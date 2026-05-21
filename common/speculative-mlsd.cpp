// speculative-mlsd.cpp — Реализация Multi-Level Speculative Decoding
//
// Класс common_speculative_impl_mlsd наследует common_speculative_impl
// и объединяет draft-model + ngram-mod + MTP в один speculative impl.
//
// Интеграция: добавляется в common/speculative.cpp при инициализации
// типа COMMON_SPECULATIVE_TYPE_DRAFT_MLSD.

#include "speculative-mlsd.h"
#include "speculative.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "log.h"

#include "../src/llama-ext.h"  // llama_set_embeddings_pre_norm

#include <algorithm>
#include <cassert>
#include <cinttypes>

// ═══════════════════════════════════════════════════════════
// common_speculative_impl_mlsd
// ═══════════════════════════════════════════════════════════
//
// Объединяет 3 источника draft-токенов:
//   1. N-gram Mod — бесплатный lookup в хеш-таблице
//   2. Draft model — forward pass через маленькую модель
//   3. MTP — Multi-Token Prediction головы draft-модели
//
// Логика draft():
//   a) Если включён ngram-mod, получаем draft из ngram
//   b) Если есть draft model, делаем forward pass + sampling
//   c) Если включён MTP, используем pre-norm embeddings bridge
//   d) Комбинируем результаты, ограничиваем n_max
//
// Логика process():
//   Зеркалируем target batch в draft context (как draft-simple)
//   и в MTP context (как draft-mtp)

struct common_speculative_impl_mlsd : public common_speculative_impl {
    common_params_speculative_draft draft_params;  // переиспользуем draft params
    common_params_speculative_mlsd  mlsd_params;

    llama_batch batch;

    // N-gram Mod (один на все seq)
    common_ngram_mod ngram_mod;

    // Samplers для draft model
    std::vector<common_sampler_ptr> smpls;

    // MTP support
    int32_t n_embd = 0;
    std::vector<std::vector<float>> pending_h;     // [n_seq][n_embd]
    std::vector<std::vector<float>> verify_h;       // [n_seq][n_rows * n_embd]
    std::vector<int32_t> verify_h_rows;
    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;
    std::vector<uint16_t> last_n_drafted;

    // Per-seq ngram state
    struct seq_info {
        size_t i_last = 0;
        size_t n_draft_last = 0;
        int n_low = 0;
    };
    std::vector<seq_info> sinfos;

    // Stats
    mlsd_stats stats;

    common_speculative_impl_mlsd(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MLSD, n_seq)
        , draft_params(params.draft)
        , mlsd_params(params.mlsd)
        , ngram_mod(params.mlsd.ngram_mod_n_match, 4 * 1024 * 1024)
    {
        auto * ctx_dft = draft_params.ctx_dft;
        auto * ctx_tgt = draft_params.ctx_tgt;

        LOG_INF("%s: adding speculative implementation 'draft-mlsd'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__,
                draft_params.n_max, draft_params.n_min, draft_params.p_min);
        LOG_INF("%s: - ngram_mod=%d, draft_mtp=%d\n", __func__,
                mlsd_params.use_ngram_mod, mlsd_params.use_draft_mtp);
        LOG_INF("%s: - ctx_tgt=%s, ctx_dft=%s\n", __func__,
                ctx_tgt ? "yes" : "no", ctx_dft ? "yes" : "no");

        // Draft model batch
        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);

        if (mlsd_params.use_draft_mtp && ctx_tgt && ctx_dft) {
            // MTP mode: need embeddings in batch
            n_embd = llama_model_n_embd(llama_get_model(ctx_dft));
            batch = llama_batch_init(n_b, n_embd, 1);
            batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

            // Enable pre-norm embeddings
            llama_set_embeddings_pre_norm(ctx_tgt, true, false);
            llama_set_embeddings_pre_norm(ctx_dft, true, true);

            pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
            verify_h.assign(n_seq, {});
            verify_h_rows.assign(n_seq, 0);
            last_n_drafted.assign(n_seq, 0);
        } else {
            // Simple draft mode (no MTP embeddings)
            n_embd = 0;
            batch = llama_batch_init(n_b, 0, 1);
        }

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        // Samplers
        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        sinfos.resize(n_seq);

        // Vocab compatibility check
        if (ctx_tgt && ctx_dft) {
            const llama_vocab * vocab_tgt = llama_model_get_vocab(llama_get_model(ctx_tgt));
            const llama_vocab * vocab_dft = llama_model_get_vocab(llama_get_model(ctx_dft));
            const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
            const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
            const int vocab_diff  = std::abs(n_vocab_tgt - n_vocab_dft);

            if (vocab_diff > 128) {
                LOG_WRN("%s: vocab size difference too large: %d vs %d\n",
                        __func__, n_vocab_tgt, n_vocab_dft);
            }
        }
    }

    ~common_speculative_impl_mlsd() override {
        if (n_embd > 0 && batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    // ─── begin: обновляем ngram-mod при новом промпте ───

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (!mlsd_params.use_ngram_mod) return;
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) return;

        auto & sinfo = sinfos[seq_id];
        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = ngram_mod.get_n();
        if (prompt.size() < n) return;

        // Добавляем n-граммы из промпта
        for (size_t i = 0; i < prompt.size() - n; ++i) {
            ngram_mod.add(prompt.data() + i);
        }
        sinfo.i_last = prompt.size() - n;

        // Проверяем заполненность
        const double f = (double)ngram_mod.get_used() / (double)ngram_mod.size();
        if (f > 0.25) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds 0.25 — resetting\n", __func__, f);
            ngram_mod.reset();
        }
    }

    // ─── process: зеркалируем batch в draft context ───

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) return true;
        if (batch_in.token == nullptr || batch_in.embd != nullptr) return true;

        auto * ctx_dft = draft_params.ctx_dft;
        auto * ctx_tgt = draft_params.ctx_tgt;

        const int32_t n_tokens = batch_in.n_tokens;

        // Запоминаем границы батча для каждой seq
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (batch_in.n_seq_id[k] > 0 && batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        common_batch_clear(batch);

        // Копируем токены в draft batch
        for (int k = 0; k < n_tokens; ++k) {
            common_batch_add(batch, batch_in.token[k], batch_in.pos[k],
                           { batch_in.seq_id[k][0] }, 0);
        }

        if (n_embd > 0 && ctx_tgt) {
            // MTP mode: заполняем embeddings
            const size_t row_bytes = (size_t) n_embd * sizeof(float);
            const float * h_tgt = llama_get_embeddings_pre_norm(ctx_tgt);

            // Shift embeddings right by one position
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt,
                       row_bytes * (n_tokens - 1));

            // Fill pending embeddings from previous run
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) continue;
                std::memcpy(batch.embd + (size_t) i_batch_beg[seq_id] * n_embd,
                           pending_h[seq_id].data(), row_bytes);
            }
        }

        const int32_t rc = llama_decode(ctx_dft, batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d\n", __func__, rc);
            return false;
        }

        // Обновляем verify_h и pending_h для MTP
        if (n_embd > 0 && ctx_tgt) {
            const size_t row_bytes = (size_t) n_embd * sizeof(float);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_end[seq_id] < 0) continue;

                const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
                verify_h_rows[seq_id] = n_rows;
                verify_h[seq_id].resize((size_t) n_rows * n_embd);

                for (int32_t i = 0; i < n_rows; ++i) {
                    const float * h = llama_get_embeddings_pre_norm_ith(
                        ctx_tgt, i_batch_beg[seq_id] + i);
                    std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd,
                               h, row_bytes);
                }

                std::memcpy(pending_h[seq_id].data(),
                           verify_h[seq_id].data() + (size_t)(n_rows - 1) * n_embd,
                           row_bytes);
            }
        }

        return true;
    }

    // ─── draft: генерация draft-токенов из 3 источников ───

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto * ctx_dft = draft_params.ctx_dft;

        common_batch_clear(batch);

        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) continue;

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            if (n_embd > 0) {
                // MTP: добавляем embedding
                const size_t row_bytes = (size_t) n_embd * sizeof(float);
                std::memcpy(batch.embd + n_embd * (batch.n_tokens - 1),
                           pending_h[seq_id].data(), row_bytes);
            }
        }

        // ── Шаг 1: N-gram Mod draft (для seq, где включён) ──
        if (mlsd_params.use_ngram_mod) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) continue;
                auto & dp = dparams[seq_id];

                // Добавляем новые n-граммы из prompt
                auto & sinfo = sinfos[seq_id];
                const auto & prompt = *dp.prompt;
                const size_t n = ngram_mod.get_n();
                if (sinfo.i_last + 32 < prompt.size()) {
                    for (size_t i = sinfo.i_last; i < prompt.size() - n; ++i) {
                        ngram_mod.add(prompt.data() + i);
                    }
                    sinfo.i_last = prompt.size() - n;
                }

                // Получаем ngram draft
                const size_t cur_len = prompt.size();
                if (cur_len >= n) {
                    std::vector<llama_token> ngram_buf(n + mlsd_params.ngram_mod_n_max);
                    for (size_t i = 0; i < n - 1; ++i) {
                        ngram_buf[i] = prompt.at(cur_len - n + 1 + i);
                    }
                    ngram_buf[n - 1] = dp.id_last;

                    for (int i = 0; i < mlsd_params.ngram_mod_n_max; ++i) {
                        const llama_token tok = ngram_mod.get(ngram_buf.data() + i);
                        if (tok == common_ngram_mod::EMPTY) break;
                        dp.result->push_back(tok);
                        ngram_buf[n + i] = tok;
                    }
                    stats.l0_ngram_drafts += dp.result->size();
                }
                sinfo.n_draft_last = dp.result->size();
                stats.l0_ngram_lookups++;
            }
        }

        // Если ngram дал достаточно токенов — пропускаем draft model
        // (ngram-mod токены бесплатные, draft model — нет)
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (!drafting[seq_id]) continue;
            auto & dp = dparams[seq_id];
            if ((int)dp.result->size() >= draft_params.n_max) {
                drafting[seq_id] = false;
                n_drafting--;
            }
        }

        // ── Шаг 2: Draft model forward pass + sampling ──
        if (n_drafting > 0) {
            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
                return;
            }

            int i = 0;
            while (n_drafting > 0) {
                int i_batch = 0;
                common_batch_clear(batch);

                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (!drafting[seq_id]) continue;

                    auto * smpl = smpls[seq_id].get();
                    common_sampler_sample(smpl, ctx_dft, i_batch, true);

                    if (n_embd > 0) {
                        // MTP: получаем embedding для следующего шага
                        const float * h_row = llama_get_embeddings_pre_norm_ith(
                            ctx_dft, i_batch);
                        const size_t row_bytes = (size_t) n_embd * sizeof(float);

                        ++i_batch;

                        const auto * cur_p = common_sampler_get_candidates(smpl, true);
                        const llama_token id = cur_p->data[0].id;

                        if (cur_p->data[0].p < draft_params.p_min) {
                            drafting[seq_id] = false;
                            n_drafting--;
                            continue;
                        }

                        common_sampler_accept(smpl, id, true);

                        auto & dp = dparams.at(seq_id);
                        dp.result->push_back(id);
                        stats.l0_draft_model_tok++;

                        if ((int)dp.result->size() >= draft_params.n_max ||
                            (dp.n_max > 0 && (int)dp.result->size() >= dp.n_max)) {
                            drafting[seq_id] = false;
                            n_drafting--;
                            continue;
                        }

                        common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                        std::memcpy(batch.embd + n_embd * (batch.n_tokens - 1),
                                   h_row, row_bytes);
                    } else {
                        // Без MTP — как draft-simple
                        ++i_batch;

                        const auto * cur_p = common_sampler_get_candidates(smpl, true);
                        const llama_token id = cur_p->data[0].id;

                        if (cur_p->data[0].p < draft_params.p_min) {
                            drafting[seq_id] = false;
                            n_drafting--;
                            continue;
                        }

                        common_sampler_accept(smpl, id, true);

                        auto & dp = dparams.at(seq_id);
                        dp.result->push_back(id);
                        stats.l0_draft_model_tok++;

                        if ((int)dp.result->size() >= draft_params.n_max ||
                            (dp.n_max > 0 && (int)dp.result->size() >= dp.n_max)) {
                            drafting[seq_id] = false;
                            n_drafting--;
                            continue;
                        }

                        common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    }
                }

                if (batch.n_tokens == 0) break;

                ret = llama_decode(ctx_dft, batch);
                if (ret != 0) {
                    LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                    break;
                }
                ++i;
            }
        }

        // ── Финализация: min filter и stats ──
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) continue;

            if (dp.result->size() < (size_t) draft_params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
            stats.l2_n_draft_total += dp.result->size();
        }
    }

    // ─── accept: обновляем ngram-mod stats и MTP state ───

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) return;

        // N-gram mod adaptive reset
        if (!is_other && mlsd_params.use_ngram_mod) {
            auto & sinfo = sinfos[seq_id];
            if (sinfo.n_draft_last > 0) {
                const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
                if (f_acc < 0.25) {
                    sinfo.n_low++;
                    if (sinfo.n_low >= 5) {
                        ngram_mod.reset();
                        sinfo.n_low = 0;
                        sinfo.i_last = 0;
                    }
                } else {
                    sinfo.n_low = 0;
                }
            }
        }

        // MTP: обновляем pending_h из verify_h
        if (n_embd > 0) {
            const int32_t n_rows = verify_h_rows[seq_id];
            if (n_rows > 0) {
                const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
                const size_t row_bytes = (size_t) n_embd * sizeof(float);
                std::memcpy(pending_h[seq_id].data(),
                           verify_h[seq_id].data() + (size_t) i_h * n_embd,
                           row_bytes);
            }
        }

        stats.l2_n_accepted += n_accepted;
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_pre_norm() const override {
        return n_embd > 0;  // true если MTP включён
    }
};
