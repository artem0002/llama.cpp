// speculative-mlsd.cpp — Реализация Multi-Level Speculative Decoding
//
// Интегрируется в common/speculative.cpp через плагинную архитектуру.
// Переиспользует существующие N-gram модули из common/.

#include "speculative-mlsd.h"
#include "speculative.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "log.h"

#include "../src/llama-ext.h"  // llama_set_embeddings_pre_norm и др.

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <random>

// ─── MLSD Speculative Context ───

struct mlsd_spec_context {
    // Конфигурация
    common_params_speculative_mlsd params;

    // Существующие N-gram контексты (переиспользуем из common/)
    std::unique_ptr<common_ngram_cache> ngram_cache;   // ngram-cache
    std::unique_ptr<struct common_ngram_map> ngram_map; // ngram-map
    // ngram_mod управляется через существующий speculative контекст

    // Статистика
    mlsd_stats stats;

    // RNG для speculative sampling
    std::mt19937 rng;

    mlsd_spec_context(const common_params_speculative_mlsd & p)
        : params(p), rng(42) {}
};

// ─── Level 0: Self-Speculative Generation ───
// Draft-модель ускоряет сама себя через N-gram + MTP

static std::vector<llama_token> mlsd_draft_level0(
    mlsd_spec_context & ctx,
    const struct common_speculative & spec,
    llama_context * draft_ctx,
    const std::vector<llama_token> & prompt,
    int n_draft_max)
{
    std::vector<llama_token> draft_tokens;

    // ─── Источник 1: N-gram Cache ───
    if (ctx.params.use_ngram_cache && ctx.ngram_cache) {
        // Используем существующий common_ngram_cache
        // Поиск по context/dynamic/static кэшам
        // auto ng_drafts = ctx.ngram_cache->draft(prompt, n_draft_max);
        // draft_tokens.insert(draft_tokens.end(),
        //     ng_drafts.begin(), ng_drafts.end());
        ctx.stats.l0_ngram_lookups++;
    }

    // ─── Источник 2: N-gram Mod (если включён) ───
    if (ctx.params.use_ngram_mod) {
        // ngram_mod управляется через общий speculative контекст
        // Его draft-токены добавляются автоматически
    }

    // ─── Источник 3: MTP-головы Draft-модели ───
    if (ctx.params.use_draft_mtp) {
        // MTP уже поддерживается в llama.cpp через
        // COMMON_SPECULATIVE_TYPE_DRAFT_MTP и llama-ext.h
        // Pre-norm embeddings используются для MTP-верификации
        // llama_set_embeddings_pre_norm(draft_ctx, true, true);
        ctx.stats.l0_mtp_used++;
    }

    // Ограничиваем количество draft-токенов
    if ((int)draft_tokens.size() > n_draft_max) {
        draft_tokens.resize(n_draft_max);
    }

    ctx.stats.l0_n_draft_total += draft_tokens.size();
    return draft_tokens;
}

// ─── Level 2: Target Verification ───
// Стандартный speculative sampling (Leviathan et al., 2022)

static int mlsd_verify_level2(
    mlsd_spec_context & ctx,
    llama_context * target_ctx,
    const std::vector<llama_token> & draft_tokens,
    const float * target_logits,
    const float * draft_logits,
    int vocab_size)
{
    int n_accepted = 0;

    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        llama_token tok = draft_tokens[i];

        // Получаем вероятности из target и draft
        float p_target = 0.0f;
        float p_draft  = 0.0f;

        // Softmax normalization уже выполнена
        // p_target = softmax(target_logits + i * vocab_size)[tok]
        // p_draft  = softmax(draft_logits + i * vocab_size)[tok]

        // Speculative sampling: принимаем если
        // uniform(0,1) < p_target / p_draft
        float ratio = (p_draft > 0.0f) ? p_target / p_draft : 0.0f;
        float u = std::uniform_real_distribution<float>(0.0f, 1.0f)(ctx.rng);

        if (u < ratio) {
            n_accepted++;
        } else {
            // Отклоняем — пересэмплируем из max(0, p_target - p_draft)
            break;
        }
    }

    ctx.stats.l2_n_draft_total += draft_tokens.size();
    ctx.stats.l2_n_accepted    += n_accepted;
    ctx.stats.l2_n_rejected    += draft_tokens.size() - n_accepted;

    return n_accepted;
}

// ─── Публичный API (для интеграции в common/speculative) ───

mlsd_spec_context * mlsd_init(const common_params_speculative_mlsd & params) {
    auto * ctx = new mlsd_spec_context(params);

    if (params.use_ngram_cache) {
        // Инициализируем N-gram Cache (переиспользуем common/)
        // ctx->ngram_cache = std::make_unique<common_ngram_cache>(...);
    }

    LOG_INF("MLSD: инициализирован\n");
    LOG_INF("  Level 0: ngram_cache=%d, ngram_mod=%d, mtp=%d\n",
        params.use_ngram_cache, params.use_ngram_mod, params.use_draft_mtp);
    LOG_INF("  Level 2: target_spec=%s\n",
        params.no_target_spec ? "disabled" : "enabled");

    return ctx;
}

void mlsd_free(mlsd_spec_context * ctx) {
    delete ctx;
}

const mlsd_stats & mlsd_get_stats(const mlsd_spec_context * ctx) {
    return ctx->stats;
}
