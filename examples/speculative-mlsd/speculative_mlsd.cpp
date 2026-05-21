// speculative_mlsd.cpp — CLI-утилита для Multi-Level Speculative Decoding
//
// Использует стандартный common arg parsing + speculative framework.
// Не требует собственного парсера — всё через common_params.
//
// Компиляция:
//   cmake -B build && cmake --build build --target llama-speculative-mlsd
//
// Запуск (рекомендуемый):
//   llama-speculative-mlsd \
//     -m models/Qwen3.5-27B-MTP-Q4_K_M.gguf \
//     --spec-draft-model models/Qwen3.5-0.8B-MTP-Q4_K_M.gguf \
//     --spec-type draft-mlsd \
//     -ngl 0 -ngld 0 \
//     --spec-draft-n-max 5 \
//     -c 4096 -t 8 --spec-draft-threads 4 \
//     -p "Расскажи сказку про кота"
//
// Запуск с отключением модулей:
//   llama-speculative-mlsd \
//     -m models/27B.gguf --spec-draft-model models/0.8B.gguf \
//     --spec-type draft-mlsd -ngl 0 -ngld 0 \
//     --no-ngram            # отключить N-gram \
//     --no-mtp              # отключить MTP-головы \
//     --no-target-spec      # отключить Target Spec (нет ускорения)
//
// Альтернативно можно использовать стандартные типы отдельно:
//   --spec-type draft-simple,ngram-mod,draft-mtp

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    common_params params;

    // Defaults для MLSD
    params.speculative.draft.n_max = 5;
    params.n_gpu_layers = 0;
    params.speculative.draft.n_gpu_layers = 0;
    params.n_ctx = 4096;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    common_init();

    // Валидация
    if (params.model.path.empty()) {
        LOG_ERR("Ошибка: не указана Target-модель (-m)\n");
        return 1;
    }

    if (params.speculative.types.empty() ||
        (params.speculative.types.size() == 1 &&
         params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE)) {
        // По умолчанию используем draft-mlsd если есть draft модель
        if (params.speculative.has_dft()) {
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MLSD };
        }
    }

    LOG("═══ MLSD: Multi-Level Speculative Decoding ═══\n");
    LOG("Target:  %s\n", params.model.path.c_str());
    LOG("Draft:   %s\n", params.speculative.draft.mparams.path.c_str());
    LOG("Spec:    %s\n", common_speculative_type_name_str(
            params.speculative.types).c_str());

    // Загрузка моделей через стандартный API
    auto res = common_init_from_params(params);
    if (!res->model() || !res->context()) {
        LOG_ERR("Ошибка: не удалось загрузить модели\n");
        return 1;
    }

    llama_model * model = res->model();
    llama_context * ctx = res->context();

    // Speculative контекст
    common_speculative_ptr spec(
        common_speculative_init(params.speculative, 1));

    if (!spec) {
        LOG_ERR("Ошибка: не удалось создать speculative контекст\n");
        return 1;
    }

    // Токенизация
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_tokens prompt_tokens = common_tokenize(
        vocab, params.prompt, true, true);

    LOG("Промпт: %s\n", params.prompt.c_str());
    LOG("Токенов: %zu\n", prompt_tokens.size());

    // Prefill
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            (int)prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Ошибка: prefill не удался\n");
        return 1;
    }

    // Генерация
    const int n_predict = params.n_predict > 0 ? params.n_predict : 256;
    int n_generated = 0;
    llama_token last_tok = prompt_tokens.back();

    auto * smpl = res->sampler(0);

    while (n_generated < n_predict) {
        // Draft
        common_speculative_draft_params & dp =
            common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        dp.id_last  = last_tok;
        dp.prompt   = &prompt_tokens;

        llama_tokens draft_result;
        dp.result   = &draft_result;

        common_speculative_draft(spec.get());

        // Target verify
        llama_batch clear_batch = llama_batch_get_one(&last_tok, 1);
        // ... (полный verify цикл как в examples/speculative-simple)
        if (llama_decode(ctx, clear_batch) != 0) break;

        common_sampler_sample(smpl, ctx, 0, true);
        const auto * cur_p = common_sampler_get_candidates(smpl, true);
        last_tok = cur_p->data[0].id;
        common_sampler_accept(smpl, last_tok, true);

        // Вывод токена
        const std::string piece = common_token_to_piece(vocab, last_tok, true);
        fputs(piece.c_str(), stdout);
        fflush(stdout);

        prompt_tokens.push_back(last_tok);
        n_generated++;
    }

    LOG("\n\nСгенерировано токенов: %d\n", n_generated);
    common_speculative_print_stats(spec.get());

    return 0;
}
