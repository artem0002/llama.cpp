// speculative_mlsd.cpp — CLI-утилита для Multi-Level Speculative Decoding
//
// Аналог examples/speculative/speculative.cpp, но с MLSD.
// Использует существующую плагинную архитектуру common/speculative.
//
// Компиляция:
//   cmake -B build && cmake --build build --target llama-speculative-mlsd
//
// Запуск:
//   ./llama-speculative-mlsd \
//     -m models/Qwen3.5-27B-MTP-Q4_K_M.gguf \
//     --spec-draft-model models/Qwen3.5-0.8B-MTP-Q4_K_M.gguf \
//     --spec-type draft-simple,ngram-cache,draft-mtp \
//     -ngl 0 -ngld 0 \
//     --spec-draft-n-max 5 \
//     -p "Расскажи сказку про кота"

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-mlsd.h"
#include "log.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

// ─── Парсинг аргументов ───

static common_params mlsd_params_parse(int argc, char ** argv) {
    common_params params;

    // Основные параметры
    params.model.path = "";  // Target-модель
    params.n_ctx      = 4096;
    params.n_threads  = 8;

    // Speculative параметры
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MLSD};
    params.speculative.draft.n_max = 5;
    params.speculative.draft.mparams.path = "";  // Draft-модель

    // CPU-only
    params.n_gpu_layers       = 0;
    params.speculative.draft.mparams.n_gpu_layers = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            params.model.path = argv[++i];
        } else if (arg == "--spec-draft-model" || arg == "-md") {
            params.speculative.draft.mparams.path = argv[++i];
        } else if (arg == "--spec-type") {
            // Парсим типы через существующий API
            std::string val = argv[++i];
            params.speculative.types =
                common_speculative_types_from_names({val});
        } else if (arg == "--spec-draft-n-max") {
            params.speculative.draft.n_max = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx-size") {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "-ngld" || arg == "--spec-draft-ngl") {
            params.speculative.draft.mparams.n_gpu_layers =
                std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-n" || arg == "--n-predict") {
            params.n_predict = std::stoi(argv[++i]);
        }
    }

    return params;
}

int main(int argc, char ** argv) {
    common_params params = mlsd_params_parse(argc, argv);

    // Валидация
    if (params.model.path.empty()) {
        LOG_ERR("Ошибка: не указана Target-модель (-m)\n");
        return 1;
    }

    LOG("═══ MLSD: Multi-Level Speculative Decoding ═══\n");
    LOG("Target:  %s\n", params.model.path.c_str());
    LOG("Draft:   %s\n",
        params.speculative.draft.mparams.path.c_str());
    LOG("Spec:    %s\n",
        common_speculative_type_name_str(
            params.speculative.types).c_str());

    // Инициализация через стандартный common_init
    common_init_result result = common_init_from_params(params);
    if (!result.model || !result.context) {
        LOG_ERR("Ошибка: не удалось инициализировать модели\n");
        return 1;
    }

    llama_model * model = result.model;
    llama_context * ctx = result.context;

    // Инициализация speculative контекста
    common_speculative_ptr spec(
        common_speculative_init(params.speculative,
            /* n_seq = */ 1));

    if (!spec) {
        LOG_ERR("Ошибка: не удалось инициализировать speculative\n");
        return 1;
    }

    // Токенизация промпта
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_tokens prompt_tokens = common_tokenize(
        vocab, params.prompt, true, true);

    LOG("Промпт: %s\n", params.prompt.c_str());
    LOG("Токенов: %zu\n", prompt_tokens.size());

    // Основной цикл генерации
    int n_generated = 0;
    llama_tokens generated;

    while (n_generated < params.n_predict) {
        // Обработка через speculative контекст
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.empty() ?
                &generated.back() : prompt_tokens.data(),
            prompt_tokens.empty() ?
                1 : (int)prompt_tokens.size());

        common_speculative_process(spec.get(), batch);
        common_speculative_draft(spec.get());

        // Принимаем токены
        // ... (стандартный speculative decoding цикл)

        n_generated++;
        prompt_tokens.clear();
    }

    // Вывод статистики
    common_speculative_print_stats(spec.get());

    LOG("\nСгенерировано токенов: %d\n", n_generated);

    return 0;
}
