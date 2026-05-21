# MLSD — Карта файлов для llama.cpp (актуальная структура)

## Новые файлы (создать)

| Файл | Описание |
|------|----------|
| common/speculative-mlsd.h | Заголовок MLSD-модуля (конфиг + статистика) |
| common/speculative-mlsd.cpp | Реализация common_speculative_impl_mlsd |
| examples/speculative-mlsd/speculative_mlsd.cpp | CLI-утилита |
| examples/speculative-mlsd/CMakeLists.txt | CMake конфигурация |

## Изменяемые файлы (4 патча)

| Файл | Что добавить |
|------|-------------|
| common/common.h | COMMON_SPECULATIVE_TYPE_DRAFT_MLSD + mlsd params + need_n_rs_seq |
| common/speculative.cpp | #include + type map + impl creation |
| common/arg.cpp | --no-self-spec, --no-mtp, --no-ngram, --no-target-spec |
| tools/server/server-context.cpp | spec_mtp check для DRAFT_MLSD |

## CMake патчи (2 строки)

| Файл | Что добавить |
|------|-------------|
| common/CMakeLists.txt | speculative-mlsd.cpp в исходники |
| examples/CMakeLists.txt | add_subdirectory(speculative-mlsd) |

## НЕ ТРОГАТЬ (уже есть в llama.cpp)

| Файл | Что используем |
|------|---------------|
| common/ngram-cache.h/.cpp | 3-уровневый N-gram кэш |
| common/ngram-map.h/.cpp | N-gram Map (simple, map-k, map-k4v) |
| common/ngram-mod.h/.cpp | N-gram Mod (LCG hash) ← используем внутри MLSD |
| common/speculative.h/.cpp | Плагинная архитектура |
| common/sampling.h/.cpp | Сэмплирование токенов |
| include/llama.h | LLAMA_CONTEXT_TYPE_MTP уже есть |
| src/llama-ext.h | Pre-norm embeddings API |
| tools/server/server.cpp | Сервер (подхватывает --spec-type) |

## Команда запуска (сервер)

llama-server \
  -m models/Qwen3.5-27B-MTP-Q4_K_M.gguf \
  --spec-draft-model models/Qwen3.5-0.8B-MTP-Q4_K_M.gguf \
  --spec-type draft-mlsd \
  -ngl 0 -ngld 0 \
  --spec-draft-n-max 5 \
  --host 0.0.0.0 --port 8080

## Команда с отключением модулей

llama-server \
  -m models/27B.gguf --spec-draft-model models/0.8B.gguf \
  --spec-type draft-mlsd -ngl 0 -ngld 0 \
  --no-ngram          # отключить N-gram \
  --no-mtp            # отключить MTP \
  --no-target-spec    # отключить ускорение
