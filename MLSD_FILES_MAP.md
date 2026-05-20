# MLSD — Карта файлов для llama.cpp (актуальная структура)

## Новые файлы (создать)

| Файл | Описание |
|------|----------|
| common/speculative-mlsd.h | Заголовок MLSD-модуля |
| common/speculative-mlsd.cpp | Реализация MLSD |
| examples/speculative-mlsd/speculative_mlsd.cpp | CLI-утилита |
| examples/speculative-mlsd/CMakeLists.txt | CMake конфигурация |

## Изменяемые файлы (патчи)

| Файл | Что изменить |
|------|-------------|
| common/common.h | Добавить COMMON_SPECULATIVE_TYPE_DRAFT_MLSD + params |
| common/speculative.cpp | Зарегистрировать "draft-mlsd" в type map |
| common/CMakeLists.txt | Добавить speculative-mlsd.cpp в llama-common |
| examples/CMakeLists.txt | Добавить add_subdirectory(speculative-mlsd) |

## Существующие файлы (НЕ трогать — переиспользуем)

| Файл | Как используем |
|------|---------------|
| common/ngram-cache.h/.cpp | N-gram Cache (3-уровневый) |
| common/ngram-map.h/.cpp | N-gram Map (simple, map-k, map-k4v) |
| common/ngram-mod.h/.cpp | N-gram Mod (LCG hash) |
| common/speculative.h/.cpp | Плагинная архитектура спекуляции |
| common/sampling.h/.cpp | Сэмплирование токенов |
| include/llama.h | LLAMA_CONTEXT_TYPE_MTP уже есть |
| src/llama-ext.h | Pre-norm embeddings API (для MTP) |
| src/llama-context.cpp | MTP контекст уже поддерживается |
| tools/server/server.cpp | Сервер (использует --spec-type) |

## Ключевые API для MLSD

### Speculative Decoding API (common/speculative.h)
- common_speculative_init() — создать speculative контекст
- common_speculative_draft() — генерация draft-токенов
- common_speculative_process() — обработка batch
- common_speculative_accept() — принятие токенов

### MTP API (src/llama-ext.h)
- llama_set_embeddings_pre_norm() — включить pre-norm эмбеддинги
- llama_get_embeddings_pre_norm_ith() — получить эмбеддинги токена

### N-gram API (common/ngram-cache.h)
- common_ngram_cache — структура кэша
- Статистики: context/dynamic/static уровни

## Команда запуска (сервер)

llama-server \
  -m models/Qwen3.5-27B-MTP-Q4_K_M.gguf \
  --spec-draft-model models/Qwen3.5-0.8B-MTP-Q4_K_M.gguf \
  --spec-type draft-simple,ngram-cache,draft-mtp \
  -ngl 0 -ngld 0 \
  --spec-draft-n-max 5 \
  --host 0.0.0.0 --port 8080

## Команда запуска (CLI)

llama-speculative-mlsd \
  -m models/Qwen3.5-27B-MTP-Q4_K_M.gguf \
  --spec-draft-model models/Qwen3.5-0.8B-MTP-Q4_K_M.gguf \
  --spec-type draft-simple,ngram-cache,draft-mtp \
  -ngl 0 -ngld 0 \
  --spec-draft-n-max 5 \
  -p "Расскажи сказку про кота"
