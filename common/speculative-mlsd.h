// speculative-mlsd.h — Multi-Level Speculative Decoding для llama.cpp
//
// Интегрируется в существующую плагинную архитектуру speculative decoding.
// Добавляет тип COMMON_SPECULATIVE_TYPE_DRAFT_MLSD, который объединяет:
//   Level 0: Self-Speculative (Draft-модель + N-gram/MTP)
//   Level 1: Накопитель draft-токенов
//   Level 2: Target Verification (27B)
//
// N-gram модули уже существуют в common/:
//   - ngram-cache.h (3-уровневый кэш: context/dynamic/static)
//   - ngram-map.h (ngram-simple, ngram-map-k, ngram-map-k4v)
//   - ngram-mod.h (LCG hash-based)
//
// MTP поддержка уже встроена:
//   - COMMON_SPECULATIVE_TYPE_DRAFT_MTP
//   - src/llama-ext.h (pre-norm embeddings API)

#pragma once

#include "speculative.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "llama.h"

#include <vector>
#include <memory>

// ─── Конфигурация MLSD ───

struct common_params_speculative_mlsd {
    // Level 0: Self-Speculative для Draft-модели
    bool use_ngram_cache = true;   // Использовать N-gram Cache (common/ngram-cache)
    bool use_ngram_mod   = false;  // Использовать N-gram Mod (common/ngram-mod)
    bool use_draft_mtp   = true;   // Использовать MTP-головы Draft-модели

    // N-gram Cache параметры (использует существующий common/ngram-cache)
    int  ngram_n_min     = 1;      // Мин. порядок N-граммы
    int  ngram_n_max     = 4;      // Макс. порядок N-граммы

    // N-gram Mod параметры (использует существующий common/ngram-mod)
    int  ngram_mod_n_match = 24;
    int  ngram_mod_n_min   = 48;
    int  ngram_mod_n_max   = 64;

    // Level 2: Target Verification
    bool no_target_spec  = false;  // Отключить Draft→Target спекуляцию
};

// ─── Статистика MLSD ───

struct mlsd_stats {
    // Level 0 статистика
    uint64_t l0_n_draft_total   = 0;  // Всего draft-токенов от L0
    uint64_t l0_n_draft_accepted = 0; // Принято L0 (self-verify)
    uint64_t l0_ngram_hits      = 0;  // N-gram попадания
    uint64_t l0_ngram_lookups   = 0;  // N-gram запросов
    uint64_t l0_mtp_used        = 0;  // MTP draft-токенов использовано

    // Level 2 статистика
    uint64_t l2_n_draft_total   = 0;  // Draft-токенов отправлено на верификацию
    uint64_t l2_n_accepted      = 0;  // Принято Target-моделью
    uint64_t l2_n_rejected      = 0;  // Отклонено Target-моделью

    // Общая статистика
    uint64_t n_tokens_generated = 0;  // Всего токенов сгенерировано
    uint64_t n_forward_passes   = 0;  // Forward passes Target-модели

    double l0_accept_rate() const {
        return l0_n_draft_total > 0
            ? 100.0 * l0_n_draft_accepted / l0_n_draft_total : 0.0;
    }
    double l2_accept_rate() const {
        return l2_n_draft_total > 0
            ? 100.0 * l2_n_accepted / l2_n_draft_total : 0.0;
    }
    double speedup() const {
        return n_forward_passes > 0
            ? (double)n_tokens_generated / n_forward_passes : 1.0;
    }
};
