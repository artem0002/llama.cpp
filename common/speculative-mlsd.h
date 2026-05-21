// speculative-mlsd.h — Multi-Level Speculative Decoding для llama.cpp
//
// MLSD объединяет 3 источника draft-токенов в одном speculative impl:
//   1. N-gram Mod (common/ngram-mod) — zero-cost токены из контекста
//   2. Draft model forward pass (как draft-simple)
//   3. MTP heads draft model (через pre-norm embeddings bridge)
//
// Использует существующую плагинную архитектуру common/speculative.

#pragma once

#include "llama.h"

#include <cstdint>

// ─── Конфигурация MLSD ───

struct common_params_speculative_mlsd {
    // Level 0: Self-Speculative для Draft-модели
    bool use_ngram_mod   = true;   // Использовать N-gram Mod (common/ngram-mod)
    bool use_draft_mtp   = true;   // Использовать MTP-головы Draft-модели

    // N-gram Mod параметры
    int  ngram_mod_n_match = 24;   // Длина lookup-ключа n-gram
    int  ngram_mod_n_min   = 48;   // Мин. draft-токенов от ngram
    int  ngram_mod_n_max   = 64;   // Макс. draft-токенов от ngram

    // Level 2: Target Verification
    bool no_target_spec  = false;  // Отключить Draft→Target спекуляцию
};

// ─── Статистика MLSD ───

struct mlsd_stats {
    // Level 0: Self-Spec статистика
    uint64_t l0_ngram_lookups   = 0;
    uint64_t l0_ngram_drafts    = 0;
    uint64_t l0_draft_model_tok = 0;
    uint64_t l0_mtp_drafts      = 0;

    // Level 2: Target Verification
    uint64_t l2_n_draft_total   = 0;
    uint64_t l2_n_accepted      = 0;

    // Общая
    uint64_t n_tokens_generated = 0;
    uint64_t n_forward_passes   = 0;

    double l2_accept_rate() const {
        return l2_n_draft_total > 0
            ? 100.0 * l2_n_accepted / l2_n_draft_total : 0.0;
    }
    double speedup() const {
        return n_forward_passes > 0
            ? (double)n_tokens_generated / n_forward_passes : 1.0;
    }
};

// ─── Factory function ───
//
// Создаёт MLSD implementation. Объявление здесь, реализация в
// speculative-mlsd.cpp. Используется в speculative.cpp.

struct common_speculative_impl;

common_speculative_impl * common_speculative_create_mlsd(
        const common_params_speculative & params,
        uint32_t n_seq);
