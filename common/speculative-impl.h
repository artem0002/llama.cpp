// speculative-impl.h — Base class for speculative decoding implementations
//
// Extracted from speculative.cpp so that derived implementations
// in separate translation units (e.g. speculative-mlsd.cpp) can
// inherit from common_speculative_impl.
//
// NOTE: common_speculative_config and common_speculative_are_compatible
// remain in speculative.cpp since they are only used there.

#pragma once

#include "llama.h"
#include "common.h"
#include "speculative.h"

#include <vector>
#include <cstddef>
#include <cstdint>

// -------------------------------------------------------------------
// draft parameters vector type (used by virtual methods)
// -------------------------------------------------------------------

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// -------------------------------------------------------------------
// common_speculative_impl — base class for all speculative impls
// -------------------------------------------------------------------
//
// Each implementation has a unique type and a state that is
// implementation-specific in a subclass of common_speculative_impl.

struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0;
    size_t n_call_draft  = 0;
    size_t n_call_accept = 0;

    size_t n_gen_drafts = 0;
    size_t n_acc_drafts = 0;
    size_t n_gen_tokens = 0;
    size_t n_acc_tokens = 0;

    const bool gen_perf = true;

    int64_t t_begin_us  = 0;
    int64_t t_draft_us  = 0;
    int64_t t_accept_us = 0;

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    virtual bool need_embd() const = 0;

    virtual bool need_embd_pre_norm() const { return false; }
};
