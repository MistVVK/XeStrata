// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_inkling.hpp - the CPU reference forward of Inkling-2-Mini (inkling2_loop).
//
// This is the correctness reference for the arch: one residual stream, hybrid sliding-window / relative attention,
// short convolutions, AttnRes every 6 layers, sigmoid-gated MoE and a fixed 3-pass loop over core layers 6-17.  It is
// reached with --device cpu (no GPU opened) and is the form later GPU work is compared against.  The graph follows
// llama.cpp's inkling2.cpp / inkling2-loop.cpp (PR #32694) and the scale re-share of PR #32704.
#pragma once

#include "strata/artifact/model_spec.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufModel;
}

namespace strata::core {

class PlainInkling {
public:
    PlainInkling();
    ~PlainInkling();

    bool load(const GgufModel& m, const ModelSpec& s, std::string& err);
    /// `tokens[n]` at `pos`, the last `rows` logits into `logits` [rows][n_vocab_unpadded].
    bool run(const int32_t* tokens, int64_t n, int64_t pos, int rows, float* logits, std::string& err);
    // Evaluate a candidate window, then commit only the rows selected by `keep`.
    bool verify(const int32_t* tokens, int count, int64_t pos, float* logits,
                const std::function<int()>& keep, std::string& err);
    /// Last token's ordered DSpark tap row, in device memory until the next run or reset.
    const float* gpu_taps() const;
    void reset();
    bool enable_gpu(std::string& err);
    /// `reserve_mib`, `vram_free_mib`, `expert_slots`: --vram-reserve-mib, --vram-free-mib and --expert-cache as
    /// PlainModel's tier takes them (each < 0: not given), for the resident experts' tier.
    bool configure_experts(const std::string& profile, int64_t context, int64_t reserve_mib, int64_t vram_free_mib,
                           int64_t expert_slots, std::string& err);
    bool save_profile(const std::string& path, std::string& err) const;

    bool load_dspark(const std::string& path, std::string& err);
    bool propose_dspark(int32_t anchor, int64_t pos, int count, std::vector<int32_t>& tokens, std::string& err) const;

    int64_t vocab() const;
    std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::core
