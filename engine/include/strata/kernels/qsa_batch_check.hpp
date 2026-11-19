// include/strata/kernels/qsa_batch_check.hpp - the argument check of native_qsa_indexer_append_batch, with the failed predicate named.
//
// #5822: the check threw one text ("bad geometry, positions, parameters or scaling") for seven different causes, so three engine exits on 0.1.58 could not be
// told apart (a position past the session's cells after a prefix reuse, a wrong geometry, a stream missing, a bad epsilon or RoPE scaling).  The text now names the
// predicate and the values; the first line of the old text is kept at the start.  Pure: no GPU.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>

namespace strata::kernels {

struct QsaBatchArgs {
    bool has_stream = true;
    int64_t idx_dim = 0, idx_block = 0, n_rot = 0;       // the shapes the caller passes
    int64_t want_dim = 0, want_block = 0, want_rot = 0;  // the kernel's compiled D, R, ROT
    int64_t n = 0, p0 = 0, max_cells = 0, pos_base = 0;
    float epsilon = 0.0f;
    const char* rope_problem = nullptr;                   // rope_scaling_invalid(scaling): null when fine
};

/// "" when the arguments are valid, else the text of the first failed predicate (with the values)
inline std::string qsa_batch_problem(const QsaBatchArgs& a) {
    auto ll = [](int64_t v) { return std::to_string((long long) v); };
    const std::string head = "native QSA indexer (batch): bad geometry, positions, parameters or scaling: ";
    if (!a.has_stream) return head + "no stream";
    if (a.idx_dim != a.want_dim || a.idx_block != a.want_block || a.n_rot != a.want_rot)
        return head + "geometry idx_dim " + ll(a.idx_dim) + " idx_block " + ll(a.idx_block) + " n_rot " + ll(a.n_rot) + " (the kernel is built for " + ll(a.want_dim) + " / " +
               ll(a.want_block) + " / " + ll(a.want_rot) + ")";
    if (a.p0 < 0) return head + "position p0 " + ll(a.p0) + " is negative";
    if (a.p0 + a.n > a.max_cells) return head + "positions p0 " + ll(a.p0) + " + n " + ll(a.n) + " pass the session's " + ll(a.max_cells) + " cells";
    if (a.max_cells > INT32_MAX) return head + "max_cells " + ll(a.max_cells) + " passes 2^31 - 1";
    if (a.pos_base < 0 || a.pos_base % (a.want_block > 0 ? a.want_block : 1) != 0) return head + "pos_base " + ll(a.pos_base) + " is not a non-negative multiple of the block " + ll(a.want_block);
    if (a.pos_base + a.max_cells > INT32_MAX) return head + "pos_base " + ll(a.pos_base) + " + max_cells " + ll(a.max_cells) + " passes 2^31 - 1";
    if (!std::isfinite(a.epsilon) || a.epsilon <= 0.0f) return head + "epsilon " + std::to_string(a.epsilon) + " is not a positive number";
    if (a.rope_problem != nullptr) return head + "RoPE scaling: " + a.rope_problem;
    return {};
}

}  // namespace strata::kernels
