// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_cpu.hpp - the dense Qwen3.5 forward (qwen35) on the CPU alone, behind PlainModel
// (PlainOptions::cpu, `--device cpu`): the same model, read from the same GGUF through the same spec, served by the
// same loop; only the computation is here instead of on a GPU.
//
// Every matrix is packed at load into the int8 panels of kernels/cpu/i8_gemm.hpp with a scale per 32 k values (an
// int8 level is finer than the file's 4 to 6 bits, so the K-quants come through nearly as they are; a Q8_0 matrix
// exactly); the activations are quantized per 32 values as well, as llama.cpp's own CPU products quantize theirs.
// The rest is FP32 on AVX2: the norms, the rotation, the gated delta-net (its convolution, gates and recurrence; the
// state of a head is 64 KiB and stays in a core's cache), the attention over the K/V cache in tiles, the gates.
// The GDN's alpha and beta are kept in FP32 (they feed exponentials).  The work goes out on kernels/cpu/cpu_ops.hpp's
// pool.  Not here: pictures, the MTP layer and drafts, batches of conversations, the KV cache's narrow forms.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_spec.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct PlainOptions;

class PlainCpu {
public:
    PlainCpu();
    ~PlainCpu();
    PlainCpu(const PlainCpu&) = delete;
    PlainCpu& operator=(const PlainCpu&) = delete;

    /// Packs the model's matrices.  The caller has checked that the CPU runs AVX2 (this file is compiled for it).
    /// Blocks [l0, l1) only (l1 < 0: to the last) for a GPU's tail (PlainModel's --gpu-layers): no embedding, no head,
    /// the rows taken from the GPU's blocks by run_blocks.
    bool load(const GgufModel& m, const ModelSpec& s, const PlainOptions& opt, std::string& err, int64_t l0 = 0,
              int64_t l1 = -1);
    int64_t vocab() const { return n_vocab_; }
    /// The pool's threads and the int8 product's kernel, for the start log.
    std::string describe() const;
    std::vector<int> worker_cpus() const;
    /// Zeroes the recurrent states: the next run starts a sequence at position 0.
    void reset();
    /// The model has recurrent layers (their state cannot go back to a prefix; the K/V cells can).
    bool recurrent() const;
    /// Tokens [0, n) at positions pos .. pos + n - 1 (following the previous run's), in chunks; `logits` gets the last
    /// `rows` tokens' rows (rows <= n), n_vocab floats each.
    bool run(const int32_t* tokens, int64_t n, int64_t pos, int rows, float* logits, std::string& err);
    /// A tail's blocks over n rows of n_embd floats `x` (in place), at positions pos .. pos + n - 1, in chunks.
    bool run_blocks(float* x, int64_t n, int64_t pos, std::string& err);
    /// Row copies of the recurrent states for a window's drafts going back (n copies: the states after rows
    /// 0 .. n-1, as the GPU window kernel's); 0 frees them.  run_blocks takes them while the rows fit one window.
    void set_snap_rows(int n);
    /// The recurrent states back to the state after row n (set_snap_rows' copy n); false without one.
    bool rollback(int n, std::string& err);
    /// The state of the first `tokens` tokens as host parts (every recurrent layer's state and convolution history;
    /// with `kv`, every attention layer's K and V cells too), and back.
    void save(int64_t tokens, bool kv, std::vector<std::vector<uint8_t>>& parts) const;
    bool restore(const std::vector<std::vector<uint8_t>>& parts, bool kv, std::string& err);
    size_t state_bytes(int64_t tokens, bool kv) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    int64_t n_vocab_ = 0;
};

}  // namespace strata::core
