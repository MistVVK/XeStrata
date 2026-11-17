// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/draft_head.hpp - the MTP drafts' head at 2 bits, for Qwen3.8-Flash-Next's drafter (mtp.cpp) and
// the other models' (PlainModel): the coarse products of every row of the draft head (its token subset, or the whole
// vocabulary), the k best of them taken again with the head's own rows and written back.  The logits it leaves are
// the coarse ones but for those k (from a long context on, those k alone), so the picks and the draft probabilities
// after it are taken as before.  A draft only proposes (the verify window decides with the model's head): a worse
// pick costs acceptance, not the answer.
//
// Which form a device takes is measured at load (`choose`), not told by its name: on the 2560 x 106,299 subset of
// Qwen3.8's Q5_K head, a draft step's head took 556 -> 288 us on the B70 and 400 -> 325 us on the RTX 4070 (k = 16,
// 2026-10-26, records/2026-10-26-plan/d1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::gpu { struct VmemSegment; }

namespace strata::core {

/// The drafts' token subset (tools/draft_vocab.py's int32 ids, setup's data/draft_vocab*.bin) and the head's rows of
/// those tokens gathered into VRAM, for both engines' MTP drafters: a draft's pick is its index here.  A head over a
/// subset reads less a draft step (upstream #474); a file with no usable ids leaves size() 0, the whole head.
class DraftSubset {
public:
    DraftSubset() = default;
    ~DraftSubset();
    DraftSubset(const DraftSubset&) = delete;
    DraftSubset& operator=(const DraftSubset&) = delete;

    /// The tokens of the file at `path` (0 when there is none), for planning the VRAM before it is loaded.
    static int64_t file_tokens(const std::string& path);
    /// Reads `path`'s ids below `n_vocab` and gathers their `row_bytes` rows of `head_rows` (device) into VRAM.
    /// False (and `err`) when they do not fit, after a hint of the smaller subsets setup ships; true with size() 0
    /// when the file has no usable id (`who` names the engine in the lines).
    bool load(const std::string& path, int64_t n_vocab, const void* head_rows, size_t row_bytes, void* stream,
              const char* who, std::string& err, bool virtual_rows = false);
    int64_t size() const { return (int64_t) ids_.size(); }
    const uint8_t* rows() const { return rows_; }
    const int32_t* ids_device() const { return d_ids_; }
    const std::vector<int32_t>& ids() const { return ids_; }
    uint64_t bytes() const { return bytes_; }
    bool can_park() const { return mapping_ != nullptr; }
    /// The rows to pinned memory and their VRAM given up; `returned` (when given): whether the device has that VRAM
    /// again (gpu::vmem_unmap_returns), without which parking makes no room.
    bool park(std::string& err, bool* returned = nullptr);
    bool restore(std::string& err);

private:
    uint8_t* rows_ = nullptr;
    int32_t* d_ids_ = nullptr;
    std::vector<int32_t> ids_;
    uint64_t bytes_ = 0;
    size_t mapping_bytes_ = 0;
    strata::gpu::VmemSegment* mapping_ = nullptr;
    void* saved_rows_ = nullptr;
};

class DraftHeadInt2 {
public:
    /// The head's own product of `n` rows (`rows`, in the head's form, gathered in candidate order) into y[0..n).
    using Exact = std::function<void(const uint8_t* rows, int n, float* y, void* stream)>;

    DraftHeadInt2() = default;
    ~DraftHeadInt2();
    DraftHeadInt2(const DraftHeadInt2&) = delete;
    DraftHeadInt2& operator=(const DraftHeadInt2&) = delete;

    /// The int2 copy of `n_rows` rows of `n_in` values of GGUF type `type`, row i's bytes at `row(i)` (host memory),
    /// and the buffers of the k candidates (`exact_row_bytes` a row of the head's own form).  False (and `err`) when
    /// the type has no host dequantizer or the memory is not there; the caller then keeps its head.
    bool build(int type, int64_t n_rows, int n_in, const std::function<const uint8_t*(int64_t)>& row,
               size_t exact_row_bytes, std::string& err);
    bool built() const { return codes_ != nullptr; }
    /// The device bytes it holds.
    uint64_t bytes() const { return bytes_; }

    /// The logits of the draft head's rows for the float row `x` (device, n_in values): coarse, but the k best, which
    /// `exact` computes from `head_rows` (the head's rows in the same order as the int2 ones).  `pos` (device, the
    /// draft row's position; null: never) at kPadFrom or more leaves the k alone, the rest at a probability of 0.
    void logits(const float* x, const uint8_t* head_rows, const Exact& exact, float* out, void* stream,
                const int32_t* pos = nullptr) const;
    static constexpr int32_t kPadFrom = 49152;

    /// Whether this device drafts faster with it than with `full` (the head's own product of every row into `out`):
    /// both timed here, after a warm-up.  STRATA_DRAFT_INT2=0|1 says it instead.  Logs the times.
    bool choose(const float* x, const uint8_t* head_rows, const Exact& exact, const std::function<void(float*, void*)>& full,
                float* out, void* stream, const char* who);

private:
    uint8_t* codes_ = nullptr;
    int32_t* cand_ = nullptr;
    uint8_t* cand_rows_ = nullptr;
    float* cand_y_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int64_t n_rows_ = 0;
    int n_in_ = 0, k_ = 0;
    size_t exact_row_bytes_ = 0;
};

}  // namespace strata::core
