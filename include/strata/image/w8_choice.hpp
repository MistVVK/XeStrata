// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
// Whether a DiT takes its 8-bit form or its FP16 one, the two fitting the VRAM alike: the one choice every image DiT
// makes (Mage-Flow's, MOSS-Image's and Anima's pipelines at each request, Qwen-Image's and SeedVR2's when they open:
// dit_rows::w8_at_open), and what the engine tells the server of the GPU (w8_here).  Whether the FP16 form fits is
// the model's own sum and stays with it.

#include "strata/platform/env.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>

namespace strata::image {

class W8Choice {
public:
    /// The seconds of the DiT's MLP's first product at `rows` rows in FP16 and in W8A8 (dit_rows::time_w8_f16; 0 a
    /// form that does not run here).
    using Timer = std::function<std::pair<double, double>(int64_t rows)>;

    /// Whether the DiT (`what`, for the log) takes W8A8 (the rows to int8, prefill::Gemm::i8) at `rows` image rows:
    /// unless it is clearly the slower here, its MLP's product timed both ways at the first request's rows and again
    /// once a request has more than twice as many.  Never where the GPU's library takes no int8 products or
    /// XESTRATA_NO_W8A8 is set.  The RX 9060 XT (2026-10-27): W8A8 1.04 s for Mage-Flow Turbo's 4 steps at 1024 x 1024
    /// against FP16's 1.36 (rocBLAS's FP16 products with FP32 sums).
    bool take(int64_t rows, const char* what, void* stream, const Timer& time) const {
        if (strata::env_get("NO_W8A8") != nullptr || !prefill::Gemm::i8_usable(stream)) {
            faster_ = 0;
            return false;
        }
        if (faster_ < 0 || rows > 2 * rows_) {
            const auto [f16, w8] = time(rows);
            if (f16 > 0.0 || w8 > 0.0) {   // (neither ran: timed again at the next request)
                // 8 bits unless 5% slower or more: a near tie stays put
                faster_ = w8 > 0.0 && (f16 <= 0.0 || w8 < f16 * 1.05) ? 1 : 0;
                rows_ = rows;
                std::fprintf(stderr, "image engine: %s in %s (its MLP's product at %lld rows: %.2f ms in FP16, %.2f in "
                                     "W8A8)\n", what, faster_ ? "8 bits" : "16 bits", (long long) rows, 1e3 * f16,
                             1e3 * w8);
            }
        }
        return faster_ == 1;
    }

    /// The form of a request's DiT, 8 bits (true) or FP16, in Mage-Flow's, MOSS-Image's and Anima's pipelines:
    /// `forced` (--dit-q8: 1 8 bits, 0 FP16, -1 auto), else 8 bits where `fp16_need` passes `budget`, else take()'s.
    /// `fp16_need` is the model's own sum: the FP16 blocks all resident beside what else the request keeps, the text
    /// encoder counted as the pipeline's plan() keeps it open, whether it is open yet or not.  Counted only once
    /// open, the first request (the encoder not open yet) chose FP16 and the next 8 bits, which opened the DiT again
    /// in the other form inside the second request's text stage (MOSS-Image: 5.5 s of its 5.6, RX 9060 XT,
    /// 2026-11-12).
    bool want_q8(int forced, uint64_t fp16_need, uint64_t budget, int64_t rows, const char* what, void* stream,
                 const Timer& time) const {
        if (forced >= 0) return forced == 1;
        if (fp16_need > budget) return true;
        return take(rows, what, stream, time);
    }

    /// Whether take() has an answer: false while neither form could be timed (no room for the timing's buffers).
    bool decided() const { return faster_ >= 0; }

private:
    mutable int faster_ = -1;      // the last timing's: W8A8 the faster (1), not (0: or not taken here), not timed (-1)
    mutable int64_t rows_ = 0;     // the rows it was timed at
};

/// Whether this GPU takes the image DiTs' W8A8 form, for the server to choose a weights file by before any DiT opens
/// (SeedVR2's int8_convrot or FP16 file, serve/images.py's seedvr2_file; the engine's W8 line): dit_rows::w8_at_open's
/// answer for SeedVR2's 7B's MLP, 3072 -> 12288, kept for the GPU as that DiT's own opening finds it.  1 W8A8, 0 FP16,
/// -1 neither form could be timed (asked again).
int w8_here(void* stream);

}  // namespace strata::image
