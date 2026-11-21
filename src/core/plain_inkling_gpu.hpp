// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::core {

// First GPU path: stream selected experts, keeping the model's FP32 reference state on the host.
class InklingProducts {
    struct Free { void operator()(void* p) const { if (p) gpu::free(p); } };
    template<class T> using Buffer = std::unique_ptr<T, Free>;
    template<class T> static Buffer<T> alloc(size_t n) {
        Buffer<T> p(static_cast<T*>(gpu::alloc_device(n * sizeof(T))));
        if (!p) throw std::runtime_error("inkling2: GPU product buffer allocation failed");
        return p;
    }
    struct Stream {
        gpu::Stream s = gpu::stream_create();
        ~Stream() { if (s) { gpu::stream_sync(s); gpu::stream_destroy(s); } }
    } stream_;
    Buffer<uint8_t> raw_;
    Buffer<uint16_t> weights_, input_;
    Buffer<float> result_;
    Buffer<int32_t> bounds_, positions_;
    Buffer<float> aq_, ak_, av_, ar_, at_, asink_, ascores_, ao_, taps_;
    std::vector<uint16_t> input_host_;
    std::vector<int32_t> bounds_host_;
public:
    InklingProducts(int64_t groups, int64_t width, int64_t ff, size_t raw_bytes, int64_t heads, int64_t kv, int64_t dim, int64_t rel, int64_t extent)
        : raw_(alloc<uint8_t>(raw_bytes)), weights_(alloc<uint16_t>((size_t) (groups * 2 * ff * width))),
          input_(alloc<uint16_t>((size_t) (groups * std::max(width, ff)))),
          result_(alloc<float>((size_t) (groups * std::max(2 * ff, width)))),
          bounds_(alloc<int32_t>((size_t) groups + 1)), positions_(alloc<int32_t>((size_t) extent)),
          aq_(alloc<float>((size_t) (heads * dim))), ak_(alloc<float>((size_t) (extent * kv * dim))),
          av_(alloc<float>((size_t) (extent * kv * dim))), ar_(alloc<float>((size_t) (heads * rel))),
          at_(alloc<float>((size_t) (extent * rel))), asink_(alloc<float>((size_t) heads)),
          ascores_(alloc<float>((size_t) (extent * heads))), ao_(alloc<float>((size_t) (heads * dim))),
          taps_(alloc<float>((size_t) width * 4)),
          input_host_((size_t) (groups * std::max(width, ff))), bounds_host_((size_t) groups + 1) {
        if (!stream_.s) throw std::runtime_error("inkling2: GPU stream creation failed");
        for (int32_t i = 0; i <= groups; ++i) bounds_host_[(size_t) i] = i;
        transfer(bounds_.get(), bounds_host_.data(), bounds_host_.size() * sizeof(int32_t));
    }

    void grouped(int type, const uint8_t* source, size_t bytes_per_expert, const int64_t* ids, int groups,
                 const float* x, bool broadcast, int64_t in, int64_t out, float* y) {
        const auto stream = stream_.s;
        for (int g = 0; g < groups; ++g) {
            // The native dequantizer consumes raw GGUF blocks; the per-expert scalar is applied at the caller.
            auto* raw = raw_.get() + (size_t) g * bytes_per_expert;
            upload(raw, source + (size_t) ids[g] * bytes_per_expert, bytes_per_expert);
            kernels::iq_dequant_f16(type, raw, in * out, weights_.get() + g * in * out, stream);
            for (int64_t i = 0; i < in; ++i)
                input_host_[(size_t) (g * in + i)] = kernels::f16_from_f32(x[(broadcast ? 0 : g * in) + i]);
        }
        upload(input_.get(), input_host_.data(), (size_t) (groups * in) * sizeof(uint16_t));
        if (!prefill::Gemm::f16_groups(input_.get(), weights_.get(), in * out, result_.get(), bounds_host_.data(),
                                      groups, out, in, stream))
            kernels::xmx_gemm_grouped(input_.get(), weights_.get(), in * out, result_.get(), bounds_.get(),
                                      groups, 1, out, in, stream);
        transfer(y, result_.get(), (size_t) (groups * out) * sizeof(float));
    }
    void attention(const float* q, const float* k, const float* v, const float* rel, const float* table,
                   const float* sink, const int32_t* positions, int64_t pos, int64_t cells, int64_t heads,
                   int64_t kv, int64_t dim, int64_t rdim, int64_t extent, int64_t window, float* out) {
        upload(aq_.get(), q, (size_t) (heads * dim) * sizeof(float));
        if (cells > 0) {
            upload(ak_.get(), k, (size_t) (cells * kv * dim) * sizeof(float));
            upload(av_.get(), v, (size_t) (cells * kv * dim) * sizeof(float));
            upload(positions_.get(), positions, (size_t) cells * sizeof(int32_t));
        }
        upload(ar_.get(), rel, (size_t) (heads * rdim) * sizeof(float));
        upload(at_.get(), table, (size_t) (extent * rdim) * sizeof(float));
        upload(asink_.get(), sink, (size_t) heads * sizeof(float));
        kernels::relative_prompt_attn(aq_.get(), ak_.get(), av_.get(), ar_.get(), at_.get(), asink_.get(), positions_.get(),
                                     pos, cells, heads, kv, dim, rdim, extent, window, ascores_.get(), ao_.get(), stream_.s);
        transfer(out, ao_.get(), (size_t) (heads * dim) * sizeof(float));
    }
    void capture(const float* p, size_t elements) { transfer(taps_.get(), p, elements * sizeof(float)); }
    const float* taps() const { return taps_.get(); }

private:
    void upload(void* dst, const void* src, size_t bytes) {
        if (!gpu::copy_async(dst, src, bytes, stream_.s))
            throw std::runtime_error("inkling2: GPU product copy failed");
    }
    void transfer(void* dst, const void* src, size_t bytes) {
        upload(dst, src, bytes);
        if (!gpu::stream_sync(stream_.s)) throw std::runtime_error("inkling2: GPU product stream failed");
    }
};

}  // namespace strata::core
