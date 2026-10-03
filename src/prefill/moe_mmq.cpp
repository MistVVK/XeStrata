// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The Xe prompt products keep GGUF weights quantized and share the decode kernels' integer dots.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/runtime.hpp"

#include <stdexcept>

namespace strata::prefill::mmq {
namespace {
struct Q8 { sycl::half2 ds; int8_t qs[32]; };
sycl::queue& queue(void* stream) {
    if (!stream) throw std::invalid_argument("prompt MMQ requires an explicit stream");
    return core::Runtime::get().stream(stream);
}
}
bool built() { return true; }
bool supported(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 42 ||
           t == 12 || t == 13 || t == 7 || t == 8 || t == 6;
}
bool fits(int t, int64_t rows) { return supported(t) && rows > 0; }
size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    const size_t row = strata::kernels::iq_row_bytes(t, cols);
    if (!supported(t) || rows <= 0 || row == 0) throw std::invalid_argument("prompt MMQ: invalid matrix");
    return (size_t) rows * row;
}
size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) ((cols + 511) / 512 * 16) * sizeof(Q8);
}
void quantize(const float* x, const int32_t* ids, void* xq, int type, int64_t cols, int64_t ld, int64_t rows,
              void* stream) {
    if (!supported(type) || !x || !xq || rows < 0 || cols <= 0 || cols % 32 || ld < cols)
        throw std::invalid_argument("prompt MMQ: invalid activation matrix");
    if (!rows) return;
    const int64_t padded = (cols + 511) / 512 * 512;
    auto* y = static_cast<Q8*>(xq);
    queue(stream).parallel_for(sycl::nd_range<1>((size_t) (rows*padded), 256),
                              [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t i = (int64_t) it.get_global_linear_id(), r = i / padded, c = i % padded;
        const auto sg = it.get_sub_group();
        const float v = c < cols ? x[(ids ? ids[r] : r) * ld + c] : 0.0f;
        float mx = sycl::fabs(v), sum = v;
        for (int o = 16; o; o >>= 1) {
            mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            sum += sycl::permute_group_by_xor(sg, sum, o);
        }
        const float d = mx / 127.0f;
        y[i/32].qs[i%32] = mx > 0 ? (int8_t) sycl::round(v/d) : 0;
        if (i%32 == 0) y[i/32].ds = sycl::half2(sycl::half(d), sycl::half(sum));
    });
}
void gather_native(const void* gate, const void* up, size_t half, const void* down, size_t db,
                   void* gu, void* dn, void* stream) {
    auto& q = queue(stream);
    q.memcpy(gu, gate, half); q.memcpy(static_cast<uint8_t*>(gu)+half, up, half); q.memcpy(dn, down, db);
}
bool gather_native_group(const GatherGroup& g, size_t up, size_t half, size_t down, size_t db,
                         void* gu, size_t gs, void* dn, size_t ds, void* stream) {
    if (!gu || !dn || g.first < 0 || g.n < g.first || g.n > kGatherGroupMax ||
        !half || !db || half % 16 || db % 16 || up % 16 || down % 16 || gs % 16 || ds % 16 ||
        reinterpret_cast<uintptr_t>(gu) % 16 || reinterpret_cast<uintptr_t>(dn) % 16 || gs < 2*half || ds < db) return false;
    for (int i=g.first; i<g.n; ++i) if (!g.blob[i] || reinterpret_cast<uintptr_t>(g.blob[i]) % 16) return false;
    const size_t chunks=(2*half+db)/16, total=(size_t) (g.n-g.first)*chunks;
    if (total) queue(stream).parallel_for(sycl::nd_range<1>((total+255)/256*256,256),
                                          [=](sycl::nd_item<1> it) {
        const size_t i=it.get_global_linear_id();
        if (i>=total) return;
        const size_t e=i/chunks+(size_t) g.first, off=i%chunks*16;
        const uint8_t* from=g.blob[e]+(off<half ? off : off<2*half ? up+off-half : down+off-2*half);
        uint8_t* to=off<2*half ? static_cast<uint8_t*>(gu)+e*gs+off : static_cast<uint8_t*>(dn)+e*ds+off-2*half;
        for (int j=0; j<4; ++j) reinterpret_cast<uint32_t*>(to)[j]=reinterpret_cast<const uint32_t*>(from)[j];
    });
    return true;
}
void gather_strata_q2(const uint8_t* blob, void* gu, void* dn, void* stream) {
    struct Q2 { sycl::half d; uint8_t qs[16]; };
    auto& q = queue(stream);
    q.parallel_for(sycl::range<1>(size_t(1280)*40+size_t(2560)*10), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const bool gate = i < size_t(1280)*40;
        const size_t b = gate ? i : i-size_t(1280)*40;
        auto* out = static_cast<Q2*>(gate ? gu : dn)+b;
        const size_t code = (gate ? 0 : size_t(1280)*640)+b*16;
        const size_t scale = size_t(1280)*640+size_t(2560)*160+(gate ? b : size_t(1280)*40+b)*2;
        out->d = *reinterpret_cast<const sycl::half*>(blob+scale);
        for (int j=0; j<16; ++j) out->qs[j] = blob[code+j];
    });
}
void swiglu(const float* gu, float* h, int64_t rows, int64_t ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    queue(stream).parallel_for(sycl::range<1>((size_t) (rows*ff)), [=](sycl::id<1> id) {
        const int64_t i=(int64_t) id[0], r=i/ff, f=i%ff;
        const float g=gu[r*2*ff+(interleaved ? 2*f : f)];
        const float u=gu[r*2*ff+(interleaved ? 2*f+1 : ff+f)];
        h[i]=(g/(1.0f+sycl::exp(-g)))*u;
    });
}
void iota(int32_t* dst, int64_t n, void* stream) {
    if (n > INT32_MAX) throw std::invalid_argument("prompt MMQ: row index exceeds int32");
    if (n > 0) queue(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { dst[id]=(int32_t) id[0]; });
}
}  // namespace strata::prefill::mmq
