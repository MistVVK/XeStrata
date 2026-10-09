// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <algorithm>
#include <limits>

namespace strata::kernels {
namespace {
sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
// 16 bytes in one read across PCIe from mapped host memory: through sycl::float4 NVPTX splits the read in two
using f32x4 = float __attribute__((ext_vector_type(4)));
void sync_if_needed(void* stream, const sycl::event& event) {
    if (!stream) core::Runtime::get().wait(event, "synchronous elementwise kernel");
}
}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    if ((code_bits != 2 && code_bits != 4 && code_bits != 8) || group_elems <= 0)
        throw core::DeviceError("invalid packed embedding geometry");
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> index) {
        const int64_t i = index[0];
        const int per_byte = 8 / code_bits;
        const unsigned mask = (1u << code_bits) - 1u;
        const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
        const float product = static_cast<float>(code + code_bias) * scales[i / group_elems];
        // This translation unit disables contraction to preserve the packed-row format's rounding.
        out[i] = product + (offsets ? offsets[i / group_elems] : 0.0f);
    });
    sync_if_needed(stream, event);
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate,
              int64_t n_tokens, int64_t h_v, void* stream) {
    if (n_tokens <= 0) return;
    if (h_v <= 0 || n_tokens > std::numeric_limits<int64_t>::max() / h_v)
        throw core::DeviceError("invalid gate dimensions");
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n_tokens * h_v), [=](sycl::id<1> index) {
        const int64_t i = index[0], h = i % h_v;
        const float x = alpha[i] + dt[h];
        gate[i] = (x > 20.0f ? x : sycl::log1p(sycl::exp(x))) * ssm_a[h];
    });
    sync_if_needed(stream, event);
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] *= s; });
    sync_if_needed(stream, event);
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i] = f16_from_f32(x[i]);
    });
    sync_if_needed(stream, event);
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        // The negative branch avoids overflow; FP32 is checked against the existing double oracle.
        const float v = x[i];
        const float e = sycl::exp(-sycl::fabs(v));
        x[i] = v >= 0.0f ? v / (1.0f + e) : (v * e) / (1.0f + e);
    });
    sync_if_needed(stream, event);
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0) return;
    if (cols <= 0 || rows > std::numeric_limits<int64_t>::max() / cols)
        throw core::DeviceError("invalid RMS dimensions");
    constexpr size_t local = 128;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rows * local), sycl::range<1>(local)), [=](sycl::nd_item<1> item) {
            const int64_t row = item.get_group_linear_id(), lane = item.get_local_linear_id();
            float sum = 0.0f;
            for (int64_t col = lane; col < cols; col += local) {
                const float v = x[row * cols + col];
                sum += v * v;
            }
            sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(cols) + eps);
            for (int64_t col = lane; col < cols; col += local) {
                const int64_t i = row * cols + col;
                x[i] = (w ? x[i] * w[col] : x[i]) * inv;
            }
        });
    sync_if_needed(stream, event);
}
void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] += src[i]; });
    sync_if_needed(stream, event);
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        y[i] = bf16_from_f32(x[i]);
    });
    sync_if_needed(stream, event);
}

// ---- the doorbell (see the header).  bench/results/2026-09-30-xe-doorbell measured what a spinning kernel sees:
// a volatile read followed by a system-scope acquire fence observes the host's write; a system-scope atomic load
// alone sometimes never does.  The GPU-to-host direction is a system-scope release fence, then the store.
namespace {
using sys_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;
inline uint32_t read_host(const uint32_t* p) {
    const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(p);
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    return v;
}
}  // namespace

// *seq += 1, read and incremented in memory so a captured graph advances it on every replay
void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    const auto event = queue_for(stream).single_task([=] {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sys_ref(*d_seq).store(read_host(d_seq) + 1u);
    });
    sync_if_needed(stream, event);
}

// Spins until the host's flag equals the ring, then everything the host wrote before the flag is visible to the
// commands that follow.  Unbounded, as in CUDA: a stalled host is reported by the runtime's watchdog on the next
// host wait, not by the kernel.
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    queue_for(stream).single_task([=] {
        const uint32_t want = read_host(d_seq);
        while (read_host(d_flag) != want) {
        }
    });
}

void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0)
        throw core::DeviceError("copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned");
    const int64_t n4 = n / 4;
    const size_t groups = (size_t) std::min<int64_t>((n4 + 255) / 256, 64);
    queue_for(stream).parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t step = (int64_t) it.get_global_range(0);
        for (int64_t i = (int64_t) it.get_global_id(0); i < n4; i += step)
            reinterpret_cast<f32x4*>(dst)[i] = reinterpret_cast<const volatile f32x4*>(src)[i];
    });
}

// the CPU rows of a verify window, +0.0 for the rows the GPU plan computes itself (hit_rows[0, *count))
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0)
        throw core::DeviceError("copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned");
    // 16-byte reads and the hit test shared by the group: the RTX 4070 took 33.8 us for 10 rows against 20.5
    const int64_t w4 = width / 4;
    queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) rows * 128, 128), [=](sycl::nd_item<1> it) {
        const int64_t row = (int64_t) it.get_group(0);
        const int c = *count;
        bool h = false;
        for (int i = (int) it.get_local_id(0); i < c; i += 128) h |= hit_rows[i] == row;
        const bool hit = sycl::any_of_group(it.get_group(), h);
        f32x4* d = reinterpret_cast<f32x4*>(dst + row * width);
        const volatile f32x4* sr = reinterpret_cast<const volatile f32x4*>(src + row * width);
        for (int64_t i = (int64_t) it.get_local_id(0); i < w4; i += 128) d[i] = hit ? f32x4(0.0f) : sr[i];
    });
}

// the routing payload and its ring in one kernel: copy, fence, then the increment the host polls
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) throw core::DeviceError("doorbell_publish: k too large");
    const int ni = (int) n, ki = (int) k;
    auto& q = queue_for(stream);
    const int wg = xe::work_group_upto_1024(q);
    q.parallel_for(sycl::nd_range<1>((size_t) wg, (size_t) wg), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_local_id(0);
        for (int i = t; i < ni; i += wg) x_out[i] = x[i];
        for (int i = t; i < ki; i += wg) { ids_out[i] = ids[i]; weights_out[i] = weights[i]; }
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (t == 0) {
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_ref(*d_seq).store(read_host(d_seq) + 1u);
        }
    });
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        dst[i] = reinterpret_cast<const volatile int32_t*>(src)[i];
    });
}

namespace {
struct MappedCopies { MappedCopy c[8]; };
}  // namespace

// upstream 4a5713fb: the draft graphs' inputs from mapped memory in one launch instead of one each
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    if (n <= 0) return;
    if (n > 8) throw core::DeviceError("copy_from_mapped_multi: at most 8 copies");
    MappedCopies a{};
    int64_t most = 0;
    for (int i = 0; i < n; ++i) {
        a.c[i] = copies[i];
        most = copies[i].words > most ? copies[i].words : most;
    }
    const int64_t units = (most + 3) / 4;
    const size_t bx = (size_t) std::clamp<int64_t>((units + 255) / 256, 1, 32);
    queue_for(stream).parallel_for(sycl::nd_range<2>({(size_t) n, bx * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
        const MappedCopy c = a.c[it.get_group(0)];
        const int64_t i0 = (int64_t) it.get_global_id(1), st = (int64_t) it.get_global_range(1);
        if ((c.words & 3) == 0 && ((uintptr_t) c.dst & 15) == 0 && ((uintptr_t) c.src & 15) == 0) {
            const volatile sycl::uint4* s = static_cast<const volatile sycl::uint4*>(c.src);
            sycl::uint4* d = static_cast<sycl::uint4*>(c.dst);
            for (int64_t i = i0; i < c.words / 4; i += st) d[i] = const_cast<const sycl::uint4*>(s)[i];
        } else {
            const volatile int32_t* s = static_cast<const volatile int32_t*>(c.src);
            int32_t* d = static_cast<int32_t*>(c.dst);
            for (int64_t i = i0; i < c.words; i += st) d[i] = s[i];
        }
    });
}
}  // namespace strata::kernels
