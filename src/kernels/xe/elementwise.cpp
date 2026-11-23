// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/f16_gpu.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_elementwise
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>

namespace strata::kernels {

class DoorbellPublishKernel;   // asked for its work-group (work_group_upto_1024_of)

namespace {
sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
// 16 bytes in one read across PCIe from mapped host memory: through sycl::float4 NVPTX splits the read in two
using f32x4 = float __attribute__((ext_vector_type(4)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
void sync_if_needed(void* stream, const sycl::event& event) {
    if (!stream) core::Runtime::get().wait(event, "synchronous elementwise kernel");
}
}  // namespace

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] *= s; });
    sync_if_needed(stream, event);
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i] = f16_gpu::f16_from_f32(x[i]);
    });
    sync_if_needed(stream, event);
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream, float a) {
    if (n <= 0) return;
    // a src is exact for a = 1, so the residual's plain sum keeps its bits
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n),
                                                      [=](sycl::id<1> i) { dst[i] += a * src[i]; });
    sync_if_needed(stream, event);
}

// ---- the doorbell (see the header).  the 2026-09-30-xe-doorbell record measured what a spinning kernel sees:
// a volatile read followed by a system-scope acquire fence observes the host's write; a system-scope atomic load
// alone sometimes never does.  The GPU-to-host direction is a system-scope fence, then a plain volatile store,
// not an atomic one: no GPU here reports usm_atomic_host_allocations, and the xe driver serves a discrete
// GPU's atomic page fault on system memory only by moving the page to VRAM (a userptr cannot move), where a read or
// a write is simply bound again (xe_vma_need_vram_for_atomic).  The ring has one writer, so it needs no atomicity;
// the B70 saw the store while the kernel ran in 2,000 of 2,000 runs, sooner than the atomic one (median 4.9 us
// against 11.0; the 2026-10-11-xe-doorbell-store record).
namespace {
inline uint32_t read_host(const uint32_t* p) {
    const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(p);
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    return v;
}
// *p += 1, with one fence between the load and the store: a release fence right after read_host's acquire fence
// crashed intel/llvm 7.1's sycl-optimize-barriers pass (an assertion in CallBase::setArgOperand).
inline void bump_host(uint32_t* p) {
    volatile uint32_t* v = p;
    const uint32_t n = *v + 1u;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    *v = n;
}
}  // namespace

// *seq += 1, read and incremented in memory so a captured graph advances it on every replay
void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    const auto event = queue_for(stream).single_task([=] {
        bump_host(d_seq);
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

void concat_rows(float* dst, const float* a, const float* b, int64_t n, int64_t rows, void* stream) {
    if (n <= 0 || rows <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) a & 15) != 0 || ((uintptr_t) b & 15) != 0)
        throw core::DeviceError("concat_rows: n must be a multiple of 4 and the pointers 16-byte aligned");
    const int64_t n4 = n / 4;
    queue_for(stream).parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t row = (int64_t) it.get_group(0);
        f32x4* d = reinterpret_cast<f32x4*>(dst + row * 2 * n);
        const f32x4* sa = reinterpret_cast<const f32x4*>(a + row * n);
        const f32x4* sb = reinterpret_cast<const f32x4*>(b + row * n);
        for (int64_t i = (int64_t) it.get_local_id(0); i < n4; i += 256) {
            d[i] = sa[i];
            d[n4 + i] = sb[i];
        }
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
    const int wg = xe::work_group_upto_1024_of<DoorbellPublishKernel>(q);
    q.parallel_for<DoorbellPublishKernel>(sycl::nd_range<1>((size_t) wg, (size_t) wg), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_local_id(0);
        for (int i = t; i < ni; i += wg) x_out[i] = x[i];
        for (int i = t; i < ki; i += wg) { ids_out[i] = ids[i]; weights_out[i] = weights[i]; }
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (t == 0) {
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            bump_host(d_seq);
        }
    });
}

void mapped_flag_store(uint32_t* flag, uint32_t value, void* stream) {
    queue_for(stream).single_task([=] {
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        reinterpret_cast<volatile uint32_t*>(flag)[0] = value;
    });
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if (((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            dst[i] = reinterpret_cast<const volatile int32_t*>(src)[i];
        });
        return;
    }
    // 16 bytes a read, as copy_from_mapped, and the last n % 4 values one by one: the prompt path's routing tables
    // cross PCIe beside the streamed experts
    const int64_t n4 = n / 4;
    const size_t groups = (size_t) std::clamp<int64_t>((n4 + 255) / 256, 1, 64);
    queue_for(stream).parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t step = (int64_t) it.get_global_range(0), g = (int64_t) it.get_global_id(0);
        for (int64_t i = g; i < n4; i += step)
            reinterpret_cast<i32x4*>(dst)[i] = reinterpret_cast<const volatile i32x4*>(src)[i];
        if (g < n - n4 * 4) dst[n4 * 4 + g] = reinterpret_cast<const volatile int32_t*>(src)[n4 * 4 + g];
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
