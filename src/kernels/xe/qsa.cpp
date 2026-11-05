// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/qsa.cpp - the Xe port of Strata's src/kernels/cuda/qsa.cu: the QSA cache's FP16 append and gather and
// the attention over the gathered cells (see include/strata/kernels/qsa.hpp).  The indexer, the selection and the
// output gate are the native kernels' (native_qsa_indexer.hpp, qsa_select.hpp, native_qsa.hpp).  A warp is a sub-group
// of 32.  The CUDA entry points exited on a bad argument; these throw core::DeviceError.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q8.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_qsa
#include "strata/kernels/device_arch_tag.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"

#include <cfloat>
#include <cmath>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;

[[noreturn]] void fail(const std::string& what) { throw core::DeviceError("qsa: " + what); }

sycl::queue& queue_for(void* stream) {
    return core::Runtime::get().stream(stream);
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

/// Geometry validation.  A silently wrong `head_dim % 4` would corrupt the gather's 8-byte copy and a silently
/// wrong `n_head % n_head_kv` would produce a plausible attention with the wrong key, so both are refused.
void validate(const QsaShapes& s, const char* who) {
    const std::string w = who;
    if (s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_dim <= 0 || s.idx_n_head <= 0 ||
        s.idx_block < 2 || s.page_size < 1)
        fail(w + ": geometry is not set up");
    if (s.n_head % s.n_head_kv != 0)
        fail(w + ": n_head " + std::to_string(s.n_head) + " is not a multiple of n_head_kv " + std::to_string(s.n_head_kv));
    if (s.head_dim % 4 != 0)
        fail(w + ": head_dim " + std::to_string(s.head_dim) + " must be a multiple of 4 (the gather copies 8 bytes)");
    if (s.n_rot <= 0 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim)
        fail(w + ": n_rot " + std::to_string(s.n_rot) + " must be even and <= head_dim and idx_dim");
    if (s.idx_n_head > 32) fail(w + ": idx_n_head " + std::to_string(s.idx_n_head) + " > 32 (one warp per indexer head)");
}

inline float h2f(uint16_t bits) { return f32_from_f16(bits); }

inline size_t grid_for(long long n, int threads) { return (size_t) ((n + threads - 1) / threads) * threads; }

// ---- the step state, uploaded (see the CUDA file): a device buffer for each GPU, written on its compute queue.
int32_t* step_scratch() {
    static core::PerDevice<int32_t*> per_device;
    auto& rt = core::Runtime::get();
    return per_device.get(rt.device(), [&rt] {
        auto* p = static_cast<int32_t*>(sycl::malloc_device(qsa_step_bytes(), rt.device(), rt.context()));
        if (!p) fail("step upload: device allocation failed");
        return p;
    });
}

// CUDA copied with a synchronous cudaMemcpy on the legacy stream; the compute queue plays that part.  The copy starts
// after every kernel already queued there (not on other engine streams, as with CUDA's non-blocking streams), and
// waiting for it keeps the host array alive until it has been read.
void step_upload_raw(const int32_t* h_step) {
    auto& rt = core::Runtime::get();
    rt.wait(rt.compute().memcpy(step_scratch(), h_step, qsa_step_bytes()), "qsa step upload");
}

const int32_t* step_upload(int64_t pos, int64_t n_kv_hint, const QsaShapes& s) {
    int32_t h[kStepCount];
    qsa_step_fill(h, pos, s);
    if (n_kv_hint >= 0 && n_kv_hint != (int64_t) h[kStepNKv])
        fail("step upload: n_kv " + std::to_string(n_kv_hint) + " disagrees with pos+1 = " + std::to_string(h[kStepNKv]));
    step_upload_raw(h);
    return step_scratch();
}

const int32_t* step_upload_width(int64_t width, const QsaShapes& s) {
    int32_t h[kStepCount];
    for (int i = 0; i < kStepCount; ++i) h[i] = 0;
    h[kStepWidth] = (int32_t) width;
    (void) s;
    step_upload_raw(h);
    return step_scratch();
}

// the FP16 appends through the cache's one append (kv_q8.hpp: kv_append_cells), its arithmetic f16(x)
void kv_append_steps_host(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                          int step_stride, const float* kcur, const float* vcur, int cur_stride, int n_tok,
                          const QsaShapes& s, void* stream, const KvHostPools* host) {
    if (n_tok == 0) return;
    KvAppend a;
    a.step = step;
    a.step_stride = step_stride;
    a.k = kcur;
    a.v = vcur;
    a.cur_stride = cur_stride;
    a.n_tok = n_tok;
    a.k_pool = k_pool;
    a.v_pool = v_pool;
    a.page_table = page_table;
    a.host = host;
    kv_append_cells(a, s, stream);
}

}  // namespace

// ================= THE CAPTURABLE ENTRY POINTS =================

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    kv_append_steps_host(k_pool, v_pool, page_table, step, 0, kcur, vcur, 0, 1, s, stream, host);
}

void kv_append_steps(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step, int step_stride,
                     const float* kcur, const float* vcur, int cur_stride, int n_tok, const QsaShapes& s, void* stream) {
    validate(s, "kv_append_steps");
    if (step == nullptr || n_tok < 0) fail("kv_append_steps: step is null or n_tok negative");
    kv_append_steps_host(k_pool, v_pool, page_table, step, step_stride, kcur, vcur, cur_stride, n_tok, s, stream,
                         nullptr);
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (step == nullptr) fail("kv_gather: step is null");
    if (max_ids <= 0) return;
    const long long cap_total = max_ids * s.n_head_kv * (s.head_dim / 4);
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const int32_t* table = page_table;
    const auto* kp = reinterpret_cast<const uint64_t*>(k_pool);
    const auto* vp = reinterpret_cast<const uint64_t*>(v_pool);
    auto* ks = reinterpret_cast<uint64_t*>(k_scratch);
    auto* vs = reinterpret_cast<uint64_t*>(v_scratch);
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>(grid_for(cap_total, 256), 256), [=](sycl::nd_item<1> it) {
        const long long n_ids = (long long) step[kStepWidth];
        const int per = head_dim / 4;                       // 4 halves per 8-byte word
        const long long total = n_ids * kv_heads * per;
        const long long i = (long long) it.get_global_id(0);
        if (i >= total) return;
        const long long id = i / (kv_heads * (long long) per);
        const int rem = (int) (i % (kv_heads * (long long) per));
        const int h = rem / per, q = rem - h * per;
        const int cell = ids[id];
        const long long page = (long long) table[cell / page_size];
        const long long src = ((page * kv_heads + h) * page_size + (cell % page_size)) * (long long) per + q;
        const long long dst = (id * kv_heads + h) * (long long) per + q;
        ks[dst] = kp[src];
        vs[dst] = vp[src];
    });
    finish(stream, e, "kv_gather");
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                     const int32_t* step, int64_t max_ids, const QsaShapes& s, float* attn, float* weights,
                     void* stream) {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    auto& queue = queue_for(stream);
    // The local size is the capacity, part of the launch configuration; the kernel reads the real count.
    const size_t smem = (size_t) (max_ids + 32) * sizeof(float);
    const size_t limit = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    if (smem > limit)
        fail("qsa_attend: max_ids " + std::to_string(max_ids) + " needs " + std::to_string(smem) +
             " B of local memory, over the " + std::to_string(limit) + " B limit");
    const int n_head = (int) s.n_head, n_head_kv = (int) s.n_head_kv, head_dim = (int) s.head_dim;
    const auto e = queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> s_w(sycl::range<1>((size_t) max_ids + 32), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) n_head * head_dim, head_dim),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto g = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const long long n_ids = (long long) step[kStepWidth];
            const int h = (int) it.get_group(0);
            const int d = (int) it.get_local_id(0);
            const int bdim = head_dim;
            // An empty selection is handled on the device: the reference returns zeros for an empty cache.
            if (n_ids == 0) {
                for (int i = d; i < head_dim; i += bdim) attn[(size_t) h * head_dim + i] = 0.0f;
                return;
            }
            const int kv = h / (n_head / n_head_kv);       // ops.cpp L8729: iv2 = iq2 / rv2, NOT h % n_head_kv
            const float scale = 1.0f / sycl::sqrt((float) head_dim);
            const int lane = (int) sg.get_local_linear_id(), wid = (int) sg.get_group_linear_id();
            const int nwarp = (bdim + WARP - 1) / WARP;
            auto red = [&](int i) -> float& { return s_w[i]; };
            auto w = [&](long long j) -> float& { return s_w[32 + j]; };

            for (long long j = d; j < n_ids; j += bdim) {
                const uint16_t* krow = k_scratch + (j * n_head_kv + kv) * head_dim;
                float acc = 0.0f;
                for (int i = 0; i < head_dim; ++i) acc += h2f(krow[i]) * q[(size_t) h * head_dim + i];
                w(j) = acc * scale;
            }
            sycl::group_barrier(g);

            float mx = -FLT_MAX;
            for (long long j = d; j < n_ids; j += bdim) mx = sycl::fmax(mx, w(j));
            for (int o = 16; o > 0; o >>= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            if (lane == 0) red(wid) = mx;
            sycl::group_barrier(g);
            if (wid == 0) {
                mx = (lane < nwarp) ? red(lane) : -FLT_MAX;
                for (int o = 16; o > 0; o >>= 1) mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
                if (lane == 0) red(0) = mx;
            }
            sycl::group_barrier(g);
            mx = red(0);
            sycl::group_barrier(g);

            float sum = 0.0f;
            for (long long j = d; j < n_ids; j += bdim) {
                const float ex = sycl::exp(w(j) - mx);
                w(j) = ex;
                sum += ex;
            }
            for (int o = 16; o > 0; o >>= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
            if (lane == 0) red(wid) = sum;
            sycl::group_barrier(g);
            if (wid == 0) {
                sum = (lane < nwarp) ? red(lane) : 0.0f;
                for (int o = 16; o > 0; o >>= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                if (lane == 0) red(0) = 1.0f / sum;
            }
            sycl::group_barrier(g);
            const float inv = red(0);
            sycl::group_barrier(g);

            float acc = 0.0f;
            for (long long j = 0; j < n_ids; ++j)
                acc += (w(j) * inv) * h2f(v_scratch[(j * n_head_kv + kv) * head_dim + d]);
            attn[(size_t) h * head_dim + d] = acc;
            if (weights != nullptr)
                for (long long j = d; j < n_ids; j += bdim) weights[(size_t) h * n_ids + j] = w(j) * inv;
        });
    });
    finish(stream, e, "qsa_attend");
}

// ================= host entry points =================

void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) fail("qsa_step_fill: pos < 0");
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    if (pos < 0) { validate(s, "kv_append"); fail("kv_append: pos < 0"); }
    kv_append_step(k_pool, v_pool, page_table, step_upload(pos, -1, s), kcur, vcur, s, stream);
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (n_ids <= 0) return;
    kv_gather_step(k_pool, v_pool, page_table, ids, step_upload_width(n_ids, s), n_ids, s, k_scratch,
                   v_scratch, stream);
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    validate(s, "qsa_attend");
    if (n_ids < 0) fail("qsa_attend: n_ids < 0");
    if (n_ids == 0) {
        qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(0, s),
                        qsa_selection_width(kTopkMaxCells, s), s, attn, weights, stream);
        return;
    }
    qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(n_ids, s), n_ids, s, attn, weights, stream);
}

}  // namespace strata::kernels
