// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/qsa.cpp - the Xe port of Strata's src/kernels/cuda/qsa.cu: the QSA cache, indexer and attention (see
// include/strata/kernels/qsa.hpp and the CUDA file for the four kernel decisions this keeps).
//
// The pooling arithmetic stays in double and order-fixed; the library's -ffp-contract=off keeps each add and multiply
// separately rounded, which CUDA spelled with __dadd_rn/__dmul_rn.  A warp is a sub-group of 32.  The CUDA entry
// points exited on a bad argument; these throw core::DeviceError.
#include "strata/kernels/qsa.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
constexpr int WARP = 32;
constexpr int TOPK_THREADS = 256;

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

/// Total order over f32 as an unsigned key (see the CUDA file): -0.0 equals +0.0, a NaN sorts below every real key.
inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

inline size_t grid_for(long long n, int threads) { return (size_t) ((n + threads - 1) / threads) * threads; }

// ---- the step state, uploaded (see the CUDA file): one process-wide device buffer written on the compute queue.
int32_t* step_scratch() {
    static int32_t* d_step = [] {
        auto& rt = core::Runtime::get();
        auto* p = static_cast<int32_t*>(sycl::malloc_device(qsa_step_bytes(), rt.device(), rt.context()));
        if (!p) fail("step upload: device allocation failed");
        return p;
    }();
    return d_step;
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

}  // namespace

// ================= THE CAPTURABLE ENTRY POINTS =================

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    const long long n = s.n_head_kv * s.head_dim;
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const KvHostPools hp = host ? *host : KvHostPools{};
    const int32_t* table = page_table;
    const auto e = queue_for(stream).parallel_for(sycl::nd_range<1>(grid_for(n, THREADS), THREADS), [=](sycl::nd_item<1> it) {
        const long long pos = (long long) step[kStepPos];
        const int i = (int) it.get_global_id(0);
        if (i >= kv_heads * head_dim) return;
        const int h = i / head_dim, d = i - h * head_dim;
        // `[page][kv_head][page_size][head_dim]`; KV streaming: the host copy always, the VRAM page if resident.
        const long long page = (long long) table[pos / page_size];
        if (page >= 0) {
            const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
            k_pool[row * head_dim + d] = f16_from_f32(kcur[i]);
            v_pool[row * head_dim + d] = f16_from_f32(vcur[i]);
        }
        if (hp.k_pool != nullptr) {
            const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
            hp.k_pool[row * head_dim + d] = f16_from_f32(kcur[i]);
            hp.v_pool[row * head_dim + d] = f16_from_f32(vcur[i]);
        }
    });
    finish(stream, e, "kv_append");
}

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    // The grid is `max_blocks`, a constant; a group past this token's completed-block count does nothing.
    const int threads = WARP * (int) s.idx_n_head;
    const int idx_n_head = (int) s.idx_n_head, idx_dim = (int) s.idx_dim;
    const long long r = s.idx_block;
    auto& queue = queue_for(stream);
    auto launch = [&](auto zero) {   // D: double where the device has FP64, float elsewhere (device_caps.hpp)
        using D = decltype(zero);
        return queue.submit([&](sycl::handler& hd) {
            sycl::local_accessor<D, 1> s_dot(sycl::range<1>(32), hd);
            hd.parallel_for(sycl::nd_range<1>((size_t) max_blocks * threads, threads),
                            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
                const long long n_bid = (long long) step[kStepNBid];
                const long long n_kv = (long long) step[kStepNKv];
                const long long b = (long long) it.get_group(0);
                if (b > n_bid) return;                       // uniform across the group
                const sycl::sub_group sg = it.get_sub_group();
                const int wid = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
                if (wid == 0) s_dot[lane] = (D) 0;             // the first work-item reads all idx_n_head entries
                sycl::group_barrier(it.get_group());
                D acc = (D) 0;
                for (int d = lane; d < idx_dim; d += WARP)
                    acc = acc + (D) pooled[(size_t) b * idx_dim + d] * (D) q_idx[(size_t) wid * idx_dim + d];
                for (int o = 16; o > 0; o >>= 1) acc = acc + sycl::permute_group_by_xor(sg, acc, o);
                if (lane == 0) s_dot[wid] = acc;
                sycl::group_barrier(it.get_group());
                if (it.get_local_id(0) != 0) return;
                D score = (D) 0;
                for (int h = 0; h < idx_n_head; ++h) score += (s_dot[h] > (D) 0) ? s_dot[h] : (D) 0;
                if (bias != nullptr) score += (D) bias[b];
                long long lo = b * r, hi = lo + r;
                if (b == n_bid) hi = n_kv;
                if (hi > n_kv) hi = n_kv;
                // llama-memory-hybrid-idx.cpp::set_input_qsa: incomplete-tail cells get a finite +1e9 bias.
                float sc = (float) score;
                if (b == n_bid && n_kv % r != 0) sc += 1e9f;
                for (long long j = lo; j < hi; ++j) cell_scores[j] = sc;
            });
        });
    };
    const auto e = xe::has_fp64(queue) ? launch(0.0) : launch(0.0f);
    finish(stream, e, "qsa_index");
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max)
        fail("topk_512: cap " + std::to_string(cap) + " < the largest possible selection width " + std::to_string(width_max));
    const float* scores = cell_scores;
    int32_t* out_ids = ids;
    const auto e = queue_for(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> s_a(sycl::range<1>(TOPK_THREADS), hd);
        sycl::local_accessor<int, 1> s_b(sycl::range<1>(TOPK_THREADS), hd);
        sycl::local_accessor<long long, 1> s_cgt(sycl::range<1>(1), hd);
        hd.parallel_for(sycl::nd_range<1>(TOPK_THREADS, TOPK_THREADS), [=](sycl::nd_item<1> it) {
            const auto g = it.get_group();
            const long long n_kv = (long long) step[kStepNKv];
            const long long width = (long long) step[kStepWidth];
            const int t = (int) it.get_local_id(0);
            const long long chunk = (n_kv + TOPK_THREADS - 1) / TOPK_THREADS;
            const long long lo = (long long) t * chunk;
            long long hi = lo + chunk;
            if (hi > n_kv) hi = n_kv;

            // ---- binary lifting: the largest key v with count(key >= v) >= width
            uint32_t v = 0u;
            for (int bit = 31; bit >= 0; --bit) {
                const uint32_t cand = v | (1u << bit);
                int c = 0;
                for (long long j = lo; j < hi; ++j) if (order_key(scores[j]) >= cand) ++c;
                s_a[t] = c;
                sycl::group_barrier(g);
                for (int sh = TOPK_THREADS / 2; sh > 0; sh >>= 1) {
                    if (t < sh) s_a[t] += s_a[t + sh];
                    sycl::group_barrier(g);
                }
                const int tot = s_a[0];
                sycl::group_barrier(g);
                if (tot >= (int) width) v = cand;
            }
            const uint32_t thr = v;

            // ---- counts and their exclusive prefixes, by the first work-item in one walk
            int gt = 0, eq = 0;
            for (long long j = lo; j < hi; ++j) {
                const uint32_t k = order_key(scores[j]);
                if (k > thr) ++gt;
                else if (k == thr) ++eq;
            }
            s_a[t] = gt;
            s_b[t] = eq;
            sycl::group_barrier(g);
            if (t == 0) {
                long long ag = 0, ae = 0;
                for (int i = 0; i < TOPK_THREADS; ++i) {
                    const int gg = s_a[i], ee = s_b[i];
                    s_a[i] = (int) ag;
                    s_b[i] = (int) ae;
                    ag += gg;
                    ae += ee;
                }
                s_cgt[0] = ag;
            }
            sycl::group_barrier(g);
            const long long off_eq = s_b[t];
            const long long eq_budget = width - s_cgt[0];

            int sel = 0;
            {
                long long ee = off_eq;
                for (long long j = lo; j < hi; ++j) {
                    const uint32_t k = order_key(scores[j]);
                    if (k > thr) ++sel;
                    else if (k == thr && ee < eq_budget) { ++sel; ++ee; }
                }
            }
            s_a[t] = sel;
            sycl::group_barrier(g);
            if (t == 0) {
                long long a = 0;
                for (int i = 0; i < TOPK_THREADS; ++i) {
                    const int c = s_a[i];
                    s_a[i] = (int) a;
                    a += c;
                }
            }
            sycl::group_barrier(g);

            // ---- emit, walking the cells ascending so the tie rule's "first by index" is the walk order itself
            long long w = s_a[t];
            long long ee = off_eq;
            for (long long j = lo; j < hi; ++j) {
                const uint32_t k = order_key(scores[j]);
                if (k > thr) out_ids[w++] = (int) j;
                else if (k == thr && ee < eq_budget) { ++ee; out_ids[w++] = (int) j; }
            }
        });
    });
    finish(stream, e, "topk_512");
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
        const int per = head_dim / 4;                       // 4 halfs per 8-byte word
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

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    const int idx_dim = (int) s.idx_dim, r = (int) s.idx_block, n_rot = (int) s.n_rot;
    float* tail = b.tail;
    float* dead = b.dead;
    float* pooled = b.pooled;
    int32_t* block_pos = b.block_pos;
    const int32_t* mtab = mrope_table();
    // One launch, no host branch on the position: the completion rotation is inside the kernel.
    auto& queue = queue_for(stream);
    auto launch = [&](auto zero) {   // D: double where the device has FP64, float elsewhere (device_caps.hpp)
        using D = decltype(zero);
        return queue.submit([&](sycl::handler& hd) {
            sycl::local_accessor<D, 1> s_mean(sycl::range<1>((size_t) idx_dim), hd);
            hd.parallel_for(sycl::nd_range<1>((size_t) idx_dim, (size_t) idx_dim), [=](sycl::nd_item<1> it) {
                const auto g = it.get_group();
                const int d = (int) it.get_local_id(0);
                const int pos = (int) pos_dev[0];
                const int slot = pos % r;

                // The raw tail: a cell that completes a block is not stored, the pool below consumes it directly.
                if (slot < r - 1) tail[(size_t) slot * idx_dim + d] = raw[d];

                // The spare slot's key is rms_norm(raw[0]), constant for the sequence.
                if (pos == 0) {
                    const D p = (D) raw[d];
                    D ss = (D) 0;
                    for (int i = 0; i < idx_dim; ++i) {
                        const D v = (D) raw[i];
                        ss = ss + v * v;
                    }
                    const D inv = (D) 1 / sycl::sqrt(ss / (D) idx_dim + (D) eps);
                    dead[d] = (float) (p * inv * (D) w_k_norm[d]);
                    // rope at position 0 is the identity (cos = 1, sin = 0 exactly), so no rotation is applied
                    pooled[d] = dead[d];
                }

                if (slot != r - 1) return;           // uniform across the group
                sycl::group_barrier(g);              // the tail rows this work-item is about to read are written above

                // The mean of this block's r raw keys, in CELL order.
                D m = (D) 0;
                for (int j = 0; j < r - 1; ++j) m = m + (D) tail[(size_t) j * idx_dim + d];
                m = m + (D) raw[d];
                m = m / (D) r;
                s_mean[d] = m;
                sycl::group_barrier(g);

                D ss = (D) 0;
                for (int i = 0; i < idx_dim; ++i) ss = ss + s_mean[i] * s_mean[i];
                const D inv = (D) 1 / sycl::sqrt(ss / (D) idx_dim + (D) eps);

                const int bl = pos / r;
                pooled[(size_t) bl * idx_dim + d] = (float) (m * inv * (D) w_k_norm[d]);
                // The spare slot moves to b+1 and is rewritten with the same constant value.
                pooled[(size_t) (bl + 1) * idx_dim + d] = dead[d];
                if (d == 0) *block_pos = (int32_t) (pos_base + bl * r);
                sycl::group_barrier(g);              // the row is complete only now; the rotation reads two elements

                // The rotation of the row that just completed, in place, at the block's first cell's position.
                const int half = n_rot / 2;
                if (d < half) {
                    float* row = pooled + (size_t) bl * idx_dim;
                    const size_t toff = (size_t) mrope_pos(mtab, pos_base + bl * r, d) * half;
                    rope_neox_pair(row[d], row[half + d], cos_tab[toff + d], sin_tab[toff + d], row[d], row[half + d]);
                }
            });
        });
    };
    const auto e = xe::has_fp64(queue) ? launch(0.0) : launch(0.0f);
    finish(stream, e, "indexer_key_append");
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (n_bid < 0 || n_kv <= 0) fail("qsa_index: n_bid < 0 or n_kv <= 0");
    if ((n_kv / s.idx_block) != n_bid)
        fail("qsa_index: n_bid " + std::to_string(n_bid) + " is not n_kv " + std::to_string(n_kv) + " / r " +
             std::to_string(s.idx_block));
    qsa_index_step(pooled, q_idx, bias, s, step_upload(n_kv - 1, n_kv, s), n_bid + 1, cell_scores, stream);
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids,
              void* stream) {
    validate(s, "topk_512");
    if (n_kv <= 0) return;
    if (n_kv > kTopkMaxCells)
        fail("topk_512: n_kv " + std::to_string(n_kv) + " > kTopkMaxCells " + std::to_string(kTopkMaxCells));
    topk_512_step(cell_scores, s, cap, step_upload(n_kv - 1, n_kv, s), ids, stream);
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

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    const int head_dim = (int) s.head_dim;
    auto& queue = queue_for(stream);
    auto launch = [&](auto zero) {   // D: double where the device has FP64, float elsewhere (device_caps.hpp)
        using D = decltype(zero);
        return queue.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const long long i = (long long) id[0];
            const int h = (int) (i / head_dim), d = (int) (i % head_dim);
            const D gv = (D) q_full[((size_t) h * 2 * head_dim) + head_dim + d];   // the SECOND half
            const D sig = (D) 1 / ((D) 1 + sycl::exp(-gv));
            out[i] = (float) ((D) attn[i] * sig);
        });
    };
    const auto e = xe::has_fp64(queue) ? launch(0.0) : launch(0.0f);
    finish(stream, e, "qsa_gate_apply_f32");
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    const int head_dim = (int) s.head_dim;
    auto& queue = queue_for(stream);
    auto launch = [&](auto zero) {   // D: double where the device has FP64, float elsewhere (device_caps.hpp)
        using D = decltype(zero);
        return queue.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const long long i = (long long) id[0];
            const int h = (int) (i / head_dim), d = (int) (i % head_dim);
            const D gv = (D) q_full[((size_t) h * 2 * head_dim) + head_dim + d];   // the SECOND half
            const D sig = (D) 1 / ((D) 1 + sycl::exp(-gv));
            out[i] = f16_from_f32((float) ((D) attn[i] * sig));
        });
    };
    const auto e = xe::has_fp64(queue) ? launch(0.0) : launch(0.0f);
    finish(stream, e, "qsa_gate_apply");
}

}  // namespace strata::kernels
