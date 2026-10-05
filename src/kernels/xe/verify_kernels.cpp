// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/verify_kernels.cpp - the Xe port of Strata's src/kernels/cuda/verify_kernels.cu (see verify_kernels.hpp):
// the verify window's multi-token GDN kernels, the resident-expert plan, the flag waits on host-mapped memory, and
// the small gathers, copies and selections around them.
//
// The per-token arithmetic of each kernel is its single-token original's (fused_gdn, elementwise) with the same
// operation order, so a verify window reproduces plain decode.  The flag waits use the reading measured in
// bench/results/2026-09-30-xe-doorbell: a volatile read followed by a system-scope acquire fence.  gpu_stamp writes
// device-clock ticks, not the nanoseconds of CUDA's %globaltimer.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"

// sycl_ext_oneapi_clock is newer than some DPC++ releases (intel/llvm 6.2 lacks it): without it there is no stamp
#if __has_include(<sycl/ext/oneapi/experimental/clock.hpp>)
#include <sycl/ext/oneapi/experimental/clock.hpp>
#endif

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;
constexpr int WARP = 32;

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
[[noreturn]] void fail(const std::string& what) { throw core::DeviceError(what); }
void done(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline uint32_t read_host(const uint32_t* p) {
    const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(p);
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    return v;
}

}  // namespace

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT)
        fail("gdn_conv_l2_multi: invalid arguments");
    const int C = channels;
    const auto e = Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / WARP), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) n_tok, (size_t) C}, {1, S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = t_begin + (int) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            const int c = (int) it.get_group(1) * S + tid;
            // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
            float win[3];
            for (int j = 0; j < 3; ++j) {
                const int src = t + j;   // index into [hist(3) | x...]
                win[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
            }
            const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
            float y = sum / (1.0f + sycl::exp(-sum));
            if ((int) it.get_group(1) < qk_heads) {   // uniform over the work-group
                const float sq = warp_sum(sg, y * y);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[(size_t) t * C + c] = y;
        });
    });
    done(stream, e, "gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    const int C = channels;
    const auto e = Q(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const int n = *n_keep;
        if (n <= 0) return;
        float seq[3];
        for (int j = 0; j < 3; ++j) {
            const int src = n + j;   // the last three of [hist(3) | x_0..x_{n-1}]
            seq[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
        }
        history[c * 3] = seq[0];
        history[c * 3 + 1] = seq[1];
        history[c * 3 + 2] = seq[2];
    });
    done(stream, e, "gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) fail("gdn_ab_multi: invalid arguments");
    const int n = n_embd, T = n_tok;
    const size_t groups = (size_t) ((2 * h_v + 7) / 8);
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int row = (int) it.get_group(0) * 8 + (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (row >= 2 * h_v) return;   // uniform over the sub-group
        const bool is_beta = row >= h_v;
        const int r = is_beta ? row - h_v : row;
        const uint32_t* w = reinterpret_cast<const uint32_t*>((is_beta ? w_beta : w_alpha) + (size_t) r * n);
        auto lo = [](uint32_t u) { return sycl::bit_cast<float>(u << 16); };
        auto hi = [](uint32_t u) { return sycl::bit_cast<float>(u & 0xffff0000u); };
        float acc[kVerifyMaxT];
        #pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
        for (int j = lane; j < n / 8; j += 32) {
            const uint32_t w0 = w[j * 4], w1 = w[j * 4 + 1], w2 = w[j * 4 + 2], w3 = w[j * 4 + 3];
            #pragma unroll
            for (int t = 0; t < kVerifyMaxT; ++t) {
                if (t >= T) break;
                const float* xa = x + (size_t) t * n + j * 8;
                float a = acc[t];
                a = sycl::fma(lo(w0), xa[0], a); a = sycl::fma(hi(w0), xa[1], a);
                a = sycl::fma(lo(w1), xa[2], a); a = sycl::fma(hi(w1), xa[3], a);
                a = sycl::fma(lo(w2), xa[4], a); a = sycl::fma(hi(w2), xa[5], a);
                a = sycl::fma(lo(w3), xa[6], a); a = sycl::fma(hi(w3), xa[7], a);
                acc[t] = a;
            }
        }
        #pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float a = warp_sum(sg, acc[t]);
            if (lane != 0) continue;
            if (is_beta) {
                beta[(size_t) t * h_v + r] = 1.0f / (1.0f + sycl::exp(-a));
            } else {
                const float v = a + dt[r];
                const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
                gate[(size_t) t * h_v + r] = sp * ssm_a[r];
            }
        }
    });
    done(stream, e, "gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* hbuf, int C, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !hbuf || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT)
        fail("gdn_step_norm_multi: invalid arguments");
    const int T = n_tok;
    const auto e = Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
            red(sycl::range<1>(RG * S), hd), wsum(sycl::range<1>(S * RG / WARP), hd);
        // one work-group per head; work-item tid = rg * 128 + col, as CUDA's (col, rg) block linearises
        hd.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0), tid = (int) it.get_local_id(0), col = tid % S, rg = tid / S;
            const int qh = head % h_k;
            const int qk = S * h_k;   // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
            const int value_dim = S * h_v;
            const int n = n_keep ? *n_keep : T;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            #pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            for (int t = 0; t < n; ++t) {
                const float* ht = hbuf + (size_t) t * C;
                sycl::group_barrier(it.get_group());   // the previous token is done with sk/sq/red/wsum
                if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                float kv = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * S + col] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                float o = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[rg * RPG + r], o);
                }
                sycl::group_barrier(it.get_group());
                red[rg * S + col] = o;
                sycl::group_barrier(it.get_group());
                float oc = 0.0f, sq_part = 0.0f;
                if (rg == 0) {
                    oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                    sq_part = oc * oc;
                }
                if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
                sq_part = warp_sum(sg, sq_part);
                if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
                sycl::group_barrier(it.get_group());
                if (rg == 0) {
                    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                    const float scale = sycl::rsqrt(ss / (float) S + eps);
                    const float zz = z[(size_t) t * value_dim + head * S + col];
                    y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::exp(-zz)));
                }
            }
            if (n_keep != nullptr && n > 0)
                #pragma unroll
                for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
        });
    });
    done(stream, e, "gdn_step_norm_multi");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    Q(stream).single_task([=] {
        while (read_host(flag) < value) {
        }
    });
}

bool doorbell_visible(void* stream) {
    auto& q = Q(stream);
    auto* m = sycl::malloc_host<uint32_t>(4, q);
    if (m == nullptr) return false;
    volatile uint32_t* hm = m;
    hm[0] = 0;
    hm[1] = 0;
    hm[2] = 0;
    uint32_t* ring = m;
    uint32_t* flag = m + 1;
    uint32_t* seen = m + 2;
    // at most 2^21 reads (well under a second on the UHD 770, a few microseconds where the flag arrives)
    q.single_task([=] {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                         sycl::access::address_space::global_space>(*ring).store(1u);
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        for (uint32_t i = 0; i < (1u << 21); ++i)
            if (read_host(flag) == 1u) { *seen = 1u; break; }
    });
    const auto t0 = std::chrono::steady_clock::now();
    bool rang = false;
    while (!(rang = hm[0] == 1u) && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {}
    std::atomic_thread_fence(std::memory_order_seq_cst);
    hm[1] = 1u;
    q.wait();
    const bool ok = rang && hm[2] == 1u;
    sycl::free(m, q);
    return ok;
}

void resident_plan(const int32_t* ids, int n, int k, const int32_t* res, int n_expert, const uint8_t* cache_base,
                   const unsigned long long* slot_off, long long blob, int32_t* pl, long long capx, uint32_t* skip,
                   uint32_t ring, void* stream) {
    // One work-item per entry (at most kVerifyMaxT * 10 of them; one work-item alone grouped them in ~70 us a layer,
    // upstream 882764d).  The plan is the host loop's exactly: groups in order of first occurrence, entries in a group
    // in index order.  Entry i: L = its expert's first index; its group = the first occurrences before L; the group
    // starts at the number of entries whose leader is before L; its place in it = the earlier entries with leader L.
    constexpr int WG = 128;
    if (n > WG) throw std::invalid_argument("resident_plan: more entries than a work-group");
    const auto e = Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> s_ids(sycl::range<1>(WG), h);
        sycl::local_accessor<int32_t, 1> s_lead(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int t = (int) it.get_local_id(0);
            bool bad = false;
            if (t < n) {
                const int32_t ex = ids[t];
                s_ids[t] = ex;
                bad = ex < 0 || ex >= n_expert || res[ex] < 0;
            }
            if (sycl::any_of_group(grp, bad)) {
                if (t == 0) *skip = 0;
                return;
            }
            if (t < n) {
                int L = t;
                for (int j = 0; j < t; ++j)
                    if (s_ids[j] == s_ids[t]) { L = j; break; }
                s_lead[t] = L;
            }
            sycl::group_barrier(grp);
            int32_t* counts = pl;
            int32_t* start = pl + 4;
            int32_t* dst = start + capx + 1;
            int32_t* tok = dst + capx;
            const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
            unsigned long long* ptr = reinterpret_cast<unsigned long long*>(pl + ptr_off);
            int32_t* start2 = pl + ptr_off + 4 * capx;
            if (t < n) {
                const int L = s_lead[t];
                int g = 0, gstart = 0, pos = 0;
                for (int j = 0; j < n; ++j) {
                    const int Lj = s_lead[j];
                    if (j < L && Lj == j) ++g;
                    if (Lj < L) ++gstart;
                    if (j < t && Lj == L) ++pos;
                }
                dst[gstart + pos] = t;
                tok[gstart + pos] = t / k;
                if (L == t) {
                    const int32_t slot = res[s_ids[t]];
                    ptr[g] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
                    start[g] = gstart;
                }
            }
            sycl::group_barrier(grp);
            if (t != 0) return;
            int groups = 0;
            for (int i = 0; i < n; ++i) groups += s_lead[i] == i;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
            *skip = ring;
        });
    });
    done(stream, e, "resident_plan");
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    Q(stream).single_task([=] {
        if (read_host(skip) == value) return;
        while (read_host(flag) < value) {
        }
    });
}

void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    const int ni = (int) n;
    Q(stream).parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) {
        if (*reinterpret_cast<const volatile uint32_t*>(skip) == value) return;
        for (int i = (int) it.get_local_id(0); i < ni; i += 128) dst[i] = reinterpret_cast<const volatile int32_t*>(src)[i];
    });
}

void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;   // the CUDA kernel moved whole float4s: n is a multiple of 4
    Q(stream).parallel_for(sycl::range<1>((size_t) (n4 * 4)), [=](sycl::id<1> id) {
        const bool zero = *reinterpret_cast<const volatile uint32_t*>(skip) == value;
        dst[id] = zero ? 0.0f : reinterpret_cast<const volatile float*>(src)[id];
    });
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        const uint64_t token = (uint64_t) tokens[t];
        const uint8_t* c = codes + token * row_codes;
        const float* sc = scales + token * row_groups;
        const float* of = offsets ? offsets + token * row_groups : nullptr;
        const int per_byte = 8 / code_bits;
        const unsigned mask = (1u << code_bits) - 1u;
        const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
        const int64_t group = i / group_elems;
        const float product = (float) (code + code_bias) * sc[group];
        out[(size_t) t * n + i] = product + (of ? of[group] : 0.0f);
    });
    done(stream, e, "embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n_embd * hc)), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        R[(size_t) t * n_embd * hc + i] = x[(size_t) t * n_embd + i % n_embd];
    });
    done(stream, e, "broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int idx = *index;
        if (idx < 0) return;
        dst[id] = src[(size_t) idx * stride + id];
    });
    done(stream, e, "copy_indexed");
}

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) fail("fetch_blobs: blob size must be a multiple of 16");
    const long long per = (long long) (blob_bytes / 16);
    // a 16-byte clang vector: through sycl::uint4, NVPTX moved each 16 bytes as two 8-byte loads, and these loads
    // cross PCIe (RTX 4070: 858 us a call against upstream's 801)
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    auto* d = reinterpret_cast<u32x4*>(dst);
    const size_t items = 48 * 8 * 256;
    const auto e = Q(stream).parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
        const long long total = (long long) *n * per;
        for (long long i = (long long) id[0]; i < total; i += (long long) items) {
            const long long k = i / per, off = i - k * per;
            d[i] = reinterpret_cast<const u32x4*>(src[k])[off];
        }
    });
    done(stream, e, "fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    const unsigned long long b = (unsigned long long) base;
    const long long bytes = (long long) blob_bytes;
    const auto e = Q(stream).parallel_for(sycl::range<1>(128), [=](sycl::id<1> id) {
        const int k = (int) id[0];
        if (k < *n) ptr[k] = b + (unsigned long long) k * (unsigned long long) bytes;
    });
    done(stream, e, "rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* ex, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n_embd * hc)), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        R[(size_t) t * n_embd * hc + i] = h[(size_t) t * n_embd * hc + i] + ex[(size_t) t * n_embd + i % n_embd];
    });
    done(stream, e, "add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) fail("ident_hits: n out of range");
    const auto e = Q(stream).parallel_for(sycl::range<1>(1024), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < n) { slot[i] = ids[i]; dst[i] = i; }
        if (i == 0) *count = n;
    });
    done(stream, e, "ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    const size_t items = 16 * 256;
    const auto e = Q(stream).parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
        const int row = *row_dev;
        for (int64_t i = (int64_t) id[0]; i < R_stride; i += (int64_t) items) R_dst[i] = R_src[(size_t) row * R_stride + i];
        if (id[0] == 0) {
            const int32_t tok = ids[row];
            *tok_dst = tok;
            if (out != nullptr) reinterpret_cast<volatile int32_t*>(out)[j] = tok;
            if (probs != nullptr && out_p != nullptr) reinterpret_cast<volatile float*>(out_p)[j] = probs[row];
        }
    });
    done(stream, e, "mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    // the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
    auto launch = [&](auto elem) {
        using E = decltype(elem);
        const long long row_e = (long long) row_bytes / (long long) sizeof(E);
        const E* s = reinterpret_cast<const E*>(src);
        E* d = reinterpret_cast<E*>(dst);
        const long long total = (long long) n * row_e;
        return Q(stream).parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
            const long long i = (long long) id[0], r = i / row_e, o = i - r * row_e;
            d[i] = s[(long long) ids[r] * row_e + o];
        });
    };
    if (n <= 0) return;
    sycl::event e;
    if (row_bytes % 16 == 0) e = launch(sycl::uint4{});
    else if (row_bytes % 4 == 0) e = launch(uint32_t{});
    else e = launch(uint8_t{});
    done(stream, e, "gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < n) ids[i] = table[ids[i]];
    });
    done(stream, e, "map_ids");
}

namespace {
template <int WG>
sycl::event row_top_prob_launch(sycl::queue& q, const float* logits, int n_rows, int n_vocab, const int32_t* ids,
                                float* probs) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(WG / 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_rows * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            const float* l = logits + (size_t) t * n_vocab;
            const float m = l[ids[t]];
            float s = 0.0f;
            for (int i = tid; i < n_vocab; i += WG) s += sycl::exp(l[i] - m);
            s = warp_sum(sg, s);
            if ((tid & 31) == 0) part[tid >> 5] = s;
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < WG / 32; ++w) tot += part[w];
                probs[t] = 1.0f / tot;
            }
        });
    });
}
}  // namespace

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    auto& q = Q(stream);
    const int wg = xe::work_group_upto_1024(q);   // the draft probabilities sum in another order below 1024
    const auto e = wg == 1024 ? row_top_prob_launch<1024>(q, logits, n_rows, n_vocab, ids, probs)
                   : wg == 512 ? row_top_prob_launch<512>(q, logits, n_rows, n_vocab, ids, probs)
                               : row_top_prob_launch<256>(q, logits, n_rows, n_vocab, ids, probs);
    done(stream, e, "row_top_prob");
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    const long long stride = (long long) ids_stride;
    const auto e = Q(stream).parallel_for(sycl::nd_range<2>({(size_t) n, 8 * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
        const int q = (int) it.get_group(0);
        int32_t* st = steps + q * 4;
        const int n_kv = st[1];
        const int start = n_kv > window ? n_kv - window : 0;
        const int width = n_kv - start;
        for (int jj = (int) it.get_global_id(1); jj < width; jj += (int) it.get_global_range(1))
            ids[q * stride + jj] = start + jj;
        sycl::group_barrier(it.get_group());
        if (it.get_group(1) == 0 && it.get_local_id(1) == 0) st[3] = width;
    });
    done(stream, e, "window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i >= n) return;
        const int c = cells[i];
        steps[i * 4 + 0] = c;
        steps[i * 4 + 1] = c + 1;
        steps[i * 4 + 2] = (c + 1) / 4;
        steps[i * 4 + 3] = c + 1;
    });
    done(stream, e, "dense_steps");
}

// CUDA wrote %globaltimer (nanoseconds).  The Xe device clock counts device ticks; stamps are compared with one another,
// and converting them to time needs the tick rate, which is not established here.
#ifdef SYCL_EXT_ONEAPI_CLOCK
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    auto& queue = Q(stream);
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_clock_device))
        fail("gpu_stamp: the device has no device-scope clock");
    queue.single_task([=] {
        buf[i] = (unsigned long long) sycl::ext::oneapi::experimental::clock<
            sycl::ext::oneapi::experimental::clock_scope::device>();
    });
}

bool gpu_stamp_available() {
    return core::Runtime::get().compute().get_device().has(sycl::aspect::ext_oneapi_clock_device);
}
#else
void gpu_stamp(unsigned long long*, int, void*) { fail("gpu_stamp: this SYCL compiler has no device clock"); }

bool gpu_stamp_available() { return false; }
#endif

}  // namespace strata::kernels
