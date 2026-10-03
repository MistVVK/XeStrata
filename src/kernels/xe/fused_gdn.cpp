// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/fused_gdn.cpp - the Xe port of Strata's src/kernels/cuda/fused_gdn.cu (see the header): the conv + SiLU + L2
// norm, the alpha/beta projections with their epilogues, and the GDN step with its output norm, each one kernel.
//
// Work-group shapes, explicit fmaf chains and summation orders are the CUDA kernels'.  CUDA's __expf and rsqrtf
// become the precise exp and SYCL's rsqrt.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per work-item
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream, const char*) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S)
        throw core::DeviceError("fused_gdn_conv_l2: invalid arguments");
    auto& q = queue_for(stream, "fused_gdn_conv_l2");
    const auto e = q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / WARP), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) channels, S), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            const int c = (int) it.get_global_id(0);
            const float v0 = history[c * 3], v1 = history[c * 3 + 1], v2 = history[c * 3 + 2], x = qkv[c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
            history[c * 3] = v1;
            history[c * 3 + 1] = v2;
            history[c * 3 + 2] = x;
            float y = sum / (1.0f + sycl::exp(-sum));
            if ((int) it.get_group(0) < qk_heads) {   // uniform over the work-group
                const float sq = warp_sum(sg, y * y);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[c] = y;
        });
    });
    finish(stream, e, "fused_gdn_conv_l2");
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0)
        throw core::DeviceError("fused_gdn_ab: invalid arguments");
    auto& q = queue_for(stream, "fused_gdn_ab");
    const int n = n_embd;
    const size_t groups = (size_t) ((2 * h_v + 7) / 8);
    const auto e = q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int row = (int) it.get_group(0) * 8 + (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (row >= 2 * h_v) return;   // uniform over the sub-group
        const bool is_beta = row >= h_v;
        const int r = is_beta ? row - h_v : row;
        const uint32_t* w = reinterpret_cast<const uint32_t*>((is_beta ? w_beta : w_alpha) + (size_t) r * n);
        auto lo = [](uint32_t u) { return sycl::bit_cast<float>(u << 16); };
        auto hi = [](uint32_t u) { return sycl::bit_cast<float>(u & 0xffff0000u); };
        float acc = 0.0f;
        for (int j = lane; j < n / 8; j += 32) {
            const uint32_t w0 = w[j * 4], w1 = w[j * 4 + 1], w2 = w[j * 4 + 2], w3 = w[j * 4 + 3];
            const float* xa = x + j * 8;
            acc = sycl::fma(lo(w0), xa[0], acc); acc = sycl::fma(hi(w0), xa[1], acc);
            acc = sycl::fma(lo(w1), xa[2], acc); acc = sycl::fma(hi(w1), xa[3], acc);
            acc = sycl::fma(lo(w2), xa[4], acc); acc = sycl::fma(hi(w2), xa[5], acc);
            acc = sycl::fma(lo(w3), xa[6], acc); acc = sycl::fma(hi(w3), xa[7], acc);
        }
        acc = warp_sum(sg, acc);
        if (lane != 0) return;
        if (is_beta) {
            beta[r] = 1.0f / (1.0f + sycl::exp(-acc));
        } else {
            const float v = acc + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
            gate[r] = sp * ssm_a[r];
        }
    });
    finish(stream, e, "fused_gdn_ab");
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k)
        throw core::DeviceError("fused_gdn_step_norm: invalid arguments");
    auto& qu = queue_for(stream, "fused_gdn_step_norm");
    const auto e = qu.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd);
        sycl::local_accessor<float, 1> sq(sycl::range<1>(S), hd);
        sycl::local_accessor<float, 1> red(sycl::range<1>(RG * S), hd);
        sycl::local_accessor<float, 1> wsum(sycl::range<1>(S * RG / WARP), hd);
        // one work-group per head; work-item tid = rg * 128 + col, as CUDA's (col, rg) block linearises
        hd.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const int col = tid % S, rg = tid / S;
            const int qh = head % h_k;
            if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            sycl::group_barrier(it.get_group());
            const float g = sycl::exp(gate[head]);
            float kv = 0.0f;
            for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
            red[rg * S + col] = kv;
            sycl::group_barrier(it.get_group());
            const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
            const float delta = (v[head * S + col] - g * kv_col) * beta[head];
            float o = 0.0f;
            for (int r = 0; r < RPG; ++r) {
                s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                o = sycl::fma(s[r], sq[rg * RPG + r], o);
                base[r * row_stride] = s[r];
            }
            sycl::group_barrier(it.get_group());   // every work-item has read red[] for kv_col
            red[rg * S + col] = o;
            sycl::group_barrier(it.get_group());
            float oc = 0.0f, sq_part = 0.0f;
            if (rg == 0) {
                oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                sq_part = oc * oc;
            }
            // RMS over the head's 128 outputs: the sub-groups of row group 0 are work-items 0..127
            sq_part = warp_sum(sg, sq_part);
            if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
            sycl::group_barrier(it.get_group());
            if (rg == 0) {
                const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                const float scale = sycl::rsqrt(ss / (float) S + eps);
                const float zz = z[head * S + col];
                y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::exp(-zz)));
            }
        });
    });
    finish(stream, e, "fused_gdn_step_norm");
}

}  // namespace strata::kernels
