// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/sycl/ink_sy_ops.h - Inkling2Loop's fused operations of the 48 visits on 16 logical lanes
// (ink_sy_common.h): the embedding, the attention residuals, the loop controllers, ShortConv, the relative-position
// attention with sinks, the router.  1..64 slots, prompts and the vision tokens.
#pragma once

#include "ink_sy_common.h"
#include "strata/kernels/qsa.hpp"

namespace ink_sycl {

constexpr int INK_D = 2304;
constexpr int INK_NH = 24;
constexpr int INK_NKV = 8;
constexpr int INK_HD = 128;
constexpr int INK_DREL = 16;
constexpr float INK_RMS_EPS = 1e-6f;

inline void k_embed_sg16(const InkTensorInfo& eti, const float* norm_w, const int* tok_ids,
                         const uint8_t* is_img, const float* img_emb, float* cur_out, int M, int V_padded) {
    const uint8_t* ew = eti.dev_ptr;
    const bool is_bf16 = (eti.fmt == FMT_BF16);
    const bool is_q8_0 = (eti.fmt == FMT_Q8_0);
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        if (is_img && is_img[m] != 0) {
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = img_emb[(size_t)m * INK_D + d];
            return;
        }
        const int id = tok_ids[m];
        float ss = 0.0f;
        if (is_bf16) {
            const uint16_t* row = reinterpret_cast<const uint16_t*>(ew) + (size_t)id * INK_D;
            for (int d = tid; d < INK_D; d += 256) { float v = bf16_to_f32(row[d]); ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = bf16_to_f32(row[d]) * inv * norm_w[d];
        } else if (is_q8_0) {
            const float* rs = reinterpret_cast<const float*>(ew) + (size_t)id * (INK_D / 32);
            const int8_t* rq = reinterpret_cast<const int8_t*>(ew + (size_t)V_padded * (INK_D / 32) * 4) + (size_t)id * INK_D;
            auto w = [rs, rq](int d) { return (float)rq[d] * rs[d >> 5]; };
            for (int d = tid; d < INK_D; d += 256) { float v = w(d); ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = w(d) * inv * norm_w[d];
        } else {
            const float sc = reinterpret_cast<const float*>(ew)[id];
            const int8_t* row = reinterpret_cast<const int8_t*>(ew + (size_t)V_padded * 4) + (size_t)id * INK_D;
            for (int d = tid; d < INK_D; d += 256) { float v = (float)row[d] * sc; ss += v * v; }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) cur_out[(size_t)m * INK_D + d] = (float)row[d] * sc * inv * norm_w[d];
        }
    });
}

inline void k_attn_res_mix_sg16(const float* blocks_buf, const float* cur_buf, const float* queries, float* out_buf,
                                int n_blocks, int block_stride, int M, int n_queries) {
    const int M_stride = M * INK_D;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        sycl::local_accessor<float, 1> inv_s(8, h);
        sycl::local_accessor<float, 1> probs(32, h);  // [4][8]
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int S = n_blocks + 1;
            for (int s = 0; s < S; ++s) {
                const float* sp = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
                float ss = 0.0f;
                for (int d = tid; d < INK_D; d += 256) ss += sp[d] * sp[d];
                const float tot = wg_sum16(it, ss, sh);
                if (tid == 0) inv_s[s] = sycl::rsqrt(tot / (float)INK_D + INK_RMS_EPS);
            }
            sycl::group_barrier(it.get_group());
            const float inv_sqrt_d = 1.0f / 48.0f;
            for (int q = 0; q < n_queries; ++q) {
                const float* qv = queries + (size_t)q * INK_D;
                for (int s = 0; s < S; ++s) {
                    const float* sp = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
                    float dt = 0.0f;
                    for (int d = tid; d < INK_D; d += 256) dt += sp[d] * qv[d];
                    const float sc = wg_sum16(it, dt, sh) * inv_s[s] * inv_sqrt_d;
                    if (tid == 0) probs[q * 8 + s] = sc;
                }
                sycl::group_barrier(it.get_group());
                if (tid == 0) {
                    float mx = -1e30f;
                    for (int s = 0; s < S; ++s) mx = sycl::fmax(mx, probs[q * 8 + s]);
                    float sum = 0.0f;
                    for (int s = 0; s < S; ++s) { probs[q * 8 + s] = sycl::exp(probs[q * 8 + s] - mx); sum += probs[q * 8 + s]; }
                    const float isum = 1.0f / sum;
                    for (int s = 0; s < S; ++s) probs[q * 8 + s] *= isum * inv_s[s];
                }
                sycl::group_barrier(it.get_group());
                float* dst = out_buf + (size_t)q * M_stride + (size_t)m * INK_D;
                for (int d = tid; d < INK_D; d += 256) {
                    float acc = 0.0f;
                    for (int s = 0; s < S; ++s) {
                        const float* sp = (s < n_blocks) ? (blocks_buf + (size_t)s * block_stride + (size_t)m * INK_D) : (cur_buf + (size_t)m * INK_D);
                        acc += probs[q * 8 + s] * sp[d];
                    }
                    dst[d] = acc;
                }
            }
        });
    });
}

inline void k_ctrl_cat_rms_sg16(const float* cur_buf, const float* msg_buf, float* cat_out, int M) {
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* cp = cur_buf + (size_t)m * INK_D;
        const float* mp = msg_buf + (size_t)m * INK_D;
        float ssc = 0.0f, ssm = 0.0f;
        for (int d = tid; d < INK_D; d += 256) { ssc += cp[d] * cp[d]; ssm += mp[d] * mp[d]; }
        const float invc = sycl::rsqrt(wg_sum16(it, ssc, sh) / (float)INK_D + INK_RMS_EPS);
        const float invm = sycl::rsqrt(wg_sum16(it, ssm, sh) / (float)INK_D + INK_RMS_EPS);
        float* dst = cat_out + (size_t)m * (2 * INK_D);
        for (int d = tid; d < INK_D; d += 256) { dst[d] = cp[d] * invc; dst[INK_D + d] = mp[d] * invm; }
    });
}

inline void k_ctrl_apply_norm_sg16(const float* cur_buf, const float* msg_buf, const float* gate_buf,
                                   const float* w_gain, const float* w_forget, const float* norm_w,
                                   float* out_buf, int M, int do_norm) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        sycl::local_accessor<float, 1> g_sh(64, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            if (tid < 64) g_sh[tid] = gate_buf[(size_t)m * 64 + tid];
            sycl::group_barrier(it.get_group());
            const float* cp = cur_buf + (size_t)m * INK_D;
            const float* mp = msg_buf + (size_t)m * INK_D;
            float* dst = out_buf + (size_t)m * INK_D;
            float ss = 0.0f;
            for (int d = tid; d < INK_D; d += 256) {
                const float* wg = w_gain + (size_t)d * 64;
                const float* wf = w_forget + (size_t)d * 64;
                float sg = 0.0f, sf = 0.0f;
                for (int r = 0; r < 64; ++r) { sg += wg[r] * g_sh[r]; sf += wf[r] * g_sh[r]; }
                const float v = (1.0f - sycl::tanh(sycl::clamp(sf, -15.0f, 15.0f))) * cp[d]
                              + sycl::tanh(sycl::clamp(sg, -15.0f, 15.0f)) * mp[d];
                dst[d] = v;
                ss += v * v;
            }
            if (do_norm) {
                const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
                for (int d = tid; d < INK_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
            }
        });
    });
}

inline void k_rms_norm_sg16(const float* x_in, const float* w, float* y_out, int D, int M) {
    Q().parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        const float* xp = x_in + (size_t)m * D;
        float ss = 0.0f;
        for (int d = tid; d < D; d += 256) ss += xp[d] * xp[d];
        const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)D + INK_RMS_EPS);
        float* yp = y_out + (size_t)m * D;
        for (int d = tid; d < D; d += 256) yp[d] = xp[d] * inv * w[d];
    });
}

inline void k_add_sg16(const float* a, const float* b, float* out, int N) {
    Q().parallel_for(NR((N + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int i = (int)it.get_global_id(0);
        if (i < N) out[i] = a[i] + b[i];
    });
}

inline void k_sconv4_multi_sg16(
    const float* x_in, const float* static_w, const float* dyn_taps, float* hist_pool,
    const int* active_slots, const int* slot_tok_off, const int* slot_tok_cnt, const int* slot_tok_idx,
    float* y_out, int n_active_slots, int C, int has_dyn,
    float* ring_out = nullptr, int save_ring = 0) {
    Q().parallel_for(NR2((C + 255) / 256, (size_t)n_active_slots, 256), [=](sycl::nd_item<2> it) INK_REQD_SG {
        const int si = (int)it.get_group(0);
        const int c = (int)it.get_group(1) * 256 + (int)it.get_local_id(1);
        if (c >= C) return;
        const int slot = active_slots[si], off = slot_tok_off[si], cnt = slot_tok_cnt[si];
        float* h = hist_pool + (size_t)slot * 3 * C;
        float h0 = h[0 * C + c], h1 = h[1 * C + c], h2 = h[2 * C + c];
        const float* sw = static_w + (size_t)c * 4;
        const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
        const float scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;
        const int hd_idx = c >> 7;
        for (int k = 0; k < cnt; ++k) {
            const int m = slot_tok_idx[off + k];
            const float cur = x_in[(size_t)m * C + c];
            float acc = h0 * sw0 + h1 * sw1 + h2 * sw2 + cur * sw3;
            if (has_dyn) {
                const float* dt = dyn_taps + ((size_t)m * 8 + hd_idx) * 4;
                acc += cur * dt[0] + h2 * dt[1] + h1 * dt[2] + h0 * dt[3];
            }
            y_out[(size_t)m * C + c] = (cur + acc) * scale;
            h0 = h1; h1 = h2; h2 = cur;
            if (save_ring && ring_out && si == 0 && k < 8) {
                float* rdst = ring_out + (size_t)k * 3 * C;
                rdst[0 * C + c] = h0; rdst[1 * C + c] = h1; rdst[2 * C + c] = h2;
            }
        }
        h[0 * C + c] = h0; h[1 * C + c] = h1; h[2 * C + c] = h2;
    });
}

inline void k_qr_clip_tau_sg16(float* q_buf, const float* q_scale, float* r_buf, const float* r_scale,
                               const int* pos_buf, int M, int is_swa) {
    Q().parallel_for(NR2(INK_NH, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), h = (int)it.get_group(1), tid = (int)it.get_local_id(1);
        const float r_pos = (float)(pos_buf[m] + 1) * (1.0f / 8192.0f);
        const float tau = is_swa ? 1.0f : (1.0f + 0.1f * sycl::log(sycl::fmax(1.0f, r_pos)));

        float* qp = q_buf + ((size_t)m * INK_NH + h) * INK_HD;
        const float qv = qp[tid] * q_scale[h * INK_HD + tid];
        const float qss = wg_sum16(it, qv * qv, sh);
        const float qfac = sycl::fmin(1.0f, 22.627416997969522f / (sycl::sqrt(qss) + 1e-6f)) * tau;
        qp[tid] = qv * qfac;

        {   // the first 16 lanes; the whole sub-group reaches the sum (a 32-wide one holds lanes 16..31 too)
            const bool rl = tid < 16;
            float* rp = r_buf + ((size_t)m * INK_NH + h) * INK_DREL;
            const float rv = rl ? rp[tid] * r_scale[h * INK_DREL + tid] : 0.0f;
            const float rss = sg_sum(it, rv * rv);
            const float rfac = sycl::fmin(1.0f, 8.0f / (sycl::sqrt(rss) + 1e-6f)) * tau;
            if (rl) rp[tid] = rv * rfac;
        }
    });
}

inline void k_kv_norm_store_f16_sg16(const float* k_in, const float* v_in, half* k_cache, half* v_cache,
                                     const int* slot_ids, const int* pos_buf, int M, int is_swa, int kv_stride) {
    Q().parallel_for(NR2(INK_NKV, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG {
        float* sh = lmem<float, 16>(it);
        const int m = (int)it.get_group(0), kh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
        const int slot = slot_ids[m], pos = pos_buf[m];
        const int kpos = is_swa ? (pos & 1023) : pos;

        const float kv = k_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        const float kss = wg_sum16(it, kv * kv, sh);
        const float kfac = sycl::fmin(1.0f, 22.627416997969522f / (sycl::sqrt(kss) + 1e-6f));

        const float vv = v_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        const float vss = wg_sum16(it, vv * vv, sh);
        const float vfac = sycl::rsqrt(vss / (float)INK_HD + INK_RMS_EPS);

        const size_t dst = (((size_t)slot * kv_stride + kpos) * INK_NKV + kh) * INK_HD + tid;
        k_cache[dst] = (half)(kv * kfac);
        v_cache[dst] = (half)(vv * vfac);
    });
}

// Fused RelPos + Sink 24:8 GQA Flash Attention + Sigmoid Gate (SG16: 8 SGs = 128 threads, 8 dims/lane!)
inline void k_relpos_sink_gqa_sg16(
    const float* q_buf, const float* r_buf, const float* rel_proj_pad, const float* sinks_buf,
    const float* gate_buf, const half* k_cache, const half* v_cache,
    const int* slot_ids, const int* pos_buf, float* attn_out,
    int M, int is_swa, int swa_win, int rel_extent, int kv_stride,
    const int32_t* sel_ids = nullptr, const int32_t* sel_steps = nullptr, const uint8_t* sel_use = nullptr,
    int sel_cap = 0) {
    // sel_ids (a full-attention visit with the QSA indexer, ink_sy_qsa.h): a query with sel_use[m] attends to the
    // cells sel_ids[m][0, width) (ascending positions; width = sel_steps[m][kStepWidth]) instead of 0..qpos
    const float scale = 1.0f / std::sqrt((float)INK_HD);
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_m(8, h);
        sycl::local_accessor<float, 1> sh_l(8, h);
        sycl::local_accessor<float, 1> sh_o(8 * INK_HD, h);
        sycl::local_accessor<float, 1> sh_r(16, h);
        h.parallel_for(NR2(INK_NH, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG {
            const int m = (int)it.get_group(0), head = (int)it.get_group(1);
            const int tid = (int)it.get_local_id(1);
            const int sg = tid >> 4, lane = tid & 15;
            const int kh = head / 3;
            const int slot = slot_ids[m], qpos = pos_buf[m];

            if (tid < 16) sh_r[tid] = r_buf[((size_t)m * INK_NH + head) * INK_DREL + tid];
            sycl::group_barrier(it.get_group());

            // Each SG16 lane holds 8 dims: d = lane*8 .. lane*8+7
            float qv[8], o_acc[8];
            const float* qp = q_buf + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
            for (int u = 0; u < 8; ++u) { qv[u] = qp[u] * scale; o_acc[u] = 0.0f; }

            const bool sel = sel_ids != nullptr && sel_use[m] != 0;
            const int32_t* ids = sel ? sel_ids + (size_t)m * sel_cap : nullptr;
            const int k_start = sel ? 0 : is_swa ? sycl::max(0, qpos - swa_win + 1) : 0;
            const int k_end = sel ? sel_steps[(size_t)m * strata::kernels::kStepCount + strata::kernels::kStepWidth]
                                  : qpos + 1;
            float m_max = (sg == 0) ? sinks_buf[(size_t)m * INK_NH + head] : -1e30f;
            float l_sum = (sg == 0) ? 1.0f : 0.0f;
            const size_t slot_base = (size_t)slot * kv_stride;

            // every logical sub-group takes the same number of steps (both halves of a 32-wide one reach the sums)
            for (int k0 = k_start; k0 < k_end; k0 += 8) {
                const int kk = sycl::min(k0 + sg, k_end - 1);   // a position, or with sel an index into ids
                const int k = sel ? ids[kk] : kk;
                const bool live = k0 + sg < k_end;
                const int kidx = is_swa ? (k & 1023) : k;
                const size_t kv_off = ((slot_base + kidx) * INK_NKV + kh) * INK_HD + lane * 8;
                float dt = 0.0f;
                for (int u = 0; u < 8; ++u) dt += qv[u] * (float)k_cache[kv_off + u];
                float sc = sg_sum(it, dt);

                const int dist = sycl::min(qpos - k, rel_extent);
                sc += sg_sum(it, sh_r[lane] * rel_proj_pad[(size_t)lane * 1025 + dist]);

                if (live) {
                    const float mn = sycl::fmax(m_max, sc);
                    const float corr = sycl::exp(m_max - mn);
                    const float p = sycl::exp(sc - mn);
                    l_sum = l_sum * corr + p;
                    m_max = mn;
                    for (int u = 0; u < 8; ++u) o_acc[u] = o_acc[u] * corr + p * (float)v_cache[kv_off + u];
                }
            }

            if (lane == 0) { sh_m[sg] = m_max; sh_l[sg] = l_sum; }
            for (int u = 0; u < 8; ++u) sh_o[sg * INK_HD + lane * 8 + u] = o_acc[u];
            sycl::group_barrier(it.get_group());

            if (sg == 0) {
                float gm = -1e30f;
                for (int s = 0; s < 8; ++s) gm = sycl::fmax(gm, sh_m[s]);
                float gl = 0.0f, go[8] = {};
                for (int s = 0; s < 8; ++s) {
                    const float w = sycl::exp(sh_m[s] - gm);
                    gl += sh_l[s] * w;
                    for (int u = 0; u < 8; ++u) go[u] += sh_o[s * INK_HD + lane * 8 + u] * w;
                }
                const float il = 1.0f / gl;
                const float* gp = gate_buf + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
                float* op = attn_out + ((size_t)m * INK_NH + head) * INK_HD + lane * 8;
                for (int u = 0; u < 8; ++u) op[u] = (go[u] * il) / (1.0f + sycl::exp(-gp[u]));
            }
        });
    });
}

inline void k_router_topk8_sg16(const float* r_logits, const float* r_bias, int* topk_idx, float* topk_w,
                                float route_scale, int M) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_p(128, h);
        sycl::local_accessor<float, 1> sh_s(128, h);
        h.parallel_for(NR(M, 128), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const float p = 1.0f / (1.0f + sycl::exp(-r_logits[(size_t)m * 128 + tid]));
            sh_p[tid] = p;
            sh_s[tid] = p + r_bias[tid];
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                int sel[8]; float pw[8]; float sum = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    int bi = 0; float bs = sh_s[0];
                    for (int e = 1; e < 128; ++e) if (sh_s[e] > bs) { bs = sh_s[e]; bi = e; }
                    sel[k] = bi; pw[k] = sh_p[bi]; sum += pw[k]; sh_s[bi] = -1e30f;
                }
                const float inv = route_scale / (sum + 1e-20f);
                for (int k = 0; k < 8; ++k) { topk_idx[(size_t)m * 8 + k] = sel[k]; topk_w[(size_t)m * 8 + k] = pw[k] * inv; }
            }
        });
    });
}

inline void k_softcap_sg16(float* logits, int total, float cap) {
    const float inv = 1.0f / cap;
    Q().parallel_for(NR((total + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
        const int i = (int)it.get_global_id(0);
        if (i < total) logits[i] = cap * sycl::tanh(sycl::clamp(logits[i] * inv, -15.0f, 15.0f));
    });
}

inline void k_argmax_rows_sg16(const float* logits, int* out_ids, int V, int M) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_v(256, h);
        sycl::local_accessor<int, 1> sh_i(256, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const float* row = logits + (size_t)m * V;
            float bv = -1e30f; int bi = 0;
            for (int i = tid; i < V; i += 256) {
                const float v = row[i];
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
            }
            sh_v[tid] = bv; sh_i[tid] = bi;
            sycl::group_barrier(it.get_group());
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    const float v1 = sh_v[tid + s]; const int i1 = sh_i[tid + s];
                    if (v1 > sh_v[tid] || (v1 == sh_v[tid] && i1 < sh_i[tid])) { sh_v[tid] = v1; sh_i[tid] = i1; }
                }
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) out_ids[m] = sh_i[0];
        });
    });
}

}  // namespace ink_sycl
