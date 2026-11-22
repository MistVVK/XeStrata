// ink_sy_dspark.h - Persistent 100% On-GPU Intel Arc SYCL SG16 DSpark Block-8 Recurrent Markov Drafter
// CRITICAL BATTLEMAGE B70 RULE: Strictly INK_REQD_SG16 (zero SG8 ICE on bmg_g31!).
// Matches inkling2_loop.py::Inkling2DSpark bit-for-bit with zero host sync across the 7 Markov steps!
// 100% MIT clean-room implementation.
#pragma once

#include "ink_sy_codecs.h"
#include "ink_sy_ops.h"

namespace ink_sycl {

constexpr int DSP_RANK = 256;

inline void k_dspark_mix_taps_sg16(const float* taps_4, const float* fuse_w, const float* norm_w,
                                   float* out_buf, int row, int M, int M_stride) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(16, h);
        sycl::local_accessor<float, 1> fw(4, h);
        h.parallel_for(NR(M, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            if (tid == 0) {
                float w0 = fuse_w[row * 4 + 0], w1 = fuse_w[row * 4 + 1];
                float w2 = fuse_w[row * 4 + 2], w3 = fuse_w[row * 4 + 3];
                const float mx = sycl::fmax(sycl::fmax(w0, w1), sycl::fmax(w2, w3));
                w0 = sycl::exp(w0 - mx); w1 = sycl::exp(w1 - mx); w2 = sycl::exp(w2 - mx); w3 = sycl::exp(w3 - mx);
                const float is = 1.0f / (w0 + w1 + w2 + w3);
                fw[0] = w0 * is; fw[1] = w1 * is; fw[2] = w2 * is; fw[3] = w3 * is;
            }
            sycl::group_barrier(it.get_group());
            float* dst = out_buf + (size_t)m * INK_D;
            float ss = 0.0f;
            for (int d = tid; d < INK_D; d += 256) {
                const float v = fw[0] * taps_4[((size_t)0 * M_stride + m) * INK_D + d]
                              + fw[1] * taps_4[((size_t)1 * M_stride + m) * INK_D + d]
                              + fw[2] * taps_4[((size_t)2 * M_stride + m) * INK_D + d]
                              + fw[3] * taps_4[((size_t)3 * M_stride + m) * INK_D + d];
                dst[d] = v;
                ss += v * v;
            }
            const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
            for (int d = tid; d < INK_D; d += 256) dst[d] = dst[d] * inv * norm_w[d];
        });
    });
}

inline void k_dspark_ctx_kv_store_sg16(const float* k_in, const float* v_in, const float* k_nw,
                                       half* k_cache, half* v_cache, const int* slot_ids, const int* pos_buf,
                                       int M, int kv_stride, float theta) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(8, h);
        sycl::local_accessor<float, 1> kn(128, h);
        h.parallel_for(NR2(INK_NKV, (size_t)M, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int m = (int)it.get_group(0), kh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
            const int slot = slot_ids[m], pos = pos_buf[m];
            const float* kp = k_in + ((size_t)m * INK_NKV + kh) * INK_HD;
            const float kv = kp[tid];
            const float inv = sycl::rsqrt(wg_sum16(it, kv * kv, sh) / (float)INK_HD + INK_RMS_EPS);
            kn[tid] = kv * inv * k_nw[tid];
            sycl::group_barrier(it.get_group());
            const size_t dst = (((size_t)slot * kv_stride + pos) * INK_NKV + kh) * INK_HD;
            if (tid < 64) {
                const float freq = 1.0f / sycl::pow(theta, (float)(2 * tid) / (float)INK_HD);
                const float ang = (float)pos * freq;
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                const float x0 = kn[tid], x1 = kn[tid + 64];
                k_cache[dst + tid] = (half)(x0 * cs - x1 * sn);
                k_cache[dst + tid + 64] = (half)(x0 * sn + x1 * cs);
            }
            v_cache[dst + tid] = (half)v_in[((size_t)m * INK_NKV + kh) * INK_HD + tid];
        });
    });
}

inline void k_dspark_init_block_sg16(const InkTensorInfo& eti, const float* norm_w, const float* mask_emb,
                                     float* cur_out, int anchor_tok, int V_padded) {
    const uint8_t* ew = eti.dev_ptr;
    const bool is_bf16 = (eti.fmt == FMT_BF16);
    Q().parallel_for(NR(8, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        float* sh = lmem<float, 16>(it);
        const int b = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        float* dst = cur_out + (size_t)b * INK_D;
        if (b == 0) {
            float ss = 0.0f;
            if (is_bf16) {
                const uint16_t* row = reinterpret_cast<const uint16_t*>(ew) + (size_t)anchor_tok * INK_D;
                for (int d = tid; d < INK_D; d += 256) { float v = bf16_to_f32(row[d]); ss += v * v; }
                const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
                for (int d = tid; d < INK_D; d += 256) dst[d] = bf16_to_f32(row[d]) * inv * norm_w[d];
            } else {
                const float sc = reinterpret_cast<const float*>(ew)[anchor_tok];
                const int8_t* row = reinterpret_cast<const int8_t*>(ew + (size_t)V_padded * 4) + (size_t)anchor_tok * INK_D;
                for (int d = tid; d < INK_D; d += 256) { float v = (float)row[d] * sc; ss += v * v; }
                const float inv = sycl::rsqrt(wg_sum16(it, ss, sh) / (float)INK_D + INK_RMS_EPS);
                for (int d = tid; d < INK_D; d += 256) dst[d] = (float)row[d] * sc * inv * norm_w[d];
            }
        } else {
            for (int d = tid; d < INK_D; d += 256) dst[d] = mask_emb[d];
        }
    });
}

inline void k_dspark_qk_norm_rope_sg16(float* x_buf, const float* nw, int NH, int apos, float theta) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(8, h);
        sycl::local_accessor<float, 1> xn(128, h);
        h.parallel_for(NR2(NH, 8, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int b = (int)it.get_group(0), hh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
            float* xp = x_buf + ((size_t)b * NH + hh) * INK_HD;
            const float v = xp[tid];
            const float inv = sycl::rsqrt(wg_sum16(it, v * v, sh) / (float)INK_HD + INK_RMS_EPS);
            xn[tid] = v * inv * nw[tid];
            sycl::group_barrier(it.get_group());
            if (tid < 64) {
                const float freq = 1.0f / sycl::pow(theta, (float)(2 * tid) / (float)INK_HD);
                const float ang = (float)(apos + b) * freq;
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                const float x0 = xn[tid], x1 = xn[tid + 64];
                xp[tid] = x0 * cs - x1 * sn;
                xp[tid + 64] = x0 * sn + x1 * cs;
            }
        });
    });
}

inline void k_dspark_block_attn_sg16(const float* q_buf, const float* kd_buf, const float* vd_buf,
                                     const half* ctx_k, const half* ctx_v, float* out_buf,
                                     int slot, int apos, int is_swa, int swa_win, int kv_stride) {
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_m(8, h);
        sycl::local_accessor<float, 1> sh_l(8, h);
        sycl::local_accessor<float, 1> sh_o(8 * INK_HD, h);
        h.parallel_for(NR2(INK_NH, 8, 128), [=](sycl::nd_item<2> it) INK_REQD_SG16 {
            const int b = (int)it.get_group(0), hh = (int)it.get_group(1), tid = (int)it.get_local_id(1);
            const int sg = tid >> 4, lane = tid & 15;
            const int kh = hh / 3;
            const float scale = 0.08838834764831845f;
            float qv[8], o_acc[8];
            const float* qp = q_buf + ((size_t)b * INK_NH + hh) * INK_HD + lane * 8;
            for (int u = 0; u < 8; ++u) { qv[u] = qp[u] * scale; o_acc[u] = 0.0f; }

            const int c0 = is_swa ? sycl::max(0, apos - swa_win + 1) : 0;
            const int n_ctx = sycl::max(0, apos - c0);
            const int total_k = n_ctx + 8;
            float m_max = -1e30f, l_sum = 0.0f;
            const size_t sbase = (size_t)slot * kv_stride;

            for (int idx = sg; idx < total_k; idx += 8) {
                float kv[8], vv[8];
                if (idx < n_ctx) {
                    const int pos = c0 + idx;
                    const size_t off = ((sbase + pos) * INK_NKV + kh) * INK_HD + lane * 8;
                    for (int u = 0; u < 8; ++u) { kv[u] = (float)ctx_k[off + u]; vv[u] = (float)ctx_v[off + u]; }
                } else {
                    const int db = idx - n_ctx;
                    const size_t off = ((size_t)db * INK_NKV + kh) * INK_HD + lane * 8;
                    for (int u = 0; u < 8; ++u) { kv[u] = kd_buf[off + u]; vv[u] = vd_buf[off + u]; }
                }
                float dt = 0.0f;
                for (int u = 0; u < 8; ++u) dt += qv[u] * kv[u];
                const float sc = sg_sum(it, dt);
                const float mn = sycl::fmax(m_max, sc);
                const float corr = sycl::exp(m_max - mn);
                const float p = sycl::exp(sc - mn);
                l_sum = l_sum * corr + p;
                m_max = mn;
                for (int u = 0; u < 8; ++u) o_acc[u] = o_acc[u] * corr + p * vv[u];
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
                float* op = out_buf + ((size_t)b * INK_NH + hh) * INK_HD + lane * 8;
                for (int u = 0; u < 8; ++u) op[u] = go[u] * il;
            }
        });
    });
}

inline void k_dspark_swiglu_sg16(const float* gate, const float* up, float* out, int N) {
    Q().parallel_for(NR((N + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = (int)it.get_global_id(0);
        if (i < N) {
            const float g = gate[i];
            out[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
        }
    });
}

inline void k_dspark_markov_joint_sg16(const float* state_buf, const InkTensorInfo& mw1,
                                       const float* cur_all, const float* ctx_vec,
                                       const int* tok_chain, float* joint_out, int slot_i, int V_padded) {
    const uint8_t* w1 = mw1.dev_ptr;
    const int fmt = mw1.fmt;
    Q().parallel_for(NR((512 + 2 * INK_D + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int g = (int)it.get_global_id(0);
        if (g < 256) {
            joint_out[g] = state_buf[g];
        } else if (g < 512) {
            const int prev_tok = tok_chain[slot_i - 1];
            if (fmt == FMT_BF16) {
                joint_out[g] = bf16_to_f32(reinterpret_cast<const uint16_t*>(w1)[(size_t)prev_tok * 256 + (g - 256)]);
            } else if (fmt == FMT_F32) {
                joint_out[g] = reinterpret_cast<const float*>(w1)[(size_t)prev_tok * 256 + (g - 256)];
            } else {
                const float sc = reinterpret_cast<const float*>(w1)[prev_tok];
                const int8_t* row = reinterpret_cast<const int8_t*>(w1 + (size_t)V_padded * 4) + (size_t)prev_tok * 256;
                joint_out[g] = (float)row[g - 256] * sc;
            }
        } else if (g < 512 + INK_D) {
            joint_out[g] = cur_all[(size_t)slot_i * INK_D + (g - 512)];
        } else if (g < 512 + 2 * INK_D) {
            joint_out[g] = ctx_vec[g - (512 + INK_D)];
        }
    });
}

inline void k_dspark_markov_rnn_update_sg16(const float* proj, const float* proj_b,
                                            float* state_buf, float* latent_out) {
    Q().parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int r = (int)it.get_local_id(0);
        const float p0 = proj[r] + proj_b[r];
        const float p1 = proj[256 + r] + proj_b[256 + r];
        const float p2 = proj[512 + r] + proj_b[512 + r];
        const float gate = 1.0f / (1.0f + sycl::exp(-p0));
        const float cand = sycl::tanh(sycl::clamp(p1, -15.0f, 15.0f));
        state_buf[r] = cand + gate * (state_buf[r] - cand);
        latent_out[r] = sycl::tanh(sycl::clamp(p2, -15.0f, 15.0f));
    });
}

inline void k_dspark_add_bias_sg16(float* x, const float* bias, int N) {
    Q().parallel_for(NR((N + 255) / 256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
        const int i = (int)it.get_global_id(0);
        if (i < N) x[i] += bias[i];
    });
}

inline void k_dspark_combine_argmax_sg16(const float* base_logits, const float* bias_logits,
                                         float* part_val, int* part_idx, int* tok_chain,
                                         int slot_i, int V, float cap) {
    const float inv_cap = 1.0f / cap;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_v(256, h);
        sycl::local_accessor<int, 1> sh_i(256, h);
        h.parallel_for(NR(256, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int tg = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            float bv = -1e30f; int bi = 0;
            for (int v = tg * 256 + tid; v < V; v += 256 * 256) {
                const float val = cap * sycl::tanh(sycl::clamp((base_logits[v] + bias_logits[v]) * inv_cap, -15.0f, 15.0f));
                if (val > bv || (val == bv && v < bi)) { bv = val; bi = v; }
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
            if (tid == 0) { part_val[tg] = sh_v[0]; part_idx[tg] = sh_i[0]; }
        });
    });
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_v(256, h);
        sycl::local_accessor<int, 1> sh_i(256, h);
        h.parallel_for(NR(1, 256), [=](sycl::nd_item<1> it) INK_REQD_SG16 {
            const int tid = (int)it.get_local_id(0);
            sh_v[tid] = part_val[tid]; sh_i[tid] = part_idx[tid];
            sycl::group_barrier(it.get_group());
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    const float v1 = sh_v[tid + s]; const int i1 = sh_i[tid + s];
                    if (v1 > sh_v[tid] || (v1 == sh_v[tid] && i1 < sh_i[tid])) { sh_v[tid] = v1; sh_i[tid] = i1; }
                }
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) tok_chain[slot_i] = sh_i[0];
        });
    });
}

}  // namespace ink_sycl
