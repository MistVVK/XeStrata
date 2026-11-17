// strata-split: load-time conversion of the small tensors of a GGUF whose types the engine does not read, to the types it does
// (what Unsloth's UD-Q4_K_XL file carries: Q8_0 for every dense matrix, the shared experts, the embeddings, the hyper-connection
// and PLE projections; F32 for the routers, the injections and the SSM gates). A set with other types for them (the ISTA-DAS IQ3_S
// set keeps K-quants, IQ4_XS and BF16 there) is dequantized to float and quantized again, once, at load. A tensor that already has the
// target type is never touched, so UD-Q4_K_XL and Q8_0 files load exactly as before. Pure host code, no CUDA.
//
// The dequantizers are ggml's reference ones (ggml-quants.c, dequantize_row_*), written out here because the engine does not link
// ggml's CPU library; tests/split/ check them against gguf-py on real tensors.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace fnx {
namespace conv {

inline float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float f;
    if (e == 0) f = std::ldexp((float) m, -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = std::ldexp((float) (m + 1024), (int) e - 25);
    return s ? -f : f;
}
inline uint16_t f2h(float x) {   // round to nearest even, as ggml's GGML_FP32_TO_FP16
    uint32_t b;
    std::memcpy(&b, &x, 4);
    const uint32_t s = (b >> 16) & 0x8000, e = (b >> 23) & 0xFF, m = b & 0x7FFFFF;
    if (e == 0xFF) return (uint16_t) (s | 0x7C00 | (m ? 0x200 : 0));
    int ne = (int) e - 127 + 15;
    if (ne >= 31) return (uint16_t) (s | 0x7C00);
    if (ne <= 0) {
        if (ne < -10) return (uint16_t) s;
        const uint32_t mm = (m | 0x800000) >> (1 - ne);
        uint32_t r = mm >> 13;
        const uint32_t rem = mm & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) ne << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t) (s | r);
}
inline float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline const int8_t* kvalues_iq4nl() {
    static const int8_t k[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return k;
}
// the K-quants' 6-bit scale and min of sub-block j (ggml get_scale_min_k4)
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// n values (a whole number of blocks) of `type` at `src`, as floats; throws on a type it has no reader for
inline void dequantize(uint32_t type, const uint8_t* src, size_t n, float* y) {
    switch (type) {
    case T_F32: std::memcpy(y, src, n * 4); return;
    case T_F16:
        for (size_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); y[i] = h2f(h); }
        return;
    case T_BF16:
        for (size_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); y[i] = bf2f(h); }
        return;
    case T_Q8_0:
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = src + b * 34;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 32; ++j) y[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return;
    case T_Q4_0:
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = src + b * 18;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 16; ++j) {
                y[b * 32 + j] = d * (float) ((int) (p[2 + j] & 0xF) - 8);
                y[b * 32 + j + 16] = d * (float) ((int) (p[2 + j] >> 4) - 8);
            }
        }
        return;
    case T_Q2_0:
        for (size_t b = 0; b < n / 64; ++b) {
            const uint8_t* p = src + b * 18;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 64; ++j) y[b * 64 + j] = d * (float) ((int) ((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
        }
        return;
    case T_IQ4_NL:
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = src + b * 18;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 16; ++j) {
                y[b * 32 + j] = d * (float) kvalues_iq4nl()[p[2 + j] & 0xF];
                y[b * 32 + j + 16] = d * (float) kvalues_iq4nl()[p[2 + j] >> 4];
            }
        }
        return;
    case T_IQ4_XS:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 136;
            uint16_t dh, sh;
            std::memcpy(&dh, p, 2);
            std::memcpy(&sh, p + 2, 2);
            const uint8_t* sl = p + 4;
            const uint8_t* qs = p + 8;
            const float d = h2f(dh);
            float* o = y + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * (float) (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    o[j] = dl * (float) kvalues_iq4nl()[qs[j] & 0xF];
                    o[j + 16] = dl * (float) kvalues_iq4nl()[qs[j] >> 4];
                }
                o += 32;
                qs += 16;
            }
        }
        return;
    case T_Q4_K:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 144;
            uint16_t dh, mh;
            std::memcpy(&dh, p, 2);
            std::memcpy(&mh, p + 2, 2);
            const float d = h2f(dh), mn = h2f(mh);
            const uint8_t* sc = p + 4;
            const uint8_t* q = p + 16;
            float* o = y + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, s1, m1);
                scale_min_k4(is + 1, sc, s2, m2);
                const float d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
                for (int l = 0; l < 32; ++l) *o++ = d1 * (float) (q[l] & 0xF) - mm1;
                for (int l = 0; l < 32; ++l) *o++ = d2 * (float) (q[l] >> 4) - mm2;
                q += 32;
                is += 2;
            }
        }
        return;
    case T_Q5_K:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 176;
            uint16_t dh, mh;
            std::memcpy(&dh, p, 2);
            std::memcpy(&mh, p + 2, 2);
            const float d = h2f(dh), mn = h2f(mh);
            const uint8_t* sc = p + 4;
            const uint8_t* qh = p + 16;
            const uint8_t* ql = p + 48;
            float* o = y + b * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, s1, m1);
                scale_min_k4(is + 1, sc, s2, m2);
                const float d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
                for (int l = 0; l < 32; ++l) *o++ = d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - mm1;
                for (int l = 0; l < 32; ++l) *o++ = d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - mm2;
                ql += 32;
                is += 2;
                u1 = (uint8_t) (u1 << 2);
                u2 = (uint8_t) (u2 << 2);
            }
        }
        return;
    case T_Q6_K:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 210;
            const uint8_t* ql = p;
            const uint8_t* qh = p + 128;
            const int8_t* sc = (const int8_t*) (p + 192);
            uint16_t dh;
            std::memcpy(&dh, p + 208, 2);
            const float d = h2f(dh);
            float* o = y + b * 256;
            for (int h = 0; h < 256; h += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    o[l] = d * (float) sc[is] * (float) q1;
                    o[l + 32] = d * (float) sc[is + 2] * (float) q2;
                    o[l + 64] = d * (float) sc[is + 4] * (float) q3;
                    o[l + 96] = d * (float) sc[is + 6] * (float) q4;
                }
                o += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    }
    throw std::runtime_error("tensor_convert: no reader for ggml type " + std::to_string(type));
}

// Q8_0 as ggml's quantize_row_q8_0_ref: per 32 values d = amax / 127 (stored as f16), q = round(x / d)
inline void quantize_q8_0(const float* x, size_t n, uint8_t* dst) {
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
        const uint16_t dh = f2h(d);
        std::memcpy(dst + b * 34, &dh, 2);
        for (int j = 0; j < 32; ++j) dst[b * 34 + 2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
}

// the type the engine reads a tensor of this name as (what UD-Q4_K_XL carries), or 0xFFFFFFFF for a tensor that is left alone
inline uint32_t wanted_type(const std::string& name) {
    auto ends = [&](const char* s) { const size_t l = std::strlen(s); return name.size() >= l && name.compare(name.size() - l, l, s) == 0; };
    static const char* q8[] = {".attn_gate.weight", ".attn_k.weight", ".attn_output.weight", ".attn_q.weight", ".attn_qkv.weight", ".attn_v.weight",
                               ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight", ".hc_attn_down.weight",
                               ".hc_attn_up.weight", ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".ple_key.weight", ".ple_value.weight",
                               ".ssm_out.weight"};
    static const char* f32[] = {".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight", ".hc_attn_inject.weight", ".hc_ffn_inject.weight",
                                ".ssm_alpha.weight", ".ssm_beta.weight", ".ple_conv1d.weight"};
    for (const char* s : q8)
        if (ends(s)) return T_Q8_0;
    for (const char* s : f32)
        if (ends(s)) return T_F32;
    if (name == "output.weight" || name == "output_hc_down.weight" || name == "output_hc_up.weight" || name == "token_embd.weight") return T_Q8_0;
    return 0xFFFFFFFFu;
}

// `t` converted to `target` (Q8_0 or F32) into `out`; the Tensor then points at `out`
inline void convert(Tensor& t, uint32_t target, std::vector<uint8_t>& out) {
    size_t n = 1;
    for (int d = 0; d < 4; ++d) n *= (size_t) t.ne[d];
    const TypeInfo from = type_info(t.type);
    if (n % (size_t) from.block) throw std::runtime_error(t.name + ": not a whole number of blocks");
    if (target == T_Q8_0 && t.ne[0] % 32) throw std::runtime_error(t.name + ": rows are not a whole number of Q8_0 blocks");
    std::vector<float> f(n);
    dequantize(t.type, t.data, n, f.data());
    if (target == T_F32) {
        out.resize(n * 4);
        std::memcpy(out.data(), f.data(), n * 4);
    } else if (target == T_Q8_0) {
        out.resize(n / 32 * 34);
        quantize_q8_0(f.data(), n, out.data());
    } else {
        throw std::runtime_error("tensor_convert: target type " + std::to_string(target));
    }
    t.type = target;
    t.data = out.data();
    t.nbytes = out.size();
    t.shard = -1;   // not in a GGUF file any more: readers that stage from the file by offset must take the bytes at t.data
    t.offset = 0;
}

// Strata's Q2_0 (64 values in 18 bytes: f16 d, two bits per value, value j in byte j / 4 at bit 2 (j % 4), meaning (code - 1) * d) as two
// Q4_0 blocks (32 values in 18 bytes: f16 d, nibble i of the low half = value i, of the high half = value i + 16, meaning (nibble - 8) * d).
// Lossless: nibble = code + 7 gives (code + 7 - 8) * d, the same product. The engine's Q4_0 kernels then read it (the draft layer's
// experts.bin goes the same way, model.cpp Mtp::experts). Twice the bytes.
inline void repack_q2_0_to_q4_0(Tensor& t, std::vector<uint8_t>& out) {
    if (t.ne[0] % 64) throw std::runtime_error(t.name + ": rows are not a whole number of Q2_0 blocks");
    const size_t nb = t.nbytes / 18;
    out.resize(nb * 36);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* in = t.data + b * 18;
        auto code = [&](int j) { return (in[2 + j / 4] >> (2 * (j % 4))) & 3; };
        for (int h = 0; h < 2; ++h) {
            uint8_t* o = out.data() + (2 * b + (size_t) h) * 18;
            o[0] = in[0];
            o[1] = in[1];
            for (int i = 0; i < 16; ++i) o[2 + i] = (uint8_t) ((code(32 * h + i) + 7) | ((code(32 * h + 16 + i) + 7) << 4));
        }
    }
    t.type = T_Q4_0;
    t.data = out.data();
    t.nbytes = out.size();
    t.shard = -1;
    t.offset = 0;
}

}  // namespace conv
}  // namespace fnx
