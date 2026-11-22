// engine/tests/quant/ink_gguf_transcode_test.cpp - Unit tests for GGUF_RAW -> S_K streaming expert transcoder
// and Level Zero V1 default guard (Battlemage B70 oneMKL crash fix).
// MIT License - Copyright (c) 2026 Strata contributors
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "inkling2/ink_gguf_transcode.h"
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/quant/sk_codecs.hpp"

using namespace strata::quant::sk;
using namespace ink2::transcode;

static float cos_sim(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    return (na > 0.0 && nb > 0.0) ? static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0f;
}

static float prng_f(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (static_cast<float>(s & 0xFFFFu) / 32768.0f - 1.0f) * 0.5f;
}

static bool test_sk_roundtrip(SkCode code, float min_cos) {
    constexpr size_t N = 2304;
    std::vector<float> src(N), dec(N);
    uint32_t rng = 0x12345678u ^ static_cast<uint32_t>(code);
    for (size_t i = 0; i < N; ++i) {
        float x = prng_f(rng);
        if ((i % 97) == 0) x *= 2.5f;
        src[i] = x;
    }
    std::vector<uint8_t> packed((N / 256) * superblock_bytes(code));
    encode_sk_row(src.data(), N, code, packed.data());
    dequant_sk_row(packed.data(), code, dec.data(), N);
    float c = cos_sim(src.data(), dec.data(), N);
    printf("[test_sk_roundtrip] code=%d cos=%.6f (min=%.4f)\n", (int)code, c, min_cos);
    return c >= min_cos;
}

static bool test_gguf_raw_expert_transcode(uint32_t ggml_type, SkCode want_code, bool interleaved, float min_cos) {
    // Use real Inkling-2-Mini-compatible dimensions (multiples of 256): hidden=512, inter=256
    constexpr size_t H = 512;
    constexpr size_t I = 256;
    const size_t gu_row_b = ink2::gguf_detail::gguf_row_bytes(ggml_type, static_cast<int64_t>(H));
    const size_t dn_row_b = ink2::gguf_detail::gguf_row_bytes(ggml_type, static_cast<int64_t>(I));
    if (gu_row_b == 0 || dn_row_b == 0) {
        fprintf(stderr, "unsupported ggml_type %u in test\n", ggml_type);
        return false;
    }

    // Build synthetic GGUF_RAW expert with F32/F16/BF16/Q8_0 weights & non-unit scales
    const size_t raw_dn_off = sizeof(GgufExpertHeader) + (2 * I) * gu_row_b;
    const size_t raw_total  = raw_dn_off + H * dn_row_b;
    std::vector<uint8_t> raw(raw_total, 0);
    GgufExpertHeader hdr{};
    hdr.magic = 0x47475546u;
    hdr.gu_type = ggml_type;
    hdr.dn_type = ggml_type;
    hdr.flags = interleaved ? 1u : 0u;
    hdr.gate_scale = 1.25f;
    hdr.up_scale = 0.85f;
    hdr.dn_scale = 1.15f;
    std::memcpy(raw.data(), &hdr, sizeof(hdr));

    auto fill_row = [&](uint8_t* dst, size_t cols, uint32_t seed) {
        std::vector<float> f(cols);
        uint32_t rng = seed * 2654435761u + 0x9E3779B9u;
        for (size_t c = 0; c < cols; ++c) f[c] = prng_f(rng);
        if (ggml_type == 0) {  // F32
            std::memcpy(dst, f.data(), cols * 4);
        } else if (ggml_type == 1) {  // F16
            uint16_t* p = reinterpret_cast<uint16_t*>(dst);
            for (size_t c = 0; c < cols; ++c) p[c] = f32_to_fp16_bits(f[c]);
        } else if (ggml_type == 30) {  // BF16
            uint16_t* p = reinterpret_cast<uint16_t*>(dst);
            for (size_t c = 0; c < cols; ++c) {
                uint32_t u = 0; std::memcpy(&u, &f[c], 4);
                p[c] = static_cast<uint16_t>(u >> 16);
            }
        } else if (ggml_type == 8) {  // Q8_0 (32 elems: 2B fp16 d + 32B int8)
            for (size_t b = 0; b < cols / 32; ++b) {
                float amax = 0.0f;
                for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(f[b * 32 + j]));
                float d = amax / 127.0f;
                uint16_t db = f32_to_fp16_bits(d);
                float df = fp16_bits_to_f32(db);
                uint8_t* blk = dst + b * 34;
                std::memcpy(blk, &db, 2);
                int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
                for (size_t j = 0; j < 32; ++j) {
                    int q = (df > 0.0f) ? static_cast<int>(std::lround(f[b * 32 + j] / df)) : 0;
                    qs[j] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
                }
            }
        }
    };

    for (size_t r = 0; r < 2 * I; ++r) fill_row(raw.data() + sizeof(GgufExpertHeader) + r * gu_row_b, H, static_cast<uint32_t>(r + 1));
    for (size_t r = 0; r < H; ++r)     fill_row(raw.data() + raw_dn_off + r * dn_row_b, I, static_cast<uint32_t>(r + 1000));

    uint32_t need_dn = 0, need_bytes = 0;
    transcoded_sk_expert_layout(want_code, want_code, H, I, need_dn, need_bytes);
    std::vector<uint8_t> sk_body(need_bytes, 0);
    uint32_t out_dn = 0;
    uint8_t out_gu = 0, out_dn_c = 0;
    std::string err;
    if (!transcode_gguf_raw_expert_into(raw.data(), raw.size(), static_cast<uint32_t>(raw_dn_off),
                                        H, I, sk_body.data(), sk_body.size(),
                                        out_dn, out_gu, out_dn_c, err)) {
        fprintf(stderr, "transcode failed: %s\n", err.c_str());
        return false;
    }
    if (out_gu != static_cast<uint8_t>(want_code) || out_dn_c != static_cast<uint8_t>(want_code) || out_dn != need_dn) {
        fprintf(stderr, "transcode metadata mismatch\n");
        return false;
    }

    // Compare eval_inkling2_expert on original GGUF_RAW vs transcoded S_K!
    std::vector<float> x(H), y_ref(H), y_sk(H);
    uint32_t xrng = 0xCAFEBABEu;
    for (size_t i = 0; i < H; ++i) x[i] = prng_f(xrng) * 1.4f;
    strata::inkling2::eval_inkling2_expert(raw.data(), SkCode::GGUF_RAW, SkCode::GGUF_RAW,
                                           static_cast<uint32_t>(raw_dn_off), x.data(), H, I, y_ref.data());
    strata::inkling2::eval_inkling2_expert(sk_body.data(), static_cast<SkCode>(out_gu), static_cast<SkCode>(out_dn_c),
                                           out_dn, x.data(), H, I, y_sk.data());
    float c = cos_sim(y_ref.data(), y_sk.data(), H);
    printf("[test_gguf_raw_expert] ggml=%u code=%u interleaved=%d cos=%.6f (min=%.4f)\n",
           ggml_type, (unsigned)out_gu, interleaved ? 1 : 0, c, min_cos);
    return c >= min_cos;
}

int main() {
    bool ok = true;
    ok &= test_sk_roundtrip(SkCode::S4_K, 0.994f);
    ok &= test_sk_roundtrip(SkCode::S6_K, 0.9997f);
    ok &= test_sk_roundtrip(SkCode::S8_K, 0.99997f);

    // Check codec auto-selection & block_geometry for IQ1_S (19) and IQ1_M (29)
    ok &= (choose_sk_code_for_ggml(12) == SkCode::S4_K);  // Q4_K -> S4_K
    ok &= (choose_sk_code_for_ggml(19) == SkCode::S4_K);  // IQ1_S -> S4_K
    ok &= (choose_sk_code_for_ggml(14) == SkCode::S6_K);  // Q6_K -> S6_K
    ok &= (choose_sk_code_for_ggml(8)  == SkCode::S8_K);  // Q8_0 -> S8_K
    ok &= (ink2::gguf_detail::gguf_row_bytes(19, 2304) == 9u * 50u);  // IQ1_S: 256 elems / 50 B
    ok &= (ink2::gguf_detail::gguf_row_bytes(29, 2304) == 9u * 56u);  // IQ1_M: 256 elems / 56 B

    // End-to-end GGUF_RAW -> S_K expert evaluation parity (split & interleaved, F16/BF16/Q8_0)
    ok &= test_gguf_raw_expert_transcode(8,  SkCode::S8_K, false, 0.9998f);
    ok &= test_gguf_raw_expert_transcode(8,  SkCode::S8_K, true,  0.9998f);
    ok &= test_gguf_raw_expert_transcode(1,  SkCode::S8_K, false, 0.9998f);
    ok &= test_gguf_raw_expert_transcode(30, SkCode::S8_K, true,  0.9998f);

    // Force S4_K & S6_K via INK_GGUF_SK_BITS override to verify both paths end-to-end
#ifdef _WIN32
    _putenv_s("INK_GGUF_SK_BITS", "4");
#else
    setenv("INK_GGUF_SK_BITS", "4", 1);
#endif
    ok &= test_gguf_raw_expert_transcode(8, SkCode::S4_K, false, 0.993f);
#ifdef _WIN32
    _putenv_s("INK_GGUF_SK_BITS", "6");
#else
    setenv("INK_GGUF_SK_BITS", "6", 1);
#endif
    ok &= test_gguf_raw_expert_transcode(8, SkCode::S6_K, true, 0.9990f);

    if (!ok) {
        fprintf(stderr, "ink_gguf_transcode_test: FAILED\n");
        return 1;
    }
    printf("ink_gguf_transcode_test: ALL PASS\n");
    return 0;
}
