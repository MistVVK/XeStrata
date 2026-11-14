// tests/quant/nvfp4_decode_model_test.cpp - the CPU model of the fused prompt kernels' NVFP4 weight decode (src/prefill/moe_fused_iq.cu: load_unit /
// convert<T_NVFP4>, ue4m3_half) against the engine's own CPU dequantizer (strata::quant, which is ggml's dequantize_row_nvfp4), no GPU, no ggml.
//
// The kernels turn a 36-byte block {uint8 d[4] (UE4M3, one per 16 values); uint8 qs[32] (E2M1 codes, split-half inside each 16-value group: byte j
// of a group holds value j in its low and value 8 + j in its high nibble)} into, per 32-value sub-block ib (0, 1): eight 32-bit words of int8 (natural
// value order; the E2M1 value doubled: 0, 1, 2, 3, 4, 6, 8, 12, signs by bit 3) and two scales s0 (values 0-15) and s1 (16-31) = the UE4M3 byte
// halved.  The weight is s * q, exact in FP32.  This file holds the same decode written from the format's definition (not from the kernel's bit
// tricks) and checks, for every scale byte and every code and for seeded random blocks, that
//   - the kernel's scale construction (the bit form of ue4m3_half, copied here word for word) equals the definition for all 256 bytes,
//   - model value s * q equals strata::quant::dequant_row (bit for bit),
//   - the int8 sums the mma forms stay inside 2^22 (the MAGIC-bias trick of the kernels needs |sum q a| < 2^22): worst case 16 * 12 * 127 per scale group.
// The GPU side of the same statement (the real kernel code against ggml's own dequantizer) is tests/cuda/prefill_decode_golden.cu.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/quant/dequant.hpp"
#include "strata/quant/types.hpp"

namespace sq = strata::quant;

static int g_fail = 0;
#define CHECK(c, ...)                                  \
    do {                                               \
        if (!(c)) {                                    \
            ++g_fail;                                  \
            if (g_fail < 20) {                         \
                std::printf("FAIL line %d: ", __LINE__); \
                std::printf(__VA_ARGS__);              \
                std::printf("\n");                     \
            }                                          \
        }                                              \
    } while (0)

namespace {

// the format's definition: E2M1 (1 sign, 2 exponent, 1 mantissa bit), doubled, as an int
int e2m1_doubled(int code) {
    const int s = (code >> 3) & 1, e = (code >> 1) & 3, m = code & 1;
    // value = (-1)^s * (e == 0 ? m * 0.5 : (1 + m / 2) * 2^(e - 1)); doubled: e == 0: m, else (2 + m) * 2^(e - 1)
    const int mag = e == 0 ? m : (2 + m) << (e - 1);
    return s ? -mag : mag;
}
// UE4M3 (4 exponent bits bias 7, 3 mantissa bits, no sign, no infinity, 0x7F = NaN) halved; NaN reads as 0 (ggml's CPU and CUDA decoders)
float ue4m3_half_def(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const double v = e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (float) (v * 0.5);
}
// moe_fused_iq.cu's ue4m3_half, the same statements
float ue4m3_half_kernel(uint32_t x) {
    x &= 0xFF;
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t ex = (x >> 3) & 0xF, man = x & 0x7;
    const uint32_t bits = ((ex + 119u) << 23) | (man << 20);
    float f;
    std::memcpy(&f, &bits, 4);
    return ex == 0 ? (float) man * 0.0009765625f : f;
}

struct Sub {
    int8_t q[32];
    float s0, s1;
};
// sub-block ib of one block, from the definition
Sub model_sub(const uint8_t* blk, int ib) {
    Sub r;
    for (int g = 0; g < 2; ++g) {          // the two 16-value groups of the sub-block
        const int grp = 2 * ib + g;        // group 0..3 of the block
        for (int j = 0; j < 8; ++j) {
            const uint8_t b = blk[4 + 8 * grp + j];
            r.q[16 * g + j] = (int8_t) e2m1_doubled(b & 15);
            r.q[16 * g + 8 + j] = (int8_t) e2m1_doubled(b >> 4);
        }
    }
    r.s0 = ue4m3_half_def(blk[2 * ib]);
    r.s1 = ue4m3_half_def(blk[2 * ib + 1]);
    return r;
}

struct SplitMix {
    uint64_t s;
    uint64_t next() {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
};

}  // namespace

int main() {
    // 1. the scale byte: the kernel's bit construction against the definition, all 256 values
    for (int x = 0; x < 256; ++x) {
        const float a = ue4m3_half_def((uint8_t) x), b = ue4m3_half_kernel((uint32_t) x);
        CHECK(std::memcmp(&a, &b, 4) == 0, "UE4M3 byte 0x%02x: definition %.9g, kernel form %.9g", x, a, b);
    }
    // the format's own anchors: 0x38 = 1.0 (exp 7, man 0) -> 0.5 halved; 0x7E = 448 -> 224; 0x01 = 2^-9 -> 2^-10
    CHECK(ue4m3_half_def(0x38) == 0.5f && ue4m3_half_def(0x7E) == 224.0f && ue4m3_half_def(0x01) == std::ldexp(1.0f, -10), "anchors");

    // 2. every (scale byte, code) pair through the engine's dequantizer: one block whose four scale bytes and all 64 codes are controlled
    int64_t pairs = 0;
    for (int x = 0; x < 256; ++x) {
        uint8_t blk[36];
        for (int s = 0; s < 4; ++s) blk[s] = (uint8_t) x;
        for (int j = 0; j < 32; ++j) blk[4 + j] = (uint8_t) (((j & 7) * 2) | (((j & 7) * 2 + 1) << 4));   // codes 0..15, each value position gets some code
        // rotate the codes over the 32 bytes so that over the 16 rounds every code meets every position
        for (int rot = 0; rot < 16; ++rot) {
            for (int j = 0; j < 32; ++j) blk[4 + j] = (uint8_t) (((j + rot) & 15) | (((j + 2 * rot + 7) & 15) << 4));
            float ref[64];
            CHECK(sq::dequant_row(sq::Type::NVFP4, blk, ref, 64), "dequant_row NVFP4");
            for (int ib = 0; ib < 2; ++ib) {
                const Sub m = model_sub(blk, ib);
                for (int j = 0; j < 32; ++j) {
                    const float mine = (j < 16 ? m.s0 : m.s1) * (float) m.q[j], want = ref[32 * ib + j];
                    CHECK(std::memcmp(&mine, &want, 4) == 0 || (mine == 0.0f && want == 0.0f), "scale 0x%02x rot %d sub %d value %d: model %.9g dequant %.9g", x, rot, ib, j, mine, want);
                    ++pairs;
                }
            }
        }
    }
    // 3. seeded random blocks (every byte random, as the golden test's): 4096 blocks, and the integer sums' bound
    SplitMix g{0x4e564650344d3300ull};
    int64_t maxsum = 0;
    for (int b = 0; b < 4096; ++b) {
        uint8_t blk[36];
        for (int i = 0; i < 36; i += 8) {
            const uint64_t r = g.next();
            for (int k = 0; k < 8 && i + k < 36; ++k) blk[i + k] = (uint8_t) (r >> (8 * k));
        }
        float ref[64];
        CHECK(sq::dequant_row(sq::Type::NVFP4, blk, ref, 64), "dequant_row NVFP4 (random)");
        for (int ib = 0; ib < 2; ++ib) {
            const Sub m = model_sub(blk, ib);
            for (int j = 0; j < 32; ++j) {
                const float mine = (j < 16 ? m.s0 : m.s1) * (float) m.q[j], want = ref[32 * ib + j];
                CHECK(std::memcmp(&mine, &want, 4) == 0 || (mine == 0.0f && want == 0.0f), "random block %d sub %d value %d: model %.9g dequant %.9g", b, ib, j, mine, want);
            }
            // worst int8 sum of the kernel's m16n8k16 (one scale group of 16) against codes +-127
            for (int h = 0; h < 2; ++h) {
                int64_t sum = 0;
                for (int j = 16 * h; j < 16 * h + 16; ++j) sum += std::abs((int) m.q[j]) * 127;
                if (sum > maxsum) maxsum = sum;
            }
        }
    }
    CHECK(maxsum <= 16 * 12 * 127 && 16 * 12 * 127 < (1 << 22), "int8 sum bound %lld", (long long) maxsum);
    std::printf("nvfp4_decode_model_test: %s (%lld value pairs, scale bytes 256 of 256, max |sum q a| per 16 values %lld of 2^22)\n", g_fail ? "FAIL" : "ok",
                (long long) pairs, (long long) maxsum);
    return g_fail ? 1 : 0;
}
