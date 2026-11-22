// Copyright (c) 2026 Strata contributors
// SPDX-License-Identifier: MIT
//
// src/kernels/hip/hip_unsloth_verify.cu - Standalone multi-arch compile check, parity verifier,
// and live micro-benchmark harness for hip_unsloth_wmma & hip_unsloth_dp4a across:
//   Q4_K (12), Q5_K (13), Q6_K (14), Q8_0 (8), IQ4_XS (23), IQ3_XXS (18), NVFP4 (40), FP16, BF16.

#include "strata/kernels/hip_unsloth_wmma.hpp"
#include "strata/kernels/hip_unsloth_dp4a.hpp"

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "strata/quant/strata-quant-common.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels::hip_unsloth;

namespace {

#define HIP_CHECK(expr) do {                                                        \
    hipError_t _e = (expr);                                                         \
    if (_e != hipSuccess) {                                                         \
        std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,           \
                     hipGetErrorString(_e));                                        \
        std::exit(1);                                                               \
    }                                                                               \
} while (0)

const char* type_name(int ty) {
    switch (ty) {
        case 8:  return "Q8_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 18: return "IQ3_XXS";
        case 23: return "IQ4_XS";
        case 40: return "NVFP4";
        default: return "UNKNOWN";
    }
}

// Host FP16/BF16 bit conversions
uint16_t f32_to_f16_bits(float v) {
    __half h = __float2half(v);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float f16_bits_to_f32(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}
uint16_t f32_to_bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}
float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float host_ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    uint32_t exp = (x >> 3) & 0xFu;
    uint32_t man = x & 0x7u;
    if (exp == 0) return static_cast<float>(man) * 0.0009765625f;
    uint32_t bits = ((exp + 119u) << 23) | (man << 20);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Host copy of codebooks (exact GGUF tables)
constexpr int8_t kHostIq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
constexpr int8_t kHostFp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Device primitive unit test kernel
__global__ void test_primitives_kernel(int* bad_count) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < 4096) {
        // Test hip_iq4_lut16 & hip_fp4_lut16 over 4-nibble patterns packed into a 32-bit word
        const uint32_t q4 = static_cast<uint32_t>(tid) | (static_cast<uint32_t>((tid * 2654435761u) >> 16) << 16);
        const int2 vi = hip_iq4_lut16(q4);
        const int2 vf = hip_fp4_lut16(q4);
        const auto* ix = reinterpret_cast<const int8_t*>(&vi.x);
        const auto* iy = reinterpret_cast<const int8_t*>(&vi.y);
        const auto* fx = reinterpret_cast<const int8_t*>(&vf.x);
        const auto* fy = reinterpret_cast<const int8_t*>(&vf.y);
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (q4 >> (8 * b)) & 0xFFu;
            if (ix[b] != kvalues_iq4nl[byte & 0xFu] || iy[b] != kvalues_iq4nl[byte >> 4]) atomicAdd(bad_count, 1);
            if (fx[b] != kvalues_fp4[byte & 0xFu]   || fy[b] != kvalues_fp4[byte >> 4])   atomicAdd(bad_count, 1);
        }
    }
    if (tid < 256 * 16) {
        const uint32_t g = iq3xxs_grid[tid >> 4];
        const uint32_t s4 = tid & 15;
        const int neg = hip_iq_sign_negate(g, s4);
        const auto* gb = reinterpret_cast<const uint8_t*>(&g);
        const auto* nb = reinterpret_cast<const int8_t*>(&neg);
        for (int b = 0; b < 4; ++b) {
            const int expect = ((s4 >> b) & 1u) ? -static_cast<int>(gb[b]) : static_cast<int>(gb[b]);
            if (nb[b] != expect) atomicAdd(bad_count, 1);
        }
    }
    if (tid < 64 * 64) {
        const uint32_t b0 = tid & 63, b1 = (tid >> 6) & 63;
        const uint32_t b2 = (b0 * 7 + 13) & 63, b3 = (b1 * 11 + 5) & 63;
        const uint32_t packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        const int res = hip_sub32_bytes(packed);
        const auto* rb = reinterpret_cast<const int8_t*>(&res);
        if (rb[0] != static_cast<int>(b0) - 32 || rb[1] != static_cast<int>(b1) - 32 ||
            rb[2] != static_cast<int>(b2) - 32 || rb[3] != static_cast<int>(b3) - 32) {
            atomicAdd(bad_count, 1);
        }
    }
    if (tid < 256) {
        const uint8_t x = static_cast<uint8_t>(tid);
        const float got = hip_ue4m3_to_f32_half(x);
        float expect = 0.0f;
        if (x != 0 && x != 0x7F) {
            const int exp = (x >> 3) & 0xF;
            const int man = x & 0x7;
            expect = (exp == 0) ? ldexpf(static_cast<float>(man), -10)
                                : ldexpf(1.0f + static_cast<float>(man) * 0.125f, exp - 8);
        }
        if (got != expect) atomicAdd(bad_count, 1);
    }
}

// Fetch iq3xxs_grid to host once for exact CPU reference
__global__ void copy_iq3xxs_grid_kernel(uint32_t* dst) {
    const int i = threadIdx.x;
    if (i < 256) dst[i] = iq3xxs_grid[i];
}

std::vector<uint32_t> get_host_iq3xxs_grid() {
    uint32_t* d_g = nullptr;
    HIP_CHECK(hipMalloc(&d_g, 256 * sizeof(uint32_t)));
    copy_iq3xxs_grid_kernel<<<1, 256>>>(d_g);
    std::vector<uint32_t> h_g(256);
    HIP_CHECK(hipMemcpy(h_g.data(), d_g, 256 * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_g));
    return h_g;
}

// Host q8_1 quantizer (exact match to hip_q8_1_quant / q8_1_finite)
void host_quantize_q8_1(const float* x, block_q8_1* y, int n_rows, int n_cols) {
    const int nb = n_cols / 32;
    for (int r = 0; r < n_rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* src = x + static_cast<std::size_t>(r) * n_cols + b * 32;
            float amax = 0.0f, sum = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::fmax(amax, std::fabs(src[i]));
                sum += src[i];
            }
            float d = amax / 127.0f;
            if (std::fabs(d) > 65504.0f) d = std::copysign(65504.0f, d);
            if (std::fabs(sum) > 65504.0f) sum = std::copysign(65504.0f, sum);
            block_q8_1& dst = y[static_cast<std::size_t>(r) * nb + b];
            dst.ds = __halves2half2(__float2half(d), __float2half(sum));
            for (int i = 0; i < 32; ++i) {
                if (amax == 0.0f) dst.qs[i] = 0;
                else {
                    float q = std::round(src[i] / d);
                    dst.qs[i] = static_cast<int8_t>(q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
                }
            }
        }
    }
}

// Fill realistic random valid quantized weight blocks (no NaN/Inf/subnormal scale hazards)
void fill_random_quant_rows(int ty, uint8_t* dst, int n_rows, int n_cols, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.15f);
    std::uniform_int_distribution<int> i8_dist(-60, 60);
    std::uniform_int_distribution<int> ue4m3_dist(0x28, 0x58);  // normal UE4M3 scales
    const std::size_t rb = hip_quant_row_bytes(ty, n_cols);
    for (int r = 0; r < n_rows; ++r) {
        uint8_t* row = dst + static_cast<std::size_t>(r) * rb;
        if (ty == 8) {
            auto* blks = reinterpret_cast<block_q8_0*>(row);
            for (int b = 0; b < n_cols / 32; ++b) {
                blks[b].d = __float2half(scale_dist(rng));
                for (int i = 0; i < 32; ++i) blks[b].qs[i] = static_cast<int8_t>(i8_dist(rng));
            }
        } else if (ty == 12) {
            auto* blks = reinterpret_cast<block_q4_K*>(row);
            for (int b = 0; b < n_cols / 256; ++b) {
                blks[b].dm = __halves2half2(__float2half(scale_dist(rng)), __float2half(scale_dist(rng) * 0.5f));
                for (int i = 0; i < 12; ++i) blks[b].scales[i] = static_cast<uint8_t>(byte_dist(rng));
                for (int i = 0; i < 128; ++i) blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        } else if (ty == 13) {
            auto* blks = reinterpret_cast<block_q5_K*>(row);
            for (int b = 0; b < n_cols / 256; ++b) {
                blks[b].dm = __halves2half2(__float2half(scale_dist(rng)), __float2half(scale_dist(rng) * 0.5f));
                for (int i = 0; i < 12; ++i) blks[b].scales[i] = static_cast<uint8_t>(byte_dist(rng));
                for (int i = 0; i < 32; ++i) blks[b].qh[i] = static_cast<uint8_t>(byte_dist(rng));
                for (int i = 0; i < 128; ++i) blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        } else if (ty == 14) {
            auto* blks = reinterpret_cast<block_q6_K*>(row);
            for (int b = 0; b < n_cols / 256; ++b) {
                for (int i = 0; i < 128; ++i) blks[b].ql[i] = static_cast<uint8_t>(byte_dist(rng));
                for (int i = 0; i < 64; ++i) blks[b].qh[i] = static_cast<uint8_t>(byte_dist(rng));
                for (int i = 0; i < 16; ++i) blks[b].scales[i] = static_cast<int8_t>(i8_dist(rng) / 4);
                blks[b].d = __float2half(scale_dist(rng));
            }
        } else if (ty == 18) {
            auto* blks = reinterpret_cast<block_iq3_xxs*>(row);
            for (int b = 0; b < n_cols / 256; ++b) {
                blks[b].d = __float2half(scale_dist(rng));
                for (int i = 0; i < 96; ++i) blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        } else if (ty == 23) {
            auto* blks = reinterpret_cast<block_iq4_xs*>(row);
            for (int b = 0; b < n_cols / 256; ++b) {
                blks[b].d = __float2half(scale_dist(rng) * 0.1f);
                blks[b].scales_h = static_cast<uint16_t>(byte_dist(rng) | (byte_dist(rng) << 8));
                for (int i = 0; i < 4; ++i) blks[b].scales_l[i] = static_cast<uint8_t>(byte_dist(rng));
                for (int i = 0; i < 128; ++i) blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        } else if (ty == 40) {
            auto* blks = reinterpret_cast<block_nvfp4*>(row);
            for (int b = 0; b < n_cols / 64; ++b) {
                for (int i = 0; i < 4; ++i) blks[b].d[i] = static_cast<uint8_t>(ue4m3_dist(rng));
                for (int i = 0; i < 32; ++i) blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }
    }
}

// Host FP64 dequantizer for exact reference
void host_dequant_row(int ty, const uint8_t* row, int K, double* out, const std::vector<uint32_t>& iq3_grid) {
    if (ty == 8) {
        const auto* blks = reinterpret_cast<const block_q8_0*>(row);
        for (int b = 0; b < K / 32; ++b) {
            const double d = __half2float(blks[b].d);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = d * static_cast<double>(blks[b].qs[i]);
        }
    } else if (ty == 12) {
        const auto* blks = reinterpret_cast<const block_q4_K*>(row);
        for (int b = 0; b < K / 256; ++b) {
            const float2 dm = __half22float2(blks[b].dm);
            for (int j = 0; j < 8; ++j) {
                int sc, m;
                if (j < 4) { sc = blks[b].scales[j] & 63; m = blks[b].scales[j + 4] & 63; }
                else {
                    sc = (blks[b].scales[j + 4] & 0xF) | ((blks[b].scales[j - 4] >> 6) << 4);
                    m  = (blks[b].scales[j + 4] >> 4)  | ((blks[b].scales[j] >> 6) << 4);
                }
                const uint8_t* qs = blks[b].qs + (j >> 1) * 32;
                const int sh = (j & 1) * 4;
                for (int i = 0; i < 32; ++i) {
                    const int q = (qs[i] >> sh) & 0xF;
                    out[b * 256 + j * 32 + i] = static_cast<double>(dm.x) * sc * q - static_cast<double>(dm.y) * m;
                }
            }
        }
    } else if (ty == 13) {
        const auto* blks = reinterpret_cast<const block_q5_K*>(row);
        for (int b = 0; b < K / 256; ++b) {
            const float2 dm = __half22float2(blks[b].dm);
            for (int j = 0; j < 8; ++j) {
                int sc, m;
                if (j < 4) { sc = blks[b].scales[j] & 63; m = blks[b].scales[j + 4] & 63; }
                else {
                    sc = (blks[b].scales[j + 4] & 0xF) | ((blks[b].scales[j - 4] >> 6) << 4);
                    m  = (blks[b].scales[j + 4] >> 4)  | ((blks[b].scales[j] >> 6) << 4);
                }
                const uint8_t* qs = blks[b].qs + (j >> 1) * 32;
                const int sh = (j & 1) * 4;
                for (int i = 0; i < 32; ++i) {
                    const int q = ((qs[i] >> sh) & 0xF) | (((blks[b].qh[i] >> j) & 1) << 4);
                    out[b * 256 + j * 32 + i] = static_cast<double>(dm.x) * sc * q - static_cast<double>(dm.y) * m;
                }
            }
        }
    } else if (ty == 14) {
        const auto* blks = reinterpret_cast<const block_q6_K*>(row);
        for (int b = 0; b < K / 256; ++b) {
            const double d = __half2float(blks[b].d);
            for (int n = 0; n < 2; ++n) {
                for (int qd = 0; qd < 4; ++qd) {
                    const uint8_t* ql = blks[b].ql + n * 64 + (qd & 1) * 32;
                    const uint8_t* qh = blks[b].qh + n * 32;
                    const int ns = (qd >= 2) ? 4 : 0;
                    const int hs = 2 * qd;
                    for (int l = 0; l < 32; ++l) {
                        const int q6 = ((ql[l] >> ns) & 0xF) | (((qh[l] >> hs) & 3) << 4);
                        const int sc = blks[b].scales[n * 8 + 2 * qd + (l >> 4)];
                        out[b * 256 + n * 128 + qd * 32 + l] = d * sc * (q6 - 32);
                    }
                }
            }
        }
    } else if (ty == 23) {
        const auto* blks = reinterpret_cast<const block_iq4_xs*>(row);
        for (int b = 0; b < K / 256; ++b) {
            const double d = __half2float(blks[b].d);
            for (int j = 0; j < 8; ++j) {
                const int ls = ((blks[b].scales_l[j / 2] >> (4 * (j & 1))) & 0xF) |
                               (((blks[b].scales_h >> (2 * j)) & 3) << 4);
                const double dl = d * (ls - 32);
                const uint8_t* qs = blks[b].qs + 16 * j;
                for (int i = 0; i < 16; ++i) {
                    out[b * 256 + j * 32 + i]      = dl * kHostIq4nl[qs[i] & 0xF];
                    out[b * 256 + j * 32 + 16 + i] = dl * kHostIq4nl[qs[i] >> 4];
                }
            }
        }
    } else if (ty == 18) {
        const auto* blks = reinterpret_cast<const block_iq3_xxs*>(row);
        for (int b = 0; b < K / 256; ++b) {
            const double d = __half2float(blks[b].d);
            for (int j = 0; j < 8; ++j) {
                uint32_t aux32;
                std::memcpy(&aux32, blks[b].qs + 64 + 4 * j, 4);
                const int ls = static_cast<int>(aux32 >> 28);
                const double dl = d * (0.5 + ls) * 0.5;
                const uint8_t* qs = blks[b].qs + 8 * j;
                for (int l0 = 0; l0 < 8; l0 += 2) {
                    const uint32_t g0 = iq3_grid[qs[l0 + 0]];
                    const uint32_t g1 = iq3_grid[qs[l0 + 1]];
                    const uint32_t v7 = (aux32 >> (7 * (l0 >> 1))) & 0x7Fu;
                    const uint32_t s8 = v7 | ((static_cast<uint32_t>(__builtin_popcount(v7)) & 1u) << 7);
                    for (int k = 0; k < 4; ++k) {
                        const int v0 = static_cast<int>((g0 >> (8 * k)) & 0xFFu);
                        const int v1 = static_cast<int>((g1 >> (8 * k)) & 0xFFu);
                        out[b * 256 + j * 32 + 4 * l0 + k]     = dl * (((s8 >> k) & 1u) ? -v0 : v0);
                        out[b * 256 + j * 32 + 4 * l0 + 4 + k] = dl * (((s8 >> (4 + k)) & 1u) ? -v1 : v1);
                    }
                }
            }
        }
    } else if (ty == 40) {
        const auto* blks = reinterpret_cast<const block_nvfp4*>(row);
        for (int b = 0; b < K / 64; ++b) {
            for (int is = 0; is < 4; ++is) {
                const double sc = host_ue4m3_half(blks[b].d[is]);
                const uint8_t* qs = blks[b].qs + 8 * is;
                for (int i = 0; i < 8; ++i) {
                    out[b * 64 + is * 16 + i]     = sc * kHostFp4[qs[i] & 0xF];
                    out[b * 64 + is * 16 + 8 + i] = sc * kHostFp4[qs[i] >> 4];
                }
            }
        }
    }
}

// Host dequantized q8_1 vector -> FP64 (so DP4A dots compare against exact quantized activation math)
void host_dequant_q8_1(const block_q8_1* x, int K, double* out) {
    for (int b = 0; b < K / 32; ++b) {
        const double d = __low2float(x[b].ds);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = d * static_cast<double>(x[b].qs[i]);
    }
}

struct ErrStats {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

ErrStats compute_err(const float* got, const double* ref, std::size_t n) {
    double max_abs = 0.0, num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double diff = static_cast<double>(got[i]) - ref[i];
        max_abs = std::max(max_abs, std::fabs(diff));
        num += diff * diff;
        den += ref[i] * ref[i];
    }
    return {max_abs, den > 0.0 ? std::sqrt(num / den) : std::sqrt(num)};
}

template <typename Fn>
float bench_ms(Fn&& fn, int warmup = 5, int iters = 30) {
    for (int i = 0; i < warmup; ++i) fn();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn();
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));
    return ms / static_cast<float>(iters);
}

}  // namespace

int main() {
    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf("=== HIP Unsloth Parity & Benchmark Harness ===\n");
    std::printf("Device %d: %s | gcnArchName: %s | CUs: %d | WMMA=%d | EXP8_AUTO=%d\n\n",
                dev, prop.name, prop.gcnArchName, prop.multiProcessorCount,
                hip_wmma_device_available() ? 1 : 0, hip_exp8_auto_enabled(dev) ? 1 : 0);

    int failures = 0;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    const std::vector<uint32_t> iq3_grid = get_host_iq3xxs_grid();

    // -----------------------------------------------------------------------
    // TEST 1: Bit-exact device primitives
    // -----------------------------------------------------------------------
    {
        int* d_bad = nullptr;
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
        test_primitives_kernel<<<16, 256>>>(d_bad);
        HIP_CHECK(hipDeviceSynchronize());
        int bad = 0;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_bad));
        std::printf("[TEST 1] Bit-exact RDNA primitives (v_perm LUT16, sign_negate, sub32, UE4M3): %s (bad=%d)\n",
                    bad == 0 ? "PASS" : "FAIL", bad);
        if (bad != 0) ++failures;
    }

    // -----------------------------------------------------------------------
    // TEST 2: Dense Wave32 WMMA FP16 & BF16 GEMM (16x16, 64x64, 128x128 BK=64)
    // -----------------------------------------------------------------------
    if (hip_wmma_device_available()) {
        struct Shape { int T, N, K; float beta; const char* tag; };
        const Shape shapes[] = {
            {19, 23, 64,  0.0f, "16x16_1w beta=0"},
            {19, 23, 64,  1.0f, "16x16_1w beta=1"},
            {45, 53, 128, 0.0f, "64x64_4w beta=0"},
            {45, 53, 128, 1.0f, "64x64_4w beta=1"},
            {140, 195, 256, 0.0f, "128x128_bk64 beta=0"},
            {140, 195, 256, 1.0f, "128x128_bk64 beta=1"},
        };
        for (int is_bf16 = 0; is_bf16 < 2; ++is_bf16) {
            for (const auto& s : shapes) {
                std::vector<uint16_t> hX(s.T * s.K), hW(s.N * s.K);
                std::vector<float> hY0(s.T * s.N), hY(s.T * s.N);
                std::vector<double> ref(s.T * s.N, 0.0);
                for (auto& v : hX) v = is_bf16 ? f32_to_bf16_bits(fdist(rng)) : f32_to_f16_bits(fdist(rng));
                for (auto& v : hW) v = is_bf16 ? f32_to_bf16_bits(fdist(rng)) : f32_to_f16_bits(fdist(rng));
                for (auto& v : hY0) v = fdist(rng);

                for (int t = 0; t < s.T; ++t) {
                    for (int n = 0; n < s.N; ++n) {
                        double acc = (s.beta == 0.0f) ? 0.0 : static_cast<double>(hY0[t * s.N + n]);
                        for (int k = 0; k < s.K; ++k) {
                            const double xv = is_bf16 ? bf16_bits_to_f32(hX[t * s.K + k]) : f16_bits_to_f32(hX[t * s.K + k]);
                            const double wv = is_bf16 ? bf16_bits_to_f32(hW[n * s.K + k]) : f16_bits_to_f32(hW[n * s.K + k]);
                            acc += xv * wv;
                        }
                        ref[t * s.N + n] = acc;
                    }
                }
                uint16_t *dX = nullptr, *dW = nullptr;
                float* dY = nullptr;
                HIP_CHECK(hipMalloc(&dX, hX.size() * 2));
                HIP_CHECK(hipMalloc(&dW, hW.size() * 2));
                HIP_CHECK(hipMalloc(&dY, hY.size() * 4));
                HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size() * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(dY, hY0.data(), hY0.size() * 4, hipMemcpyHostToDevice));

                bool ok = is_bf16
                    ? hip_unsloth_wmma_gemm_bf16(dX, s.K, dW, s.K, dY, s.T, s.N, s.K, s.N, s.beta, nullptr)
                    : hip_unsloth_wmma_gemm_f16(dX, s.K, dW, s.K, dY, s.T, s.N, s.K, s.N, s.beta, nullptr);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(hY.data(), dY, hY.size() * 4, hipMemcpyDeviceToHost));
                HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dY));

                ErrStats err = compute_err(hY.data(), ref.data(), ref.size());
                const bool pass = ok && err.rel_l2 < 1e-4;
                std::printf("[TEST 2] WMMA %s %-20s (%3dx%3dx%3d): %s (max_abs=%.2e, rel_l2=%.2e)\n",
                            is_bf16 ? "BF16" : "FP16", s.tag, s.T, s.N, s.K,
                            pass ? "PASS" : "FAIL", err.max_abs, err.rel_l2);
                if (!pass) ++failures;
            }
        }
    }

    // -----------------------------------------------------------------------
    // TEST 3: Wave32 INT8 WMMA Q8_0 x Q8_1 Verify GEMM (T = 8, 12, 16)
    // -----------------------------------------------------------------------
    if (hip_wmma_device_available()) {
        const int K = 512, N = 140;
        const int cols_list[] = {8, 12, 16};
        std::vector<uint8_t> hW(N * hip_quant_row_bytes(8, K));
        fill_random_quant_rows(8, hW.data(), N, K, rng);
        void* dW = nullptr;
        HIP_CHECK(hipMalloc(&dW, hW.size()));
        HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));

        for (int nc : cols_list) {
            std::vector<float> hXf(nc * K);
            for (auto& v : hXf) v = fdist(rng);
            std::vector<block_q8_1> hXq(nc * (K / 32));
            host_quantize_q8_1(hXf.data(), hXq.data(), nc, K);

            std::vector<double> ref(nc * N, 0.0);
            std::vector<double> wrow(K), xrow(K);
            for (int c = 0; c < nc; ++c) {
                host_dequant_q8_1(hXq.data() + c * (K / 32), K, xrow.data());
                for (int r = 0; r < N; ++r) {
                    host_dequant_row(8, hW.data() + r * hip_quant_row_bytes(8, K), K, wrow.data(), iq3_grid);
                    double s = 0.0;
                    for (int k = 0; k < K; ++k) s += wrow[k] * xrow[k];
                    ref[c * N + r] = s;
                }
            }
            void* dX = nullptr;
            float* dY = nullptr;
            HIP_CHECK(hipMalloc(&dX, hXq.size() * sizeof(block_q8_1)));
            HIP_CHECK(hipMalloc(&dY, nc * N * sizeof(float)));
            HIP_CHECK(hipMemcpy(dX, hXq.data(), hXq.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
            bool ok = hip_unsloth_wmma_q80_mmvq(dW, dX, dY, K, N, nc, nullptr);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> hY(nc * N);
            HIP_CHECK(hipMemcpy(hY.data(), dY, hY.size() * 4, hipMemcpyDeviceToHost));
            HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dY));

            ErrStats err = compute_err(hY.data(), ref.data(), ref.size());
            const bool pass = ok && err.rel_l2 < 1e-5;
            std::printf("[TEST 3] WMMA INT8 Q8_0 verify T=%2d (%dx%d): %s (max_abs=%.2e, rel_l2=%.2e)\n",
                        nc, N, K, pass ? "PASS" : "FAIL", err.max_abs, err.rel_l2);
            if (!pass) ++failures;
        }
        HIP_CHECK(hipFree(dW));
    }

    // -----------------------------------------------------------------------
    // TEST 4: Wave32 DP4A MMVQ & Fused Dequant-to-LDS WMMA Prefill (ALL 7 Unsloth quants)
    // -----------------------------------------------------------------------
    const int all_types[] = {12, 13, 14, 8, 23, 18, 40};
    for (int ty : all_types) {
        const int K = 512, N = 131, NC = 5, T_pf = 73;
        const float gscale = (ty == 40) ? 0.37f : 1.0f;
        const std::size_t rb = hip_quant_row_bytes(ty, K);
        std::vector<uint8_t> hW(N * rb);
        fill_random_quant_rows(ty, hW.data(), N, K, rng);

        // Pre-dequantize weight matrix in FP64
        std::vector<double> W64(N * K);
        for (int r = 0; r < N; ++r) host_dequant_row(ty, hW.data() + r * rb, K, W64.data() + r * K, iq3_grid);

        void* dW = nullptr;
        HIP_CHECK(hipMalloc(&dW, hW.size()));
        HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));

        // 4A. DP4A MMVQ (NC=5)
        {
            std::vector<float> hXf(NC * K);
            for (auto& v : hXf) v = fdist(rng);
            std::vector<block_q8_1> hXq(NC * (K / 32));
            host_quantize_q8_1(hXf.data(), hXq.data(), NC, K);
            std::vector<double> ref(NC * N, 0.0), xrow(K);
            for (int c = 0; c < NC; ++c) {
                host_dequant_q8_1(hXq.data() + c * (K / 32), K, xrow.data());
                for (int r = 0; r < N; ++r) {
                    double s = 0.0;
                    for (int k = 0; k < K; ++k) s += W64[r * K + k] * xrow[k];
                    ref[c * N + r] = s * gscale;
                }
            }
            void* dX = nullptr;
            float* dY = nullptr;
            HIP_CHECK(hipMalloc(&dX, hXq.size() * sizeof(block_q8_1)));
            HIP_CHECK(hipMalloc(&dY, NC * N * sizeof(float)));
            HIP_CHECK(hipMemcpy(dX, hXq.data(), hXq.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
            bool ok = hip_unsloth_dp4a_mmvq(ty, dW, dX, dY, K, N, NC, gscale, nullptr);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> hY(NC * N);
            HIP_CHECK(hipMemcpy(hY.data(), dY, hY.size() * 4, hipMemcpyDeviceToHost));
            HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dY));

            ErrStats err = compute_err(hY.data(), ref.data(), ref.size());
            // IQ3_XXS uses integer truncating division `(ls*sumi + sumi/2)/2`, so tolerance is 3e-3; others < 5e-4
            const double tol = (ty == 18) ? 5e-3 : 5e-4;
            const bool pass = ok && err.rel_l2 < tol;
            std::printf("[TEST 4A] DP4A MMVQ     %-7s (T=%d, %dx%d): %s (max_abs=%.2e, rel_l2=%.2e)\n",
                        type_name(ty), NC, N, K, pass ? "PASS" : "FAIL", err.max_abs, err.rel_l2);
            if (!pass) ++failures;
        }

        // 4B. Fused Dequant-to-LDS FP16 + Wave32 WMMA Prefill
        if (hip_wmma_device_available()) {
            std::vector<uint16_t> hX16(T_pf * K);
            for (auto& v : hX16) v = f32_to_f16_bits(fdist(rng));
            std::vector<double> ref(T_pf * N, 0.0);
            for (int t = 0; t < T_pf; ++t) {
                for (int r = 0; r < N; ++r) {
                    double s = 0.0;
                    for (int k = 0; k < K; ++k) s += static_cast<double>(f16_bits_to_f32(hX16[t * K + k])) * W64[r * K + k];
                    ref[t * N + r] = s * gscale;
                }
            }
            uint16_t* dX = nullptr;
            float* dY = nullptr;
            HIP_CHECK(hipMalloc(&dX, hX16.size() * 2));
            HIP_CHECK(hipMalloc(&dY, T_pf * N * 4));
            HIP_CHECK(hipMemcpy(dX, hX16.data(), hX16.size() * 2, hipMemcpyHostToDevice));
            bool ok = hip_unsloth_wmma_dequant_prefill(ty, dW, dX, K, dY, T_pf, N, K, N, gscale, nullptr);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> hY(T_pf * N);
            HIP_CHECK(hipMemcpy(hY.data(), dY, hY.size() * 4, hipMemcpyDeviceToHost));
            HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dY));

            ErrStats err = compute_err(hY.data(), ref.data(), ref.size());
            const bool pass = ok && err.rel_l2 < 2e-3;
            std::printf("[TEST 4B] WMMA DeqPref  %-7s (T=%d, %dx%d): %s (max_abs=%.2e, rel_l2=%.2e)\n",
                        type_name(ty), T_pf, N, K, pass ? "PASS" : "FAIL", err.max_abs, err.rel_l2);
            if (!pass) ++failures;
        }
        HIP_CHECK(hipFree(dW));
    }

    // -----------------------------------------------------------------------
    // TEST 5: Fused EXP_MODE=8 (gate+up+SwiGLU+q8_1) + LDS Down (ALL 7 types + NVFP4 tail + had_block guard)
    // -----------------------------------------------------------------------
    for (int ty : all_types) {
        const int64_t n_embd = 512;
        const int64_t n_ff = (ty == 40 || ty == 8) ? 640 : 512;  // 640 exercises use_sub8 (n_items=20 for NVFP4)!
        const HipExpertLayout L = hip_make_expert_layout(ty, ty, n_embd, n_ff, 0);

        // Verify had_block > 0 guard rejects fused EXP_MODE=8
        {
            HipExpertLayout L_had = hip_make_expert_layout(ty, ty, n_embd, n_ff, 128);
            int dummy = 0;
            bool rejected = !hip_unsloth_exp8_gu_fused(L_had, reinterpret_cast<const unsigned long long*>(&dummy),
                                                       &dummy, &dummy, &dummy, 1, &dummy, &dummy, nullptr);
            if (!rejected) {
                std::printf("[TEST 5] had_block guard FAILED for %s!\n", type_name(ty));
                ++failures;
            }
        }

        const int n_grp = 2;
        const int n_ent = 5;  // group 0 has 3 entries, group 1 has 2 entries
        std::vector<std::vector<uint8_t>> h_blobs(n_grp, std::vector<uint8_t>(L.bytes));
        std::vector<void*> d_blobs(n_grp, nullptr);
        std::vector<unsigned long long> h_ptrs(n_grp);
        const float tails[2][4] = {{0.63f, 1.37f, 0.29f, 0.0f}, {1.15f, 0.72f, 0.44f, 0.0f}};

        for (int g = 0; g < n_grp; ++g) {
            fill_random_quant_rows(ty, h_blobs[g].data(), static_cast<int>(n_ff), static_cast<int>(n_embd), rng);
            fill_random_quant_rows(ty, h_blobs[g].data() + L.up_off, static_cast<int>(n_ff), static_cast<int>(n_embd), rng);
            fill_random_quant_rows(ty, h_blobs[g].data() + L.down_off, static_cast<int>(n_embd), static_cast<int>(n_ff), rng);
            if (L.tail_off != 0) std::memcpy(h_blobs[g].data() + L.tail_off, tails[g], 16);
            HIP_CHECK(hipMalloc(&d_blobs[g], L.bytes));
            HIP_CHECK(hipMemcpy(d_blobs[g], h_blobs[g].data(), L.bytes, hipMemcpyHostToDevice));
            h_ptrs[g] = reinterpret_cast<unsigned long long>(d_blobs[g]);
        }

        const int32_t h_gs[3] = {0, 3, 5};
        const int32_t h_ng = n_grp;
        const int32_t h_tok[5] = {0, 2, 1, 0, 2};
        const int32_t h_dst[5] = {4, 1, 3, 0, 2};
        const int n_tok = 3;

        std::vector<float> hXf(n_tok * n_embd);
        for (auto& v : hXf) v = fdist(rng);
        std::vector<block_q8_1> hXq(n_tok * (n_embd / 32));
        host_quantize_q8_1(hXf.data(), hXq.data(), n_tok, static_cast<int>(n_embd));

        // Exact host reference: dequant xq -> gate/up -> SwiGLU -> quantize hq -> dequant hq -> down
        std::vector<double> ref_out(n_ent * n_embd, 0.0);
        std::vector<double> wrow(std::max(n_embd, n_ff)), xrow(n_embd), hrow_q(n_ff);
        for (int g = 0; g < n_grp; ++g) {
            const float sg = (ty == 40) ? tails[g][0] : 1.0f;
            const float su = (ty == 40) ? tails[g][1] : 1.0f;
            const float sd = (ty == 40) ? tails[g][2] : 1.0f;
            for (int e = h_gs[g]; e < h_gs[g + 1]; ++e) {
                host_dequant_q8_1(hXq.data() + h_tok[e] * (n_embd / 32), static_cast<int>(n_embd), xrow.data());
                std::vector<float> hf(n_ff);
                for (int r = 0; r < n_ff; ++r) {
                    host_dequant_row(ty, h_blobs[g].data() + r * L.gu_row, static_cast<int>(n_embd), wrow.data(), iq3_grid);
                    double dg = 0.0;
                    for (int k = 0; k < n_embd; ++k) dg += wrow[k] * xrow[k];
                    host_dequant_row(ty, h_blobs[g].data() + L.up_off + r * L.gu_row, static_cast<int>(n_embd), wrow.data(), iq3_grid);
                    double du = 0.0;
                    for (int k = 0; k < n_embd; ++k) du += wrow[k] * xrow[k];
                    const float gg = static_cast<float>(dg) * sg;
                    const float uu = static_cast<float>(du) * su;
                    hf[r] = (gg / (1.0f + std::exp(-gg))) * uu;
                }
                std::vector<block_q8_1> hq_e(n_ff / 32);
                host_quantize_q8_1(hf.data(), hq_e.data(), 1, static_cast<int>(n_ff));
                host_dequant_q8_1(hq_e.data(), static_cast<int>(n_ff), hrow_q.data());
                for (int r = 0; r < n_embd; ++r) {
                    host_dequant_row(ty, h_blobs[g].data() + L.down_off + r * L.d_row, static_cast<int>(n_ff), wrow.data(), iq3_grid);
                    double acc = 0.0;
                    for (int k = 0; k < n_ff; ++k) acc += wrow[k] * hrow_q[k];
                    ref_out[h_dst[e] * n_embd + r] = acc * sd;
                }
            }
        }

        unsigned long long* d_ptrs = nullptr;
        int32_t *d_gs = nullptr, *d_ng = nullptr, *d_tok = nullptr, *d_dst = nullptr;
        void *d_xq = nullptr, *d_hq = nullptr;
        float* d_out = nullptr;
        HIP_CHECK(hipMalloc(&d_ptrs, n_grp * sizeof(unsigned long long)));
        HIP_CHECK(hipMalloc(&d_gs, (n_grp + 1) * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&d_ng, sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&d_tok, n_ent * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&d_dst, n_ent * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&d_xq, hXq.size() * sizeof(block_q8_1)));
        HIP_CHECK(hipMalloc(&d_hq, n_ent * (n_ff / 32) * sizeof(block_q8_1)));
        HIP_CHECK(hipMalloc(&d_out, n_ent * n_embd * sizeof(float)));

        HIP_CHECK(hipMemcpy(d_ptrs, h_ptrs.data(), n_grp * sizeof(unsigned long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_gs, h_gs, (n_grp + 1) * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_ng, &h_ng, sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_tok, h_tok, n_ent * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_dst, h_dst, n_ent * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_xq, hXq.data(), hXq.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));

        bool ok1 = hip_unsloth_exp8_gu_fused(L, d_ptrs, d_gs, d_ng, d_tok, n_grp, d_xq, d_hq, nullptr);
        bool ok2 = hip_unsloth_down_lds(L, d_ptrs, d_gs, d_ng, d_dst, n_grp, d_hq, d_out, nullptr);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> h_out(n_ent * n_embd);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * 4, hipMemcpyDeviceToHost));

        for (void* p : d_blobs) HIP_CHECK(hipFree(p));
        HIP_CHECK(hipFree(d_ptrs)); HIP_CHECK(hipFree(d_gs)); HIP_CHECK(hipFree(d_ng));
        HIP_CHECK(hipFree(d_tok)); HIP_CHECK(hipFree(d_dst)); HIP_CHECK(hipFree(d_xq));
        HIP_CHECK(hipFree(d_hq)); HIP_CHECK(hipFree(d_out));

        ErrStats err = compute_err(h_out.data(), ref_out.data(), ref_out.size());
        const double tol = (ty == 18) ? 2e-2 : 2e-3;
        const bool pass = ok1 && ok2 && err.rel_l2 < tol;
        std::printf("[TEST 5] EXP8 Fused+Down %-7s (%lldx%lld): %s (max_abs=%.2e, rel_l2=%.2e)\n",
                    type_name(ty), (long long) n_embd, (long long) n_ff,
                    pass ? "PASS" : "FAIL", err.max_abs, err.rel_l2);
        if (!pass) ++failures;
    }

    if (failures != 0) {
        std::printf("\nFAILED: %d parity checks failed!\n", failures);
        return 1;
    }
    std::printf("\nALL PARITY CHECKS PASSED (0 failures)!\n\n");

    // -----------------------------------------------------------------------
    // LIVE MICRO-BENCHMARKS
    // -----------------------------------------------------------------------
    std::printf("=== Live Micro-Benchmarks on %s (%s) ===\n", prop.name, prop.gcnArchName);

    // Benchmark A: RDNA4/3 Wave32 WMMA FP16 & BF16 Prefill GEMM
    if (hip_wmma_device_available()) {
        struct BShape { int T, N, K; };
        const BShape bshapes[] = {{2048, 2560, 2560}, {4096, 4096, 4096}};
        for (const auto& s : bshapes) {
            uint16_t *dX = nullptr, *dW = nullptr;
            float* dY = nullptr;
            HIP_CHECK(hipMalloc(&dX, static_cast<std::size_t>(s.T) * s.K * 2));
            HIP_CHECK(hipMalloc(&dW, static_cast<std::size_t>(s.N) * s.K * 2));
            HIP_CHECK(hipMalloc(&dY, static_cast<std::size_t>(s.T) * s.N * 4));
            HIP_CHECK(hipMemset(dX, 0x3c, static_cast<std::size_t>(s.T) * s.K * 2));
            HIP_CHECK(hipMemset(dW, 0x3c, static_cast<std::size_t>(s.N) * s.K * 2));

            const double flops = 2.0 * s.T * s.N * s.K;
            const float ms_f16 = bench_ms([&] {
                hip_unsloth_wmma_gemm_f16(dX, s.K, dW, s.K, dY, s.T, s.N, s.K, s.N, 0.0f, nullptr);
            });
            const float ms_bf16 = bench_ms([&] {
                hip_unsloth_wmma_gemm_bf16(dX, s.K, dW, s.K, dY, s.T, s.N, s.K, s.N, 0.0f, nullptr);
            });
            std::printf("  WMMA FP16 (%dx%dx%d): %7.3f ms | %6.2f TFLOPS\n",
                        s.T, s.N, s.K, ms_f16, (flops / (ms_f16 * 1e-3)) / 1e12);
            std::printf("  WMMA BF16 (%dx%dx%d): %7.3f ms | %6.2f TFLOPS\n",
                        s.T, s.N, s.K, ms_bf16, (flops / (ms_bf16 * 1e-3)) / 1e12);
            HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dY));
        }
    }

    // Benchmark B: Wave32 DP4A Decode MMVQ (N=2560, K=2560, T=1 & T=4) + INT8 WMMA Q8_0 (T=8, 16)
    std::printf("\n  --- Wave32 DP4A Decode MMVQ (N=2560, K=2560) ---\n");
    {
        const int N = 2560, K = 2560;
        void* dX = nullptr;
        float* dY = nullptr;
        HIP_CHECK(hipMalloc(&dX, 16 * (K / 32) * sizeof(block_q8_1)));
        HIP_CHECK(hipMalloc(&dY, 16 * N * sizeof(float)));
        HIP_CHECK(hipMemset(dX, 1, 16 * (K / 32) * sizeof(block_q8_1)));
        for (int ty : all_types) {
            const std::size_t wbytes = static_cast<std::size_t>(N) * hip_quant_row_bytes(ty, K);
            void* dW = nullptr;
            HIP_CHECK(hipMalloc(&dW, wbytes));
            HIP_CHECK(hipMemset(dW, 0x22, wbytes));
            const float ms1 = bench_ms([&] { hip_unsloth_dp4a_mmvq(ty, dW, dX, dY, K, N, 1, 1.0f, nullptr); });
            const float ms4 = bench_ms([&] { hip_unsloth_dp4a_mmvq(ty, dW, dX, dY, K, N, 4, 1.0f, nullptr); });
            const double bw1 = (static_cast<double>(wbytes) / (ms1 * 1e-3)) / 1e9;
            const double bw4 = (static_cast<double>(wbytes) / (ms4 * 1e-3)) / 1e9;
            std::printf("  DP4A %-7s | T=1: %6.1f us (%5.1f GB/s) | T=4: %6.1f us (%5.1f GB/s eff)\n",
                        type_name(ty), ms1 * 1e3f, bw1, ms4 * 1e3f, bw4);
            if (ty == 8 && hip_wmma_device_available()) {
                const float ms8 = bench_ms([&] { hip_unsloth_wmma_q80_mmvq(dW, dX, dY, K, N, 8, nullptr); });
                const float ms16 = bench_ms([&] { hip_unsloth_wmma_q80_mmvq(dW, dX, dY, K, N, 16, nullptr); });
                const double tops16 = (2.0 * 16.0 * N * K / (ms16 * 1e-3)) / 1e12;
                std::printf("  WMMA Q8_0 I8 | T=8: %6.1f us | T=16: %6.1f us (%.2f TOPS)\n",
                            ms8 * 1e3f, ms16 * 1e3f, tops16);
            }
            HIP_CHECK(hipFree(dW));
        }
        HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dY));
    }

    // Benchmark C: Fused Dequant-to-LDS FP16 + Wave32 WMMA Prefill (T=512, N=2560, K=2560)
    if (hip_wmma_device_available()) {
        std::printf("\n  --- Fused Dequant-to-LDS + Wave32 WMMA Prefill (T=512, N=2560, K=2560) ---\n");
        const int T = 512, N = 2560, K = 2560;
        uint16_t* dX = nullptr;
        float* dY = nullptr;
        HIP_CHECK(hipMalloc(&dX, static_cast<std::size_t>(T) * K * 2));
        HIP_CHECK(hipMalloc(&dY, static_cast<std::size_t>(T) * N * 4));
        HIP_CHECK(hipMemset(dX, 0x3c, static_cast<std::size_t>(T) * K * 2));
        const double flops = 2.0 * T * N * K;
        for (int ty : all_types) {
            const std::size_t wbytes = static_cast<std::size_t>(N) * hip_quant_row_bytes(ty, K);
            void* dW = nullptr;
            HIP_CHECK(hipMalloc(&dW, wbytes));
            HIP_CHECK(hipMemset(dW, 0x22, wbytes));
            const float ms = bench_ms([&] {
                hip_unsloth_wmma_dequant_prefill(ty, dW, dX, K, dY, T, N, K, N, 1.0f, nullptr);
            });
            std::printf("  WMMA DeqPref %-7s | %6.3f ms | %6.2f TFLOPS\n",
                        type_name(ty), ms, (flops / (ms * 1e-3)) / 1e12);
            HIP_CHECK(hipFree(dW));
        }
        HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dY));
    }

    // Benchmark D: Fused EXP_MODE=8 MoE Expert Pass (10 active experts, T=1 & T=4 tokens, 2560x512/640)
    std::printf("\n  --- Fused EXP_MODE=8 MoE Expert Pass (10 experts x 1 & 4 tokens, n_embd=2560) ---\n");
    {
        const int n_grp = 10;
        const int64_t n_embd = 2560;
        for (int ty : all_types) {
            const int64_t n_ff = (ty == 40 || ty == 8) ? 640 : 512;
            const HipExpertLayout L = hip_make_expert_layout(ty, ty, n_embd, n_ff, 0);
            std::vector<void*> d_blobs(n_grp, nullptr);
            std::vector<unsigned long long> h_ptrs(n_grp);
            for (int g = 0; g < n_grp; ++g) {
                HIP_CHECK(hipMalloc(&d_blobs[g], L.bytes));
                HIP_CHECK(hipMemset(d_blobs[g], 0x22, L.bytes));
                h_ptrs[g] = reinterpret_cast<unsigned long long>(d_blobs[g]);
            }
            const int max_ent = n_grp * 4;
            unsigned long long* d_ptrs = nullptr;
            int32_t *d_gs1 = nullptr, *d_gs4 = nullptr, *d_ng = nullptr, *d_tok = nullptr, *d_dst = nullptr;
            void *d_xq = nullptr, *d_hq = nullptr;
            float* d_out = nullptr;
            HIP_CHECK(hipMalloc(&d_ptrs, n_grp * sizeof(unsigned long long)));
            HIP_CHECK(hipMalloc(&d_gs1, (n_grp + 1) * sizeof(int32_t)));
            HIP_CHECK(hipMalloc(&d_gs4, (n_grp + 1) * sizeof(int32_t)));
            HIP_CHECK(hipMalloc(&d_ng, sizeof(int32_t)));
            HIP_CHECK(hipMalloc(&d_tok, max_ent * sizeof(int32_t)));
            HIP_CHECK(hipMalloc(&d_dst, max_ent * sizeof(int32_t)));
            HIP_CHECK(hipMalloc(&d_xq, 4 * (n_embd / 32) * sizeof(block_q8_1)));
            HIP_CHECK(hipMalloc(&d_hq, max_ent * (n_ff / 32) * sizeof(block_q8_1)));
            HIP_CHECK(hipMalloc(&d_out, max_ent * n_embd * sizeof(float)));

            std::vector<int32_t> gs1(n_grp + 1), gs4(n_grp + 1), tok(max_ent), dst(max_ent);
            for (int g = 0; g <= n_grp; ++g) { gs1[g] = g; gs4[g] = g * 4; }
            for (int i = 0; i < max_ent; ++i) { tok[i] = i & 3; dst[i] = i; }
            const int32_t ng = n_grp;
            HIP_CHECK(hipMemcpy(d_ptrs, h_ptrs.data(), n_grp * sizeof(unsigned long long), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_gs1, gs1.data(), gs1.size() * 4, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_gs4, gs4.data(), gs4.size() * 4, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_ng, &ng, 4, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tok, tok.data(), tok.size() * 4, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_dst, dst.data(), dst.size() * 4, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_xq, 1, 4 * (n_embd / 32) * sizeof(block_q8_1)));

            const float ms1 = bench_ms([&] {
                hip_unsloth_exp8_gu_fused(L, d_ptrs, d_gs1, d_ng, d_tok, n_grp, d_xq, d_hq, nullptr);
                hip_unsloth_down_lds(L, d_ptrs, d_gs1, d_ng, d_dst, n_grp, d_hq, d_out, nullptr);
            });
            const float ms4 = bench_ms([&] {
                hip_unsloth_exp8_gu_fused(L, d_ptrs, d_gs4, d_ng, d_tok, n_grp, d_xq, d_hq, nullptr);
                hip_unsloth_down_lds(L, d_ptrs, d_gs4, d_ng, d_dst, n_grp, d_hq, d_out, nullptr);
            });
            const double tot_bytes = static_cast<double>(n_grp) * static_cast<double>(L.bytes);
            std::printf("  EXP8 %-7s (2560x%lld) | 10x1: %6.1f us (%5.1f GB/s) | 10x4: %6.1f us (%5.1f GB/s eff)\n",
                        type_name(ty), (long long) n_ff, ms1 * 1e3f, (tot_bytes / (ms1 * 1e-3)) / 1e9,
                        ms4 * 1e3f, (tot_bytes / (ms4 * 1e-3)) / 1e9);

            for (void* p : d_blobs) HIP_CHECK(hipFree(p));
            HIP_CHECK(hipFree(d_ptrs)); HIP_CHECK(hipFree(d_gs1)); HIP_CHECK(hipFree(d_gs4));
            HIP_CHECK(hipFree(d_ng)); HIP_CHECK(hipFree(d_tok)); HIP_CHECK(hipFree(d_dst));
            HIP_CHECK(hipFree(d_xq)); HIP_CHECK(hipFree(d_hq)); HIP_CHECK(hipFree(d_out));
        }
    }

    return 0;
}
