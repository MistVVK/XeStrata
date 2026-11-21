// ink_metal_codecs.metal - High-speed Apple Silicon Metal kernels for Inkling-2-Mini (.strata v1.1)
// Optimizations:
//   - Zero-stack-spill register-only S_K decoders (float4 pairs, 2-row/SG linear, 8-tok MoE micro-batching)
//   - Vectorized 64-bit uint2 SF4 nibble loads & 32-bit uint MX8 loads
//   - simdgroup_matrix 8x8 FP16/BF16 GEMM for M > 8 (prefill & c=32/64 batching)
//   - GPU-side MoE binning (ink_moe_plan -> indirect dispatch grid, zero host sync)
//   - Fused w13 + SITU(beta=4.0, lb=25.0) + w2 scatter-add across Top-8/128 experts
// 100% MIT clean-room implementation.
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 1. Lossless NVFP4 (E2M1) & FP8 (E4M3FN) & UE8M0 Register-LUT Decoders
// ============================================================================

constant float kE2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e4m3_to_f32(uint c) {
    const uint mag = c & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    const uint exp = mag >> 3;
    const uint man = mag & 7u;
    float v = (exp == 0u) ? (float(man) * (1.0f / 512.0f))
                          : as_type<float>(((exp + 120u) << 23) | (man << 20));
    return (c & 0x80u) ? -v : v;
}

static inline float ue8m0_to_f32(uint e) {
    if (e == 0u) return 5.877471754111438e-39f;
    if (e == 255u) return 0.0f;
    return as_type<float>(e << 23);
}

static inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

static inline float4 bf16x4_to_f32(ushort4 b) {
    return float4(as_type<float>(uint(b.x) << 16), as_type<float>(uint(b.y) << 16),
                  as_type<float>(uint(b.z) << 16), as_type<float>(uint(b.w) << 16));
}

// Exact SITU(g, u, beta=4.0, lb=25.0)
static inline float situ_act(float g, float u) {
    const float sig = 1.0f / (1.0f + exp(-g));
    const float g_act = 4.0f * tanh(clamp(g * 0.25f, -15.0f, 15.0f)) * sig;
    const float u_act = 25.0f * tanh(clamp(u * 0.04f, -15.0f, 15.0f));
    return g_act * u_act;
}

// ============================================================================
// 2. Register-Only S_K Super-Block Decoders (ZERO thread-local array spills!)
// ============================================================================

static inline constexpr int sk_sb_bytes(int F) {
    return F == 2 ? 70 : F == 3 ? 102 : F == 4 ? 134 : F == 5 ? 166 : F == 6 ? 198 : F == 8 ? 262 : 0;
}

static inline constexpr float sk_half(int F) {
    return F == 2 ? 1.5f : F == 3 ? 3.5f : F == 4 ? 7.5f : F == 5 ? 15.5f : F == 6 ? 31.5f : 0.0f;
}

static inline ushort ld16(device const uchar* p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

template <int F>
static inline void sk_w8(device const uchar* pl, int j, thread float4& w0, thread float4& w1) {
    constexpr float H = sk_half(F);
    if (F == 2) {
        const uint b0 = pl[j >> 2], b1 = pl[(j >> 2) + 1];
        w0 = float4(float(b0 & 3u), float((b0 >> 2) & 3u), float((b0 >> 4) & 3u), float((b0 >> 6) & 3u)) - H;
        w1 = float4(float(b1 & 3u), float((b1 >> 2) & 3u), float((b1 >> 4) & 3u), float((b1 >> 6) & 3u)) - H;
    } else if (F == 3) {
        const uint b0 = pl[j >> 2], b1 = pl[(j >> 2) + 1];
        const uint h = pl[64 + (j >> 3)];
        w0 = float4(float((b0 & 3u) | ((h & 1u) << 2)),
                    float(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2)),
                    float(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2)),
                    float(((b0 >> 6) & 3u) | (((h >> 3) & 1u) << 2))) - H;
        w1 = float4(float((b1 & 3u) | (((h >> 4) & 1u) << 2)),
                    float(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2)),
                    float(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2)),
                    float(((b1 >> 6) & 3u) | (((h >> 7) & 1u) << 2))) - H;
    } else if (F == 4) {
        const uint o = uint(j >> 1);
        const uint b0 = pl[o], b1 = pl[o + 1], b2 = pl[o + 2], b3 = pl[o + 3];
        w0 = float4(float(b0 & 15u), float(b0 >> 4), float(b1 & 15u), float(b1 >> 4)) - H;
        w1 = float4(float(b2 & 15u), float(b2 >> 4), float(b3 & 15u), float(b3 >> 4)) - H;
    } else if (F == 5) {
        const uint o = uint(j >> 1);
        const uint b0 = pl[o], b1 = pl[o + 1], b2 = pl[o + 2], b3 = pl[o + 3];
        const uint h = pl[128 + (j >> 3)];
        w0 = float4(float((b0 & 15u) | ((h & 1u) << 4)),
                    float((b0 >> 4) | (((h >> 1) & 1u) << 4)),
                    float((b1 & 15u) | (((h >> 2) & 1u) << 4)),
                    float((b1 >> 4) | (((h >> 3) & 1u) << 4))) - H;
        w1 = float4(float((b2 & 15u) | (((h >> 4) & 1u) << 4)),
                    float((b2 >> 4) | (((h >> 5) & 1u) << 4)),
                    float((b3 & 15u) | (((h >> 6) & 1u) << 4)),
                    float((b3 >> 4) | (((h >> 7) & 1u) << 4))) - H;
    } else if (F == 6) {
        const uint o = uint(j >> 1);
        const uint b0 = pl[o], b1 = pl[o + 1], b2 = pl[o + 2], b3 = pl[o + 3];
        const uint h0 = pl[128 + (j >> 2)], h1 = pl[128 + (j >> 2) + 1];
        w0 = float4(float((b0 & 15u) | ((h0 & 3u) << 4)),
                    float((b0 >> 4) | (((h0 >> 2) & 3u) << 4)),
                    float((b1 & 15u) | (((h0 >> 4) & 3u) << 4)),
                    float((b1 >> 4) | (((h0 >> 6) & 3u) << 4))) - H;
        w1 = float4(float((b2 & 15u) | ((h1 & 3u) << 4)),
                    float((b2 >> 4) | (((h1 >> 2) & 3u) << 4)),
                    float((b3 & 15u) | (((h1 >> 4) & 3u) << 4)),
                    float((b3 >> 4) | (((h1 >> 6) & 3u) << 4))) - H;
    } else {
        device const char* qp = (device const char*)(pl + j);
        w0 = float4(float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3]));
        w1 = float4(float(qp[4]), float(qp[5]), float(qp[6]), float(qp[7]));
    }
}

// ============================================================================
// 3. Dense Trunk & DSpark Linear Kernels (BF16, Q8_ROW, F32, S_K)
// ============================================================================

// --- BF16 GEMV for M=1..8 (8 SGs = 256 threads/TG, 4 rows/SG = 32 rows/TG when M==1, 2 rows/SG for M=2..8)
kernel void ink_linear_bf16_mv(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool has1 = (row0 + 1) < N;
    const int K4 = K >> 2;
    device const ushort4* w0p = (device const ushort4*)(W + (size_t)row0 * K);
    device const ushort4* w1p = (device const ushort4*)(W + (size_t)(row0 + (has1 ? 1 : 0)) * K);

    if (M == 1) {
        float a0 = 0.0f, a1 = 0.0f;
        device const float4* xp = (device const float4*)X;
        for (int k = int(lane); k < K4; k += 32) {
            const float4 xv = xp[k];
            a0 += dot(bf16x4_to_f32(w0p[k]), xv);
            if (has1) a1 += dot(bf16x4_to_f32(w1p[k]), xv);
        }
        const float v0 = simd_sum(a0);
        if (lane == 0) Y[row0] = v0;
        if (has1) {
            const float v1 = simd_sum(a1);
            if (lane == 0) Y[row0 + 1] = v1;
        }
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 w0 = bf16x4_to_f32(w0p[k]);
            const float4 w1 = has1 ? bf16x4_to_f32(w1p[k]) : float4(0.f);
            for (int i = 0; i < mm; ++i) {
                const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
                a0[i] += dot(w0, xv);
                if (has1) a1[i] += dot(w1, xv);
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]);
            if (lane == 0) Y[(size_t)(m0 + i) * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
            }
        }
    }
}

// --- BF16 simdgroup_matrix 8x8 GEMM for M > 8 (BM=32, BN=64, BK=32, 8 SGs = 256 threads)
kernel void ink_linear_bf16_mm(
    device const ushort* W [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    constant int& K        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 32];
    threadgroup half sB[64 * 32];
    const int m0 = int(tgp.y) * 32;
    const int n0 = int(tgp.x) * 64;
    if (m0 >= M || n0 >= N) return;

    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (uint idx = tid * 4; idx < 32 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gm = m0 + r;
            if (gm < M && k0 + c + 3 < K) {
                *((threadgroup half4*)(sA + r * 32 + c)) = half4(*((device const float4*)(X + (size_t)gm * K + k0 + c)));
            } else {
                for (int u = 0; u < 4; ++u)
                    sA[r * 32 + c + u] = (gm < M && k0 + c + u < K) ? half(X[(size_t)gm * K + k0 + c + u]) : half(0.0h);
            }
        }
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K) {
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(bf16x4_to_f32(*((device const ushort4*)(W + (size_t)gn * K + k0 + c))));
            } else {
                for (int u = 0; u < 4; ++u)
                    sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(bf16_to_f32(W[(size_t)gn * K + k0 + c + u])) : half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, sA + (sgm * 16 + 0) * 32 + kk * 8, 32);
            simdgroup_load(a1, sA + (sgm * 16 + 8) * 32 + kk * 8, 32);
            simdgroup_load(b0, sB + (sgn * 16 + 0) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (sgn * 16 + 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 2; ++i) {
        const int gm = m0 + sgm * 16 + i * 8 + int(fm);
        if (gm < M) {
            for (int j = 0; j < 2; ++j) {
                const int gn = n0 + sgn * 16 + j * 8 + int(fn);
                if (gn < N) Y[(size_t)gm * N + gn] = acc[i][j].thread_elements()[0];
                if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc[i][j].thread_elements()[1];
            }
        }
    }
}

// --- Q8_ROW GEMV for M=1..8 (vectorized char4, specialized M==1 fast path!)
kernel void ink_linear_q8row_mv(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool has1 = (row0 + 1) < N;
    device const float* scales = (device const float*)W;
    const float sc0 = scales[row0];
    const float sc1 = has1 ? scales[row0 + 1] : 0.0f;
    device const char4* q0p = (device const char4*)(W + (size_t)N * 4 + (size_t)row0 * K);
    device const char4* q1p = (device const char4*)(W + (size_t)N * 4 + (size_t)(row0 + (has1 ? 1 : 0)) * K);
    const int K4 = K >> 2;

    if (M == 1) {
        float a0 = 0.0f, a1 = 0.0f;
        device const float4* xp = (device const float4*)X;
        for (int k = int(lane); k < K4; k += 32) {
            const float4 xv = xp[k];
            a0 += dot(float4(q0p[k]), xv);
            if (has1) a1 += dot(float4(q1p[k]), xv);
        }
        const float v0 = simd_sum(a0) * sc0;
        if (lane == 0) Y[row0] = v0;
        if (has1) {
            const float v1 = simd_sum(a1) * sc1;
            if (lane == 0) Y[row0 + 1] = v1;
        }
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 w0 = float4(q0p[k]);
            const float4 w1 = has1 ? float4(q1p[k]) : float4(0.f);
            for (int i = 0; i < mm; ++i) {
                const float4 xv = ((device const float4*)(X + (size_t)(m0 + i) * K))[k];
                a0[i] += dot(w0, xv);
                if (has1) a1[i] += dot(w1, xv);
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]) * sc0;
            if (lane == 0) Y[(size_t)(m0 + i) * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]) * sc1;
                if (lane == 0) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
            }
        }
    }
}

// --- Q8_ROW simdgroup_matrix 8x8 GEMM for M > 8
kernel void ink_linear_q8row_mm(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 32];
    threadgroup half sB[64 * 32];
    const int m0 = int(tgp.y) * 32, n0 = int(tgp.x) * 64;
    if (m0 >= M || n0 >= N) return;
    device const float* scales = (device const float*)W;
    device const char* qw = (device const char*)(W + (size_t)N * 4);
    const int sgm = int(sg >> 2), sgn = int(sg & 3u);
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (uint idx = tid * 4; idx < 32 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gm = m0 + r;
            if (gm < M && k0 + c + 3 < K)
                *((threadgroup half4*)(sA + r * 32 + c)) = half4(*((device const float4*)(X + (size_t)gm * K + k0 + c)));
            else
                for (int u = 0; u < 4; ++u) sA[r * 32 + c + u] = (gm < M && k0 + c + u < K) ? half(X[(size_t)gm * K + k0 + c + u]) : half(0.0h);
        }
        for (uint idx = tid * 4; idx < 64 * 32; idx += 256 * 4) {
            const int r = int(idx >> 5), c = int(idx & 31u), gn = n0 + r;
            if (gn < N && k0 + c + 3 < K)
                *((threadgroup half4*)(sB + r * 32 + c)) = half4(float4(*((device const char4*)(qw + (size_t)gn * K + k0 + c))));
            else
                for (int u = 0; u < 4; ++u) sB[r * 32 + c + u] = (gn < N && k0 + c + u < K) ? half(float(qw[(size_t)gn * K + k0 + c + u])) : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, sA + (sgm * 16 + 0) * 32 + kk * 8, 32);
            simdgroup_load(a1, sA + (sgm * 16 + 8) * 32 + kk * 8, 32);
            simdgroup_load(b0, sB + (sgn * 16 + 0) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_load(b1, sB + (sgn * 16 + 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 2; ++i) {
        const int gm = m0 + sgm * 16 + i * 8 + int(fm);
        if (gm < M) {
            for (int j = 0; j < 2; ++j) {
                const int gn = n0 + sgn * 16 + j * 8 + int(fn);
                if (gn < N) Y[(size_t)gm * N + gn] = acc[i][j].thread_elements()[0] * scales[gn];
                if (gn + 1 < N) Y[(size_t)gm * N + gn + 1] = acc[i][j].thread_elements()[1] * scales[gn + 1];
            }
        }
    }
}

// --- F32 Linear
kernel void ink_linear_f32(
    device const float* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = int(tg) * 8 + int(sg);
    if (row >= N) return;
    device const float* wp = W + (size_t)row * K;
    const int K4 = K >> 2;
    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a[4] = {0, 0, 0, 0};
        for (int k = int(lane); k < K4; k += 32) {
            const float4 wv = ((device const float4*)wp)[k];
            for (int i = 0; i < mm; ++i) a[i] += dot(wv, ((device const float4*)(X + (size_t)(m0 + i) * K))[k]);
        }
        for (int k = K4 * 4 + int(lane); k < K; k += 32) {
            const float wv = wp[k];
            for (int i = 0; i < mm; ++i) a[i] += wv * X[(size_t)(m0 + i) * K + k];
        }
        for (int i = 0; i < mm; ++i) {
            const float v = simd_sum(a[i]);
            if (lane == 0) Y[(size_t)(m0 + i) * N + row] = v;
        }
    }
}

// --- FAST 2-Row-per-SG Register-Only S_K Linear (16 rows/TG! Huge speedup on S6_K unembed & S5_K DSpark!)
template <int F>
kernel void ink_linear_sk_mv(
    device const uchar* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& M       [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool has1 = (row0 + 1) < N;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t)nsb * SB;
    device const uchar* r0 = W + (size_t)row0 * rowb;
    device const uchar* r1 = has1 ? (r0 + rowb) : r0;

    if (M == 1) {
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int unit = int(lane); unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const float4* xp = (device const float4*)(X + sbi * 256 + j);
            const float4 x0 = xp[0], x1 = xp[1];
            {
                device const uchar* blk = r0 + (size_t)sbi * SB;
                const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                float4 w0, w1;
                sk_w8<F>(blk + 6, j, w0, w1);
                acc0 = fma(eff, dot(w0, x0) + dot(w1, x1), acc0);
            }
            if (has1) {
                device const uchar* blk = r1 + (size_t)sbi * SB;
                const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                float4 w0, w1;
                sk_w8<F>(blk + 6, j, w0, w1);
                acc1 = fma(eff, dot(w0, x0) + dot(w1, x1), acc1);
            }
        }
        const float v0 = simd_sum(acc0);
        if (lane == 0) Y[row0] = v0;
        if (has1) {
            const float v1 = simd_sum(acc1);
            if (lane == 0) Y[row0 + 1] = v1;
        }
        return;
    }

    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mm = min(4, M - m0);
        float a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        for (int unit = int(lane); unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const uchar* b0 = r0 + (size_t)sbi * SB;
            const float e0 = float(as_type<half>(ld16(b0))) * (float(b0[2 + g]) * (1.0f / 255.0f));
            float4 w00, w01;
            sk_w8<F>(b0 + 6, j, w00, w01);
            float e1 = 0.0f; float4 w10 = 0.0f, w11 = 0.0f;
            if (has1) {
                device const uchar* b1 = r1 + (size_t)sbi * SB;
                e1 = float(as_type<half>(ld16(b1))) * (float(b1[2 + g]) * (1.0f / 255.0f));
                sk_w8<F>(b1 + 6, j, w10, w11);
            }
            for (int i = 0; i < mm; ++i) {
                device const float4* xp = (device const float4*)(X + (size_t)(m0 + i) * K + sbi * 256 + j);
                const float4 x0 = xp[0], x1 = xp[1];
                a0[i] = fma(e0, dot(w00, x0) + dot(w01, x1), a0[i]);
                if (has1) a1[i] = fma(e1, dot(w10, x0) + dot(w11, x1), a1[i]);
            }
        }
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]);
            if (lane == 0) Y[(size_t)(m0 + i) * N + row0] = v0;
            if (has1) {
                const float v1 = simd_sum(a1[i]);
                if (lane == 0) Y[(size_t)(m0 + i) * N + row0 + 1] = v1;
            }
        }
    }
}

#define INK_LINSK(F) template [[host_name("ink_linear_sk_mv_" #F)]] kernel void ink_linear_sk_mv<F>(device const uchar*, device const float*, device float*, constant int&, constant int&, constant int&, uint, uint);
INK_LINSK(2) INK_LINSK(3) INK_LINSK(4) INK_LINSK(5) INK_LINSK(6) INK_LINSK(8)

// ============================================================================
// 4. Inkling-2-Mini MoE Pipeline (Register-Only Decoders + 8-Token Micro-Batching!)
// ============================================================================

struct InkExpertDev {
    ulong body_off;
    uint dn_off;
    uchar gu_code;
    uchar dn_code;
    ushort pad;
};

kernel void ink_moe_plan(
    device const int* topk_idx [[buffer(0)]],
    constant int& P            [[buffer(1)]],
    device int* uexp           [[buffer(2)]],
    device int* uoff           [[buffer(3)]],
    device int* ucnt           [[buffer(4)]],
    device int* pair_list      [[buffer(5)]],
    device uint* meta          [[buffer(6)]],
    uint e    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sg_u[4];
    threadgroup int sg_p[4];
    int c = 0;
    for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) ++c;
    const int has = c > 0 ? 1 : 0;
    const int ex_u = simd_prefix_exclusive_sum(has);
    const int ex_p = simd_prefix_exclusive_sum(c);
    if (lane == 31) { sg_u[sg] = ex_u + has; sg_p[sg] = ex_p + c; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int base_u = 0, base_p = 0;
    for (uint w = 0; w < sg; ++w) { base_u += sg_u[w]; base_p += sg_p[w]; }
    const int pos_u = base_u + ex_u, pos_p = base_p + ex_p;
    if (has) {
        uexp[pos_u] = int(e); uoff[pos_u] = pos_p; ucnt[pos_u] = c;
        int k = 0;
        for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) pair_list[pos_p + k++] = i;
    }
    if (e == 127) {
        const uint n_uniq = uint(base_u + ex_u + has);
        meta[0] = n_uniq; meta[1] = uint(P);
        meta[2] = 144u; meta[3] = n_uniq; meta[4] = 1u;
        meta[5] = 144u; meta[6] = n_uniq; meta[7] = 1u;
    }
}

// Vectorized 64-bit uint2 SF4 w13 (16 weights per 64-bit load!)
static inline void sf4_w13_chunk(
    device const uchar* body, int r, device const float* x_in,
    thread const int* tk, int mm, int lane, thread float* ag, thread float* au) {
    device const float* hdr = (device const float*)body;
    const float outer = hdr[0] / hdr[1];
    device const uchar* p_base = body + 16;
    device const uchar* s_base = p_base + 2654208;
    const int rg = 2 * r, ru = 2 * r + 1;
    device const uint2* pg = (device const uint2*)(p_base + (size_t)rg * 1152);
    device const uchar* sg = s_base + (size_t)rg * 144;
    device const uint2* pu = (device const uint2*)(p_base + (size_t)ru * 1152);
    device const uchar* su = s_base + (size_t)ru * 144;

    for (int b = lane; b < 144; b += 32) {
        const float scg = e4m3_to_f32(sg[b]) * outer;
        const float scu = e4m3_to_f32(su[b]) * outer;
        const uint2 bg = pg[b], bu = pu[b];
        const float4 wg0 = float4(kE2M1[bg.x & 15u], kE2M1[(bg.x >> 4) & 15u], kE2M1[(bg.x >> 8) & 15u], kE2M1[(bg.x >> 12) & 15u]);
        const float4 wg1 = float4(kE2M1[(bg.x >> 16) & 15u], kE2M1[(bg.x >> 20) & 15u], kE2M1[(bg.x >> 24) & 15u], kE2M1[bg.x >> 28]);
        const float4 wg2 = float4(kE2M1[bg.y & 15u], kE2M1[(bg.y >> 4) & 15u], kE2M1[(bg.y >> 8) & 15u], kE2M1[(bg.y >> 12) & 15u]);
        const float4 wg3 = float4(kE2M1[(bg.y >> 16) & 15u], kE2M1[(bg.y >> 20) & 15u], kE2M1[(bg.y >> 24) & 15u], kE2M1[bg.y >> 28]);
        const float4 wu0 = float4(kE2M1[bu.x & 15u], kE2M1[(bu.x >> 4) & 15u], kE2M1[(bu.x >> 8) & 15u], kE2M1[(bu.x >> 12) & 15u]);
        const float4 wu1 = float4(kE2M1[(bu.x >> 16) & 15u], kE2M1[(bu.x >> 20) & 15u], kE2M1[(bu.x >> 24) & 15u], kE2M1[bu.x >> 28]);
        const float4 wu2 = float4(kE2M1[bu.y & 15u], kE2M1[(bu.y >> 4) & 15u], kE2M1[(bu.y >> 8) & 15u], kE2M1[(bu.y >> 12) & 15u]);
        const float4 wu3 = float4(kE2M1[(bu.y >> 16) & 15u], kE2M1[(bu.y >> 20) & 15u], kE2M1[(bu.y >> 24) & 15u], kE2M1[bu.y >> 28]);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(x_in + (size_t)tk[i] * 2304 + b * 16);
            const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            ag[i] = fma(scg, dot(wg0, x0) + dot(wg1, x1) + dot(wg2, x2) + dot(wg3, x3), ag[i]);
            au[i] = fma(scu, dot(wu0, x0) + dot(wu1, x1) + dot(wu2, x2) + dot(wu3, x3), au[i]);
        }
    }
}

static inline void mx8_w13_chunk(
    device const uchar* body, int r, device const float* x_in,
    thread const int* tk, int mm, int lane, thread float* ag, thread float* au) {
    device const uchar* w_base = body;
    device const uchar* s_base = body + 5308416;
    const int rg = 2 * r, ru = 2 * r + 1;
    device const uint* wg = (device const uint*)(w_base + (size_t)rg * 2304);
    device const uchar* sg = s_base + (size_t)rg * 72;
    device const uint* wu = (device const uint*)(w_base + (size_t)ru * 2304);
    device const uchar* su = s_base + (size_t)ru * 72;

    for (int b = lane; b < 72; b += 32) {
        const float scg = ue8m0_to_f32(sg[b]);
        const float scu = ue8m0_to_f32(su[b]);
        device const uint* bg = wg + b * 8;
        device const uint* bu = wu + b * 8;
        for (int s = 0; s < 8; s += 2) {
            const uint g0w = bg[s], g1w = bg[s + 1];
            const uint u0w = bu[s], u1w = bu[s + 1];
            const float4 g0 = float4(e4m3_to_f32(g0w & 255u), e4m3_to_f32((g0w >> 8) & 255u), e4m3_to_f32((g0w >> 16) & 255u), e4m3_to_f32(g0w >> 24));
            const float4 g1 = float4(e4m3_to_f32(g1w & 255u), e4m3_to_f32((g1w >> 8) & 255u), e4m3_to_f32((g1w >> 16) & 255u), e4m3_to_f32(g1w >> 24));
            const float4 u0 = float4(e4m3_to_f32(u0w & 255u), e4m3_to_f32((u0w >> 8) & 255u), e4m3_to_f32((u0w >> 16) & 255u), e4m3_to_f32(u0w >> 24));
            const float4 u1 = float4(e4m3_to_f32(u1w & 255u), e4m3_to_f32((u1w >> 8) & 255u), e4m3_to_f32((u1w >> 16) & 255u), e4m3_to_f32(u1w >> 24));
            for (int i = 0; i < mm; ++i) {
                device const float4* xp = (device const float4*)(x_in + (size_t)tk[i] * 2304 + b * 32 + s * 4);
                const float4 x0 = xp[0], x1 = xp[1];
                ag[i] = fma(scg, dot(g0, x0) + dot(g1, x1), ag[i]);
                au[i] = fma(scu, dot(u0, x0) + dot(u1, x1), au[i]);
            }
        }
    }
}

template <int F>
static inline void sk_w13_chunk(
    device const uchar* body, int r, device const float* x_in,
    thread const int* tk, int mm, int lane, thread float* ag, thread float* au) {
    constexpr int SB = sk_sb_bytes(F);
    const size_t rowb = 9ul * SB;
    device const uchar* gu = body + 2304;
    device const uchar* rg = gu + (size_t)(2 * r) * rowb;
    device const uchar* ru = gu + (size_t)(2 * r + 1) * rowb;
    for (int unit = lane; unit < 288; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        device const uchar* bg = rg + (size_t)sbi * SB;
        device const uchar* bu = ru + (size_t)sbi * SB;
        const float effg = float(as_type<half>(ld16(bg))) * (float(bg[2 + g]) * (1.0f / 255.0f));
        const float effu = float(as_type<half>(ld16(bu))) * (float(bu[2 + g]) * (1.0f / 255.0f));
        float4 wg0, wg1, wu0, wu1;
        sk_w8<F>(bg + 6, j, wg0, wg1);
        sk_w8<F>(bu + 6, j, wu0, wu1);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(x_in + (size_t)tk[i] * 2304 + sbi * 256 + j);
            const float4 x0 = xp[0], x1 = xp[1];
            ag[i] = fma(effg, dot(wg0, x0) + dot(wg1, x1), ag[i]);
            au[i] = fma(effu, dot(wu0, x0) + dot(wu1, x1), au[i]);
        }
    }
}

kernel void ink_moe_gu_situ(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;

    const int e = uexp[y], off = uoff[y], n = ucnt[y];
    const InkExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16);
    const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(ld16(body + 2 * r)))) : 1.0f;

    for (int t0 = 0; t0 < n; t0 += 8) {
        const int mm = min(8, n - t0);
        int pid[8], tk[8];
        for (int i = 0; i < mm; ++i) { pid[i] = pair_list[off + t0 + i]; tk[i] = pid[i] >> 3; }
        float ag[8] = {0,0,0,0,0,0,0,0};
        float au[8] = {0,0,0,0,0,0,0,0};
        if (ed.gu_code == 15)      sf4_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 16) mx8_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 3)  sk_w13_chunk<3>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 2)  sk_w13_chunk<2>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 4)  sk_w13_chunk<4>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 5)  sk_w13_chunk<5>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 6)  sk_w13_chunk<6>(body, r, x_in, tk, mm, int(lane), ag, au);
        else if (ed.gu_code == 8)  sk_w13_chunk<8>(body, r, x_in, tk, mm, int(lane), ag, au);

        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]);
            const float u = simd_sum(au[i]);
            if (lane == 0) h_act[(size_t)pid[i] * 1152 + r] = situ_act(g, u) * inv_sq;
        }
    }
}

static inline void sf4_w2_pair(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const int* pid, int mm, int lane, thread float* a0, thread float* a1) {
    device const float* hdr = (device const float*)body;
    const float outer = hdr[2] / hdr[3];
    device const uchar* p_base = body + dn_off;
    device const uchar* s_base = p_base + 1327104;
    const int r0 = 2 * rp, r1 = 2 * rp + 1;
    device const uint2* p0 = (device const uint2*)(p_base + (size_t)r0 * 576);
    device const uchar* s0 = s_base + (size_t)r0 * 72;
    device const uint2* p1 = (device const uint2*)(p_base + (size_t)r1 * 576);
    device const uchar* s1 = s_base + (size_t)r1 * 72;

    for (int b = lane; b < 72; b += 32) {
        const float sc0 = e4m3_to_f32(s0[b]) * outer;
        const float sc1 = e4m3_to_f32(s1[b]) * outer;
        const uint2 b0 = p0[b], b1 = p1[b];
        const float4 w00 = float4(kE2M1[b0.x & 15u], kE2M1[(b0.x >> 4) & 15u], kE2M1[(b0.x >> 8) & 15u], kE2M1[(b0.x >> 12) & 15u]);
        const float4 w01 = float4(kE2M1[(b0.x >> 16) & 15u], kE2M1[(b0.x >> 20) & 15u], kE2M1[(b0.x >> 24) & 15u], kE2M1[b0.x >> 28]);
        const float4 w02 = float4(kE2M1[b0.y & 15u], kE2M1[(b0.y >> 4) & 15u], kE2M1[(b0.y >> 8) & 15u], kE2M1[(b0.y >> 12) & 15u]);
        const float4 w03 = float4(kE2M1[(b0.y >> 16) & 15u], kE2M1[(b0.y >> 20) & 15u], kE2M1[(b0.y >> 24) & 15u], kE2M1[b0.y >> 28]);
        const float4 w10 = float4(kE2M1[b1.x & 15u], kE2M1[(b1.x >> 4) & 15u], kE2M1[(b1.x >> 8) & 15u], kE2M1[(b1.x >> 12) & 15u]);
        const float4 w11 = float4(kE2M1[(b1.x >> 16) & 15u], kE2M1[(b1.x >> 20) & 15u], kE2M1[(b1.x >> 24) & 15u], kE2M1[b1.x >> 28]);
        const float4 w12 = float4(kE2M1[b1.y & 15u], kE2M1[(b1.y >> 4) & 15u], kE2M1[(b1.y >> 8) & 15u], kE2M1[(b1.y >> 12) & 15u]);
        const float4 w13 = float4(kE2M1[(b1.y >> 16) & 15u], kE2M1[(b1.y >> 20) & 15u], kE2M1[(b1.y >> 24) & 15u], kE2M1[b1.y >> 28]);
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + (size_t)pid[i] * 1152 + b * 16);
            const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            a0[i] = fma(sc0, dot(w00, x0) + dot(w01, x1) + dot(w02, x2) + dot(w03, x3), a0[i]);
            a1[i] = fma(sc1, dot(w10, x0) + dot(w11, x1) + dot(w12, x2) + dot(w13, x3), a1[i]);
        }
    }
}

static inline void mx8_w2_pair(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const int* pid, int mm, int lane, thread float* a0, thread float* a1) {
    device const uchar* w_base = body + dn_off;
    device const uchar* s_base = w_base + 2654208;
    const int r0 = 2 * rp, r1 = 2 * rp + 1;
    device const uint* w0 = (device const uint*)(w_base + (size_t)r0 * 1152);
    device const uchar* s0 = s_base + (size_t)r0 * 36;
    device const uint* w1 = (device const uint*)(w_base + (size_t)r1 * 1152);
    device const uchar* s1 = s_base + (size_t)r1 * 36;

    for (int b = lane; b < 36; b += 32) {
        const float sc0 = ue8m0_to_f32(s0[b]), sc1 = ue8m0_to_f32(s1[b]);
        device const uint* b0 = w0 + b * 8;
        device const uint* b1 = w1 + b * 8;
        for (int s = 0; s < 8; s += 2) {
            const uint u0w = b0[s], u1w = b0[s + 1];
            const uint v0w = b1[s], v1w = b1[s + 1];
            const float4 u0 = float4(e4m3_to_f32(u0w & 255u), e4m3_to_f32((u0w >> 8) & 255u), e4m3_to_f32((u0w >> 16) & 255u), e4m3_to_f32(u0w >> 24));
            const float4 u1 = float4(e4m3_to_f32(u1w & 255u), e4m3_to_f32((u1w >> 8) & 255u), e4m3_to_f32((u1w >> 16) & 255u), e4m3_to_f32(u1w >> 24));
            const float4 v0 = float4(e4m3_to_f32(v0w & 255u), e4m3_to_f32((v0w >> 8) & 255u), e4m3_to_f32((v0w >> 16) & 255u), e4m3_to_f32(v0w >> 24));
            const float4 v1 = float4(e4m3_to_f32(v1w & 255u), e4m3_to_f32((v1w >> 8) & 255u), e4m3_to_f32((v1w >> 16) & 255u), e4m3_to_f32(v1w >> 24));
            for (int i = 0; i < mm; ++i) {
                device const float4* xp = (device const float4*)(h_act + (size_t)pid[i] * 1152 + b * 32 + s * 4);
                const float4 x0 = xp[0], x1 = xp[1];
                a0[i] = fma(sc0, dot(u0, x0) + dot(u1, x1), a0[i]);
                a1[i] = fma(sc1, dot(v0, x0) + dot(v1, x1), a1[i]);
            }
        }
    }
}

template <int F>
static inline void sk_w2_paired2(
    device const uchar* body, uint dn_off, int rp, device const float* h_act,
    thread const int* pid, int mm, int lane, thread float* a0, thread float* a1) {
    constexpr int SB = sk_sb_bytes(F);
    device const uchar* pr = body + dn_off + (size_t)rp * 9ul * SB;
    for (int unit = lane; unit < 288; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        device const uchar* blk = pr + (size_t)sbi * SB;
        const float eff = float(as_type<half>(ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
        float4 w0, w1;
        sk_w8<F>(blk + 6, j, w0, w1);
        const bool is_r1 = (unit >= 144);
        const int col = (is_r1 ? (unit - 144) : unit) << 3;
        for (int i = 0; i < mm; ++i) {
            device const float4* xp = (device const float4*)(h_act + (size_t)pid[i] * 1152 + col);
            const float val = eff * (dot(w0, xp[0]) + dot(w1, xp[1]));
            if (is_r1) a1[i] += val; else a0[i] += val;
        }
    }
}

kernel void ink_moe_dn(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int rp = int(tgp.x) * 8 + int(sg);
    if (rp >= 1152) return;

    const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
    const InkExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;

    for (int t0 = 0; t0 < n; t0 += 8) {
        const int mm = min(8, n - t0);
        int pid[8];
        for (int i = 0; i < mm; ++i) pid[i] = pair_list[off + t0 + i];
        float a0[8] = {0,0,0,0,0,0,0,0};
        float a1[8] = {0,0,0,0,0,0,0,0};
        if (ed.dn_code == 15)      sf4_w2_pair(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 16) mx8_w2_pair(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 2)  sk_w2_paired2<2>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 3)  sk_w2_paired2<3>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 4)  sk_w2_paired2<4>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 5)  sk_w2_paired2<5>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 6)  sk_w2_paired2<6>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        else if (ed.dn_code == 8)  sk_w2_paired2<8>(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);

        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]);
            const float v1 = simd_sum(a1[i]);
            if (lane == 0) {
                const float rw = topk_w[pid[i]];
                y_pairs[(size_t)pid[i] * 2304 + 2 * rp]     = v0 * rw;
                y_pairs[(size_t)pid[i] * 2304 + 2 * rp + 1] = v1 * rw;
            }
        }
    }
}

kernel void ink_moe_reduce8(
    device const float* y_pairs [[buffer(0)]],
    device float* moe_out       [[buffer(1)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int d = int(tgp.y) * 256 + int(tid);
    if (d >= 2304) return;
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += y_pairs[((size_t)m * 8 + k) * 2304 + d];
    moe_out[(size_t)m * 2304 + d] = s;
}

// ---- Low-Register Specialized SF4 & MX8 MoE Shaders (high occupancy on Apple Silicon for SQ-20!) ----
kernel void ink_moe_gu_sf4(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;
    const int e = uexp[y], off = uoff[y], n = ucnt[y];
    device const uchar* body = sec4 + ex[e].body_off;
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        int pid[4], tk[4];
        for (int i = 0; i < mm; ++i) { pid[i] = pair_list[off + t0 + i]; tk[i] = pid[i] >> 3; }
        float ag[4] = {0,0,0,0}, au[4] = {0,0,0,0};
        sf4_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0) h_act[(size_t)pid[i] * 1152 + r] = situ_act(g, u);
        }
    }
}

kernel void ink_moe_dn_sf4(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int rp = int(tgp.x) * 8 + int(sg);
    if (rp >= 1152) return;
    const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
    const InkExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        int pid[4];
        for (int i = 0; i < mm; ++i) pid[i] = pair_list[off + t0 + i];
        float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
        sf4_w2_pair(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]), v1 = simd_sum(a1[i]);
            if (lane == 0) {
                const float rw = topk_w[pid[i]];
                y_pairs[(size_t)pid[i] * 2304 + 2 * rp]     = v0 * rw;
                y_pairs[(size_t)pid[i] * 2304 + 2 * rp + 1] = v1 * rw;
            }
        }
    }
}

kernel void ink_moe_gu_mx8(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* x_in        [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device float* h_act             [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    if (r >= 1152) return;
    const int e = uexp[y], off = uoff[y], n = ucnt[y];
    device const uchar* body = sec4 + ex[e].body_off;
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        int pid[4], tk[4];
        for (int i = 0; i < mm; ++i) { pid[i] = pair_list[off + t0 + i]; tk[i] = pid[i] >> 3; }
        float ag[4] = {0,0,0,0}, au[4] = {0,0,0,0};
        mx8_w13_chunk(body, r, x_in, tk, mm, int(lane), ag, au);
        for (int i = 0; i < mm; ++i) {
            const float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0) h_act[(size_t)pid[i] * 1152 + r] = situ_act(g, u);
        }
    }
}

kernel void ink_moe_dn_mx8(
    device const uchar* sec4        [[buffer(0)]],
    device const InkExpertDev* ex   [[buffer(1)]],
    device const float* topk_w      [[buffer(2)]],
    device const int* uexp          [[buffer(3)]],
    device const int* uoff          [[buffer(4)]],
    device const int* ucnt          [[buffer(5)]],
    device const int* pair_list     [[buffer(6)]],
    device const uint* meta         [[buffer(7)]],
    device const float* h_act       [[buffer(8)]],
    device float* y_pairs           [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int rp = int(tgp.x) * 8 + int(sg);
    if (rp >= 1152) return;
    const int e = uexp[yy], off = uoff[yy], n = ucnt[yy];
    const InkExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    for (int t0 = 0; t0 < n; t0 += 4) {
        const int mm = min(4, n - t0);
        int pid[4];
        for (int i = 0; i < mm; ++i) pid[i] = pair_list[off + t0 + i];
        float a0[4] = {0,0,0,0}, a1[4] = {0,0,0,0};
        mx8_w2_pair(body, ed.dn_off, rp, h_act, pid, mm, int(lane), a0, a1);
        for (int i = 0; i < mm; ++i) {
            const float v0 = simd_sum(a0[i]), v1 = simd_sum(a1[i]);
            if (lane == 0) {
                const float rw = topk_w[pid[i]];
                y_pairs[(size_t)pid[i] * 2304 + 2 * rp]     = v0 * rw;
                y_pairs[(size_t)pid[i] * 2304 + 2 * rp + 1] = v1 * rw;
            }
        }
    }
}
