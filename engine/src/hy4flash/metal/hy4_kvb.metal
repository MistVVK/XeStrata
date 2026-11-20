// hy4_kvb.metal - Quantized S4/S5/S6/S8 kv_b_proj decode/verify kernels (-0.75 GB/tok, -1.05 GB wired RAM),
// per-head simdgroup_matrix prefill prep_q/vup, fast simdgroup_matrix IndexPool scorer, and fused small kernels.
// Compiled with hy4.metal (sk_sb_bytes, sk_half, sk_q8, sk_q, h2f_bits, ld16, rope_pair, sigmoidf_, tg_sum, HID, NHC, MLA_ROW, MLA_KV).

// ================================================================ 1. Quantized kv_b_proj decode / verify ================================================================

// qa[m][h][0:512] = sum_{i<192} q_b[m][h][i] * Wk[h*448+i][0:512]  (+ rope on [512:576]).
// Grid (NH, 2), 256 threads (8 simdgroups). Block (h, s) handles super-block s (256 cols).
// Bitwise M-invariant across MT = 1..8: every row m accumulates in the exact same wave/row order.
template <int F, int MT>
kernel void mla_prep_q_dec(device const uchar* kvb [[buffer(0)]], device const float* q_b [[buffer(1)]], device half* qa [[buffer(2)]],
                           constant int& M [[buffer(3)]], constant int& NH [[buffer(4)]], constant int& p0 [[buffer(5)]], constant float& theta [[buffer(6)]],
                           uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int h = int(tgp.x), s = int(tgp.y);
    const uint wave = tid >> 5, lane = tid & 31u;
    const int g = int(lane >> 3);
    const int j = int(lane << 3);
    threadgroup float red[8][256];

    float acc[MT][8];
    for (int m = 0; m < MT; ++m)
        for (int k = 0; k < 8; ++k) acc[m][k] = 0.0f;

    for (int ii = 0; ii < 24; ++ii) {
        const int i = int(wave) * 24 + ii;
        device const uchar* blk = kvb + ((size_t) h * 448 + i) * (2 * SB) + (size_t) s * SB;
        const float D = float(as_type<half>(*((device const ushort*) blk)));
        const float eff = D * (float(blk[2 + g]) * (1.0f / 255.0f));
        int q[8];
        sk_q8<F>(blk + 6, j, q);
        float qh[8];
        for (int k = 0; k < 8; ++k) qh[k] = float(q[k]) - H;
        for (int m = 0; m < MT; ++m) {
            const float qe = q_b[((size_t) m * NH + h) * 256 + i] * eff;
            for (int k = 0; k < 8; ++k) acc[m][k] = fma(qe, qh[k], acc[m][k]);
        }
    }

    for (int m = 0; m < MT; ++m) {
        for (int k = 0; k < 8; ++k) red[wave][j + k] = acc[m][k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v = red[0][tid] + red[1][tid] + red[2][tid] + red[3][tid] + red[4][tid] + red[5][tid] + red[6][tid] + red[7][tid];
        qa[((size_t) m * NH + h) * MLA_ROW + s * 256 + int(tid)] = half(v);
        if (m + 1 < MT) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (s != 0) return;
    for (int idx = int(tid); idx < MT * 32; idx += 256) {
        const int m = idx >> 5, i = idx & 31;
        device const float* qp = q_b + ((size_t) m * NH + h) * 256 + 192 + 2 * i;
        float o0, o1;
        rope_pair(qp[0], qp[1], p0 + m, i, theta, o0, o1);
        *((device half2*) (qa + ((size_t) m * NH + h) * MLA_ROW + MLA_KV + 2 * i)) = half2(half(o0), half(o1));
    }
}

// o[m][h][j] = half(sigmoid(gate) * sum_{c<512} Wv[h*448+192+j][c] * ctx[m][h][c]).
// Grid (NH, 32), 256 threads (8 simdgroups, 1 row per simdgroup).
// Bitwise M-invariant across MT = 1..8.
template <int F, int MT>
kernel void mla_vup_dec(device const uchar* kvb [[buffer(0)]], device const half* ctx [[buffer(1)]], device const float* gate [[buffer(2)]], device half* o [[buffer(3)]],
                        constant int& M [[buffer(4)]], constant int& NH [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    const int h = int(tgp.x);
    const uint sg = tid >> 5, lane = tid & 31u;
    const int row_j = int(tgp.y) * 8 + int(sg);
    const int g = int(lane >> 3);
    const int k0 = int(lane << 3);

    threadgroup half sc[MT][512];
    for (int idx = int(tid); idx < MT * 256; idx += 256) {
        const int m = idx >> 8, c2 = idx & 255;
        *((threadgroup half2*) (sc[m] + 2 * c2)) = *((device const half2*) (ctx + ((size_t) m * NH + h) * MLA_KV + 2 * c2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* rp = kvb + ((size_t) h * 448 + 192 + row_j) * (2 * SB);
    float acc[MT];
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

    for (int s = 0; s < 2; ++s) {
        device const uchar* blk = rp + (size_t) s * SB;
        const float D = float(as_type<half>(*((device const ushort*) blk)));
        const float eff = D * (float(blk[2 + g]) * (1.0f / 255.0f));
        int q[8];
        sk_q8<F>(blk + 6, k0, q);
        const float4 w0 = float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) - H;
        const float4 w1 = float4(float(q[4]), float(q[5]), float(q[6]), float(q[7])) - H;
        for (int m = 0; m < MT; ++m) {
            threadgroup const half4* xp = (threadgroup const half4*) (sc[m] + s * 256 + k0);
            const float d = dot(w0, float4(xp[0])) + dot(w1, float4(xp[1]));
            acc[m] = fma(eff, d, acc[m]);
        }
    }

    for (int m = 0; m < MT; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            const size_t gi = (size_t) m * NH * 256 + (size_t) h * 256 + row_j;
            o[gi] = half(v * sigmoidf_(gate[gi]));
        }
    }
}

#define INST_KVB_MT(F, MT)                                                                                                                                                   \
    template [[host_name("mla_prep_q_dec_" #F "_" #MT)]] kernel void mla_prep_q_dec<F, MT>(device const uchar*, device const float*, device half*, constant int&,            \
                                                                                           constant int&, constant int&, constant float&, uint2, uint);                      \
    template [[host_name("mla_vup_dec_" #F "_" #MT)]] kernel void mla_vup_dec<F, MT>(device const uchar*, device const half*, device const float*, device half*,             \
                                                                                     constant int&, constant int&, uint2, uint);
#define INST_KVB_F(F) INST_KVB_MT(F, 1) INST_KVB_MT(F, 2) INST_KVB_MT(F, 3) INST_KVB_MT(F, 4) INST_KVB_MT(F, 5) INST_KVB_MT(F, 6) INST_KVB_MT(F, 7) INST_KVB_MT(F, 8)
INST_KVB_F(4)
INST_KVB_F(5)
INST_KVB_F(6)
INST_KVB_F(8)

// ================================================================ 2. Single-layer scratch dequant + per-head prefill GEMM ================================================

// Dequantizes kv_b_proj [NH*448][512] into wk_t [NH][512][192] (transposed for GEMM A@B^T!) and wv [NH][256][512] half.
// Grid (NH * 448), 128 threads.
template <int F>
kernel void split_kvb_t(device const uchar* W [[buffer(0)]], device half* wk_t [[buffer(1)]], device half* wv [[buffer(2)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    const int h = int(r) / 448;
    const int j = int(r) % 448;
    constexpr int SB = sk_sb_bytes(F);
    device const uchar* rp = W + (size_t) r * 2 * SB;
    if (j < 192) {
        device half* dst = wk_t + (size_t) h * 512 * 192 + j;
        for (uint k = tid; k < 512; k += nthr) {
            device const uchar* blk = rp + (size_t) (k >> 8) * SB;
            const float D = h2f_bits(ld16(blk));
            const int i = int(k & 255u);
            const float eff = D * (float(blk[2 + (i >> 6)]) * (1.0f / 255.0f));
            dst[(size_t) k * 192] = half((float(sk_q<F>(blk + 6, i)) - sk_half(F)) * eff);
        }
    } else {
        device half* dst = wv + ((size_t) h * 256 + (j - 192)) * 512;
        for (uint k = tid; k < 512; k += nthr) {
            device const uchar* blk = rp + (size_t) (k >> 8) * SB;
            const float D = h2f_bits(ld16(blk));
            const int i = int(k & 255u);
            const float eff = D * (float(blk[2 + (i >> 6)]) * (1.0f / 255.0f));
            dst[k] = half((float(sk_q<F>(blk + 6, i)) - sk_half(F)) * eff);
        }
    }
}
template [[host_name("split_kvb_t_4")]] kernel void split_kvb_t<4>(device const uchar*, device half*, device half*, uint, uint, uint);
template [[host_name("split_kvb_t_5")]] kernel void split_kvb_t<5>(device const uchar*, device half*, device half*, uint, uint, uint);
template [[host_name("split_kvb_t_6")]] kernel void split_kvb_t<6>(device const uchar*, device half*, device half*, uint, uint, uint);
template [[host_name("split_kvb_t_8")]] kernel void split_kvb_t<8>(device const uchar*, device half*, device half*, uint, uint, uint);

// Per-head simdgroup_matrix q-absorb + RoPE for prefill (M > 8):
// qa[t][h][0:512] = q_b[t][h][0:192] @ wk_t[h][0:512][0:192]^T, qa[t][h][512:576] = rope(q_b[t][h][192:256]).
// Grid (512/64 = 8, ceil(M/32), NH), 128 threads (4 simdgroups, each 32 x 16).
kernel void mla_prep_q_mm(device const float* q_b [[buffer(0)]], device const half* wk_t [[buffer(1)]], device half* qa [[buffer(2)]],
                          constant int& M [[buffer(3)]], constant int& NH [[buffer(4)]], constant int& p0 [[buffer(5)]], constant float& theta [[buffer(6)]],
                          uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64]; // 4 KB
    threadgroup half sB[64 * 64]; // 8 KB
    const int n0 = int(tgp.x) * 64;
    const int t0 = int(tgp.y) * 32;
    const int h = int(tgp.z);

    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    device const half* B = wk_t + ((size_t) h * 512 + n0) * 192;
    for (int k0 = 0; k0 < 192; k0 += 64) {
        // Load A: 32 x 64 float -> half (2048 elems / 128 threads = 16 elems = 4 float4/thread)
        for (int p = 0; p < 4; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint r = idx >> 4;
            const uint c4 = (idx & 15u) << 2;
            const int t = t0 + int(r);
            float4 v = t < M ? *((device const float4*) (q_b + ((size_t) t * NH + h) * 256 + k0 + int(c4))) : float4(0.0f);
            *((threadgroup half4*) (sA + r * 64u + c4)) = half4(v);
        }
        // Load B: 64 x 64 half (512 uint4 / 128 threads = 4 uint4/thread)
        for (int p = 0; p < 4; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint r = idx >> 3;
            const uint c8 = (idx & 7u) << 3;
            *((threadgroup uint4*) (sB + r * 64u + c8)) = *((device const uint4*) (B + (size_t) r * 192 + k0 + int(c8)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 a[4], b[2];
            for (int i = 0; i < 4; ++i) simdgroup_load(a[i], sA + (i * 8) * 64 + kk * 8, 64);
            for (int j = 0; j < 2; ++j) simdgroup_load(b[j], sB + (sg * 16 + j * 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + i * 8 + int(fm);
        if (t < M) {
            device half* dst = qa + ((size_t) t * NH + h) * MLA_ROW + n0 + sg * 16 + fn;
            for (int j = 0; j < 2; ++j)
                *((device half2*) (dst + j * 8)) = half2(half(acc[i][j].thread_elements()[0]), half(acc[i][j].thread_elements()[1]));
        }
    }

    if (tgp.x == 0) {
        for (int idx = int(tid); idx < 32 * 32; idx += 128) {
            const int tt = idx >> 5, i = idx & 31, t = t0 + tt;
            if (t >= M) continue;
            device const float* qp = q_b + ((size_t) t * NH + h) * 256 + 192 + 2 * i;
            float o0, o1;
            rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
            *((device half2*) (qa + ((size_t) t * NH + h) * MLA_ROW + MLA_KV + 2 * i)) = half2(half(o0), half(o1));
        }
    }
}

// Per-head simdgroup_matrix v-up + output gate for prefill (M > 8):
// o[t][h][0:256] = half(sigmoid(gate) * (ctx[t][h][0:512] @ wv[h][0:256][0:512]^T)).
// Grid (256/64 = 4, ceil(M/32), NH), 128 threads (4 simdgroups, each 32 x 16).
kernel void mla_vup_mm(device const half* ctx [[buffer(0)]], device const half* wv [[buffer(1)]], device const float* gate [[buffer(2)]], device half* o [[buffer(3)]],
                       constant int& M [[buffer(4)]], constant int& NH [[buffer(5)]],
                       uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[32 * 64]; // 4 KB
    threadgroup half sB[64 * 64]; // 8 KB
    const int n0 = int(tgp.x) * 64;
    const int t0 = int(tgp.y) * 32;
    const int h = int(tgp.z);

    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    device const half* B = wv + ((size_t) h * 256 + n0) * 512;
    for (int k0 = 0; k0 < 512; k0 += 64) {
        // Load A: 32 x 64 half (256 uint4 / 128 threads = 2 uint4/thread)
        for (int p = 0; p < 2; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint r = idx >> 3;
            const uint c8 = (idx & 7u) << 3;
            const int t = t0 + int(r);
            const uint4 av = t < M ? *((device const uint4*) (ctx + ((size_t) t * NH + h) * MLA_KV + k0 + int(c8))) : uint4(0u);
            *((threadgroup uint4*) (sA + r * 64u + c8)) = av;
        }
        // Load B: 64 x 64 half (512 uint4 / 128 threads = 4 uint4/thread)
        for (int p = 0; p < 4; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint r = idx >> 3;
            const uint c8 = (idx & 7u) << 3;
            *((threadgroup uint4*) (sB + r * 64u + c8)) = *((device const uint4*) (B + (size_t) r * 512 + k0 + int(c8)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 a[4], b[2];
            for (int i = 0; i < 4; ++i) simdgroup_load(a[i], sA + (i * 8) * 64 + kk * 8, 64);
            for (int j = 0; j < 2; ++j) simdgroup_load(b[j], sB + (sg * 16 + j * 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + i * 8 + int(fm);
        if (t < M) {
            const size_t base = (size_t) t * NH * 256 + (size_t) h * 256 + n0 + sg * 16 + fn;
            for (int j = 0; j < 2; ++j) {
                const float2 g2 = *((device const float2*) (gate + base + j * 8));
                const float v0 = acc[i][j].thread_elements()[0] * sigmoidf_(g2.x);
                const float v1 = acc[i][j].thread_elements()[1] * sigmoidf_(g2.y);
                *((device half2*) (o + base + j * 8)) = half2(half(v0), half(v1));
            }
        }
    }
}

// ================================================================ 3. Fast simdgroup_matrix IndexPool scorer =============================================================

// score[t][g] = sum_{h<16} relu(iq[t][h] . pooled[g]) * hw[t][h] * hscale for g < (p0+t)/4, else -INFINITY.
// Grid ((G+63)/64, T), 128 threads (4 simdgroups, each owns 16 groups x 16 heads).
// Eliminates 16x redundant pooled reads in scalar idx_score; 100% M-invariant (one threadgroup per (t, g_block)).
kernel void idx_score_sg(device const half* iq [[buffer(0)]], device const float* hw [[buffer(1)]], device const half* pooled [[buffer(2)]], device float* score [[buffer(3)]],
                         constant int& p0 [[buffer(4)]], constant int& G [[buffer(5)]], constant int& ldS [[buffer(6)]], constant float& hscale [[buffer(7)]],
                         uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int t = int(tgp.y);
    const int g0 = int(tgp.x) * 64;
    const int grp = (p0 + t) >> 2;
    if (g0 >= grp) {
        if (int(tid) < 64 && g0 + int(tid) < G) score[(size_t) t * ldS + g0 + int(tid)] = -INFINITY;
        return;
    }

    threadgroup half sQ[16 * 128]; // 4096 B
    threadgroup half sK[64 * 64];  // 8192 B
    threadgroup float shw[16];     // 64 B

    for (int p = 0; p < 2; ++p) {
        const uint idx = uint(p * 128) + tid;
        *((threadgroup uint4*) (sQ + idx * 8u)) = *((device const uint4*) (iq + (size_t) t * 2048 + idx * 8u));
    }
    if (tid < 16) shw[tid] = hw[(size_t) t * 16 + tid] * hscale;

    simdgroup_float8x8 s[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) s[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int hk = 0; hk < 2; ++hk) {
        for (int p = 0; p < 4; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint gr = idx >> 3;
            const uint gc = (idx & 7u) << 3;
            const int g = g0 + int(gr);
            const uint4 kv = g < grp ? *((device const uint4*) (pooled + (size_t) g * 128 + hk * 64 + int(gc))) : uint4(0u);
            *((threadgroup uint4*) (sK + gr * 64u + gc)) = kv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 qf0, qf1, kf0, kf1;
            simdgroup_load(qf0, sQ + hk * 64 + kk * 8, 128);
            simdgroup_load(qf1, sQ + 8 * 128 + hk * 64 + kk * 8, 128);
            simdgroup_load(kf0, sK + (sg * 16) * 64 + kk * 8, 64, ulong2(0, 0), true);
            simdgroup_load(kf1, sK + (sg * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s[0][0], qf0, kf0, s[0][0]);
            simdgroup_multiply_accumulate(s[0][1], qf0, kf1, s[0][1]);
            simdgroup_multiply_accumulate(s[1][0], qf1, kf0, s[1][0]);
            simdgroup_multiply_accumulate(s[1][1], qf1, kf1, s[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const float w0 = shw[fm];
    const float w1 = shw[8u + fm];
    for (int j = 0; j < 2; ++j) {
        for (int e = 0; e < 2; ++e) {
            float val = max(s[0][j].thread_elements()[e], 0.0f) * w0 + max(s[1][j].thread_elements()[e], 0.0f) * w1;
            // Reduce across all 8 rows fm = 0..7 holding column fn+e fixed (bits 1, 2, 4 -> xor 2, 4, 16)
            val += simd_shuffle_xor(val, ushort(2));
            val += simd_shuffle_xor(val, ushort(4));
            val += simd_shuffle_xor(val, ushort(16));
            if (fm == 0) {
                const int g = g0 + int(sg) * 16 + j * 8 + int(fn) + e;
                if (g < G) score[(size_t) t * ldS + g] = g < grp ? val : -INFINITY;
            }
        }
    }
}

// ================================================================ 4. Fused small kernels (cut dispatch count) ===========================================================

// Fused moe_reduce + f2h: y float [T][16][1536] -> z_h half [T][1536] (saves 41 dispatches + 41 barriers/step!)
kernel void moe_reduce_h(device const float* y [[buffer(0)]], device half* zh [[buffer(1)]],
                         uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    for (uint n = tid; n < 1536; n += nthr) {
        float a = 0.0f;
        for (int s = 0; s < 16; ++s) a += y[((size_t) t * 16 + s) * 1536 + n];
        zh[(size_t) t * 1536 + n] = half(a);
    }
}

// Fused moe_combine + f2h for the MPP MoE prefill path
kernel void moe_combine_h(device const float* y [[buffer(0)]], device const int* pos [[buffer(1)]], device const float* w [[buffer(2)]], device half* zh [[buffer(3)]],
                          uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    int rows[16];
    float wt[16];
    for (int s = 0; s < 16; ++s) { rows[s] = pos[t * 16 + s]; wt[s] = w[t * 16 + s]; }
    for (uint n = tid; n < 1536; n += 256) {
        float a = 0.0f;
        for (int s = 0; s < 16; ++s) a += y[(size_t) rows[s] * 1536 + n] * wt[s];
        zh[(size_t) t * 1536 + n] = half(a);
    }
}

// Fused dfly_rms_resid + post_attention_layernorm: H[row] += rms(o[row])*w_post; hn[row] = half(rms(H[row])*w_ln). Saves 5 dispatches/draft step!
kernel void dfly_rms_resid_norm(device const float* o [[buffer(0)]], device const float* w_post [[buffer(1)]], device float* H [[buffer(2)]],
                                device const float* w_ln [[buffer(3)]], device half* hn [[buffer(4)]], constant float& eps [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    device const float* x = o + (size_t) row * HID;
    device float* h = H + (size_t) row * HID;
    float ox[HID / 256];
    float ss = 0.0f;
    for (int i = 0; i < HID / 256; ++i) {
        const int d = int(tid) + i * 256;
        const float v = x[d];
        ox[i] = v;
        ss += v * v;
    }
    const float inv1 = rsqrt(tg_sum(ss, sh, tid, 256) / float(HID) + eps);
    float ss2 = 0.0f;
    for (int i = 0; i < HID / 256; ++i) {
        const int d = int(tid) + i * 256;
        const float hv = h[d] + ox[i] * inv1 * w_post[d];
        h[d] = hv;
        ox[i] = hv;
        ss2 += hv * hv;
    }
    const float inv2 = rsqrt(tg_sum(ss2, sh, tid, 256) / float(HID) + eps);
    for (int i = 0; i < HID / 256; ++i) {
        const int d = int(tid) + i * 256;
        hn[(size_t) row * HID + d] = half(ox[i] * inv2 * w_ln[d]);
    }
}

// Split-iHC reduce+norm for head() (nout == 4: pre only, no post). Same math as hc_reduce_norm_acc.
kernel void hc_reduce_norm_acc4(device const float* x [[buffer(0)]], device const float* part [[buffer(1)]], device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                                device float* pre [[buffer(4)]], device const float* w [[buffer(5)]], device half* outh [[buffer(6)]],
                                constant float& eps [[buffer(7)]], constant float& hc_eps [[buffer(8)]],
                                uint tok [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    threadgroup float tot[9];
    threadgroup float sp[4];
    if (tid < 9) {
        float a = 0.0f;
        for (int b = 0; b < 24; ++b) a += part[((size_t) tok * 24 + b) * 9 + tid];
        tot[tid] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float inv = rsqrt(tot[8] / float(4 * HID) + eps);
        for (int h = 0; h < NHC; ++h) {
            const float pr = sigmoidf_(tot[h] * inv * scale[0] + base[h]) + hc_eps;
            sp[h] = pr;
            pre[(size_t) tok * NHC + h] = pr;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* xt = x + (size_t) tok * NHC * HID;
    const float p0 = sp[0], p1 = sp[1], p2 = sp[2], p3 = sp[3];
    float ss = 0.0f;
    float v[HID / 256];
    for (int i = 0; i < HID / 256; ++i) {
        const int d = int(tid) + i * 256;
        const float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
        v[i] = c;
        ss += c * c;
    }
    const float inv = rsqrt(tg_sum(ss, sh, tid, 256) / float(HID) + eps);
    for (int i = 0; i < HID / 256; ++i) {
        const int d = int(tid) + i * 256;
        outh[(size_t) tok * HID + d] = half(v[i] * inv * w[d]);
    }
}

// Zero-barrier 1-simdgroup (32 threads) router_topk for E = 512, K = 16: each lane holds 16 experts in registers!
// Exact same sigmoid/bias/tie-break (lower index wins) and fp32 normalization as router_topk in hy4.metal; eliminates 32 threadgroup barriers per layer (-0.5 ms/tok!).
kernel void router_topk_fast(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]], device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                             constant int& E [[buffer(4)]], constant int& K [[buffer(5)]], constant float& scale [[buffer(6)]],
                             uint tok [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device const float* lg = logits + (size_t) tok * E;
    float lp[16], ls[16];
    for (int j = 0; j < 16; ++j) {
        const int e = int(lane) + j * 32;
        const float p = sigmoidf_(lg[e]);
        lp[j] = p;
        ls[j] = p + bias[e];
    }
    int chosen[16];
    float cp[16];
    for (int k = 0; k < 16; ++k) {
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
        for (int j = 0; j < 16; ++j) {
            const int e = int(lane) + j * 32;
            if (ls[j] > bv || (ls[j] == bv && e < bi)) { bv = ls[j]; bi = e; }
        }
        for (uint o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, ushort(o));
            const int oi = simd_shuffle_xor(bi, ushort(o));
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        chosen[k] = bi;
        const int owner_j = bi >> 5;
        const float p_win = simd_broadcast(lp[owner_j], ushort(bi & 31));
        cp[k] = p_win;
        if ((bi & 31) == int(lane)) ls[owner_j] = -3.0e38f;
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 16; ++k) sum += cp[k];
        const float denom = sum + 1e-20f;
        for (int k = 0; k < 16; ++k) {
            idx[(size_t) tok * 16 + k] = chosen[k];
            w[(size_t) tok * 16 + k] = cp[k] / denom * scale;
        }
    }
}
