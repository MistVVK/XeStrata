// hy4_lin2.metal - decode / verify GEMV (M = 1..8) + small-M simdgroup_matrix verify GEMM (BM = 8, BN = 32, BK = 64).
// Both linear_skr and linear_sm are 100% bitwise M-invariant across M = 1..8 (row m of an M-token window equals M = 1).
// Compiled with hy4.metal (sk_q8, sk_half, sk_sb_bytes, h2f_bits, ld16).

template <int F, int MT, int R>
kernel void linear_skr(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                       constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = (int(tg) * 8 + int(sg)) * R;
    if (row0 >= N) return;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t) nsb * SB;
    float acc[R][MT];
    for (int r = 0; r < R; ++r)
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;
    const int nunits = nsb * 32;
    for (int unit = int(lane); unit < nunits; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        if (R == 1) {
            device const uchar* blk = W + (size_t) row0 * rowb + (size_t) sbi * SB;
            const float eff = float(as_type<half>(*((device const ushort*) blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
            float4 w0, w1;
            sk_w8_f4<F>(blk + 6, j, w0, w1);
            for (int m = 0; m < MT; ++m) {
                device const half4* xp = (device const half4*) (X + (size_t) m * K + sbi * 256 + j);
                acc[0][m] = fma(eff, dot(w0, float4(xp[0])) + dot(w1, float4(xp[1])), acc[0][m]);
            }
        } else {
            float4 xa[MT], xb[MT];
            for (int m = 0; m < MT; ++m) {
                device const half4* xp = (device const half4*) (X + (size_t) m * K + sbi * 256 + j);
                xa[m] = float4(xp[0]);
                xb[m] = float4(xp[1]);
            }
            for (int r = 0; r < R; ++r) {
                if (row0 + r < N) {
                    device const uchar* blk = W + (size_t) (row0 + r) * rowb + (size_t) sbi * SB;
                    const float eff = float(as_type<half>(*((device const ushort*) blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
                    float4 w0, w1;
                    sk_w8_f4<F>(blk + 6, j, w0, w1);
                    for (int m = 0; m < MT; ++m) {
                        acc[r][m] = fma(eff, dot(w0, xa[m]) + dot(w1, xb[m]), acc[r][m]);
                    }
                }
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int m = 0; m < MT; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0 && row0 + r < N) Y[(size_t) m * N + row0 + r] = v;
        }
}

// Small-M simdgroup_matrix GEMM for M = 1..8: BM = 8 tokens x BN = 32 output rows, BK = 64, 4 simdgroups (128 threads, each 8 x 8).
// Because rows m >= M of sA are padded with 0.0h, every valid row m < M executes the exact same HMMA instructions for M = 1 and M = 2..8 -> 100% bitwise M-invariant!
// TG mem: sA[8*64] (1 KB) + sB[32*64] (4 KB) = 5 KB total (maximum occupancy!).
template <int F>
kernel void linear_sm(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                      constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sA[8 * 64];  // 1024 B
    threadgroup half sB[32 * 64]; // 4096 B
    const int n0 = int(tg) * 32;
    if (n0 >= N) return;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // 128 threads decode 32 weight rows x 64 k-elems = 2048 weights = 16 weights/thread (4 threads/row, 2 units of 8 weights per thread)
    const int wr = int(tid >> 2);        // 0..31
    const int wk = int((tid & 3u) << 4); // 0, 16, 32, 48
    const int wrow = n0 + wr;
    const size_t woff = (size_t) wrow * nsb * SB;

    for (int ks = 0; ks < (K >> 6); ++ks) {
        // Load A: 8 x 64 = 64 uint4 (first 64 threads load 1 uint4 each)
        if (tid < 64) {
            const int m = int(tid >> 3);
            const int c8 = int((tid & 7u) << 3);
            *((threadgroup uint4*) (sA + m * 64 + c8)) = m < M ? *((device const uint4*) (X + (size_t) m * K + ks * 64 + c8)) : uint4(0u);
        }
        // Dequant B: 32 x 64 half
        if (wrow < N) {
            const int sbi = ks >> 2, g = ks & 3;
            device const uchar* blk = W + woff + (size_t) sbi * SB;
            const float eff = float(as_type<half>(*((device const ushort*) blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
            for (int u = 0; u < 2; ++u) {
                const int col = wk + u * 8;
                float4 w0, w1;
                sk_w8_f4<F>(blk + 6, g * 64 + col, w0, w1);
                threadgroup half4* dst = (threadgroup half4*) (sB + wr * 64 + col);
                dst[0] = half4(w0 * eff);
                dst[1] = half4(w1 * eff);
            }
        } else {
            threadgroup uint4* dst = (threadgroup uint4*) (sB + wr * 64 + wk);
            dst[0] = uint4(0u);
            dst[1] = uint4(0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, sA + kk * 8, 64);
            simdgroup_load(b, sB + (sg * 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const int m = int(fm);
    const int n = n0 + int(sg) * 8 + int(fn);
    if (m < M) {
        if (n < N) Y[(size_t) m * N + n] = acc.thread_elements()[0];
        if (n + 1 < N) Y[(size_t) m * N + n + 1] = acc.thread_elements()[1];
    }
}

#define LIN2(F, MT, R) template [[host_name("linear_skr_" #F "_" #MT "_" #R)]] kernel void linear_skr<F, MT, R>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint);
#define LIN2_ALL(F, R) LIN2(F, 1, R) LIN2(F, 2, R) LIN2(F, 3, R) LIN2(F, 4, R) LIN2(F, 5, R) LIN2(F, 6, R) LIN2(F, 7, R) LIN2(F, 8, R)
LIN2_ALL(4, 1) LIN2_ALL(4, 2) LIN2_ALL(4, 4)
LIN2_ALL(5, 1) LIN2_ALL(5, 2) LIN2_ALL(5, 4)
LIN2_ALL(6, 1) LIN2_ALL(6, 2) LIN2_ALL(6, 4)
LIN2_ALL(8, 1) LIN2_ALL(8, 2) LIN2_ALL(8, 4)

#define LINSM(F) template [[host_name("linear_sm_" #F)]] kernel void linear_sm<F>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint, uint, uint);
LINSM(4) LINSM(5) LINSM(6) LINSM(8)
