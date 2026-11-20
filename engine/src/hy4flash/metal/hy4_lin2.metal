// hy4_lin2.metal - decode / verify GEMV with row blocking: one simdgroup computes R consecutive rows against MT tokens, so the activation loads (half4 x2 per token per unit, which cost about 3x the
// weight bytes in the single-row kernel) are shared by R rows and the weights of a unit are decoded once for all MT tokens. MT == M (1..8), per (row, token) the arithmetic is the same
// expression as linear_sk (the sum order over units/lanes is unchanged), so a row of a window equals the same row computed alone.
// Compiled with hy4.metal (sk_w8 etc.).

template <int F, int MT, int R>
kernel void linear_skr(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                       constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row0 = (tg * 8 + sg) * R;
    if (row0 >= N) return;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    const size_t rowb = (size_t) nsb * SB;
    float acc[R][MT];
    for (int r = 0; r < R; ++r)
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;
    const int nunits = nsb * 32;
    for (int unit = lane; unit < nunits; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        float4 xa[MT], xb[MT];
        for (int m = 0; m < MT; ++m) {
            device const half4* xp = (device const half4*) (X + (size_t) m * K + sbi * 256 + j);
            xa[m] = float4(xp[0]);
            xb[m] = float4(xp[1]);
        }
        for (int r = 0; r < R; ++r) {
            if (row0 + r < N) {
                float w[8];
                sk_w8<F>(W + (size_t) (row0 + r) * rowb + (size_t) sbi * SB, g, j, w);
                for (int m = 0; m < MT; ++m) {
                    float a = 0.f;
                    a += w[0] * xa[m].x; a += w[1] * xa[m].y; a += w[2] * xa[m].z; a += w[3] * xa[m].w;
                    a += w[4] * xb[m].x; a += w[5] * xb[m].y; a += w[6] * xb[m].z; a += w[7] * xb[m].w;
                    acc[r][m] += a;
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

#define LIN2(F, MT, R) template [[host_name("linear_skr_" #F "_" #MT "_" #R)]] kernel void linear_skr<F, MT, R>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint);
#define LIN2_ALL(F, R) LIN2(F, 1, R) LIN2(F, 2, R) LIN2(F, 3, R) LIN2(F, 4, R) LIN2(F, 5, R) LIN2(F, 6, R) LIN2(F, 7, R) LIN2(F, 8, R)
LIN2_ALL(4, 1) LIN2_ALL(4, 4)
LIN2_ALL(5, 1) LIN2_ALL(5, 4)
LIN2_ALL(6, 1) LIN2_ALL(6, 4)
LIN2_ALL(8, 1) LIN2_ALL(8, 4)
LIN2(4, 1, 2) LIN2(5, 1, 2) LIN2(6, 1, 2) LIN2(8, 1, 2)
LIN2(4, 2, 2) LIN2(5, 2, 2) LIN2(6, 2, 2) LIN2(8, 2, 2)
