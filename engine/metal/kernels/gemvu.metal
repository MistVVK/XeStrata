// metal/kernels/gemvu.metal - gemv.metal's kernel with the unit loop unrolled UF times, the weights of the UF units decoded (their
// loads issued) before any of them is applied.  The sums are accumulated in the original's order (unit u, then u + LPR, ...), so the
// result is bitwise gemv's; what changes is that a lane waits one memory latency for UF units instead of one for each.  After gemv.metal.
template <int F, int LPR, int NR, int SG, int KS, int UF>
kernel void gemv_u(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                   device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]], constant GemvArgs& a [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RS = SG / KS, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup float red[KS > 1 ? KS * RS * RG * NR * 8 : 1];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ks = sg / RS, rs = sg % RS;
    const int li = lane % LPR;
    const int lrow0 = (rs * RG + lane / LPR) * NR;
    const int row0 = int(tg.x) * RS * RG * NR + lrow0;
    const int U = int(a.n_in) / UNIT;
    const int u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    float acc[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
    device const uchar* rp[NR];
    for (int r = 0; r < NR; ++r) rp[r] = W + (ulong)min(row0 + r, int(a.n_out) - 1) * a.row_bytes;
    const int s16n = int(a.n_in) / 16;
    for (int u = u0 + li; u < u1; u += LPR * UF) {
        float q[UF][NR][UNIT];
        float4 sc[UF][NR];
        for (int k = 0; k < UF; ++k)
            if (u + k * LPR < u1)
                for (int r = 0; r < NR; ++r) Dec<F>::run(rp[r], u + k * LPR, q[k][r], sc[k][r], tab);
        for (int k = 0; k < UF; ++k)
            if (u + k * LPR < u1) {
                float xs[UNIT], xsum[2];
                for (int t = 0; t < NTOK; ++t) {
                    X<F>::load(xp + t * int(a.x_stride), s16 + t * s16n, u + k * LPR, xs, xsum);
                    for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine<UNIT>(q[k][r], sc[k][r], xs, xsum);
                }
            }
    }
    constexpr int RPT = RS * RG * NR;
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (KS == 1) {
                if (li == 0 && row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
            } else if (li == 0) {
                red[(ks * RPT + lrow0 + r) * 8 + t] = v;
            }
        }
    if (KS > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks == 0 && li == 0)
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) {
                    float v = 0.0f;
                    for (int k = 0; k < KS; ++k) v += red[(k * RPT + lrow0 + r) * 8 + t];
                    if (row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
                }
    }
}

#define GVU_(F, L, R, S, K, U) template [[host_name("gemvu_" #F "_" #L "_" #R "_" #S "_" #K "_" #U)]] kernel decltype(gemv_u<F, L, R, S, K, U>) gemv_u<F, L, R, S, K, U>;
#define GVU(F, L, R, S, K, U) GVU_(F, L, R, S, K, U)
#define GVU_SET(F, U) GVU(F, 16, 1, 4, 1, U) GVU(F, 32, 1, 8, 2, U) GVU(F, 8, 1, 4, 1, U) GVU(F, 32, 1, 4, 1, U)
#define GVU_ALL(U) GVU_SET(T_F32, U) GVU_SET(T_BF16, U) GVU_SET(T_Q8_0, U) GVU_SET(T_Q4_0, U) GVU_SET(T_Q5_0, U) GVU_SET(T_IQ4_NL, U) GVU_SET(T_IQ4_XS, U) \
                  GVU_SET(T_Q2_0, U) GVU_SET(T_IQ3_S, U) GVU_SET(T_Q3_K, U) GVU_SET(T_Q4_K, U) GVU_SET(T_Q5_K, U) GVU_SET(T_Q6_K, U)
GVU_ALL(2)
