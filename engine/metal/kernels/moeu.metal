// metal/kernels/moeu.metal - moe.metal's decode gate/up and down kernels with the unit loop unrolled UF times: the weights of UF
// units are decoded (their loads issued) before the first is applied.  Each lane adds its units in the original order (u, u + LPR,
// ...), so the sums are bitwise moe.metal's; a lane waits one memory latency for UF units instead of one for each.  After moe.metal.
// Names: moe_guu_<type>_<LPR>_<NR>_<SG>_<UF>, moe_dnu_<type>_<LPR>_<NR>_<SG>_<UF>.

template <int F, int LPR, int NR, int NE, int UF>
static inline void gu_body_u(device const uchar* g0, device const uchar* u0, device const float* x, device const float* xs16,
                             device float* h, device const int* etok, int c, int row0, int li, constant MoeArgs& a,
                             threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    float ag[NR][NE], au[NR][NE];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) ag[r][t] = au[r][t] = 0.0f;
    const int U = int(a.n_embd) / UNIT;
    for (int u = li; u < U; u += LPR * UF) {
        float qg[UF][NR][UNIT], qu[UF][NR][UNIT];
        float4 cg[UF][NR], cu[UF][NR];
        for (int k = 0; k < UF; ++k)
            if (u + k * LPR < U)
                for (int r = 0; r < NR; ++r) {
                    Dec<F>::run(g0 + r * a.gu_row, u + k * LPR, qg[k][r], cg[k][r], tab);
                    Dec<F>::run(u0 + r * a.gu_row, u + k * LPR, qu[k][r], cu[k][r], tab);
                }
        for (int k = 0; k < UF; ++k)
            if (u + k * LPR < U)
                for (int t = 0; t < NE; ++t) {
                    const int tok = etok[t];
                    float xs[UNIT], xsum[2];
                    X<F>::load(x + tok * int(a.n_embd), xs16 + tok * int(a.n_embd / 16), u + k * LPR, xs, xsum);
                    for (int r = 0; r < NR; ++r) {
                        ag[r][t] += unit_combine<UNIT>(qg[k][r], cg[k][r], xs, xsum);
                        au[r][t] += unit_combine<UNIT>(qu[k][r], cu[k][r], xs, xsum);
                    }
                }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) {
            const float gv = lane_sum<LPR>(ag[r][t]), uv = lane_sum<LPR>(au[r][t]);
            if (li == 0) h[(c + t) * int(a.n_ff) + row0 + r] = gv / (1.0f + exp(-gv)) * uv;
        }
}

template <int F, int LPR, int NR, int SG, int UF>
kernel void moe_gu_u(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* h [[buffer(2)]], device const int* plan [[buffer(3)]],
                     constant MoeArgs& a [[buffer(4)]], device const float* xs16 [[buffer(5)]],
                     device const uint* smap [[buffer(6)]], device const uchar* W1 [[buffer(7)]],
                     device const uchar* W2 [[buffer(8)]], device const uchar* W3 [[buffer(9)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    if (moe_probe != 1) {
        load_tab<F>(tab, tid, SG * 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int g = int(tg.y);
    if (g >= plan[P_NG]) return;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const ulong ex = (ulong)plan[P_GEXP + g];
    const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
    device const int* etok = plan + P_ETOK + c;
    device const uchar* E = kArena ? arena_slot(smap[ex], W, W1, W2, W3) : W + ex * a.gu_stride;
    device const uchar* g0 = E + a.gate_off + (ulong)row0 * a.gu_row;
    device const uchar* u0 = E + a.up_off + (ulong)row0 * a.gu_row;
    switch (ne) {
        case 1: gu_body_u<F, LPR, NR, 1, UF>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        case 2: gu_body_u<F, LPR, NR, 2, UF>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        case 3: gu_body_u<F, LPR, NR, 3, UF>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        case 4: gu_body_u<F, LPR, NR, 4, UF>(g0, u0, x, xs16, h, etok, c, row0, li, a, tab); break;
        default:
            for (int k = 0; k < ne; k += 4) {
                const int m = min(4, ne - k);
                if (m == 4) gu_body_u<F, LPR, NR, 4, UF>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
                else if (m == 3) gu_body_u<F, LPR, NR, 3, UF>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
                else if (m == 2) gu_body_u<F, LPR, NR, 2, UF>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
                else gu_body_u<F, LPR, NR, 1, UF>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
            }
    }
}

template <int F, int LPR, int NR, int NE, int UF>
static inline void dn_body_u(device const uchar* d0, device const float* h, device const float* hs16, device float* ye,
                             device const int* plan, int c, int row0, int li, constant MoeArgs& a, threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    float ac[NR][NE];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) ac[r][t] = 0.0f;
    const int U = int(a.n_ff) / UNIT;
    for (int u = li; u < U; u += LPR * UF) {
        float q[UF][NR][UNIT];
        float4 sc[UF][NR];
        for (int k = 0; k < UF; ++k)
            if (u + k * LPR < U)
                for (int r = 0; r < NR; ++r) Dec<F>::run(d0 + r * a.d_row, u + k * LPR, q[k][r], sc[k][r], tab);
        for (int k = 0; k < UF; ++k)
            if (u + k * LPR < U)
                for (int t = 0; t < NE; ++t) {
                    float xs[UNIT], xsum[2];
                    X<F>::load(h + (c + t) * int(a.n_ff), hs16 + (c + t) * int(a.n_ff / 16), u + k * LPR, xs, xsum);
                    for (int r = 0; r < NR; ++r) ac[r][t] += unit_combine<UNIT>(q[k][r], sc[k][r], xs, xsum);
                }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) {
            const float v = lane_sum<LPR>(ac[r][t]);
            if (li == 0) ye[(c + t) * int(a.n_embd) + row0 + r] = as_type<float>(plan[P_EW + c + t]) * v;
        }
}

template <int F, int LPR, int NR, int SG, int UF>
kernel void moe_down_u(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                       device float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                       constant MoeArgs& a [[buffer(4)]], device const float* hs16 [[buffer(5)]],
                       device const uint* smap [[buffer(6)]], device const uchar* W1 [[buffer(7)]],
                       device const uchar* W2 [[buffer(8)]], device const uchar* W3 [[buffer(9)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int g = int(tg.y);
    if (g >= plan[P_NG]) return;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const ulong ex = (ulong)plan[P_GEXP + g];
    const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
    device const uchar* d0 = (kArena ? arena_slot(smap[ex], W, W1, W2, W3) : W + ex * a.d_stride) + a.down_off + (ulong)row0 * a.d_row;
    switch (ne) {
        case 1: dn_body_u<F, LPR, NR, 1, UF>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        case 2: dn_body_u<F, LPR, NR, 2, UF>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        case 3: dn_body_u<F, LPR, NR, 3, UF>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        case 4: dn_body_u<F, LPR, NR, 4, UF>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
        default:
            for (int k = 0; k < ne; k += 4) {
                const int m = min(4, ne - k);
                if (m == 4) dn_body_u<F, LPR, NR, 4, UF>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                else if (m == 3) dn_body_u<F, LPR, NR, 3, UF>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                else if (m == 2) dn_body_u<F, LPR, NR, 2, UF>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                else dn_body_u<F, LPR, NR, 1, UF>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
            }
    }
}

#define GUU_(F, L, R, S, U) template [[host_name("moe_guu_" #F "_" #L "_" #R "_" #S "_" #U)]] kernel decltype(moe_gu_u<F, L, R, S, U>) moe_gu_u<F, L, R, S, U>;
#define GUU(F, L, R, S, U) GUU_(F, L, R, S, U)
#define DNU_(F, L, R, S, U) template [[host_name("moe_dnu_" #F "_" #L "_" #R "_" #S "_" #U)]] kernel decltype(moe_down_u<F, L, R, S, U>) moe_down_u<F, L, R, S, U>;
#define DNU(F, L, R, S, U) DNU_(F, L, R, S, U)
// the tiles the Q2_0 and IQ3_S packs use (window.cpp / token_chain.cpp pick 32,1,4 for gate/up; down 8,1,4 for Q2_0 and 4,2,4 otherwise)
#define GUU_ALL(U) GUU(T_Q2_0, 32, 1, 4, U) GUU(T_IQ3_XXS, 32, 1, 4, U) GUU(T_IQ2_S, 32, 1, 4, U) GUU(T_IQ3_S, 32, 1, 4, U) \
                   GUU(T_IQ4_XS, 32, 1, 4, U)
#define DNU_ALL(U) DNU(T_Q2_0, 8, 1, 4, U) DNU(T_Q2_0R, 8, 1, 4, U) DNU(T_IQ4_NL, 4, 2, 4, U) DNU(T_Q8_0, 4, 2, 4, U) DNU(T_Q5_1, 4, 2, 4, U) DNU(T_Q5_0, 4, 2, 4, U)
GUU_ALL(2) DNU_ALL(2)
