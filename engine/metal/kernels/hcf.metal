// metal/kernels/hcf.metal - the hyper-connection read's gr_norm folded into the kernel that makes the sublayer output
// (decode windows; after gemv.metal, gr.metal and glue.metal).
//
// gr_norm did, per stream c and token t:   R'[c][d] = fma(bo[d], g_c, R[c][d])   (written back in place)
//                                          xn[c][d] = (R'[c][d] * w_norm[c][d]) * rsqrt(mean(R'[c]^2) + eps)
// with g_c = 2 / (1 + exp(-inj[c] / hc)).  Here the producer of bo (the mixer's output GEMV, the MoE combine) does the
// first line itself for the outputs it computes (4 streams a value), and the LAST threadgroup of its dispatch to finish
// (the router_route pattern: a device counter, reset by the last) does the second line for every stream and token.  The
// reduction is gr_norm's: thread (tid mod 256) sums its 10 elements d = tid + 256 j with fma in j order, simd_sum over
// the 32 lanes, the 8 simdgroup partials added in order.  A producer threadgroup of 128 threads runs two of gr_norm's
// virtual simdgroups a real one.  All values are bitwise gr_norm's after the producer's own dispatch; one dispatch (a
// barrier, ~7 us in the decode chain) is gone per read.  n_embd 2560, hc 4.

struct HcfArgs {
    uint n_embd;
    uint hc;
    float eps;
    uint pad;
};

static inline void hcf_apply(device float* R, device const float* inj, constant HcfArgs& h, int t, int row, float v) {
#ifdef HCF_NO_APPLY
    return;
#endif
    const int n = int(h.n_embd), hc = int(h.hc);
    for (int c = 0; c < hc; ++c) {
        const float g = 2.0f / (1.0f + exp(-inj[t * hc + c] / float(hc)));
        device float* r = R + (t * hc + c) * n + row;
        *r = fma(v, g, *r);
    }
}

// R' (every stream, token) -> xn.  NTH threads (128 or 256); part: 4 * 8 floats of threadgroup memory
template <int NTH>
static inline void hcf_norm(device const float* R, device const float* w_norm, device float* xn, constant HcfArgs& h, int ntok,
                            ushort tid, ushort sg, ushort lane, threadgroup float* part) {
    constexpr int NV = 256 / NTH, NJ = 10;
    const int n = int(h.n_embd);
    for (int t = 0; t < ntok; ++t) {
        float v[4][NV][NJ];
        for (int c = 0; c < 4; ++c) {
            device const float* r = R + (t * 4 + c) * n;
            for (int m = 0; m < NV; ++m) {
                const int vt = int(tid) + NTH * m;
                float ss = 0.0f;
                for (int j = 0; j < NJ; ++j) {
                    const float x = r[vt + 256 * j];
                    v[c][m][j] = x;
                    ss = fma(x, x, ss);
                }
                ss = simd_sum(ss);
                if (lane == 0) part[c * 8 + int(sg) + (NTH / 32) * m] = ss;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < 4; ++c) {
            float tot = 0.0f;
            for (int i = 0; i < 8; ++i) tot += part[c * 8 + i];
            const float rs = rsqrt(tot / float(n) + h.eps);
            device float* o = xn + (t * 4 + c) * n;
            device const float* w = w_norm + c * n;
            for (int m = 0; m < NV; ++m) {
                const int vt = int(tid) + NTH * m;
                for (int j = 0; j < NJ; ++j) {
                    const int d = vt + 256 * j;
                    o[d] = v[c][m][j] * w[d] * rs;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// the last-arrival protocol (router_route's): true in the one threadgroup that runs the follow-on work
static inline bool hcf_last(device atomic_uint* counter, uint ngx, ushort tid, threadgroup uint* last) {
#if !defined(HCF_NOFENCE1) && !defined(HCF_T0FENCE1)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
    threadgroup_barrier(mem_flags::mem_device);
#ifdef HCF_T0FENCE1
    if (tid == 0) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
    if (tid == 0) *last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == ngx - 1 ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool l = *last != 0;
#ifndef HCF_NOFENCE2
    if (l) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
    return l;
}

// gemv (gemv.metal's arithmetic and tile, written as the original) with bo = y applied to R and the last threadgroup forming xn.
// Buffers 0..4 as gemv; R, inj, w_norm, xn, counter, HcfArgs.
template <int F, int LPR, int NR, int SG, int KS>
kernel void gemv_hcf(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                     device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]], constant GemvArgs& a [[buffer(4)]],
                     device float* R [[buffer(5)]], device const float* inj [[buffer(6)]],
                     device const float* w_norm [[buffer(7)]], device float* xn [[buffer(8)]],
                     device atomic_uint* counter [[buffer(9)]], constant HcfArgs& h [[buffer(10)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint3 ngv [[threadgroups_per_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RS = SG / KS, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup float red[KS > 1 ? KS * RS * RG * NR * 8 : 1];
    threadgroup float part[32];
    threadgroup uint last;
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
    for (int u = u0 + li; u < u1; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(rp[r], u, q[r], sc[r], tab);
        float xs[UNIT], xsum[2];
        for (int t = 0; t < NTOK; ++t) {
            X<F>::load(xp + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    constexpr int RPT = RS * RG * NR;
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (KS == 1) {
                if (li == 0 && row0 + r < int(a.n_out)) {
                    y[t * int(a.y_stride) + row0 + r] = v;
                    hcf_apply(R, inj, h, t, row0 + r, v);
                }
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
                    if (row0 + r < int(a.n_out)) {
                        y[t * int(a.y_stride) + row0 + r] = v;
                        hcf_apply(R, inj, h, t, row0 + r, v);
                    }
                }
    }
#ifdef HCF_NO_LAST
    return;
#endif
    if (!hcf_last(counter, ngv.x, tid, &last)) return;
#ifndef HCF_NO_NORM
    hcf_norm<SG * 32>(R, w_norm, xn, h, int(a.n_tok), tid, sg, lane, part);
#endif
    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

// moe_combine_sh (glue.metal) with the same follow-on.  One thread an output; threadgroups of 256.
kernel void moe_combine_hcf(device const float* ye [[buffer(0)]], device const int* plan [[buffer(1)]],
                            device const float* ysh [[buffer(2)]], device const float* sg [[buffer(3)]],
                            device float* bo [[buffer(4)]], constant GlueArgs& a [[buffer(5)]],
                            device float* R [[buffer(6)]], device const float* inj [[buffer(7)]],
                            device const float* w_norm [[buffer(8)]], device float* xn [[buffer(9)]],
                            device atomic_uint* counter [[buffer(10)]], constant HcfArgs& h [[buffer(11)]],
                            uint3 tg [[threadgroup_position_in_grid]], uint3 ngv [[threadgroups_per_grid]],
                            ushort sgi [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                            ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[32];
    threadgroup uint last;
    const uint gid = tg.x * 256 + tid;
    const uint t = gid / a.n_embd, k = gid % a.n_embd;
    if (t < a.n_tok) {
        float s = 0.0f;
        for (uint j = 0; j < a.topk; ++j) s += ye[plan[P_TOKENT + t * a.topk + j] * int(a.n_embd) + k];
        const float gs = 1.0f / (1.0f + exp(-sg[t]));
        const float v = fma(gs, ysh[gid], s);
        bo[gid] = v;
        hcf_apply(R, inj, h, int(t), int(k), v);
    }
#ifdef HCF_NO_LAST
    return;
#endif
    if (!hcf_last(counter, ngv.x, tid, &last)) return;
#ifndef HCF_NO_NORM
    hcf_norm<256>(R, w_norm, xn, h, int(a.n_tok), tid, sgi, lane, part);
#endif
    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

#define GVH_(F) template [[host_name("gemvh_" #F)]] kernel decltype(gemv_hcf<F, 32, 1, 8, 2>) gemv_hcf<F, 32, 1, 8, 2>;
#define GVH(F) GVH_(F)
GVH(T_F32) GVH(T_BF16) GVH(T_Q8_0) GVH(T_Q4_0) GVH(T_Q5_0) GVH(T_IQ4_NL) GVH(T_IQ4_XS) GVH(T_Q2_0) GVH(T_IQ3_S) GVH(T_Q3_K)
GVH(T_Q4_K) GVH(T_Q5_K) GVH(T_Q6_K)

// timing probe: an empty kernel (the cost of one dependent dispatch)
kernel void noop_k(device float* x [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    if (x[0] == 12345.678f) x[t + 1] = 1.0f;
}
