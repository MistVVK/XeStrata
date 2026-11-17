// The recurrent layer's core over a prompt chunk of C rows (included by kernels.cu after gdn_core.cuh): the arithmetic
// of k_gdn_core, row for row and in the same order, without its speculative snapshots, in three launches instead of one
// per MAXT rows:
//   k_gdn_pf_conv: conv + silu for every row and channel at once; q and k L2-normed per row (into xc, rows of ch)
//   k_gdn_pf_hist: the chunk's last three raw rows into the history the next window reads
//   k_gdn_pf_rec:  the delta rule, each block (value head, GDN_QC columns) walking the rows in order with the state in
//                  registers, rows staged GDN_PF_TT at a time; the outputs (into y32) and their square sums per block
//   k_gdn_pf_norm: the gated RMS norm, in place in y32
// ctl holds the chunk's groups of MAXT rows as (position, tag) pairs from ctl[2]: the history is read at the first
// group's parity and written at the parity the window after the last group reads.
namespace {

constexpr int GDN_PF_TT = 32;

__global__ void __launch_bounds__(HS) k_gdn_pf_conv(const float* __restrict__ qkvz, const float* __restrict__ hist,
                                                    const float* __restrict__ convw, const int* __restrict__ ctl,
                                                    float* __restrict__ xc, int nk, int nv) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x, c = h * HS + i;
    const int ch = (2 * nk + nv) * HS, row_stride = ch + nv * HS;
    const float* hin = hist + (size_t) (((const unsigned*) ctl)[3] & 1u) * 3 * ch;
    float win[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int r = t - 3 + k;
        win[k] = r >= 0 ? qkvz[(size_t) r * row_stride + c] : hin[(size_t) (3 + r) * ch + c];
    }
    const float w0 = convw[c * 4], w1 = convw[c * 4 + 1], w2 = convw[c * 4 + 2], w3 = convw[c * 4 + 3];
    const float x = w0 * win[0] + w1 * win[1] + w2 * win[2] + w3 * win[3];
    float y = x / (1.0f + __expf(-x));
    if (h < 2 * nk) {   // q or k: the L2 norm over the head's HS channels, summed as k_gdn_core sums it
        __shared__ float red[HS / 32];
        const float ss = wsum(y * y);
        if ((i & 31) == 0) red[i >> 5] = ss;
        __syncthreads();
        float sq = 0.f;
#pragma unroll
        for (int w = 0; w < HS / 32; ++w) sq += red[w];
        y = y * rsqrtf(sq + 1e-6f);
    }
    xc[(size_t) t * ch + c] = y;
}

__global__ void k_gdn_pf_hist(const float* __restrict__ qkvz, float* __restrict__ hist, const int* __restrict__ ctl, int ng,
                              int C, int nk, int nv) {
    const int ch = (2 * nk + nv) * HS, row_stride = ch + nv * HS;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const unsigned* tags = (const unsigned*) ctl;
    const float* hin = hist + (size_t) (tags[3] & 1u) * 3 * ch;
    float* hout = hist + (size_t) ((tags[3 + 2 * (ng - 1)] & 1u) ^ 1u) * 3 * ch;
    float v[3];   // read before writing: hin and hout can be the same buffer
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int r = C - 3 + k;
        v[k] = r >= 0 ? qkvz[(size_t) r * row_stride + c] : hin[(size_t) (3 + r) * ch + c];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) hout[(size_t) k * ch + c] = v[k];
}

// threads per column: each walks HS / GDN_PF_PARTS rows of the column's state (more threads, a shorter chain per row
// than k_gdn_core's four)
constexpr int GDN_PF_PARTS = 8, GDN_PF_RP = HS / GDN_PF_PARTS, GDN_PF_THREADS = GDN_QC * GDN_PF_PARTS;
static_assert(GDN_PF_RP % 4 == 0, "rows per part in fours");

__global__ void __launch_bounds__(GDN_PF_THREADS) k_gdn_pf_rec(const float* __restrict__ xc, const float* __restrict__ ab,
                                                              const float* __restrict__ dt, const float* __restrict__ a,
                                                              float* __restrict__ S, float* __restrict__ y32, float* __restrict__ sq,
                                                              int nk, int nv, const int* __restrict__ vmap, int C) {
    const int lv = blockIdx.x / GDN_Q, qq = blockIdx.x % GDN_Q, kh = vmap[lv] & 255, tid = threadIdx.x;
    const int ch = (2 * nk + nv) * HS;
    __shared__ float xs[2][GDN_PF_TT][GDN_PF_RP][GDN_PF_PARTS];   // q, k: element i at [i % RP][i / RP]
    __shared__ float vs[GDN_PF_TT][GDN_QC];
    __shared__ float os[GDN_PF_TT][GDN_QC];
    __shared__ float gates[GDN_PF_TT][2];
    __shared__ float red[GDN_PF_THREADS / 32][GDN_PF_TT];
    const int warp = tid >> 5, lane = tid & 31, cl = tid / GDN_PF_PARTS, part = tid % GDN_PF_PARTS, col = qq * GDN_QC + cl;
    float* st = S + (size_t) lv * HS * HS + (size_t) part * GDN_PF_RP * HS + col;
    float s[GDN_PF_RP];
#pragma unroll
    for (int r = 0; r < GDN_PF_RP; ++r) s[r] = st[(size_t) r * HS];
    const float scale = rsqrtf((float) HS);
    for (int t0 = 0; t0 < C; t0 += GDN_PF_TT) {
        const int n = min(GDN_PF_TT, C - t0);
        __syncthreads();
        for (int e = tid; e < n * HS; e += GDN_PF_THREADS) {
            const int tt = e / HS, i = e % HS;
            const float* row = xc + (size_t) (t0 + tt) * ch;
            xs[0][tt][i % GDN_PF_RP][i / GDN_PF_RP] = row[kh * HS + i];
            xs[1][tt][i % GDN_PF_RP][i / GDN_PF_RP] = row[nk * HS + kh * HS + i];
        }
        for (int e = tid; e < n * GDN_QC; e += GDN_PF_THREADS) {
            const int tt = e / GDN_QC, j = e % GDN_QC;
            vs[tt][j] = xc[(size_t) (t0 + tt) * ch + 2 * nk * HS + lv * HS + qq * GDN_QC + j];
        }
        if (tid < n) {
            const float* abt = ab + (size_t) (t0 + tid) * 2 * nv;
            const float x = abt[lv] + dt[lv];
            gates[tid][0] = __expf((x > 20.0f ? x : log1pf(__expf(x))) * a[lv]);
            gates[tid][1] = 1.0f / (1.0f + __expf(-abt[nv + lv]));
        }
        __syncthreads();
#pragma unroll 1
        for (int t = 0; t < n; ++t) {
            const float gv = gates[t][0], bv = gates[t][1];
            float kvd0 = 0.f, kvd1 = 0.f, kvd2 = 0.f, kvd3 = 0.f;
#pragma unroll
            for (int r = 0; r < GDN_PF_RP; r += 4) {
                kvd0 += s[r] * xs[1][t][r][part];
                kvd1 += s[r + 1] * xs[1][t][r + 1][part];
                kvd2 += s[r + 2] * xs[1][t][r + 2][part];
                kvd3 += s[r + 3] * xs[1][t][r + 3][part];
            }
            float kvd = (kvd0 + kvd1) + (kvd2 + kvd3);
#pragma unroll
            for (int o = 1; o < GDN_PF_PARTS; o <<= 1) kvd += __shfl_xor_sync(0xffffffffu, kvd, o);
            const float delta = (vs[t][cl] - gv * kvd) * bv;
            float at0 = 0.f, at1 = 0.f, at2 = 0.f, at3 = 0.f;
#pragma unroll
            for (int r = 0; r < GDN_PF_RP; r += 4) {
                s[r] = gv * s[r] + xs[1][t][r][part] * delta;
                s[r + 1] = gv * s[r + 1] + xs[1][t][r + 1][part] * delta;
                s[r + 2] = gv * s[r + 2] + xs[1][t][r + 2][part] * delta;
                s[r + 3] = gv * s[r + 3] + xs[1][t][r + 3][part] * delta;
                at0 += s[r] * xs[0][t][r][part];
                at1 += s[r + 1] * xs[0][t][r + 1][part];
                at2 += s[r + 2] * xs[0][t][r + 2][part];
                at3 += s[r + 3] * xs[0][t][r + 3][part];
            }
            float at = (at0 + at1) + (at2 + at3);
#pragma unroll
            for (int o = 1; o < GDN_PF_PARTS; o <<= 1) at += __shfl_xor_sync(0xffffffffu, at, o);
            if (part == 0) os[t][cl] = at * scale;
        }
        __syncthreads();
        // the outputs and this block's square sums per row (the part-0 lanes hold a column's output, as in k_gdn_core)
        for (int t = 0; t < n; ++t) {
            const float v = part == 0 ? os[t][cl] : 0.f;
            if (part == 0) y32[(size_t) (t0 + t) * nv * HS + lv * HS + col] = v;
            const float ss = wsum(v * v);
            if (lane == 0) red[warp][t] = ss;
        }
        __syncthreads();
        if (tid < n) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < GDN_PF_THREADS / 32; ++w) v += red[w][tid];
            sq[((size_t) (t0 + tid) * nv + lv) * GDN_Q + qq] = v;
        }
    }
#pragma unroll
    for (int r = 0; r < GDN_PF_RP; ++r) st[(size_t) r * HS] = s[r];
}

// ---------------------------------------------------------------- the delta rule in sub-chunks (FNX_PF_GDN_CHUNK, default)
// The same recurrence, S_i = g_i S_{i-1} + k_i u_i^T with u_i = b_i (v_i - g_i S_{i-1}^T k_i) and o_i = S_i^T q_i / sqrt(HS),
// GCH rows at a time. Within a sub-chunk, with G_i the decay since its start (the product of g up to i, kept as a sum
// of logs) and S0 the state before it:
//   u_i = b_i (v_i - G_i S0^T k_i) - sum_{j<i} b_i (G_i / G_j) (k_i . k_j) u_j,   so U = T R, T = (I + A)^{-1},
//   R_i = b_i (v_i - G_i S0^T k_i), A_ij = b_i (G_i / G_j) (k_i . k_j) for j < i;
//   o_t = G_t S0^T q_t + sum_{i<=t} (G_t / G_i) (q_t . k_i) u_i = G_t S0^T q_t + (P U)_t;
//   S_end = G_last S0 + sum_i (G_last / G_i) k_i u_i^T.
// k_gdn_ch_prep makes what does not depend on the state for every sub-chunk and head at once (T, P, G, the decays to the
// end, b); k_gdn_ch_rec walks the sub-chunks in order per (value head, GCB columns), the state's columns in shared
// memory. The sums come in another order than k_gdn_core's, row by row (fp32 either way).
#if FNX_HIP
// gfx12 has 64 KB of LDS a block: k_gdn_ch_rec takes 16 state columns (58 KB) in 128 threads, twice the blocks a head; every output's sum
// runs in the same order as with 32 columns, so the results are bit-identical to the CUDA build's layout
constexpr int GCH = 32, GCB = 16, GCH_THREADS = 256, GCR_THREADS = 128, GCH_REC_BLOCKS_PER_HEAD = HS / GCB;
#else
constexpr int GCH = 32, GCB = 32, GCH_THREADS = 256, GCR_THREADS = 256, GCH_REC_BLOCKS_PER_HEAD = HS / GCB;
#endif
constexpr int GCR_CG = GCB / 4;   // column groups of 4 in a block's columns
// the decays from sums of up to GCH logs: --use_fast_math makes expf __expf, whose error grows with the argument (2 +
// 1.16 |x| ulp), so these go through double, which fast math leaves exact
__device__ __forceinline__ float gch_exp(float x) { return (float) exp((double) x); }
constexpr int GCH_PREP_FLOATS = 2 * GCH * GCH + 3 * GCH + 1;   // per sub-chunk and head: T, P, G, decay to end, b, G_last
static_assert(GCR_THREADS == GCH * GCB / 8 * 2 && GCR_THREADS == HS / 4 * (GCB / 4), "k_gdn_ch_rec's thread tiles");

__global__ void __launch_bounds__(GCH_THREADS) k_gdn_ch_prep(const float* __restrict__ xc, const float* __restrict__ ab,
                                                            const float* __restrict__ dt, const float* __restrict__ a,
                                                            const int* __restrict__ vmap, int nk, int nv, int C,
                                                            float* __restrict__ out) {
    const int sc = blockIdx.x, lv = blockIdx.y, kh = vmap[lv] & 255, tid = threadIdx.x, t0 = sc * GCH;
    const int n = min(GCH, C - t0), ch = (2 * nk + nv) * HS;
    __shared__ float ks[GCH][HS + 1], qs[GCH][HS + 1], kk[GCH][GCH + 1], pp[GCH][GCH + 1], tt[GCH][GCH + 1];
    __shared__ float lg[GCH], lgam[GCH], bb[GCH];
    for (int e = tid; e < GCH * HS; e += GCH_THREADS) {
        const int i = e / HS, r = e % HS;
        const float* row = xc + (size_t) (t0 + i) * ch;
        qs[i][r] = i < n ? row[kh * HS + r] : 0.f;
        ks[i][r] = i < n ? row[nk * HS + kh * HS + r] : 0.f;
    }
    if (tid < GCH) {   // rows past the chunk: g = 1, b = 0 (no effect)
        float l = 0.f, b = 0.f;
        if (tid < n) {
            const float* abt = ab + (size_t) (t0 + tid) * 2 * nv;
            const float x = abt[lv] + dt[lv];
            l = (x > 20.0f ? x : log1pf(__expf(x))) * a[lv];
            b = 1.0f / (1.0f + __expf(-abt[nv + lv]));
        }
        lg[tid] = l, bb[tid] = b;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int i = 0; i < GCH; ++i) lgam[i] = s += lg[i];
    }
    {   // k_i . k_j and q_i . k_j, each thread a 2 x 2 corner of both (the masks below keep j < i and j <= i)
        static_assert(GCH_THREADS == GCH / 2 * (GCH / 2), "a 2 x 2 corner a thread");
        const int i0 = (tid / (GCH / 2)) * 2, j0 = (tid % (GCH / 2)) * 2;
        float k00 = 0.f, k01 = 0.f, k10 = 0.f, k11 = 0.f, q00 = 0.f, q01 = 0.f, q10 = 0.f, q11 = 0.f;
#pragma unroll 4
        for (int r = 0; r < HS; ++r) {
            const float a0 = ks[i0][r], a1 = ks[i0 + 1][r], b0 = qs[i0][r], b1 = qs[i0 + 1][r], c0 = ks[j0][r], c1 = ks[j0 + 1][r];
            k00 = fmaf(a0, c0, k00), k01 = fmaf(a0, c1, k01), k10 = fmaf(a1, c0, k10), k11 = fmaf(a1, c1, k11);
            q00 = fmaf(b0, c0, q00), q01 = fmaf(b0, c1, q01), q10 = fmaf(b1, c0, q10), q11 = fmaf(b1, c1, q11);
        }
        kk[i0][j0] = k00, kk[i0][j0 + 1] = k01, kk[i0 + 1][j0] = k10, kk[i0 + 1][j0 + 1] = k11;
        pp[i0][j0] = q00, pp[i0][j0 + 1] = q01, pp[i0 + 1][j0] = q10, pp[i0 + 1][j0 + 1] = q11;
    }
    __syncthreads();
    for (int e = tid; e < GCH * GCH; e += GCH_THREADS) {
        const int i = e / GCH, j = e % GCH;
        const float dec = gch_exp(lgam[i] - lgam[j]);
        kk[i][j] = j < i ? bb[i] * dec * kk[i][j] : 0.f;   // A
        pp[i][j] = j <= i ? dec * pp[i][j] : 0.f;          // P
    }
    __syncthreads();
    // T = (I + A)^{-1}, lower unit-triangular: row i = e_i - sum_{j<i} A_ij row j (a warp, a column each)
    if (tid < GCH) {
        const int m = tid;
        for (int i = 0; i < GCH; ++i) {
            float v = i == m ? 1.f : 0.f;
            for (int j = m; j < i; ++j) v = fmaf(-kk[i][j], tt[j][m], v);
            tt[i][m] = m <= i ? v : 0.f;
            __syncwarp();
        }
    }
    __syncthreads();
    float* o = out + ((size_t) sc * nv + lv) * GCH_PREP_FLOATS;
    for (int e = tid; e < GCH * GCH; e += GCH_THREADS) {
        o[e] = tt[e / GCH][e % GCH];
        o[GCH * GCH + e] = pp[e / GCH][e % GCH];
    }
    if (tid < GCH) {
        o[2 * GCH * GCH + tid] = gch_exp(lgam[tid]);
        o[2 * GCH * GCH + GCH + tid] = gch_exp(lgam[GCH - 1] - lgam[tid]);
        o[2 * GCH * GCH + 2 * GCH + tid] = bb[tid];
    }
    if (tid == 0) o[2 * GCH * GCH + 3 * GCH] = gch_exp(lgam[GCH - 1]);
}

constexpr size_t GCH_REC_SMEM = sizeof(float) * (2 * GCH * (HS + 1) + HS * GCB + 4 * GCH * GCB + GCH_PREP_FLOATS);
// a thread's share of a sub-chunk's inputs, fetched into registers while the sub-chunk before computes
constexpr int GCH_KQ_PER = GCH * HS / GCR_THREADS, GCH_V_PER = GCH * GCB / GCR_THREADS,
              GCH_PRE_PER = (GCH_PREP_FLOATS + GCR_THREADS - 1) / GCR_THREADS;
static_assert(GCH * HS % GCR_THREADS == 0 && GCH * GCB % GCR_THREADS == 0, "even shares");

__global__ void __launch_bounds__(GCR_THREADS) k_gdn_ch_rec(const float* __restrict__ xc, const float* __restrict__ prep,
                                                           float* __restrict__ S, float* __restrict__ y32, int nk, int nv,
                                                           const int* __restrict__ vmap, int C) {
    const int lv = blockIdx.x / GCH_REC_BLOCKS_PER_HEAD, c0 = (blockIdx.x % GCH_REC_BLOCKS_PER_HEAD) * GCB;
    const int kh = vmap[lv] & 255, tid = threadIdx.x, ch = (2 * nk + nv) * HS;
    extern __shared__ float gsm[];
    float(*ks)[HS + 1] = reinterpret_cast<float(*)[HS + 1]>(gsm);
    float(*qs)[HS + 1] = reinterpret_cast<float(*)[HS + 1]>(gsm + GCH * (HS + 1));
    float* ss = gsm + 2 * GCH * (HS + 1);   // [HS][GCB]: the state's columns c0.. (rows: the key dimension)
    float* vv = ss + HS * GCB;              // [GCH][GCB] each: v; R; Y; U
    float* rr = vv + GCH * GCB;
    float* yy = rr + GCH * GCB;
    float* uu = yy + GCH * GCB;
    float* pre = uu + GCH * GCB;            // the sub-chunk's T, P, G, decay to end, b, G_last
    float* sst = S + (size_t) lv * HS * HS;
    for (int e = tid; e < HS * GCB; e += GCR_THREADS) ss[e] = sst[(size_t) (e / GCB) * HS + c0 + e % GCB];
    const float scale = rsqrtf((float) HS);
    const int nsub = (C + GCH - 1) / GCH;
    float kq[2][GCH_KQ_PER], vr[GCH_V_PER], pr[GCH_PRE_PER];
    auto fetch = [&](int sc) {
        const int t0 = sc * GCH, n = min(GCH, C - t0);
#pragma unroll
        for (int j = 0; j < GCH_KQ_PER; ++j) {
            const int e = tid + j * GCR_THREADS, i = e / HS, r = e % HS;
            const float* row = xc + (size_t) (t0 + i) * ch;
            kq[0][j] = i < n ? row[kh * HS + r] : 0.f;
            kq[1][j] = i < n ? row[nk * HS + kh * HS + r] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < GCH_V_PER; ++j) {
            const int e = tid + j * GCR_THREADS, i = e / GCB, c = e % GCB;
            vr[j] = i < n ? xc[(size_t) (t0 + i) * ch + 2 * nk * HS + lv * HS + c0 + c] : 0.f;
        }
        const float* pg = prep + ((size_t) sc * nv + lv) * GCH_PREP_FLOATS;
#pragma unroll
        for (int j = 0; j < GCH_PRE_PER; ++j) {
            const int e = tid + j * GCR_THREADS;
            pr[j] = e < GCH_PREP_FLOATS ? pg[e] : 0.f;
        }
    };
    fetch(0);
    for (int sc = 0; sc < nsub; ++sc) {
        const int t0 = sc * GCH, n = min(GCH, C - t0);
        __syncthreads();   // the last sub-chunk's reads are done
#pragma unroll
        for (int j = 0; j < GCH_KQ_PER; ++j) {
            const int e = tid + j * GCR_THREADS;
            qs[e / HS][e % HS] = kq[0][j];
            ks[e / HS][e % HS] = kq[1][j];
        }
#pragma unroll
        for (int j = 0; j < GCH_V_PER; ++j) vv[tid + j * GCR_THREADS] = vr[j];
#pragma unroll
        for (int j = 0; j < GCH_PRE_PER; ++j)
            if (tid + j * GCR_THREADS < GCH_PREP_FLOATS) pre[tid + j * GCR_THREADS] = pr[j];
        __syncthreads();
        if (sc + 1 < nsub) fetch(sc + 1);   // in flight while this one computes
        const float* T = pre;
        const float* P = pre + GCH * GCH;
        const float* G = pre + 2 * GCH * GCH;
        const float* dec = G + GCH;
        const float* B = dec + GCH;
        const float glast = B[GCH];
        // W = K S, Y = Q S: each thread 2 rows x 4 columns of both over half of the key dimension (half 1's partial
        // sums through rr / yy, half 0 adds them); then R = b (v - G W)
        {
            const int half = tid / (GCR_THREADS / 2), t7 = tid % (GCR_THREADS / 2), i0 = (t7 / GCR_CG) * 2, cc = (t7 % GCR_CG) * 4;
            float w[2][4] = {}, y[2][4] = {};
#pragma unroll 4
            for (int r = half * (HS / 2); r < (half + 1) * (HS / 2); ++r) {
                const float4 s = *reinterpret_cast<const float4*>(ss + r * GCB + cc);
                const float sv[4] = {s.x, s.y, s.z, s.w};
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    const float k = ks[i0 + a][r], q = qs[i0 + a][r];
#pragma unroll
                    for (int b = 0; b < 4; ++b) w[a][b] = fmaf(k, sv[b], w[a][b]), y[a][b] = fmaf(q, sv[b], y[a][b]);
                }
            }
            if (half) {
#pragma unroll
                for (int a = 0; a < 2; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) rr[(i0 + a) * GCB + cc + b] = w[a][b], yy[(i0 + a) * GCB + cc + b] = y[a][b];
            }
            __syncthreads();
            if (!half) {
#pragma unroll
                for (int a = 0; a < 2; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int e = (i0 + a) * GCB + cc + b;
                        rr[e] = B[i0 + a] * (vv[e] - G[i0 + a] * (w[a][b] + rr[e]));
                        yy[e] = y[a][b] + yy[e];
                    }
            }
        }
        __syncthreads();
        // U = T R (T lower-triangular): each thread column tid % GCB, rows tid / GCB + 8 h
        {
            const int c = tid % GCB;
#pragma unroll
            for (int h = 0; h < GCH * GCB / GCR_THREADS; ++h) {
                const int i = tid / GCB + h * (GCR_THREADS / GCB);
                float u = 0.f;
                for (int m = 0; m <= i; ++m) u = fmaf(T[i * GCH + m], rr[m * GCB + c], u);
                uu[i * GCB + c] = u;
            }
        }
        __syncthreads();
        // o = scale (G Y + P U)
        {
            const int c = tid % GCB;
#pragma unroll
            for (int h = 0; h < GCH * GCB / GCR_THREADS; ++h) {
                const int t = tid / GCB + h * (GCR_THREADS / GCB);
                float o = G[t] * yy[t * GCB + c];
                for (int i = 0; i <= t; ++i) o = fmaf(P[t * GCH + i], uu[i * GCB + c], o);
                if (t < n) y32[(size_t) (t0 + t) * nv * HS + lv * HS + c0 + c] = o * scale;
            }
        }
        // S = G_last S + K^T (decay U): each thread 4 rows x 4 columns of the state
        {
            const int r0 = (tid / GCR_CG) * 4, cc = (tid % GCR_CG) * 4;
            float s[4][4];
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const float4 v = *reinterpret_cast<const float4*>(ss + (r0 + a) * GCB + cc);
                s[a][0] = glast * v.x, s[a][1] = glast * v.y, s[a][2] = glast * v.z, s[a][3] = glast * v.w;
            }
#pragma unroll 4
            for (int i = 0; i < GCH; ++i) {
                const float4 u = *reinterpret_cast<const float4*>(uu + i * GCB + cc);
                const float uv[4] = {u.x, u.y, u.z, u.w}, d = dec[i];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const float kd = d * ks[i][r0 + a];
#pragma unroll
                    for (int b = 0; b < 4; ++b) s[a][b] = fmaf(kd, uv[b], s[a][b]);
                }
            }
#pragma unroll
            for (int a = 0; a < 4; ++a)
                *reinterpret_cast<float4*>(ss + (r0 + a) * GCB + cc) = make_float4(s[a][0], s[a][1], s[a][2], s[a][3]);
        }
    }
    __syncthreads();
    for (int e = tid; e < HS * GCB; e += GCR_THREADS) sst[(size_t) (e / GCB) * HS + c0 + e % GCB] = ss[e];
}

// the gated RMS norm with the row's square sum made here (the sub-chunk path keeps no per-block sums)
__global__ void __launch_bounds__(HS) k_gdn_pf_norm_own(const float* __restrict__ qkvz, const float* __restrict__ nw,
                                                        float* __restrict__ y32, int nk, int nv, float eps) {
    const int t = blockIdx.x, lv = blockIdx.y, c = threadIdx.x;
    const int ch = (2 * nk + nv) * HS, row_stride = ch + nv * HS;
    float* y = y32 + (size_t) t * nv * HS + lv * HS + c;
    const float v = *y;
    __shared__ float red[HS / 32];
    const float ss = wsum(v * v);
    if ((c & 31) == 0) red[c >> 5] = ss;
    __syncthreads();
    float sq = 0.f;
#pragma unroll
    for (int w = 0; w < HS / 32; ++w) sq += red[w];
    const float z = qkvz[(size_t) t * row_stride + ch + lv * HS + c];
#if FNX_27B   // the 27B gates with silu(z), Flash-Next with sigmoid(z)
    *y = v * rsqrtf(sq / (float) HS + eps) * nw[c] * z / (1.0f + __expf(-z));
#else
    *y = v * rsqrtf(sq / (float) HS + eps) * nw[c] / (1.0f + __expf(-z));
#endif
}

__global__ void __launch_bounds__(HS) k_gdn_pf_norm(const float* __restrict__ qkvz, const float* __restrict__ nw,
                                                    const float* __restrict__ sq, float* __restrict__ y32, int nk, int nv,
                                                    float eps) {
    const int t = blockIdx.x, lv = blockIdx.y, c = threadIdx.x;
    const int ch = (2 * nk + nv) * HS, row_stride = ch + nv * HS;
    const float* sqv = sq + ((size_t) t * nv + lv) * GDN_Q;
    const float ms = (sqv[0] + sqv[1] + sqv[2] + sqv[3]) / (float) HS;
    float* y = y32 + (size_t) t * nv * HS + lv * HS + c;
    const float z = qkvz[(size_t) t * row_stride + ch + lv * HS + c];
#if FNX_27B   // the 27B gates with silu(z), Flash-Next with sigmoid(z)
    *y = *y * rsqrtf(ms + eps) * nw[c] * z / (1.0f + __expf(-z));
#else
    *y = *y * rsqrtf(ms + eps) * nw[c] / (1.0f + __expf(-z));
#endif
}

}  // namespace
