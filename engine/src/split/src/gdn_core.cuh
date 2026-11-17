// The recurrent layer's core in one launch (included by kernels.cu): GDN_Q blocks per local value head, each owning
// GDN_QC columns of the head's state (value dims; every column's delta rule is independent of the others).
// qkvz rows: [q (nk heads) | k (nk heads) | v (nv heads) | z (nv heads)] x HS; ab rows: [alpha (nv) | beta (nv)].
// hist: [2 parities][3][ch]; this window reads parity tag & 1 and writes the other (the head's q/k channels are read by
// its three value heads' blocks, so only block 0 of the head with s == 0 writes them, into the other buffer).
// S: [nv][HS rows (key dim)][HS cols (value dim)], cols contiguous.
// Thread (col, part) owns rows [part*32, +32) of its column; the four parts of a column are adjacent lanes, so the sums
// over rows are two shuffles. The gated RMS norm spans the head's HS columns: each block leaves its outputs and their
// square sums in scratch, and the head's last block to finish norms, gates and quantizes the head.
// Output: y = rms_norm(o) * w * sigmoid(z) (the 27B: silu(z)), as q8_1 rows of nv*HS (blocks lv*4 .. lv*4+3 of row t).
// For a speculative window: snap [MAXT - 1][nv][HS][HS] gets the state after each row but the last, hsnap [MAXT][3][ch]
// the history the window leaves when only its first n rows are kept (variant n - 1, for n < T).
namespace {

constexpr int GDN_Q = 4, GDN_QC = HS / GDN_Q, GDN_THREADS = GDN_QC * 4;

template <int T>
__global__ void __launch_bounds__(GDN_THREADS) k_gdn_core(const float* __restrict__ qkvz, const float* __restrict__ ab,
                                                         float* __restrict__ hist, const float* __restrict__ convw,
                                                         const float* __restrict__ dt, const float* __restrict__ a,
                                                         const float* __restrict__ nw, float* __restrict__ S,
                                                         block_q8_1* __restrict__ yq, const int* __restrict__ ctl,
                                                         float* __restrict__ scratch, float* __restrict__ snap,
                                                         float* __restrict__ hsnap, int nk, int nv, const int* __restrict__ vmap,
                                                         float eps,
                                                         int8_t* __restrict__ tq, float* __restrict__ td,
                                                         float* __restrict__ y32, GdnLanes gl) {
    const int lv = blockIdx.x / GDN_Q, qq = blockIdx.x % GDN_Q, vm = vmap[lv], kh = vm & 255, s0 = vm >> 8, tid = threadIdx.x;
    const int ch = (2 * nk + nv) * HS, row_stride = ch + nv * HS;
    // with lanes: blockIdx.y is a lane's segment of T rows from window row r0 (row0 null: one row, r0 = the segment), on
    // its lane's state, history and ctl; its rows of every buffer, its snapshots at r0 (rows r0 .. r0 + T - 2 of them)
    const int seg = (int) blockIdx.y;
    const int r0 = gl.S ? (gl.row0 ? gl.row0[seg] : seg) : 0;
    if (gl.S) {
        S = gl.S[seg];
        hist = gl.hist[seg];
        ctl = gl.ctl[seg];
        if (gl.snap) {
            snap = gl.snap + (size_t) r0 * nv * HS * HS;
            hsnap = gl.hsnap + (size_t) r0 * 3 * ch;
        }
        qkvz += (size_t) r0 * row_stride;
        ab += (size_t) r0 * 2 * nv;
        yq += (size_t) r0 * (nv * HS / 32);
        if (tq) tq += (size_t) r0 * nv * HS, td += (size_t) r0 * (nv * HS / 32);
        if (y32) y32 += (size_t) r0 * nv * HS;
    }
    const unsigned parity = ((const unsigned*) ctl)[1] & 1u;
    const float* hin = hist + (size_t) parity * 3 * ch;
    float* hout = hist + (size_t) (parity ^ 1u) * 3 * ch;
    // scratch: o [MAXT][nv * HS], square sums [nv][MAXT][GDN_Q], a counter per row and head ([MAXT][nv])
    float* o_all = scratch + (size_t) r0 * nv * HS;
    float* sq_all = scratch + (size_t) MAXT * nv * HS + (size_t) r0 * GDN_Q;
    unsigned* counters = (unsigned*) (scratch + (size_t) MAXT * nv * HS + (size_t) nv * MAXT * GDN_Q) + (size_t) r0 * nv;
    // q, k after conv, silu and the L2 norm; element i at [i % 32][i / 32] (row r of part p: [r][p])
    __shared__ float xs[2][T][32][4];
    __shared__ float vs[T][GDN_QC];        // this block's v columns after conv and silu
    __shared__ float red[GDN_THREADS / 32][2 * T];
    __shared__ float gates[T][2];          // per row: the decay exp(softplus(alpha + dt) * a), beta = sigmoid(b)
    if (tid < T) {
        const float* abt = ab + (size_t) tid * 2 * nv;
        const float x = abt[lv] + dt[lv];
#if FNX_27B   // the decay through double, as gdn_prompt's (gch_exp): __expf's error grows with |x| and compounds in the
              // state row after row (the 27B's edit1 read through windows: KL 1.1e-6 against llama.cpp, the bar 1.5e-7)
        gates[tid][0] = (float) exp((double) ((x > 20.0f ? x : (float) log1p(exp((double) x))) * a[lv]));
#else
        gates[tid][0] = __expf((x > 20.0f ? x : log1pf(__expf(x))) * a[lv]);
#endif
        gates[tid][1] = 1.0f / (1.0f + __expf(-abt[nv + lv]));
    }
    // ---- conv + silu: q and k channels (all HS of each), this block's v channels
#pragma unroll
    for (int e = tid; e < 2 * HS + GDN_QC; e += GDN_THREADS) {
        const int part = e < HS ? 0 : e < 2 * HS ? 1 : 2, i = part < 2 ? e - part * HS : e - 2 * HS;
        const int c = part == 0 ? kh * HS + i : part == 1 ? nk * HS + kh * HS + i : 2 * nk * HS + lv * HS + qq * GDN_QC + i;
        float win[3 + T];
#pragma unroll
        for (int k = 0; k < 3; ++k) win[k] = hin[k * ch + c];
#pragma unroll
        for (int t = 0; t < T; ++t) win[3 + t] = qkvz[(size_t) t * row_stride + c];
        const float w0 = convw[c * 4], w1 = convw[c * 4 + 1], w2 = convw[c * 4 + 2], w3 = convw[c * 4 + 3];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float x = w0 * win[t] + w1 * win[t + 1] + w2 * win[t + 2] + w3 * win[t + 3];
            const float y = x / (1.0f + __expf(-x));
            if (part < 2) xs[part][t][i % 32][i / 32] = y;
            else vs[t][i] = y;
        }
        if (part == 2 || (s0 && qq == 0)) {
#pragma unroll
            for (int k = 0; k < 3; ++k) hout[k * ch + c] = win[T + k];
#pragma unroll
            for (int n = 1; n < T; ++n)
#pragma unroll
                for (int k = 0; k < 3; ++k) hsnap[((size_t) (n - 1) * 3 + k) * ch + c] = win[n + k];
        }
    }
    __syncthreads();
    // ---- L2 norms of q and k: x / sqrt(sum x^2 + 1e-6), per row (HS = GDN_THREADS: one element of each per thread)
    static_assert(HS == GDN_THREADS, "one q and one k element per thread");
    const int warp = tid >> 5, lane = tid & 31;
    {
        float qv[T], kv[T], ss[2 * T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            qv[t] = xs[0][t][tid % 32][tid / 32];
            kv[t] = xs[1][t][tid % 32][tid / 32];
            ss[t] = wsum(qv[t] * qv[t]);
            ss[T + t] = wsum(kv[t] * kv[t]);
        }
        if (lane == 0)
#pragma unroll
            for (int i = 0; i < 2 * T; ++i) red[warp][i] = ss[i];
        __syncthreads();
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float sq = 0.f, sk = 0.f;
#pragma unroll
            for (int w = 0; w < GDN_THREADS / 32; ++w) {
                sq += red[w][t];
                sk += red[w][T + t];
            }
            xs[0][t][tid % 32][tid / 32] = qv[t] * rsqrtf(sq + 1e-6f);
            xs[1][t][tid % 32][tid / 32] = kv[t] * rsqrtf(sk + 1e-6f);
        }
    }
    __syncthreads();
    // ---- the delta rule over the rows
    const int cl = tid >> 2, part = tid & 3, col = qq * GDN_QC + cl;
    float* st = S + (size_t) lv * HS * HS + (size_t) part * 32 * HS + col;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = st[(size_t) r * HS];
    const float scale = rsqrtf((float) HS);
    float ov[T];
    // rolled: unrolled over T the loop is several KB of code, fetched cold at every launch (the instruction cache does
    // not keep a kernel between layers)
#pragma unroll 1
    for (int t = 0; t < T; ++t) {
        const float gv = gates[t][0], bv = gates[t][1];
        float kvd0 = 0.f, kvd1 = 0.f, kvd2 = 0.f, kvd3 = 0.f;
#pragma unroll
        for (int r = 0; r < 32; r += 4) {
            kvd0 += s[r] * xs[1][t][r][part];
            kvd1 += s[r + 1] * xs[1][t][r + 1][part];
            kvd2 += s[r + 2] * xs[1][t][r + 2][part];
            kvd3 += s[r + 3] * xs[1][t][r + 3][part];
        }
        float kvd = (kvd0 + kvd1) + (kvd2 + kvd3);
        kvd += __shfl_xor_sync(0xffffffffu, kvd, 1);
        kvd += __shfl_xor_sync(0xffffffffu, kvd, 2);
        const float delta = (vs[t][cl] - gv * kvd) * bv;
        float at0 = 0.f, at1 = 0.f, at2 = 0.f, at3 = 0.f;
#pragma unroll
        for (int r = 0; r < 32; r += 4) {
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
        at += __shfl_xor_sync(0xffffffffu, at, 1);
        at += __shfl_xor_sync(0xffffffffu, at, 2);
        ov[t] = at * scale;
        if (t < T - 1) {
            float* sp = snap + (size_t) t * nv * HS * HS + (st - S);
#pragma unroll
            for (int r = 0; r < 32; ++r) sp[(size_t) r * HS] = s[r];
        }
    }
#pragma unroll
    for (int r = 0; r < 32; ++r) st[(size_t) r * HS] = s[r];
    // ---- this block's outputs and their square sums per row (the part-0 lanes hold the column's output)
    {
        float ss[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float v = part == 0 ? ov[t] : 0.f;
            if (part == 0) o_all[(size_t) t * nv * HS + lv * HS + col] = v;
            ss[t] = wsum(v * v);
        }
        if (lane == 0)
#pragma unroll
            for (int t = 0; t < T; ++t) red[warp][t] = ss[t];
        __syncthreads();
        if (tid < T) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < GDN_THREADS / 32; ++w) v += red[w][tid];
            sq_all[((size_t) lv * MAXT + tid) * GDN_Q + qq] = v;
        }
    }
    // ---- the head's last block: gated RMS norm and q8_1 for all HS columns
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(&counters[lv], 1u) == GDN_Q - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int w = warp; w < 4 * T; w += GDN_THREADS / 32) {
        const int t = w / 4, b = w % 4, c = b * 32 + lane;
        const volatile float* sqv = sq_all + ((size_t) lv * MAXT + t) * GDN_Q;
        const float ms = (sqv[0] + sqv[1] + sqv[2] + sqv[3]) / (float) HS;
        const float v = ((const volatile float*) o_all)[(size_t) t * nv * HS + lv * HS + c];
        const float z = qkvz[(size_t) t * row_stride + ch + lv * HS + c];
#if FNX_27B   // the 27B gates with silu(z), Flash-Next with sigmoid(z)
        const float xi = v * rsqrtf(ms + eps) * nw[c] * z / (1.0f + __expf(-z));
#else
        const float xi = v * rsqrtf(ms + eps) * nw[c] / (1.0f + __expf(-z));
#endif
        if (y32) y32[(size_t) t * nv * HS + lv * HS + c] = xi;
        const float amax = wmax(fabsf(xi)), sm = wsum(xi);
        const float d = amax / 127.0f;
        block_q8_1& q = yq[(size_t) t * (nv * HS / 32) + lv * 4 + b];
        const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
        q.qs[lane] = qv;
        if (lane == 0) q.ds = make_half2(__float2half(d), __float2half(sm));
        if (tq) {
            tq[(size_t) t * nv * HS + lv * HS + b * 32 + lane] = qv;
            if (lane == 0) td[(size_t) t * (nv * HS / 32) + lv * 4 + b] = __half2float(__float2half(d));
        }
    }
    if (tid == 0) counters[lv] = 0u;
}

}  // namespace
