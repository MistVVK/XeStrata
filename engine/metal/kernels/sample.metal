// metal/kernels/sample.metal - greedy sampling over the LM head's logits: the largest logit, ties to the lowest
// token id (NaN never wins).  Two stages: argmax_part (grid (parts, n_tok), 256 threads) writes one (value, id)
// pair per threadgroup; argmax_final (grid (n_tok), 256 threads) reduces them in order.

struct ArgArgs {
    uint n_vocab;
    uint parts;
};

static inline void better(thread float& bv, thread int& bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

static inline void tg_argmax(thread float& bv, thread int& bi, threadgroup float* sv, threadgroup int* si, ushort sg,
                             ushort lane) {
    for (ushort m = 16; m > 0; m >>= 1) {
        const float ov = simd_shuffle_xor(bv, m);
        const int oi = simd_shuffle_xor(bi, m);
        better(bv, bi, ov, oi);
    }
    if (lane == 0) { sv[sg] = bv; si[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = -INFINITY;
    bi = 0x7fffffff;
    for (int k = 0; k < 8; ++k) better(bv, bi, sv[k], si[k]);
}

kernel void argmax_part(device const float* logits [[buffer(0)]], device float* pv [[buffer(1)]],
                        device int* pi [[buffer(2)]], constant ArgArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup int si[8];
    const uint t = tg.y, per = (a.n_vocab + a.parts - 1) / a.parts;
    const uint b0 = tg.x * per, b1 = min(b0 + per, a.n_vocab);
    device const float* l = logits + t * a.n_vocab;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint i = b0 + tid; i < b1; i += 256) {
        const float v = l[i];
        if (!isnan(v)) better(bv, bi, v, int(i));
    }
    tg_argmax(bv, bi, sv, si, sg, lane);
    if (tid == 0) { pv[t * a.parts + tg.x] = bv; pi[t * a.parts + tg.x] = bi; }
}

kernel void argmax_final(device const float* pv [[buffer(0)]], device const int* pi [[buffer(1)]],
                         device int* out [[buffer(2)]], constant ArgArgs& a [[buffer(3)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup int si[8];
    const uint t = tg.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint k = tid; k < a.parts; k += 256) better(bv, bi, pv[t * a.parts + k], pi[t * a.parts + k]);
    tg_argmax(bv, bi, sv, si, sg, lane);
    if (tid == 0) out[t] = bi;
}

// ---------------------------------------------------------------------------------------------------------------
// The sampler chain (llama.cpp's order, as src/kernels/cuda/sampler.cu `sampler_kernel`, the reference):
//     logit bias -> penalties -> top_k -> top_p -> min_p -> temperature -> one Philox draw keyed (seed, position)
// Two stages.  samp_part: grid (parts, rows), 256 threads: a part's slice of the vocabulary (<= 4096 logits) goes to threadgroup
// memory with the bias and the penalties applied, and k rounds of a block argmax (ties to the lowest id; NaN and -inf never win)
// leave that part's k best.  samp_pick: grid (rows): the parts' k lists are merged by k more rounds (the same order), then one
// thread runs the chain over the k kept candidates and writes the token.  Metal has no double: the sums and the cumulative
// probabilities are float (the CUDA reference uses double); the draw is the same Philox 4x32-10 value as CUDA's for equal
// (seed, counter).  A row's counter is the position of the token it samples, so the text does not depend on how it was cut
// into windows.  greedy (flag 2): the first candidate of the penalised / biased logits.

constant uint SF_BIAS = 1u, SF_GREEDY = 2u, SF_PEN = 4u;

struct SampArgs {
    uint n_vocab, parts, k, hist_len, pen_n, min_keep, flags;
    float top_p, min_p, inv_t, rep, freq, pres;
    uint seed_lo, seed_hi, ctr_lo, ctr_hi;
};

static inline float philox_u(ulong seed, ulong ctr) {
    uint c0 = uint(ctr), c1 = uint(ctr >> 32), c2 = uint(seed), c3 = uint(seed >> 32);
    for (uint i = 0; i < 10; ++i) {
        const uint hi0 = mulhi(0x9E3779B9u, c0), hi1 = mulhi(0xBB67AE85u, c2);
        const uint lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return float(c0 >> 8) * (1.0f / 16777216.0f);
}

kernel void samp_part(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]], device const int* hist [[buffer(2)]],
                      device float* cv [[buffer(3)]], device int* ci [[buffer(4)]], constant SampArgs& a [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[4096];
    threadgroup float rv[8];
    threadgroup int ri[8];
    const uint row = tg.y, part = tg.x;
    const uint per = (a.n_vocab + a.parts - 1) / a.parts;
    const uint b0 = part * per, b1 = min(b0 + per, a.n_vocab);
    const uint cnt = b1 > b0 ? b1 - b0 : 0;
    device const float* l = logits + row * a.n_vocab;
    device const int* h = hist + row * a.hist_len + (a.hist_len - a.pen_n);   // the penalty window is the row's last pen_n tokens
    for (uint i = tid; i < 4096; i += 256) {
        float s = -INFINITY;
        if (i < cnt) {
            const uint v = b0 + i;
            s = l[v];
            if (a.flags & SF_BIAS) s += bias[v];
            if (a.flags & SF_PEN) {
                int c = 0;
                for (uint j = 0; j < a.pen_n; ++j) c += (h[j] == int(v)) ? 1 : 0;
                if (c > 0) {
                    if (s <= 0.0f) s *= a.rep; else s /= a.rep;
                    s -= float(c) * a.freq + a.pres;
                }
            }
            if (isnan(s)) s = -INFINITY;
        }
        sv[i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* ov = cv + (row * a.parts + part) * a.k;
    device int* oi = ci + (row * a.parts + part) * a.k;
    for (uint r = 0; r < a.k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (uint i = tid; i < cnt; i += 256) {
            const float s = sv[i];
            if (s > bv) { bv = s; bi = int(i); }
        }
        tg_argmax(bv, bi, rv, ri, sg, lane);
        if (tid == 0) {
            ov[r] = bv;
            oi[r] = bi < int(cnt) ? int(b0) + bi : 0x7fffffff;
            if (bi < int(cnt)) sv[bi] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void samp_pick(device const float* cv [[buffer(0)]], device const int* ci [[buffer(1)]], device int* out [[buffer(2)]],
                      constant SampArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float cvs[4096];
    threadgroup float rv[8];
    threadgroup int ri[8];
    threadgroup int sel_id[64];
    threadgroup float sel_lg[64];
    const uint row = tg.x, n = a.parts * a.k;
    device const int* cid = ci + row * n;   // the candidates' ids stay in device memory (the threadgroup holds the values only)
    for (uint i = tid; i < n; i += 256) cvs[i] = cv[row * n + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < a.k; ++r) {
        // 1. the largest value left
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (uint i = tid; i < n; i += 256) if (cvs[i] > bv) { bv = cvs[i]; bi = int(i); }
        tg_argmax(bv, bi, rv, ri, sg, lane);
        // 2. among the candidates holding it, the lowest id (the selection order: value descending, ties to the lowest id)
        int mn = 0x7fffffff;
        if (bv > -INFINITY)
            for (uint i = tid; i < n; i += 256) if (cvs[i] == bv) mn = min(mn, cid[i]);
        mn = simd_min(mn);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) ri[sg] = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < 8; ++k) mn = min(mn, ri[k]);
        if (bv > -INFINITY)
            for (uint i = tid; i < n; i += 256) if (cvs[i] == bv && cid[i] == mn) cvs[i] = -INFINITY;
        if (tid == 0) { sel_id[r] = (bv > -INFINITY && mn < int(a.n_vocab)) ? mn : 0; sel_lg[r] = bv; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0) return;
    const uint k = a.k;
    int pick = sel_id[0];
    if (!(a.flags & SF_GREEDY)) {
        uint n_keep = k;
        float mx = sel_lg[0];
        for (uint i = 1; i < k; ++i) mx = max(mx, sel_lg[i]);
        if (a.top_p < 1.0f) {
            float sum = 0.0f;
            for (uint i = 0; i < k; ++i) sum += exp(sel_lg[i] - mx);
            float cum = 0.0f;
            uint cut = k;
            for (uint i = 0; i < k; ++i) {
                cum += exp(sel_lg[i] - mx) / sum;
                if (cum >= a.top_p) { cut = i + 1; break; }
            }
            if (cut < a.min_keep) cut = a.min_keep < k ? a.min_keep : k;
            n_keep = cut;
        }
        if (a.min_p > 0.0f) {
            const float thresh = sel_lg[0] + log(a.min_p);
            for (uint i = 0; i < n_keep; ++i)
                if (sel_lg[i] < thresh) { n_keep = i; break; }
        }
        float smx = sel_lg[0] * a.inv_t;
        for (uint i = 1; i < n_keep; ++i) smx = max(smx, sel_lg[i] * a.inv_t);
        float sum = 0.0f;
        for (uint i = 0; i < n_keep; ++i) sum += exp(sel_lg[i] * a.inv_t - smx);
        pick = sel_id[n_keep - 1];
        const float u = philox_u(ulong(a.seed_lo) | (ulong(a.seed_hi) << 32), (ulong(a.ctr_lo) | (ulong(a.ctr_hi) << 32)) + ulong(row));
        float cum = 0.0f;
        for (uint i = 0; i < n_keep; ++i) {
            cum += exp(sel_lg[i] * a.inv_t - smx) / sum;
            if (u < cum) { pick = sel_id[i]; break; }
        }
    }
    out[row] = pick;
}
