// The sampler's candidates (included by kernels.cu): per verify row, this card's slice of the head's logits with the
// row's penalties applied (llama.cpp's penalties stage, as Strata applies it: repeat multiplies a non-positive logit
// and divides a positive one, then frequency x count and presence once), then its TOPK_N highest, ordered by logit
// (descending) and token id (ascending) on ties. A block per row: a radix select of the TOPK_N-th highest key, then
// everything at or above it sorted. More than TOPK_CAP at or above it (a tie at the threshold that wide: a row of equal
// logits) takes a second pass that picks the same TOPK_N, so the result never depends on how many tie. A NaN or inf
// among the logits sets err bit 16.
namespace {

constexpr int TOPK_THREADS = 1024, TOPK_CAP = 256;

__device__ __forceinline__ unsigned tk_key(float v) {   // order-preserving: a larger float, a larger key
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(TOPK_THREADS) k_topk_rows(float* __restrict__ logits, int n, int v0, const int* __restrict__ pen_tok,
                                                           const int* __restrict__ pen_cnt, const int* __restrict__ pen_n, int pen_stride,
                                                           const float* __restrict__ pen_par, float* __restrict__ out_v,
                                                           int* __restrict__ out_i, unsigned* err) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    float* L = logits + (size_t) row * n;
    // a NaN or inf among the head's logits (before the penalties, which a request sets): err bit 16
    bool bad = false;
    for (int j = tid; j < n; j += TOPK_THREADS) bad = bad || !isfinite(L[j]);
    if (bad) atomicOr(err, 16u);
    __syncthreads();   // every logit checked before any penalty changes one
    // the row's penalties on this card's tokens (each token once in the list)
    const float repeat = pen_par[row * 3], freq = pen_par[row * 3 + 1], present = pen_par[row * 3 + 2];   // the row's penalty parameters
    for (int i = tid; i < pen_n[row]; i += TOPK_THREADS) {
        const int tok = pen_tok[row * pen_stride + i] - v0, cnt = pen_cnt[row * pen_stride + i];
        if (tok < 0 || tok >= n || cnt <= 0) continue;
        float l = L[tok];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[tok] = l - (float) cnt * freq - present;
    }
    __syncthreads();
    // the TOPK_N-th highest key: four passes of 8 bits
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    unsigned prefix = 0, mask = 0, want = TOPK_N;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += TOPK_THREADS) hist[b] = 0;
        __syncthreads();
        for (int jb = 0; jb < n; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const unsigned k = j < n ? tk_key(L[j]) : 0u;
            const bool in = j < n && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#else
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
#endif
            if (in && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned) __popc(peers));
        }
        __syncthreads();
        if (tid == 0) {
            unsigned acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= want) break;
                acc += hist[b];
            }
            prefix_s = prefix | ((unsigned) b << shift);
            want_s = want - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // everything at or above the threshold, then sorted: (key desc, id asc) as one 64-bit key, descending
    __shared__ unsigned long long cand[TOPK_CAP];
    __shared__ unsigned n_cand;
    for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS) cand[i] = 0ull;
    if (tid == 0) n_cand = 0;
    __syncthreads();
    for (int j = tid; j < n; j += TOPK_THREADS) {
        const unsigned k = tk_key(L[j]);
        if (k >= prefix) {
            const unsigned s = atomicAdd(&n_cand, 1u);
            if (s < TOPK_CAP) cand[s] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
    }
    __syncthreads();
    const bool wide = n_cand > TOPK_CAP;
    __syncthreads();   // every thread has read n_cand before the wide pass resets it
    if (wide) {   // block-uniform. The first TOPK_N of the sorted set are every key above the threshold
        // (TOPK_N - want of them) and the `want` lowest ids at it: those exactly, the ties taken in id order a chunk at a
        // time (a block-wide rank from each warp's ballot), stopping once `want` are in
        __shared__ unsigned warp_n[TOPK_THREADS / 32];
        const int warp = tid >> 5;
        for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS) cand[i] = 0ull;
        if (tid == 0) n_cand = 0;
        __syncthreads();
        for (int j = tid; j < n; j += TOPK_THREADS) {
            const unsigned k = tk_key(L[j]);
            if (k > prefix) cand[atomicAdd(&n_cand, 1u)] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
        const unsigned base = TOPK_N - want;
        unsigned taken = 0;
        for (int jb = 0; jb < n && taken < want; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const bool eq = j < n && tk_key(L[j]) == prefix;
            const unsigned bal = __ballot_sync(0xffffffffu, eq);
            if (lane == 0) warp_n[warp] = (unsigned) __popc(bal);
            __syncthreads();
            unsigned before = (unsigned) __popc(bal & ((1u << lane) - 1u)), total = 0;
            for (int w = 0; w < TOPK_THREADS / 32; ++w) {
                before += w < warp ? warp_n[w] : 0u;
                total += warp_n[w];
            }
            if (eq && taken + before < want) cand[base + taken + before] = ((unsigned long long) prefix << 32) | (0xffffffffu - (unsigned) j);
            taken += total;
            __syncthreads();   // warp_n read by every thread before the next chunk writes it
        }
        __syncthreads();
    }
    for (int size = 2; size <= TOPK_CAP; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < TOPK_CAP) {
                const int o = tid ^ stride;
                if (o > tid) {
                    const bool desc = (tid & size) == 0;
                    const unsigned long long a = cand[tid], b = cand[o];
                    if (desc ? a < b : a > b) {
                        cand[tid] = b;
                        cand[o] = a;
                    }
                }
            }
            __syncthreads();
        }
    if (tid < TOPK_N) {
        const unsigned long long c = cand[tid];
        const int j = (int) (0xffffffffu - (unsigned) (c & 0xffffffffu));
        out_i[row * TOPK_N + tid] = v0 + j;
        out_v[row * TOPK_N + tid] = L[j];
    }
}

}  // namespace
