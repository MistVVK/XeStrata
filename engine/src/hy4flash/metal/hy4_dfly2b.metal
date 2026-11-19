// hy4_dfly2b.metal - reduced-vocabulary draft chain of DFly2: candidate ids = the top-N of the base logits lm_head(final_norm(H)[j]); the chain scores only those lm_head rows with the corrected
// hidden state. A draft change can only change acceptance, never the output (the verify pass is the target). Compiled with hy4.metal (f2key is defined there).

// top-N ids of each logits row [n][V]: threshold search on the order-preserving key, then emit every id with key >= thr until N are written. grid n, 1024 threads
kernel void dfly_topn(device const float* logits [[buffer(0)]], constant int& V [[buffer(1)]], constant int& N [[buffer(2)]], device int* cand [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int red[32];
    threadgroup atomic_int cnt;
    threadgroup uint thr_s;
    device const float* lg = logits + (size_t) row * V;
    device int* out = cand + (size_t) row * N;
    uint lo = 0, hi = 0xFFFFFFFFu;
    for (int it = 0; it < 33 && lo < hi; ++it) {
        const uint mid = (uint) (lo + (((ulong) hi - lo + 1) >> 1));
        int c = 0;
        for (int i = tid; i < V; i += 1024) c += (f2key(lg[i]) >= mid) ? 1 : 0;
        c = simd_sum(c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & 31) == 0) red[tid >> 5] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tot = 0;
        for (int w = 0; w < 32; ++w) tot += red[w];
        if (tot >= N) lo = mid; else hi = mid - 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { thr_s = lo; atomic_store_explicit(&cnt, 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint thr = thr_s;
    for (int i = tid; i < V; i += 1024)
        if (f2key(lg[i]) >= thr) {
            const int s = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
            if (s < N) out[s] = i;
        }
}

// score[k] = lm_head row cand[k] . h   (lm_head S8_K [V][3072], a row = 12 super-blocks of 262 B); one simdgroup (32 threads) per candidate
kernel void dfly_cand_score(device const uchar* W [[buffer(0)]], device const int* cand [[buffer(1)]], device const half* h [[buffer(2)]], device float* score [[buffer(3)]], uint k [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]]) {
    device const uchar* row = W + (size_t) cand[k] * 12 * 262;
    float acc = 0;
    for (int sb = 0; sb < 12; ++sb) {
        device const uchar* blk = row + sb * 262;
        const float D = h2f_bits(ld16(blk));
        for (int i = lane; i < 256; i += 32) {
            const float eff = D * (float(blk[2 + (i >> 6)]) / 255.0f);
            acc += float(as_type<char>(blk[6 + i])) * eff * float(h[sb * 256 + i]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) score[k] = acc;
}

kernel void dfly_cand_argmax(device const float* score [[buffer(0)]], device const int* cand [[buffer(1)]], constant int& N [[buffer(2)]], device int* out [[buffer(3)]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = tid; i < N; i += 1024) {
        const int id = cand[i];
        if (score[i] > bv || (score[i] == bv && id < bi)) { bv = score[i]; bi = id; }
    }
    for (uint o = 16; o > 0; o >>= 1) {
        const float ov = simd_shuffle_xor(bv, o);
        const int oi = simd_shuffle_xor(bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b2 = sv[0];
        int i2 = si[0];
        for (uint q = 1; q < 32; ++q)
            if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
        *out = i2;
    }
}
