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

// ================================================================ Tree-8 Speculative Verification Kernels ===============================================================
// Shared 76-byte TreeArgs descriptor (bit-identical to sequential 1-token decode along any tree branch).
struct TreeArgs {
    int n;              // 1..8 tree nodes
    uchar depth[8];     // logical position of node t is p0 + depth[t]
    uchar anc[8][8];    // anc[t][d] = tree node index of node t's ancestor at depth d (0 <= d <= depth[t])
};

// KV cache write for tree nodes: writes to physical slot (p0 + t) [% ring], rotates k_pe at logical position p0 + depth[t]. grid T, 256 threads
kernel void mla_write_kv_tree(device const float* kv_a [[buffer(0)]], device const float* norm_w [[buffer(1)]], device half* cache [[buffer(2)]], constant int& p0 [[buffer(3)]],
                              constant int& ring [[buffer(4)]], constant float& theta [[buffer(5)]], constant float& eps [[buffer(6)]], constant TreeArgs& tr [[buffer(7)]],
                              uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* row = kv_a + (size_t) t * MLA_ROW;
    const float v0 = row[2 * tid], v1 = row[2 * tid + 1];
    const float ss = tg_sum(v0 * v0 + v1 * v1, sh, tid, 256);
    const float inv = 1.0f / sqrt(ss / float(MLA_KV) + eps);
    const int phys = p0 + int(t);
    const int slot = ring > 0 ? phys % ring : phys;
    const int lpos = p0 + int(tr.depth[t]);
    device half* dst = cache + (size_t) slot * MLA_ROW;
    dst[2 * tid] = half(v0 * inv * norm_w[2 * tid]);
    dst[2 * tid + 1] = half(v1 * inv * norm_w[2 * tid + 1]);
    if (tid < 32) {
        float o0, o1;
        rope_pair(row[MLA_KV + 2 * tid], row[MLA_KV + 2 * tid + 1], lpos, int(tid), theta, o0, o1);
        dst[MLA_KV + 2 * tid] = half(o0);
        dst[MLA_KV + 2 * tid + 1] = half(o1);
    }
}

// Overwrites q_pe RoPE dims [512..575] of qa[t][h] using logical position p0 + depth[t]. grid (NH, T), 32 threads
kernel void mla_q_rope_tree(device const float* q_b [[buffer(0)]], device half* qa [[buffer(1)]], constant int& NH [[buffer(2)]], constant int& p0 [[buffer(3)]],
                            constant float& theta [[buffer(4)]], constant TreeArgs& tr [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const int h = int(tgp.x), t = int(tgp.y);
    device const float* qp = q_b + ((size_t) t * NH + h) * 256 + 192 + 2 * int(i);
    float o0, o1;
    rope_pair(qp[0], qp[1], p0 + int(tr.depth[t]), int(i), theta, o0, o1);
    *((device half2*) (qa + ((size_t) t * NH + h) * MLA_ROW + MLA_KV + 2 * int(i))) = half2(half(o0), half(o1));
}

// Tree-aware split-K MLA attention partial (exact same math & ws[t][h][s][514] output as mla_attend_part in hy4_split.metal). grid (NH, S, T), 256 threads
kernel void mla_attend_part_tree(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device float* ws [[buffer(3)]],
                                 constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]], constant int& S [[buffer(7)]],
                                 constant TreeArgs& tr [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = int(tgp.x), s = int(tgp.y), t = int(tgp.z);
    const uint sg = tid >> 5, lane = tid & 31;
    threadgroup float sm[8], sl[8];
    threadgroup float sacc[8][512];
    device const half2* qrow = (device const half2*) (qa + ((size_t) t * a.NH + h) * MLA_ROW);
    float qv[18];
    for (int i = 0; i < 9; ++i) {
        half2 q = qrow[lane + 32 * i];
        qv[2 * i] = float(q.x);
        qv[2 * i + 1] = float(q.y);
    }
    const int pos = a.p0 + int(tr.depth[t]);
    int nk, lo = 0;
    if (a.mode == 0) nk = pos + 1;
    else if (a.mode == 1) {
        lo = max(0, pos - a.window + 1);
        nk = pos - lo + 1;
    } else nk = nkeys[t];
    const int per = (nk + S - 1) / S;
    const int j_lo = s * per, j_hi = min(nk, j_lo + per);
    float m = (s == 0 && sg == 0) ? sinks[h] : -INFINITY;
    float l = (s == 0 && sg == 0) ? 1.0f : 0.0f;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int j = j_lo + (int) sg; j < j_hi; j += 8) {
        int lp;
        if (a.mode == 0) lp = j;
        else if (a.mode == 1) lp = lo + j;
        else lp = keys[(size_t) t * a.kmax + j];
        const int phys = (lp >= a.p0) ? (a.p0 + int(tr.anc[t][(lp - a.p0) & 7])) : lp;
        const int slot = a.ring > 0 ? (phys % a.ring) : phys;
        device const half2* kr = (device const half2*) (cache + (size_t) slot * MLA_ROW);
        float d = 0.f;
        for (int i = 0; i < 9; ++i) {
            half2 kk = kr[lane + 32 * i];
            d += qv[2 * i] * float(kk.x) + qv[2 * i + 1] * float(kk.y);
        }
        d = simd_sum(d) * a.scale;
        float nm = max(m, d);
        float corr = exp(m - nm);
        float p = exp(d - nm);
        l = l * corr + p;
        for (int i = 0; i < 8; ++i) {
            half2 v = kr[lane + 32 * i];
            acc[2 * i] = acc[2 * i] * corr + p * float(v.x);
            acc[2 * i + 1] = acc[2 * i + 1] * corr + p * float(v.y);
        }
        m = nm;
    }
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    for (int i = 0; i < 8; ++i) {
        sacc[sg][2 * (lane + 32 * i)] = acc[2 * i];
        sacc[sg][2 * (lane + 32 * i) + 1] = acc[2 * i + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = sm[0];
    for (int q = 1; q < 8; ++q) M = max(M, sm[q]);
    device float* o = ws + (((size_t) t * a.NH + h) * S + s) * 514;
    if (M == -INFINITY) {
        if (tid == 0) { o[0] = -INFINITY; o[1] = 0.f; }
        o[2 + 2 * tid] = 0.f;
        o[2 + 2 * tid + 1] = 0.f;
        return;
    }
    float L = 0.f, o0 = 0.f, o1 = 0.f;
    for (int q = 0; q < 8; ++q) {
        float f = exp(sm[q] - M);
        L += sl[q] * f;
        o0 += sacc[q][2 * tid] * f;
        o1 += sacc[q][2 * tid + 1] * f;
    }
    if (tid == 0) { o[0] = M; o[1] = L; }
    o[2 + 2 * tid] = o0;
    o[2 + 2 * tid + 1] = o1;
}

// Tree-aware IndexPool: temporary per-node pooled keys for up to 2 groups g in [p0/4, grp_t). grid (T, 2), 128 threads
kernel void idx_pool_tree(device const half* ivg [[buffer(0)]], device const float* ape [[buffer(1)]], device const float* knw [[buffer(2)]], device half* tmp_pooled [[buffer(3)]],
                          constant int& p0 [[buffer(4)]], constant TreeArgs& tr [[buffer(5)]], constant float& theta [[buffer(6)]], constant float& eps [[buffer(7)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float sv[128];
    const int t = int(tgp.x), k = int(tgp.y);
    const int g0 = p0 >> 2;
    const int grp_t = (p0 + int(tr.depth[t])) >> 2;
    const int g = g0 + k;
    if (g >= grp_t) return;
    int sl[4];
    for (int j = 0; j < 4; ++j) {
        const int lp = 4 * g + j;
        sl[j] = lp < p0 ? lp : p0 + int(tr.anc[t][(lp - p0) & 7]);
    }
    float l[4], mx = -3.0e38f;
    for (int j = 0; j < 4; ++j) {
        l[j] = float(ivg[(size_t) sl[j] * 256 + 128 + d]) + ape[j * 128 + d];
        mx = max(mx, l[j]);
    }
    float s = 0.0f;
    for (int j = 0; j < 4; ++j) { l[j] = exp(l[j] - mx); s += l[j]; }
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) acc += (l[j] / s) * float(ivg[(size_t) sl[j] * 256 + d]);
    const float ss = tg_sum(acc * acc, sh, d, 128);
    const float inv = rsqrt(ss / 128.0f + eps);
    float v = acc * inv * knw[d];
    sv[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= 64) {
        const int i = int(d - 64) >> 1;
        float o0, o1;
        rope_pair(sv[64 + 2 * i], sv[64 + 2 * i + 1], 4 * g, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    tmp_pooled[(size_t) (t * 2 + k) * 128 + d] = half(v);
}

// Tree-aware IndexPool query RoPE at p0 + depth[t]. grid (T, 16), 128 threads
kernel void idx_q_rope_tree(device const float* iq [[buffer(0)]], device half* out [[buffer(1)]], constant int& p0 [[buffer(2)]], constant TreeArgs& tr [[buffer(3)]],
                            constant float& theta [[buffer(4)]], uint2 th [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    const int t = int(th.x), h = int(th.y);
    device const float* q = iq + (size_t) t * 2048 + h * 128;
    float v = q[d];
    if (d >= 64) {
        const int i = int(d - 64) >> 1;
        float o0, o1;
        rope_pair(q[64 + 2 * i], q[64 + 2 * i + 1], p0 + int(tr.depth[t]), i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    out[(size_t) t * 2048 + h * 128 + d] = half(v);
}

// Tree-aware IndexPool scorer: reads committed pooled[g] for g < p0/4 and tmp_pooled[t*2 + g - g0] for g >= p0/4. grid ((G+63)/64, T), 64 threads
kernel void idx_score_tree(device const half* iq [[buffer(0)]], device const float* hw [[buffer(1)]], device const half* pooled [[buffer(2)]], device const half* tmp_pooled [[buffer(3)]],
                           device float* score [[buffer(4)]], constant int& p0 [[buffer(5)]], constant TreeArgs& tr [[buffer(6)]], constant int& G [[buffer(7)]],
                           constant int& ldS [[buffer(8)]], constant float& hscale [[buffer(9)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sq[16 * 128];
    threadgroup float shw[16];
    const int t = int(tgp.y), g = int(tgp.x) * 64 + int(tid);
    for (uint i = tid; i < 16 * 128; i += 64) sq[i] = float(iq[(size_t) t * 2048 + i]);
    if (tid < 16) shw[tid] = hw[(size_t) t * 16 + tid] * hscale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g >= G) return;
    const int g0 = p0 >> 2;
    const int grp = (p0 + int(tr.depth[t])) >> 2;
    float res;
    if (g >= grp) {
        res = -INFINITY;
    } else {
        device const half* pr = (g < g0) ? (pooled + (size_t) g * 128) : (tmp_pooled + (size_t) (t * 2 + (g - g0)) * 128);
        res = 0.0f;
        for (int h = 0; h < 16; ++h) {
            float dsum = 0.0f;
            for (int k = 0; k < 128; ++k) dsum += sq[h * 128 + k] * float(pr[k]);
            res += max(dsum, 0.0f) * shw[h];
        }
    }
    score[(size_t) t * ldS + g] = res;
}

// Tree-aware IndexPool selector: emits logical positions 0..pos (pos = p0 + depth[t]) in deterministic increasing group order. grid T, 256 threads
kernel void idx_select_tree(device const float* score [[buffer(0)]], constant int& ldS [[buffer(1)]], constant int& p0 [[buffer(2)]], constant TreeArgs& tr [[buffer(3)]],
                            constant int& n_top [[buffer(4)]], constant int& kmax [[buffer(5)]], device int* keys [[buffer(6)]], device int* nkeys [[buffer(7)]],
                            uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int red[8];
    threadgroup atomic_int cnt_sel;
    threadgroup uint thr_s;
    threadgroup int n_gt_s;
    const int pos = p0 + int(tr.depth[t]);
    const int grp = pos >> 2;
    device int* kl = keys + (size_t) t * kmax;
    if (grp <= n_top) {
        for (int i = int(tid); i <= pos; i += 256) kl[i] = i;
        if (tid == 0) nkeys[t] = pos + 1;
        return;
    }
    device const float* sc = score + (size_t) t * ldS;
    uint lo = 0, hi = 0xFFFFFFFFu;
    for (int it = 0; it < 33 && lo < hi; ++it) {
        uint mid = (uint) (lo + (((ulong) hi - lo + 1) >> 1));
        int c = 0;
        for (int g = int(tid); g < grp; g += 256) c += (f2key(sc[g]) >= mid) ? 1 : 0;
        c = simd_sum(c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & 31) == 0) red[tid >> 5] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tot = 0;
        for (int w = 0; w < 8; ++w) tot += red[w];
        if (tot >= n_top) lo = mid; else hi = mid - 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { thr_s = lo; atomic_store_explicit(&cnt_sel, 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint thr = thr_s;
    int c = 0;
    for (int g = int(tid); g < grp; g += 256) c += (f2key(sc[g]) > thr) ? 1 : 0;
    c = simd_sum(c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) red[tid >> 5] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int tot = 0;
        for (int w = 0; w < 8; ++w) tot += red[w];
        n_gt_s = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int n_eq_need = n_top - n_gt_s;
    {
        const int per = (grp + 255) / 256;
        const int g0 = int(tid) * per, g1 = min(grp, g0 + per);
        int cnt = 0;
        for (int g = g0; g < g1; ++g) cnt += (f2key(sc[g]) > thr) ? 1 : 0;
        const int ex = simd_prefix_exclusive_sum(cnt);
        if ((tid & 31) == 31) red[tid >> 5] = ex + cnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int base = 0;
        for (uint w = 0; w < (tid >> 5); ++w) base += red[w];
        int sidx = base + ex;
        for (int g = g0; g < g1; ++g)
            if (f2key(sc[g]) > thr) {
                for (int j = 0; j < 4; ++j) kl[4 * sidx + j] = 4 * g + j;
                ++sidx;
            }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int s = n_gt_s, need = n_eq_need;
        for (int g = 0; g < grp && need > 0; ++g) {
            if (f2key(sc[g]) == thr) {
                for (int j = 0; j < 4; ++j) kl[4 * s + j] = 4 * g + j;
                ++s;
                --need;
            }
        }
        atomic_store_explicit(&cnt_sel, s, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int npool = atomic_load_explicit(&cnt_sel, memory_order_relaxed);
    const int nown = pos - 4 * grp + 1;
    for (int i = int(tid); i < nown; i += 256) kl[4 * npool + i] = 4 * grp + i;
    if (tid == 0) nkeys[t] = 4 * npool + nown;
}

// Post-verify in-place sequential KV compaction: tr.depth[k] = path[k] >= k. grid 1, 256 threads
kernel void mla_compact_kv(device half* cache [[buffer(0)]], constant int& p0 [[buffer(1)]], constant int& ring [[buffer(2)]], constant TreeArgs& tr [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]]) {
    for (int k = 0; k < tr.n; ++k) {
        const int src_n = int(tr.depth[k]);
        if (src_n != k) {
            const int s_slot = ring > 0 ? ((p0 + src_n) % ring) : (p0 + src_n);
            const int d_slot = ring > 0 ? ((p0 + k) % ring) : (p0 + k);
            device const half* srow = cache + (size_t) s_slot * MLA_ROW;
            device half* drow = cache + (size_t) d_slot * MLA_ROW;
            for (int c = int(tid); c < MLA_ROW; c += 256) drow[c] = srow[c];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Post-verify in-place sequential IndexPool ivg compaction: tr.depth[k] = path[k] >= k. grid 1, 256 threads
kernel void idx_compact_ivg(device half* ivg [[buffer(0)]], constant int& p0 [[buffer(1)]], constant TreeArgs& tr [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    for (int k = 0; k < tr.n; ++k) {
        const int src_n = int(tr.depth[k]);
        if (src_n != k) {
            ivg[(size_t) (p0 + k) * 256 + tid] = ivg[(size_t) (p0 + src_n) * 256 + tid];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Copy one captured key list row src -> dst inside dkeys/dnkc. grid 1, 256 threads
kernel void copy_key_one(device int* dkeys [[buffer(0)]], device int* dnkc [[buffer(1)]], constant int& src [[buffer(2)]], constant int& dst [[buffer(3)]],
                         constant int& stride [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    const int n = dnkc[src];
    for (int i = int(tid); i < n; i += 256) dkeys[(size_t) dst * stride + i] = dkeys[(size_t) src * stride + i];
    if (tid == 0) dnkc[dst] = n;
}
