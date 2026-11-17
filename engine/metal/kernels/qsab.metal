// metal/kernels/qsab.metal - qsa_attn (qsa.metal) with its loads batched (after qsa.metal).  The score loop and the V loop each
// waited one memory latency per cell (64 cells a chunk: ~30 us a layer in a 1,000-cell context); here the K rows of NS cells
// and the V values of NB cells are loaded before they are used.  Every dot, sum and accumulation is done with the same
// operands in the same order as qsa_attn, so the partials are bitwise the same.
kernel void qsa_attn_b(device const float* qcur [[buffer(0)]], device const char* kq [[buffer(1)]],
                       device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                       device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                       device float* part [[buffer(6)]], constant QsaArgs& a [[buffer(7)]],
                       device const uint* rinfo [[buffer(8), function_constant(qsa_rows)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV;
    constexpr int NS = 4, NB = 8;
    threadgroup float qs[HPG][QD];
    threadgroup float s[HPG][QCH];
    threadgroup int cid[QCH];
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = (qsa_rows ? rinfo[8 + 4 * t + 1] : a.pos0 + uint(t)) + 1, width = min(n_kv, uint(QBUDGET));
    if (qsa_rows) { const ulong sl = rinfo[8 + 4 * t]; kq += sl * rinfo[0]; vq += sl * rinfo[0]; ksc += sl * rinfo[1]; vsc += sl * rinfo[1]; }
    const int c0 = chunk * QCH;
    const int nchunks = (QBUDGET + QCH - 1) / QCH;
    device float* pp = part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    if (uint(c0) >= width) {
        for (int h = 0; h < HPG; ++h)
            if (tid == 0) { pp[(g * HPG + h) * (QD + 2) + QD] = -INFINITY; pp[(g * HPG + h) * (QD + 2) + QD + 1] = 0.0f; }
        return;
    }
    const int nc = min(QCH, int(width) - c0);
    for (int h = 0; h < HPG; ++h) qs[h][tid] = qcur[(t * QH + g * HPG + h) * QD + tid];
    if (tid < QCH) cid[tid] = tid < nc ? ids[t * QBUDGET + c0 + tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: simdgroup sg takes cells sg, sg + 8, ...; NS of them at a time
    for (int cb = sg; cb < nc; cb += 8 * NS) {
        float kv[NS][8], scl[NS];
        for (int b = 0; b < NS; ++b) {
            const int c = cb + 8 * b;
            if (c < nc) {
                const uint rowi = uint(cid[c]) * QKV + uint(g);
                device const char* kr = kq + rowi * QD;
                scl[b] = float(ksc[rowi * 4 + (lane >> 3)]);
                for (int i = 0; i < 8; ++i) kv[b][i] = float(kr[lane * 8 + i]);
            }
        }
        for (int b = 0; b < NS; ++b) {
            const int c = cb + 8 * b;
            if (c < nc) {
                for (int i = 0; i < 8; ++i) kv[b][i] = kv[b][i] * scl[b];
                for (int h = 0; h < HPG; ++h) {
                    float d = 0.0f;
                    for (int i = 0; i < 8; ++i) d = fma(qs[h][lane * 8 + i], kv[b][i], d);
                    d = simd_sum(d);
                    if (lane == 0) s[h][c] = d * 0.0625f;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float mh[HPG], lh[HPG];
    for (int h = sg; h < HPG; h += 8) {
        float m = -INFINITY;
        for (int c = lane; c < nc; c += 32) m = max(m, s[h][c]);
        m = simd_max(m);
        float l = 0.0f;
        for (int c = lane; c < nc; c += 32) {
            const float p = exp(s[h][c] - m);
            s[h][c] = p;
            l += p;
        }
        l = simd_sum(l);
        if (lane == 0) { mh[h] = m; lh[h] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // acc[h][j] = sum_c p[h][c] v[c][j]: thread = dim j; NB cells' values loaded together, added in cell order
    float acc[HPG];
    for (int h = 0; h < HPG; ++h) acc[h] = 0.0f;
    for (int cb = 0; cb < nc; cb += NB) {
        float v[NB];
        for (int b = 0; b < NB; ++b)
            if (cb + b < nc) {
                const uint rowi = uint(cid[cb + b]) * QKV + uint(g);
                v[b] = float(vq[rowi * QD + tid]) * float(vsc[rowi * 4 + (tid >> 6)]);
            }
        for (int b = 0; b < NB; ++b)
            if (cb + b < nc)
                for (int h = 0; h < HPG; ++h) acc[h] = fma(s[h][cb + b], v[b], acc[h]);
    }
    for (int h = 0; h < HPG; ++h) {
        device float* o = pp + (g * HPG + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = mh[h]; o[QD + 1] = lh[h]; }
    }
}
