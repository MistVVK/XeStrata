// hy4_split.metal - split-K MLA attention for decode (T <= 4): the key range of every head is divided over S threadgroups (grid (NH, S, T)), each returns (m, l, unnormalised acc[512]) relative to its
// own running maximum; mla_attend_merge combines the S partials. Same math as mla_attend (hy4.metal), 8 simdgroups per threadgroup, the sink belongs to split 0.
// ws layout: [t][h][s][514] floats = { m, l, acc[512] }

kernel void mla_attend_part(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device float* ws [[buffer(3)]],
                            constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]], constant int& S [[buffer(7)]],
                            uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = tgp.x, s = tgp.y, t = tgp.z;
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
    const int pos = a.p0 + t;
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
        int slot;
        if (a.mode == 0) slot = a.ring > 0 ? j % a.ring : j;
        else if (a.mode == 1) slot = (lo + j) % a.ring;
        else slot = keys[(size_t) t * a.kmax + j];
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
    if (M == -INFINITY) {  // an empty split
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

// grid (NH, T), 256 threads
kernel void mla_attend_merge(device const float* ws [[buffer(0)]], device half* ctx [[buffer(1)]], constant MlaArgs& a [[buffer(2)]], constant int& S [[buffer(3)]],
                             uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = tgp.x, t = tgp.y;
    device const float* base = ws + ((size_t) t * a.NH + h) * S * 514;
    float M = -INFINITY;
    for (int s = 0; s < S; ++s) M = max(M, base[s * 514]);
    float L = 0.f, o0 = 0.f, o1 = 0.f;
    for (int s = 0; s < S; ++s) {
        const float ms = base[s * 514];
        const float f = ms == -INFINITY ? 0.0f : exp(ms - M);
        L += base[s * 514 + 1] * f;
        o0 += base[s * 514 + 2 + 2 * tid] * f;
        o1 += base[s * 514 + 2 + 2 * tid + 1] * f;
    }
    const float il = 1.0f / L;
    *((device half2*) (ctx + ((size_t) t * a.NH + h) * MLA_KV + 2 * tid)) = half2(half(o0 * il), half(o1 * il));
}
