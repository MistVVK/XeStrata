// hy4_moe2.metal - LatentMoE experts for decode / verify windows (T <= 8 tokens): every distinct expert of the window is read once and applied to all the window tokens that picked it.
// Plan kernel: counts the experts, lists the distinct ones in increasing id order with their (token, slot) pairs in increasing pair order (deterministic), the gate/up and down kernels run
// one simdgroup per weight row with NT (1, 2, 4 or 8) tokens per pass, selected inside the kernel from the expert's token count. Per (token, row) the arithmetic is the same expression
// as moe_gu_pair / moe_dn_pair (sum over units in lane order, simd_sum), so a window row equals the same token alone.
// Compiled with hy4.metal (ExpertDev, sk_w8, code_row_bytes).

// grid 1, 512 threads. idx int [P] (P = T*16 <= 128). meta[0] = number of distinct experts.
kernel void moe2_plan(device const int* idx [[buffer(0)]], constant int& P [[buffer(1)]], device int* uexp [[buffer(2)]], device int* ucnt [[buffer(3)]], device int* upair [[buffer(4)]],
                      device int* meta [[buffer(5)]], uint e [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sums[16];
    int mine[8];
    int c = 0;
    for (int i = 0; i < P; ++i)
        if (idx[i] == (int) e && c < 8) mine[c++] = i;
    const int has = c > 0 ? 1 : 0;
    const int ex = simd_prefix_exclusive_sum(has);
    if (lane == 31) sums[sg] = ex + has;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int base = 0;
    for (uint w = 0; w < sg; ++w) base += sums[w];
    const int pos = base + ex;
    if (has) {
        uexp[pos] = (int) e;
        ucnt[pos] = c;
        for (int k = 0; k < c; ++k) upair[pos * 8 + k] = mine[k];
    }
    if (e == 511) meta[0] = base + ex + has;
}

template <int F, int NT>
static inline void m2_gu_rows(device const uchar* rg, device const uchar* ru, device const half* z, thread const int* tk, int lane, thread float* ag, thread float* au) {
    constexpr int SB = sk_sb_bytes(F);
    for (int unit = lane; unit < 6 * 32; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        float wg[8], wu[8];
        sk_w8<F>(rg + (size_t) sbi * SB, g, j, wg);
        sk_w8<F>(ru + (size_t) sbi * SB, g, j, wu);
        for (int t = 0; t < NT; ++t) {
            device const half4* x4 = (device const half4*) (z + (size_t) tk[t] * 1536 + sbi * 256 + j);
            const float4 xa = float4(x4[0]), xb = float4(x4[1]);
            float a = 0.f, b = 0.f;
            a += wg[0] * xa.x; a += wg[1] * xa.y; a += wg[2] * xa.z; a += wg[3] * xa.w;
            a += wg[4] * xb.x; a += wg[5] * xb.y; a += wg[6] * xb.z; a += wg[7] * xb.w;
            b += wu[0] * xa.x; b += wu[1] * xa.y; b += wu[2] * xa.z; b += wu[3] * xa.w;
            b += wu[4] * xb.x; b += wu[5] * xb.y; b += wu[6] * xb.z; b += wu[7] * xb.w;
            ag[t] += a;
            au[t] += b;
        }
    }
}
template <int NT>
static inline void m2_gu_code(int code, device const uchar* rg, device const uchar* ru, device const half* z, thread const int* tk, int lane, thread float* ag, thread float* au) {
    switch (code) {
        case 2: m2_gu_rows<2, NT>(rg, ru, z, tk, lane, ag, au); break;
        case 3: m2_gu_rows<3, NT>(rg, ru, z, tk, lane, ag, au); break;
        case 4: m2_gu_rows<4, NT>(rg, ru, z, tk, lane, ag, au); break;
        case 5: m2_gu_rows<5, NT>(rg, ru, z, tk, lane, ag, au); break;
        case 6: m2_gu_rows<6, NT>(rg, ru, z, tk, lane, ag, au); break;
        case 8: m2_gu_rows<8, NT>(rg, ru, z, tk, lane, ag, au); break;
        default: break;
    }
}

// grid (192, 16*T), 256 threads. h2 half [P][1536] indexed by pair id
kernel void moe2_gu(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const half* z [[buffer(2)]], device const int* uexp [[buffer(3)]],
                    device const int* ucnt [[buffer(4)]], device const int* upair [[buffer(5)]], device const int* meta [[buffer(6)]], device half* h2 [[buffer(7)]], uint2 tgp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    const int y = tgp.y;
    if (y >= meta[0]) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = tgp.x * 8 + sg;
    const int e = uexp[y], n = ucnt[y];
    const ExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    device const half* sqx = (device const half*) body;
    const int rb = code_row_bytes(ed.gu_code, 1536);
    device const uchar* gu = body + 3072;
    device const uchar* rg = gu + (size_t) r * rb;
    device const uchar* ru = gu + (size_t) (1536 + r) * rb;
    int pid[8], tk[8];
    for (int i = 0; i < 8; ++i) {
        pid[i] = i < n ? upair[y * 8 + i] : 0;
        tk[i] = pid[i] / 16;
    }
    float ag[8], au[8];
    for (int i = 0; i < 8; ++i) { ag[i] = 0.f; au[i] = 0.f; }
    if (n == 1) m2_gu_code<1>(ed.gu_code, rg, ru, z, tk, lane, ag, au);
    else if (n == 2) m2_gu_code<2>(ed.gu_code, rg, ru, z, tk, lane, ag, au);
    else if (n <= 4) m2_gu_code<4>(ed.gu_code, rg, ru, z, tk, lane, ag, au);
    else m2_gu_code<8>(ed.gu_code, rg, ru, z, tk, lane, ag, au);
    const float sq = float(sqx[r]);
    for (int i = 0; i < 8; ++i) {
        if (i >= n) break;
        const float g = simd_sum(ag[i]), u = simd_sum(au[i]);
        if (lane == 0) {
            const float sg_ = g / (1.0f + exp(-g));
            const float hh = min(sg_, 10.0f) * clamp(u, -10.0f, 10.0f);
            h2[(size_t) pid[i] * 1536 + r] = half(hh / sq);
        }
    }
}

template <int F, int NT>
static inline void m2_dn_rows(device const uchar* rp, device const half* h2, thread const int* pid, int lane, thread float* acc) {
    constexpr int SB = sk_sb_bytes(F);
    for (int unit = lane; unit < 6 * 32; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        float w[8];
        sk_w8<F>(rp + (size_t) sbi * SB, g, j, w);
        for (int t = 0; t < NT; ++t) {
            device const half4* x4 = (device const half4*) (h2 + (size_t) pid[t] * 1536 + sbi * 256 + j);
            const float4 xa = float4(x4[0]), xb = float4(x4[1]);
            float a = 0.f;
            a += w[0] * xa.x; a += w[1] * xa.y; a += w[2] * xa.z; a += w[3] * xa.w;
            a += w[4] * xb.x; a += w[5] * xb.y; a += w[6] * xb.z; a += w[7] * xb.w;
            acc[t] += a;
        }
    }
}
template <int NT>
static inline void m2_dn_code(int code, device const uchar* rp, device const half* h2, thread const int* pid, int lane, thread float* acc) {
    switch (code) {
        case 2: m2_dn_rows<2, NT>(rp, h2, pid, lane, acc); break;
        case 3: m2_dn_rows<3, NT>(rp, h2, pid, lane, acc); break;
        case 4: m2_dn_rows<4, NT>(rp, h2, pid, lane, acc); break;
        case 5: m2_dn_rows<5, NT>(rp, h2, pid, lane, acc); break;
        case 6: m2_dn_rows<6, NT>(rp, h2, pid, lane, acc); break;
        case 8: m2_dn_rows<8, NT>(rp, h2, pid, lane, acc); break;
        default: break;
    }
}

// grid (192, 16*T), 256 threads. y float [P][1536] = router weight * down(h2)
kernel void moe2_dn(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const float* topk_w [[buffer(2)]], device const int* uexp [[buffer(3)]],
                    device const int* ucnt [[buffer(4)]], device const int* upair [[buffer(5)]], device const int* meta [[buffer(6)]], device const half* h2 [[buffer(7)]], device float* y [[buffer(8)]],
                    uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int yy = tgp.y;
    if (yy >= meta[0]) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = tgp.x * 8 + sg;
    const int e = uexp[yy], n = ucnt[yy];
    const ExpertDev ed = ex[e];
    device const uchar* dn = sec4 + ed.body_off + ed.dn_off;
    device const uchar* rp = dn + (size_t) r * code_row_bytes(ed.dn_code, 1536);
    int pid[8];
    for (int i = 0; i < 8; ++i) pid[i] = i < n ? upair[yy * 8 + i] : 0;
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    if (n == 1) m2_dn_code<1>(ed.dn_code, rp, h2, pid, lane, acc);
    else if (n == 2) m2_dn_code<2>(ed.dn_code, rp, h2, pid, lane, acc);
    else if (n <= 4) m2_dn_code<4>(ed.dn_code, rp, h2, pid, lane, acc);
    else m2_dn_code<8>(ed.dn_code, rp, h2, pid, lane, acc);
    for (int i = 0; i < 8; ++i) {
        if (i >= n) break;
        const float v = simd_sum(acc[i]);
        if (lane == 0) y[(size_t) pid[i] * 1536 + r] = v * topk_w[pid[i]];
    }
}
