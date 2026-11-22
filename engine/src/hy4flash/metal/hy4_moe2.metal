// hy4_moe2.metal - LatentMoE experts for decode / verify windows (T <= 8 tokens): every distinct expert of the window is read once and applied to all the window tokens that picked it.
// Upgrades for HY4-METAL-SPEED:
//   - Indirect dispatch grid written by moe2_plan into meta[2..4] = {192, u_count, 1} (eliminates 30-50% wasted empty threadgroups at M=2..8).
//   - Exact NT = 1..8 specialization (no wasted n=3->4 or n=5..7->8 work).
//   - Hoisted sub-group scale + float4 dot inner loop (100% bitwise M-invariant across M=1..8).
// Compiled with hy4.metal (ExpertDev, sk_q8, sk_half, sk_sb_bytes, code_row_bytes).

// grid 1, 512 threads. idx int [P] (P = T*16 <= 128). meta[0] = distinct experts, meta[2..4] = indirect grid {192, u_count, 1}.
kernel void moe2_plan(device const int* idx [[buffer(0)]], constant int& P [[buffer(1)]], device int* uexp [[buffer(2)]], device int* ucnt [[buffer(3)]], device int* upair [[buffer(4)]],
                      device uint* meta [[buffer(5)]], uint e [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sums[16];
    threadgroup int s_idx[128];
    if ((int) e < P) s_idx[e] = idx[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int mine[8];
    int c = 0;
    for (int i = 0; i < P; ++i)
        if (s_idx[i] == (int) e && c < 8) mine[c++] = i;
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
    if (e == 511) {
        const uint n_uniq = uint(base + ex + has);
        meta[0] = n_uniq;
        meta[1] = 0u;
        meta[2] = 192u;
        meta[3] = n_uniq;
        meta[4] = 1u;
    }
}

template <int F, int NT>
static inline void m2_gu_rows(device const uchar* rg, device const uchar* ru, device const half* z, thread const int* tk, uint tid, int lane, threadgroup half* sx, thread float* ag, thread float* au) {
    constexpr int SB = sk_sb_bytes(F);
    const int g = lane >> 3, j = (g << 6) + ((lane & 7) << 3);
    for (int sbi = 0; sbi < 6; ++sbi) {
        if (NT >= 2) {
            // Cooperatively stage sbi's 256 halves for all NT tokens into threadgroup memory (<= 4 KB) -> 8x fewer device loads!
            for (int idx = int(tid); idx < NT * 64; idx += 256) {
                const int t = idx >> 6, c4 = (idx & 63) << 2;
                *((threadgroup half4*) (sx + t * 256 + c4)) = *((device const half4*) (z + (size_t) tk[t] * 1536 + sbi * 256 + c4));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        device const uchar* bg = rg + (size_t) sbi * SB;
        device const uchar* bu = ru + (size_t) sbi * SB;
        const float effg = float(as_type<half>(*((device const ushort*) bg))) * (float(bg[2 + g]) * (1.0f / 255.0f));
        const float effu = float(as_type<half>(*((device const ushort*) bu))) * (float(bu[2 + g]) * (1.0f / 255.0f));
        float4 wg0, wg1, wu0, wu1;
        sk_w8_f4<F>(bg + 6, j, wg0, wg1);
        sk_w8_f4<F>(bu + 6, j, wu0, wu1);
        if (NT == 1) {
            device const half4* x4 = (device const half4*) (z + (size_t) tk[0] * 1536 + sbi * 256 + j);
            const float4 xa = float4(x4[0]), xb = float4(x4[1]);
            ag[0] = fma(effg, dot(wg0, xa) + dot(wg1, xb), ag[0]);
            au[0] = fma(effu, dot(wu0, xa) + dot(wu1, xb), au[0]);
        } else {
            for (int t = 0; t < NT; ++t) {
                threadgroup const half4* x4 = (threadgroup const half4*) (sx + t * 256 + j);
                const float4 xa = float4(x4[0]), xb = float4(x4[1]);
                ag[t] = fma(effg, dot(wg0, xa) + dot(wg1, xb), ag[t]);
                au[t] = fma(effu, dot(wu0, xa) + dot(wu1, xb), au[t]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
template <int NT>
static inline void m2_gu_code(int code, device const uchar* rg, device const uchar* ru, device const half* z, thread const int* tk, uint tid, int lane, threadgroup half* sx, thread float* ag, thread float* au) {
    switch (code) {
        case 2: m2_gu_rows<2, NT>(rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 3: m2_gu_rows<3, NT>(rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 4: m2_gu_rows<4, NT>(rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 5: m2_gu_rows<5, NT>(rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 6: m2_gu_rows<6, NT>(rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 8: m2_gu_rows<8, NT>(rg, ru, z, tk, tid, lane, sx, ag, au); break;
        default: break;
    }
}

// indirect grid (192, u_count, 1) from meta[2..4], 256 threads. h2 half [P][1536] indexed by pair id
kernel void moe2_gu(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const half* z [[buffer(2)]], device const int* uexp [[buffer(3)]],
                    device const int* ucnt [[buffer(4)]], device const int* upair [[buffer(5)]], device const uint* meta [[buffer(6)]], device half* h2 [[buffer(7)]], uint2 tgp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sx[8 * 256]; // 4096 B
    const int y = int(tgp.y);
    if (y >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
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
    switch (n) {
        case 1: m2_gu_code<1>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 2: m2_gu_code<2>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 3: m2_gu_code<3>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 4: m2_gu_code<4>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 5: m2_gu_code<5>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 6: m2_gu_code<6>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        case 7: m2_gu_code<7>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
        default: m2_gu_code<8>(ed.gu_code, rg, ru, z, tk, tid, lane, sx, ag, au); break;
    }
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
    const int g = lane >> 3, j = (g << 6) + ((lane & 7) << 3);
    for (int sbi = 0; sbi < 6; ++sbi) {
        device const uchar* blk = rp + (size_t) sbi * SB;
        const float eff = float(as_type<half>(*((device const ushort*) blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
        float4 w0, w1;
        sk_w8_f4<F>(blk + 6, j, w0, w1);
        for (int t = 0; t < NT; ++t) {
            device const half4* x4 = (device const half4*) (h2 + (size_t) pid[t] * 1536 + sbi * 256 + j);
            acc[t] = fma(eff, dot(w0, float4(x4[0])) + dot(w1, float4(x4[1])), acc[t]);
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

// indirect grid (192, u_count, 1) from meta[2..4], 256 threads. y float [P][1536] = router weight * down(h2)
kernel void moe2_dn(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const float* topk_w [[buffer(2)]], device const int* uexp [[buffer(3)]],
                    device const int* ucnt [[buffer(4)]], device const int* upair [[buffer(5)]], device const uint* meta [[buffer(6)]], device const half* h2 [[buffer(7)]], device float* y [[buffer(8)]],
                    uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int yy = int(tgp.y);
    if (yy >= int(meta[0])) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = int(tgp.x) * 8 + int(sg);
    const int e = uexp[yy], n = ucnt[yy];
    const ExpertDev ed = ex[e];
    device const uchar* dn = sec4 + ed.body_off + ed.dn_off;
    device const uchar* rp = dn + (size_t) r * code_row_bytes(ed.dn_code, 1536);
    int pid[8];
    for (int i = 0; i < 8; ++i) pid[i] = i < n ? upair[yy * 8 + i] : 0;
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    switch (n) {
        case 1: m2_dn_code<1>(ed.dn_code, rp, h2, pid, lane, acc); break;
        case 2: m2_dn_code<2>(ed.dn_code, rp, h2, pid, lane, acc); break;
        case 3: m2_dn_code<3>(ed.dn_code, rp, h2, pid, lane, acc); break;
        case 4: m2_dn_code<4>(ed.dn_code, rp, h2, pid, lane, acc); break;
        case 5: m2_dn_code<5>(ed.dn_code, rp, h2, pid, lane, acc); break;
        case 6: m2_dn_code<6>(ed.dn_code, rp, h2, pid, lane, acc); break;
        case 7: m2_dn_code<7>(ed.dn_code, rp, h2, pid, lane, acc); break;
        default: m2_dn_code<8>(ed.dn_code, rp, h2, pid, lane, acc); break;
    }
    for (int i = 0; i < 8; ++i) {
        if (i >= n) break;
        const float v = simd_sum(acc[i]);
        if (lane == 0) y[(size_t) pid[i] * 1536 + r] = v * topk_w[pid[i]];
    }
}
