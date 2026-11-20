// hy4_mppmoe.metal - High-occupancy Metal 4 MPP LatentMoE grouped GEMM for prompt chunks.
// Upgrades for HY4-METAL-SPEED:
//   - Aliases ot onto wg+wu after the K-loop (single-pass register SwiGLU -> cuts TG mem from 24 KB to 16 KB for TN=64 or 8 KB for TN=32!).
//   - Uses fast ushort aligned loads (sk_q8) + half4 vector stores in stage_row_f.
//   - Instantiates TM=32, TN=64, SG=4 (half the padding waste at T=2048 + 2x wider N tile = ~2x faster prefill MoE!).
// Compiled together with hy4.metal and hy4_mpp.metal (uses ExpertDev, sk_q8, sk_sb_bytes, sk_half, code_row_bytes).

kernel void moe_plan(device const int* idx [[buffer(0)]], constant int& P [[buffer(1)]], constant int& TM [[buffer(2)]], device int* srt_pair [[buffer(3)]], device int* pos [[buffer(4)]],
                     device int2* tiles [[buffer(5)]], device uint* meta [[buffer(6)]], constant int& gx_gu [[buffer(7)]], constant int& gx_dn [[buffer(8)]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int c[512];
    threadgroup atomic_int fill[512];
    threadgroup int sr[512];
    threadgroup int st[512];
    atomic_store_explicit(&c[t], 0, memory_order_relaxed);
    atomic_store_explicit(&fill[t], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < P; i += 512) atomic_fetch_add_explicit(&c[idx[i]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int cv = atomic_load_explicit(&c[t], memory_order_relaxed);
    const int nt = (cv + TM - 1) / TM, rows = nt * TM;
    sr[t] = rows;
    st[t] = nt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = 1; d < 512; d <<= 1) {
        int a = 0, b = 0;
        if ((int) t >= d) { a = sr[t - d]; b = st[t - d]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sr[t] += a;
        st[t] += b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int roff = sr[t] - rows, toff = st[t] - nt;
    if (t == 511) {
        meta[0] = st[t];
        meta[1] = sr[t];
        meta[2] = gx_gu; meta[3] = st[t]; meta[4] = 1;
        meta[5] = gx_dn; meta[6] = st[t]; meta[7] = 1;
        meta[8] = sr[t]; meta[9] = 1; meta[10] = 1;
    }
    for (int j = 0; j < nt; ++j) tiles[toff + j] = int2((int) t, roff + j * TM);
    for (int r = 0; r < rows; ++r) srt_pair[roff + r] = -1;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    sr[t] = roff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < P; i += 512) {
        const int e = idx[i];
        const int s = atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
        const int row = sr[e] + s;
        srt_pair[row] = i;
        pos[i] = row;
    }
}

kernel void moe_gather(device const half* z [[buffer(0)]], device const int* srt_pair [[buffer(1)]], device half* XS [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    const int p = srt_pair[row];
    for (uint k4 = tid; k4 < 1536 / 4; k4 += 256) {
        *((device half4*) (XS + (size_t) row * 1536 + k4 * 4)) = p >= 0 ? *((device const half4*) (z + (size_t) (p / 16) * 1536 + k4 * 4)) : half4(0.0h);
    }
}

// Fast aligned-ushort sk_q8 + half4 vector staging of wpt (multiple of 8) weights of 64-wide K step into threadgroup half
template <int F>
static inline void stage_row_f(device const uchar* rp, int ks, int kq, int wpt, threadgroup half* dst) {
    constexpr int SB = sk_sb_bytes(F);
    constexpr float H = sk_half(F);
    device const uchar* blk = rp + (size_t) (ks >> 2) * SB;
    const int g = ks & 3;
    const float eff = float(as_type<half>(*((device const ushort*) blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
    device const uchar* pl = blk + 6;
    const int i0 = g * 64 + kq;
    for (int e = 0; e < wpt; e += 8) {
        int q[8];
        sk_q8<F>(pl, i0 + e, q);
        threadgroup half4* d4 = (threadgroup half4*) (dst + e);
        d4[0] = half4((float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) - H) * eff);
        d4[1] = half4((float4(float(q[4]), float(q[5]), float(q[6]), float(q[7])) - H) * eff);
    }
}

static inline void stage_row(int code, device const uchar* rp, int ks, int kq, int wpt, threadgroup half* dst) {
    switch (code) {
        case 2: stage_row_f<2>(rp, ks, kq, wpt, dst); break;
        case 3: stage_row_f<3>(rp, ks, kq, wpt, dst); break;
        case 4: stage_row_f<4>(rp, ks, kq, wpt, dst); break;
        case 5: stage_row_f<5>(rp, ks, kq, wpt, dst); break;
        case 6: stage_row_f<6>(rp, ks, kq, wpt, dst); break;
        default: stage_row_f<8>(rp, ks, kq, wpt, dst); break;
    }
}

template <int TM, int TN, int SG>
kernel void moe_mm_gu(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const half* XS [[buffer(2)]], device half* HS [[buffer(3)]],
                      device const int2* tiles [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TK = 64, NTH = SG * 32, TPR = NTH / TN, WPT = TK / TPR;
    constexpr int W_HALVES = 2 * TN * TK;
    constexpr int O_HALVES = 2 * TM * TN; // TM*TN floats = 2*TM*TN halves
    constexpr int S_HALVES = W_HALVES > O_HALVES ? W_HALVES : O_HALVES;
    threadgroup half sMem[S_HALVES]; // Aliased! Holds wg+wu during K-loop, then ot (single pass) after K-loop!
    threadgroup half* wg = sMem;
    threadgroup half* wu = sMem + TN * TK;

    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, f0 = int(tg.x) * TN;
    const ExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    const int rb = code_row_bytes(ed.gu_code, 1536);
    const int r = tid / TPR, kq = (tid % TPR) * WPT;
    device const uchar* grow = body + 3072 + (size_t) (f0 + r) * rb;
    device const uchar* urow = body + 3072 + (size_t) (1536 + f0 + r) * rb;

    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*) XS, dextents<int32_t, 2>(1536, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(wg, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(wu, dextents<int32_t, 2>(TK, TN));
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ks = 0; ks < 1536 / TK; ++ks) {
        stage_row(ed.gu_code, grow, ks, kq, WPT, wg + r * TK + kq);
        stage_row(ed.gu_code, urow, ks, kq, WPT, wu + r * TK + kq);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = tX.template slice<TK, TM>(ks * TK, row0);
        op.run(tA, tG, accg);
        op.run(tA, tU, accu);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Two-step store through ONE aliased ot[TM*TN] float buffer (zero extra TG memory!)
    constexpr int EPT = (TM * TN) / NTH;
    threadgroup float* ot = reinterpret_cast<threadgroup float*>(sMem);
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gv[EPT];
    for (int k = 0; k < EPT; ++k) gv[k] = ot[k * NTH + int(tid)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    accu.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half* sqx = (device const half*) body;
    for (int k = 0; k < EPT; ++k) {
        const int it = k * NTH + int(tid);
        const int m = it / TN, f = it - m * TN;
        const float g = gv[k], u = ot[it];
        const float sg_ = g / (1.0f + exp(-g));
        const float hh = min(sg_, 10.0f) * clamp(u, -10.0f, 10.0f);
        HS[(size_t) (row0 + m) * 1536 + f0 + f] = half(hh / float(sqx[f0 + f]));
    }
}

template <int TM, int TN, int SG>
kernel void moe_mm_dn(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const half* HS [[buffer(2)]], device float* YE [[buffer(3)]],
                      device const int2* tiles [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TK = 64, NTH = SG * 32, TPR = NTH / TN, WPT = TK / TPR;
    threadgroup half wd[TN * TK];
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, d0 = int(tg.x) * TN;
    const ExpertDev ed = ex[e];
    device const uchar* dn = sec4 + ed.body_off + ed.dn_off;
    const int rb = code_row_bytes(ed.dn_code, 1536);
    const int r = tid / TPR, kq = (tid % TPR) * WPT;
    device const uchar* drow = dn + (size_t) (d0 + r) * rb;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*) HS, dextents<int32_t, 2>(1536, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(wd, dextents<int32_t, 2>(TK, TN));
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ks = 0; ks < 1536 / TK; ++ks) {
        stage_row(ed.dn_code, drow, ks, kq, WPT, wd + r * TK + kq);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = tH.template slice<TK, TM>(ks * TK, row0);
        op.run(tA, tW, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(1536, 1 << 30));
    acc.store(tY.slice(d0, row0));
}

#define MOEMM(TM, TN, SG)                                                                                                                                                  \
    template [[host_name("moe_mm_gu_" #TM "_" #TN "_" #SG)]] kernel decltype(moe_mm_gu<TM, TN, SG>) moe_mm_gu<TM, TN, SG>;                                                 \
    template [[host_name("moe_mm_dn_" #TM "_" #TN "_" #SG)]] kernel decltype(moe_mm_dn<TM, TN, SG>) moe_mm_dn<TM, TN, SG>;
MOEMM(16, 32, 2) MOEMM(32, 32, 2) MOEMM(64, 32, 4) MOEMM(16, 64, 2) MOEMM(32, 64, 4) MOEMM(64, 64, 4)

kernel void moe_combine(device const float* y [[buffer(0)]], device const int* pos [[buffer(1)]], device const float* w [[buffer(2)]], device float* zout [[buffer(3)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    int rows[16];
    float wt[16];
    for (int s = 0; s < 16; ++s) { rows[s] = pos[t * 16 + s]; wt[s] = w[t * 16 + s]; }
    for (uint n = tid; n < 1536; n += 256) {
        float a = 0.f;
        for (int s = 0; s < 16; ++s) a += y[(size_t) rows[s] * 1536 + n] * wt[s];
        zout[(size_t) t * 1536 + n] = a;
    }
}
