// hy4_mppflash.metal - Metal 4 MPP matmul2d Flash-MLA (prefill + split-K decode/verify), per-head MPP prep_q/vup, and Block-Tiled MPP attention.
// Compiled together with hy4.metal and hy4_mpp.metal (uses MlaArgs, MLA_ROW, MLA_KV, rope_pair, sigmoidf_, mpp::tensor_ops).

// ================================================================ 1. Fused MPP Flash-MLA (TN=128, Deferred-Sink Exact Online Softmax) ==================================
// Tile: 1 threadgroup (4 simdgroups = 128 threads) = 1 query token x 16 heads (grid.x = NH / 16, grid.y = M).
// Keys in blocks of BN = 64.
// Deferred-sink online softmax: sm starts at -INFINITY, sl starts at 0.0f so sP (half) always uses the full 11-bit mantissa relative to real keys
// even on sink-dominated heads; learnable sink is combined accurately in fp32 at the final normalization!
// Uses TN = 128 for P@V (4 tiles acc0..acc3 instead of 8 -> half the barriers on MLA_LIST and half the MPP calls on causal/SWA!).
kernel void mla_flash_mpp(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device half* ctx [[buffer(3)]],
                          constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sBuf[64 * 128]; // 16384 B: K[64][128] (2x faster list staging), V[64][128] (4 passes instead of 8), S[16][64] (4 KB), Oout[16][128] (8 KB)
    threadgroup half sP[16 * 64];    // 2048 B
    threadgroup float sm[16], sl[16], sc[16];
    threadgroup int sslot[64];

    threadgroup float* sF = reinterpret_cast<threadgroup float*>(sBuf);

    const int h0 = int(tgp.x) * 16;
    const int t = int(tgp.y);
    const int pos = a.p0 + t;
    int nk, lo = 0;
    if (a.mode == 0) {
        nk = pos + 1;
    } else if (a.mode == 1) {
        lo = max(0, pos - a.window + 1);
        nk = pos - lo + 1;
    } else {
        nk = nkeys[t];
    }

    if (tid < 16) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
    }

    device half* qbase = (device half*) (qa + ((size_t) t * a.NH + h0) * MLA_ROW);
    device half* cbase = (device half*) cache;
    device const int* klist = keys + (size_t) t * a.kmax;

    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qbase, dextents<int32_t, 2>(MLA_ROW, 16));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tCache(cbase, dextents<int32_t, 2>(MLA_ROW, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK64(sBuf, dextents<int32_t, 2>(64, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK128(sBuf, dextents<int32_t, 2>(128, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(64, 16));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sF, dextents<int32_t, 2>(64, 16));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO128(sF, dextents<int32_t, 2>(128, 16));

    constexpr auto dQK_full = matmul2d_descriptor(16, 64, MLA_ROW, false, true, false);
    constexpr auto dQK_128  = matmul2d_descriptor(16, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dQK_64   = matmul2d_descriptor(16, 64, 64,  false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dPV_128  = matmul2d_descriptor(16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dQK_full, execution_simdgroups<4>> opQK_full;
    matmul2d<dQK_128,  execution_simdgroups<4>> opQK_128;
    matmul2d<dQK_64,   execution_simdgroups<4>> opQK_64;
    matmul2d<dPV_128,  execution_simdgroups<4>> opPV;

    auto vdummy = tCache.template slice<128, 64>(0, 0);
    auto acc0 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc1 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc2 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc3 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j0 = 0; j0 < nk; j0 += 64) {
        bool contig = false;
        int slot0 = 0;
        if (j0 + 64 <= nk) {
            if (a.mode == 0) {
                slot0 = a.ring > 0 ? (j0 % a.ring) : j0;
                contig = (a.ring == 0) || (slot0 + 64 <= a.ring);
            } else if (a.mode == 1) {
                slot0 = (lo + j0) % a.ring;
                contig = (slot0 + 64 <= a.ring);
            }
        }

        if (contig) {
            auto mQ = tQ.template slice<MLA_ROW, 16>(0, 0);
            auto mK = tCache.template slice<MLA_ROW, 64>(0, slot0);
            opQK_full.run(mQ, mK, tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            if (tid < 64) {
                const int j = j0 + int(tid);
                int sl_i = -1;
                if (j < nk) {
                    if (a.mode == 0) sl_i = a.ring > 0 ? j % a.ring : j;
                    else if (a.mode == 1) sl_i = (lo + j) % a.ring;
                    else sl_i = klist[j];
                }
                sslot[tid] = sl_i;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto mQ0 = tQ.template slice<128, 16>(0, 0);
            auto accS = opQK_128.template get_destination_cooperative_tensor<decltype(mQ0), decltype(tK128), float>();
            for (uint16_t i = 0; i < accS.get_capacity(); ++i)
                if (accS.is_valid_element(i)) accS[i] = 0.0f;

            // 4 steps of BK = 128 (dims 0..511)
            for (int ks = 0; ks < 4; ++ks) {
                for (int p = 0; p < 8; ++p) {
                    const uint idx = uint(p * 128) + tid;
                    const uint kr = idx >> 4;
                    const uint kc = (idx & 15u) << 3;
                    const int sl_i = sslot[kr];
                    const uint4 kv = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + ks * 128 + int(kc))) : uint4(0u);
                    *((threadgroup uint4*) (sBuf + kr * 128u + kc)) = kv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mQk = tQ.template slice<128, 16>(ks * 128, 0);
                opQK_128.run(mQk, tK128, accS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // Final step of BK = 64 (RoPE dims 512..575)
            {
                for (int p = 0; p < 4; ++p) {
                    const uint idx = uint(p * 128) + tid;
                    const uint kr = idx >> 3;
                    const uint kc = (idx & 7u) << 3;
                    const int sl_i = sslot[kr];
                    const uint4 kv = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + 512 + int(kc))) : uint4(0u);
                    *((threadgroup uint4*) (sBuf + kr * 64u + kc)) = kv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mQlast = tQ.template slice<64, 16>(512, 0);
                opQK_64.run(mQlast, tK64, accS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            accS.store(tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Exact natural-exp online softmax over sF[16][64] (8 threads per head row) -> sP[16][64]
        {
            const uint r = tid >> 3;
            const uint lane8 = tid & 7u;
            const uint kbase = lane8 * 8u;
            float vals[8];
            float mx = -INFINITY;
            for (uint e = 0; e < 8u; ++e) {
                const int kidx = j0 + int(kbase + e);
                const float x = (kidx < nk) ? sF[r * 64u + kbase + e] * a.scale : -INFINITY;
                vals[e] = x;
                mx = max(mx, x);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(4)));
            const float m_old = sm[r];
            const float mn = max(m_old, mx);
            const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            float ps = 0.0f;
            for (uint e = 0; e < 8u; ++e) {
                const int kidx = j0 + int(kbase + e);
                const float p = (kidx < nk) ? exp(vals[e] - mn) : 0.0f;
                ps += p;
                sP[r * 64u + kbase + e] = half(p);
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(2));
            ps += simd_shuffle_xor(ps, ushort(4));
            if (lane8 == 0) {
                sm[r] = mn;
                sl[r] = sl[r] * corr + ps;
                sc[r] = corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (j0 > 0) {
            for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
                if (acc0.is_valid_element(i)) {
                    const int row = acc0.get_multidimensional_index(i)[1];
                    const float c = sc[row];
                    acc0[i] *= c; acc1[i] *= c; acc2[i] *= c; acc3[i] *= c;
                }
            }
        }

        if (contig) {
            auto v0 = tCache.template slice<128, 64>(0,   slot0); opPV.run(tP, v0, acc0);
            auto v1 = tCache.template slice<128, 64>(128, slot0); opPV.run(tP, v1, acc1);
            auto v2 = tCache.template slice<128, 64>(256, slot0); opPV.run(tP, v2, acc2);
            auto v3 = tCache.template slice<128, 64>(384, slot0); opPV.run(tP, v3, acc3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            #define PV128_STAGE(COL0, ACC)                                                                                \
                for (int p = 0; p < 8; ++p) {                                                                             \
                    const uint idx = uint(p * 128) + tid;                                                                 \
                    const uint vr = idx >> 4;                                                                             \
                    const uint vc = (idx & 15u) << 3;                                                                     \
                    const int sl_i = sslot[vr];                                                                           \
                    const uint4 vv = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + (COL0) + int(vc))) : uint4(0u); \
                    *((threadgroup uint4*) (sBuf + vr * 128u + vc)) = vv;                                                 \
                }                                                                                                         \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                          \
                opPV.run(tP, tK128, ACC);                                                                                 \
                threadgroup_barrier(mem_flags::mem_threadgroup);

            PV128_STAGE(0,   acc0)
            PV128_STAGE(128, acc1)
            PV128_STAGE(256, acc2)
            PV128_STAGE(384, acc3)
            #undef PV128_STAGE
        }
    }

    // Final normalization combining deferred learnable sink in exact fp32
    if (tid < 16) {
        const float sk = sinks[h0 + int(tid)];
        const float m_k = sm[tid];
        if (m_k == -INFINITY) {
            sc[tid] = 0.0f;
        } else {
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so / (sl[tid] * so + exp(sk - m_fin));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #define STORE_O128(COL0, ACC)                                                                                         \
        ACC.store(tO128);                                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
        for (int p = 0; p < 8; ++p) {                                                                                     \
            const uint idx = uint(p * 128) + tid;                                                                         \
            const uint r = idx >> 6;                                                                                      \
            const uint c2 = (idx & 63u) * 2u;                                                                             \
            const float inv = sc[r];                                                                                      \
            *((device half2*) (ctx + ((size_t) t * a.NH + h0 + int(r)) * MLA_KV + (COL0) + int(c2))) =                    \
                half2(half(sF[r * 128u + c2] * inv), half(sF[r * 128u + c2 + 1u] * inv));                                 \
        }                                                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);

    STORE_O128(0,   acc0)
    STORE_O128(128, acc1)
    STORE_O128(256, acc2)
    STORE_O128(384, acc3)
    #undef STORE_O128
}

// ================================================================ 2. Split-K MPP Flash-MLA for Decode / Verify (T <= 8) =================================================
// Grid (NH / 16, S, T), 128 threads. Writes exact same ws[t][h][s][514] = {m, l, acc[512]} layout consumed by mla_attend_merge!
// 100% bitwise M-invariant across T = 1..8 (each token t runs independent threadgroups with fixed S).
kernel void mla_flash_mpp_part(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device float* ws [[buffer(3)]],
                               constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]], constant int& S [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sBuf[64 * 128];
    threadgroup half sP[16 * 64];
    threadgroup float sm[16], sl[16], sc[16];
    threadgroup int sslot[64];

    threadgroup float* sF = reinterpret_cast<threadgroup float*>(sBuf);

    const int h0 = int(tgp.x) * 16;
    const int sp = int(tgp.y);
    const int t = int(tgp.z);
    const int pos = a.p0 + t;
    int nk, lo = 0;
    if (a.mode == 0) nk = pos + 1;
    else if (a.mode == 1) { lo = max(0, pos - a.window + 1); nk = pos - lo + 1; }
    else nk = nkeys[t];

    // Partition nk into S tile-aligned ranges (multiple of 64 keys per split)
    const int ntiles = (nk + 63) >> 6;
    const int tps = (ntiles + S - 1) / S;
    const int j_start = (sp * tps) << 6;
    const int j_end = min(nk, ((sp + 1) * tps) << 6);

    if (j_start >= j_end) {
        // Empty split (except sp == 0 carries the sink if nk == 0, which cannot happen for pos >= 0)
        for (int r = 0; r < 16; ++r) {
            device float* o = ws + (((size_t) t * a.NH + h0 + r) * S + sp) * 514;
            if (tid == 0) {
                o[0] = (sp == 0) ? sinks[h0 + r] : -INFINITY;
                o[1] = (sp == 0) ? 1.0f : 0.0f;
            }
            for (int c = int(tid); c < 512; c += 128) o[2 + c] = 0.0f;
        }
        return;
    }

    if (tid < 16) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
    }

    device half* qbase = (device half*) (qa + ((size_t) t * a.NH + h0) * MLA_ROW);
    device half* cbase = (device half*) cache;
    device const int* klist = keys + (size_t) t * a.kmax;

    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qbase, dextents<int32_t, 2>(MLA_ROW, 16));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tCache(cbase, dextents<int32_t, 2>(MLA_ROW, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK64(sBuf, dextents<int32_t, 2>(64, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK128(sBuf, dextents<int32_t, 2>(128, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(64, 16));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sF, dextents<int32_t, 2>(64, 16));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO128(sF, dextents<int32_t, 2>(128, 16));

    constexpr auto dQK_full = matmul2d_descriptor(16, 64, MLA_ROW, false, true, false);
    constexpr auto dQK_128  = matmul2d_descriptor(16, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dQK_64   = matmul2d_descriptor(16, 64, 64,  false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dPV_128  = matmul2d_descriptor(16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dQK_full, execution_simdgroups<4>> opQK_full;
    matmul2d<dQK_128,  execution_simdgroups<4>> opQK_128;
    matmul2d<dQK_64,   execution_simdgroups<4>> opQK_64;
    matmul2d<dPV_128,  execution_simdgroups<4>> opPV;

    auto vdummy = tCache.template slice<128, 64>(0, 0);
    auto acc0 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc1 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc2 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc3 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j0 = j_start; j0 < j_end; j0 += 64) {
        bool contig = false;
        int slot0 = 0;
        if (j0 + 64 <= j_end) {
            if (a.mode == 0) { slot0 = a.ring > 0 ? (j0 % a.ring) : j0; contig = (a.ring == 0) || (slot0 + 64 <= a.ring); }
            else if (a.mode == 1) { slot0 = (lo + j0) % a.ring; contig = (slot0 + 64 <= a.ring); }
        }

        if (contig) {
            auto mQ = tQ.template slice<MLA_ROW, 16>(0, 0);
            auto mK = tCache.template slice<MLA_ROW, 64>(0, slot0);
            opQK_full.run(mQ, mK, tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            if (tid < 64) {
                const int j = j0 + int(tid);
                int sl_i = -1;
                if (j < j_end) {
                    if (a.mode == 0) sl_i = a.ring > 0 ? j % a.ring : j;
                    else if (a.mode == 1) sl_i = (lo + j) % a.ring;
                    else sl_i = klist[j];
                }
                sslot[tid] = sl_i;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mQ0 = tQ.template slice<128, 16>(0, 0);
            auto accS = opQK_128.template get_destination_cooperative_tensor<decltype(mQ0), decltype(tK128), float>();
            for (uint16_t i = 0; i < accS.get_capacity(); ++i)
                if (accS.is_valid_element(i)) accS[i] = 0.0f;
            for (int ks = 0; ks < 4; ++ks) {
                for (int p = 0; p < 8; ++p) {
                    const uint idx = uint(p * 128) + tid;
                    const uint kr = idx >> 4, kc = (idx & 15u) << 3;
                    const int sl_i = sslot[kr];
                    *((threadgroup uint4*) (sBuf + kr * 128u + kc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + ks * 128 + int(kc))) : uint4(0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mQk = tQ.template slice<128, 16>(ks * 128, 0);
                opQK_128.run(mQk, tK128, accS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            {
                for (int p = 0; p < 4; ++p) {
                    const uint idx = uint(p * 128) + tid;
                    const uint kr = idx >> 3, kc = (idx & 7u) << 3;
                    const int sl_i = sslot[kr];
                    *((threadgroup uint4*) (sBuf + kr * 64u + kc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + 512 + int(kc))) : uint4(0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mQlast = tQ.template slice<64, 16>(512, 0);
                opQK_64.run(mQlast, tK64, accS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            accS.store(tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        {
            const uint r = tid >> 3, lane8 = tid & 7u, kbase = lane8 * 8u;
            float vals[8];
            float mx = -INFINITY;
            for (uint e = 0; e < 8u; ++e) {
                const int kidx = j0 + int(kbase + e);
                const float x = (kidx < j_end) ? sF[r * 64u + kbase + e] * a.scale : -INFINITY;
                vals[e] = x;
                mx = max(mx, x);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(4)));
            const float m_old = sm[r];
            const float mn = max(m_old, mx);
            const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            float ps = 0.0f;
            for (uint e = 0; e < 8u; ++e) {
                const int kidx = j0 + int(kbase + e);
                const float p = (kidx < j_end) ? exp(vals[e] - mn) : 0.0f;
                ps += p;
                sP[r * 64u + kbase + e] = half(p);
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(2));
            ps += simd_shuffle_xor(ps, ushort(4));
            if (lane8 == 0) { sm[r] = mn; sl[r] = sl[r] * corr + ps; sc[r] = corr; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (j0 > j_start) {
            for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
                if (acc0.is_valid_element(i)) {
                    const float c = sc[acc0.get_multidimensional_index(i)[1]];
                    acc0[i] *= c; acc1[i] *= c; acc2[i] *= c; acc3[i] *= c;
                }
            }
        }

        if (contig) {
            auto v0 = tCache.template slice<128, 64>(0,   slot0); opPV.run(tP, v0, acc0);
            auto v1 = tCache.template slice<128, 64>(128, slot0); opPV.run(tP, v1, acc1);
            auto v2 = tCache.template slice<128, 64>(256, slot0); opPV.run(tP, v2, acc2);
            auto v3 = tCache.template slice<128, 64>(384, slot0); opPV.run(tP, v3, acc3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            #define PV128P(COL0, ACC)                                                                                     \
                for (int p = 0; p < 8; ++p) {                                                                             \
                    const uint idx = uint(p * 128) + tid;                                                                 \
                    const uint vr = idx >> 4, vc = (idx & 15u) << 3;                                                      \
                    const int sl_i = sslot[vr];                                                                           \
                    *((threadgroup uint4*) (sBuf + vr * 128u + vc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + (COL0) + int(vc))) : uint4(0u); \
                }                                                                                                         \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                          \
                opPV.run(tP, tK128, ACC);                                                                                 \
                threadgroup_barrier(mem_flags::mem_threadgroup);
            PV128P(0, acc0) PV128P(128, acc1) PV128P(256, acc2) PV128P(384, acc3)
            #undef PV128P
        }
    }

    // Fold learnable sink into split 0 cleanly before storing ws[t][h][sp][514]
    if (tid < 16) {
        if (sp == 0) {
            const float sk = sinks[h0 + int(tid)];
            const float m_k = sm[tid];
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so;
            sm[tid] = m_fin;
            sl[tid] = sl[tid] * so + exp(sk - m_fin);
        } else {
            sc[tid] = 1.0f;
        }
        device float* o = ws + (((size_t) t * a.NH + h0 + int(tid)) * S + sp) * 514;
        o[0] = sm[tid];
        o[1] = sl[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #define STORE_WS128(COL0, ACC)                                                                                        \
        ACC.store(tO128);                                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
        for (int p = 0; p < 16; ++p) {                                                                                    \
            const uint idx = uint(p * 128) + tid;                                                                         \
            const uint r = idx >> 7;                                                                                      \
            const uint c = idx & 127u;                                                                                    \
            device float* o = ws + (((size_t) t * a.NH + h0 + int(r)) * S + sp) * 514;                                    \
            o[2 + (COL0) + int(c)] = sF[r * 128u + c] * sc[r];                                                            \
        }                                                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);
    STORE_WS128(0, acc0) STORE_WS128(128, acc1) STORE_WS128(256, acc2) STORE_WS128(384, acc3)
    #undef STORE_WS128
}

// ================================================================ 2b. Tree-8 Split-K MPP Flash-MLA Verify (3.2x faster than scalar mla_attend_part_tree!) =================
// Bit-identical to mla_flash_mpp_part at T=1 along every tree branch. Grid (NH / 16, S, T), 128 threads.
kernel void mla_flash_mpp_part_tree(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device float* ws [[buffer(3)]],
                                    constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]], constant int& S [[buffer(7)]],
                                    constant TreeArgs& tr [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sBuf[64 * 128];
    threadgroup half sP[16 * 64];
    threadgroup float sm[16], sl[16], sc[16];
    threadgroup int sslot[64];

    threadgroup float* sF = reinterpret_cast<threadgroup float*>(sBuf);

    const int h0 = int(tgp.x) * 16;
    const int sp = int(tgp.y);
    const int t = int(tgp.z);
    const int pos = a.p0 + int(tr.depth[t]);
    int nk, lo = 0;
    if (a.mode == 0) nk = pos + 1;
    else if (a.mode == 1) { lo = max(0, pos - a.window + 1); nk = pos - lo + 1; }
    else nk = nkeys[t];

    const int ntiles = (nk + 63) >> 6;
    const int tps = (ntiles + S - 1) / S;
    const int j_start = (sp * tps) << 6;
    const int j_end = min(nk, ((sp + 1) * tps) << 6);

    if (j_start >= j_end) {
        for (int r = 0; r < 16; ++r) {
            device float* o = ws + (((size_t) t * a.NH + h0 + r) * S + sp) * 514;
            if (tid == 0) {
                o[0] = (sp == 0) ? sinks[h0 + r] : -INFINITY;
                o[1] = (sp == 0) ? 1.0f : 0.0f;
            }
            for (int c = int(tid); c < 512; c += 128) o[2 + c] = 0.0f;
        }
        return;
    }

    if (tid < 16u) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    device half* qbase = (device half*) (qa + ((size_t) t * a.NH + h0) * MLA_ROW);
    device half* cbase = (device half*) cache;
    device const int* klist = keys + (size_t) t * a.kmax;

    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qbase, dextents<int32_t, 2>(MLA_ROW, 16));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tCache(cbase, dextents<int32_t, 2>(MLA_ROW, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK64(sBuf, dextents<int32_t, 2>(64, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK128(sBuf, dextents<int32_t, 2>(128, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(64, 16));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sF, dextents<int32_t, 2>(64, 16));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO128(sF, dextents<int32_t, 2>(128, 16));

    constexpr auto dQK_full = matmul2d_descriptor(16, 64, MLA_ROW, false, true, false);
    constexpr auto dQK_128  = matmul2d_descriptor(16, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dQK_64   = matmul2d_descriptor(16, 64, 64,  false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dPV_128  = matmul2d_descriptor(16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dQK_full, execution_simdgroups<4>> opQK_full;
    matmul2d<dQK_128,  execution_simdgroups<4>> opQK_128;
    matmul2d<dQK_64,   execution_simdgroups<4>> opQK_64;
    matmul2d<dPV_128,  execution_simdgroups<4>> opPV;

    auto vdummy = tCache.template slice<128, 64>(0, 0);
    auto acc0 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc1 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc2 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    auto acc3 = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(vdummy), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j0 = j_start; j0 < j_end; j0 += 64) {
        bool contig = false;
        int slot0 = 0;
        if (j0 + 64 <= j_end) {
            if (a.mode == 0 && j0 + 64 <= a.p0) {
                slot0 = a.ring > 0 ? (j0 % a.ring) : j0;
                contig = (a.ring == 0) || (slot0 + 64 <= a.ring);
            } else if (a.mode == 1 && lo + j0 + 64 <= a.p0) {
                slot0 = (lo + j0) % a.ring;
                contig = (slot0 + 64 <= a.ring);
            }
        }

        if (contig) {
            auto mQ = tQ.template slice<MLA_ROW, 16>(0, 0);
            auto mK = tCache.template slice<MLA_ROW, 64>(0, slot0);
            opQK_full.run(mQ, mK, tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            if (tid < 64u) {
                const int j = j0 + int(tid);
                int sl_i = -1;
                if (j < j_end) {
                    int lp;
                    if (a.mode == 0) lp = j;
                    else if (a.mode == 1) lp = lo + j;
                    else lp = klist[j];
                    const int phys = (lp >= a.p0) ? (a.p0 + int(tr.anc[t][(lp - a.p0) & 7])) : lp;
                    sl_i = a.ring > 0 ? (phys % a.ring) : phys;
                }
                sslot[tid] = sl_i;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mQ0 = tQ.template slice<128, 16>(0, 0);
            auto accS = opQK_128.template get_destination_cooperative_tensor<decltype(mQ0), decltype(tK128), float>();
            for (uint16_t i = 0; i < accS.get_capacity(); ++i)
                if (accS.is_valid_element(i)) accS[i] = 0.0f;
            for (int ks = 0; ks < 4; ++ks) {
                for (int p = 0; p < 8; ++p) {
                    const uint idx = uint(p * 128) + tid;
                    const uint kr = idx >> 4, kc = (idx & 15u) << 3;
                    const int sl_i = sslot[kr];
                    *((threadgroup uint4*) (sBuf + kr * 128u + kc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + ks * 128 + int(kc))) : uint4(0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mQk = tQ.template slice<128, 16>(ks * 128, 0);
                opQK_128.run(mQk, tK128, accS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            {
                for (int p = 0; p < 4; ++p) {
                    const uint idx = uint(p * 128) + tid;
                    const uint kr = idx >> 3, kc = (idx & 7u) << 3;
                    const int sl_i = sslot[kr];
                    *((threadgroup uint4*) (sBuf + kr * 64u + kc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + 512 + int(kc))) : uint4(0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mQlast = tQ.template slice<64, 16>(512, 0);
                opQK_64.run(mQlast, tK64, accS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            accS.store(tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        {
            const uint r = tid >> 3, lane8 = tid & 7u, kbase = lane8 * 8u;
            float vals[8];
            float mx = -INFINITY;
            for (uint e = 0; e < 8u; ++e) {
                const int kidx = j0 + int(kbase + e);
                const float x = (kidx < j_end) ? sF[r * 64u + kbase + e] * a.scale : -INFINITY;
                vals[e] = x;
                mx = max(mx, x);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(4)));
            const float m_old = sm[r];
            const float mn = max(m_old, mx);
            const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            float ps = 0.0f;
            for (uint e = 0; e < 8u; ++e) {
                const int kidx = j0 + int(kbase + e);
                const float p = (kidx < j_end) ? exp(vals[e] - mn) : 0.0f;
                ps += p;
                sP[r * 64u + kbase + e] = half(p);
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(2));
            ps += simd_shuffle_xor(ps, ushort(4));
            if (lane8 == 0u) { sm[r] = mn; sl[r] = sl[r] * corr + ps; sc[r] = corr; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (j0 > j_start) {
            for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
                if (acc0.is_valid_element(i)) {
                    const float c = sc[acc0.get_multidimensional_index(i)[1]];
                    acc0[i] *= c; acc1[i] *= c; acc2[i] *= c; acc3[i] *= c;
                }
            }
        }

        if (contig) {
            auto v0 = tCache.template slice<128, 64>(0,   slot0); opPV.run(tP, v0, acc0);
            auto v1 = tCache.template slice<128, 64>(128, slot0); opPV.run(tP, v1, acc1);
            auto v2 = tCache.template slice<128, 64>(256, slot0); opPV.run(tP, v2, acc2);
            auto v3 = tCache.template slice<128, 64>(384, slot0); opPV.run(tP, v3, acc3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            #define PV128PT(COL0, ACC)                                                                                    \
                for (int p = 0; p < 8; ++p) {                                                                             \
                    const uint idx = uint(p * 128) + tid;                                                                 \
                    const uint vr = idx >> 4, vc = (idx & 15u) << 3;                                                      \
                    const int sl_i = sslot[vr];                                                                           \
                    *((threadgroup uint4*) (sBuf + vr * 128u + vc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + (COL0) + int(vc))) : uint4(0u); \
                }                                                                                                         \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                          \
                opPV.run(tP, tK128, ACC);                                                                                 \
                threadgroup_barrier(mem_flags::mem_threadgroup);
            PV128PT(0, acc0) PV128PT(128, acc1) PV128PT(256, acc2) PV128PT(384, acc3)
            #undef PV128PT
        }
    }

    if (tid < 16u) {
        if (sp == 0) {
            const float sk = sinks[h0 + int(tid)];
            const float m_k = sm[tid];
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so;
            sm[tid] = m_fin;
            sl[tid] = sl[tid] * so + exp(sk - m_fin);
        } else {
            sc[tid] = 1.0f;
        }
        device float* o = ws + (((size_t) t * a.NH + h0 + int(tid)) * S + sp) * 514;
        o[0] = sm[tid];
        o[1] = sl[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #define STORE_WST128(COL0, ACC)                                                                                       \
        ACC.store(tO128);                                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
        for (int p = 0; p < 16; ++p) {                                                                                    \
            const uint idx = uint(p * 128) + tid;                                                                         \
            const uint r = idx >> 7, c = idx & 127u;                                                                      \
            device float* o = ws + (((size_t) t * a.NH + h0 + int(r)) * S + sp) * 514;                                    \
            o[2 + (COL0) + int(c)] = sF[r * 128u + c] * sc[r];                                                            \
        }                                                                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);
    STORE_WST128(0, acc0) STORE_WST128(128, acc1) STORE_WST128(256, acc2) STORE_WST128(384, acc3)
    #undef STORE_WST128
}

// ================================================================ 3. Per-Head MPP Prefill q-absorb & v-up (50 TFLOP/s!) =================================================

// Per-head MPP q-absorb + RoPE: qa[t][h][0:512] = q_b[t][h][0:192] @ wk_t[h][0:512][0:192]^T
// Grid (512/64 = 8, ceil(M/64), NH), 128 threads (4 simdgroups).
kernel void mla_prep_q_mpp(device const float* q_b [[buffer(0)]], device const half* wk_t [[buffer(1)]], device half* qa [[buffer(2)]],
                           constant int& M [[buffer(3)]], constant int& NH [[buffer(4)]], constant int& p0 [[buffer(5)]], constant float& theta [[buffer(6)]],
                           uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sQ[64 * 64];  // 8 KB
    threadgroup float sO[64 * 64]; // 16 KB (only used at store — could alias, 16 KB fits 2 TGs/core cleanly)
    const int n0 = int(tgp.x) * 64;
    const int t0 = int(tgp.y) * 64;
    const int h = int(tgp.z);

    device half* B = (device half*) (wk_t + ((size_t) h * 512 + n0) * 192);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(sQ, dextents<int32_t, 2>(64, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(192, 64));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tC(sO, dextents<int32_t, 2>(64, 64));

    constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto b0 = tB.template slice<64, 64>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(b0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;

    for (int k0 = 0; k0 < 192; k0 += 64) {
        for (int p = 0; p < 8; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint r = idx >> 4;
            const uint c4 = (idx & 15u) << 2;
            const int t = t0 + int(r);
            const float4 v = t < M ? *((device const float4*) (q_b + ((size_t) t * NH + h) * 256 + k0 + int(c4))) : float4(0.0f);
            *((threadgroup half4*) (sQ + r * 64u + c4)) = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto bk = tB.template slice<64, 64>(k0, 0);
        op.run(tA, bk, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < 16; ++p) {
        const uint idx = uint(p * 128) + tid;
        const uint r = idx >> 5;
        const uint c2 = (idx & 31u) * 2u;
        const int t = t0 + int(r);
        if (t < M) {
            *((device half2*) (qa + ((size_t) t * NH + h) * MLA_ROW + n0 + int(c2))) =
                half2(half(sO[r * 64u + c2]), half(sO[r * 64u + c2 + 1u]));
        }
    }

    if (tgp.x == 0) {
        for (int idx = int(tid); idx < 64 * 32; idx += 128) {
            const int tt = idx >> 5, i = idx & 31, t = t0 + tt;
            if (t >= M) continue;
            device const float* qp = q_b + ((size_t) t * NH + h) * 256 + 192 + 2 * i;
            float o0, o1;
            rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
            *((device half2*) (qa + ((size_t) t * NH + h) * MLA_ROW + MLA_KV + 2 * i)) = half2(half(o0), half(o1));
        }
    }
}

// Per-head MPP v-up + sigmoid gate: o[t][h][0:256] = half(sigmoid(gate) * (ctx[t][h][0:512] @ wv[h][0:256][0:512]^T))
// Grid (256/64 = 4, ceil(M/64), NH), 128 threads (4 simdgroups).
kernel void mla_vup_mpp(device const half* ctx [[buffer(0)]], device const half* wv [[buffer(1)]], device const float* gate [[buffer(2)]], device half* o [[buffer(3)]],
                        constant int& M [[buffer(4)]], constant int& NH [[buffer(5)]],
                        uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sO[64 * 64]; // 16 KB
    const int n0 = int(tgp.x) * 64;
    const int t0 = int(tgp.y) * 64;
    const int h = int(tgp.z);

    // Zero-copy strided view of ctx[0..M-1][h][0..511]: row stride = NH * 512!
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*) ctx, dextents<int32_t, 2>(NH * MLA_KV, M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*) (wv + ((size_t) h * 256 + n0) * MLA_KV), dextents<int32_t, 2>(MLA_KV, 64));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tC(sO, dextents<int32_t, 2>(64, 64));

    constexpr auto desc = matmul2d_descriptor(64, 64, MLA_KV, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tA.slice(h * MLA_KV, t0);
    auto mB = tB.template slice<MLA_KV, 64>(0, 0);
    op.run(mA, mB, tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int p = 0; p < 16; ++p) {
        const uint idx = uint(p * 128) + tid;
        const uint r = idx >> 5;
        const uint c2 = (idx & 31u) * 2u;
        const int t = t0 + int(r);
        if (t < M) {
            const size_t base = (size_t) t * NH * 256 + (size_t) h * 256 + n0 + int(c2);
            const float2 g2 = *((device const float2*) (gate + base));
            *((device half2*) (o + base)) = half2(half(sO[r * 64u + c2] * sigmoidf_(g2.x)), half(sO[r * 64u + c2 + 1u] * sigmoidf_(g2.y)));
        }
    }
}

// ================================================================ 4. Block-Tiled MPP SWA/Causal Attention (for attb / causal) ===========================================
kernel void mla_blk_kv(device const half* cache [[buffer(0)]], device half* K_blk [[buffer(1)]], device half* Vt_blk [[buffer(2)]],
                       constant int& k_lo [[buffer(3)]], constant int& nk [[buffer(4)]], constant int& nk_pad [[buffer(5)]], constant int& ring [[buffer(6)]],
                       uint j [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const bool valid = int(j) < nk;
    const int pos = k_lo + int(j);
    const int slot = ring > 0 ? (pos % ring) : pos;
    device const half* src = valid ? (cache + (size_t) slot * MLA_ROW) : nullptr;
    for (int c = int(tid); c < MLA_ROW; c += 256) {
        const half v = valid ? src[c] : half(0.0h);
        K_blk[(size_t) j * MLA_ROW + c] = v;
        if (c < MLA_KV) Vt_blk[(size_t) c * nk_pad + j] = v;
    }
}

kernel void mla_blk_softmax(device const float* S [[buffer(0)]], device const float* sinks [[buffer(1)]], device half* P [[buffer(2)]],
                            constant int& NH [[buffer(3)]], constant int& p0_blk [[buffer(4)]], constant int& k_lo [[buffer(5)]],
                            constant int& nk_pad [[buffer(6)]], constant int& window [[buffer(7)]], constant int& mode [[buffer(8)]], constant float& scale [[buffer(9)]],
                            uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    const int tt = int(row) / NH;
    const int h = int(row) - tt * NH;
    const int pos = p0_blk + tt;
    const int lo = (mode == 1) ? max(0, pos - window + 1) : 0;
    const int j_lo = max(0, lo - k_lo);
    const int j_hi = min(nk_pad, pos - k_lo + 1);
    const float sk = sinks[h];

    device const float* sr = S + (size_t) row * nk_pad;
    device half* pr = P + (size_t) row * nk_pad;

    float mx = sk;
    for (int j = j_lo + int(tid); j < j_hi; j += 256) mx = max(mx, sr[j] * scale);
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = sh[0];
    for (int w = 1; w < 8; ++w) mx = max(mx, sh[w]);

    float sum = (tid == 0) ? exp(sk - mx) : 0.0f;
    for (int j = j_lo + int(tid); j < j_hi; j += 256) sum += exp(sr[j] * scale - mx);
    sum = simd_sum(sum);
    if ((tid & 31u) == 0u) sh[tid >> 5] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int w = 0; w < 8; ++w) tot += sh[w];
    const float inv = 1.0f / tot;

    for (int j = int(tid); j < nk_pad; j += 256) {
        pr[j] = (j >= j_lo && j < j_hi) ? half(exp(sr[j] * scale - mx) * inv) : half(0.0h);
    }
}
