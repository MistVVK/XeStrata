// hy4_flash.metal - High-occupancy K-streamed Flash-MLA attention on simdgroup_matrix (works on M1..M5).
// Compiled with hy4.metal (MlaArgs, MLA_ROW, MLA_KV).
//
// Tile: 1 threadgroup (8 simdgroups = 256 threads) = 1 query token x 32 heads (grid.x = NH / 32, grid.y = M).
// Keys are processed in blocks of BN = 64 with K-streaming BK = 64 (for Q@K^T) and BVK = 8 (for P@V).
// Threadgroup memory: 13.6 KB (sKV 8 KB reused for K/V + sQP 4 KB reused for Q/P + 1.6 KB softmax/slots) -> 2-3x occupancy vs the 28 KB v1 kernel.
// Online softmax runs in registers across all 256 threads using Apple Silicon's 8x8 fragment layout (no sS buffer).
// Contiguous full tiles (causal / SWA non-wrap) load Q, K, V directly from L1-cached device memory with zero staging barriers;
// gathered / ring-wrap / partial tail tiles stage through sKV with exact vmask zeroing of out-of-bounds slots.

template <bool DIRECT_CONTIG>
kernel void mla_flash_sg(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device half* ctx [[buffer(3)]],
                         constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]],
                         uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sKV[64 * 64];  // 8192 B: K[64][64] during Q@K^T, V[8][512] during P@V
    threadgroup half sQP[32 * 64];  // 4096 B: Q[32][64] during Q@K^T, P[32][64] during P@V
    threadgroup float sm_sg[8][16]; // 512 B
    threadgroup float sl_sg[8][16]; // 512 B
    threadgroup float sm[32], sl[32], sc[32]; // 384 B
    threadgroup int sslot[64];      // 256 B

    const int h0 = int(tgp.x) * 32;
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

    if (tid < 32) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
    }

    simdgroup_float8x8 acc[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint rb = sg >> 2;  // 0: heads 0..15, 1: heads 16..31
    const uint cb = sg & 3u;  // 0..3: keys 16*cb .. 16*cb+15
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    device const half* qbase = qa + ((size_t) t * a.NH + h0) * MLA_ROW;
    device const int* klist = keys + (size_t) t * a.kmax;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j0 = 0; j0 < nk; j0 += 64) {
        bool contig = false;
        int slot0 = 0;
        if (DIRECT_CONTIG && j0 + 64 <= nk) {
            if (a.mode == 0) {
                slot0 = a.ring > 0 ? (j0 % a.ring) : j0;
                contig = (a.ring == 0) || (slot0 + 64 <= a.ring);
            } else if (a.mode == 1) {
                slot0 = (lo + j0) % a.ring;
                contig = (slot0 + 64 <= a.ring);
            }
        }

        // ---- 1. S[32][64] = Q[32][576] @ K[64][576]^T ----
        simdgroup_float8x8 s[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                s[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        if (contig) {
            device const half* q0 = qbase + (size_t) (rb * 16) * MLA_ROW;
            device const half* k0 = cache + (size_t) (slot0 + int(cb) * 16) * MLA_ROW;
            for (int kk = 0; kk < 72; ++kk) {
                simdgroup_half8x8 qf0, qf1, kf0, kf1;
                simdgroup_load(qf0, q0 + kk * 8, MLA_ROW);
                simdgroup_load(qf1, q0 + 8 * MLA_ROW + kk * 8, MLA_ROW);
                simdgroup_load(kf0, k0 + kk * 8, MLA_ROW, ulong2(0, 0), true);
                simdgroup_load(kf1, k0 + 8 * MLA_ROW + kk * 8, MLA_ROW, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[0][0], qf0, kf0, s[0][0]);
                simdgroup_multiply_accumulate(s[0][1], qf0, kf1, s[0][1]);
                simdgroup_multiply_accumulate(s[1][0], qf1, kf0, s[1][0]);
                simdgroup_multiply_accumulate(s[1][1], qf1, kf1, s[1][1]);
            }
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

            const uint qr = tid >> 3;
            const uint qc = (tid & 7u) << 3;
            for (int ks = 0; ks < 9; ++ks) {
                const int kcol = ks * 64 + int(qc);
                *((threadgroup uint4*) (sQP + qr * 64 + qc)) = *((device const uint4*) (qbase + (size_t) qr * MLA_ROW + kcol));
                for (int p = 0; p < 2; ++p) {
                    const uint kr = uint(p * 32) + qr;
                    const int sl_i = sslot[kr];
                    const uint4 kv = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + kcol)) : uint4(0u);
                    *((threadgroup uint4*) (sKV + kr * 64 + qc)) = kv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int kk = 0; kk < 8; ++kk) {
                    simdgroup_half8x8 qf0, qf1, kf0, kf1;
                    simdgroup_load(qf0, sQP + (rb * 16) * 64 + kk * 8, 64);
                    simdgroup_load(qf1, sQP + (rb * 16 + 8) * 64 + kk * 8, 64);
                    simdgroup_load(kf0, sKV + (cb * 16) * 64 + kk * 8, 64, ulong2(0, 0), true);
                    simdgroup_load(kf1, sKV + (cb * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[0][0], qf0, kf0, s[0][0]);
                    simdgroup_multiply_accumulate(s[0][1], qf0, kf1, s[0][1]);
                    simdgroup_multiply_accumulate(s[1][0], qf1, kf0, s[1][0]);
                    simdgroup_multiply_accumulate(s[1][1], qf1, kf1, s[1][1]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // ---- 2. In-register exact natural-exp online softmax (all 256 threads) -> sP in sQP[32][64] ----
        float v[2][2][2];
        for (int i = 0; i < 2; ++i) {
            float mx = -INFINITY;
            for (int j = 0; j < 2; ++j) {
                const int kbase = j0 + int(cb) * 16 + j * 8 + int(fn);
                const float x0 = (kbase < nk) ? s[i][j].thread_elements()[0] * a.scale : -INFINITY;
                const float x1 = (kbase + 1 < nk) ? s[i][j].thread_elements()[1] * a.scale : -INFINITY;
                v[i][j][0] = x0;
                v[i][j][1] = x1;
                mx = max(mx, max(x0, x1));
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
            if (fn == 0) sm_sg[sg][i * 8 + fm] = mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 32) {
            const uint b = (tid >> 4) * 4u;
            const uint rl = tid & 15u;
            const float bmax = max(max(sm_sg[b][rl], sm_sg[b + 1][rl]), max(sm_sg[b + 2][rl], sm_sg[b + 3][rl]));
            const float m_old = sm[tid];
            const float mn = max(m_old, bmax);
            sc[tid] = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            sm[tid] = mn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 2; ++i) {
            const uint row = rb * 16u + uint(i) * 8u + fm;
            const float mn = sm[row];
            float ps = 0.0f;
            for (int j = 0; j < 2; ++j) {
                const int kcol = int(cb) * 16 + j * 8 + int(fn);
                const float p0 = (j0 + kcol < nk) ? exp(v[i][j][0] - mn) : 0.0f;
                const float p1 = (j0 + kcol + 1 < nk) ? exp(v[i][j][1] - mn) : 0.0f;
                ps += p0 + p1;
                *((threadgroup half2*) (sQP + row * 64u + uint(kcol))) = half2(half(p0), half(p1));
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(8));
            if (fn == 0) sl_sg[sg][i * 8 + fm] = ps;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 32) {
            const uint b = (tid >> 4) * 4u;
            const uint rl = tid & 15u;
            sl[tid] = sl[tid] * sc[tid] + (sl_sg[b][rl] + sl_sg[b + 1][rl] + sl_sg[b + 2][rl] + sl_sg[b + 3][rl]);
        }

        // ---- 3. Rescale O accumulators ----
        if (j0 > 0) {
            for (int i = 0; i < 4; ++i) {
                const float c = sc[i * 8 + int(fm)];
                if (simd_any(c != 1.0f)) {
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j].thread_elements()[0] *= c;
                        acc[i][j].thread_elements()[1] *= c;
                    }
                }
            }
        }

        // ---- 4. O[32][512] += P[32][64] @ V[64][512] ----
        if (contig) {
            for (int vk = 0; vk < 64; vk += 8) {
                simdgroup_half8x8 pf[4];
                for (int i = 0; i < 4; ++i) simdgroup_load(pf[i], sQP + (i * 8) * 64 + vk, 64);
                device const half* vbase = cache + (size_t) (slot0 + vk) * MLA_ROW + sg * 64;
                for (int j = 0; j < 8; ++j) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, vbase + j * 8, MLA_ROW);
                    for (int i = 0; i < 4; ++i) simdgroup_multiply_accumulate(acc[i][j], pf[i], vf, acc[i][j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (int vk = 0; vk < 64; vk += 8) {
                if (j0 + vk >= nk) break;
                for (int p = 0; p < 2; ++p) {
                    const uint idx = uint(p * 256) + tid;
                    const uint vr = idx >> 6;
                    const uint vc = (idx & 63u) << 3;
                    const int sl_i = sslot[vk + int(vr)];
                    const uint4 vv = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + vc)) : uint4(0u);
                    *((threadgroup uint4*) (sKV + vr * 512u + vc)) = vv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_half8x8 pf[4];
                for (int i = 0; i < 4; ++i) simdgroup_load(pf[i], sQP + (i * 8) * 64 + vk, 64);
                for (int j = 0; j < 8; ++j) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, sKV + sg * 64 + j * 8, 512);
                    for (int i = 0; i < 4; ++i) simdgroup_multiply_accumulate(acc[i][j], pf[i], vf, acc[i][j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // ---- 5. Fold deferred learnable sink in exact fp32 and write ctx[t][h0..h0+31][0..511] ----
    if (tid < 32) {
        const float sk = sinks[h0 + int(tid)];
        const float m_k = sm[tid];
        if (m_k == -INFINITY) sc[tid] = 0.0f;
        else {
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so / (sl[tid] * so + exp(sk - m_fin));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < 4; ++i) {
        const int row = i * 8 + int(fm);
        const float inv = sc[row];
        device half* o = ctx + ((size_t) t * a.NH + h0 + row) * MLA_KV + sg * 64 + fn;
        for (int j = 0; j < 8; ++j) {
            *((device half2*) (o + j * 8)) = half2(half(acc[i][j].thread_elements()[0] * inv), half(acc[i][j].thread_elements()[1] * inv));
        }
    }
}

template [[host_name("mla_flash")]] kernel void mla_flash_sg<true>(device const half*, device const half*, device const float*, device half*, constant MlaArgs&, device const int*, device const int*,
                                                                   uint2, uint, uint, uint);
template [[host_name("mla_flash_tg")]] kernel void mla_flash_sg<false>(device const half*, device const half*, device const float*, device half*, constant MlaArgs&, device const int*, device const int*,
                                                                       uint2, uint, uint, uint);

// Low-register-pressure variant (M1..M4 + M5): BM = 16 heads x BN = 64 keys, 8 simdgroups (256 threads), ONLY 16 accumulators (32 floats/thread) -> zero stack spills!
kernel void mla_flash_sg16(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device half* ctx [[buffer(3)]],
                           constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]],
                           uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sKV[64 * 64];  // 8192 B: K[64][64] during Q@K^T, V[8][512] during P@V
    threadgroup half sQP[16 * 64];  // 2048 B: Q[16][64] during Q@K^T, P[16][64] during P@V
    threadgroup float sm_sg[8][8];  // 256 B
    threadgroup float sl_sg[8][8];  // 256 B
    threadgroup float sm[16], sl[16], sc[16]; // 192 B
    threadgroup int sslot[64];      // 256 B  -> Total TG mem = 11.2 KB!

    const int h0 = int(tgp.x) * 16;
    const int t = int(tgp.y);
    const int pos = a.p0 + t;
    int nk, lo = 0;
    if (a.mode == 0) nk = pos + 1;
    else if (a.mode == 1) { lo = max(0, pos - a.window + 1); nk = pos - lo + 1; }
    else nk = nkeys[t];

    if (tid < 16) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
    }

    simdgroup_float8x8 acc[2][8]; // ONLY 16 tiles = 32 floats/thread (half of mla_flash_sg!)
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 8; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint rb = sg >> 2;  // 0: heads 0..7, 1: heads 8..15
    const uint cb = sg & 3u;  // 0..3: keys 16*cb .. 16*cb+15
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    device const half* qbase = qa + ((size_t) t * a.NH + h0) * MLA_ROW;
    device const int* klist = keys + (size_t) t * a.kmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j0 = 0; j0 < nk; j0 += 64) {
        bool contig = false;
        int slot0 = 0;
        if (j0 + 64 <= nk) {
            if (a.mode == 0) { slot0 = a.ring > 0 ? (j0 % a.ring) : j0; contig = (a.ring == 0) || (slot0 + 64 <= a.ring); }
            else if (a.mode == 1) { slot0 = (lo + j0) % a.ring; contig = (slot0 + 64 <= a.ring); }
        }

        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        if (contig) {
            device const half* q0 = qbase + (size_t) (rb * 8) * MLA_ROW;
            device const half* k0 = cache + (size_t) (slot0 + int(cb) * 16) * MLA_ROW;
            for (int kk = 0; kk < 72; ++kk) {
                simdgroup_half8x8 qf, kf0, kf1;
                simdgroup_load(qf, q0 + kk * 8, MLA_ROW);
                simdgroup_load(kf0, k0 + kk * 8, MLA_ROW, ulong2(0, 0), true);
                simdgroup_load(kf1, k0 + 8 * MLA_ROW + kk * 8, MLA_ROW, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s0, qf, kf0, s0);
                simdgroup_multiply_accumulate(s1, qf, kf1, s1);
            }
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

            const uint qr = tid >> 3;
            const uint qc = (tid & 7u) << 3;
            for (int ks = 0; ks < 9; ++ks) {
                const int kcol = ks * 64 + int(qc);
                if (tid < 128) *((threadgroup uint4*) (sQP + qr * 64 + qc)) = *((device const uint4*) (qbase + (size_t) qr * MLA_ROW + kcol));
                for (int p = 0; p < 2; ++p) {
                    const uint kr = uint(p * 32) + qr;
                    const int sl_i = sslot[kr];
                    *((threadgroup uint4*) (sKV + kr * 64 + qc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + kcol)) : uint4(0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int kk = 0; kk < 8; ++kk) {
                    simdgroup_half8x8 qf, kf0, kf1;
                    simdgroup_load(qf, sQP + (rb * 8) * 64 + kk * 8, 64);
                    simdgroup_load(kf0, sKV + (cb * 16) * 64 + kk * 8, 64, ulong2(0, 0), true);
                    simdgroup_load(kf1, sKV + (cb * 16 + 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s0, qf, kf0, s0);
                    simdgroup_multiply_accumulate(s1, qf, kf1, s1);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        const int kb0 = j0 + int(cb) * 16 + int(fn);
        const float x00 = (kb0 < nk) ? s0.thread_elements()[0] * a.scale : -INFINITY;
        const float x01 = (kb0 + 1 < nk) ? s0.thread_elements()[1] * a.scale : -INFINITY;
        const float x10 = (kb0 + 8 < nk) ? s1.thread_elements()[0] * a.scale : -INFINITY;
        const float x11 = (kb0 + 9 < nk) ? s1.thread_elements()[1] * a.scale : -INFINITY;
        float mx = max(max(x00, x01), max(x10, x11));
        mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
        mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
        if (fn == 0) sm_sg[sg][fm] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16) {
            const uint b = (tid >> 3) * 4u, rl = tid & 7u;
            const float bmax = max(max(sm_sg[b][rl], sm_sg[b + 1][rl]), max(sm_sg[b + 2][rl], sm_sg[b + 3][rl]));
            const float m_old = sm[tid];
            const float mn = max(m_old, bmax);
            sc[tid] = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            sm[tid] = mn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint row = rb * 8u + fm;
        const float mn = sm[row];
        const float p00 = (kb0 < nk) ? exp(x00 - mn) : 0.0f;
        const float p01 = (kb0 + 1 < nk) ? exp(x01 - mn) : 0.0f;
        const float p10 = (kb0 + 8 < nk) ? exp(x10 - mn) : 0.0f;
        const float p11 = (kb0 + 9 < nk) ? exp(x11 - mn) : 0.0f;
        const uint kcol = cb * 16u + fn;
        *((threadgroup half2*) (sQP + row * 64u + kcol)) = half2(half(p00), half(p01));
        *((threadgroup half2*) (sQP + row * 64u + kcol + 8u)) = half2(half(p10), half(p11));
        float ps = p00 + p01 + p10 + p11;
        ps += simd_shuffle_xor(ps, ushort(1));
        ps += simd_shuffle_xor(ps, ushort(8));
        if (fn == 0) sl_sg[sg][fm] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 16) {
            const uint b = (tid >> 3) * 4u, rl = tid & 7u;
            sl[tid] = sl[tid] * sc[tid] + (sl_sg[b][rl] + sl_sg[b + 1][rl] + sl_sg[b + 2][rl] + sl_sg[b + 3][rl]);
        }

        for (int i = 0; i < 2; ++i) {
            const float c = sc[i * 8 + int(fm)];
            if (simd_any(c != 1.0f)) {
                for (int j = 0; j < 8; ++j) {
                    acc[i][j].thread_elements()[0] *= c;
                    acc[i][j].thread_elements()[1] *= c;
                }
            }
        }

        if (contig) {
            for (int vk = 0; vk < 64; vk += 8) {
                simdgroup_half8x8 pf0, pf1;
                simdgroup_load(pf0, sQP + vk, 64);
                simdgroup_load(pf1, sQP + 8 * 64 + vk, 64);
                device const half* vbase = cache + (size_t) (slot0 + vk) * MLA_ROW + sg * 64;
                for (int j = 0; j < 8; ++j) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, vbase + j * 8, MLA_ROW);
                    simdgroup_multiply_accumulate(acc[0][j], pf0, vf, acc[0][j]);
                    simdgroup_multiply_accumulate(acc[1][j], pf1, vf, acc[1][j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (int vk = 0; vk < 64; vk += 8) {
                if (j0 + vk >= nk) break;
                for (int p = 0; p < 2; ++p) {
                    const uint idx = uint(p * 256) + tid;
                    const uint vr = idx >> 6, vc = (idx & 63u) << 3;
                    const int sl_i = sslot[vk + int(vr)];
                    *((threadgroup uint4*) (sKV + vr * 512u + vc)) = sl_i >= 0 ? *((device const uint4*) (cache + (size_t) sl_i * MLA_ROW + vc)) : uint4(0u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_half8x8 pf0, pf1;
                simdgroup_load(pf0, sQP + vk, 64);
                simdgroup_load(pf1, sQP + 8 * 64 + vk, 64);
                for (int j = 0; j < 8; ++j) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, sKV + sg * 64 + j * 8, 512);
                    simdgroup_multiply_accumulate(acc[0][j], pf0, vf, acc[0][j]);
                    simdgroup_multiply_accumulate(acc[1][j], pf1, vf, acc[1][j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (tid < 16) {
        const float sk = sinks[h0 + int(tid)];
        const float m_k = sm[tid];
        if (m_k == -INFINITY) sc[tid] = 0.0f;
        else {
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so / (sl[tid] * so + exp(sk - m_fin));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < 2; ++i) {
        const int row = i * 8 + int(fm);
        const float inv = sc[row];
        device half* o = ctx + ((size_t) t * a.NH + h0 + row) * MLA_KV + sg * 64 + fn;
        for (int j = 0; j < 8; ++j)
            *((device half2*) (o + j * 8)) = half2(half(acc[i][j].thread_elements()[0] * inv), half(acc[i][j].thread_elements()[1] * inv));
    }
}
