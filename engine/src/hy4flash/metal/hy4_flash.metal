// hy4_flash.metal - MLA attention for PREFILL chunks on simdgroup_matrix (flash style, online softmax). Compiled with hy4.metal (MlaArgs, MLA_ROW, MLA_KV).
//
// One threadgroup (8 simdgroups) = one query token x 32 heads (NH is 32 or 64: grid.x = NH / 32). The 32 heads are the rows of the matrices, they share every key of the token
// (latent MQA). Keys are taken in blocks of 16: the block's rows (576 halves each: 512 latent | 64 rope) are gathered into threadgroup memory (slots by mode: causal, ring window, list),
//   S[32][16] = Q . K^T      simdgroup (rb = sg/2, cb = sg%2) owns one 8x8 tile; its 72 Q fragments (8 dims each) stay in registers
//   m, l, correction per row (one thread per row), P = exp(S - m) as fp16
//   O[32][512] = O * corr + P . V    simdgroup sg owns the 64 columns sg*64.., all 4 row blocks: 32 accumulators of 8x8
// The learnable sink is the initial state of every row (m = sink, l = 1, no value). Output ctx half [T][NH][512] = O / l. The rescale of the register accumulators by the per-row correction uses the
// 8x8 fragment layout (row fm, columns fn, fn+1 of every lane).
template <bool QREG>
kernel void mla_flash_t(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device half* ctx [[buffer(3)]], constant MlaArgs& a [[buffer(4)]],
                      device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sK[16 * MLA_ROW];
    threadgroup float sS[4][32 * 16];
    threadgroup half sP[32 * 16];
    threadgroup float sm[32], sl[32], sc[32];
    threadgroup int sslot[16];
    const int h0 = tgp.x * 32, t = tgp.y;
    const int pos = a.p0 + t;
    int nk, lo = 0;
    if (a.mode == 0) nk = pos + 1;
    else if (a.mode == 1) {
        lo = max(0, pos - a.window + 1);
        nk = pos - lo + 1;
    } else nk = nkeys[t];
    const int rb = sg >> 1, cb = sg & 1;
    // Q fragments of this simdgroup's row block: rows h0 + rb*8 .. +7 of token t
    device const half* qrow = qa + ((size_t) t * a.NH + h0 + rb * 8) * MLA_ROW;
    simdgroup_half8x8 qf[QREG ? 72 : 1];
    if (QREG)
        for (int kk = 0; kk < 72; ++kk) simdgroup_load(qf[QREG ? kk : 0], qrow + kk * 8, MLA_ROW);
    simdgroup_float8x8 acc[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (tid < 32) {
        sm[tid] = sinks[h0 + tid];
        sl[tid] = 1.0f;
    }
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int j0 = 0; j0 < nk; j0 += 16) {
        // slots of the 16 keys
        if (tid < 16) {
            const int j = j0 + tid;
            int slot = 0;
            if (j < nk) {
                if (a.mode == 0) slot = a.ring > 0 ? j % a.ring : j;
                else if (a.mode == 1) slot = (lo + j) % a.ring;
                else slot = keys[(size_t) t * a.kmax + j];
            }
            sslot[tid] = j < nk ? slot : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // stage K rows (half4 loads): 16 rows x 144 half4
        for (int idx = tid; idx < 16 * (MLA_ROW / 4); idx += 256) {
            const int r = idx / (MLA_ROW / 4), c4 = idx - r * (MLA_ROW / 4);
            const int slot = sslot[r];
            half4 v = slot >= 0 ? *((device const half4*) (cache + (size_t) slot * MLA_ROW) + c4) : half4(0.0h);
            *((threadgroup half4*) (sK + r * MLA_ROW) + c4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T: four independent partial products (k steps kk % 4), summed by the softmax thread
        {
            simdgroup_float8x8 s[4];
            for (int i = 0; i < 4; ++i) s[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kk = 0; kk < 72; kk += 4) {
                simdgroup_half8x8 kf[4], qq[4];
                for (int i = 0; i < 4; ++i) {
                    simdgroup_load(kf[i], sK + (cb * 8) * MLA_ROW + (kk + i) * 8, MLA_ROW, ulong2(0, 0), true);
                    if (QREG) qq[i] = qf[QREG ? kk + i : 0];
                    else simdgroup_load(qq[i], qrow + (kk + i) * 8, MLA_ROW);
                }
                for (int i = 0; i < 4; ++i) simdgroup_multiply_accumulate(s[i], qq[i], kf[i], s[i]);
            }
            for (int i = 0; i < 4; ++i) simdgroup_store(s[i], sS[i] + (rb * 8) * 16 + cb * 8, 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax statistics: one thread per row
        if (tid < 32) {
            const int nval = min(16, nk - j0);
            float mx = sm[tid];
            float v[16];
            for (int j = 0; j < 16; ++j) {
                v[j] = j < nval ? (sS[0][tid * 16 + j] + sS[1][tid * 16 + j] + sS[2][tid * 16 + j] + sS[3][tid * 16 + j]) * a.scale : -INFINITY;
                mx = max(mx, v[j]);
            }
            const float corr = exp(sm[tid] - mx);
            float ls = sl[tid] * corr;
            for (int j = 0; j < 16; ++j) {
                const float p = j < nval ? exp(v[j] - mx) : 0.0f;
                ls += p;
                sP[tid * 16 + j] = half(p);
            }
            sm[tid] = mx;
            sl[tid] = ls;
            sc[tid] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = O * corr + P . V
        for (int i = 0; i < 4; ++i) {
            const float c = sc[i * 8 + fm];
            for (int j = 0; j < 8; ++j) {
                acc[i][j].thread_elements()[0] *= c;
                acc[i][j].thread_elements()[1] *= c;
            }
        }
        for (int kk2 = 0; kk2 < 2; ++kk2) {
            simdgroup_half8x8 pf[4];
            for (int i = 0; i < 4; ++i) simdgroup_load(pf[i], sP + (i * 8) * 16 + kk2 * 8, 16);
            for (int j = 0; j < 8; ++j) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, sK + (kk2 * 8) * MLA_ROW + sg * 64 + j * 8, MLA_ROW);
                for (int i = 0; i < 4; ++i) simdgroup_multiply_accumulate(acc[i][j], pf[i], vf, acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // output
    for (int i = 0; i < 4; ++i) {
        const int row = i * 8 + fm;
        const float inv = 1.0f / sl[row];
        device half* o = ctx + ((size_t) t * a.NH + h0 + row) * MLA_KV + sg * 64 + fn;
        for (int j = 0; j < 8; ++j) {
            *((device half2*) (o + j * 8)) = half2(half(acc[i][j].thread_elements()[0] * inv), half(acc[i][j].thread_elements()[1] * inv));
        }
    }
}

template [[host_name("mla_flash")]] kernel void mla_flash_t<false>(device const half*, device const half*, device const float*, device half*, constant MlaArgs&, device const int*, device const int*, uint2,
                                                                  uint, uint, uint);
template [[host_name("mla_flash_q")]] kernel void mla_flash_t<true>(device const half*, device const half*, device const float*, device half*, constant MlaArgs&, device const int*, device const int*, uint2,
                                                                   uint, uint, uint);
