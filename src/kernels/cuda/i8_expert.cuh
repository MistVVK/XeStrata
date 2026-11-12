// src/kernels/cuda/i8_expert.cuh - our own int8 dot-product kernels for a prompt layer of experts (native_expert_i8, declared in
// include/strata/kernels/iq_kernels.hpp), for cards that have no matrix cores the fused kernels can use (RDNA2, Pascal, any
// card with only a byte dot product).  Included once by iq_kernels.cu, INSIDE its anonymous namespace (it uses that file's
// block types, codebook tables, q8_1 helpers and the dp4a wrapper).
//
// What it does, against native_expert_grouped.  The grouped kernels decode every weight block again for each pass of
// GRP_NC = 4 activation rows; an expert of a prompt chunk carries 40-390 rows.  Here a block owns a tile of
//   64 weight rows x BM = 16 TM entries of one expert
// and walks K in stages of 128 values.  Per stage every thread decodes ONE 32-value sub-block of ONE weight row into int8 words
// in shared memory (once per tile, not once per 4 entries), the tile's activations (q8_1) are staged beside them, and every
// thread then multiplies a TM x 4 register tile with `__dp4a`:
//     per 32 values and (entry, weight row):  8 dp4a, the format's integer scale(s), one float fma with d_w * d_x.
// The arithmetic is the quantized-dot arithmetic of the other native kernels: w = d_w * s * q (q the int8 code, s a small
// integer scale), x = d_x * a (a in -127..127), sum w x = d_w d_x sum s (q . a).  The scale algebra is ggml's: IQ2_XXS d (2s+1)/8
// per 32, IQ2_XS / IQ2_S d (2s+1)/8 per 16, IQ3_XXS d (2s+1)/4 per 32, IQ3_S d (2s+1) per 32, IQ4_NL / Q2_0 d.  Unlike the
// decode kernels the integer division inside the dot is not done (the exact value is used), so the error against FP32 is at
// most theirs.
//
// Two kernels and a small prologue per call:
//   i8_tiles_kernel  the tile list: tile_off[g] = first tile of group g (device side, no host sync: graph capture is fine)
//   i8_kernel<T, GU>  GU: gate/up rows of 32 features (32 gate + the same 32 up rows), SwiGLU and the q8_1 quantization of H in the
//                     epilogue (a thread holds gate and up of the same feature; the 16 lanes of a row hold the 32 features of one
//                     q8_1 block), so gate/up never go to memory; !GU: down rows, FP32 into out[ent_dst].
// Formats: gate/up IQ2_XXS(16) IQ2_XS(17) IQ3_XXS(18) IQ3_S(21) IQ2_S(22); down IQ4_NL(20) Q2_0(42).
constexpr int I8_KS = 128;     // values of K per stage
constexpr int I8_LD = 36;      // dwords per shared row of a stage: 32 + 4 (no bank conflicts for the 16-byte reads)
constexpr int I8_NT = 256;

__host__ __device__ constexpr int i8_block_bytes(int t) {
    return t == 16 ? 66 : t == 17 ? 74 : t == 22 ? 82 : t == 18 ? 98 : t == 21 ? 110 : t == 20 ? 18 : t == 42 ? 18 : 0;
}
__host__ __device__ constexpr int i8_block_vals(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }
__host__ __device__ constexpr int i8_grid_words(int t) { return t == 16 ? 512 : t == 17 ? 1024 : t == 22 ? 2048 : t == 18 ? 256 : t == 21 ? 512 : 4; }
__host__ __device__ constexpr bool i8_two_scales(int t) { return t == 17 || t == 22; }
__host__ __device__ constexpr bool i8_gu_type(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }
__host__ __device__ constexpr bool i8_d_type(int t) { return t == 20 || t == 42; }

__device__ __forceinline__ uint32_t i8_ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t i8_ld32(const uint8_t* p) { return i8_ld16(p) | (i8_ld16(p + 2) << 16); }   // 2-byte aligned only
__device__ __forceinline__ float i8_half(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// 7 sign bits of the field (the 8th is their parity), as the sign byte broadcast to 4 bytes
__device__ __forceinline__ uint32_t i8_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 codebook bytes with the sign byte `s` (broadcast) applied: bits 0-3 to qx, 4-7 to qy
__device__ __forceinline__ void i8_signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}

// bytes of sub-block `ib` of the block at `bp` (the 256-value super-block, ib 0..7; Q2_0: the 64-value block, ib 0..1; IQ4_NL: the block)
template <int T>
__device__ __forceinline__ void i8_load(const uint8_t* bp, int ib, uint32_t (&w)[5]) {
    if constexpr (T == 16) {
        w[0] = i8_ld32(bp + 2 + 8 * ib); w[1] = i8_ld32(bp + 6 + 8 * ib); w[2] = i8_ld16(bp);
    } else if constexpr (T == 17) {
        w[0] = i8_ld32(bp + 2 + 8 * ib); w[1] = i8_ld32(bp + 6 + 8 * ib); w[2] = i8_ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == 22) {
        w[0] = i8_ld32(bp + 2 + 4 * ib); w[1] = i8_ld32(bp + 34 + 4 * ib);
        w[2] = i8_ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == 18) {
        w[0] = i8_ld32(bp + 2 + 8 * ib); w[1] = i8_ld32(bp + 6 + 8 * ib); w[2] = i8_ld32(bp + 66 + 4 * ib); w[3] = i8_ld16(bp);
    } else if constexpr (T == 21) {
        w[0] = i8_ld32(bp + 2 + 8 * ib); w[1] = i8_ld32(bp + 6 + 8 * ib); w[2] = i8_ld32(bp + 74 + 4 * ib);
        w[3] = i8_ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == 20) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = i8_ld32(bp + 2 + 4 * k);
        w[4] = i8_ld16(bp);
    } else {   // Q2_0
        w[0] = i8_ld32(bp + 2 + 8 * ib); w[1] = i8_ld32(bp + 6 + 8 * ib); w[2] = i8_ld16(bp);
    }
}

// 8 nibbles through a 16-entry int8 table (4 words): the low nibbles' values in lo, the high ones' in hi
__device__ __forceinline__ void i8_table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}

// the sub-block as 8 words of 4 int8 (natural order), the integer scales of its values 0-15 (m0) and 16-31 (m1), and the float base
// scale: value = base * m * q
template <int T>
__device__ __forceinline__ void i8_convert(const uint32_t (&w)[5], const uint32_t* grid, const uint32_t (&kv)[4], uint32_t (&q)[8], int& m0,
                                           int& m1, float& base) {
    if constexpr (T == 16) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            i8_signed8(e.x, e.y, i8_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        m0 = m1 = (int) ((w[1] >> 27) | 1);
        base = i8_half(w[2]) * 0.125f;
    } else if constexpr (T == 17) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            i8_signed8(e.x, e.y, i8_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const uint32_t sc = w[2] >> 16;
        m0 = (int) (2 * (sc & 15) + 1);
        m1 = (int) (2 * ((sc >> 4) & 15) + 1);
        base = i8_half(w[2]) * 0.125f;
    } else if constexpr (T == 22) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            i8_signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const uint32_t sc = w[2] >> 24;
        m0 = (int) (2 * (sc & 15) + 1);
        m1 = (int) (2 * (sc >> 4) + 1);
        base = i8_half(w[2]) * 0.125f;
    } else if constexpr (T == 18) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            i8_signed8(grid[i0], grid[i1], i8_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        m0 = m1 = (int) (2 * (w[2] >> 28) + 1);
        base = i8_half(w[3]) * 0.25f;
    } else if constexpr (T == 21) {
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            i8_signed8(grid[i0 | ((qh << (8 - 2 * l)) & 256)], grid[i1 | ((qh << (7 - 2 * l)) & 256)], ((w[2] >> (8 * l)) & 255) * 0x01010101u,
                       q[2 * l], q[2 * l + 1]);
        }
        m0 = m1 = (int) (1 + 2 * (w[3] >> 24));
        base = i8_half(w[3]);
    } else if constexpr (T == 20) {
#pragma unroll
        for (int k = 0; k < 4; ++k) i8_table16(w[k], kv, q[k], q[4 + k]);
        m0 = m1 = 1;
        base = i8_half(w[4]);
    } else {   // Q2_0: code - 1 through the byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        m0 = m1 = 1;
        base = i8_half(w[2]);
    }
}

template <int T>
__device__ __forceinline__ const uint32_t* i8_grid_src() {
    if constexpr (T == 16) return (const uint32_t*) iq2xxs_grid;
    else if constexpr (T == 17) return (const uint32_t*) iq2xs_grid;
    else if constexpr (T == 22) return (const uint32_t*) iq2s_grid;
    else if constexpr (T == 18) return (const uint32_t*) iq3xxs_grid;
    else if constexpr (T == 21) return (const uint32_t*) iq3s_grid;
    else return nullptr;
}

// shared memory of an instance, in dwords: grid | stage weights x2 | stage activations x2 | integer scales x2 | float base scales x2 |
// activation scales x2 | source rows
template <int T, int TM>
struct I8Smem {
    static constexpr int BM = 16 * TM;
    static constexpr int GW = (i8_grid_words(T) + 3) / 4 * 4;
    static constexpr int SZ_W = 64 * I8_LD, SZ_A = BM * I8_LD;
    static constexpr int o_w = GW, o_a = o_w + 2 * SZ_W, o_m = o_a + 2 * SZ_A, o_wd = o_m + 2 * 256, o_ad = o_wd + 2 * 256,
                         o_src = o_ad + 2 * BM * 4, total = (o_src + BM + 3) / 4 * 4;
    static constexpr size_t bytes = (size_t) total * 4;
};

__global__ void __launch_bounds__(256) i8_tiles_kernel(const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups, int bm,
                                                       int32_t* __restrict__ tile_off) {
    __shared__ int s[256];
    const int ng = *n_groups, tid = threadIdx.x;
    const int per = (ng + 255) / 256;
    const int g0 = min(ng, tid * per), g1 = min(ng, g0 + per);
    int sum = 0;
    for (int g = g0; g < g1; ++g) sum += (grp_start[g + 1] - grp_start[g] + bm - 1) / bm;
    s[tid] = sum;
    __syncthreads();
    for (int o = 1; o < 256; o <<= 1) {
        const int v = tid >= o ? s[tid - o] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }
    int run = s[tid] - sum;
    for (int g = g0; g < g1; ++g) {
        tile_off[g] = run;
        run += (grp_start[g + 1] - grp_start[g] + bm - 1) / bm;
    }
    if (tid == 255) tile_off[ng] = s[255];
}

template <int T, bool GU, int TM>
__global__ void __launch_bounds__(I8_NT)
i8_kernel(const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
          const int32_t* __restrict__ tile_off, const int32_t* __restrict__ ent_idx, const block_q8_1* __restrict__ A,
          block_q8_1* __restrict__ hq, float* __restrict__ out, NativeExpertLayout L) {
    using SM = I8Smem<T, TM>;
    constexpr int BM = SM::BM, NA = (BM * I8_LD + I8_NT - 1) / I8_NT;
    extern __shared__ uint4 i8_dyn[];
    uint32_t* const smem = (uint32_t*) i8_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_w = smem + SM::o_w;
    uint32_t* const s_a = smem + SM::o_a;
    uint32_t* const s_m = smem + SM::o_m;
    float* const s_wd = (float*) (smem + SM::o_wd);
    float* const s_ad = (float*) (smem + SM::o_ad);
    int* const s_src = (int*) (smem + SM::o_src);

    const int ng = *n_groups;
    const int ti = blockIdx.x;
    if (ti >= tile_off[ng]) return;
    int lo = 0, hi = ng - 1;
    while (lo < hi) {   // the largest group whose first tile is <= ti
        const int mid = (lo + hi + 1) >> 1;
        if (tile_off[mid] <= ti) lo = mid; else hi = mid - 1;
    }
    const int g = lo;
    const int e_lo = grp_start[g] + (ti - tile_off[g]) * BM;
    const int e_hi = min(grp_start[g + 1], e_lo + BM);
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
#if defined(STRATA_I8_DBG)
    if (tid == 0 && blockIdx.y == 0 && ti < 3)
        printf("i8 dbg T=%d GU=%d TM=%d ti=%d ng=%d total=%d g=%d e_lo=%d e_hi=%d blob=%p A=%p hq=%p out=%p tok=%p\n", T, (int) GU, TM, ti, ng,
               tile_off[ng], g, e_lo, e_hi, (const void*) blob, (const void*) A, (const void*) hq, (const void*) out, (const void*) ent_idx);
#endif
    const int K = GU ? (int) L.n_embd : (int) L.n_ff;
    const int ns = K / I8_KS;
    const size_t wrb = GU ? L.gu_row : L.d_row;

    if constexpr (i8_grid_words(T) > 4) {
        const uint32_t* src = i8_grid_src<T>();
        for (int i = tid; i < i8_grid_words(T); i += I8_NT) s_grid[i] = src[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == 20) {
        const uint32_t* t = (const uint32_t*) kvalues_iq4nl;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv[i] = t[i];
    }
    if (tid < BM) s_src[tid] = (e_lo + tid < e_hi) ? (GU ? ent_idx[e_lo + tid] : e_lo + tid) : -1;
    __syncthreads();

    // the weight sub-block of this thread: row wr of the tile, 32 values wp of each stage
    const int wr = tid >> 2, wp = tid & 3;
    const uint8_t* wrow;
    if constexpr (GU) wrow = blob + (wr >= 32 ? L.up_off : (size_t) 0) + (size_t) (blockIdx.y * 32 + (wr & 31)) * wrb;
    else wrow = blob + L.down_off + (size_t) (blockIdx.y * 64 + wr) * wrb;
    // the activation dwords of this thread: item it of the stage's BM rows x 36 dwords
    const uint32_t* const Ad = (const uint32_t*) A;
    const int arow_dw = (K / 32) * 9;
    int a_off[NA], a_dst[NA];
#pragma unroll
    for (int q = 0; q < NA; ++q) {
        const int it = tid + q * I8_NT;
        if (it >= BM * 36) { a_off[q] = -1; a_dst[q] = 0x7fffffff; continue; }
        const int mr = it / 36, d = it - mr * 36, blk = d / 9, k9 = d - blk * 9;
        const int src = s_src[mr];
        a_off[q] = src >= 0 ? src * arow_dw + d : -1;
        a_dst[q] = k9 == 0 ? ~(mr * 4 + blk) : mr * I8_LD + blk * 8 + (k9 - 1);
    }

    auto load_w = [&](int st, uint32_t (&w)[5]) {
        const int sb = st * 4 + wp;
        int bi, ib;
        if constexpr (i8_block_vals(T) == 256) { bi = sb >> 3; ib = sb & 7; }
        else if constexpr (i8_block_vals(T) == 64) { bi = sb >> 1; ib = sb & 1; }
        else { bi = sb; ib = 0; }
        i8_load<T>(wrow + (size_t) bi * i8_block_bytes(T), ib, w);
    };
    auto store_w = [&](int b, const uint32_t (&w)[5]) {
        uint32_t q[8];
        int m0, m1;
        float base;
        i8_convert<T>(w, s_grid, kv, q, m0, m1, base);
        uint4* dst = (uint4*) (s_w + b * SM::SZ_W + wr * I8_LD + wp * 8);
        dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
        dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
        s_m[b * 256 + wr * 4 + wp] = (uint32_t) (m0 & 0xff) | ((uint32_t) (m1 & 0xff) << 8);
        s_wd[b * 256 + wr * 4 + wp] = base;
    };
    auto load_a = [&](int st, uint32_t (&ar)[NA]) {
#pragma unroll
        for (int q = 0; q < NA; ++q) ar[q] = a_off[q] >= 0 ? Ad[a_off[q] + st * 36] : 0u;
    };
    auto store_a = [&](int b, const uint32_t (&ar)[NA]) {
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int d = a_dst[q];
            if (d == 0x7fffffff) continue;
            if (d >= 0) s_a[b * SM::SZ_A + d] = ar[q];
            else s_ad[b * BM * 4 + ~d] = i8_half(ar[q] & 0xFFFFu);
        }
    };

    float acc[TM][4];
#pragma unroll
    for (int j = 0; j < TM; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

    uint32_t wraw[5], ar[NA];
    load_w(0, wraw);
    load_a(0, ar);
    store_w(0, wraw);
    store_a(0, ar);
    __syncthreads();
    for (int st = 0; st < ns; ++st) {
        const int b = st & 1;
        if (st + 1 < ns) { load_w(st + 1, wraw); load_a(st + 1, ar); }
        const uint32_t* const W = s_w + b * SM::SZ_W;
        const uint32_t* const Ab = s_a + b * SM::SZ_A;
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            int m0[4], m1[4];
            float bw[4];
            uint4 wv[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = tx + 16 * i;
                const uint32_t mm = s_m[b * 256 + row * 4 + blk];
                m0[i] = (int) (int8_t) (mm & 0xff);
                m1[i] = (int) (int8_t) ((mm >> 8) & 0xff);
                bw[i] = s_wd[b * 256 + row * 4 + blk];
                wv[i][0] = *(const uint4*) (W + row * I8_LD + blk * 8);
                wv[i][1] = *(const uint4*) (W + row * I8_LD + blk * 8 + 4);
            }
#pragma unroll
            for (int j = 0; j < TM; ++j) {
                const int arow = ty + 16 * j;
                const float ax = s_ad[b * BM * 4 + arow * 4 + blk];
                const uint4 a0 = *(const uint4*) (Ab + arow * I8_LD + blk * 8);
                const uint4 a1 = *(const uint4*) (Ab + arow * I8_LD + blk * 8 + 4);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int s0 = ggml_cuda_dp4a((int) wv[i][0].x, (int) a0.x, 0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].y, (int) a0.y, s0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].z, (int) a0.z, s0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].w, (int) a0.w, s0);
                    int t;
                    if constexpr (i8_two_scales(T)) {
                        int s1 = ggml_cuda_dp4a((int) wv[i][1].x, (int) a1.x, 0);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].y, (int) a1.y, s1);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].z, (int) a1.z, s1);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].w, (int) a1.w, s1);
                        t = s0 * m0[i] + s1 * m1[i];
                    } else {
                        s0 = ggml_cuda_dp4a((int) wv[i][1].x, (int) a1.x, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].y, (int) a1.y, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].z, (int) a1.z, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].w, (int) a1.w, s0);
                        t = s0 * m0[i];
                    }
                    acc[j][i] = fmaf((float) t, bw[i] * ax, acc[j][i]);
                }
            }
        }
        if (st + 1 < ns) { store_w(b ^ 1, wraw); store_a(b ^ 1, ar); }
        __syncthreads();
    }

    // epilogues
    if constexpr (GU) {
        // thread rows i = 0, 1: gate features tx, tx + 16 of the tile; i = 2, 3: the up rows of the same features
        const int nb = (int) (L.n_ff / 32);
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int e = e_lo + ty + 16 * j;
            const float g0 = acc[j][0], g1 = acc[j][1];
            const float h0 = (g0 / (1.0f + __expf(-g0))) * acc[j][2];
            const float h1 = (g1 / (1.0f + __expf(-g1))) * acc[j][3];
            float amax = fmaxf(fabsf(h0), fabsf(h1)), sum = h0 + h1;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            if (e < e_hi) {
                const float d = q8_1_finite(amax / 127.0f);
                block_q8_1* hb = hq + (size_t) e * nb + blockIdx.y;
                hb->qs[tx] = q8_1_quant(h0, d, amax);
                hb->qs[tx + 16] = q8_1_quant(h1, d, amax);
                if (tx == 0) hb->ds = q8_1_ds(d, sum);
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int e = e_lo + ty + 16 * j;
            if (e < e_hi) {
                float* o = out + (size_t) ent_idx[e] * L.n_embd + (size_t) blockIdx.y * 64 + tx;
#pragma unroll
                for (int i = 0; i < 4; ++i) o[16 * i] = acc[j][i];
            }
        }
    }
}

// ---------------------------------------------------------------- host side
struct I8Args {
    const unsigned long long* grp_ptr;
    const int32_t *grp_start, *n_groups, *tile_off, *ent_tok, *ent_dst;
    const block_q8_1* x;
    block_q8_1* hq;
    float* out;
    NativeExpertLayout L;
    int64_t tiles_ub;
};

template <int T, bool GU, int TM>
void i8_launch_tm(const I8Args& a, cudaStream_t s) {
    constexpr size_t smem = I8Smem<T, TM>::bytes;
#if !defined(__HIPCC__)
    if (smem > 48 * 1024) cudaFuncSetAttribute(i8_kernel<T, GU, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
#endif
    const dim3 grid((unsigned) a.tiles_ub, (unsigned) (GU ? a.L.n_ff / 32 : a.L.n_embd / 64));
    if constexpr (GU)
        i8_kernel<T, true, TM><<<grid, I8_NT, smem, s>>>(a.grp_ptr, a.grp_start, a.n_groups, a.tile_off, a.ent_tok, a.x, a.hq, a.out, a.L);
    else
        i8_kernel<T, false, TM><<<grid, I8_NT, smem, s>>>(a.grp_ptr, a.grp_start, a.n_groups, a.tile_off, a.ent_dst, a.hq, nullptr, a.out, a.L);
}
template <int T, bool GU>
void i8_launch(int tm, const I8Args& a, cudaStream_t s) {
    if (tm <= 2) i8_launch_tm<T, GU, 2>(a, s);
    else if (tm == 3) i8_launch_tm<T, GU, 3>(a, s);
    else i8_launch_tm<T, GU, 4>(a, s);
}

int i8_env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}
