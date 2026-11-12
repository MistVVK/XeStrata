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
//
// Iteration 4 (what changed against the iteration-3 kernel, all of it exact: the output is bit-identical):
//   * the byte dot product of a chain starts with the three-operand v_dot4_i32_i8 on gfx1030-1032 (src2 = 0), where the compiler emitted a
//     v_mov 0 and the accumulating v_dot4c for every one of the 16-32 chains of a sub-block;
//   * the integer scales and the float base scale of a (row, sub-block) are one 16-byte record in shared memory ({m0, m1, d, 0}):
//     one read per row, and the two-scale formats use mul24 + mad24 on clean registers (the packed form cost mul, mul, add);
//     the unit-scale formats (IQ4_NL, Q2_0) skip the scale multiply;
//   * the sign pattern of the i-quant grids comes from a 256-entry table in shared memory ({mask lo, +1 lo, mask hi, +1 hi}) and is
//     applied as (grid ^ mask) + 1s: the grids have no zero byte, so no borrow crosses a byte (checked: par/owni8/grid_zero_check.py);
//   * the activation stage is one thread per (row, q8_1 block): 9 dword loads from one base + immediate offsets, two 16-byte and one
//     4-byte store, no branch (the old mapping branched per dword); the weight pieces are read with a uniform base plus a per-thread
//     32-bit offset (global SADDR form) instead of 64-bit address arithmetic per piece;
//   * a tile whose last entries are empty skips the 16-entry slices that hold no entry (uniform branch; bit-identical for the others).
#ifndef I8_BLK_UNROLL
#define I8_BLK_UNROLL 1     // unroll factor of the 32-value sub-block loop of one stage (the full unroll hoisted every LDS read: 256 VGPRs and 55-200 spilled on gfx1031)
#endif
#define I8_STR_(x) #x
#define I8_PRAGMA_UNROLL(n) _Pragma(I8_STR_(unroll n))
constexpr int I8_KS = 128;     // values of K per stage
constexpr int I8_LD = 36;      // dwords per shared row of a stage: 32 + 4 (no bank conflicts for the 16-byte reads)
constexpr int I8_RS = 20;      // dwords per row of the scale records: 4 x 4 + 4 (no bank conflicts for the 16-byte reads)
constexpr int I8_NT = 256;

// Second wave (same arithmetic, new decodes): gate/up IQ4_XS (23: the IQ4_NL table with a 6-bit scale per 32 values) and IQ1_M (29: the grid of +-1 / 0 times 8, with the
// block's two delta signs folded into the int8 values: 8 (g - 1 +- 1/8) is an integer, so no activation-sum term is needed); down Q5_0 (6) and Q8_0 (8), the legacy
// 32-value blocks with one scale (the 5th bit of Q5_0 comes from the 32-bit qh word; 16 is subtracted per byte).
__host__ __device__ constexpr int i8_block_bytes(int t) {
    return t == 16 ? 66 : t == 17 ? 74 : t == 22 ? 82 : t == 18 ? 98 : t == 21 ? 110 : t == 20 ? 18 : t == 42 ? 18 : t == 23 ? 136 : t == 29 ? 56 : t == 6 ? 22 : t == 8 ? 34 : 0;
}
__host__ __device__ constexpr int i8_block_vals(int t) { return t == 20 || t == 6 || t == 8 ? 32 : t == 42 ? 64 : 256; }
__host__ __device__ constexpr int i8_grid_words(int t) { return t == 16 ? 512 : t == 17 ? 1024 : t == 22 ? 2048 : t == 18 ? 256 : t == 21 ? 512 : t == 29 ? 2048 : 4; }
__host__ __device__ constexpr bool i8_two_scales(int t) { return t == 17 || t == 22 || t == 29; }
__host__ __device__ constexpr bool i8_unit_scale(int t) { return t == 20 || t == 42 || t == 6 || t == 8; }
__host__ __device__ constexpr bool i8_wide_scale(int t) { return t == 23; }   // scale -32..31: |sum x scale| can pass 2^23, plain 32-bit multiply
__host__ __device__ constexpr bool i8_sign_type(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }
__host__ __device__ constexpr bool i8_gu_type(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42; }   // (42: the Q2_0 pack's gate/up)
__host__ __device__ constexpr bool i8_d_type(int t) { return t == 20 || t == 42 || t == 6 || t == 8; }
// raw words one thread loads for a sub-block, and the packed words the decode reads (Q8_0: the 8 code words and the scale)
template <int T> constexpr int I8NR = T == 8 ? 9 : 6;
template <int T> constexpr int I8NW = T == 8 ? 9 : T == 6 ? 6 : 5;

// Weight bytes are read through a GLOBAL address-space pointer on HIP: a flat load also counts in lgkmcnt, so every LDS wait of the dot loop then waited
// for the prefetch that was meant to overlap it (iteration 2 compiled to flat_load + an immediate s_waitcnt vmcnt(0) before the first dot).
#if defined(__HIPCC__)
#define I8_AS __attribute__((address_space(1)))
#else
#define I8_AS
#endif
using I8P = const I8_AS uint8_t*;
using I8P16 = const I8_AS uint16_t*;
using I8P32 = const I8_AS uint32_t*;
#define I8_GPTR(x) ((I8P) (uintptr_t) (x))
__device__ __forceinline__ uint32_t i8_ld16(I8P p) { return *(I8P16) p; }   // 2-byte aligned only
__device__ __forceinline__ uint32_t i8_ld8(I8P p) { return *p; }
__device__ __forceinline__ uint32_t i8_ldw(I8P p) { return *(I8P32) p; }   // 4-byte aligned
__device__ __forceinline__ float i8_half(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// byte 0 of v (v < 256) in all four bytes (one v_perm instead of a quarter-rate 32-bit multiply)
__device__ __forceinline__ uint32_t i8_bcast(uint32_t v) { return __byte_perm(v, 0u, 0x0000); }
// a pointer that is the same in every lane, as scalar registers (the compiler cannot tell: it came out of a binary search over loads)
__device__ __forceinline__ I8P i8_uniform(I8P p) {
#if defined(__HIPCC__)
    const uint64_t v = (uint64_t) (uintptr_t) p;
    const uint32_t lo = (uint32_t) __builtin_amdgcn_readfirstlane((int) (uint32_t) v), hi = (uint32_t) __builtin_amdgcn_readfirstlane((int) (uint32_t) (v >> 32));
    return (I8P) (uintptr_t) (((uint64_t) hi << 32) | lo);
#else
    return p;
#endif
}
// the first dot product of a chain: 4 byte products plus 0.  gfx1030-1032 have the three-operand v_dot4_i32_i8 (the accumulating v_dot4c needs the
// accumulator zeroed first by a v_mov, one extra instruction per chain; the compiler picks the v_dot4c form for sdot4(a, b, 0)).
__device__ __forceinline__ int i8_dot0(int a, int b) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
    int r;
    __asm__("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(r) : "v"(a), "v"(b));
    return r;
#else
    return ggml_cuda_dp4a(a, b, 0);
#endif
}

// bytes of sub-block `ib` of the block at `bp` (the 256-value super-block, ib 0..7; Q2_0: the 64-value block, ib 0..1; IQ4_NL: the block)
// The raw loads (32-bit words read at 2-byte alignment as one load, a 16-bit scale, byte pieces; nothing combined here) so that the first use of the
// loaded registers sits after the dot loop of the stage: i8_load issues them, i8_pack (called in the decode, after the dot loop) builds the words that
// i8_convert reads.  (Read as 16-bit halves the compiler merged them into one dword load and split the dword again right after the load, which put an
// s_waitcnt vmcnt in front of the dot loop.)
constexpr int I8_NRAW = 6;
typedef uint32_t i8_u32a __attribute__((aligned(2)));
#if defined(__HIPCC__)
__device__ __forceinline__ uint32_t i8_ld32(I8P p) { return *(const I8_AS i8_u32a*) p; }   // AMD global loads may be unaligned
#else
__device__ __forceinline__ uint32_t i8_ld32(I8P p) { return i8_ld16(p) | (i8_ld16(p + 2) << 16); }   // NVIDIA: a 32-bit load must be 4-byte aligned ("misaligned address" otherwise)
#endif
// per-thread byte offsets of the pieces of the thread's sub-block (the thread of sub-block wp of the stage's four), all relative to the expert's blob:
// the row's start plus wp times the stride of the piece's family (8-byte, 4-byte, 1-byte pieces per sub-block; o_h: half a byte), see i8_load
struct I8Off { uint32_t o8, o4, o1, oh, o0, o2; };
template <int T>
__device__ __forceinline__ I8Off i8_offs(uint32_t row_off, int wp) {
    I8Off o;
    o.o0 = row_off;
    o.o8 = row_off + 8 * wp;
    o.o4 = row_off + 4 * wp;
    o.o1 = row_off + wp;
    o.oh = row_off + (wp >> 1);
    o.o2 = row_off + 2 * wp;
    if constexpr (T == 20) o.o4 = row_off + 18 * wp;                                   // IQ4_NL: the block of sub-block wp
    if constexpr (T == 6) o.o4 = row_off + 22 * wp;                                    // Q5_0
    if constexpr (T == 8) o.o4 = row_off + 34 * wp;                                    // Q8_0
    if constexpr (T == 23) o.o4 = row_off + 16 * wp;                                   // IQ4_XS: the 16 code bytes of sub-block wp
    if constexpr (T == 42) { o.o0 = row_off + (wp >> 1) * 18; o.o8 = o.o0 + 8 * (wp & 1); }   // Q2_0: block wp / 2, half wp % 2
    return o;
}
// the stage's pieces.  Per stage the bases are uniform (scalar), the offsets per thread: super-block bi = st / 2 and half par = st % 2 of it
// (ib = 4 par + wp); IQ4_NL: block 4 st + wp; Q2_0: block 2 st + wp / 2.
template <int T>
__device__ __forceinline__ void i8_load(I8P blob, const I8Off& of, int st, uint32_t (&r)[I8NR<T>]) {
    const uint32_t bi = (uint32_t) st >> 1, par = (uint32_t) st & 1;
    if constexpr (T == 23) {   // IQ4_XS: super-block bi (136 bytes), sub-blocks 4 par + wp: 16 code bytes at 8 + 16 ib, d + scales_h, scales_l
        I8P a = blob + (bi * 136 + 8 + 64 * par) + of.o4;
        r[0] = i8_ld32(a); r[1] = i8_ld32(a + 4); r[2] = i8_ld32(a + 8); r[3] = i8_ld32(a + 12);
        r[4] = i8_ld32(blob + (bi * 136) + of.o0); r[5] = i8_ld32(blob + (bi * 136 + 4) + of.o0);
    } else if constexpr (T == 29) {   // IQ1_M: super-block bi (56 bytes): 4 qs bytes at 4 ib, 2 qh bytes at 32 + 2 ib, the 8 scale bytes at 48
        r[0] = i8_ld32(blob + (bi * 56 + 16 * par) + of.o4);
        r[1] = i8_ld16(blob + (bi * 56 + 32 + 8 * par) + of.o2);
        r[2] = i8_ld32(blob + (bi * 56 + 48) + of.o0); r[3] = i8_ld32(blob + (bi * 56 + 52) + of.o0);
    } else if constexpr (T == 6) {   // Q5_0: four 22-byte blocks a stage: d, qh (4 bytes), 16 code bytes
        I8P a = blob + (88u * (uint32_t) st) + of.o4;
        r[0] = i8_ld32(a + 6); r[1] = i8_ld32(a + 10); r[2] = i8_ld32(a + 14); r[3] = i8_ld32(a + 18);
        r[4] = i8_ld32(a + 2); r[5] = i8_ld16(a);
    } else if constexpr (T == 8) {   // Q8_0: four 34-byte blocks a stage: d, 32 code bytes
        I8P a = blob + (136u * (uint32_t) st) + of.o4;
#pragma unroll
        for (int k = 0; k < 8; ++k) r[k] = i8_ld32(a + 2 + 4 * k);
        r[8] = i8_ld16(a);
    } else if constexpr (T == 16 || T == 17) {
        constexpr uint32_t BB = T == 16 ? 66 : 74;
        I8P a8 = blob + (bi * BB + 32 * par) + of.o8;
        r[0] = i8_ld32(a8 + 2); r[1] = i8_ld32(a8 + 6); r[2] = i8_ld16(blob + (bi * BB) + of.o0);
        if constexpr (T == 17) r[3] = i8_ld8(blob + (bi * BB + 4 * par) + of.o1 + 66);
    } else if constexpr (T == 42) {
        I8P a8 = blob + (36u * (uint32_t) st) + of.o8;
        r[0] = i8_ld32(a8 + 2); r[1] = i8_ld32(a8 + 6); r[2] = i8_ld16(blob + (36u * (uint32_t) st) + of.o0);
    } else if constexpr (T == 22) {
        I8P a4 = blob + (bi * 82 + 16 * par) + of.o4, b1 = blob + (bi * 82 + 4 * par) + of.o1;
        r[0] = i8_ld32(a4 + 2); r[1] = i8_ld32(a4 + 34);
        r[2] = i8_ld16(blob + (bi * 82) + of.o0); r[3] = i8_ld8(b1 + 66); r[4] = i8_ld8(b1 + 74);
    } else if constexpr (T == 18) {
        I8P a8 = blob + (bi * 98 + 32 * par) + of.o8, a4 = blob + (bi * 98 + 16 * par) + of.o4;
        r[0] = i8_ld32(a8 + 2); r[1] = i8_ld32(a8 + 6); r[2] = i8_ld32(a4 + 66); r[3] = i8_ld16(blob + (bi * 98) + of.o0);
    } else if constexpr (T == 21) {
        I8P a8 = blob + (bi * 110 + 32 * par) + of.o8, a4 = blob + (bi * 110 + 16 * par) + of.o4;
        r[0] = i8_ld32(a8 + 2); r[1] = i8_ld32(a8 + 6); r[2] = i8_ld32(a4 + 74);
        r[3] = i8_ld16(blob + (bi * 110) + of.o0); r[4] = i8_ld8(blob + (bi * 110 + 4 * par) + of.o1 + 66);
        r[5] = i8_ld8(blob + (bi * 110 + 2 * par) + of.oh + 106);
    } else {   // IQ4_NL
        I8P a = blob + (72u * (uint32_t) st) + of.o4;
#pragma unroll
        for (int k = 0; k < 4; ++k) r[k] = i8_ld32(a + 2 + 4 * k);
        r[4] = i8_ld16(a);
    }
}
template <int T>
__device__ __forceinline__ void i8_pack(const uint32_t (&r)[I8NR<T>], int ib, uint32_t (&w)[I8NW<T>]) {
    if constexpr (T == 23) {
        // scale of sub-block ib: 4 low bits from scales_l[ib / 2] (nibble ib & 1) and 2 high bits from scales_h; w[4] = d | (ls << 16)
        const uint32_t ls = ((r[5] >> (4 * (ib & 1) + 8 * (ib >> 1))) & 15u) | (((r[4] >> (16 + 2 * ib)) & 3u) << 4);
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2]; w[3] = r[3];
        w[4] = (r[4] & 0xFFFFu) | (ls << 16);
    } else if constexpr (T == 29) {
        // the block's fp16 scale is spread over the top nibbles of its four 16-bit scale words; the 3-bit scales of the two halves of sub-block ib
        const uint32_t d16 = ((r[2] >> 12) & 0xFu) | ((r[2] >> 24) & 0xF0u) | ((r[3] >> 4) & 0xF00u) | ((r[3] >> 16) & 0xF000u);
        const uint32_t v = (((ib >> 2) ? r[3] : r[2]) >> (16 * ((ib >> 1) & 1))) >> (6 * (ib & 1));
        w[0] = r[0]; w[1] = r[1] & 0xFFFFu;
        w[2] = d16 | ((2u * (v & 7u) + 1u) << 16) | ((2u * ((v >> 3) & 7u) + 1u) << 24);
    } else if constexpr (T == 6) {
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2]; w[3] = r[3]; w[4] = r[4]; w[5] = r[5];
    } else if constexpr (T == 8) {
#pragma unroll
        for (int k = 0; k < 9; ++k) w[k] = r[k];
    } else if constexpr (T == 16 || T == 42) {
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2];
    } else if constexpr (T == 17) {
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2] | (r[3] << 16);
    } else if constexpr (T == 22) {
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2] | (r[3] << 16) | (r[4] << 24);
    } else if constexpr (T == 18) {
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2]; w[3] = r[3];
    } else if constexpr (T == 21) {
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2];
        w[3] = r[3] | (r[4] << 16) | (((r[5] >> (4 * (ib & 1))) & 15) << 24);
    } else {   // IQ4_NL
        w[0] = r[0]; w[1] = r[1]; w[2] = r[2]; w[3] = r[3]; w[4] = r[4];
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

// 8 grid bytes with a sign pattern applied: t = {byte mask of the 4 low values, byte mask of the 4 high values} (the +1 of a negated byte is its mask's low bit); per byte
// (g ^ 0xFF) + 1 = 256 - g = -g for a negated value (g in 1..127, so no carry), g itself otherwise
__device__ __forceinline__ void i8_signed8t(uint32_t gx, uint32_t gy, const uint2 t, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ t.x) + (t.x & 0x01010101u);
    qy = (gy ^ t.y) + (t.y & 0x01010101u);
}

// the sub-block as 8 words of 4 int8 (natural order), the integer scales of its values 0-15 (m0) and 16-31 (m1), and the float base
// scale: value = base * m * q.  sg = the sign table (shared memory) of the sign formats.
template <int T>
__device__ __forceinline__ void i8_convert(const uint32_t (&w)[I8NW<T>], const uint32_t* grid, const uint2* sg, const uint32_t (&kv)[4], uint32_t (&q)[8],
                                           int& m0, int& m1, float& base) {
    if constexpr (T == 23) {   // IQ4_XS: the IQ4_NL codes, integer scale ls - 32
#pragma unroll
        for (int k = 0; k < 4; ++k) i8_table16(w[k], kv, q[k], q[4 + k]);
        m0 = m1 = (int) (w[4] >> 16) - 32;
        base = i8_half(w[4]);
    } else if constexpr (T == 29) {   // IQ1_M: 8 (g - 1 +- 1/8) = 8 g - 7 or 8 g - 9 per byte; g (0..2) is a nibble of the grid word, so 8 g + 121 (or 119) has no carry between bytes
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t n = (w[1] >> (4 * l)) & 15u;
            const uint32_t e = grid[((w[0] >> (8 * l)) & 255u) | ((n & 7u) << 8)];
            const uint32_t K = (n & 8u) ? 0x77777777u : 0x79797979u;
            q[2 * l] = (((e & 0x0F0F0F0Fu) << 3) + K) ^ 0x80808080u;
            q[2 * l + 1] = ((((e >> 4) & 0x0F0F0F0Fu) << 3) + K) ^ 0x80808080u;
        }
        m0 = (int) ((w[2] >> 16) & 255u);
        m1 = (int) (w[2] >> 24);
        base = i8_half(w[2]) * 0.125f;
    } else if constexpr (T == 6) {   // Q5_0: (nibble | 5th bit << 4) - 16 per byte, values 0-15 from the low nibbles (qh bits 4p..), 16-31 from the high ones (qh bits 16 + 4p..)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t bl = ((((w[4] >> (4 * p)) & 15u) * 0x00204081u) & 0x01010101u) << 4;
            const uint32_t bh = ((((w[4] >> (16 + 4 * p)) & 15u) * 0x00204081u) & 0x01010101u) << 4;
            q[p] = ((((w[p] & 0x0F0F0F0Fu) | bl) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
            q[4 + p] = (((((w[p] >> 4) & 0x0F0F0F0Fu) | bh) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        m0 = m1 = 1;
        base = i8_half(w[5]);
    } else if constexpr (T == 8) {   // Q8_0: the code bytes as they are
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = w[k];
        m0 = m1 = 1;
        base = i8_half(w[8]);
    } else if constexpr (T == 16) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            i8_signed8t(e.x, e.y, sg[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
        }
        m0 = m1 = (int) ((w[1] >> 27) | 1);
        base = i8_half(w[2]) * 0.125f;
    } else if constexpr (T == 17) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            i8_signed8t(e.x, e.y, sg[c >> 9], q[2 * l], q[2 * l + 1]);
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
            i8_signed8t(e.x, e.y, sg[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
        }
        const uint32_t sc = w[2] >> 24;
        m0 = (int) (2 * (sc & 15) + 1);
        m1 = (int) (2 * (sc >> 4) + 1);
        base = i8_half(w[2]) * 0.125f;
    } else if constexpr (T == 18) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            i8_signed8t(grid[i0], grid[i1], sg[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
        }
        m0 = m1 = (int) (2 * (w[2] >> 28) + 1);
        base = i8_half(w[3]) * 0.25f;
    } else if constexpr (T == 21) {
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            i8_signed8t(grid[i0 | ((qh << (8 - 2 * l)) & 256)], grid[i1 | ((qh << (7 - 2 * l)) & 256)], sg[(w[2] >> (8 * l)) & 255], q[2 * l],
                        q[2 * l + 1]);
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
    else if constexpr (T == 29) return (const uint32_t*) iq1s_grid_gpu;   // 2048 words: 8 values of 2 bits' worth in nibbles, as the other IQ1_M kernels read it
    else return nullptr;
}

// shared memory of an instance, in dwords: grid | sign table | stage weights x2 | stage activations x2 | scale records x2 |
// activation scales x2 | source rows
template <int T, int TM>
struct I8Smem {
    static constexpr int BM = 16 * TM;
    static constexpr int GW = (i8_grid_words(T) + 3) / 4 * 4;
    static constexpr int SG = i8_sign_type(T) ? 512 : 0;
    static constexpr int SZ_W = 64 * I8_LD, SZ_A = BM * I8_LD, SZ_R = 64 * I8_RS, SZ_D = BM * 4;
    static constexpr int o_sg = GW, o_w = o_sg + SG, o_a = o_w + 2 * SZ_W, o_r = o_a + 2 * SZ_A, o_d = o_r + 2 * SZ_R, o_src = o_d + 2 * SZ_D,
                         total = (o_d + 2 * SZ_D + 3) / 4 * 4;   // (the source rows s_src: in the pad words of s_r, see i8_kernel)
    static constexpr size_t bytes = (size_t) total * 4;
};

// The smallest tile (TM 2) of every covered format fits 48 KB of shared memory per block, the limit of Pascal cards (49,152 bytes on the P100: no opt-in above
// it).  A format whose decode tables grow past it fails the build here, not on a card nobody has at hand.
constexpr size_t I8_SMEM_48K = 48 * 1024;
static_assert(I8Smem<16, 2>::bytes <= I8_SMEM_48K && I8Smem<17, 2>::bytes <= I8_SMEM_48K && I8Smem<18, 2>::bytes <= I8_SMEM_48K &&
              I8Smem<21, 2>::bytes <= I8_SMEM_48K && I8Smem<22, 2>::bytes <= I8_SMEM_48K && I8Smem<23, 2>::bytes <= I8_SMEM_48K &&
              I8Smem<29, 2>::bytes <= I8_SMEM_48K && I8Smem<20, 2>::bytes <= I8_SMEM_48K && I8Smem<42, 2>::bytes <= I8_SMEM_48K &&
              I8Smem<6, 2>::bytes <= I8_SMEM_48K && I8Smem<8, 2>::bytes <= I8_SMEM_48K,
              "the TM 2 tile of a covered format must fit 48 KB of shared memory per block");

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
          block_q8_1* __restrict__ hq, float* __restrict__ out, NativeExpertLayout L, int bisect) {
    using SM = I8Smem<T, TM>;
    constexpr int BM = SM::BM;
    extern __shared__ uint4 i8_dyn[];
    uint32_t* const smem = (uint32_t*) i8_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_sg = smem + SM::o_sg;
    uint32_t* const s_w = smem + SM::o_w;
    uint32_t* const s_a = smem + SM::o_a;
    uint32_t* const s_r = smem + SM::o_r;
    float* const s_ad = (float*) (smem + SM::o_d);
    // the BM source rows of the tile: word (i & 3) of the pad (words 16..19) of row i / 4 of the first record buffer (the records fill words 0..15 of a row)
    auto s_src = [&](int i) -> int& { return *(int*) (s_r + (i >> 2) * I8_RS + 16 + (i & 3)); };

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
    const int nv = e_hi - e_lo;   // entries of this tile (the rest of the tile is empty)
    I8P blob = i8_uniform(I8_GPTR(grp_ptr[g]));
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
#if defined(STRATA_I8_DBG)
    if (tid == 0 && blockIdx.y == 0 && ti < 3)
        printf("i8 dbg T=%d GU=%d TM=%d ti=%d ng=%d total=%d g=%d e_lo=%d e_hi=%d blob=%p A=%p hq=%p out=%p tok=%p\n", T, (int) GU, TM, ti, ng,
               tile_off[ng], g, e_lo, e_hi, (const void*) blob, (const void*) A, (const void*) hq, (const void*) out, (const void*) ent_idx);
#endif
    if (bisect == 1) return;   // research (STRATA_I8_BISECT = n): return at point n, uniform over the block (the HIP fault hunt)
    const int K = GU ? (int) L.n_embd : (int) L.n_ff;
    const int ns = K / I8_KS;
    const size_t wrb = GU ? L.gu_row : L.d_row;

    if constexpr (i8_grid_words(T) > 4) {
        const uint32_t* src = i8_grid_src<T>();
        for (int i = tid; i < i8_grid_words(T); i += I8_NT) s_grid[i] = src[i];
    }
    if constexpr (i8_sign_type(T)) {   // entry b = tid: the 8 sign bits (the ksigns formats: 7 bits and their parity), as byte masks and +1s
        const uint32_t b = (uint32_t) tid;
        uint32_t sb = b;
        if constexpr (T != 21 && T != 22) sb = (b & 127u) | (((uint32_t) __popc(b & 127u) & 1u) << 7);
        uint32_t mlo = 0, mhi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            mlo |= ((sb >> k) & 1u) ? (0xFFu << (8 * k)) : 0u;
            mhi |= ((sb >> (4 + k)) & 1u) ? (0xFFu << (8 * k)) : 0u;
        }
        *(uint2*) (s_sg + 2 * b) = make_uint2(mlo, mhi);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == 20 || T == 23) {
        const uint32_t* t = (const uint32_t*) kvalues_iq4nl;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv[i] = t[i];
    }
    if (bisect == 2) return;
    if (tid < BM) s_src(tid) = (e_lo + tid < e_hi) ? (GU ? ent_idx[e_lo + tid] : e_lo + tid) : -1;
    __syncthreads();
    if (bisect == 3) return;

    // the weight sub-block of this thread: row wr of the tile, 32 values wp of each stage
    const int wr = tid >> 2, wp = tid & 3;
    uint32_t row_off;
    if constexpr (GU) row_off = (uint32_t) ((wr >= 32 ? L.up_off : (size_t) 0) + (size_t) (blockIdx.y * 32 + (wr & 31)) * wrb);
    else row_off = (uint32_t) (L.down_off + (size_t) (blockIdx.y * 64 + wr) * wrb);
    const I8Off woff = i8_offs<T>(row_off, wp);
    // the activation blocks of this thread (the first 4 BM threads, a whole number of waves): q8_1 block (tid & 3) of row (tid >> 2) of the tile, 9 dwords
    I8P const abase = I8_GPTR(A);
    const bool a_thr = tid < 4 * BM;
    uint32_t a_vo = 0;
    if (a_thr) {
        const int src = s_src(tid >> 2);
        a_vo = (uint32_t) ((size_t) (src >= 0 ? src : 0) * (size_t) ((K / 32) * 36) + (size_t) (tid & 3) * 36);
    }

    auto load_w = [&](int st, uint32_t (&r)[I8NR<T>]) { i8_load<T>(blob, woff, st, r); };
    auto store_w = [&](int b, int st, const uint32_t (&r)[I8NR<T>]) {
        uint32_t q[8], w[I8NW<T>];
        int m0, m1;
        float base;
        i8_pack<T>(r, (st * 4 + wp) & 7, w);   // (ib of a 256-value super-block; only IQ3_S reads it)
        i8_convert<T>(w, s_grid, (const uint2*) s_sg, kv, q, m0, m1, base);
        uint4* dst = (uint4*) (s_w + b * SM::SZ_W + wr * I8_LD + wp * 8);
        dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
        dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
        *(uint4*) (s_r + b * SM::SZ_R + wr * I8_RS + wp * 4) = make_uint4((uint32_t) m0, (uint32_t) m1, __float_as_uint(base), 0u);
    };
    auto load_a = [&](int st, uint32_t (&ar)[9]) {
        if (a_thr) {
            I8P p = abase + (size_t) st * 144 + a_vo;
#pragma unroll
            for (int k = 0; k < 9; ++k) ar[k] = i8_ldw(p + 4 * k);
        }
    };
    auto store_a = [&](int b, const uint32_t (&ar)[9]) {
        if (a_thr) {
            uint4* d = (uint4*) (s_a + b * SM::SZ_A + (tid >> 2) * I8_LD + (tid & 3) * 8);
            d[0] = make_uint4(ar[1], ar[2], ar[3], ar[4]);
            d[1] = make_uint4(ar[5], ar[6], ar[7], ar[8]);
            s_ad[b * SM::SZ_D + tid] = i8_half(ar[0] & 0xFFFFu);
        }
    };

    float acc[TM][4];
#pragma unroll
    for (int j = 0; j < TM; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

    uint32_t wraw[I8NR<T>], ar[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) ar[k] = 0;
    load_w(0, wraw);
    if (bisect == 4) { if (wraw[0] == 0x12345678u) out[0] = 0.0f; return; }
    load_a(0, ar);
    if (bisect == 5) { if (wraw[0] + ar[0] == 0x12345678u) out[0] = 0.0f; return; }
    store_w(0, 0, wraw);
    if (bisect == 6) return;
    store_a(0, ar);
    __syncthreads();
    if (bisect == 7) return;
    for (int st = 0; st < ns; ++st) {
        const int b = st & 1;
        if (st + 1 < ns) { load_w(st + 1, wraw); load_a(st + 1, ar); }
        const uint32_t* const W = s_w + b * SM::SZ_W;
        const uint32_t* const Ab = s_a + b * SM::SZ_A;
        const uint32_t* const Rb = s_r + b * SM::SZ_R;
        const float* const Db = s_ad + b * SM::SZ_D;
        I8_PRAGMA_UNROLL(I8_BLK_UNROLL)
        for (int blk = 0; blk < 4; ++blk) {
            uint4 rc[4], wv[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = tx + 16 * i;
                rc[i] = *(const uint4*) (Rb + row * I8_RS + blk * 4);
                wv[i][0] = *(const uint4*) (W + row * I8_LD + blk * 8);
                wv[i][1] = *(const uint4*) (W + row * I8_LD + blk * 8 + 4);
            }
#pragma unroll
            for (int j = 0; j < TM; ++j) {
                if (16 * j >= nv) break;   // uniform: the 16-entry slices at and beyond nv hold no entry of this tile (the loop is unrolled: a forward branch)
                const int arow = ty + 16 * j;
                const float ax = Db[arow * 4 + blk];
                const uint4 a0 = *(const uint4*) (Ab + arow * I8_LD + blk * 8);
                const uint4 a1 = *(const uint4*) (Ab + arow * I8_LD + blk * 8 + 4);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int s0 = i8_dot0((int) wv[i][0].x, (int) a0.x);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].y, (int) a0.y, s0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].z, (int) a0.z, s0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].w, (int) a0.w, s0);
                    int t;
                    if constexpr (i8_two_scales(T)) {
                        int s1 = i8_dot0((int) wv[i][1].x, (int) a1.x);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].y, (int) a1.y, s1);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].z, (int) a1.z, s1);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].w, (int) a1.w, s1);
                        t = __mul24(s1, (int) rc[i].y) + __mul24(s0, (int) rc[i].x);   // |s * m| < 8.0e6 < 2^23 for every format (IQ3_XXS: 32 x 62 x 127 x 31): exact; v_mul_i32_i24 + v_mad_i32_i24
                    } else {
                        s0 = ggml_cuda_dp4a((int) wv[i][1].x, (int) a1.x, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].y, (int) a1.y, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].z, (int) a1.z, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].w, (int) a1.w, s0);
                        if constexpr (i8_unit_scale(T)) t = s0;
                        else if constexpr (i8_wide_scale(T)) t = s0 * (int) rc[i].x;
                        else t = __mul24(s0, (int) rc[i].x);
                    }
                    acc[j][i] = fmaf((float) t, __uint_as_float(rc[i].z) * ax, acc[j][i]);
                }
            }
        }
        if (st + 1 < ns) { store_w(b ^ 1, st + 1, wraw); store_a(b ^ 1, ar); }
        __syncthreads();
    }

    // epilogues
    if constexpr (GU) {
        // thread rows i = 0, 1: gate features tx, tx + 16 of the tile; i = 2, 3: the up rows of the same features
        const int nb = (int) (L.n_ff / 32);
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            if (16 * j >= nv) break;   // uniform
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

// ---------------------------------------------------------------- variant B (tile code 8): 128 weight rows x 128 entries, 4 rows x 16 entries per thread
// Iteration 5.  Iteration 4 cut the instructions of the dot loop by a quarter and the 20K time by 3 %: the loop is limited by the shared-memory reads
// (a b128 read takes 4 cycles of the LDS pipe whatever the lanes read; the 4 x 4 register tile needs 16 of them per 128 dots).  Here a lane owns 4 weight
// rows (lane + 32 i: a wave covers 32 rows with full 16-byte reads) and a wave owns 16 entries, which every lane of the wave reads from the same address
// (one request): 8 + 32 b128 reads per 512 dots instead of 4 x 16 = 64 (and half the scale reads).  The arithmetic per (row, entry) is the one of the
// other tiles in the same order (so the output is bit-identical); the stage is 64 values (two sub-blocks) because 128 rows x 128 values would not fit.
// Per stage: thread (row tid / 2, sub-block tid % 2) decodes one sub-block, thread (entry tid / 2, q8_1 block tid % 2) stages one activation block.
constexpr int I8B_BM = 128, I8B_LD = 20, I8B_KS = 64;
template <int T>
struct I8BSmem {
    static constexpr int GW = (i8_grid_words(T) + 3) / 4 * 4;
    static constexpr int SG = i8_sign_type(T) ? 512 : 0;
    static constexpr int SZ_W = 128 * I8B_LD, SZ_A = 128 * I8B_LD, SZ_R = 2 * 128 * 4, SZ_D = 2 * 128;
    static constexpr int o_sg = GW, o_w = o_sg + SG, o_a = o_w + 2 * SZ_W, o_r = o_a + 2 * SZ_A, o_d = o_r + 2 * SZ_R, o_src = o_d + 2 * SZ_D,
                         total = (o_src + I8B_BM + 3) / 4 * 4;
    static constexpr size_t bytes = (size_t) total * 4;
};
// the raw pieces of sub-block `ib` of the block at bp (one load per piece, see i8_load for the stage-uniform form used by the other tiles)
template <int T>
__device__ __forceinline__ void i8_load_bp(I8P bp, int ib, uint32_t (&r)[I8NR<T>]) {
    if constexpr (T == 23) {
        r[0] = i8_ld32(bp + 8 + 16 * ib); r[1] = i8_ld32(bp + 12 + 16 * ib); r[2] = i8_ld32(bp + 16 + 16 * ib); r[3] = i8_ld32(bp + 20 + 16 * ib);
        r[4] = i8_ld32(bp); r[5] = i8_ld32(bp + 4);
    } else if constexpr (T == 29) {
        r[0] = i8_ld32(bp + 4 * ib); r[1] = i8_ld16(bp + 32 + 2 * ib); r[2] = i8_ld32(bp + 48); r[3] = i8_ld32(bp + 52);
    } else if constexpr (T == 6) {
        r[0] = i8_ld32(bp + 6); r[1] = i8_ld32(bp + 10); r[2] = i8_ld32(bp + 14); r[3] = i8_ld32(bp + 18); r[4] = i8_ld32(bp + 2); r[5] = i8_ld16(bp);
    } else if constexpr (T == 8) {
#pragma unroll
        for (int k = 0; k < 8; ++k) r[k] = i8_ld32(bp + 2 + 4 * k);
        r[8] = i8_ld16(bp);
    } else if constexpr (T == 16 || T == 17 || T == 42) {
        r[0] = i8_ld32(bp + 2 + 8 * ib); r[1] = i8_ld32(bp + 6 + 8 * ib); r[2] = i8_ld16(bp);
        if constexpr (T == 17) r[3] = i8_ld8(bp + 66 + ib);
    } else if constexpr (T == 22) {
        r[0] = i8_ld32(bp + 2 + 4 * ib); r[1] = i8_ld32(bp + 34 + 4 * ib);
        r[2] = i8_ld16(bp); r[3] = i8_ld8(bp + 66 + ib); r[4] = i8_ld8(bp + 74 + ib);
    } else if constexpr (T == 18) {
        r[0] = i8_ld32(bp + 2 + 8 * ib); r[1] = i8_ld32(bp + 6 + 8 * ib); r[2] = i8_ld32(bp + 66 + 4 * ib); r[3] = i8_ld16(bp);
    } else if constexpr (T == 21) {
        r[0] = i8_ld32(bp + 2 + 8 * ib); r[1] = i8_ld32(bp + 6 + 8 * ib); r[2] = i8_ld32(bp + 74 + 4 * ib);
        r[3] = i8_ld16(bp); r[4] = i8_ld8(bp + 66 + ib); r[5] = i8_ld8(bp + 106 + ib / 2);
    } else {   // IQ4_NL
#pragma unroll
        for (int k = 0; k < 4; ++k) r[k] = i8_ld32(bp + 2 + 4 * k);
        r[4] = i8_ld16(bp);
    }
}

template <int T, bool GU>
__global__ void __launch_bounds__(I8_NT)
i8b_kernel(const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
           const int32_t* __restrict__ tile_off, const int32_t* __restrict__ ent_idx, const block_q8_1* __restrict__ A,
           block_q8_1* __restrict__ hq, float* __restrict__ out, NativeExpertLayout L, int bisect) {
    using SM = I8BSmem<T>;
    extern __shared__ uint4 i8_dyn[];
    uint32_t* const smem = (uint32_t*) i8_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_sg = smem + SM::o_sg;
    uint32_t* const s_w = smem + SM::o_w;
    uint32_t* const s_a = smem + SM::o_a;
    uint32_t* const s_r = smem + SM::o_r;
    float* const s_ad = (float*) (smem + SM::o_d);
    int* const s_src = (int*) (smem + SM::o_src);

    const int ng = *n_groups;
    const int ti = blockIdx.x;
    if (ti >= tile_off[ng]) return;
    int lo = 0, hi = ng - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_off[mid] <= ti) lo = mid; else hi = mid - 1;
    }
    const int g = lo;
    const int e_lo = grp_start[g] + (ti - tile_off[g]) * I8B_BM;
    const int e_hi = min(grp_start[g + 1], e_lo + I8B_BM);
    const int nv = e_hi - e_lo;
    I8P blob = i8_uniform(I8_GPTR(grp_ptr[g]));
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int K = GU ? (int) L.n_embd : (int) L.n_ff;
    const int ns = K / I8B_KS;
    const size_t wrb = GU ? L.gu_row : L.d_row;

    if constexpr (i8_grid_words(T) > 4) {
        const uint32_t* src = i8_grid_src<T>();
        for (int i = tid; i < i8_grid_words(T); i += I8_NT) s_grid[i] = src[i];
    }
    if constexpr (i8_sign_type(T)) {
        const uint32_t b = (uint32_t) tid;
        uint32_t sb = b;
        if constexpr (T != 21 && T != 22) sb = (b & 127u) | (((uint32_t) __popc(b & 127u) & 1u) << 7);
        uint32_t mlo = 0, mhi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            mlo |= ((sb >> k) & 1u) ? (0xFFu << (8 * k)) : 0u;
            mhi |= ((sb >> (4 + k)) & 1u) ? (0xFFu << (8 * k)) : 0u;
        }
        *(uint2*) (s_sg + 2 * b) = make_uint2(mlo, mhi);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == 20 || T == 23) {
        const uint32_t* t = (const uint32_t*) kvalues_iq4nl;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv[i] = t[i];
    }
    if (tid < I8B_BM) s_src[tid] = (e_lo + tid < e_hi) ? (GU ? ent_idx[e_lo + tid] : e_lo + tid) : -1;
    __syncthreads();

    // weights: thread = (tile row wr, sub-block wp of the stage's two); the tile's rows 0..63 are gate features, 64..127 up features (GU), or 128 down rows
    const int wr = tid >> 1, wp = tid & 1;
    size_t row_off;
    if constexpr (GU) row_off = (wr >= 64 ? L.up_off : (size_t) 0) + (size_t) (blockIdx.y * 64 + (wr & 63)) * wrb;
    else row_off = L.down_off + (size_t) (blockIdx.y * 128 + wr) * wrb;
    I8P const wrowb = blob + row_off;
    // activations: thread = (entry ea, q8_1 block ba of the stage's two)
    const int ea = tid >> 1, ba = tid & 1;
    I8P const abase = I8_GPTR(A);
    uint32_t a_vo;
    {
        const int src = s_src[ea];
        a_vo = (uint32_t) ((size_t) (src >= 0 ? src : 0) * (size_t) ((K / 32) * 36) + (size_t) ba * 36);
    }

    auto load_w = [&](int st, uint32_t (&r)[I8NR<T>]) {
        const int sb = 2 * st + wp;
        int bi, ib;
        if constexpr (i8_block_vals(T) == 256) { bi = sb >> 3; ib = sb & 7; }
        else if constexpr (i8_block_vals(T) == 64) { bi = sb >> 1; ib = sb & 1; }
        else { bi = sb; ib = 0; }
        i8_load_bp<T>(wrowb + (size_t) bi * i8_block_bytes(T), ib, r);
    };
    auto store_w = [&](int b, int st, const uint32_t (&r)[I8NR<T>]) {
        uint32_t q[8], w[I8NW<T>];
        int m0, m1;
        float base;
        i8_pack<T>(r, (2 * st + wp) & 7, w);
        i8_convert<T>(w, s_grid, (const uint2*) s_sg, kv, q, m0, m1, base);
        uint4* dst = (uint4*) (s_w + b * SM::SZ_W + wr * I8B_LD + wp * 8);
        dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
        dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
        *(uint4*) (s_r + b * SM::SZ_R + (wp * 128 + wr) * 4) = make_uint4((uint32_t) m0, (uint32_t) m1, __float_as_uint(base), 0u);
    };
    auto load_a = [&](int st, uint32_t (&ar)[9]) {
        I8P p = abase + (size_t) st * 72 + a_vo;
#pragma unroll
        for (int k = 0; k < 9; ++k) ar[k] = i8_ldw(p + 4 * k);
    };
    auto store_a = [&](int b, const uint32_t (&ar)[9]) {
        uint4* d = (uint4*) (s_a + b * SM::SZ_A + ea * I8B_LD + ba * 8);
        d[0] = make_uint4(ar[1], ar[2], ar[3], ar[4]);
        d[1] = make_uint4(ar[5], ar[6], ar[7], ar[8]);
        s_ad[b * SM::SZ_D + ba * 128 + ea] = i8_half(ar[0] & 0xFFFFu);
    };

    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

    uint32_t wraw[I8NR<T>], ar[9];
    load_w(0, wraw);
    load_a(0, ar);
    store_w(0, 0, wraw);
    store_a(0, ar);
    __syncthreads();
    const int ebase = 16 * wave;
    for (int st = 0; st < ns; ++st) {
        const int b = st & 1;
        if (st + 1 < ns) { load_w(st + 1, wraw); load_a(st + 1, ar); }
        const uint32_t* const W = s_w + b * SM::SZ_W;
        const uint32_t* const Ab = s_a + b * SM::SZ_A;
        const uint32_t* const Rb = s_r + b * SM::SZ_R;
        const float* const Db = s_ad + b * SM::SZ_D;
#pragma unroll 1
        for (int blk = 0; blk < 2; ++blk) {
            uint4 rc[4], wv[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = lane + 32 * i;
                rc[i] = *(const uint4*) (Rb + blk * 512 + row * 4);
                wv[i][0] = *(const uint4*) (W + row * I8B_LD + blk * 8);
                wv[i][1] = *(const uint4*) (W + row * I8B_LD + blk * 8 + 4);
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                if (ebase + j < nv) {   // uniform per wave: no entry of this tile from here on otherwise
                const int e = ebase + j;
                const float ax = Db[blk * 128 + e];
                const uint4 a0 = *(const uint4*) (Ab + e * I8B_LD + blk * 8);
                const uint4 a1 = *(const uint4*) (Ab + e * I8B_LD + blk * 8 + 4);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int s0 = i8_dot0((int) wv[i][0].x, (int) a0.x);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].y, (int) a0.y, s0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].z, (int) a0.z, s0);
                    s0 = ggml_cuda_dp4a((int) wv[i][0].w, (int) a0.w, s0);
                    int t;
                    if constexpr (i8_two_scales(T)) {
                        int s1 = i8_dot0((int) wv[i][1].x, (int) a1.x);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].y, (int) a1.y, s1);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].z, (int) a1.z, s1);
                        s1 = ggml_cuda_dp4a((int) wv[i][1].w, (int) a1.w, s1);
                        t = __mul24(s1, (int) rc[i].y) + __mul24(s0, (int) rc[i].x);
                    } else {
                        s0 = ggml_cuda_dp4a((int) wv[i][1].x, (int) a1.x, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].y, (int) a1.y, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].z, (int) a1.z, s0);
                        s0 = ggml_cuda_dp4a((int) wv[i][1].w, (int) a1.w, s0);
                        if constexpr (i8_unit_scale(T)) t = s0;
                        else if constexpr (i8_wide_scale(T)) t = s0 * (int) rc[i].x;
                        else t = __mul24(s0, (int) rc[i].x);
                    }
                    acc[j][i] = fmaf((float) t, __uint_as_float(rc[i].z) * ax, acc[j][i]);
                }
            }}
        }
        if (st + 1 < ns) { store_w(b ^ 1, st + 1, wraw); store_a(b ^ 1, ar); }
        __syncthreads();
    }

    if constexpr (GU) {
        // lane = feature; thread rows i = 0, 1: gate features lane, lane + 32; i = 2, 3: up features of the same; the 32 lanes of a wave are one q8_1 block of H
        const int nb = (int) (L.n_ff / 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (ebase + j < nv) {   // uniform
            const int e = e_lo + ebase + j;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float g0 = acc[j][half], u0 = acc[j][2 + half];
                const float h0 = (g0 / (1.0f + __expf(-g0))) * u0;
                float amax = fabsf(h0), sum = h0;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                    sum += __shfl_xor_sync(0xffffffffu, sum, o);
                }
                const float d = q8_1_finite(amax / 127.0f);
                block_q8_1* hb = hq + (size_t) e * nb + 2 * blockIdx.y + half;
                hb->qs[lane] = q8_1_quant(h0, d, amax);
                if (lane == 0) hb->ds = q8_1_ds(d, sum);
            }
            }}
    } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (ebase + j < nv) {
            const int e = e_lo + ebase + j;
            float* o = out + (size_t) ent_idx[e] * L.n_embd + (size_t) blockIdx.y * 128 + lane;
#pragma unroll
            for (int i = 0; i < 4; ++i) o[32 * i] = acc[j][i];
        }}
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
    int bisect;
};

// what the last native_expert_i8 call asked its two launches for (the tile code, the dynamic shared memory in bytes passed as the launch's third argument, and
// the launch's own status, read without clearing it): native_expert_i8_last_launch, printed by the bench
struct I8LastLaunch {
    int tile = 0, gu_err = 0, down_err = 0;
    size_t gu_bytes = 0, down_bytes = 0;
};
I8LastLaunch g_i8_last;
void i8_note_launch(bool gu, int tile, size_t bytes) {
    const int e = (int) cudaPeekAtLastError();
    g_i8_last.tile = tile;
    if (gu) { g_i8_last.gu_bytes = bytes; g_i8_last.gu_err = e; }
    else { g_i8_last.down_bytes = bytes; g_i8_last.down_err = e; }
}

template <int T, bool GU, int TM>
void i8_launch_tm(const I8Args& a, cudaStream_t s) {
    constexpr size_t smem = I8Smem<T, TM>::bytes;
#if !defined(__HIPCC__)
    if (smem > 48 * 1024) cudaFuncSetAttribute(i8_kernel<T, GU, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
#endif
    const dim3 grid((unsigned) a.tiles_ub, (unsigned) (GU ? a.L.n_ff / 32 : a.L.n_embd / 64));
    if constexpr (GU)
        i8_kernel<T, true, TM><<<grid, I8_NT, smem, s>>>(a.grp_ptr, a.grp_start, a.n_groups, a.tile_off, a.ent_tok, a.x, a.hq, a.out, a.L, a.bisect);
    else
        i8_kernel<T, false, TM><<<grid, I8_NT, smem, s>>>(a.grp_ptr, a.grp_start, a.n_groups, a.tile_off, a.ent_dst, a.hq, nullptr, a.out, a.L, a.bisect);
    i8_note_launch(GU, TM, smem);
}
template <int T, bool GU>
void i8b_launch(const I8Args& a, cudaStream_t s) {
    constexpr size_t smem = I8BSmem<T>::bytes;
#if !defined(__HIPCC__)
    if (smem > 48 * 1024) cudaFuncSetAttribute(i8b_kernel<T, GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
#endif
    const dim3 grid((unsigned) a.tiles_ub, (unsigned) (GU ? a.L.n_ff / 64 : a.L.n_embd / 128));
    if constexpr (GU)
        i8b_kernel<T, true><<<grid, I8_NT, smem, s>>>(a.grp_ptr, a.grp_start, a.n_groups, a.tile_off, a.ent_tok, a.x, a.hq, a.out, a.L, a.bisect);
    else
        i8b_kernel<T, false><<<grid, I8_NT, smem, s>>>(a.grp_ptr, a.grp_start, a.n_groups, a.tile_off, a.ent_dst, a.hq, nullptr, a.out, a.L, a.bisect);
    i8_note_launch(GU, 8, smem);
}
// shared memory of the instances of a gate/up and a down type for tile code tm (2, 3, 4: rows of 16 entries; 8: variant B), and the device's limit per block
template <int T>
size_t i8_sm(int tm) {
    return tm == 8 ? I8BSmem<T>::bytes : tm <= 2 ? I8Smem<T, 2>::bytes : tm == 3 ? I8Smem<T, 3>::bytes : I8Smem<T, 4>::bytes;
}
size_t i8_smem_bytes(int gu, int dt, int tm) {
    const size_t a = gu == 16 ? i8_sm<16>(tm) : gu == 17 ? i8_sm<17>(tm) : gu == 18 ? i8_sm<18>(tm) : gu == 21 ? i8_sm<21>(tm) : gu == 23 ? i8_sm<23>(tm) :
                     gu == 29 ? i8_sm<29>(tm) : gu == 42 ? i8_sm<42>(tm) : i8_sm<22>(tm);
    const size_t b = dt == 20 ? i8_sm<20>(tm) : dt == 6 ? i8_sm<6>(tm) : dt == 8 ? i8_sm<8>(tm) : i8_sm<42>(tm);
    return a > b ? a : b;
}
int i8_env_int(const char* name, int dflt);
#if !defined(__HIPCC__)
// Diagnostic only (native_expert_i8_probe_launch, used by native_i8_parity): does a launch of an empty kernel with this much dynamic shared memory run?  The
// attribute call is made first and its refusal ignored (the launch decides, as in the real launch path); every failed attempt is read and cleared at once and an
// accepted launch is waited for, so the probe leaves the context clean.  Nothing in the engine's path calls it.
__global__ void i8_smem_probe_kernel() {
    extern __shared__ uint4 i8_probe_dyn[];
    if (threadIdx.x == 0xFFFFu) i8_probe_dyn[0] = make_uint4(0, 0, 0, 0);
}
bool i8_smem_launches(size_t bytes) {
    cudaGetLastError();
    if (bytes > 48 * 1024 && cudaFuncSetAttribute(i8_smem_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) != cudaSuccess) cudaGetLastError();
    i8_smem_probe_kernel<<<1, 1, bytes>>>();
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}
#endif
size_t i8_smem_limit() {
    int dev = 0, v = 0;
    size_t lim = 48 * 1024;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess && v > 0) lim = (size_t) v;
    const int o = i8_env_int("STRATA_I8_SMEM_LIMIT", 0);   // research / the parity test: pretend a smaller device
    return o > 0 && (size_t) o < lim ? (size_t) o : lim;
}
template <int T, bool GU>
void i8_launch(int tm, const I8Args& a, cudaStream_t s) {
    if (tm == 8) i8b_launch<T, GU>(a, s);
    else if (tm <= 2) i8_launch_tm<T, GU, 2>(a, s);
    else if (tm == 3) i8_launch_tm<T, GU, 3>(a, s);
    else i8_launch_tm<T, GU, 4>(a, s);
}

int i8_env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}
