// The prompt path's weight tiles (moe_mma.cuh) for the 4-bit codebook and i-quant routed experts of the IQ3_S set: Q4_0 (what the Q2_0
// experts are repacked to at load, tensor_convert.hpp), IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS and IQ2_S. Each value is ggml's own dequantization
// (ggml-cuda/dequantize.cuh dequantize_*: the f32 product d * scale * grid * sign) rounded once to f16, so the f16 operands are the ones
// llama.cpp's cuBLAS path would have made. A thread takes one 32-value sub-block of a row per k-step (MM_TPR = 2 of them side by side
// in the step's 64 columns) like the Q5_1 and Q8_0 tiles; the 256-value types find theirs at (k0 % 256) / 32 + u in the block.
// First version: the raw bytes are read with 16- and 32-bit loads (the blocks of 110, 98 and 82 bytes are only 2-byte aligned) and the
// grids come from ggml's tables in global memory (cached); not yet tuned for speed.

__device__ __forceinline__ uint32_t mq_ld16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ uint32_t mq_ld32(const uint8_t* p) { return mq_ld16(p) | (mq_ld16(p + 2) << 16); }
__device__ __forceinline__ float mq_half(uint32_t h) { return __half2float(__ushort_as_half((uint16_t) h)); }
// 32 f32 values to f16 pairs and into the tile row
__device__ __forceinline__ void mq_put32(__half* dst, const float (&f)[32]) {
    __half2 y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    mm_store8(dst, y);
    mm_store8(dst + 16, y + 8);
}
__device__ __forceinline__ void mq_zero32(__half* dst) {
    float z[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) z[i] = 0.f;
    mq_put32(dst, z);
}

// Q4_0 (value (nibble - 8) * d) and IQ4_NL (kvalues_iq4nl[nibble] * d): 18 bytes, low nibbles are values 0..15, high 16..31
template <bool NL>
struct MmTile4 {
    uint16_t v[9];
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 32 + u) * 18;
#pragma unroll
            for (int l = 0; l < 9; ++l) v[l] = (uint16_t) mq_ld16(b + 2 * l);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const float d = mq_half(v[0]);
        float f[32];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t q = (uint32_t) v[1 + 2 * w] | ((uint32_t) v[2 + 2 * w] << 16);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int lo = (q >> (8 * k)) & 0xF, hi = (q >> (8 * k + 4)) & 0xF;
                f[4 * w + k] = d * (float) (NL ? (int) kvalues_iq4nl[lo] : lo - 8);
                f[16 + 4 * w + k] = d * (float) (NL ? (int) kvalues_iq4nl[hi] : hi - 8);
            }
        }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};
template <> struct MmTile<T_Q4_0> : MmTile4<false> {};
template <> struct MmTile<T_IQ4_NL> : MmTile4<true> {};

// IQ4_XS: 136 bytes = d (f16), scales_h (u16), scales_l[4], qs[128]; sub-block ib has 16 qs bytes at 8 + 16 ib and a 6-bit scale
template <> struct MmTile<T_IQ4_XS> {
    uint32_t hd[2], q[4];
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 136;
            hd[0] = mq_ld32(b);
            hd[1] = mq_ld32(b + 4);
#pragma unroll
            for (int l = 0; l < 4; ++l) q[l] = mq_ld32(b + 8 + 16 * ib + 4 * l);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const uint32_t sh = hd[0] >> 16, sl = (hd[1] >> (8 * (ib / 2))) & 0xFF;
        const int ls = (int) (((sl >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4));
        const float d = mq_half(hd[0] & 0xFFFF) * (float) (ls - 32);
        float f[32];
#pragma unroll
        for (int l = 0; l < 4; ++l)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t byte = (q[l] >> (8 * k)) & 0xFF;
                f[4 * l + k] = d * (float) kvalues_iq4nl[byte & 0xF];
                f[16 + 4 * l + k] = d * (float) kvalues_iq4nl[byte >> 4];
            }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};

__device__ __forceinline__ void mq_signed4(float* y, uint32_t grid, float d, uint32_t signs, int s0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) y[j] = d * (float) ((grid >> (8 * j)) & 0xFF) * ((signs & kmask_iq2xs[s0 + j]) ? -1.f : 1.f);
}

// IQ3_S: 110 bytes = d, qs[64], qh[8], signs[32], scales[4]; sub-block ib: qs[8 ib..], qh[ib], signs[4 ib..], a 4-bit scale (1 + 2 s) d
template <> struct MmTile<T_IQ3_S> {
    uint32_t dq, qs[2], sg, qh, sc;   // d (low 16 bits of dq), 8 qs bytes, 4 sign bytes, qh byte, scales byte
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 110;
            dq = mq_ld16(b);
            qs[0] = mq_ld32(b + 2 + 8 * ib);
            qs[1] = mq_ld32(b + 6 + 8 * ib);
            qh = b[66 + ib];
            sg = mq_ld32(b + 74 + 4 * ib);
            sc = b[106 + ib / 2];
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const float d = mq_half(dq) * (float) (1 + 2 * ((sc >> (4 * (ib % 2))) & 0xF));
        float f[32];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q0 = (qs[(2 * il) / 4] >> (8 * ((2 * il) % 4))) & 0xFF, q1 = (qs[(2 * il + 1) / 4] >> (8 * ((2 * il + 1) % 4))) & 0xFF;
            const uint32_t g1 = iq3s_grid[q0 | ((qh << (8 - 2 * il)) & 256)], g2 = iq3s_grid[q1 | ((qh << (7 - 2 * il)) & 256)];
            const uint32_t sb = (sg >> (8 * il)) & 0xFF;
            mq_signed4(f + 8 * il, g1, d, sb, 0);
            mq_signed4(f + 8 * il + 4, g2, d, sb, 4);
        }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};

// IQ3_XXS: 98 bytes = d, qs[96]: 64 grid indices (8 per sub-block) then per sub-block a u32 of four 7-bit sign codes and a 4-bit scale at the top
template <> struct MmTile<T_IQ3_XXS> {
    uint32_t dq, qs[2], aux;
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 98;
            dq = mq_ld16(b);
            qs[0] = mq_ld32(b + 2 + 8 * ib);
            qs[1] = mq_ld32(b + 6 + 8 * ib);
            aux = mq_ld32(b + 66 + 4 * ib);
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        const float d = mq_half(dq) * (0.5f + (float) (aux >> 28)) * 0.5f;
        float f[32];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q0 = (qs[(2 * il) / 4] >> (8 * ((2 * il) % 4))) & 0xFF, q1 = (qs[(2 * il + 1) / 4] >> (8 * ((2 * il + 1) % 4))) & 0xFF;
            const uint32_t sb = ksigns_iq2xs[(aux >> (7 * il)) & 127];
            mq_signed4(f + 8 * il, iq3xxs_grid[q0], d, sb, 0);
            mq_signed4(f + 8 * il + 4, iq3xxs_grid[q1], d, sb, 4);
        }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};

// IQ2_S: 82 bytes = d, qs[32] (grid index low bytes, 4 per sub-block), signs[32] (4 per sub-block), qh[8], scales[8]; 10-bit grid index of 8 values
template <> struct MmTile<T_IQ2_S> {
    uint32_t dq, qs, sg, qh, sc;
    int ib;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        ib = (k0 % 256) / 32 + u;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 82;
            dq = mq_ld16(b);
            qs = mq_ld32(b + 2 + 4 * ib);
            sg = mq_ld32(b + 34 + 4 * ib);
            qh = b[66 + ib];
            sc = b[74 + ib];
        }
    }
    __device__ __forceinline__ void put(int, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        if (!live) {
            mq_zero32(sw + r * MM_LD + u * 32);
            return;
        }
        float f[32];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint64_t grid = iq2s_grid[((qs >> (8 * il)) & 0xFF) | ((qh << (8 - 2 * il)) & 0x300)];
            const float d = mq_half(dq) * (0.5f + (float) ((sc >> (4 * (il / 2))) & 0xF)) * 0.25f;
            const uint32_t sb = (sg >> (8 * il)) & 0xFF;
            mq_signed4(f + 8 * il, (uint32_t) grid, d, sb, 0);
            mq_signed4(f + 8 * il + 4, (uint32_t) (grid >> 32), d, sb, 4);
        }
        mq_put32(sw + r * MM_LD + u * 32, f);
    }
};
