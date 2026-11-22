// hy4.metal - Hy4-Flash (hy_v4_flash, .strata v1.0 SQ-64) kernels for Apple Silicon (MSL, compiled at run time by mtl_rt).
//
// Strata's own code (no llama.cpp). The math follows engine/src/hy4flash/run/*.hip.h of lane HY4-STRIX (the HIP reference), which already matches the python
// reference: residual stream x fp32 [T][4][3072]; GEMM activations fp16, GEMM outputs fp32, fp32 accumulation; S_K weights dequantised on the fly,
// w = (q - H) * D * (u/255) per 64-weight sub-group (H = 1.5/3.5/7.5/15.5/31.5 for S2..S6, 0 for the signed-int8 S8).
//
// Buffer layout of every kernel: arguments in order, [[buffer(i)]] = the i-th entry of the host Dispatch (buffers and inline bytes share the index space).
#define HID 3072
#define NHC 4
#define MLA_ROW 576
#define MLA_KV 512

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float siluf_(float x) { return x / (1.0f + exp(-x)); }

// sum over a threadgroup (nthr multiple of 32); sh[>= nthr/32]
static inline float tg_sum(float v, threadgroup float* sh, uint tid, uint nthr) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint i = 0; i < (nthr + 31) / 32; ++i) r += sh[i];
    return r;
}

static inline uint ld16(device const uchar* p) { return uint(p[0]) | (uint(p[1]) << 8); }
static inline float h2f_bits(uint b) { return float(as_type<half>((ushort) b)); }

constexpr int sk_sb_bytes(int f) { return f == 2 ? 70 : f == 3 ? 102 : f == 4 ? 134 : f == 5 ? 166 : f == 6 ? 198 : f == 8 ? 262 : 0; }
constexpr float sk_half(int f) { return f == 2 ? 1.5f : f == 3 ? 3.5f : f == 4 ? 7.5f : f == 5 ? 15.5f : f == 6 ? 31.5f : 0.0f; }

// integer of weight i (0..255) of a super-block payload pl (= block + 6)
template <int F>
static inline int sk_q(device const uchar* pl, int i) {
    if (F == 2) return (pl[i >> 2] >> ((i & 3) * 2)) & 3;
    if (F == 3) return ((pl[i >> 2] >> ((i & 3) * 2)) & 3) | (((pl[64 + (i >> 3)] >> (i & 7)) & 1) << 2);
    if (F == 4) return (pl[i >> 1] >> ((i & 1) * 4)) & 15;
    if (F == 5) return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 3)] >> (i & 7)) & 1) << 4);
    if (F == 6) return ((pl[i >> 1] >> ((i & 1) * 4)) & 15) | (((pl[128 + (i >> 2)] >> ((i & 3) * 2)) & 3) << 4);
    return int(as_type<char>(pl[i]));  // S8
}

// the 8 integers of weights i..i+7 (i a multiple of 8) of a super-block payload, each payload byte loaded once
template <int F>
static inline void sk_q8(device const uchar* pl, int i, thread int* q) {
    // payload offsets are even (block size and header are even), so 16-bit loads are aligned: half the load instructions of the byte version
    if (F == 2) {
        const uint v = uint(*((device const ushort*) (pl + (i >> 2))));
        const uint b0 = v & 255u, b1 = v >> 8;
        for (int k = 0; k < 4; ++k) { q[k] = int((b0 >> (2 * k)) & 3); q[4 + k] = int((b1 >> (2 * k)) & 3); }
    } else if (F == 3) {
        const uint v = uint(*((device const ushort*) (pl + (i >> 2))));
        const uint b0 = v & 255u, b1 = v >> 8, hb = pl[64 + (i >> 3)];
        for (int k = 0; k < 4; ++k) {
            q[k] = int(((b0 >> (2 * k)) & 3) | (((hb >> k) & 1) << 2));
            q[4 + k] = int(((b1 >> (2 * k)) & 3) | (((hb >> (4 + k)) & 1) << 2));
        }
    } else if (F == 8) {
        device const ushort* p = (device const ushort*) (pl + i);
        const uint v0 = uint(p[0]) | (uint(p[1]) << 16), v1 = uint(p[2]) | (uint(p[3]) << 16);
        for (int k = 0; k < 4; ++k) {
            q[k] = int(as_type<char>((uchar) ((v0 >> (8 * k)) & 255u)));
            q[4 + k] = int(as_type<char>((uchar) ((v1 >> (8 * k)) & 255u)));
        }
    } else {  // 4, 5, 6: nibbles + the high bits
        device const ushort* p = (device const ushort*) (pl + (i >> 1));
        const uint v = uint(p[0]) | (uint(p[1]) << 16);
        for (int k = 0; k < 4; ++k) {
            const uint b = (v >> (8 * k)) & 255u;
            q[2 * k] = int(b & 15);
            q[2 * k + 1] = int(b >> 4);
        }
        if (F == 5) {
            const uint hb = pl[128 + (i >> 3)];
            for (int k = 0; k < 8; ++k) q[k] |= int((hb >> k) & 1) << 4;
        } else if (F == 6) {
            const uint hb = uint(*((device const ushort*) (pl + 128 + (i >> 2))));
            for (int k = 0; k < 8; ++k) q[k] |= int((hb >> (2 * k)) & 3) << 4;
        }
    }
}

// the 8 weights j..j+7 (j multiple of 8, inside one 64-weight sub-group g) of super-block blk, dequantised
template <int F>
static inline void sk_w8(device const uchar* blk, int g, int j, thread float* w) {
    const float D = float(as_type<half>(*((device const ushort*) blk)));
    const float eff = D * (float(blk[2 + g]) / 255.0f);
    int q[8];
    sk_q8<F>(blk + 6, j, q);
    for (int t = 0; t < 8; ++t) w[t] = (float(q[t]) - sk_half(F)) * eff;
}

// Constant-cache LUTs for S2_K..S6_K centered weight decode (exact to the bit vs float(q) - sk_half(F); 10x fewer ALU ops).
constant half4 kHyS2Lut[256] = {
    half4(-1.5h,-1.5h,-1.5h,-1.5h), half4(-0.5h,-1.5h,-1.5h,-1.5h), half4(0.5h,-1.5h,-1.5h,-1.5h), half4(1.5h,-1.5h,-1.5h,-1.5h),
    half4(-1.5h,-0.5h,-1.5h,-1.5h), half4(-0.5h,-0.5h,-1.5h,-1.5h), half4(0.5h,-0.5h,-1.5h,-1.5h), half4(1.5h,-0.5h,-1.5h,-1.5h),
    half4(-1.5h,0.5h,-1.5h,-1.5h),  half4(-0.5h,0.5h,-1.5h,-1.5h),  half4(0.5h,0.5h,-1.5h,-1.5h),  half4(1.5h,0.5h,-1.5h,-1.5h),
    half4(-1.5h,1.5h,-1.5h,-1.5h),  half4(-0.5h,1.5h,-1.5h,-1.5h),  half4(0.5h,1.5h,-1.5h,-1.5h),  half4(1.5h,1.5h,-1.5h,-1.5h),
    half4(-1.5h,-1.5h,-0.5h,-1.5h), half4(-0.5h,-1.5h,-0.5h,-1.5h), half4(0.5h,-1.5h,-0.5h,-1.5h), half4(1.5h,-1.5h,-0.5h,-1.5h),
    half4(-1.5h,-0.5h,-0.5h,-1.5h), half4(-0.5h,-0.5h,-0.5h,-1.5h), half4(0.5h,-0.5h,-0.5h,-1.5h), half4(1.5h,-0.5h,-0.5h,-1.5h),
    half4(-1.5h,0.5h,-0.5h,-1.5h),  half4(-0.5h,0.5h,-0.5h,-1.5h),  half4(0.5h,0.5h,-0.5h,-1.5h),  half4(1.5h,0.5h,-0.5h,-1.5h),
    half4(-1.5h,1.5h,-0.5h,-1.5h),  half4(-0.5h,1.5h,-0.5h,-1.5h),  half4(0.5h,1.5h,-0.5h,-1.5h),  half4(1.5h,1.5h,-0.5h,-1.5h),
    half4(-1.5h,-1.5h,0.5h,-1.5h),  half4(-0.5h,-1.5h,0.5h,-1.5h),  half4(0.5h,-1.5h,0.5h,-1.5h),  half4(1.5h,-1.5h,0.5h,-1.5h),
    half4(-1.5h,-0.5h,0.5h,-1.5h),  half4(-0.5h,-0.5h,0.5h,-1.5h),  half4(0.5h,-0.5h,0.5h,-1.5h),  half4(1.5h,-0.5h,0.5h,-1.5h),
    half4(-1.5h,0.5h,0.5h,-1.5h),   half4(-0.5h,0.5h,0.5h,-1.5h),   half4(0.5h,0.5h,0.5h,-1.5h),   half4(1.5h,0.5h,0.5h,-1.5h),
    half4(-1.5h,1.5h,0.5h,-1.5h),   half4(-0.5h,1.5h,0.5h,-1.5h),   half4(0.5h,1.5h,0.5h,-1.5h),   half4(1.5h,1.5h,0.5h,-1.5h),
    half4(-1.5h,-1.5h,1.5h,-1.5h),  half4(-0.5h,-1.5h,1.5h,-1.5h),  half4(0.5h,-1.5h,1.5h,-1.5h),  half4(1.5h,-1.5h,1.5h,-1.5h),
    half4(-1.5h,-0.5h,1.5h,-1.5h),  half4(-0.5h,-0.5h,1.5h,-1.5h),  half4(0.5h,-0.5h,1.5h,-1.5h),  half4(1.5h,-0.5h,1.5h,-1.5h),
    half4(-1.5h,0.5h,1.5h,-1.5h),   half4(-0.5h,0.5h,1.5h,-1.5h),   half4(0.5h,0.5h,1.5h,-1.5h),   half4(1.5h,0.5h,1.5h,-1.5h),
    half4(-1.5h,1.5h,1.5h,-1.5h),   half4(-0.5h,1.5h,1.5h,-1.5h),   half4(0.5h,1.5h,1.5h,-1.5h),   half4(1.5h,1.5h,1.5h,-1.5h),
    half4(-1.5h,-1.5h,-1.5h,-0.5h), half4(-0.5h,-1.5h,-1.5h,-0.5h), half4(0.5h,-1.5h,-1.5h,-0.5h), half4(1.5h,-1.5h,-1.5h,-0.5h),
    half4(-1.5h,-0.5h,-1.5h,-0.5h), half4(-0.5h,-0.5h,-1.5h,-0.5h), half4(0.5h,-0.5h,-1.5h,-0.5h), half4(1.5h,-0.5h,-1.5h,-0.5h),
    half4(-1.5h,0.5h,-1.5h,-0.5h),  half4(-0.5h,0.5h,-1.5h,-0.5h),  half4(0.5h,0.5h,-1.5h,-0.5h),  half4(1.5h,0.5h,-1.5h,-0.5h),
    half4(-1.5h,1.5h,-1.5h,-0.5h),  half4(-0.5h,1.5h,-1.5h,-0.5h),  half4(0.5h,1.5h,-1.5h,-0.5h),  half4(1.5h,1.5h,-1.5h,-0.5h),
    half4(-1.5h,-1.5h,-0.5h,-0.5h), half4(-0.5h,-1.5h,-0.5h,-0.5h), half4(0.5h,-1.5h,-0.5h,-0.5h), half4(1.5h,-1.5h,-0.5h,-0.5h),
    half4(-1.5h,-0.5h,-0.5h,-0.5h), half4(-0.5h,-0.5h,-0.5h,-0.5h), half4(0.5h,-0.5h,-0.5h,-0.5h), half4(1.5h,-0.5h,-0.5h,-0.5h),
    half4(-1.5h,0.5h,-0.5h,-0.5h),  half4(-0.5h,0.5h,-0.5h,-0.5h),  half4(0.5h,0.5h,-0.5h,-0.5h),  half4(1.5h,0.5h,-0.5h,-0.5h),
    half4(-1.5h,1.5h,-0.5h,-0.5h),  half4(-0.5h,1.5h,-0.5h,-0.5h),  half4(0.5h,1.5h,-0.5h,-0.5h),  half4(1.5h,1.5h,-0.5h,-0.5h),
    half4(-1.5h,-1.5h,0.5h,-0.5h),  half4(-0.5h,-1.5h,0.5h,-0.5h),  half4(0.5h,-1.5h,0.5h,-0.5h),  half4(1.5h,-1.5h,0.5h,-0.5h),
    half4(-1.5h,-0.5h,0.5h,-0.5h),  half4(-0.5h,-0.5h,0.5h,-0.5h),  half4(0.5h,-0.5h,0.5h,-0.5h),  half4(1.5h,-0.5h,0.5h,-0.5h),
    half4(-1.5h,0.5h,0.5h,-0.5h),   half4(-0.5h,0.5h,0.5h,-0.5h),   half4(0.5h,0.5h,0.5h,-0.5h),   half4(1.5h,0.5h,0.5h,-0.5h),
    half4(-1.5h,1.5h,0.5h,-0.5h),   half4(-0.5h,1.5h,0.5h,-0.5h),   half4(0.5h,1.5h,0.5h,-0.5h),   half4(1.5h,1.5h,0.5h,-0.5h),
    half4(-1.5h,-1.5h,1.5h,-0.5h),  half4(-0.5h,-1.5h,1.5h,-0.5h),  half4(0.5h,-1.5h,1.5h,-0.5h),  half4(1.5h,-1.5h,1.5h,-0.5h),
    half4(-1.5h,-0.5h,1.5h,-0.5h),  half4(-0.5h,-0.5h,1.5h,-0.5h),  half4(0.5h,-0.5h,1.5h,-0.5h),  half4(1.5h,-0.5h,1.5h,-0.5h),
    half4(-1.5h,0.5h,1.5h,-0.5h),   half4(-0.5h,0.5h,1.5h,-0.5h),   half4(0.5h,0.5h,1.5h,-0.5h),   half4(1.5h,0.5h,1.5h,-0.5h),
    half4(-1.5h,1.5h,1.5h,-0.5h),   half4(-0.5h,1.5h,1.5h,-0.5h),   half4(0.5h,1.5h,1.5h,-0.5h),   half4(1.5h,1.5h,1.5h,-0.5h),
    half4(-1.5h,-1.5h,-1.5h,0.5h),  half4(-0.5h,-1.5h,-1.5h,0.5h),  half4(0.5h,-1.5h,-1.5h,0.5h),  half4(1.5h,-1.5h,-1.5h,0.5h),
    half4(-1.5h,-0.5h,-1.5h,0.5h),  half4(-0.5h,-0.5h,-1.5h,0.5h),  half4(0.5h,-0.5h,-1.5h,0.5h),  half4(1.5h,-0.5h,-1.5h,0.5h),
    half4(-1.5h,0.5h,-1.5h,0.5h),   half4(-0.5h,0.5h,-1.5h,0.5h),   half4(0.5h,0.5h,-1.5h,0.5h),   half4(1.5h,0.5h,-1.5h,0.5h),
    half4(-1.5h,1.5h,-1.5h,0.5h),   half4(-0.5h,1.5h,-1.5h,0.5h),   half4(0.5h,1.5h,-1.5h,0.5h),   half4(1.5h,1.5h,-1.5h,0.5h),
    half4(-1.5h,-1.5h,-0.5h,0.5h),  half4(-0.5h,-1.5h,-0.5h,0.5h),  half4(0.5h,-1.5h,-0.5h,0.5h),  half4(1.5h,-1.5h,-0.5h,0.5h),
    half4(-1.5h,-0.5h,-0.5h,0.5h),  half4(-0.5h,-0.5h,-0.5h,0.5h),  half4(0.5h,-0.5h,-0.5h,0.5h),  half4(1.5h,-0.5h,-0.5h,0.5h),
    half4(-1.5h,0.5h,-0.5h,0.5h),   half4(-0.5h,0.5h,-0.5h,0.5h),   half4(0.5h,0.5h,-0.5h,0.5h),   half4(1.5h,0.5h,-0.5h,0.5h),
    half4(-1.5h,1.5h,-0.5h,0.5h),   half4(-0.5h,1.5h,-0.5h,0.5h),   half4(0.5h,1.5h,-0.5h,0.5h),   half4(1.5h,1.5h,-0.5h,0.5h),
    half4(-1.5h,-1.5h,0.5h,0.5h),   half4(-0.5h,-1.5h,0.5h,0.5h),   half4(0.5h,-1.5h,0.5h,0.5h),   half4(1.5h,-1.5h,0.5h,0.5h),
    half4(-1.5h,-0.5h,0.5h,0.5h),   half4(-0.5h,-0.5h,0.5h,0.5h),   half4(0.5h,-0.5h,0.5h,0.5h),   half4(1.5h,-0.5h,0.5h,0.5h),
    half4(-1.5h,0.5h,0.5h,0.5h),    half4(-0.5h,0.5h,0.5h,0.5h),    half4(0.5h,0.5h,0.5h,0.5h),    half4(1.5h,0.5h,0.5h,0.5h),
    half4(-1.5h,1.5h,0.5h,0.5h),    half4(-0.5h,1.5h,0.5h,0.5h),    half4(0.5h,1.5h,0.5h,0.5h),    half4(1.5h,1.5h,0.5h,0.5h),
    half4(-1.5h,-1.5h,1.5h,0.5h),   half4(-0.5h,-1.5h,1.5h,0.5h),   half4(0.5h,-1.5h,1.5h,0.5h),   half4(1.5h,-1.5h,1.5h,0.5h),
    half4(-1.5h,-0.5h,1.5h,0.5h),   half4(-0.5h,-0.5h,1.5h,0.5h),   half4(0.5h,-0.5h,1.5h,0.5h),   half4(1.5h,-0.5h,1.5h,0.5h),
    half4(-1.5h,0.5h,1.5h,0.5h),    half4(-0.5h,0.5h,1.5h,0.5h),    half4(0.5h,0.5h,1.5h,0.5h),    half4(1.5h,0.5h,1.5h,0.5h),
    half4(-1.5h,1.5h,1.5h,0.5h),    half4(-0.5h,1.5h,1.5h,0.5h),    half4(0.5h,1.5h,1.5h,0.5h),    half4(1.5h,1.5h,1.5h,0.5h),
    half4(-1.5h,-1.5h,-1.5h,1.5h),  half4(-0.5h,-1.5h,-1.5h,1.5h),  half4(0.5h,-1.5h,-1.5h,1.5h),  half4(1.5h,-1.5h,-1.5h,1.5h),
    half4(-1.5h,-0.5h,-1.5h,1.5h),  half4(-0.5h,-0.5h,-1.5h,1.5h),  half4(0.5h,-0.5h,-1.5h,1.5h),  half4(1.5h,-0.5h,-1.5h,1.5h),
    half4(-1.5h,0.5h,-1.5h,1.5h),   half4(-0.5h,0.5h,-1.5h,1.5h),   half4(0.5h,0.5h,-1.5h,1.5h),   half4(1.5h,0.5h,-1.5h,1.5h),
    half4(-1.5h,1.5h,-1.5h,1.5h),   half4(-0.5h,1.5h,-1.5h,1.5h),   half4(0.5h,1.5h,-1.5h,1.5h),   half4(1.5h,1.5h,-1.5h,1.5h),
    half4(-1.5h,-1.5h,-0.5h,1.5h),  half4(-0.5h,-1.5h,-0.5h,1.5h),  half4(0.5h,-1.5h,-0.5h,1.5h),  half4(1.5h,-1.5h,-0.5h,1.5h),
    half4(-1.5h,-0.5h,-0.5h,1.5h),  half4(-0.5h,-0.5h,-0.5h,1.5h),  half4(0.5h,-0.5h,-0.5h,1.5h),  half4(1.5h,-0.5h,-0.5h,1.5h),
    half4(-1.5h,0.5h,-0.5h,1.5h),   half4(-0.5h,0.5h,-0.5h,1.5h),   half4(0.5h,0.5h,-0.5h,1.5h),   half4(1.5h,0.5h,-0.5h,1.5h),
    half4(-1.5h,1.5h,-0.5h,1.5h),   half4(-0.5h,1.5h,-0.5h,1.5h),   half4(0.5h,1.5h,-0.5h,1.5h),   half4(1.5h,1.5h,-0.5h,1.5h),
    half4(-1.5h,-1.5h,0.5h,1.5h),   half4(-0.5h,-1.5h,0.5h,1.5h),   half4(0.5h,-1.5h,0.5h,1.5h),   half4(1.5h,-1.5h,0.5h,1.5h),
    half4(-1.5h,-0.5h,0.5h,1.5h),   half4(-0.5h,-0.5h,0.5h,1.5h),   half4(0.5h,-0.5h,0.5h,1.5h),   half4(1.5h,-0.5h,0.5h,1.5h),
    half4(-1.5h,0.5h,0.5h,1.5h),    half4(-0.5h,0.5h,0.5h,1.5h),    half4(0.5h,0.5h,0.5h,1.5h),    half4(1.5h,0.5h,0.5h,1.5h),
    half4(-1.5h,1.5h,0.5h,1.5h),    half4(-0.5h,1.5h,0.5h,1.5h),    half4(0.5h,1.5h,0.5h,1.5h),    half4(1.5h,1.5h,0.5h,1.5h),
    half4(-1.5h,-1.5h,1.5h,1.5h),   half4(-0.5h,-1.5h,1.5h,1.5h),   half4(0.5h,-1.5h,1.5h,1.5h),   half4(1.5h,-1.5h,1.5h,1.5h),
    half4(-1.5h,-0.5h,1.5h,1.5h),   half4(-0.5h,-0.5h,1.5h,1.5h),   half4(0.5h,-0.5h,1.5h,1.5h),   half4(1.5h,-0.5h,1.5h,1.5h),
    half4(-1.5h,0.5h,1.5h,1.5h),    half4(-0.5h,0.5h,1.5h,1.5h),    half4(0.5h,0.5h,1.5h,1.5h),    half4(1.5h,0.5h,1.5h,1.5h),
    half4(-1.5h,1.5h,1.5h,1.5h),    half4(-0.5h,1.5h,1.5h,1.5h),    half4(0.5h,1.5h,1.5h,1.5h),    half4(1.5h,1.5h,1.5h,1.5h),
};
constant half4 kHyS3Hi[16] = {
    half4(-2.0h,-2.0h,-2.0h,-2.0h), half4(2.0h,-2.0h,-2.0h,-2.0h), half4(-2.0h,2.0h,-2.0h,-2.0h), half4(2.0h,2.0h,-2.0h,-2.0h),
    half4(-2.0h,-2.0h,2.0h,-2.0h),  half4(2.0h,-2.0h,2.0h,-2.0h),  half4(-2.0h,2.0h,2.0h,-2.0h),  half4(2.0h,2.0h,2.0h,-2.0h),
    half4(-2.0h,-2.0h,-2.0h,2.0h),  half4(2.0h,-2.0h,-2.0h,2.0h),  half4(-2.0h,2.0h,-2.0h,2.0h),  half4(2.0h,2.0h,-2.0h,2.0h),
    half4(-2.0h,-2.0h,2.0h,2.0h),   half4(2.0h,-2.0h,2.0h,2.0h),   half4(-2.0h,2.0h,2.0h,2.0h),   half4(2.0h,2.0h,2.0h,2.0h),
};
constant half2 kHyS4Lut[256] = {
    half2(-7.5h,-7.5h), half2(-6.5h,-7.5h), half2(-5.5h,-7.5h), half2(-4.5h,-7.5h), half2(-3.5h,-7.5h), half2(-2.5h,-7.5h), half2(-1.5h,-7.5h), half2(-0.5h,-7.5h),
    half2(0.5h,-7.5h),  half2(1.5h,-7.5h),  half2(2.5h,-7.5h),  half2(3.5h,-7.5h),  half2(4.5h,-7.5h),  half2(5.5h,-7.5h),  half2(6.5h,-7.5h),  half2(7.5h,-7.5h),
    half2(-7.5h,-6.5h), half2(-6.5h,-6.5h), half2(-5.5h,-6.5h), half2(-4.5h,-6.5h), half2(-3.5h,-6.5h), half2(-2.5h,-6.5h), half2(-1.5h,-6.5h), half2(-0.5h,-6.5h),
    half2(0.5h,-6.5h),  half2(1.5h,-6.5h),  half2(2.5h,-6.5h),  half2(3.5h,-6.5h),  half2(4.5h,-6.5h),  half2(5.5h,-6.5h),  half2(6.5h,-6.5h),  half2(7.5h,-6.5h),
    half2(-7.5h,-5.5h), half2(-6.5h,-5.5h), half2(-5.5h,-5.5h), half2(-4.5h,-5.5h), half2(-3.5h,-5.5h), half2(-2.5h,-5.5h), half2(-1.5h,-5.5h), half2(-0.5h,-5.5h),
    half2(0.5h,-5.5h),  half2(1.5h,-5.5h),  half2(2.5h,-5.5h),  half2(3.5h,-5.5h),  half2(4.5h,-5.5h),  half2(5.5h,-5.5h),  half2(6.5h,-5.5h),  half2(7.5h,-5.5h),
    half2(-7.5h,-4.5h), half2(-6.5h,-4.5h), half2(-5.5h,-4.5h), half2(-4.5h,-4.5h), half2(-3.5h,-4.5h), half2(-2.5h,-4.5h), half2(-1.5h,-4.5h), half2(-0.5h,-4.5h),
    half2(0.5h,-4.5h),  half2(1.5h,-4.5h),  half2(2.5h,-4.5h),  half2(3.5h,-4.5h),  half2(4.5h,-4.5h),  half2(5.5h,-4.5h),  half2(6.5h,-4.5h),  half2(7.5h,-4.5h),
    half2(-7.5h,-3.5h), half2(-6.5h,-3.5h), half2(-5.5h,-3.5h), half2(-4.5h,-3.5h), half2(-3.5h,-3.5h), half2(-2.5h,-3.5h), half2(-1.5h,-3.5h), half2(-0.5h,-3.5h),
    half2(0.5h,-3.5h),  half2(1.5h,-3.5h),  half2(2.5h,-3.5h),  half2(3.5h,-3.5h),  half2(4.5h,-3.5h),  half2(5.5h,-3.5h),  half2(6.5h,-3.5h),  half2(7.5h,-3.5h),
    half2(-7.5h,-2.5h), half2(-6.5h,-2.5h), half2(-5.5h,-2.5h), half2(-4.5h,-2.5h), half2(-3.5h,-2.5h), half2(-2.5h,-2.5h), half2(-1.5h,-2.5h), half2(-0.5h,-2.5h),
    half2(0.5h,-2.5h),  half2(1.5h,-2.5h),  half2(2.5h,-2.5h),  half2(3.5h,-2.5h),  half2(4.5h,-2.5h),  half2(5.5h,-2.5h),  half2(6.5h,-2.5h),  half2(7.5h,-2.5h),
    half2(-7.5h,-1.5h), half2(-6.5h,-1.5h), half2(-5.5h,-1.5h), half2(-4.5h,-1.5h), half2(-3.5h,-1.5h), half2(-2.5h,-1.5h), half2(-1.5h,-1.5h), half2(-0.5h,-1.5h),
    half2(0.5h,-1.5h),  half2(1.5h,-1.5h),  half2(2.5h,-1.5h),  half2(3.5h,-1.5h),  half2(4.5h,-1.5h),  half2(5.5h,-1.5h),  half2(6.5h,-1.5h),  half2(7.5h,-1.5h),
    half2(-7.5h,-0.5h), half2(-6.5h,-0.5h), half2(-5.5h,-0.5h), half2(-4.5h,-0.5h), half2(-3.5h,-0.5h), half2(-2.5h,-0.5h), half2(-1.5h,-0.5h), half2(-0.5h,-0.5h),
    half2(0.5h,-0.5h),  half2(1.5h,-0.5h),  half2(2.5h,-0.5h),  half2(3.5h,-0.5h),  half2(4.5h,-0.5h),  half2(5.5h,-0.5h),  half2(6.5h,-0.5h),  half2(7.5h,-0.5h),
    half2(-7.5h,0.5h),  half2(-6.5h,0.5h),  half2(-5.5h,0.5h),  half2(-4.5h,0.5h),  half2(-3.5h,0.5h),  half2(-2.5h,0.5h),  half2(-1.5h,0.5h),  half2(-0.5h,0.5h),
    half2(0.5h,0.5h),   half2(1.5h,0.5h),   half2(2.5h,0.5h),   half2(3.5h,0.5h),   half2(4.5h,0.5h),   half2(5.5h,0.5h),   half2(6.5h,0.5h),   half2(7.5h,0.5h),
    half2(-7.5h,1.5h),  half2(-6.5h,1.5h),  half2(-5.5h,1.5h),  half2(-4.5h,1.5h),  half2(-3.5h,1.5h),  half2(-2.5h,1.5h),  half2(-1.5h,1.5h),  half2(-0.5h,1.5h),
    half2(0.5h,1.5h),   half2(1.5h,1.5h),   half2(2.5h,1.5h),   half2(3.5h,1.5h),   half2(4.5h,1.5h),   half2(5.5h,1.5h),   half2(6.5h,1.5h),   half2(7.5h,1.5h),
    half2(-7.5h,2.5h),  half2(-6.5h,2.5h),  half2(-5.5h,2.5h),  half2(-4.5h,2.5h),  half2(-3.5h,2.5h),  half2(-2.5h,2.5h),  half2(-1.5h,2.5h),  half2(-0.5h,2.5h),
    half2(0.5h,2.5h),   half2(1.5h,2.5h),   half2(2.5h,2.5h),   half2(3.5h,2.5h),   half2(4.5h,2.5h),   half2(5.5h,2.5h),   half2(6.5h,2.5h),   half2(7.5h,2.5h),
    half2(-7.5h,3.5h),  half2(-6.5h,3.5h),  half2(-5.5h,3.5h),  half2(-4.5h,3.5h),  half2(-3.5h,3.5h),  half2(-2.5h,3.5h),  half2(-1.5h,3.5h),  half2(-0.5h,3.5h),
    half2(0.5h,3.5h),   half2(1.5h,3.5h),   half2(2.5h,3.5h),   half2(3.5h,3.5h),   half2(4.5h,3.5h),   half2(5.5h,3.5h),   half2(6.5h,3.5h),   half2(7.5h,3.5h),
    half2(-7.5h,4.5h),  half2(-6.5h,4.5h),  half2(-5.5h,4.5h),  half2(-4.5h,4.5h),  half2(-3.5h,4.5h),  half2(-2.5h,4.5h),  half2(-1.5h,4.5h),  half2(-0.5h,4.5h),
    half2(0.5h,4.5h),   half2(1.5h,4.5h),   half2(2.5h,4.5h),   half2(3.5h,4.5h),   half2(4.5h,4.5h),   half2(5.5h,4.5h),   half2(6.5h,4.5h),   half2(7.5h,4.5h),
    half2(-7.5h,5.5h),  half2(-6.5h,5.5h),  half2(-5.5h,5.5h),  half2(-4.5h,5.5h),  half2(-3.5h,5.5h),  half2(-2.5h,5.5h),  half2(-1.5h,5.5h),  half2(-0.5h,5.5h),
    half2(0.5h,5.5h),   half2(1.5h,5.5h),   half2(2.5h,5.5h),   half2(3.5h,5.5h),   half2(4.5h,5.5h),   half2(5.5h,5.5h),   half2(6.5h,5.5h),   half2(7.5h,5.5h),
    half2(-7.5h,6.5h),  half2(-6.5h,6.5h),  half2(-5.5h,6.5h),  half2(-4.5h,6.5h),  half2(-3.5h,6.5h),  half2(-2.5h,6.5h),  half2(-1.5h,6.5h),  half2(-0.5h,6.5h),
    half2(0.5h,6.5h),   half2(1.5h,6.5h),   half2(2.5h,6.5h),   half2(3.5h,6.5h),   half2(4.5h,6.5h),   half2(5.5h,6.5h),   half2(6.5h,6.5h),   half2(7.5h,6.5h),
    half2(-7.5h,7.5h),  half2(-6.5h,7.5h),  half2(-5.5h,7.5h),  half2(-4.5h,7.5h),  half2(-3.5h,7.5h),  half2(-2.5h,7.5h),  half2(-1.5h,7.5h),  half2(-0.5h,7.5h),
    half2(0.5h,7.5h),   half2(1.5h,7.5h),   half2(2.5h,7.5h),   half2(3.5h,7.5h),   half2(4.5h,7.5h),   half2(5.5h,7.5h),   half2(6.5h,7.5h),   half2(7.5h,7.5h),
};
constant half4 kHyS5Hi[16] = {
    half4(-8.0h,-8.0h,-8.0h,-8.0h), half4(8.0h,-8.0h,-8.0h,-8.0h), half4(-8.0h,8.0h,-8.0h,-8.0h), half4(8.0h,8.0h,-8.0h,-8.0h),
    half4(-8.0h,-8.0h,8.0h,-8.0h),  half4(8.0h,-8.0h,8.0h,-8.0h),  half4(-8.0h,8.0h,8.0h,-8.0h),  half4(8.0h,8.0h,8.0h,-8.0h),
    half4(-8.0h,-8.0h,-8.0h,8.0h),  half4(8.0h,-8.0h,-8.0h,8.0h),  half4(-8.0h,8.0h,-8.0h,8.0h),  half4(8.0h,8.0h,-8.0h,8.0h),
    half4(-8.0h,-8.0h,8.0h,8.0h),   half4(8.0h,-8.0h,8.0h,8.0h),   half4(-8.0h,8.0h,8.0h,8.0h),   half4(8.0h,8.0h,8.0h,8.0h),
};

// Fast aligned-ushort centered weight decode (exact to the bit vs float(q) - sk_half(F)):
//   - S2_K / S3_K: constant-cache LUT (4 weights/lookup -> -16% on MoE experts)
//   - S4_K / S5_K / S6_K: pure-ALU aligned-ushort sk_q8 (avoids constant-cache lookup pressure on 4-lookup formats)
//   - S8_K: aligned-ushort char2 unpack (-7% on S8 linear projections)
template <int F>
static inline void sk_w8_f4(device const uchar* pl, int j, thread float4& w0, thread float4& w1) {
    if (F == 2) {
        const uint v = uint(*((device const ushort*) (pl + (j >> 2))));
        w0 = float4(kHyS2Lut[v & 255u]);
        w1 = float4(kHyS2Lut[v >> 8]);
    } else if (F == 3) {
        const uint v = uint(*((device const ushort*) (pl + (j >> 2))));
        const uint h = uint(pl[64 + (j >> 3)]);
        w0 = float4(kHyS2Lut[v & 255u] + kHyS3Hi[h & 15u]);
        w1 = float4(kHyS2Lut[v >> 8] + kHyS3Hi[h >> 4]);
    } else if (F == 8) {
        device const ushort* p = (device const ushort*) (pl + j);
        const char2 c0 = as_type<char2>(p[0]), c1 = as_type<char2>(p[1]);
        const char2 c2 = as_type<char2>(p[2]), c3 = as_type<char2>(p[3]);
        w0 = float4(float(c0.x), float(c0.y), float(c1.x), float(c1.y));
        w1 = float4(float(c2.x), float(c2.y), float(c3.x), float(c3.y));
    } else {
        constexpr float H = sk_half(F);
        int q[8];
        sk_q8<F>(pl, j, q);
        w0 = float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) - H;
        w1 = float4(float(q[4]), float(q[5]), float(q[6]), float(q[7])) - H;
    }
}

// ================================================================ elementwise / norms / iHC =====================================================================

// embedding: S8_K row -> 4 identical hc streams. W: S8_K [V][3072]
kernel void embed_hc(device const uchar* W [[buffer(0)]], device const int* ids [[buffer(1)]], device float* x [[buffer(2)]], constant int& T [[buffer(3)]], uint t [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    if ((int) t >= T) return;
    device const uchar* row = W + (size_t) ids[t] * 12 * 262;
    for (uint d = tid; d < HID; d += nthr) {
        int sb = d >> 8, i = d & 255, g = i >> 6;
        device const uchar* blk = row + sb * 262;
        float D = h2f_bits(ld16(blk));
        float eff = D * (float(blk[2 + g]) / 255.0f);
        float v = float(as_type<char>(blk[6 + i])) * eff;
        for (int h = 0; h < NHC; ++h) x[((size_t) t * NHC + h) * HID + d] = v;
    }
}

// rmsnorm over D (one threadgroup per row); w optional (has_w), outputs fp32 and/or fp16
kernel void rms_rows(device const float* in [[buffer(0)]], device const float* w [[buffer(1)]], device float* outf [[buffer(2)]], device half* outh [[buffer(3)]], constant int& D [[buffer(4)]],
                     constant float& eps [[buffer(5)]], constant int& flags [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device const float* x = in + (size_t) row * D;
    float ss = 0;
    for (uint i = tid; i < (uint) D; i += nthr) ss += x[i] * x[i];
    ss = tg_sum(ss, sh, tid, nthr);
    float inv = rsqrt(ss / float(D) + eps);
    for (uint i = tid; i < (uint) D; i += nthr) {
        float v = x[i] * inv * ((flags & 1) ? w[i] : 1.0f);
        if (flags & 2) outf[(size_t) row * D + i] = v;
        if (flags & 4) outh[(size_t) row * D + i] = half(v);
    }
}

// iHC mixes: per token, rms of the 12288 flat state, nout dot products with fn [nout][12288] -> pre[4], post[4] (nout 8) or pre only (nout 4, head)
kernel void hc_mix(device const float* x [[buffer(0)]], device const float* fn [[buffer(1)]], device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                   device float* pre [[buffer(4)]], device float* post [[buffer(5)]], constant int& nout [[buffer(6)]], constant float& eps [[buffer(7)]], constant float& hc_eps [[buffer(8)]],
                   constant float& hc_mag [[buffer(9)]], uint tok [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sh[9][8];
    device const float* xt = x + (size_t) tok * NHC * HID;
    const int K = NHC * HID;
    float ss = 0;
    float acc[8];
    for (int j = 0; j < 8; ++j) acc[j] = 0;
    for (uint k = tid; k < (uint) K; k += nthr) {
        float v = xt[k];
        ss += v * v;
        for (int j = 0; j < 8; ++j)
            if (j < nout) acc[j] += v * fn[(size_t) j * K + k];
    }
    uint sg = tid >> 5, lane = tid & 31;
    ss = simd_sum(ss);
    if (lane == 0) sh[8][sg] = ss;
    for (int j = 0; j < 8; ++j) {
        float a = simd_sum(acc[j]);
        if (lane == 0) sh[j][sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0;
        for (uint i = 0; i < nthr / 32; ++i) tot += sh[8][i];
        float inv = rsqrt(tot / float(K) + eps);
        float m[8];
        for (int j = 0; j < 8; ++j) {
            float a = 0;
            for (uint i = 0; i < nthr / 32; ++i) a += sh[j][i];
            m[j] = a * inv;
        }
        if (nout == 8) {
            for (int h = 0; h < NHC; ++h) {
                pre[(size_t) tok * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
                post[(size_t) tok * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
            }
        } else {
            for (int h = 0; h < NHC; ++h) pre[(size_t) tok * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
        }
    }
}

// Prefill multi-token hc_mix: 4 tokens per threadgroup of 256 threads, reading fn[nout][12288] ONCE for all 4 tokens (4x less fn L2 traffic!)
// Uses exact lane-strided k += 256 summation order matching hc_mix bit-for-bit (zero router drift / KL = 0!).
kernel void hc_mix_blk(device const float* x [[buffer(0)]], device const float* fn [[buffer(1)]], device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                       device float* pre [[buffer(4)]], device float* post [[buffer(5)]], constant int& nout [[buffer(6)]], constant int& T [[buffer(7)]],
                       constant float& eps [[buffer(8)]], constant float& hc_eps [[buffer(9)]], constant float& hc_mag [[buffer(10)]],
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4][9][8];
    const int t0 = int(tg) * 4;
    const int nt = min(4, T - t0);
    if (nt <= 0) return;
    const int K = NHC * HID;
    float ss[4] = {0.f, 0.f, 0.f, 0.f};
    float acc[4][8];
    for (int t = 0; t < 4; ++t)
        for (int j = 0; j < 8; ++j) acc[t][j] = 0.f;
    for (uint k = tid; k < (uint) K; k += 256u) {
        float vx[4];
        for (int t = 0; t < 4; ++t) {
            if (t < nt) {
                const float v = x[(size_t) (t0 + t) * K + k];
                vx[t] = v;
                ss[t] += v * v;
            }
        }
        for (int j = 0; j < 8; ++j) {
            if (j < nout) {
                const float vf = fn[(size_t) j * K + k];
                for (int t = 0; t < 4; ++t)
                    if (t < nt) acc[t][j] += vx[t] * vf;
            }
        }
    }
    const uint sg = tid >> 5, lane = tid & 31u;
    for (int t = 0; t < 4; ++t) {
        if (t < nt) {
            const float s = simd_sum(ss[t]);
            if (lane == 0) sh[t][8][sg] = s;
            for (int j = 0; j < 8; ++j) {
                const float a = simd_sum(acc[t][j]);
                if (lane == 0) sh[t][j][sg] = a;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int) tid < nt) {
        const int t = int(tid);
        const int tok = t0 + t;
        float tot = 0.f;
        for (int i = 0; i < 8; ++i) tot += sh[t][8][i];
        const float inv = rsqrt(tot / float(K) + eps);
        float m[8];
        for (int j = 0; j < 8; ++j) {
            float a = 0.f;
            for (int i = 0; i < 8; ++i) a += sh[t][j][i];
            m[j] = a * inv;
        }
        if (nout == 8) {
            for (int h = 0; h < NHC; ++h) {
                pre[(size_t) tok * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
                post[(size_t) tok * NHC + h] = hc_mag * sigmoidf_(m[NHC + h] * scale[1] + base[NHC + h]) + hc_eps;
            }
        } else {
            for (int h = 0; h < NHC; ++h) pre[(size_t) tok * NHC + h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
        }
    }
}

// cur = sum_h x[h]*pre[h]; rmsnorm(cur, w) -> fp16 (GEMM input) and optionally fp32 (flags bit0 fp32 out). blockDim must be 256
kernel void hc_reduce_norm(device const float* x [[buffer(0)]], device const float* pre [[buffer(1)]], device const float* w [[buffer(2)]], device half* outh [[buffer(3)]],
                           device float* outf [[buffer(4)]], constant float& eps [[buffer(5)]], constant int& flags [[buffer(6)]], uint tok [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    device const float* xt = x + (size_t) tok * NHC * HID;
    device const float* p = pre + (size_t) tok * NHC;
    float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    float ss = 0;
    float v[HID / 256];
    for (int i = 0; i < HID / 256; ++i) {
        int d = tid + i * 256;
        float c = xt[d] * p0 + xt[HID + d] * p1 + xt[2 * HID + d] * p2 + xt[3 * HID + d] * p3;
        v[i] = c;
        ss += c * c;
    }
    ss = tg_sum(ss, sh, tid, 256);
    float inv = rsqrt(ss / float(HID) + eps);
    for (int i = 0; i < HID / 256; ++i) {
        int d = tid + i * 256;
        float o = v[i] * inv * w[d];
        outh[(size_t) tok * HID + d] = half(o);
        if (flags & 1) outf[(size_t) tok * HID + d] = o;
    }
}

// x[h] += sub * post[h]
kernel void hc_post(device float* x [[buffer(0)]], device const float* sub [[buffer(1)]], device const float* post [[buffer(2)]], uint tok [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    device const float* s = sub + (size_t) tok * HID;
    device const float* p = post + (size_t) tok * NHC;
    device float* xt = x + (size_t) tok * NHC * HID;
    for (uint d = tid; d < HID; d += nthr) {
        float sv = s[d];
        for (int h = 0; h < NHC; ++h) xt[h * HID + d] += sv * p[h];
    }
}

// out = rmsnorm(a (+ b), w), fp32 (has_b: b used)
kernel void add_rms(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const float* w [[buffer(2)]], device float* out [[buffer(3)]], constant int& D [[buffer(4)]],
                    constant float& eps [[buffer(5)]], constant int& has_b [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    size_t o = (size_t) row * D;
    float ss = 0;
    for (uint i = tid; i < (uint) D; i += nthr) {
        float v = a[o + i] + (has_b ? b[o + i] : 0.0f);
        ss += v * v;
    }
    ss = tg_sum(ss, sh, tid, nthr);
    float inv = rsqrt(ss / float(D) + eps);
    for (uint i = tid; i < (uint) D; i += nthr) {
        float v = a[o + i] + (has_b ? b[o + i] : 0.0f);
        out[o + i] = v * inv * w[i];
    }
}

kernel void add_rms_post(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const float* w [[buffer(2)]], device float* x [[buffer(3)]],
                         device const float* post [[buffer(4)]], constant float& eps [[buffer(5)]], constant int& has_b [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    size_t o = (size_t) row * HID;
    float v[HID / 256];
    float ss = 0;
    for (int i = 0; i < HID / 256; ++i) {
        int d = tid + i * 256;
        float t = a[o + d] + (has_b ? b[o + d] : 0.0f);
        v[i] = t;
        ss += t * t;
    }
    ss = tg_sum(ss, sh, tid, 256);
    float inv = rsqrt(ss / float(HID) + eps);
    device const float* p = post + (size_t) row * NHC;
    float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    device float* xt = x + (size_t) row * NHC * HID;
    for (int i = 0; i < HID / 256; ++i) {
        int d = tid + i * 256;
        float sv = v[i] * inv * w[d];
        xt[d] += sv * p0;
        xt[HID + d] += sv * p1;
        xt[2 * HID + d] += sv * p2;
        xt[3 * HID + d] += sv * p3;
    }
}

kernel void f2h(device const float* in [[buffer(0)]], device half* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = half(in[i]);
}

// h = silu(g) * u, fp32 in -> fp16 out
kernel void swiglu_h(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device half* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = half(siluf_(g[i]) * u[i]);
}

// router: logits [T][512] -> top-16 by sigmoid+bias; weights = p/(sum p + 1e-20)*scale. One threadgroup (256 threads) per token
kernel void router_topk(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]], device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                        constant int& E [[buffer(4)]], constant int& K [[buffer(5)]], constant float& scale [[buffer(6)]], uint tok [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sp[512], ss[512];
    threadgroup float rv[8];
    threadgroup int ri[8];
    threadgroup int chosen[16];
    device const float* lg = logits + (size_t) tok * E;
    for (uint e = tid; e < (uint) E; e += nthr) {
        float p = sigmoidf_(lg[e]);
        sp[e] = p;
        ss[e] = p + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < K; ++k) {
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
        for (uint e = tid; e < (uint) E; e += nthr)
            if (ss[e] > bv || (ss[e] == bv && (int) e < bi)) { bv = ss[e]; bi = e; }
        for (uint o = 16; o > 0; o >>= 1) {
            float ov = simd_shuffle_xor(bv, o);
            int oi = simd_shuffle_xor(bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { rv[tid >> 5] = bv; ri[tid >> 5] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b2 = rv[0];
            int i2 = ri[0];
            for (uint q = 1; q < nthr / 32; ++q)
                if (rv[q] > b2 || (rv[q] == b2 && ri[q] < i2)) { b2 = rv[q]; i2 = ri[q]; }
            chosen[k] = i2;
            ss[i2] = -3.0e38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += sp[chosen[k]];
        for (int k = 0; k < K; ++k) {
            idx[(size_t) tok * K + k] = chosen[k];
            w[(size_t) tok * K + k] = sp[chosen[k]] / (sum + 1e-20f) * scale;
        }
    }
}

// argmax over logits [V] (one row) -> token id; 1 threadgroup of 1024
kernel void argmax_row(device const float* logits [[buffer(0)]], constant int& V [[buffer(1)]], device int* out [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                       uint nthr [[threads_per_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    float bv = -3.0e38f;
    int bi = 0;
    for (uint i = tid; i < (uint) V; i += nthr)
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    for (uint o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_xor(bv, o);
        int oi = simd_shuffle_xor(bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b2 = sv[0];
        int i2 = si[0];
        for (uint q = 1; q < nthr / 32; ++q)
            if (sv[q] > b2 || (sv[q] == b2 && si[q] < i2)) { b2 = sv[q]; i2 = si[q]; }
        *out = i2;
    }
}

// ---- Engram: raw rows [T][16][100 B] (f32 scale + 96 int8) -> normed rows r[T][1536] fp16; 512 threads (16 rows x 32 lanes)
kernel void engram_rows(device const uchar* raw [[buffer(0)]], device const float* lnw [[buffer(1)]], device const uchar* valid [[buffer(2)]], device half* r [[buffer(3)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint c = tid >> 5, lane = tid & 31;
    device const uchar* row = raw + ((size_t) t * 16 + c) * 100;
    float scale = *((device const float*) row);
    float v[3];
    float ss = 0;
    for (int j = 0; j < 3; ++j) {
        v[j] = float(as_type<char>(row[4 + lane + 32 * j])) * scale;
        ss += v[j] * v[j];
    }
    ss = simd_sum(ss);
    float inv = rsqrt(ss / 96.0f + 1.1920929e-7f);
    float vm = valid[(size_t) t * 16 + c] ? 1.0f : 0.0f;
    for (int j = 0; j < 3; ++j) r[(size_t) t * 1536 + c * 96 + lane + 32 * j] = half(v[j] * inv * lnw[c * 96 + lane + 32 * j] * vm);
}

// ---- Engram apply: kv [5][T][3072] fp32 (bias added here), x [T][4][3072]; threadgroup per (token, stream j), 256 threads
kernel void engram_apply(device float* x [[buffer(0)]], device const float* kv [[buffer(1)]], device const float* kvb [[buffer(2)]], device const float* egate [[buffer(3)]],
                         device const float* pre_gamma [[buffer(4)]], device const float* hcs [[buffer(5)]], device const uchar* valid [[buffer(6)]], constant int& T [[buffer(7)]],
                         uint2 tj [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const uint nthr = 256;
    uint t = tj.x, j = tj.y;
    const float EPS = 1.1920929e-7f;
    device float* xs = x + ((size_t) t * NHC + j) * HID;
    device const float* key = kv + ((size_t) j * T + t) * HID;
    device const float* val = kv + ((size_t) 4 * T + t) * HID;
    float sx = 0, sk = 0, sv = 0;
    for (uint d = tid; d < HID; d += nthr) {
        float a = xs[d], k = key[d] + kvb[j * HID + d], v = val[d] + kvb[4 * HID + d];
        sx += a * a; sk += k * k; sv += v * v;
    }
    sx = tg_sum(sx, sh, tid, nthr);
    sk = tg_sum(sk, sh, tid, nthr);
    sv = tg_sum(sv, sh, tid, nthr);
    float xi = rsqrt(sx / float(HID) + EPS), ki = rsqrt(sk / float(HID) + EPS), vi = rsqrt(sv / float(HID) + EPS);
    float dot = 0;
    for (uint d = tid; d < HID; d += nthr) {
        float a = xs[d], k = key[d] + kvb[j * HID + d];
        dot += (a * xi) * (k * ki) * egate[j * HID + d];
    }
    dot = tg_sum(dot, sh, tid, nthr) / sqrt(float(HID));
    float mag = sqrt(max(fabs(dot), 1e-6f));
    float gate = sigmoidf_(dot < 0 ? -mag : mag);
    int anyv = 0;
    for (int c = 0; c < 16; ++c) anyv |= valid[(size_t) t * 16 + c] ? 1 : 0;
    float anyf = anyv ? 1.0f : 0.0f;
    for (uint d = tid; d < HID; d += nthr) {
        float v = val[d] + kvb[4 * HID + d];
        xs[d] += (v * vi * pre_gamma[d]) * (gate + hcs[j * HID + d]) * anyf;
    }
}

// ================================================================ linear algebra ===============================================================================

// Y[m][n] = sum_k W[n][k] * X[m][k]   (X half [M][K], Y float [M][N]); GEMV style: one simdgroup per row, MT tokens per pass; 8 rows per threadgroup (256 threads)
template <int F, int MT>
kernel void linear_sk(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                      constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = tg * 8 + sg;
    if (row >= N) return;
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    device const uchar* rp = W + (size_t) row * nsb * SB;
    for (int m0 = 0; m0 < M; m0 += MT) {
        float acc[MT];
        for (int i = 0; i < MT; ++i) acc[i] = 0.f;
        for (int unit = lane; unit < nsb * 32; unit += 32) {
            const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
            device const uchar* blk = rp + (size_t) sbi * SB;
            float w[8];
            sk_w8<F>(blk, g, j, w);
            for (int mm = 0; mm < MT; ++mm) {
                if (m0 + mm < M) {
                    device const half4* xp = (device const half4*) (X + (size_t) (m0 + mm) * K + sbi * 256 + j);
                    const float4 xa = float4(xp[0]), xb = float4(xp[1]);
                    float a = 0.f;
                    a += w[0] * xa.x; a += w[1] * xa.y; a += w[2] * xa.z; a += w[3] * xa.w;
                    a += w[4] * xb.x; a += w[5] * xb.y; a += w[6] * xb.z; a += w[7] * xb.w;
                    acc[mm] += a;
                }
            }
        }
        for (int mm = 0; mm < MT; ++mm) {
            float v = simd_sum(acc[mm]);
            if (lane == 0 && m0 + mm < M) Y[(size_t) (m0 + mm) * N + row] = v;
        }
    }
}

// Q8ROW: [N f32 scales][N*K int8]
template <int MT>
kernel void linear_q8row(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                         constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = tg * 8 + sg;
    if (row >= N) return;
    const float scale = ((device const float*) W)[row];
    device const char* q = (device const char*) (W + (size_t) N * 4) + (size_t) row * K;
    for (int m0 = 0; m0 < M; m0 += MT) {
        float acc[MT];
        for (int i = 0; i < MT; ++i) acc[i] = 0.f;
        for (int k = lane; k < K; k += 32) {
            const float w = float(q[k]);
            for (int mm = 0; mm < MT; ++mm)
                if (m0 + mm < M) acc[mm] += w * float(X[(size_t) (m0 + mm) * K + k]);
        }
        for (int mm = 0; mm < MT; ++mm) {
            float v = simd_sum(acc[mm]);
            if (lane == 0 && m0 + mm < M) Y[(size_t) (m0 + mm) * N + row] = v * scale;
        }
    }
}

// F32 weights, F32 activations (router, hc)
kernel void linear_f32(device const float* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                       constant int& M [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int row = tg * 8 + sg;
    if (row >= N) return;
    device const float* wp = W + (size_t) row * K;
    for (int m0 = 0; m0 < M; m0 += 8) {
        float acc[8];
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
        for (int k = lane; k < K; k += 32) {
            const float w = wp[k];
            for (int mm = 0; mm < 8; ++mm)
                if (m0 + mm < M) acc[mm] += w * X[(size_t) (m0 + mm) * K + k];
        }
        for (int mm = 0; mm < 8; ++mm) {
            float v = simd_sum(acc[mm]);
            if (lane == 0 && m0 + mm < M) Y[(size_t) (m0 + mm) * N + row] = v;
        }
    }
}

// fp32 x fp32 GEMM on simdgroup_matrix (router): Y[m][n] = sum_k W[n][k] X[m][k]; tile 32 tokens x 64 rows per threadgroup of 4 simdgroups (each 32 x 16), operands from device memory.
// X is read up to the next multiple of 32 rows (the buffer is allocated for the whole chunk).
kernel void linear_f32_mm(device const float* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                          constant int& M [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sC[32 * 64];
    const int n0 = tgp.x * 64, m0 = tgp.y * 32;
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < K; k += 8) {
        simdgroup_float8x8 a[4], b[2];
        for (int i = 0; i < 4; ++i) simdgroup_load(a[i], X + (size_t) (m0 + i * 8) * K + k, K);
        for (int j = 0; j < 2; ++j) simdgroup_load(b[j], W + (size_t) (n0 + sg * 16 + j * 8) * K + k, K, ulong2(0, 0), true);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], sC + (i * 8) * 64 + sg * 16 + j * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < 32 * 64; idx += 128) {
        int m = idx >> 6, n = idx & 63;
        if (m0 + m < M && n0 + n < N) Y[(size_t) (m0 + m) * N + n0 + n] = sC[idx];
    }
}

// GEMM for prefill (M >= 16): simdgroup_matrix half x half -> float. Threadgroup = 4 simdgroups, tile 32 tokens x 64 rows, K step 64 (one scale sub-group).
// F = 2..8 (S_K) or 100 (Q8ROW). Weights are dequantised in fp32 and rounded to fp16 for the matrix unit.
template <int F>
kernel void linear_mm(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]], device float* Y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& K [[buffer(4)]],
                      constant int& M [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sA[32 * 64];
    threadgroup half sB[64 * 64];
    threadgroup float sC[32 * 64];
    const int n0 = tgp.x * 64, m0 = tgp.y * 32;
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    constexpr int SB = sk_sb_bytes(F);
    const int nsb = K >> 8;
    const int r = tid >> 1, kh = (tid & 1) * 32;  // weight tile row / half of the 64-wide k step this thread decodes
    const int arow = tid >> 2, acol = (tid & 3) * 16;  // activation tile
    const int wrow = n0 + r;
    float q8scale = 0.0f;
    if (F == 100 && wrow < N) q8scale = ((device const float*) W)[wrow];
    for (int ks = 0; ks < (K >> 6); ++ks) {
        // activation tile
        {
            const int m = m0 + arow;
            for (int e = 0; e < 16; ++e) sA[arow * 64 + acol + e] = m < M ? X[(size_t) m * K + ks * 64 + acol + e] : half(0.0f);
        }
        // weight tile
        if (wrow < N) {
            if (F == 100) {
                device const char* q = (device const char*) (W + (size_t) N * 4) + (size_t) wrow * K + ks * 64 + kh;
                for (int e = 0; e < 32; ++e) sB[r * 64 + kh + e] = half(float(q[e]) * q8scale);
            } else {
                const int sbi = ks >> 2, g = ks & 3;
                device const uchar* blk = W + ((size_t) wrow * nsb + sbi) * SB;
                const float D = h2f_bits(ld16(blk));
                const float eff = D * (float(blk[2 + g]) / 255.0f);
                device const uchar* pl = blk + 6;
                for (int e = 0; e < 32; ++e) sB[r * 64 + kh + e] = half((float(sk_q<F>(pl, g * 64 + kh + e)) - sk_half(F)) * eff);
            }
        } else {
            for (int e = 0; e < 32; ++e) sB[r * 64 + kh + e] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 a[4], b[2];
            for (int i = 0; i < 4; ++i) simdgroup_load(a[i], sA + (i * 8) * 64 + kk * 8, 64);
            for (int j = 0; j < 2; ++j) simdgroup_load(b[j], sB + (sg * 16 + j * 8) * 64 + kk * 8, 64, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], sC + (i * 8) * 64 + sg * 16 + j * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < 32 * 64; idx += 128) {
        int m = idx >> 6, n = idx & 63;
        if (m0 + m < M && n0 + n < N) Y[(size_t) (m0 + m) * N + n0 + n] = sC[idx];
    }
}

#define INST_LIN(F)                                                                                                                                                       \
    template [[host_name("linear_sk_" #F "_8")]] kernel void linear_sk<F, 8>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint); \
    template [[host_name("linear_sk_" #F "_1")]] kernel void linear_sk<F, 1>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint); \
    template [[host_name("linear_mm_" #F)]] kernel void linear_mm<F>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint2, uint, uint);
INST_LIN(2)
INST_LIN(3)
INST_LIN(4)
INST_LIN(5)
INST_LIN(6)
INST_LIN(8)
template [[host_name("linear_q8row_8")]] kernel void linear_q8row<8>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint);
template [[host_name("linear_q8row_1")]] kernel void linear_q8row<1>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint, uint);
template [[host_name("linear_mm_100")]] kernel void linear_mm<100>(device const uchar*, device const half*, device float*, constant int&, constant int&, constant int&, uint2, uint, uint);

// kv_b_proj [NH*448][512] (any S_K) -> wk [NH][192][512] and wv [NH][256][512] fp16 (absorbed MLA weights); one threadgroup per row of kv_b_proj
template <int F>
kernel void split_kvb(device const uchar* W [[buffer(0)]], device half* wk [[buffer(1)]], device half* wv [[buffer(2)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint nthr [[threads_per_threadgroup]]) {
    int h = r / 448, j = r % 448;
    constexpr int SB = sk_sb_bytes(F);
    device const uchar* rp = W + (size_t) r * 2 * SB;
    device half* dst = j < 192 ? wk + ((size_t) h * 192 + j) * 512 : wv + ((size_t) h * 256 + (j - 192)) * 512;
    for (uint k = tid; k < 512; k += nthr) {
        device const uchar* blk = rp + (size_t) (k >> 8) * SB;
        const float D = h2f_bits(ld16(blk));
        const int i = k & 255;
        const float eff = D * (float(blk[2 + (i >> 6)]) / 255.0f);
        dst[k] = half((float(sk_q<F>(blk + 6, i)) - sk_half(F)) * eff);
    }
}
template [[host_name("split_kvb_5")]] kernel void split_kvb<5>(device const uchar*, device half*, device half*, uint, uint, uint);
template [[host_name("split_kvb_4")]] kernel void split_kvb<4>(device const uchar*, device half*, device half*, uint, uint, uint);
template [[host_name("split_kvb_6")]] kernel void split_kvb<6>(device const uchar*, device half*, device half*, uint, uint, uint);
template [[host_name("split_kvb_8")]] kernel void split_kvb<8>(device const uchar*, device half*, device half*, uint, uint, uint);

// ================================================================ MoE ==========================================================================================
struct ExpertDev {
    ulong body_off;  // relative to the Section 4 base
    uint dn_off;     // relative to body_off
    uchar gu_code, dn_code;
    ushort pad;
};

// dot of one weight row (K = 1536, format F) with MT x vectors (half, row stride 1536 elements; xp[m] pointers; null = skipped)
template <int F, int MT>
static inline void row_dot1536(device const uchar* rp, device const half* xbase, thread const int* xoff, int lane, thread float* acc) {
    constexpr int SB = sk_sb_bytes(F);
    for (int unit = lane; unit < 6 * 32; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        device const uchar* blk = rp + (size_t) sbi * SB;
        float w[8];
        sk_w8<F>(blk, g, j, w);
        for (int mm = 0; mm < MT; ++mm) {
            if (xoff[mm] >= 0) {
                device const half4* x4 = (device const half4*) (xbase + (size_t) xoff[mm] * 1536 + sbi * 256 + j);
                const float4 xa = float4(x4[0]), xb = float4(x4[1]);
                float a = 0.f;
                a += w[0] * xa.x; a += w[1] * xa.y; a += w[2] * xa.z; a += w[3] * xa.w;
                a += w[4] * xb.x; a += w[5] * xb.y; a += w[6] * xb.z; a += w[7] * xb.w;
                acc[mm] += a;
            }
        }
    }
}
template <int MT>
static inline void row_dot_sw(int code, device const uchar* rp, device const half* xbase, thread const int* xoff, int lane, thread float* acc) {
    switch (code) {
        case 2: row_dot1536<2, MT>(rp, xbase, xoff, lane, acc); break;
        case 3: row_dot1536<3, MT>(rp, xbase, xoff, lane, acc); break;
        case 4: row_dot1536<4, MT>(rp, xbase, xoff, lane, acc); break;
        case 5: row_dot1536<5, MT>(rp, xbase, xoff, lane, acc); break;
        case 6: row_dot1536<6, MT>(rp, xbase, xoff, lane, acc); break;
        case 8: row_dot1536<8, MT>(rp, xbase, xoff, lane, acc); break;
        default: break;
    }
}
// two rows (gate, up) of the same format against ONE x vector: the x loads are shared (decode pair path)
template <int F>
static inline void row_dot1536_gu(device const uchar* rg, device const uchar* ru, device const half* x, int lane, thread float& ag, thread float& au) {
    constexpr int SB = sk_sb_bytes(F);
    for (int unit = lane; unit < 6 * 32; unit += 32) {
        const int sbi = unit >> 5, u = unit & 31, g = u >> 3, j = (g << 6) + ((u & 7) << 3);
        device const half4* x4 = (device const half4*) (x + sbi * 256 + j);
        const float4 xa = float4(x4[0]), xb = float4(x4[1]);
        float wg[8], wu[8];
        sk_w8<F>(rg + (size_t) sbi * SB, g, j, wg);
        sk_w8<F>(ru + (size_t) sbi * SB, g, j, wu);
        float a = 0.f, b = 0.f;
        a += wg[0] * xa.x; a += wg[1] * xa.y; a += wg[2] * xa.z; a += wg[3] * xa.w;
        a += wg[4] * xb.x; a += wg[5] * xb.y; a += wg[6] * xb.z; a += wg[7] * xb.w;
        b += wu[0] * xa.x; b += wu[1] * xa.y; b += wu[2] * xa.z; b += wu[3] * xa.w;
        b += wu[4] * xb.x; b += wu[5] * xb.y; b += wu[6] * xb.z; b += wu[7] * xb.w;
        ag += a;
        au += b;
    }
}
static inline void row_dot_gu_sw(int code, device const uchar* rg, device const uchar* ru, device const half* x, int lane, thread float& ag, thread float& au) {
    switch (code) {
        case 2: row_dot1536_gu<2>(rg, ru, x, lane, ag, au); break;
        case 3: row_dot1536_gu<3>(rg, ru, x, lane, ag, au); break;
        case 4: row_dot1536_gu<4>(rg, ru, x, lane, ag, au); break;
        case 5: row_dot1536_gu<5>(rg, ru, x, lane, ag, au); break;
        case 6: row_dot1536_gu<6>(rg, ru, x, lane, ag, au); break;
        case 8: row_dot1536_gu<8>(rg, ru, x, lane, ag, au); break;
        default: break;
    }
}
static inline int code_row_bytes(int code, int K) { return (K >> 8) * (code == 2 ? 70 : code == 3 ? 102 : code == 4 ? 134 : code == 5 ? 166 : code == 6 ? 198 : 262); }

// ---- pair path (decode, few tokens): grid (192, P), 256 threads; every (token, slot) pair reads its own expert
kernel void moe_gu_pair(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const half* z [[buffer(2)]], device const int* idx [[buffer(3)]],
                        device half* h2 [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = tgp.x * 8 + sg;
    const int p = tgp.y;
    const int e = idx[p];
    const ExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    device const half* sqx = (device const half*) body;
    const int rb = code_row_bytes(ed.gu_code, 1536);
    device const uchar* gu = body + 3072;
    float ag = 0.f, au = 0.f;
    row_dot_gu_sw(ed.gu_code, gu + (size_t) r * rb, gu + (size_t) (1536 + r) * rb, z + (size_t) (p / 16) * 1536, lane, ag, au);
    float g = simd_sum(ag), u = simd_sum(au);
    if (lane == 0) {
        const float sg_ = g / (1.0f + exp(-g));
        const float hh = min(sg_, 10.0f) * clamp(u, -10.0f, 10.0f);
        h2[(size_t) p * 1536 + r] = half(hh / float(sqx[r]));
    }
}
kernel void moe_dn_pair(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const float* topk_w [[buffer(2)]], device const int* idx [[buffer(3)]],
                        device const half* h2 [[buffer(4)]], device float* y [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = tgp.x * 8 + sg;
    const int p = tgp.y;
    const ExpertDev ed = ex[idx[p]];
    device const uchar* dn = sec4 + ed.body_off + ed.dn_off;
    int xo[1] = {p};
    float a[1] = {0.f};
    row_dot_sw<1>(ed.dn_code, dn + (size_t) r * code_row_bytes(ed.dn_code, 1536), h2, xo, lane, a);
    float v = simd_sum(a[0]);
    if (lane == 0) y[(size_t) p * 1536 + r] = v * topk_w[p];
}
kernel void moe_reduce(device const float* y [[buffer(0)]], device float* zout [[buffer(1)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint nthr [[threads_per_threadgroup]]) {
    for (uint n = tid; n < 1536; n += nthr) {
        float a = 0.f;
        for (int s = 0; s < 16; ++s) a += y[((size_t) t * 16 + s) * 1536 + n];
        zout[(size_t) t * 1536 + n] = a;
    }
}

// ---- sorted path (prefill): routing = counting sort of the P = T*16 pairs by expert (single threadgroup of 512)
kernel void moe_route(device const int* idx [[buffer(0)]], constant int& P [[buffer(1)]], device int* cnt [[buffer(2)]], device int* off [[buffer(3)]], device int* pairs [[buffer(4)]],
                      uint t [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int c[512];
    threadgroup atomic_int fill[512];
    threadgroup int sc[512];
    atomic_store_explicit(&c[t], 0, memory_order_relaxed);
    atomic_store_explicit(&fill[t], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < P; i += 512) atomic_fetch_add_explicit(&c[idx[i]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int cv = atomic_load_explicit(&c[t], memory_order_relaxed);
    sc[t] = cv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = 1; d < 512; d <<= 1) {
        int a = 0;
        if ((int) t >= d) a = sc[t - d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sc[t] += a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int incl = sc[t];
    cnt[t] = cv;
    off[t] = incl - cv;
    if (t == 511) off[512] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sc[t] = incl - cv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = t; i < P; i += 512) {
        const int e = idx[i];
        const int pos = atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
        pairs[sc[e] + pos] = i;
    }
}

// grid (192, 512 experts); each simdgroup = one (gate row r, up row 1536+r) pair, up to 8 (token, slot) pairs per weight pass
kernel void moe_gu_sorted(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const half* z [[buffer(2)]], device const int* cnt [[buffer(3)]],
                          device const int* off [[buffer(4)]], device const int* pairs [[buffer(5)]], device half* h2 [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const int e = tgp.y;
    const int n = cnt[e];
    if (n == 0) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = tgp.x * 8 + sg;
    const ExpertDev ed = ex[e];
    device const uchar* body = sec4 + ed.body_off;
    device const half* sqx = (device const half*) body;
    const int rb = code_row_bytes(ed.gu_code, 1536);
    device const uchar* gu = body + 3072;
    device const uchar* grow = gu + (size_t) r * rb;
    device const uchar* urow = gu + (size_t) (1536 + r) * rb;
    const int base = off[e];
    const float sq = float(sqx[r]);
    for (int c0 = 0; c0 < n; c0 += 8) {
        int xo[8];
        int pid[8];
        for (int i = 0; i < 8; ++i) {
            if (c0 + i < n) {
                pid[i] = pairs[base + c0 + i];
                xo[i] = pid[i] / 16;
            } else {
                pid[i] = -1;
                xo[i] = -1;
            }
        }
        float ag[8], au[8];
        for (int i = 0; i < 8; ++i) { ag[i] = 0.f; au[i] = 0.f; }
        row_dot_sw<8>(ed.gu_code, grow, z, xo, lane, ag);
        row_dot_sw<8>(ed.gu_code, urow, z, xo, lane, au);
        for (int i = 0; i < 8; ++i) {
            float g = simd_sum(ag[i]), u = simd_sum(au[i]);
            if (lane == 0 && pid[i] >= 0) {
                const float sg_ = g / (1.0f + exp(-g));
                const float hh = min(sg_, 10.0f) * clamp(u, -10.0f, 10.0f);
                h2[(size_t) (base + c0 + i) * 1536 + r] = half(hh / sq);
            }
        }
    }
}
kernel void moe_dn_sorted(device const uchar* sec4 [[buffer(0)]], device const ExpertDev* ex [[buffer(1)]], device const float* topk_w [[buffer(2)]], device const int* cnt [[buffer(3)]],
                          device const int* off [[buffer(4)]], device const int* pairs [[buffer(5)]], device const half* h2 [[buffer(6)]], device float* y [[buffer(7)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int e = tgp.y;
    const int n = cnt[e];
    if (n == 0) return;
    const uint sg = tid >> 5, lane = tid & 31;
    const int r = tgp.x * 8 + sg;
    const ExpertDev ed = ex[e];
    device const uchar* dn = sec4 + ed.body_off + ed.dn_off;
    device const uchar* drow = dn + (size_t) r * code_row_bytes(ed.dn_code, 1536);
    const int base = off[e];
    for (int c0 = 0; c0 < n; c0 += 8) {
        int xo[8];
        int pid[8];
        for (int i = 0; i < 8; ++i) {
            if (c0 + i < n) {
                pid[i] = pairs[base + c0 + i];
                xo[i] = base + c0 + i;
            } else {
                pid[i] = -1;
                xo[i] = -1;
            }
        }
        float a[8];
        for (int i = 0; i < 8; ++i) a[i] = 0.f;
        row_dot_sw<8>(ed.dn_code, drow, h2, xo, lane, a);
        for (int i = 0; i < 8; ++i) {
            float v = simd_sum(a[i]);
            if (lane == 0 && pid[i] >= 0) y[(size_t) pid[i] * 1536 + r] = v * topk_w[pid[i]];
        }
    }
}

// ================================================================ MLA ==========================================================================================
static inline void rope_pair(float x0, float x1, int pos, int i, float theta, thread float& o0, thread float& o1) {
    float freq = 1.0f / pow(theta, float(2 * i) / 64.0f);
    float ang = float(pos) * freq;
    float sn = precise::sin(ang), cs = precise::cos(ang);
    o0 = x0 * cs - x1 * sn;
    o1 = x0 * sn + x1 * cs;
}

// KV cache write: cache row (576 halves) = [ rmsnorm(kv_a[0:512]) * norm_w | rope(kv_a[512:576]) ] at slot (p0+t) % ring (ring 0: slot = position)
kernel void mla_write_kv(device const float* kv_a [[buffer(0)]], device const float* norm_w [[buffer(1)]], device half* cache [[buffer(2)]], constant int& p0 [[buffer(3)]],
                         constant int& ring [[buffer(4)]], constant float& theta [[buffer(5)]], constant float& eps [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* row = kv_a + (size_t) t * MLA_ROW;
    float v0 = row[2 * tid], v1 = row[2 * tid + 1];
    float ss = tg_sum(v0 * v0 + v1 * v1, sh, tid, 256);
    float inv = 1.0f / sqrt(ss / float(MLA_KV) + eps);
    int pos = p0 + t;
    int slot = ring > 0 ? pos % ring : pos;
    device half* dst = cache + (size_t) slot * MLA_ROW;
    dst[2 * tid] = half(v0 * inv * norm_w[2 * tid]);
    dst[2 * tid + 1] = half(v1 * inv * norm_w[2 * tid + 1]);
    if (tid < 32) {
        float o0, o1;
        rope_pair(row[MLA_KV + 2 * tid], row[MLA_KV + 2 * tid + 1], pos, tid, theta, o0, o1);
        dst[MLA_KV + 2 * tid] = half(o0);
        dst[MLA_KV + 2 * tid + 1] = half(o1);
    }
}

// q absorb + rope.  q_b fp32 [M][NH][256] (192 nope | 64 pe) -> qa half [M][NH][576] = [ sum_i q_nope[i] * wk_b[h][i][:] (512) | rope(q_pe) (64) ].  wk_b half [NH][192][512]
// grid (NH, ceil(M/TT)), 256 threads
template <int TT>
kernel void mla_prep_q(device const float* q_b [[buffer(0)]], device const half* wk_b [[buffer(1)]], device half* qa [[buffer(2)]], constant int& M [[buffer(3)]], constant int& NH [[buffer(4)]],
                       constant int& p0 [[buffer(5)]], constant float& theta [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = tgp.x, t0 = tgp.y * TT;
    threadgroup float qs[TT][192];
    for (int idx = tid; idx < TT * 192; idx += 256) {
        int tt = idx / 192, i = idx % 192, t = t0 + tt;
        qs[tt][i] = t < M ? q_b[((size_t) t * NH + h) * 256 + i] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc0[TT], acc1[TT];
    for (int tt = 0; tt < TT; ++tt) acc0[tt] = acc1[tt] = 0.f;
    device const half2* w2 = (device const half2*) (wk_b + (size_t) h * 192 * 512);
    for (int i = 0; i < 192; ++i) {
        half2 w = w2[(size_t) i * 256 + tid];
        float wx = float(w.x), wy = float(w.y);
        for (int tt = 0; tt < TT; ++tt) {
            float q = qs[tt][i];
            acc0[tt] += q * wx;
            acc1[tt] += q * wy;
        }
    }
    for (int tt = 0; tt < TT; ++tt) {
        int t = t0 + tt;
        if (t < M) *((device half2*) (qa + ((size_t) t * NH + h) * MLA_ROW + 2 * tid)) = half2(half(acc0[tt]), half(acc1[tt]));
    }
    for (int idx = tid; idx < TT * 32; idx += 256) {
        int tt = idx / 32, i = idx % 32, t = t0 + tt;
        if (t >= M) continue;
        device const float* qp = q_b + ((size_t) t * NH + h) * 256 + 192 + 2 * i;
        float o0, o1;
        rope_pair(qp[0], qp[1], p0 + t, i, theta, o0, o1);
        *((device half2*) (qa + ((size_t) t * NH + h) * MLA_ROW + MLA_KV + 2 * i)) = half2(half(o0), half(o1));
    }
}
template [[host_name("mla_prep_q_1")]] kernel void mla_prep_q<1>(device const float*, device const half*, device half*, constant int&, constant int&, constant int&, constant float&, uint2, uint);
template [[host_name("mla_prep_q_8")]] kernel void mla_prep_q<8>(device const float*, device const half*, device half*, constant int&, constant int&, constant int&, constant float&, uint2, uint);

struct MlaArgs {
    int M, NH, p0, mode, window, ring, kmax;
    float scale;
};

// attention in the latent space, one threadgroup (8 simdgroups) per (head, query).  Keys are split over the simdgroups (key j -> simdgroup j % 8), each with its own online
// softmax; the learnable sink is the initial state of simdgroup 0 (m = sink, l = 1, no value).  mode 0 causal, 1 sliding window (ring), 2 explicit slot lists.
kernel void mla_attend(device const half* qa [[buffer(0)]], device const half* cache [[buffer(1)]], device const float* sinks [[buffer(2)]], device half* ctx [[buffer(3)]],
                       constant MlaArgs& a [[buffer(4)]], device const int* keys [[buffer(5)]], device const int* nkeys [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    const int h = tgp.x, t = tgp.y;
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
    float m = sg == 0 ? sinks[h] : -INFINITY;
    float l = sg == 0 ? 1.0f : 0.0f;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    for (int j = sg; j < nk; j += 8) {
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
        float p = fast::exp(d - nm);
        if (nm == m) {  // the running maximum did not change (the common case): no rescale
            l += p;
            for (int i = 0; i < 8; ++i) {
                half2 v = kr[lane + 32 * i];
                acc[2 * i] += p * float(v.x);
                acc[2 * i + 1] += p * float(v.y);
            }
        } else {
            float corr = fast::exp(m - nm);
            l = l * corr + p;
            for (int i = 0; i < 8; ++i) {
                half2 v = kr[lane + 32 * i];
                acc[2 * i] = acc[2 * i] * corr + p * float(v.x);
                acc[2 * i + 1] = acc[2 * i + 1] * corr + p * float(v.y);
            }
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
    for (int s = 1; s < 8; ++s) M = max(M, sm[s]);
    float L = 0.f, o0 = 0.f, o1 = 0.f;
    for (int s = 0; s < 8; ++s) {
        float f = exp(sm[s] - M);
        L += sl[s] * f;
        o0 += sacc[s][2 * tid] * f;
        o1 += sacc[s][2 * tid + 1] * f;
    }
    float il = 1.0f / L;
    *((device half2*) (ctx + ((size_t) t * a.NH + h) * MLA_KV + 2 * tid)) = half2(half(o0 * il), half(o1 * il));
}

// v up-projection (+ output gate): ctx half [M][NH][512], wv_b half [NH][256][512], gate fp32 [M][NH*256] (raw logits) -> o half [M][NH*256]. grid (NH, ceil(M/TT)), 256 threads
template <int TT>
kernel void mla_vup(device const half* ctx [[buffer(0)]], device const half* wv_b [[buffer(1)]], device const float* gate [[buffer(2)]], device half* o [[buffer(3)]], constant int& M [[buffer(4)]],
                    constant int& NH [[buffer(5)]], uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = tgp.x, t0 = tgp.y * TT;
    threadgroup float cs[TT][512];
    for (int idx = tid; idx < TT * 512; idx += 256) {
        int tt = idx / 512, c = idx % 512, t = t0 + tt;
        cs[tt][c] = t < M ? float(ctx[((size_t) t * NH + h) * 512 + c]) : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint w = tid >> 5, lane = tid & 31;
    for (int j = w; j < 256; j += 8) {
        device const half2* row = (device const half2*) (wv_b + ((size_t) h * 256 + j) * 512);
        float acc[TT];
        for (int tt = 0; tt < TT; ++tt) acc[tt] = 0.f;
        for (int r = 0; r < 8; ++r) {
            int c2 = lane + 32 * r;
            half2 wv = row[c2];
            float wx = float(wv.x), wy = float(wv.y);
            for (int tt = 0; tt < TT; ++tt) acc[tt] += cs[tt][2 * c2] * wx + cs[tt][2 * c2 + 1] * wy;
        }
        for (int tt = 0; tt < TT; ++tt) acc[tt] = simd_sum(acc[tt]);
        if (lane == 0) {
            for (int tt = 0; tt < TT; ++tt) {
                int t = t0 + tt;
                if (t < M) {
                    float v = acc[tt] * sigmoidf_(gate[(size_t) t * NH * 256 + (size_t) h * 256 + j]);
                    o[(size_t) t * NH * 256 + (size_t) h * 256 + j] = half(v);
                }
            }
        }
    }
}
template [[host_name("mla_vup_1")]] kernel void mla_vup<1>(device const half*, device const half*, device const float*, device half*, constant int&, constant int&, uint2, uint);
template [[host_name("mla_vup_8")]] kernel void mla_vup<8>(device const half*, device const half*, device const float*, device half*, constant int&, constant int&, uint2, uint);

// ================================================================ IndexPool (sparse layers) ====================================================================
static inline void idx_rope(float x0, float x1, int pos, int i, float theta, thread float& o0, thread float& o1) { rope_pair(x0, x1, pos, i, theta, o0, o1); }

// store iv, ig (fp32 [T][128] each) of tokens p0..p0+T-1 into the fp16 cache rows [ctx][256]
kernel void idx_store(device const float* iv [[buffer(0)]], device const float* ig [[buffer(1)]], device half* ivg [[buffer(2)]], constant int& p0 [[buffer(3)]], uint t [[threadgroup_position_in_grid]],
                      uint d [[thread_index_in_threadgroup]]) {
    ivg[(size_t) (p0 + t) * 256 + d] = half(iv[(size_t) t * 128 + d]);
    ivg[(size_t) (p0 + t) * 256 + 128 + d] = half(ig[(size_t) t * 128 + d]);
}

// pool completed groups g in [g_lo, g_lo + n): softmax over the 4 tokens of (ig + ape), weighted sum of iv, rmsnorm(key_norm), rope at 4g on dims 64..127. 128 threads
kernel void idx_pool(device const half* ivg [[buffer(0)]], device const float* ape [[buffer(1)]], device const float* knw [[buffer(2)]], device half* pooled [[buffer(3)]], constant int& g_lo [[buffer(4)]],
                     constant float& theta [[buffer(5)]], constant float& eps [[buffer(6)]], uint bx [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    threadgroup float sh[4];
    threadgroup float sv[128];
    int g = g_lo + bx;
    float l[4], mx = -3.0e38f;
    for (int j = 0; j < 4; ++j) {
        l[j] = float(ivg[(size_t) (4 * g + j) * 256 + 128 + d]) + ape[j * 128 + d];
        mx = max(mx, l[j]);
    }
    float s = 0;
    for (int j = 0; j < 4; ++j) {
        l[j] = exp(l[j] - mx);
        s += l[j];
    }
    float acc = 0;
    for (int j = 0; j < 4; ++j) acc += (l[j] / s) * float(ivg[(size_t) (4 * g + j) * 256 + d]);
    float ss = tg_sum(acc * acc, sh, d, 128);
    float inv = rsqrt(ss / 128.0f + eps);
    float v = acc * inv * knw[d];
    sv[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= 64) {
        int i = (d - 64) >> 1;
        float x0 = sv[64 + 2 * i], x1 = sv[64 + 2 * i + 1], o0, o1;
        idx_rope(x0, x1, 4 * g, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    pooled[(size_t) g * 128 + d] = half(v);
}

// iq fp32 [T][2048] (16 heads x 128) -> rope on dims 64..127 of every head at position p0+t -> half [T][2048]. grid (T, 16), 128 threads
kernel void idx_q_rope(device const float* iq [[buffer(0)]], device half* out [[buffer(1)]], constant int& p0 [[buffer(2)]], constant float& theta [[buffer(3)]], uint2 th [[threadgroup_position_in_grid]],
                       uint d [[thread_index_in_threadgroup]]) {
    int t = th.x, h = th.y;
    device const float* q = iq + (size_t) t * 2048 + h * 128;
    float v = q[d];
    if (d >= 64) {
        int i = (d - 64) >> 1;
        float x0 = q[64 + 2 * i], x1 = q[64 + 2 * i + 1], o0, o1;
        idx_rope(x0, x1, p0 + t, i, theta, o0, o1);
        v = (d & 1) ? o1 : o0;
    }
    out[(size_t) t * 2048 + h * 128 + d] = half(v);
}

// score[t][g] = sum_h relu(iq[t][h] . pooled[g]) * hw[t][h] * hscale for g < grp(t) = (p0+t)/4, else -inf. grid ((G+63)/64, T), 64 threads
kernel void idx_score(device const half* iq [[buffer(0)]], device const float* hw [[buffer(1)]], device const half* pooled [[buffer(2)]], device float* score [[buffer(3)]],
                      constant int& p0 [[buffer(4)]], constant int& G [[buffer(5)]], constant int& ldS [[buffer(6)]], constant float& hscale [[buffer(7)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sq[16 * 128];
    threadgroup float shw[16];
    int t = tgp.y, g = tgp.x * 64 + tid;
    for (uint i = tid; i < 16 * 128; i += 64) sq[i] = float(iq[(size_t) t * 2048 + i]);
    if (tid < 16) shw[tid] = hw[(size_t) t * 16 + tid] * hscale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g >= G) return;
    int grp = (p0 + t) >> 2;
    float res;
    if (g >= grp) {
        res = -INFINITY;
    } else {
        device const half* pr = pooled + (size_t) g * 128;
        res = 0;
        for (int h = 0; h < 16; ++h) {
            float dsum = 0;
            for (int k = 0; k < 128; ++k) dsum += sq[h * 128 + k] * float(pr[k]);
            res += max(dsum, 0.0f) * shw[h];
        }
    }
    score[(size_t) t * ldS + g] = res;
}

static inline uint f2key(float f) {
    uint b = as_type<uint>(f);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// per token: select the top n_top pools among g < grp (all when grp <= n_top) and emit the key list (cache slots = positions): the 4 tokens of each selected pool, then the own
// group's tokens 4*grp..t.  keys [T][kmax], nkeys [T].  256 threads
kernel void idx_select(device const float* score [[buffer(0)]], constant int& ldS [[buffer(1)]], constant int& p0 [[buffer(2)]], constant int& n_top [[buffer(3)]], constant int& kmax [[buffer(4)]],
                       device int* keys [[buffer(5)]], device int* nkeys [[buffer(6)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int red[8];
    int pos = p0 + t;
    int grp = pos >> 2;
    device int* kl = keys + (size_t) t * kmax;
    if (grp <= n_top) {
        for (int i = tid; i <= pos; i += 256) kl[i] = i;
        if (tid == 0) nkeys[t] = pos + 1;
        return;
    }
    device const float* sc = score + (size_t) t * ldS;
    threadgroup atomic_uint hist[256];
    threadgroup uint s_byte;
    threadgroup int s_above;
    uint kc[8];
    const bool cached = (grp <= 2048);
    if (cached) {
        for (int r = 0; r < 8; ++r) {
            const int g = int(tid) + r * 256;
            kc[r] = g < grp ? f2key(sc[g]) : 0u;
        }
    }
    uint prefix = 0u;
    int rem = n_top;
    const uint sg = tid >> 5, lane = tid & 31u;
    for (int pass = 0; pass < 4; ++pass) {
        const uint shift = uint((3 - pass) * 8);
        const uint mask = pass == 0 ? 0u : (~0u << (shift + 8u));
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cached) {
            for (int r = 0; r < 8; ++r) {
                const int g = int(tid) + r * 256;
                if (g < grp) {
                    const uint k = kc[r];
                    if (pass == 0 || (k & mask) == prefix)
                        atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
                }
            }
        } else {
            for (int g = int(tid); g < grp; g += 256) {
                const uint k = f2key(sc[g]);
                if (pass == 0 || (k & mask) == prefix)
                    atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c0 = atomic_load_explicit(&hist[255u - tid], memory_order_relaxed);
        const uint sg_inc = simd_prefix_inclusive_sum(c0);
        if (lane == 31u) red[sg] = int(sg_inc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint base = 0u;
        for (uint w = 0; w < sg; ++w) base += uint(red[w]);
        const uint cum = base + sg_inc;
        const uint prev = cum - c0;
        if (prev < uint(rem) && cum >= uint(rem)) {
            s_byte = 255u - tid;
            s_above = int(prev);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= (s_byte << shift);
        rem -= s_above;
    }
    // Exact counts from 4-pass radix descent: n_gt = elements strictly > thr, n_eq_need = ties == thr needed.
    const uint thr = prefix;
    const int n_eq_need = rem;
    const int n_gt = n_top - n_eq_need;
    threadgroup int red_eq[8];
    {   // Emit > thr at [0..n_gt-1] and first n_eq_need ties == thr at [n_gt..n_top-1] in deterministic increasing g order (100% parallel!)
        const int per = (grp + 255) / 256;
        const int g0 = int(tid) * per, g1 = min(grp, g0 + per);
        int cnt_gt = 0, cnt_eq = 0;
        for (int g = g0; g < g1; ++g) {
            const uint k = f2key(sc[g]);
            cnt_gt += (k > thr) ? 1 : 0;
            cnt_eq += (k == thr) ? 1 : 0;
        }
        const int ex_gt = simd_prefix_exclusive_sum(cnt_gt);
        const int ex_eq = simd_prefix_exclusive_sum(cnt_eq);
        if (lane == 31u) { red[sg] = ex_gt + cnt_gt; red_eq[sg] = ex_eq + cnt_eq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int base_gt = 0, base_eq = 0;
        for (uint w = 0; w < sg; ++w) { base_gt += red[w]; base_eq += red_eq[w]; }
        int sidx_gt = base_gt + ex_gt;
        int sidx_eq = base_eq + ex_eq;
        for (int g = g0; g < g1; ++g) {
            const uint k = f2key(sc[g]);
            if (k > thr) {
                *((device int4*) (kl + 4 * sidx_gt)) = int4(4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3);
                ++sidx_gt;
            } else if (k == thr && sidx_eq < n_eq_need) {
                const int d = n_gt + sidx_eq;
                *((device int4*) (kl + 4 * d)) = int4(4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3);
                ++sidx_eq;
            }
        }
    }
    const int nown = pos - 4 * grp + 1;
    for (int i = int(tid); i < nown; i += 256) kl[4 * n_top + i] = 4 * grp + i;
    if (tid == 0) nkeys[t] = 4 * n_top + nown;
}
