// hy4_cpu_experts.cpp - see hy4_cpu_experts.h
#include "hy4_cpu_experts.h"
#include "hy4_cpu_k512.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "hy4_io.h"
#include "strata/core/moe_wave_plan.hpp"
#include "strata/core/ssd_stream_scheduler.hpp"
#include "strata/platform/cgroup_mem.hpp"
#include "strata/platform/direct_file.hpp"
#include "strata/quant/sk_codecs.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
static inline float _cvtsh_ss(unsigned short h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}
#endif
#define HY4_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define HY4_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#define HY4_RELAX() std::this_thread::yield()
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <deque>
#include <unordered_map>

// Fast kernels:
//   k512    = AVX-512 BW + VNNI + VBMI + F16C (Ice Lake / Rocket Lake / Zen 4+), compiled in hy4_cpu_k512.cpp
//             so MSVC builds ONLY that TU with /arch:AVX512 while THIS TU compiles strictly with /arch:AVX2
//             (zero EVEX 0x62 instructions in AVX2, AVX-VNNI, or dispatch paths on Alder/Raptor/Arrow Lake).
//   k2_vnni = AVX2 + FMA + F16C + VEX AVX-VNNI (_mm256_dpbusd_avx_epi32, Alder/Raptor/Meteor/Arrow Lake).
//   k2      = AVX2 + FMA + F16C (Haswell / Zen 1-3+).
// Chosen at init by CPUID + XGETBV; order of preference: k512, k2_vnni, k2, library gemm_sk_q8_1.
// Override: STRATA_CPUEXP_KERNEL=avx512|avx2-vnni|avx2|lib; STRATA_FORCE_ISA=avx2|avx2-vnni|no-avx512|avx|sse.
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(HY4_NO_FAST)
#define HY4_KERNELS 1
#define HY4_NEON_KERNELS 0
#elif defined(__aarch64__) && !defined(HY4_NO_FAST)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif
#define HY4_KERNELS 0
#define HY4_NEON_KERNELS 1
#else
#define HY4_KERNELS 0
#define HY4_NEON_KERNELS 0
#endif
#define HY4_ANY_FAST (HY4_KERNELS || HY4_NEON_KERNELS)
#if HY4_KERNELS && ((defined(__clang__) && __clang_major__ >= 12) || (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(_MSC_VER) && _MSC_VER >= 1930))
#define HY4_AVX_VNNI 1
#else
#define HY4_AVX_VNNI 0
#endif
#if defined(__clang__)
#define HY4_PUSH_AVX2_VNNI _Pragma("clang attribute push (__attribute__((target(\"avx2,fma,f16c,avxvnni\"))), apply_to=function)")
#define HY4_PUSH_AVX2 _Pragma("clang attribute push (__attribute__((target(\"avx2,fma,f16c\"))), apply_to=function)")
#define HY4_POP_TARGET _Pragma("clang attribute pop")
#elif defined(__GNUC__)
#define HY4_PUSH_AVX2_VNNI _Pragma("GCC push_options") _Pragma("GCC target(\"avx2,fma,f16c,avxvnni\")")
#define HY4_PUSH_AVX2 _Pragma("GCC push_options") _Pragma("GCC target(\"avx2,fma,f16c\")")
#define HY4_POP_TARGET _Pragma("GCC pop_options")
#else
#define HY4_PUSH_AVX2_VNNI
#define HY4_PUSH_AVX2
#define HY4_POP_TARGET
#endif
#if HY4_NEON_KERNELS && !defined(__ARM_FEATURE_DOTPROD)
#if defined(__clang__)
#define HY4_PUSH_NEON_DP _Pragma("clang attribute push (__attribute__((target(\"+dotprod\"))), apply_to=function)")
#define HY4_POP_NEON _Pragma("clang attribute pop")
#elif defined(__GNUC__)
#define HY4_PUSH_NEON_DP _Pragma("GCC push_options") _Pragma("GCC target(\"+dotprod\")")
#define HY4_POP_NEON _Pragma("GCC pop_options")
#else
#define HY4_PUSH_NEON_DP
#define HY4_POP_NEON
#endif
#else
#define HY4_PUSH_NEON_DP
#define HY4_POP_NEON
#endif
#if HY4_KERNELS && !defined(_MSC_VER)
#include <cpuid.h>
#endif
#if HY4_KERNELS && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hy4 {
namespace {

using strata::quant::sk::BlockQ8_1_G64;
using strata::quant::sk::SkCode;
using strata::quant::sk::gemm_sk_q8_1;
using strata::quant::sk::quantize_q8_1_g64;
using strata::quant::sk::superblock_bytes;

constexpr int E = CpuExperts::kExperts, TOPK = CpuExperts::kTopK, D = CpuExperts::kDim;
constexpr int NB = D / 64;          // 24 activation blocks per row
constexpr int TC = CpuExperts::kChunkTokens;
constexpr int MG = 16;              // tokens per gemm_sk_q8_1 call (its m cap), library path
constexpr int MGF = 32;             // tokens per unit pass, fast path (the kernel itself works on tiles of 8 tokens)
constexpr int RB = 64;              // output rows per work unit (one Q8_1 block of h)
constexpr int NUNIT = D / RB;       // 24 units per expert per phase
constexpr float SWIGLU_LIMIT = 10.0f;

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void* aligned_alloc64(size_t bytes) {
    bytes = (bytes + 63) & ~(size_t)63;
    if (bytes == 0) bytes = 64;
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) p = nullptr;
#endif
    if (p) std::memset(p, 0, bytes);
    return p;
}
void aligned_free64(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

template <class T>
struct ABuf {
    T* p = nullptr;
    size_t n = 0;
    ABuf() = default;
    ABuf(const ABuf&) = delete;
    ABuf& operator=(const ABuf&) = delete;
    ~ABuf() { aligned_free64(p); }
    bool alloc(size_t cnt) {
        aligned_free64(p);
        n = cnt;
        p = (T*)aligned_alloc64(cnt * sizeof(T));
        return p != nullptr;
    }
};

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, ex = (h >> 10) & 0x1Fu, mn = h & 0x03FFu;
    uint32_t f;
    if (ex == 0) {
        if (mn == 0) f = sign;
        else {
            uint32_t e2 = 127 - 14, m2 = mn;
            while ((m2 & 0x0400u) == 0) { m2 <<= 1; --e2; }
            m2 &= 0x03FFu;
            f = sign | (e2 << 23) | (m2 << 13);
        }
    } else if (ex == 31) f = sign | 0x7F800000u | (mn << 13);
    else f = sign | ((ex + 112) << 23) | (mn << 13);
    float o;
    std::memcpy(&o, &f, 4);
    return o;
}

// Vectorizable expf (Cephes polynomial, ~1 ulp), used for silu on 64-element rows.
inline float fexpf(float x) {
    x = std::min(std::max(x, -87.0f), 88.0f);
    const float t = x * 1.44269504088896341f;
    const float n = (t + 12582912.0f) - 12582912.0f;  // round to nearest integer
    float r = x - n * 0.693359375f;
    r = r - n * (-2.12194440e-4f);
    float p = 1.9875691500E-4f;
    p = p * r + 1.3981999507E-3f;
    p = p * r + 8.3334519073E-3f;
    p = p * r + 4.1665795894E-2f;
    p = p * r + 1.6666665459E-1f;
    p = p * r + 5.0000001201E-1f;
    p = p * (r * r) + r + 1.0f;
    const int32_t ni = (int32_t)n + 127;
    const uint32_t bits = (uint32_t)ni << 23;
    float s;
    std::memcpy(&s, &bits, 4);
    return p * s;
}

#if HY4_KERNELS
HY4_PUSH_AVX2
static void swiglu_64_avx2(const float* g, const float* u, const float* sq, float* h) {
    const __m256 v_lo = _mm256_set1_ps(-87.0f);
    const __m256 v_hi = _mm256_set1_ps(88.0f);
    const __m256 v_log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 v_magic = _mm256_set1_ps(12582912.0f);
    const __m256 v_ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 v_ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 c0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201E-1f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_lim = _mm256_set1_ps(SWIGLU_LIMIT);
    const __m256 v_nlim = _mm256_set1_ps(-SWIGLU_LIMIT);
    const __m256i v_127 = _mm256_set1_epi32(127);
    for (int d = 0; d < RB; d += 8) {
        const __m256 gv = _mm256_loadu_ps(g + d);
        const __m256 uv = _mm256_loadu_ps(u + d);
        const __m256 sqv = _mm256_loadu_ps(sq + d);
        __m256 x = _mm256_sub_ps(_mm256_setzero_ps(), gv);
        x = _mm256_min_ps(_mm256_max_ps(x, v_lo), v_hi);
        const __m256 t = _mm256_mul_ps(x, v_log2e);
        const __m256 n = _mm256_sub_ps(_mm256_add_ps(t, v_magic), v_magic);
        __m256 r = _mm256_fnmadd_ps(n, v_ln2_hi, x);
        r = _mm256_fnmadd_ps(n, v_ln2_lo, r);
        __m256 p = _mm256_fmadd_ps(c0, r, c1);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c4);
        p = _mm256_fmadd_ps(p, r, c5);
        p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, v_one));
        const __m256i ni = _mm256_add_epi32(_mm256_cvttps_epi32(n), v_127);
        const __m256 s = _mm256_castsi256_ps(_mm256_slli_epi32(ni, 23));
        const __m256 ex = _mm256_mul_ps(p, s);
        const __m256 si = _mm256_div_ps(gv, _mm256_add_ps(v_one, ex));
        const __m256 sc = _mm256_min_ps(si, v_lim);
        const __m256 uc = _mm256_max_ps(v_nlim, _mm256_min_ps(uv, v_lim));
        _mm256_storeu_ps(h + d, _mm256_div_ps(_mm256_mul_ps(sc, uc), sqv));
    }
}
HY4_POP_TARGET
#endif

struct Rec {
    const uint8_t* body = nullptr;  // resident arena pointer (null for cold and GPU-owned experts)
    uint32_t dn_off = 0, body_bytes = 0;
    uint8_t gu = 0, dn = 0;
    uint32_t gu_sb = 0, dn_sb = 0;  // superblock bytes
    bool cold = false;              // not in the arena: the body is read from the .strata file on demand (set_cold_experts)
    uint64_t file_off = 0;          // absolute file offset of the body (4 KiB aligned)
};

struct DirE {
    uint16_t expert_id;
    uint8_t gu, dn;
    uint64_t body_off;
    uint32_t body_bytes, dn_off;
};

struct Job {
    int layer = 0, T = 0;
    const float* z = nullptr;
    const int32_t* idx = nullptr;
    const float* w = nullptr;
    const uint8_t* use = nullptr;
    float* zout = nullptr;
    bool accum = false;
    bool hi = false;
    double t_post = 0;
};

struct SpinBarrier {
    alignas(64) std::atomic<int> count{0};  // arrivals (RMW storm) and the release flag live on different cache lines:
    alignas(64) std::atomic<int> sense{0};  // the spinners only ever poll `sense`
    int n = 1;
    void wait(int& local_sense) {
        local_sense ^= 1;
        if (n == 1) return;
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(local_sense, std::memory_order_release);
        } else {
            uint32_t spins = 0;
            while (sense.load(std::memory_order_acquire) != local_sense) {
                HY4_RELAX();
                if ((++spins & 0xFFFF) == 0) std::this_thread::yield();
            }
        }
    }
};

struct alignas(64) PadCnt {  // one counter per cache line
    std::atomic<int> v{0};
};

static int g_pf_rows = 0;  // software prefetch distance in weight rows for the fast kernels (STRATA_CPUEXP_PF, 0 = hw prefetcher)

struct alignas(64) TScratch {
    ABuf<BlockQ8_1_G64> xg, xg2;  // gathered z blocks for up to 16 tokens (library path; xg2 = residual term, hi mode)
    ABuf<float> tg, tu, tg2;      // [16..32][64] gate / up (or down) results (tg2: second term, library path hi mode)
    int sense = 0;
};

// ==================================================================================================== fast kernels
// ActRow (4224 bytes) and hy4::k512 AVX-512 kernels live in hy4_cpu_k512.h / hy4_cpu_k512.cpp so this TU compiles
// strictly with /arch:AVX2 under MSVC (zero EVEX 0x62 instructions anywhere in hy4_cpu_experts.obj).
using hy4::ActRow;

#if HY4_KERNELS
namespace k2 {
// ==================================================================================================== AVX2 kernel
// Same idea as the AVX-512 kernel (planes P_j of unsigned weight codes, activations pre-permuted, float accumulation, one horizontal
// sum per row) on 256-bit vectors: vpmaddubsw (u8 code x s8 activation, no int16 saturation: codes <= 63 so a pair is <= 16002 and
// four planes added in int16 stay <= 31496 for S1.25..S5) + vpmaddwd with ones. S6 (codes up to 63) does two vpmaddwd.
// Layout "Z" (per super-block, per half h = groups 2h, 2h+1 of 64 weights; one ymm = 32 bytes):
//   xj[sb][j][32*h + 16*(k>>3) + 8*g1 + (k&7)] = x[(sb*4 + 2h + g1)*64 + 4*k + j]     (g1 = group within the half, k = 0..15)
//   i.e. a ymm holds [g0 k0-7 | g1 k0-7 | g0 k8-15 | g1 k8-15] of plane j.
//   xd/xs4[sb][8*h + 2*g1 + {0,1,4,5}] = d / 0.25*s of that block (each group on 4 dword lanes, matching vpmaddwd's output).
struct FastConsts2 {
    alignas(32) uint8_t hsidx[32];       // vpshufb control: hi-bit mask byte for each plane position (S3/S5 high bits)
    alignas(32) uint8_t oddmask[32];     // 0x80 on odd byte positions (they need the upper nibble of the mask byte)
    alignas(32) uint8_t cb_lo[4][32];    // S1.25 codebook column c, entries 0..15 (both lanes), +2 bias
    alignas(32) uint8_t cb_hi[4][32];    // entries 16..31
    alignas(32) uint8_t ctl125[32];      // S1.25: pshufb control, word j <- bytes (5j>>3, +1)
    alignas(32) int16_t mul125[16];      // S1.25: 2^(11 - (5j&7)): (word * mul) >> 11 = 5-bit field
    alignas(32) uint8_t tr8[32];         // S8_K: pshufb de-interleave 4x4 bytes per 128-bit lane
    alignas(32) int32_t effidx[2][8];
    FastConsts2() {
        static const int8_t cbk[32][4] = {  // = S125_CODEBOOK of sk_codecs.cpp / tools/squant/codecs.py
            {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
            {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
            {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},    {0, 0, 0, -2},
            {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},   {2, 0, 2, 0}};
        for (int p = 0; p < 32; ++p) {
            const int g1 = (p >> 3) & 1, kk = p & 7, up = p >> 4;
            hsidx[p] = (uint8_t)(8 * g1 + 4 * up + (kk >> 1));
            oddmask[p] = (p & 1) ? 0x80 : 0;
        }
        for (int c = 0; c < 4; ++c)
            for (int e = 0; e < 16; ++e) {
                cb_lo[c][e] = cb_lo[c][16 + e] = (uint8_t)(cbk[e][c] + 2);
                cb_hi[c][e] = cb_hi[c][16 + e] = (uint8_t)(cbk[16 + e][c] + 2);
            }
        static const int b0[8] = {0, 0, 1, 1, 2, 3, 3, 4}, off[8] = {0, 5, 2, 7, 4, 1, 6, 3};
        for (int lane = 0; lane < 2; ++lane)
            for (int j = 0; j < 8; ++j) {
                ctl125[16 * lane + 2 * j] = (uint8_t)b0[j];
                ctl125[16 * lane + 2 * j + 1] = (uint8_t)(b0[j] + 1);
                mul125[8 * lane + j] = (int16_t)(1 << (11 - off[j]));
            }
        for (int lane = 0; lane < 2; ++lane)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k) tr8[16 * lane + 4 * j + k] = (uint8_t)(4 * k + j);
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < 8; ++i) effidx[h][i] = 2 * h + ((i >> 1) & 1);
    }
};
static const FastConsts2 kc2;
HY4_PUSH_AVX2

static inline void store_qs(int8_t* base, int L, const int8_t* qs) {  // base = &xj[sb][0][0]
    const int h = L >> 1, g1 = L & 1;
    for (int j = 0; j < 4; ++j) {
        int8_t* dst = base + j * 64 + 32 * h + 8 * g1;
        for (int k = 0; k < 16; ++k) dst[16 * (k >> 3) + (k & 7)] = qs[4 * k + j];
    }
}
static inline void store_scale(float* base, int L, float v) {  // base = xd[sb] / xs4[sb] (16 floats)
    float* d = base + 8 * (L >> 1) + 2 * (L & 1);
    d[0] = d[1] = d[4] = d[5] = v;
}
static inline void store_block(ActRow* a, int b, const BlockQ8_1_G64& blk) {
    const int sb = b >> 2, L = b & 3;
    store_qs(&a->xj[sb][0][0], L, blk.qs);
    store_scale(a->xd[sb], L, blk.d);
    store_scale(a->xs4[sb], L, 0.25f * blk.s);
}
static inline void store_block_hi(ActRow* a, int b, const BlockQ8_1_G64& blk, const BlockQ8_1_G64& blk2) {
    const int sb = b >> 2, L = b & 3;
    store_qs(&a->xj[sb][0][0], L, blk.qs);
    store_qs(&a->xjb[sb][0][0], L, blk2.qs);
    store_scale(a->xd[sb], L, blk.d);
    store_scale(a->xdb[sb], L, blk2.d);
    store_scale(a->xs4[sb], L, 0.25f * (blk.s + blk2.s));
}

// byte vector with, for every plane position, the mask byte holding that element's bit, already shifted so the bit sits at bit j
// of the byte (odd positions need the upper nibble): hsel >> j & 1 = high bit of plane j
static inline __m256i make_hsel(const uint8_t* p16) {
    const __m256i hb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)p16));
    const __m256i hs = _mm256_shuffle_epi8(hb, _mm256_load_si256((const __m256i*)kc2.hsidx));
    return _mm256_blendv_epi8(hs, _mm256_srli_epi16(hs, 4), _mm256_load_si256((const __m256i*)kc2.oddmask));
}

// P[j] = plane j (unsigned codes) of half h of one super-block payload, layout Z
template <int FMT>
static inline void unpack_half(const uint8_t* pl, int h, __m256i (&P)[4]) {
    const __m256i m3 = _mm256_set1_epi8(3), m4 = _mm256_set1_epi8(4), m15 = _mm256_set1_epi8(15), m16 = _mm256_set1_epi8(16),
                  m30 = _mm256_set1_epi8(0x30), m00FF = _mm256_set1_epi16(0x00FF);
    (void)m3; (void)m4; (void)m16; (void)m30; (void)m00FF; (void)m15;
    if constexpr (FMT == 0) {
        // S1.25: 64 five-bit indices -> 32 bytes (Z order straight out of the pack), then 4 codebook columns by two pshufb + blend
        const uint8_t* g = pl + 20 * h;
        const __m256i ctl = _mm256_load_si256((const __m256i*)kc2.ctl125), mulc = _mm256_load_si256((const __m256i*)kc2.mul125);
        const __m256i X = _mm256_loadu2_m128i((const __m128i*)(g + 5), (const __m128i*)g);
        const __m256i Y = _mm256_loadu2_m128i((const __m128i*)(g + 15), (const __m128i*)(g + 10));
        const __m256i Xf = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(X, ctl), mulc), 11);
        const __m256i Yf = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(Y, ctl), mulc), 11);
        const __m256i idx = _mm256_packus_epi16(Xf, Yf);
        const __m256i sel = _mm256_slli_epi16(idx, 3);
        for (int j = 0; j < 4; ++j)
            P[j] = _mm256_blendv_epi8(_mm256_shuffle_epi8(_mm256_load_si256((const __m256i*)kc2.cb_lo[j]), idx),
                                      _mm256_shuffle_epi8(_mm256_load_si256((const __m256i*)kc2.cb_hi[j]), idx), sel);
    } else if constexpr (FMT == 2 || FMT == 3) {
        const __m256i wz = _mm256_permute4x64_epi64(_mm256_loadu_si256((const __m256i*)(pl + 32 * h)), 0xD8);
        P[0] = _mm256_and_si256(wz, m3);
        P[1] = _mm256_and_si256(_mm256_srli_epi16(wz, 2), m3);
        P[2] = _mm256_and_si256(_mm256_srli_epi16(wz, 4), m3);
        P[3] = _mm256_and_si256(_mm256_srli_epi16(wz, 6), m3);
        if constexpr (FMT == 3) {
            const __m256i hsel = make_hsel(pl + 64 + 16 * h);
            P[0] = _mm256_or_si256(P[0], _mm256_and_si256(_mm256_slli_epi16(hsel, 2), m4));
            P[1] = _mm256_or_si256(P[1], _mm256_and_si256(_mm256_slli_epi16(hsel, 1), m4));
            P[2] = _mm256_or_si256(P[2], _mm256_and_si256(hsel, m4));
            P[3] = _mm256_or_si256(P[3], _mm256_and_si256(_mm256_srli_epi16(hsel, 1), m4));
        }
    } else {
        // nibble plane (S4/S5/S6): byte i of group G = q[2i] | q[2i+1] << 4 -> E = even bytes, O = odd bytes of the 32-byte group
        const __m256i wa = _mm256_loadu_si256((const __m256i*)(pl + 64 * h)), wb = _mm256_loadu_si256((const __m256i*)(pl + 64 * h + 32));
        const __m256i E = _mm256_packus_epi16(_mm256_and_si256(wa, m00FF), _mm256_and_si256(wb, m00FF));
        const __m256i O = _mm256_packus_epi16(_mm256_srli_epi16(wa, 8), _mm256_srli_epi16(wb, 8));
        P[0] = _mm256_and_si256(E, m15);
        P[1] = _mm256_and_si256(_mm256_srli_epi16(E, 4), m15);
        P[2] = _mm256_and_si256(O, m15);
        P[3] = _mm256_and_si256(_mm256_srli_epi16(O, 4), m15);
        if constexpr (FMT == 5) {
            const __m256i hsel = make_hsel(pl + 128 + 16 * h);
            P[0] = _mm256_or_si256(P[0], _mm256_and_si256(_mm256_slli_epi16(hsel, 4), m16));
            P[1] = _mm256_or_si256(P[1], _mm256_and_si256(_mm256_slli_epi16(hsel, 3), m16));
            P[2] = _mm256_or_si256(P[2], _mm256_and_si256(_mm256_slli_epi16(hsel, 2), m16));
            P[3] = _mm256_or_si256(P[3], _mm256_and_si256(_mm256_slli_epi16(hsel, 1), m16));
        } else if constexpr (FMT == 6) {  // 2-bit high planes: byte i -> q[4i+j] in bits 2j
            const __m256i w6 = _mm256_permute4x64_epi64(_mm256_loadu_si256((const __m256i*)(pl + 128 + 32 * h)), 0xD8);
            P[0] = _mm256_or_si256(P[0], _mm256_and_si256(_mm256_slli_epi16(w6, 4), m30));
            P[1] = _mm256_or_si256(P[1], _mm256_and_si256(_mm256_slli_epi16(w6, 2), m30));
            P[2] = _mm256_or_si256(P[2], _mm256_and_si256(w6, m30));
            P[3] = _mm256_or_si256(P[3], _mm256_and_si256(_mm256_srli_epi16(w6, 2), m30));
        }
    }
}

// S8_K specialization for k2: 128 signed int8 weights for half h -> 4 signed int8 planes in layout Z
template <>
inline void unpack_half<8>(const uint8_t* pl, int h, __m256i (&P)[4]) {
    const uint8_t* g0 = pl + 128 * h;
    const uint8_t* g1 = g0 + 64;
    const __m256i tr = _mm256_load_si256((const __m256i*)kc2.tr8);
    const __m256i a0 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g0 + 32), (const __m128i*)(g0)), tr);
    const __m256i a1 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g0 + 48), (const __m128i*)(g0 + 16)), tr);
    const __m256i b0 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g1 + 32), (const __m128i*)(g1)), tr);
    const __m256i b1 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g1 + 48), (const __m128i*)(g1 + 16)), tr);
    const __m256i u0 = _mm256_unpacklo_epi32(a0, a1);
    const __m256i u1 = _mm256_unpackhi_epi32(a0, a1);
    const __m256i v0 = _mm256_unpacklo_epi32(b0, b1);
    const __m256i v1 = _mm256_unpackhi_epi32(b0, b1);
    P[0] = _mm256_unpacklo_epi64(u0, v0);
    P[1] = _mm256_unpackhi_epi64(u0, v0);
    P[2] = _mm256_unpacklo_epi64(u1, v1);
    P[3] = _mm256_unpackhi_epi64(u1, v1);
}

template <int FMT>
struct FmtInfo;
template <> struct FmtInfo<0> { static constexpr int SB = 48;  static constexpr float HALF = 2.0f; };
template <> struct FmtInfo<2> { static constexpr int SB = 70;  static constexpr float HALF = 1.5f; };
template <> struct FmtInfo<3> { static constexpr int SB = 102; static constexpr float HALF = 3.5f; };
template <> struct FmtInfo<4> { static constexpr int SB = 134; static constexpr float HALF = 7.5f; };
template <> struct FmtInfo<5> { static constexpr int SB = 166; static constexpr float HALF = 15.5f; };
template <> struct FmtInfo<6> { static constexpr int SB = 198; static constexpr float HALF = 31.5f; };
template <> struct FmtInfo<8> { static constexpr int SB = 262; static constexpr float HALF = 0.0f; };

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// int32 lanes [g0 g0 g1 g1 g0 g0 g1 g1] = sum over the 4 planes of (P_j . x_j) for one half
template <int FMT>
static inline __m256i dot_half(const __m256i (&P)[4], const int8_t* xj, const __m256i ones) {
    if constexpr (FMT == 8) {
        // Signed int8 x signed int8 widened to int16 -> madd_epi16 (zero int16 saturation even for -128 * -128)
        __m256i acc_lo = _mm256_setzero_si256(), acc_hi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i xv = _mm256_load_si256((const __m256i*)(xj + j * 64));
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(P[j])),
                                                                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv))));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(P[j], 1)),
                                                                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1))));
        }
        return _mm256_permute4x64_epi64(_mm256_hadd_epi32(acc_lo, acc_hi), 0xD8);
    }
    const __m256i m0 = _mm256_maddubs_epi16(P[0], _mm256_load_si256((const __m256i*)(xj)));
    const __m256i m1 = _mm256_maddubs_epi16(P[1], _mm256_load_si256((const __m256i*)(xj + 64)));
    const __m256i m2 = _mm256_maddubs_epi16(P[2], _mm256_load_si256((const __m256i*)(xj + 128)));
    const __m256i m3 = _mm256_maddubs_epi16(P[3], _mm256_load_si256((const __m256i*)(xj + 192)));
    const __m256i s01 = _mm256_add_epi16(m0, m1), s23 = _mm256_add_epi16(m2, m3);
    if constexpr (FMT == 6) return _mm256_add_epi32(_mm256_madd_epi16(s01, ones), _mm256_madd_epi16(s23, ones));  // 4 x 16002 would overflow int16
    return _mm256_madd_epi16(_mm256_add_epi16(s01, s23), ones);
}

// M tokens x 1 weight row (6 super-blocks of format FMT) -> out[t]
template <int FMT, int M, bool HI>
static inline void row_dot(const uint8_t* row, const ActRow* const* X, float* out) {
    constexpr int SB = FmtInfo<FMT>::SB;
    __m256 facc[M];
    for (int t = 0; t < M; ++t) facc[t] = _mm256_setzero_ps();
    const __m128 inv255 = _mm_set1_ps(255.0f);
    const __m256 halfv = _mm256_set1_ps(FmtInfo<FMT>::HALF);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = row + (size_t)s * SB;
        uint16_t dh;
        uint32_t u32;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&u32, blk + 2, 4);
        // eff_g = D * (u_g / 255) exactly as in sk_codecs (float division, then multiply)
        __m128 u4 = _mm_div_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)u32))), inv255);
#ifdef _MSC_VER
        u4 = _mm_mul_ps(u4, _mm_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)dh)))));
#else
        u4 = _mm_mul_ps(u4, _mm_set1_ps(_cvtsh_ss(dh)));
#endif
        if constexpr (FMT == 0) {  // S1.25: x fine[g], fine = 0.5 + nibble/15 (nibbles at payload bytes 40, 41)
            uint16_t fn;
            std::memcpy(&fn, blk + 46, 2);
            __m128i fv = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)fn), _mm_setr_epi32(0, 4, 8, 12)), _mm_set1_epi32(15));
            u4 = _mm_mul_ps(u4, _mm_add_ps(_mm_div_ps(_mm_cvtepi32_ps(fv), _mm_set1_ps(15.0f)), _mm_set1_ps(0.5f)));
        }
        const __m256 effc = _mm256_castps128_ps256(u4);
        for (int h = 0; h < 2; ++h) {
            const __m256 effv = _mm256_permutevar8x32_ps(effc, _mm256_load_si256((const __m256i*)kc2.effidx[h]));
            const __m256 effh = _mm256_mul_ps(effv, halfv);
            __m256i P[4];
            unpack_half<FMT>(blk + 6, h, P);
            for (int t = 0; t < M; ++t) {
                const ActRow* a = X[t];
                __m256i acc = dot_half<FMT>(P, &a->xj[s][0][32 * h], ones);
                const __m256 sc = _mm256_mul_ps(effv, _mm256_load_ps(&a->xd[s][8 * h]));
                facc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sc, facc[t]);
                if constexpr (HI) {  // residual term: same unpacked weights
                    const __m256i accb = dot_half<FMT>(P, &a->xjb[s][0][32 * h], ones);
                    facc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(accb), _mm256_mul_ps(effv, _mm256_load_ps(&a->xdb[s][8 * h])), facc[t]);
                }
                if constexpr (FMT != 8) facc[t] = _mm256_fnmadd_ps(effh, _mm256_load_ps(&a->xs4[s][8 * h]), facc[t]);
            }
        }
    }
    for (int t = 0; t < M; ++t) out[t] = hsum256(facc[t]);
}

// nrows rows x M tokens; out[i*nrows + r]
template <int FMT, int M, bool HI>
static void rows_m(const uint8_t* W, int nrows, const ActRow* const* X, float* out) {
    constexpr size_t RBYTES = 6 * (size_t)FmtInfo<FMT>::SB;
    const int pf = g_pf_rows;
    for (int r = 0; r < nrows; ++r) {
        if (pf && r + pf < nrows) {  // pull in the row `pf` rows ahead (crosses 4 KiB page boundaries, which the hardware prefetcher does not)
            const char* pp = (const char*)(W + (size_t)(r + pf) * RBYTES);
            for (size_t o = 0; o < RBYTES; o += 64) _mm_prefetch(pp + o, _MM_HINT_T0);
        }
        float o[M];
        row_dot<FMT, M, HI>(W + r * RBYTES, X, o);
        for (int i = 0; i < M; ++i) out[(size_t)i * nrows + r] = o[i];
    }
}

template <int FMT, bool HI>
static void rows_fmt(const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out) {
    for (int g0 = 0; g0 < m; g0 += 8) {
        const int mg = std::min(8, m - g0);
        const ActRow* const* x = X + g0;
        float* o = out + (size_t)g0 * nrows;
        switch (mg) {
            case 1: rows_m<FMT, 1, HI>(W, nrows, x, o); break;
            case 2: rows_m<FMT, 2, HI>(W, nrows, x, o); break;
            case 3: rows_m<FMT, 3, HI>(W, nrows, x, o); break;
            case 4: rows_m<FMT, 4, HI>(W, nrows, x, o); break;
            case 5: rows_m<FMT, 5, HI>(W, nrows, x, o); break;
            case 6: rows_m<FMT, 6, HI>(W, nrows, x, o); break;
            case 7: rows_m<FMT, 7, HI>(W, nrows, x, o); break;
            default: rows_m<FMT, 8, HI>(W, nrows, x, o); break;
        }
    }
}

// W: nrows rows of K=1536 in format `code` (0 = S1.25, 2..6, 8); X[m] activation rows; out[m][nrows]
static void rows_gemm(int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi = false) {
    if (hi) {
        switch (code) {
            case 0: rows_fmt<0, true>(W, nrows, X, m, out); break;
            case 2: rows_fmt<2, true>(W, nrows, X, m, out); break;
            case 3: rows_fmt<3, true>(W, nrows, X, m, out); break;
            case 4: rows_fmt<4, true>(W, nrows, X, m, out); break;
            case 5: rows_fmt<5, true>(W, nrows, X, m, out); break;
            case 8: rows_fmt<8, true>(W, nrows, X, m, out); break;
            default: rows_fmt<6, true>(W, nrows, X, m, out); break;
        }
        return;
    }
    switch (code) {
        case 0: rows_fmt<0, false>(W, nrows, X, m, out); break;
        case 2: rows_fmt<2, false>(W, nrows, X, m, out); break;
        case 3: rows_fmt<3, false>(W, nrows, X, m, out); break;
        case 4: rows_fmt<4, false>(W, nrows, X, m, out); break;
        case 5: rows_fmt<5, false>(W, nrows, X, m, out); break;
        case 8: rows_fmt<8, false>(W, nrows, X, m, out); break;
        default: rows_fmt<6, false>(W, nrows, X, m, out); break;
    }
}
HY4_POP_TARGET
}  // namespace k2

#if HY4_AVX_VNNI
namespace k2_vnni {
// ==================================================================================================== AVX2 + AVX-VNNI kernel
// Targets CPUs without AVX-512 that support VEX AVX-VNNI (Intel Alder Lake / Raptor Lake / Meteor Lake / Arrow Lake, e.g. Core Ultra 9 275HX).
// Reuses k2's ActRow layout ("Z") and constant tables (k2::kc2), replacing k2's 8-10 op maddubs+add+madd chain with 4x _mm256_dpbusd_epi32.
HY4_PUSH_AVX2_VNNI

static inline __m256i make_hsel(const uint8_t* p16) {
    const __m256i hb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)p16));
    const __m256i hs = _mm256_shuffle_epi8(hb, _mm256_load_si256((const __m256i*)k2::kc2.hsidx));
    return _mm256_blendv_epi8(hs, _mm256_srli_epi16(hs, 4), _mm256_load_si256((const __m256i*)k2::kc2.oddmask));
}

template <int FMT>
static inline void unpack_half(const uint8_t* pl, int h, __m256i (&P)[4]) {
    const __m256i m3 = _mm256_set1_epi8(3), m4 = _mm256_set1_epi8(4), m15 = _mm256_set1_epi8(15), m16 = _mm256_set1_epi8(16),
                  m30 = _mm256_set1_epi8(0x30), m00FF = _mm256_set1_epi16(0x00FF);
    (void)m3; (void)m4; (void)m16; (void)m30; (void)m00FF; (void)m15;
    if constexpr (FMT == 0) {
        const uint8_t* g = pl + 20 * h;
        const __m256i ctl = _mm256_load_si256((const __m256i*)k2::kc2.ctl125), mulc = _mm256_load_si256((const __m256i*)k2::kc2.mul125);
        const __m256i X = _mm256_loadu2_m128i((const __m128i*)(g + 5), (const __m128i*)g);
        const __m256i Y = _mm256_loadu2_m128i((const __m128i*)(g + 15), (const __m128i*)(g + 10));
        const __m256i Xf = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(X, ctl), mulc), 11);
        const __m256i Yf = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(Y, ctl), mulc), 11);
        const __m256i idx = _mm256_packus_epi16(Xf, Yf);
        const __m256i sel = _mm256_slli_epi16(idx, 3);
        for (int j = 0; j < 4; ++j)
            P[j] = _mm256_blendv_epi8(_mm256_shuffle_epi8(_mm256_load_si256((const __m256i*)k2::kc2.cb_lo[j]), idx),
                                      _mm256_shuffle_epi8(_mm256_load_si256((const __m256i*)k2::kc2.cb_hi[j]), idx), sel);
    } else if constexpr (FMT == 2 || FMT == 3) {
        const __m256i wz = _mm256_permute4x64_epi64(_mm256_loadu_si256((const __m256i*)(pl + 32 * h)), 0xD8);
        P[0] = _mm256_and_si256(wz, m3);
        P[1] = _mm256_and_si256(_mm256_srli_epi16(wz, 2), m3);
        P[2] = _mm256_and_si256(_mm256_srli_epi16(wz, 4), m3);
        P[3] = _mm256_and_si256(_mm256_srli_epi16(wz, 6), m3);
        if constexpr (FMT == 3) {
            const __m256i hsel = make_hsel(pl + 64 + 16 * h);
            P[0] = _mm256_or_si256(P[0], _mm256_and_si256(_mm256_slli_epi16(hsel, 2), m4));
            P[1] = _mm256_or_si256(P[1], _mm256_and_si256(_mm256_slli_epi16(hsel, 1), m4));
            P[2] = _mm256_or_si256(P[2], _mm256_and_si256(hsel, m4));
            P[3] = _mm256_or_si256(P[3], _mm256_and_si256(_mm256_srli_epi16(hsel, 1), m4));
        }
    } else if constexpr (FMT == 8) {
        const uint8_t* g0 = pl + 128 * h;
        const uint8_t* g1 = g0 + 64;
        const __m256i tr = _mm256_load_si256((const __m256i*)k2::kc2.tr8);
        const __m256i x80 = _mm256_set1_epi8((char)0x80);
        const __m256i a0 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g0 + 32), (const __m128i*)(g0)), tr);
        const __m256i a1 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g0 + 48), (const __m128i*)(g0 + 16)), tr);
        const __m256i b0 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g1 + 32), (const __m128i*)(g1)), tr);
        const __m256i b1 = _mm256_shuffle_epi8(_mm256_loadu2_m128i((const __m128i*)(g1 + 48), (const __m128i*)(g1 + 16)), tr);
        const __m256i u0 = _mm256_unpacklo_epi32(a0, a1);
        const __m256i u1 = _mm256_unpackhi_epi32(a0, a1);
        const __m256i v0 = _mm256_unpacklo_epi32(b0, b1);
        const __m256i v1 = _mm256_unpackhi_epi32(b0, b1);
        P[0] = _mm256_xor_si256(_mm256_unpacklo_epi64(u0, v0), x80);
        P[1] = _mm256_xor_si256(_mm256_unpackhi_epi64(u0, v0), x80);
        P[2] = _mm256_xor_si256(_mm256_unpacklo_epi64(u1, v1), x80);
        P[3] = _mm256_xor_si256(_mm256_unpackhi_epi64(u1, v1), x80);
    } else {
        const __m256i wa = _mm256_loadu_si256((const __m256i*)(pl + 64 * h)), wb = _mm256_loadu_si256((const __m256i*)(pl + 64 * h + 32));
        const __m256i E = _mm256_packus_epi16(_mm256_and_si256(wa, m00FF), _mm256_and_si256(wb, m00FF));
        const __m256i O = _mm256_packus_epi16(_mm256_srli_epi16(wa, 8), _mm256_srli_epi16(wb, 8));
        P[0] = _mm256_and_si256(E, m15);
        P[1] = _mm256_and_si256(_mm256_srli_epi16(E, 4), m15);
        P[2] = _mm256_and_si256(O, m15);
        P[3] = _mm256_and_si256(_mm256_srli_epi16(O, 4), m15);
        if constexpr (FMT == 5) {
            const __m256i hsel = make_hsel(pl + 128 + 16 * h);
            P[0] = _mm256_or_si256(P[0], _mm256_and_si256(_mm256_slli_epi16(hsel, 4), m16));
            P[1] = _mm256_or_si256(P[1], _mm256_and_si256(_mm256_slli_epi16(hsel, 3), m16));
            P[2] = _mm256_or_si256(P[2], _mm256_and_si256(_mm256_slli_epi16(hsel, 2), m16));
            P[3] = _mm256_or_si256(P[3], _mm256_and_si256(_mm256_slli_epi16(hsel, 1), m16));
        } else if constexpr (FMT == 6) {
            const __m256i w6 = _mm256_permute4x64_epi64(_mm256_loadu_si256((const __m256i*)(pl + 128 + 32 * h)), 0xD8);
            P[0] = _mm256_or_si256(P[0], _mm256_and_si256(_mm256_slli_epi16(w6, 4), m30));
            P[1] = _mm256_or_si256(P[1], _mm256_and_si256(_mm256_slli_epi16(w6, 2), m30));
            P[2] = _mm256_or_si256(P[2], _mm256_and_si256(w6, m30));
            P[3] = _mm256_or_si256(P[3], _mm256_and_si256(_mm256_srli_epi16(w6, 2), m30));
        }
    }
}

template <int FMT>
struct FmtInfo;
template <> struct FmtInfo<0> { static constexpr int SB = 48;  static constexpr float HALF = 2.0f; };
template <> struct FmtInfo<2> { static constexpr int SB = 70;  static constexpr float HALF = 1.5f; };
template <> struct FmtInfo<3> { static constexpr int SB = 102; static constexpr float HALF = 3.5f; };
template <> struct FmtInfo<4> { static constexpr int SB = 134; static constexpr float HALF = 7.5f; };
template <> struct FmtInfo<5> { static constexpr int SB = 166; static constexpr float HALF = 15.5f; };
template <> struct FmtInfo<6> { static constexpr int SB = 198; static constexpr float HALF = 31.5f; };
template <> struct FmtInfo<8> { static constexpr int SB = 262; static constexpr float HALF = 128.0f; };

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template <int FMT>
static inline __m256i dot_half(const __m256i (&P)[4], const int8_t* xj) {
    // VEX-encoded AVX-VNNI ({vex} vpdpbusd, 0xC4 prefix) — NEVER _mm256_dpbusd_epi32 which emits EVEX (0x62, AVX512-VNNI+VL) on MSVC!
    __m256i acc = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), P[0], _mm256_load_si256((const __m256i*)(xj)));
    acc = _mm256_dpbusd_avx_epi32(acc, P[1], _mm256_load_si256((const __m256i*)(xj + 64)));
    acc = _mm256_dpbusd_avx_epi32(acc, P[2], _mm256_load_si256((const __m256i*)(xj + 128)));
    acc = _mm256_dpbusd_avx_epi32(acc, P[3], _mm256_load_si256((const __m256i*)(xj + 192)));
    return acc;
}

template <int FMT, int M, bool HI>
static inline void row_dot(const uint8_t* row, const ActRow* const* X, float* out) {
    constexpr int SB = FmtInfo<FMT>::SB;
    __m256 facc[M];
    for (int t = 0; t < M; ++t) facc[t] = _mm256_setzero_ps();
    const __m128 inv255 = _mm_set1_ps(255.0f);
    const __m256 halfv = _mm256_set1_ps(FmtInfo<FMT>::HALF);
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = row + (size_t)s * SB;
        uint16_t dh;
        uint32_t u32;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&u32, blk + 2, 4);
        __m128 u4 = _mm_div_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)u32))), inv255);
#ifdef _MSC_VER
        u4 = _mm_mul_ps(u4, _mm_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)dh)))));
#else
        u4 = _mm_mul_ps(u4, _mm_set1_ps(_cvtsh_ss(dh)));
#endif
        if constexpr (FMT == 0) {
            uint16_t fn;
            std::memcpy(&fn, blk + 46, 2);
            __m128i fv = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)fn), _mm_setr_epi32(0, 4, 8, 12)), _mm_set1_epi32(15));
            u4 = _mm_mul_ps(u4, _mm_add_ps(_mm_div_ps(_mm_cvtepi32_ps(fv), _mm_set1_ps(15.0f)), _mm_set1_ps(0.5f)));
        }
        const __m256 effc = _mm256_castps128_ps256(u4);
        for (int h = 0; h < 2; ++h) {
            const __m256 effv = _mm256_permutevar8x32_ps(effc, _mm256_load_si256((const __m256i*)k2::kc2.effidx[h]));
            const __m256 effh = _mm256_mul_ps(effv, halfv);
            __m256i P[4];
            unpack_half<FMT>(blk + 6, h, P);
            for (int t = 0; t < M; ++t) {
                const ActRow* a = X[t];
                const __m256i acc = dot_half<FMT>(P, &a->xj[s][0][32 * h]);
                const __m256 sc = _mm256_mul_ps(effv, _mm256_load_ps(&a->xd[s][8 * h]));
                facc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sc, facc[t]);
                if constexpr (HI) {
                    const __m256i accb = dot_half<FMT>(P, &a->xjb[s][0][32 * h]);
                    facc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(accb), _mm256_mul_ps(effv, _mm256_load_ps(&a->xdb[s][8 * h])), facc[t]);
                }
                facc[t] = _mm256_fnmadd_ps(effh, _mm256_load_ps(&a->xs4[s][8 * h]), facc[t]);
            }
        }
    }
    for (int t = 0; t < M; ++t) out[t] = hsum256(facc[t]);
}

template <int FMT, int M, bool HI>
static void rows_m(const uint8_t* W, int nrows, const ActRow* const* X, float* out) {
    constexpr size_t RBYTES = 6 * (size_t)FmtInfo<FMT>::SB;
    const int pf = g_pf_rows;
    for (int r = 0; r < nrows; ++r) {
        if (pf && r + pf < nrows) {
            const char* pp = (const char*)(W + (size_t)(r + pf) * RBYTES);
            for (size_t o = 0; o < RBYTES; o += 64) _mm_prefetch(pp + o, _MM_HINT_T0);
        }
        float o[M];
        row_dot<FMT, M, HI>(W + r * RBYTES, X, o);
        for (int i = 0; i < M; ++i) out[(size_t)i * nrows + r] = o[i];
    }
}

template <int FMT, bool HI>
static void rows_fmt(const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out) {
    for (int g0 = 0; g0 < m; g0 += 8) {
        const int mg = std::min(8, m - g0);
        const ActRow* const* x = X + g0;
        float* o = out + (size_t)g0 * nrows;
        switch (mg) {
            case 1: rows_m<FMT, 1, HI>(W, nrows, x, o); break;
            case 2: rows_m<FMT, 2, HI>(W, nrows, x, o); break;
            case 3: rows_m<FMT, 3, HI>(W, nrows, x, o); break;
            case 4: rows_m<FMT, 4, HI>(W, nrows, x, o); break;
            case 5: rows_m<FMT, 5, HI>(W, nrows, x, o); break;
            case 6: rows_m<FMT, 6, HI>(W, nrows, x, o); break;
            case 7: rows_m<FMT, 7, HI>(W, nrows, x, o); break;
            default: rows_m<FMT, 8, HI>(W, nrows, x, o); break;
        }
    }
}

static void rows_gemm(int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi = false) {
    if (hi) {
        switch (code) {
            case 0: rows_fmt<0, true>(W, nrows, X, m, out); break;
            case 2: rows_fmt<2, true>(W, nrows, X, m, out); break;
            case 3: rows_fmt<3, true>(W, nrows, X, m, out); break;
            case 4: rows_fmt<4, true>(W, nrows, X, m, out); break;
            case 5: rows_fmt<5, true>(W, nrows, X, m, out); break;
            case 8: rows_fmt<8, true>(W, nrows, X, m, out); break;
            default: rows_fmt<6, true>(W, nrows, X, m, out); break;
        }
        return;
    }
    switch (code) {
        case 0: rows_fmt<0, false>(W, nrows, X, m, out); break;
        case 2: rows_fmt<2, false>(W, nrows, X, m, out); break;
        case 3: rows_fmt<3, false>(W, nrows, X, m, out); break;
        case 4: rows_fmt<4, false>(W, nrows, X, m, out); break;
        case 5: rows_fmt<5, false>(W, nrows, X, m, out); break;
        case 8: rows_fmt<8, false>(W, nrows, X, m, out); break;
        default: rows_fmt<6, false>(W, nrows, X, m, out); break;
    }
}
HY4_POP_TARGET
}  // namespace k2_vnni
#endif  // HY4_AVX_VNNI
#endif  // HY4_KERNELS (k2)

#if HY4_NEON_KERNELS
namespace kneon {
// ==================================================================================================== ARM64 NEON dotprod kernel
// 128-bit NEON vdotq_s32 (__ARM_FEATURE_DOTPROD) kernel for S1.25 and S2..S6.
// Layout in ActRow:
//   xj[sb][j][L*16 + k] = qs[4*k + j]  (4-plane de-interleaved via vld4q_s8; L = 0..3 sub-group, k = 0..15)
//   xd[sb][L]           = d of block sb*4+L
//   xs4[sb][L]          = s of block sb*4+L (or s_a + s_b in hi mode; 1 float lane per sub-group, no x4 replication)
// All unpacked weight codes are in 0..63 (fit in signed int8), so vdotq_s32(acc, P[j], xj) is exact.
struct NeonConsts {
    alignas(16) uint8_t cb125[4][32];  // S1.25 codebook column c (+2 bias -> 0..4), 32 entries for vqtbl2q_u8
    alignas(16) uint8_t hmask[4][16];  // bit test mask for S3/S5 high bits: even k -> bit j, odd k -> bit 4+j
    NeonConsts() {
        static const int8_t cbk[32][4] = {
            {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
            {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
            {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},    {0, 0, 0, -2},
            {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},   {2, 0, 2, 0}};
        for (int c = 0; c < 4; ++c)
            for (int e = 0; e < 32; ++e) cb125[c][e] = (uint8_t)(cbk[e][c] + 2);
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 16; ++k) hmask[j][k] = (uint8_t)(1u << ((k & 1) ? (4 + j) : j));
    }
};
static const NeonConsts knc;

template <int FMT>
struct FmtInfo;
template <> struct FmtInfo<0> { static constexpr int SB = 48;  static constexpr float HALF = 2.0f; };
template <> struct FmtInfo<2> { static constexpr int SB = 70;  static constexpr float HALF = 1.5f; };
template <> struct FmtInfo<3> { static constexpr int SB = 102; static constexpr float HALF = 3.5f; };
template <> struct FmtInfo<4> { static constexpr int SB = 134; static constexpr float HALF = 7.5f; };
template <> struct FmtInfo<5> { static constexpr int SB = 166; static constexpr float HALF = 15.5f; };
template <> struct FmtInfo<6> { static constexpr int SB = 198; static constexpr float HALF = 31.5f; };
template <> struct FmtInfo<8> { static constexpr int SB = 262; static constexpr float HALF = 0.0f; };

HY4_PUSH_NEON_DP

static inline void store_block(ActRow* a, int b, const BlockQ8_1_G64& blk) {
    const int sb = b >> 2, L = b & 3;
    const int8x16x4_t v = vld4q_s8(blk.qs);
    vst1q_s8(&a->xj[sb][0][L * 16], v.val[0]);
    vst1q_s8(&a->xj[sb][1][L * 16], v.val[1]);
    vst1q_s8(&a->xj[sb][2][L * 16], v.val[2]);
    vst1q_s8(&a->xj[sb][3][L * 16], v.val[3]);
    a->xd[sb][L] = blk.d;
    a->xs4[sb][L] = blk.s;
}

static inline void store_block_hi(ActRow* a, int b, const BlockQ8_1_G64& blk, const BlockQ8_1_G64& blk2) {
    const int sb = b >> 2, L = b & 3;
    const int8x16x4_t v = vld4q_s8(blk.qs);
    const int8x16x4_t v2 = vld4q_s8(blk2.qs);
    vst1q_s8(&a->xj[sb][0][L * 16], v.val[0]);
    vst1q_s8(&a->xj[sb][1][L * 16], v.val[1]);
    vst1q_s8(&a->xj[sb][2][L * 16], v.val[2]);
    vst1q_s8(&a->xj[sb][3][L * 16], v.val[3]);
    vst1q_s8(&a->xjb[sb][0][L * 16], v2.val[0]);
    vst1q_s8(&a->xjb[sb][1][L * 16], v2.val[1]);
    vst1q_s8(&a->xjb[sb][2][L * 16], v2.val[2]);
    vst1q_s8(&a->xjb[sb][3][L * 16], v2.val[3]);
    a->xd[sb][L] = blk.d;
    a->xdb[sb][L] = blk2.d;
    a->xs4[sb][L] = blk.s + blk2.s;
}

template <int FMT>
static inline void unpack_group(const uint8_t* pl, int L, int8x16_t (&P)[4]) {
    if constexpr (FMT == 0) {
        alignas(16) uint8_t idx_arr[16];
        for (int g = 0; g < 2; ++g) {
            const uint8_t* p = pl + L * 10 + g * 5;
            const uint64_t b40 = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
            for (int i = 0; i < 8; ++i) idx_arr[g * 8 + i] = (uint8_t)((b40 >> (5 * i)) & 31u);
        }
        const uint8x16_t vidx = vld1q_u8(idx_arr);
        for (int j = 0; j < 4; ++j) {
            uint8x16x2_t tab;
            tab.val[0] = vld1q_u8(knc.cb125[j]);
            tab.val[1] = vld1q_u8(knc.cb125[j] + 16);
            P[j] = vreinterpretq_s8_u8(vqtbl2q_u8(tab, vidx));
        }
    } else if constexpr (FMT == 2 || FMT == 3) {
        const uint8x16_t w = vld1q_u8(pl + L * 16);
        const uint8x16_t m3 = vdupq_n_u8(3);
        uint8x16_t p0 = vandq_u8(w, m3);
        uint8x16_t p1 = vandq_u8(vshrq_n_u8(w, 2), m3);
        uint8x16_t p2 = vandq_u8(vshrq_n_u8(w, 4), m3);
        uint8x16_t p3 = vshrq_n_u8(w, 6);
        if constexpr (FMT == 3) {
            const uint8x8_t h8 = vld1_u8(pl + 64 + L * 8);
            const uint8x16_t hd = vcombine_u8(vzip1_u8(h8, h8), vzip2_u8(h8, h8));
            const uint8x16_t m4 = vdupq_n_u8(4);
            p0 = vorrq_u8(p0, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[0])), m4));
            p1 = vorrq_u8(p1, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[1])), m4));
            p2 = vorrq_u8(p2, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[2])), m4));
            p3 = vorrq_u8(p3, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[3])), m4));
        }
        P[0] = vreinterpretq_s8_u8(p0);
        P[1] = vreinterpretq_s8_u8(p1);
        P[2] = vreinterpretq_s8_u8(p2);
        P[3] = vreinterpretq_s8_u8(p3);
    } else {
        const uint8x16x2_t w = vld2q_u8(pl + L * 32);
        const uint8x16_t m15 = vdupq_n_u8(15);
        uint8x16_t p0 = vandq_u8(w.val[0], m15);
        uint8x16_t p1 = vshrq_n_u8(w.val[0], 4);
        uint8x16_t p2 = vandq_u8(w.val[1], m15);
        uint8x16_t p3 = vshrq_n_u8(w.val[1], 4);
        if constexpr (FMT == 5) {
            const uint8x8_t h8 = vld1_u8(pl + 128 + L * 8);
            const uint8x16_t hd = vcombine_u8(vzip1_u8(h8, h8), vzip2_u8(h8, h8));
            const uint8x16_t m16 = vdupq_n_u8(16);
            p0 = vorrq_u8(p0, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[0])), m16));
            p1 = vorrq_u8(p1, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[1])), m16));
            p2 = vorrq_u8(p2, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[2])), m16));
            p3 = vorrq_u8(p3, vandq_u8(vtstq_u8(hd, vld1q_u8(knc.hmask[3])), m16));
        } else if constexpr (FMT == 6) {
            const uint8x16_t w6 = vld1q_u8(pl + 128 + L * 16);
            const uint8x16_t m30 = vdupq_n_u8(0x30);
            p0 = vorrq_u8(p0, vandq_u8(vshlq_n_u8(w6, 4), m30));
            p1 = vorrq_u8(p1, vandq_u8(vshlq_n_u8(w6, 2), m30));
            p2 = vorrq_u8(p2, vandq_u8(w6, m30));
            p3 = vorrq_u8(p3, vandq_u8(vshrq_n_u8(w6, 2), m30));
        }
        P[0] = vreinterpretq_s8_u8(p0);
        P[1] = vreinterpretq_s8_u8(p1);
        P[2] = vreinterpretq_s8_u8(p2);
        P[3] = vreinterpretq_s8_u8(p3);
    }
}

template <>
inline void unpack_group<8>(const uint8_t* pl, int L, int8x16_t (&P)[4]) {
    const int8x16x4_t v = vld4q_s8(reinterpret_cast<const int8_t*>(pl + L * 64));
    P[0] = v.val[0];
    P[1] = v.val[1];
    P[2] = v.val[2];
    P[3] = v.val[3];
}

template <int FMT, int M, bool HI>
static inline void row_dot(const uint8_t* row, const ActRow* const* X, float* out) {
    constexpr int SB = FmtInfo<FMT>::SB;
    float32x4_t facc[M];
    for (int t = 0; t < M; ++t) facc[t] = vdupq_n_f32(0.0f);
    const float32x4_t halfv = vdupq_n_f32(FmtInfo<FMT>::HALF);
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = row + (size_t)s * SB;
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        const float D = fp16_to_fp32(dh);
        alignas(16) float eff[4];
        for (int g = 0; g < 4; ++g) eff[g] = D * ((float)blk[2 + g] / 255.0f);
        if constexpr (FMT == 0) {
            const uint8_t f0 = blk[46], f1 = blk[47];
            eff[0] *= 0.5f + (float)(f0 & 15u) / 15.0f;
            eff[1] *= 0.5f + (float)((f0 >> 4) & 15u) / 15.0f;
            eff[2] *= 0.5f + (float)(f1 & 15u) / 15.0f;
            eff[3] *= 0.5f + (float)((f1 >> 4) & 15u) / 15.0f;
        }
        const float32x4_t effv = vld1q_f32(eff);
        const float32x4_t effh = vmulq_f32(effv, halfv);
        int32x4_t acc_a[M][4];
        int32x4_t acc_b[HI ? M : 1][4];
        for (int L = 0; L < 4; ++L) {
            int8x16_t P[4];
            unpack_group<FMT>(blk + 6, L, P);
            for (int t = 0; t < M; ++t) {
                const ActRow* a = X[t];
                int32x4_t acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, P[0], vld1q_s8(&a->xj[s][0][L * 16]));
                acc = vdotq_s32(acc, P[1], vld1q_s8(&a->xj[s][1][L * 16]));
                acc = vdotq_s32(acc, P[2], vld1q_s8(&a->xj[s][2][L * 16]));
                acc = vdotq_s32(acc, P[3], vld1q_s8(&a->xj[s][3][L * 16]));
                acc_a[t][L] = acc;
                if constexpr (HI) {
                    int32x4_t accb = vdupq_n_s32(0);
                    accb = vdotq_s32(accb, P[0], vld1q_s8(&a->xjb[s][0][L * 16]));
                    accb = vdotq_s32(accb, P[1], vld1q_s8(&a->xjb[s][1][L * 16]));
                    accb = vdotq_s32(accb, P[2], vld1q_s8(&a->xjb[s][2][L * 16]));
                    accb = vdotq_s32(accb, P[3], vld1q_s8(&a->xjb[s][3][L * 16]));
                    acc_b[t][L] = accb;
                }
            }
        }
        for (int t = 0; t < M; ++t) {
            const ActRow* a = X[t];
            const int32x4_t sum4 = vpaddq_s32(vpaddq_s32(acc_a[t][0], acc_a[t][1]), vpaddq_s32(acc_a[t][2], acc_a[t][3]));
            facc[t] = vfmaq_f32(facc[t], vcvtq_f32_s32(sum4), vmulq_f32(effv, vld1q_f32(a->xd[s])));
            if constexpr (HI) {
                const int32x4_t sum4b = vpaddq_s32(vpaddq_s32(acc_b[t][0], acc_b[t][1]), vpaddq_s32(acc_b[t][2], acc_b[t][3]));
                facc[t] = vfmaq_f32(facc[t], vcvtq_f32_s32(sum4b), vmulq_f32(effv, vld1q_f32(a->xdb[s])));
            }
            if constexpr (FMT != 8) facc[t] = vfmsq_f32(facc[t], effh, vld1q_f32(a->xs4[s]));
        }
    }
    for (int t = 0; t < M; ++t) out[t] = vaddvq_f32(facc[t]);
}

template <int FMT, int M, bool HI>
static void rows_m(const uint8_t* W, int nrows, const ActRow* const* X, float* out) {
    constexpr size_t RBYTES = 6 * (size_t)FmtInfo<FMT>::SB;
    const int pf = g_pf_rows;
    for (int r = 0; r < nrows; ++r) {
        if (pf) {
            const char* pp = (const char*)(W + (size_t)(r + pf) * RBYTES);
            for (size_t o = 0; o < RBYTES; o += 64) __builtin_prefetch(pp + o, 0, 3);
        }
        float o[M];
        row_dot<FMT, M, HI>(W + r * RBYTES, X, o);
        for (int i = 0; i < M; ++i) out[(size_t)i * nrows + r] = o[i];
    }
}

template <int FMT, bool HI>
static void rows_fmt(const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out) {
    for (int g0 = 0; g0 < m; g0 += 8) {
        const int mg = std::min(8, m - g0);
        const ActRow* const* x = X + g0;
        float* o = out + (size_t)g0 * nrows;
        switch (mg) {
            case 1: rows_m<FMT, 1, HI>(W, nrows, x, o); break;
            case 2: rows_m<FMT, 2, HI>(W, nrows, x, o); break;
            case 3: rows_m<FMT, 3, HI>(W, nrows, x, o); break;
            case 4: rows_m<FMT, 4, HI>(W, nrows, x, o); break;
            case 5: rows_m<FMT, 5, HI>(W, nrows, x, o); break;
            case 6: rows_m<FMT, 6, HI>(W, nrows, x, o); break;
            case 7: rows_m<FMT, 7, HI>(W, nrows, x, o); break;
            default: rows_m<FMT, 8, HI>(W, nrows, x, o); break;
        }
    }
}

static void rows_gemm(int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi = false) {
    if (hi) {
        switch (code) {
            case 0: rows_fmt<0, true>(W, nrows, X, m, out); break;
            case 2: rows_fmt<2, true>(W, nrows, X, m, out); break;
            case 3: rows_fmt<3, true>(W, nrows, X, m, out); break;
            case 4: rows_fmt<4, true>(W, nrows, X, m, out); break;
            case 5: rows_fmt<5, true>(W, nrows, X, m, out); break;
            case 8: rows_fmt<8, true>(W, nrows, X, m, out); break;
            default: rows_fmt<6, true>(W, nrows, X, m, out); break;
        }
        return;
    }
    switch (code) {
        case 0: rows_fmt<0, false>(W, nrows, X, m, out); break;
        case 2: rows_fmt<2, false>(W, nrows, X, m, out); break;
        case 3: rows_fmt<3, false>(W, nrows, X, m, out); break;
        case 4: rows_fmt<4, false>(W, nrows, X, m, out); break;
        case 5: rows_fmt<5, false>(W, nrows, X, m, out); break;
        case 8: rows_fmt<8, false>(W, nrows, X, m, out); break;
        default: rows_fmt<6, false>(W, nrows, X, m, out); break;
    }
}
HY4_POP_NEON
}  // namespace kneon
#endif  // HY4_NEON_KERNELS

static const char* kern_name(int k) {
    return k == 4 ? "fast(avx2-vnni)" : (k == 3 ? "fast(neon-dotprod)" : (k == 2 ? "fast(avx512-vbmi)" : (k == 1 ? "fast(avx2)" : "lib")));
}

static inline bool fast_code_ok(uint8_t c) { return c == 0 || (c >= 2 && c <= 6) || c == 8; }

#if HY4_KERNELS
// ---- CPU probe (same checks as the main engine's quant/isa.cpp: CPUID + XGETBV for OS support) and per-call dispatch ------------
struct KernCaps {
    bool avx2 = false, avx_vnni = false, avx512 = false;
};
static void cpuid_ex(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER)
    int a[4];
    __cpuidex(a, (int)leaf, (int)sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned)a[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
static unsigned long long xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
#endif
}
static KernCaps probe_kernel_caps() {
    KernCaps c;
    unsigned r[4] = {0, 0, 0, 0};
    cpuid_ex(0, 0, r);
    const unsigned maxleaf = r[0];
    cpuid_ex(1, 0, r);
    const unsigned ecx1 = r[2];
    const bool osxsave = (ecx1 >> 27) & 1;
    const unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
    const bool ymm = (xcr0 & 6) == 6, zmm = ymm && (xcr0 & 0xE0) == 0xE0;
    const bool avx = ymm && ((ecx1 >> 28) & 1), fma = (ecx1 >> 12) & 1, f16c = (ecx1 >> 29) & 1;
    if (maxleaf >= 7) {
        cpuid_ex(7, 0, r);
        const unsigned maxsub = r[0], ebx = r[1], ecx = r[2];
        c.avx2 = avx && fma && f16c && ((ebx >> 5) & 1);
        // AVX-512 F, BW, VL + VNNI + VBMI (+ the OS saving ZMM state)
        c.avx512 = c.avx2 && zmm && ((ebx >> 16) & 1) && ((ebx >> 30) & 1) && ((ebx >> 31) & 1) && ((ecx >> 11) & 1) && ((ecx >> 1) & 1);
#if HY4_AVX_VNNI
        if (c.avx2 && maxsub >= 1) {
            cpuid_ex(7, 1, r);
            c.avx_vnni = ((r[0] >> 4) & 1) != 0;  // CPUID.07H.01H:EAX[4] = AVX-VNNI
        }
#else
        (void)maxsub;
#endif
    }
    if (const char* f = std::getenv("STRATA_FORCE_ISA")) {  // as in the main engine: avx2 = no AVX-512/VNNI, no-avx512 / avx2-vnni = no AVX-512, avx / sse = no AVX2 kernels either
        if (!std::strcmp(f, "avx2")) { c.avx512 = false; c.avx_vnni = false; }
        else if (!std::strcmp(f, "avx2-vnni") || !std::strcmp(f, "no-avx512")) c.avx512 = false;
        else if (!std::strcmp(f, "avx") || !std::strcmp(f, "sse")) c.avx512 = c.avx_vnni = c.avx2 = false;
    }
    return c;
}
static inline void kern_store(int k, ActRow* a, int b, const BlockQ8_1_G64& blk) {
    if (k == 2) hy4::k512::store_block(a, b, blk);
    else k2::store_block(a, b, blk);
}
static inline void kern_store_hi(int k, ActRow* a, int b, const BlockQ8_1_G64& blk, const BlockQ8_1_G64& blk2) {
    if (k == 2) hy4::k512::store_block_hi(a, b, blk, blk2);
    else k2::store_block_hi(a, b, blk, blk2);
}
static void kern_gemm(int k, int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi) {
    if (k == 2) hy4::k512::rows_gemm(code, W, nrows, X, m, out, hi, g_pf_rows);
#if HY4_AVX_VNNI
    else if (k == 4) k2_vnni::rows_gemm(code, W, nrows, X, m, out, hi);
#endif
    else k2::rows_gemm(code, W, nrows, X, m, out, hi);
}
#elif HY4_NEON_KERNELS
struct KernCaps {
    bool neon_dp = false;
};
static KernCaps probe_kernel_caps() {
    KernCaps c;
#if defined(__linux__)
    c.neon_dp = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__ARM_FEATURE_DOTPROD) || defined(__APPLE__)
    c.neon_dp = true;
#endif
    if (const char* f = std::getenv("STRATA_FORCE_ISA")) {
        if (!std::strcmp(f, "scalar") || !std::strcmp(f, "lib")) c.neon_dp = false;
    }
    return c;
}
static inline void kern_store(int, ActRow* a, int b, const BlockQ8_1_G64& blk) { kneon::store_block(a, b, blk); }
static inline void kern_store_hi(int, ActRow* a, int b, const BlockQ8_1_G64& blk, const BlockQ8_1_G64& blk2) { kneon::store_block_hi(a, b, blk, blk2); }
static void kern_gemm(int, int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi) {
    kneon::rows_gemm(code, W, nrows, X, m, out, hi);
}
#endif

#if HY4_ANY_FAST
// Reconstruct BlockQ8_1_G64[24] from an ActRow for rare per-call library fallback (e.g. S1.5 matrix while fast kernel handles S1.25/S2..S6/S8).
static void unpack_act_row_blocks(int kern, const ActRow* a, bool hi, BlockQ8_1_G64* q1, BlockQ8_1_G64* q2) {
    const bool z_layout = (kern == 1 || kern == 4);
    for (int b = 0; b < NB; ++b) {
        const int sb = b >> 2, L = b & 3;
        int32_t sum1 = 0, sum2 = 0;
        if (z_layout) {
            const int h = L >> 1, g1 = L & 1;
            for (int j = 0; j < 4; ++j) {
                const int8_t* s1 = &a->xj[sb][j][32 * h + 8 * g1];
                const int8_t* s2 = hi ? &a->xjb[sb][j][32 * h + 8 * g1] : nullptr;
                for (int k = 0; k < 16; ++k) {
                    const int idx = 16 * (k >> 3) + (k & 7);
                    q1[b].qs[4 * k + j] = s1[idx];
                    sum1 += (int32_t)s1[idx];
                    if (hi) { q2[b].qs[4 * k + j] = s2[idx]; sum2 += (int32_t)s2[idx]; }
                }
            }
            q1[b].d = a->xd[sb][8 * h + 2 * g1];
            q1[b].s = q1[b].d * (float)sum1;
            if (hi) { q2[b].d = a->xdb[sb][8 * h + 2 * g1]; q2[b].s = q2[b].d * (float)sum2; }
        } else {
            for (int j = 0; j < 4; ++j) {
                const int8_t* s1 = &a->xj[sb][j][L * 16];
                const int8_t* s2 = hi ? &a->xjb[sb][j][L * 16] : nullptr;
                for (int k = 0; k < 16; ++k) {
                    q1[b].qs[4 * k + j] = s1[k];
                    sum1 += (int32_t)s1[k];
                    if (hi) { q2[b].qs[4 * k + j] = s2[k]; sum2 += (int32_t)s2[k]; }
                }
            }
            q1[b].d = (kern == 3) ? a->xd[sb][L] : a->xd[sb][L * 4];
            q1[b].s = q1[b].d * (float)sum1;
            if (hi) { q2[b].d = (kern == 3) ? a->xdb[sb][L] : a->xdb[sb][L * 4]; q2[b].s = q2[b].d * (float)sum2; }
        }
    }
}

static void kern_gemm_dispatch(int kern, int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out, bool hi) {
    if (fast_code_ok((uint8_t)code)) {
        kern_gemm(kern, code, W, nrows, X, m, out, hi);
        return;
    }
    // Per-call library fallback for non-fast formats (e.g. S1.5) so one matrix never disables fast kernels globally.
    BlockQ8_1_G64 q1[MG * NB], q2[MG * NB];
    float tmp2[MG * RB];
    for (int g0 = 0; g0 < m; g0 += MG) {
        const int mg = std::min(MG, m - g0);
        for (int i = 0; i < mg; ++i) unpack_act_row_blocks(kern, X[g0 + i], hi, q1 + (size_t)i * NB, q2 + (size_t)i * NB);
        float* o = out + (size_t)g0 * nrows;
        gemm_sk_q8_1(W, (SkCode)code, q1, o, nrows, D, mg);
        if (hi) {
            gemm_sk_q8_1(W, (SkCode)code, q2, tmp2, nrows, D, mg);
            for (int i = 0; i < mg * nrows; ++i) o[i] += tmp2[i];
        }
    }
}
#endif

// hi mode: q2 = Q8_1(x - dq(q)) per 64-block (x: nblocks*64 floats, q: its Q8_1 blocks)
static inline void residual_blocks(const float* x, const BlockQ8_1_G64* q, BlockQ8_1_G64* q2, int nblocks) {
    float r[64];
    for (int b = 0; b < nblocks; ++b) {
        const float d = q[b].d;
        for (int i = 0; i < 64; ++i) r[i] = x[b * 64 + i] - d * (float)q[b].qs[i];
        quantize_q8_1_g64(r, &q2[b], 64);
    }
}

}  // namespace

// ==================================================================================================== cold experts
// On-demand reader for experts that live in the .strata file instead of the arena: a small slot cache (LRU, slot = one expert
// body, 4 KiB aligned) filled by a few reader threads with large preads (O_DIRECT by default, so the page cache does not eat the
// RAM budget). Slot management is done by pool thread 0 only; the readers only see (dst, offset, length) requests.
struct ColdStore {
    struct Slot {
        int64_t key = -1;  // layer * 512 + expert, -1 = empty
        uint64_t tick = 0;
        int pin = 0;       // > 0: needed by the wave being computed / in flight
        int io_active = 0; // > 0: reader thread currently writing into this slot
    };
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    int io_mode = 0;  // 0 = direct (O_DIRECT / FILE_FLAG_NO_BUFFERING), 1 = buffered, 2 = buffered + POSIX_FADV_DONTNEED
    size_t slot_bytes = 0;
    size_t sector_align = 4096;
    int nslots = 0;
    uint8_t* mem = nullptr;
    std::mutex slot_mu;
    std::vector<Slot> slots;
    std::unordered_map<int64_t, int> where;
    uint64_t tick = 0;
    uint64_t gen = 1;
    strata::core::TwoPriorityExpertScheduler sched;
    std::atomic<int64_t> last_done_ns{0};
    std::atomic<bool> err{false};
    std::atomic<uint64_t> short_reads{0};

    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool read_chunk(uint64_t off, uint8_t* dst, size_t len) {
        size_t got = 0;
#ifdef _WIN32
        HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        bool ok = true;
        while (got < len) {
            OVERLAPPED ov{};
            const uint64_t o = off + got;
            ov.Offset = (DWORD)(o & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD)(o >> 32);
            ov.hEvent = ev;
            if (ev) ResetEvent(ev);
            const DWORD chunk = (DWORD)((len - got) > (1u << 30) ? (1u << 30) : (len - got));
            DWORD n = 0;
            if (!ReadFile(h, dst + got, chunk, &n, &ov)) {
                const DWORD le = GetLastError();
                if (le == ERROR_IO_PENDING) {
                    if (!GetOverlappedResult(h, &ov, &n, TRUE)) {
                        if (GetLastError() == ERROR_HANDLE_EOF) break;
                        ok = false;
                        break;
                    }
                } else if (le == ERROR_HANDLE_EOF) {
                    break;
                } else {
                    ok = false;
                    break;
                }
            }
            if (n == 0) break;
            got += (size_t)n;
        }
        if (ev) CloseHandle(ev);
        if (!ok) { err.store(true); return false; }
        if (got < len) std::memset(dst + got, 0, len - got);
#else
        while (got < len) {
            const ssize_t n = ::pread(fd, dst + got, len - got, (off_t)(off + got));
            if (n < 0) {
                if (errno == EINTR) continue;
                err.store(true);
                return false;
            }
            if (n == 0) break;
            got += (size_t)n;
        }
        if (got < len) std::memset(dst + got, 0, len - got);
        if (io_mode == 2) posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
#endif
        const int64_t t = now_ns();
        int64_t prev = last_done_ns.load(std::memory_order_relaxed);
        while (prev < t && !last_done_ns.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
        return true;
    }

    bool open(const std::string& path, size_t slot_bytes_, int nslots_, int nreaders, int mode, std::string& e) {
        slot_bytes = slot_bytes_;
        nslots = nslots_;
        io_mode = mode;
#ifdef _WIN32
        if (mode == 2) io_mode = 0;
        DWORD fl = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS | (io_mode == 0 ? FILE_FLAG_NO_BUFFERING : 0);
        h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, fl, nullptr);
        if (h == INVALID_HANDLE_VALUE && io_mode == 0) {
            std::fprintf(stderr, "[cpuexp] cold: FILE_FLAG_NO_BUFFERING open failed, using buffered overlapped reads\n");
            io_mode = 1;
            fl = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
            h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, fl, nullptr);
        }
        if (h == INVALID_HANDLE_VALUE) { e = "cannot open " + path + " for cold reads"; return false; }
        {
            FILE_STORAGE_INFO fsi{};
            if (GetFileInformationByHandleEx(h, FileStorageInfo, &fsi, sizeof(fsi))) {
                const size_t phys = fsi.PhysicalBytesPerSectorForPerformance ? (size_t)fsi.PhysicalBytesPerSectorForPerformance
                                                                             : (size_t)fsi.LogicalBytesPerSector;
                sector_align = strata::platform::round_up_sector_alignment(phys);
            }
            if (sector_align > 4096) {
                slot_bytes = (slot_bytes + sector_align - 1u) & ~(sector_align - 1u);
            }
        }
        mem = (uint8_t*)VirtualAlloc(nullptr, slot_bytes * (size_t)nslots, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!mem) { CloseHandle(h); h = INVALID_HANDLE_VALUE; e = "cannot allocate the cold-expert cache"; return false; }
#else
        if (mode == 0) {
            fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) {
                std::fprintf(stderr, "[cpuexp] cold: O_DIRECT open failed, using buffered reads + FADV_DONTNEED\n");
                io_mode = 2;
            }
        }
        if (fd < 0) fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { e = "cannot open " + path + " for cold reads"; return false; }
        if (posix_memalign((void**)&mem, 4096, slot_bytes * (size_t)nslots) != 0) { mem = nullptr; e = "cannot allocate the cold-expert cache"; return false; }
        std::memset(mem, 0, slot_bytes * (size_t)nslots);
#endif
        slots.assign(nslots, Slot());
        sched.start(
            std::max(1, nreaders), nslots,
            [this](int /*file_idx*/, uint64_t off, uint8_t* dst, size_t len) -> bool {
                return read_chunk(off, dst, len);
            },
            [this](const strata::core::StreamExpertReq& req, int& out_slot, uint8_t*& out_dst) -> bool {
                std::lock_guard<std::mutex> lk(slot_mu);
                if (where.find(req.key) != where.end()) return false;
                int best = -1;
                uint64_t bt = ~0ull;
                for (int s = 0; s < nslots; ++s) {
                    if (slots[s].pin > 0 || slots[s].io_active > 0) continue;
                    if (slots[s].key < 0) { best = s; break; }
                    if (slots[s].tick < bt) { bt = slots[s].tick; best = s; }
                }
                if (best < 0) return false;
                if (slots[best].key >= 0) where.erase(slots[best].key);
                slots[best].key = req.key;
                slots[best].tick = ++tick;
                slots[best].pin = 0;
                slots[best].io_active++;
                if (auto* st = sched.slot_state(best)) st->reset_for_load(req.key);
                where[req.key] = best;
                out_slot = best;
                out_dst = mem + (size_t)best * slot_bytes;
                return true;
            },
            [this](const strata::core::StreamExpertReq& req, bool ok, bool /*was_pf*/) {
                if (req.slot >= 0 && req.slot < nslots) {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    if (slots[req.slot].io_active > 0) slots[req.slot].io_active--;
                    if (!ok && slots[req.slot].key == req.key) {
                        where.erase(slots[req.slot].key);
                        slots[req.slot].key = -1;
                    }
                }
            });
        return true;
    }

    void enqueue_demand(int64_t key, int layer, int slot, uint8_t* dst, uint64_t off, size_t len, size_t dn_off) {
        const size_t align = std::max<size_t>(4096u, sector_align);
        const size_t alen = (len + align - 1u) & ~(align - 1u);
        auto req = strata::core::StreamExpertReq::make_contiguous_split(
            key, layer, gen, slot, dst, 0, off, alen, dn_off, align);
        sched.enqueue_demand(std::move(req));
    }

    void enqueue_prefetch(int64_t key, int layer, uint64_t off, size_t len, size_t dn_off) {
        {
            std::lock_guard<std::mutex> lk(slot_mu);
            if (where.find(key) != where.end()) return;
        }
        const size_t align = std::max<size_t>(4096u, sector_align);
        const size_t alen = (len + align - 1u) & ~(align - 1u);
        auto req = strata::core::StreamExpertReq::make_contiguous_split(
            key, layer, gen, -1, nullptr, 0, off, alen, dn_off, align);
        sched.enqueue_prefetch(std::move(req));
    }

    ~ColdStore() {
        sched.stop();
#ifdef _WIN32
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        if (mem) VirtualFree(mem, 0, MEM_RELEASE);
#else
        if (fd >= 0) ::close(fd);
        free(mem);
#endif
    }
};

struct CpuExperts::Impl {
    // ---- model
    int nlayers = 0;
    std::vector<Rec> recs;
    uint8_t* arena = nullptr;
    void* arena_base = nullptr;  // start of the mmap (arena is 2 MiB aligned inside it)
    size_t arena_size = 0, arena_map_size = 0;
    uint64_t n_resident = 0;
    bool locked = false;
    double t_load = 0;
    // ---- pool
    int nth = 0;
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cv_work, cv_done;
    std::atomic<uint64_t> gen{0}, done_gen{0};
    std::atomic<bool> stop{false};
    Job job;
    SpinBarrier bar;
    std::unique_ptr<TScratch[]> ts;
    int spin_us = 300;
    // ---- scratch (grown in submit, never while a job runs)
    ABuf<BlockQ8_1_G64> tokq, h2q;  // library path: [Tc][24], [P][24]
    ABuf<ActRow> tokA, h2A;         // fast path:    [Tc],     [P]
    ABuf<BlockQ8_1_G64> tokq2, h2q2;  // library path, hi mode: residual terms
    bool fast = false;              // kern != 0
    int kern = 0;                   // 0 = library gemm_sk_q8_1, 1 = AVX2 kernel (k2), 2 = AVX-512 kernel (k512); chosen at init
    bool hi = false;                // high-precision activations (STRATA_CPUEXP_ACT=hi / set_hi_activations)
    ABuf<float> Y;                  // [P][1536]
    size_t cap_tok = 0;
    std::vector<int> cnt, aoff, act_e, act_o, act_n, pair_tok, posof;
    std::vector<uint8_t> tok_used;
    PadCnt qctr, guctr, dnctr, cbctr, fctr;  // unit counters, one cache line each
    std::unique_ptr<PadCnt[]> gdone, dn_cc;  // flow scheduler: finished gate/up units per active expert, finished down units per column chunk
    int sched = 0;                           // 0 = barriered phases, 1 = gate/up -> down -> combine dataflow (STRATA_CPUEXP_SCHED)
    // ---- cold experts (set_cold_experts): body read from the file on demand; jobs touching them run in waves
    std::string path;
    std::vector<DirE> dir;
    uint64_t s4_off = 0;
    std::unique_ptr<ColdStore> cs;
    std::vector<const uint8_t*> act_body;    // body pointer per active expert (arena, or cold slot)
    std::vector<int> act_slot;               // cold slot per active expert (-1 = resident)
    bool cold_job = false, wave_finished = false;
    int nA = 0;                              // active experts [0, nA) are resident / cache hits (wave 0); the rest are cold misses
    int cur_a0 = 0, cur_a1 = 0, nxt_a0 = 0, nxt_a1 = 0, next_miss = 0;
    int64_t batch_start_ns = 0;
    bool batch_open = false;
    const Rec* cold_lr = nullptr;            // recs of the layer of the current job (for cold_issue)
    int cold_layer = 0;
    bool cold_err_reported = false;
    void cold_issue();
    void cold_wait();
    void cold_advance(int w);
    bool set_cold(const std::vector<std::vector<bool>>& cold, std::string& err);
    int nact = 0;
    // ---- stats (written by thread 0 only, read after wait())
    CpuExpertStats st;
    std::atomic<uint64_t> posted{0};
    // ---- next-layer L3 prefetch (STRATA_L3_PF, preempted immediately when gen != seen)
    bool l3_pf_enabled = true;
    std::atomic<uint64_t> pf_gen{0};
    std::atomic<int> pf_cursor{0};
    int pf_layer = -1;
    int pf_n = 0;
    int32_t pf_experts[64] = {};
    void do_l3_prefetch(int tid, uint64_t seen);

    ~Impl() {
        stop.store(true);
        {
            std::lock_guard<std::mutex> lk(m);
        }
        cv_work.notify_all();
        for (auto& t : threads) t.join();
#ifdef _WIN32
        if (arena) VirtualFree(arena, 0, MEM_RELEASE);
#else
        if (arena_base) munmap(arena_base, arena_map_size);
#endif
    }

    // ------------------------------------------------------------------------------------------------ load
    bool alloc_arena(size_t bytes, std::string& err) {
        arena_size = bytes;
        locked = false;
        if (bytes == 0) {
            arena = nullptr;
            arena_map_size = 0;
            return true;
        }
        const char* mlock_env = std::getenv("HY4_CPUEXP_MLOCK");
        bool want_lock = (mlock_env && mlock_env[0] == '1');
        if (!mlock_env) {
            // Only lock RAM pages when there is generous free RAM headroom (>= 16 GiB above arena) so
            // tight 32-64 GB hosts (and Windows WDDM commit limits) never suffer swap pressure or working-set faults.
#ifdef _WIN32
            MEMORYSTATUSEX ms{};
            ms.dwLength = sizeof(ms);
            if (GlobalMemoryStatusEx(&ms) && ms.ullAvailPhys > (uint64_t)bytes + (16ull << 30)) want_lock = true;
#else
            unsigned long long avail_kb = 0;
            if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
                char ln[256];
                while (std::fgets(ln, sizeof ln, f))
                    if (std::sscanf(ln, "MemAvailable: %llu kB", &avail_kb) == 1) break;
                std::fclose(f);
            }
            if (avail_kb * 1024ull > (unsigned long long)bytes + (16ull << 30)) want_lock = true;
#endif
        }
#ifdef _WIN32
        arena = (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        arena_map_size = bytes;
        if (!arena) { err = "VirtualAlloc failed for expert arena"; return false; }
        if (want_lock) locked = VirtualLock(arena, bytes) != 0;
#else
        const size_t HP = 2u << 20;
        arena_map_size = bytes + HP;
        void* p = mmap(nullptr, arena_map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { err = "mmap failed for expert arena (" + std::to_string(bytes >> 20) + " MiB)"; return false; }
        arena_base = p;
        uint8_t* a = (uint8_t*)(((uintptr_t)p + HP - 1) & ~(uintptr_t)(HP - 1));
#ifdef MADV_HUGEPAGE
        madvise(a, bytes, MADV_HUGEPAGE);
#endif
        arena = a;
        if (want_lock) locked = mlock(arena, bytes) == 0;  // best effort (RLIMIT_MEMLOCK)
#endif
        return true;
    }

    bool load(const std::string& path, int n_threads, const std::vector<std::vector<bool>>* mask, std::string& err);

    // ------------------------------------------------------------------------------------------------ pool
    void start_threads(int n);
    void worker(int tid);
    void run_job(int tid);
    void plan(const Job& J, int t0, int Tc);
    void gu_unit(int tid, const Job& J, int a, int b);
    void dn_unit(int tid, const Job& J, int a, int c);
    void combine_unit(const Job& J, int t0, int tl, int c0);
    void combine_cols(const Job& J, int t0, int Tc, int c);
    void flow_phase(int tid, const Job& J, int t0, int Tc);
};

// ==================================================================================================== load
namespace {
struct CpuTarget {
    uint16_t group = 0;
    int cpu = 0;
};
std::vector<CpuTarget> pin_list(int want);
void pin_current_thread(const CpuTarget& t);
}  // namespace

// header + Section 3 directory of a .strata (no zstd needed)
static bool read_dir(const std::string& path, std::vector<DirE>& dir, uint64_t& s4_off, uint64_t& s4_len, std::string& err) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { err = "cannot open " + path; return false; }
    auto seek = [&](uint64_t off) {
#ifdef _WIN32
        return _fseeki64(fp, (__int64)off, SEEK_SET) == 0;
#else
        return fseeko(fp, (off_t)off, SEEK_SET) == 0;
#endif
    };
    uint8_t frame[4096];
    if (std::fread(frame, 1, 4096, fp) != 4096 || std::memcmp(frame, "STRATA\0\1", 8) != 0) {
        std::fclose(fp);
        err = "not a .strata file: " + path;
        return false;
    }
    auto sec = [&](int i, uint64_t& off, uint64_t& len) {
        std::memcpy(&off, frame + 0x10 + i * 64, 8);
        std::memcpy(&len, frame + 0x10 + i * 64 + 8, 8);
    };
    uint64_t s3_off, s3_len;
    sec(3, s3_off, s3_len);
    sec(4, s4_off, s4_len);
    if (s3_len == 0 || s3_len % 32 != 0 || s4_len == 0) { std::fclose(fp); err = "no expert directory / Section 4"; return false; }
    std::vector<uint8_t> raw(s3_len);
    if (!seek(s3_off) || std::fread(raw.data(), 1, raw.size(), fp) != raw.size()) { std::fclose(fp); err = "short read of Section 3"; return false; }
    std::fclose(fp);
    const size_t ndir = raw.size() / 32;
    if ((ndir % E) != 0 || ndir == 0) { err = "expert directory is not layers x 512"; return false; }
    dir.resize(ndir);
    for (size_t i = 0; i < ndir; ++i) {
        const uint8_t* r = raw.data() + i * 32;
        DirE& d = dir[i];
        std::memcpy(&d.expert_id, r, 2);
        d.gu = r[2];
        d.dn = r[3];
        std::memcpy(&d.body_off, r + 8, 8);
        std::memcpy(&d.body_bytes, r + 16, 4);
        std::memcpy(&d.dn_off, r + 20, 4);
    }
    return true;
}

bool CpuExperts::Impl::load(const std::string& path, int n_threads, const std::vector<std::vector<bool>>* mask, std::string& err) {
    const double t0 = now_s();
    std::vector<DirE> dir;
    uint64_t s4_off = 0, s4_len = 0;
    if (!read_dir(path, dir, s4_off, s4_len, err)) return false;
    this->path = path;
    this->dir = dir;
    this->s4_off = s4_off;
    const size_t ndir = dir.size();
    nlayers = (int)(ndir / E);
    // residency + arena layout
    recs.assign((size_t)nlayers * E, Rec());
    std::vector<uint64_t> slot(ndir, ~0ull);  // arena offset per directory entry
    size_t cursor = 0;
    for (size_t i = 0; i < ndir; ++i) {
        const DirE& d = dir[i];
        const int layer = (int)(i / E);
        if (d.expert_id >= E) { err = "bad expert_id in directory"; return false; }
        bool res = true;
        if (mask) res = layer < (int)mask->size() && d.expert_id < (*mask)[layer].size() && (*mask)[layer][d.expert_id];
        if (!res) continue;
        const uint32_t gsb = (uint32_t)superblock_bytes((SkCode)d.gu), dsb = (uint32_t)superblock_bytes((SkCode)d.dn);
        if (!gsb || !dsb) { err = "unsupported expert code in directory"; return false; }
        if (d.dn_off != 3072u + 18432u * gsb || (uint64_t)d.dn_off + 9216ull * dsb > d.body_bytes) { err = "inconsistent expert body layout"; return false; }
        if ((s4_off + d.body_off) % 4096) { err = "expert body not 4 KiB aligned in file"; return false; }
        slot[i] = cursor;
        cursor += ((size_t)d.body_bytes + 4095) & ~(size_t)4095;
        ++n_resident;
    }
    if (cursor == 0) cursor = 4096;
    if (!alloc_arena(cursor, err)) return false;
    // read runs of adjacent resident bodies
    struct Run { uint64_t file_off, arena_off, bytes; };
    std::vector<Run> runs;
    for (size_t i = 0; i < ndir; ++i) {
        if (slot[i] == ~0ull) continue;
        const DirE& d = dir[i];
        const uint64_t fo = s4_off + d.body_off, len = ((uint64_t)d.body_bytes + 4095) & ~4095ull;
        if (!runs.empty()) {
            Run& r = runs.back();
            if (r.file_off + r.bytes == fo && r.arena_off + r.bytes == slot[i] && r.bytes + len <= (128ull << 20)) {
                r.bytes += len;
                continue;
            }
        }
        runs.push_back({fo, slot[i], len});
    }
    hy4::AlignedFile f;
    if (!f.open(path, true)) { err = "cannot open " + path + " for body reads"; return false; }
    std::atomic<size_t> next{0};
    std::atomic<bool> fail{false};
    auto reader = [&]() {
        for (;;) {
            const size_t k = next.fetch_add(1);
            if (k >= runs.size() || fail.load()) return;
            const Run& r = runs[k];
            size_t want = (size_t)r.bytes, got = f.read_at(r.file_off, arena + r.arena_off, want);
            // the very last body of the file may be clipped at EOF (padding stays zero)
            if (got < want && r.file_off + got < f.size()) fail.store(true);
        }
    };
    const int nr = (int)std::min<size_t>(4, std::max<size_t>(1, runs.size()));
    std::vector<CpuTarget> rpins;
    const char* numa_env = std::getenv("STRATA_NUMA_ARENA");
    if (!numa_env || numa_env[0] != '0') rpins = pin_list(nr);
    std::vector<std::thread> rt;
    for (int i = 0; i < nr; ++i) {
        rt.emplace_back([&, i]() {
            if ((size_t)i < rpins.size()) pin_current_thread(rpins[i]);
            reader();
        });
    }
    for (auto& t : rt) t.join();
    if (fail.load()) { err = "short read while loading expert bodies"; return false; }
    for (size_t i = 0; i < ndir; ++i) {
        if (slot[i] == ~0ull) continue;
        const DirE& d = dir[i];
        Rec& r = recs[(size_t)(i / E) * E + d.expert_id];
        r.body = arena + slot[i];
        r.dn_off = d.dn_off;
        r.body_bytes = d.body_bytes;
        r.gu = d.gu;
        r.dn = d.dn;
        r.gu_sb = (uint32_t)superblock_bytes((SkCode)d.gu);
        r.dn_sb = (uint32_t)superblock_bytes((SkCode)d.dn);
    }
#ifndef _WIN32
    {   // transparent huge pages: report how much of the arena is backed by 2 MiB pages; STRATA_CPUEXP_COLLAPSE=1 asks the kernel
        // (MADV_COLLAPSE, Linux >= 6.1) to collapse the populated arena into huge pages now (fewer TLB misses on the streaming reads)
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
        auto anon_huge_kb = []() {
            long v = -1;
            if (std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r")) {
                char line[256];
                while (std::fgets(line, sizeof(line), f))
                    if (!std::strncmp(line, "AnonHugePages:", 14)) { std::sscanf(line + 14, " %ld", &v); break; }
                std::fclose(f);
            }
            return v;
        };
        const long before = anon_huge_kb();
        size_t okb = 0, badb = 0;
        const double tc0 = now_s();
        const char* ce = std::getenv("STRATA_CPUEXP_COLLAPSE");
        if (ce && ce[0] == '1') {
            const size_t CH = 64u << 20;
            for (size_t off = 0; off < arena_size; off += CH) {
                const size_t len = std::min(CH, arena_size - off) & ~(size_t)((2u << 20) - 1);
                if (!len) continue;
                if (madvise(arena + off, len, MADV_COLLAPSE) == 0) okb += len;
                else badb += len;
            }
        }
        std::fprintf(stderr, "[cpuexp] THP: AnonHugePages %.2f GB -> %.2f GB of %.2f GB arena (collapse %s: %zu MiB ok, %zu MiB failed, %.1f s)\n",
                     before / 1048576.0, anon_huge_kb() / 1048576.0, arena_size / 1073741824.0, (ce && ce[0] == '1') ? "on" : "off", okb >> 20, badb >> 20, now_s() - tc0);
    }
#endif
    t_load = now_s() - t0;
    kern = 0;
#if HY4_KERNELS
    {
        const KernCaps kcaps = probe_kernel_caps();
        const int best = kcaps.avx512 ? 2 : (kcaps.avx_vnni ? 4 : (kcaps.avx2 ? 1 : 0));
        kern = best;
        if (const char* k = std::getenv("STRATA_CPUEXP_KERNEL")) {
            int want = -1;
            if (!std::strcmp(k, "lib")) want = 0;
            else if (!std::strcmp(k, "avx2")) want = 1;
            else if (!std::strcmp(k, "avx2-vnni") || !std::strcmp(k, "avxvnni")) want = 4;
            else if (!std::strcmp(k, "avx512")) want = 2;
            else std::fprintf(stderr, "[cpuexp] STRATA_CPUEXP_KERNEL=%s is not avx512, avx2-vnni, avx2 or lib; ignored\n", k);
            if (want >= 0) {
                const bool ok_want = (want == 0) || (want == 1 && kcaps.avx2) || (want == 4 && kcaps.avx_vnni) || (want == 2 && kcaps.avx512);
                if (ok_want) kern = want;
                else std::fprintf(stderr, "[cpuexp] STRATA_CPUEXP_KERNEL=%s is not supported by this CPU/OS; using %s\n", k, kern_name(best));
            }
        }
    }
#elif HY4_NEON_KERNELS
    {
        const KernCaps kcaps = probe_kernel_caps();
        const int best = kcaps.neon_dp ? 3 : 0;
        kern = best;
        if (const char* k = std::getenv("STRATA_CPUEXP_KERNEL")) {
            int want = -1;
            if (!std::strcmp(k, "lib")) want = 0;
            else if (!std::strcmp(k, "neon")) want = 3;
            else std::fprintf(stderr, "[cpuexp] STRATA_CPUEXP_KERNEL=%s is not neon or lib; ignored\n", k);
            if (want >= 0) {
                if (want == 0 || kcaps.neon_dp) kern = want;
                else std::fprintf(stderr, "[cpuexp] STRATA_CPUEXP_KERNEL=%s is not supported by this CPU/OS; using %s\n", k, kern_name(best));
            }
        }
    }
#endif
    fast = kern != 0;
    hi = (kern == 1);  // plain AVX2 is DRAM-bound so high-precision residual costs ~1%; AVX2-VNNI/AVX-512/NEON default to q8 for 2x throughput (STRATA_CPUEXP_ACT=hi overrides)
    if (const char* a = std::getenv("STRATA_CPUEXP_ACT")) {
        if (!std::strcmp(a, "hi")) hi = true;
        else if (!std::strcmp(a, "q8")) hi = false;
    }
    std::fprintf(stderr, "[cpuexp] %llu experts resident, %.2f GB in %.1f s (%.2f GB/s), %zu reads, mlock=%s, isa=%s, kernel=%s, act=%s\n",
                 (unsigned long long)n_resident, arena_size / 1e9, t_load, arena_size / 1e9 / std::max(t_load, 1e-9), runs.size(),
                 locked ? "yes" : "no", strata::quant::sk::active_sk_isa(), kern_name(kern), hi ? "hi(2xQ8_1)" : "q8");
    start_threads(n_threads);
    return true;
}

// ==================================================================================================== pool
namespace {
static inline int total_logical_cpus() {
#ifdef _WIN32
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n > 0) return (int)n;
#endif
    return (int)std::max(1u, std::thread::hardware_concurrency());
}

static inline void pin_native_thread(std::thread::native_handle_type h, const CpuTarget& t) {
#ifdef _WIN32
    GROUP_AFFINITY ga{};
    ga.Group = t.group;
    ga.Mask = (KAFFINITY)1ull << t.cpu;
    SetThreadGroupAffinity((HANDLE)h, &ga, nullptr);
#else
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (t.cpu >= 0 && t.cpu < CPU_SETSIZE) {
        CPU_SET(t.cpu, &cs);
        pthread_setaffinity_np(h, sizeof(cs), &cs);
    }
#endif
}

void pin_current_thread(const CpuTarget& t) {
#ifdef _WIN32
    pin_native_thread((std::thread::native_handle_type)GetCurrentThread(), t);
#else
    pin_native_thread(pthread_self(), t);
#endif
}

// physical-core-first list of logical CPU targets to pin to, spread round-robin across L3/CCD/NUMA domains
// so multi-CCD / multi-socket CPUs (EPYC / Threadripper / dual Xeon, including Windows >64 logical CPUs) engage all memory channels.
std::vector<CpuTarget> pin_list(int want) {
    const char* pe = std::getenv("STRATA_CPUEXP_PIN");
    const bool linear = pe && !std::strcmp(pe, "linear");
    std::vector<std::vector<CpuTarget>> prim_g, sec_g;
#ifdef _WIN32
    auto get_ex = [](LOGICAL_PROCESSOR_RELATIONSHIP rel, std::vector<uint8_t>& buf) -> bool {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(rel, nullptr, &len);
        if (!len) return false;
        buf.resize(len);
        return GetLogicalProcessorInformationEx(rel, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len) != FALSE;
    };
    struct L3Dom { uint16_t group; KAFFINITY mask; };
    std::vector<L3Dom> l3_doms;
    std::vector<uint8_t> cbuf, pbuf;
    if (!linear && get_ex(RelationCache, cbuf)) {
        size_t off = 0;
        while (off < cbuf.size()) {
            auto* ex = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(cbuf.data() + off);
            if (ex->Size == 0) break;
            if (ex->Relationship == RelationCache && ex->Cache.Level == 3) {
                l3_doms.push_back({ex->Cache.GroupMask.Group, ex->Cache.GroupMask.Mask});
            }
            off += ex->Size;
        }
    }
    if (get_ex(RelationProcessorCore, pbuf)) {
        const size_t ng = std::max<size_t>(1, l3_doms.size());
        prim_g.resize(ng);
        sec_g.resize(ng);
        size_t off = 0;
        while (off < pbuf.size()) {
            auto* ex = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(pbuf.data() + off);
            if (ex->Size == 0) break;
            if (ex->Relationship == RelationProcessorCore) {
                const uint16_t gc = std::max<uint16_t>(1, ex->Processor.GroupCount);
                for (uint16_t gi = 0; gi < gc; ++gi) {
                    const GROUP_AFFINITY& gm = ex->Processor.GroupMask[gi];
                    size_t g = 0;
                    for (size_t k = 0; k < l3_doms.size(); ++k) {
                        if (l3_doms[k].group == gm.Group && (l3_doms[k].mask & gm.Mask)) { g = k; break; }
                    }
                    bool first = true;
                    for (int b = 0; b < (int)(8 * sizeof(KAFFINITY)); ++b) {
                        if ((gm.Mask >> b) & 1ull) {
                            (first ? prim_g[g] : sec_g[g]).push_back({gm.Group, b});
                            first = false;
                        }
                    }
                }
            }
            off += ex->Size;
        }
    }
#else
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return {};
    std::unordered_map<std::string, size_t> dom_id;
    auto domain_of = [&](int c) -> size_t {
        if (linear) {
            if (prim_g.empty()) { prim_g.emplace_back(); sec_g.emplace_back(); }
            return 0;
        }
        char fn[160], buf[128] = {0};
        std::string key = "0";
        std::snprintf(fn, sizeof(fn), "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", c);
        if (std::FILE* f = std::fopen(fn, "r")) {
            if (std::fgets(buf, sizeof(buf), f)) key = buf;
            std::fclose(f);
        } else {
            std::snprintf(fn, sizeof(fn), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
            if (std::FILE* f = std::fopen(fn, "r")) {
                if (std::fgets(buf, sizeof(buf), f)) key = buf;
                std::fclose(f);
            }
        }
        auto it = dom_id.find(key);
        if (it != dom_id.end()) return it->second;
        const size_t id = dom_id.size();
        dom_id[key] = id;
        prim_g.emplace_back();
        sec_g.emplace_back();
        return id;
    };
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &cs)) continue;
        int first = c;
        char fn[128];
        std::snprintf(fn, sizeof(fn), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        if (std::FILE* f = std::fopen(fn, "r")) {
            int a = -1;
            if (std::fscanf(f, "%d", &a) == 1) first = a;
            std::fclose(f);
        }
        const size_t g = domain_of(c);
        (first == c ? prim_g[g] : sec_g[g]).push_back({0, c});
    }
#endif
    std::vector<CpuTarget> all;
    auto spread = [&](const std::vector<std::vector<CpuTarget>>& groups) {
        if (linear) {
            for (const auto& v : groups) all.insert(all.end(), v.begin(), v.end());
            return;
        }
        size_t mx = 0;
        for (const auto& v : groups) mx = std::max(mx, v.size());
        for (size_t r = 0; r < mx; ++r)
            for (const auto& v : groups)
                if (r < v.size()) all.push_back(v[r]);
    };
    spread(prim_g);
    spread(sec_g);
    if (all.empty() || want > (int)all.size()) return {};
    return all;
}
// Linux cgroup CPU quota in cores (> 0 when limited, else 0.0)
static double cgroup_cpu_limit() {
#ifdef _WIN32
    return 0.0;
#else
    double lim = 0.0;
    if (std::FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char qbuf[64] = {0};
        long long period = 0;
        if (std::fscanf(f, "%63s %lld", qbuf, &period) == 2 && std::strcmp(qbuf, "max") != 0 && period > 0) {
            const long long quota = std::atoll(qbuf);
            if (quota > 0) lim = (double)quota / (double)period;
        }
        std::fclose(f);
    }
    if (lim <= 0.0) {
        long long quota = -1, period = 0;
        if (std::FILE* fq = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
            if (std::fscanf(fq, "%lld", &quota) != 1) quota = -1;
            std::fclose(fq);
        }
        if (std::FILE* fp = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(fp, "%lld", &period) != 1) period = 0;
            std::fclose(fp);
        }
        if (quota > 0 && period > 0) lim = (double)quota / (double)period;
    }
    return lim;
#endif
}
}  // namespace

void CpuExperts::Impl::start_threads(int n) {
    nth = std::max(1, n);
    bar.n = nth;
    ts.reset(new TScratch[nth]);
    gdone.reset(new PadCnt[E]);
    dn_cc.reset(new PadCnt[NUNIT]);
    if (const char* e = std::getenv("STRATA_CPUEXP_SCHED")) sched = std::atoi(e);
    else if (nth >= 8) sched = 1;  // dataflow scheduler avoids phase barriers on multi-core CPUs
    if (const char* e = std::getenv("STRATA_CPUEXP_PF")) g_pf_rows = std::atoi(e);
    else if (fast) g_pf_rows = 2;  // prefetch 2 weight rows ahead to cross 4 KiB page boundaries on fast kernels
    for (int ti = 0; ti < nth; ++ti) {
        TScratch& s = ts[ti];
        if (!fast) { s.xg.alloc((size_t)MG * NB); s.xg2.alloc((size_t)MG * NB); }
        s.tg2.alloc((size_t)MGF * RB);
        s.tg.alloc((size_t)MGF * RB);
        s.tu.alloc((size_t)MGF * RB);
    }
    if (const char* e = std::getenv("STRATA_CPUEXP_SPIN_US")) spin_us = std::atoi(e);
    if (const char* e = std::getenv("STRATA_L3_PF")) l3_pf_enabled = (e[0] != '0');
    std::vector<CpuTarget> pins;
    const char* pe = std::getenv("STRATA_CPUEXP_PIN");
    const int host_cpus = total_logical_cpus();
    const double cg_cpus = cgroup_cpu_limit();
    const bool cg_constrained = (cg_cpus > 0.0 && cg_cpus < (double)host_cpus);
    // In a CFS-quota container (cgroup_cpus < host_cpus), rigid pinning hurts scheduling across unthrottled cores; skip unless explicitly enabled.
    if (pe ? (pe[0] != '0') : !cg_constrained) pins = pin_list(nth);
    cnt.assign(E, 0);
    aoff.assign(E + 1, 0);
    act_e.assign(E, 0);
    act_o.assign(E, 0);
    act_n.assign(E, 0);
    act_body.assign(E, nullptr);
    act_slot.assign(E, -1);
    threads.reserve(nth);
    for (int i = 0; i < nth; ++i) {
        threads.emplace_back([this, i]() { worker(i); });
        if (!pins.empty()) pin_native_thread(threads.back().native_handle(), pins[i]);
    }
    const char* pin_note = !pins.empty() ? " (pinned, physical cores first)" : (cg_constrained ? " (not pinned: cgroup CPU quota)" : " (not pinned)");
    std::fprintf(stderr, "[cpuexp] %d pool threads%s, spin %d us, sched=%d, pf=%d rows, l3_pf=%s\n",
                 nth, pin_note, spin_us, sched, g_pf_rows, l3_pf_enabled ? "on" : "off");
}

void CpuExperts::Impl::do_l3_prefetch(int, uint64_t seen) {
    const int layer = pf_layer;
    const int n = pf_n;
    if (layer < 0 || layer >= nlayers || n <= 0) return;
    const Rec* lr = &recs[(size_t)layer * E];
    // Prefetch 16 KiB stripes per claimed unit using T1 (L2/L3 cache) so we warm L3 without thrashing L1,
    // checking gen != seen every stripe so a real compute job preempts immediately (<1 us).
    constexpr size_t STRIPE = 16384;
    for (;;) {
        if (gen.load(std::memory_order_acquire) != seen || stop.load(std::memory_order_relaxed)) return;
        const int u = pf_cursor.fetch_add(1, std::memory_order_relaxed);
        int rem = u;
        const uint8_t* ptr = nullptr;
        size_t len = 0;
        for (int i = 0; i < n; ++i) {
            const Rec& r = lr[pf_experts[i]];
            if (!r.body || !r.body_bytes) continue;
            const int nstripes = (int)((r.body_bytes + STRIPE - 1) / STRIPE);
            if (rem < nstripes) {
                const size_t off = (size_t)rem * STRIPE;
                ptr = r.body + off;
                len = std::min(STRIPE, (size_t)r.body_bytes - off);
                break;
            }
            rem -= nstripes;
        }
        if (!ptr) return;
#if defined(__x86_64__) || defined(_M_X64)
        for (size_t o = 0; o < len; o += 64) _mm_prefetch((const char*)(ptr + o), _MM_HINT_T1);
#elif defined(__GNUC__) || defined(__clang__)
        for (size_t o = 0; o < len; o += 64) __builtin_prefetch((const char*)(ptr + o), 0, 2);
#endif
    }
}

void CpuExperts::Impl::worker(int tid) {
    uint64_t seen = 0;
    uint64_t pf_seen = 0;
    for (;;) {
        const double t_spin = now_s();
        uint32_t it = 0;
        while (gen.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_relaxed)) {
            if (l3_pf_enabled) {
                const uint64_t pg = pf_gen.load(std::memory_order_acquire);
                if (pg != pf_seen) {
                    pf_seen = pg;
                    do_l3_prefetch(tid, seen);
                    continue;
                }
            }
            HY4_RELAX();
            if ((++it & 255) == 0 && (now_s() - t_spin) * 1e6 > spin_us) break;
        }
        if (gen.load(std::memory_order_acquire) == seen && !stop.load()) {
            std::unique_lock<std::mutex> lk(m);
            cv_work.wait(lk, [&] { return gen.load(std::memory_order_acquire) != seen || stop.load(); });
        }
        if (stop.load()) return;
        seen = gen.load(std::memory_order_acquire);
        run_job(tid);
    }
}

// thread 0: group the CPU-resident (token, slot) pairs of this chunk by expert (stable, so slots stay in ascending order)
void CpuExperts::Impl::plan(const Job& J, int t0, int Tc) {
    const Rec* lr = &recs[(size_t)J.layer * E];
    std::fill(cnt.begin(), cnt.end(), 0);
    const size_t P = (size_t)Tc * TOPK;
    uint64_t skipped = 0;
    std::fill(tok_used.begin(), tok_used.begin() + Tc, 0);
    for (int t = 0; t < Tc; ++t) {
        for (int s = 0; s < TOPK; ++s) {
            const size_t q = (size_t)(t0 + t) * TOPK + s;
            const bool want = J.use ? J.use[q] != 0 : true;
            if (!want) continue;
            const int32_t e = J.idx[q];
            if (e < 0 || e >= E || !(lr[e].body || lr[e].cold)) { ++skipped; continue; }
            ++cnt[e];
            tok_used[t] = 1;
        }
    }
    nact = 0;
    int off = 0;
    uint64_t bytes = 0;
    for (int e = 0; e < E; ++e) {
        aoff[e] = off;
        if (cnt[e] > 0) {
            act_e[nact] = e;
            act_o[nact] = off;
            act_n[nact] = cnt[e];
            act_body[nact] = lr[e].body;
            act_slot[nact] = -1;
            ++nact;
            bytes += lr[e].body_bytes;
        }
        off += cnt[e];
    }
    aoff[E] = off;
    cold_job = false;
    if (cs) {
        // cold experts: wave 0 = resident experts + cold cache hits, then the cold misses in waves limited by the slot cache
        bool any_cold = false;
        for (int a = 0; a < nact; ++a)
            if (!lr[act_e[a]].body) { any_cold = true; break; }
        if (any_cold) {
            std::vector<int> ea, oa, na, mi;
            std::vector<const uint8_t*> ba;
            std::vector<int> sa;
            // Compute pe128 prompt-end recency weights when Tc > 1 so Order-A waves keep prompt-end experts resident!
            std::vector<float> w_pe;
            if (Tc > 1) {
                w_pe = strata::core::compute_pe128_scores(J.idx + (size_t)t0 * TOPK, Tc, TOPK, E);
            }
            {
                std::lock_guard<std::mutex> lk(cs->slot_mu);
                cs->sched.drop_stale(J.layer, cs->gen);
                for (int a = 0; a < nact; ++a) {
                    const int e = act_e[a];
                    const Rec& r = lr[e];
                    if (r.body) {
                        ea.push_back(e); oa.push_back(act_o[a]); na.push_back(act_n[a]); ba.push_back(r.body); sa.push_back(-1);
                        continue;
                    }
                    auto it = cs->where.find((int64_t)J.layer * E + e);
                    if (it != cs->where.end()) {  // cache hit (or in-flight prefetch promoted to demand!)
                        const int s = it->second;
                        cs->slots[s].pin++;
                        cs->slots[s].tick = ++cs->tick;
                        cs->sched.promote_to_demand((int64_t)J.layer * E + e, s, cs->mem + (size_t)s * cs->slot_bytes);
                        ea.push_back(e); oa.push_back(act_o[a]); na.push_back(act_n[a]); ba.push_back(cs->mem + (size_t)s * cs->slot_bytes); sa.push_back(s);
                        ++st.cold_hits;
                    } else {
                        mi.push_back(a);
                    }
                }
            }
            // Order-A pe128 miss ordering: coldest misses stream first, hottest prompt-end misses stream LAST!
            if (Tc > 1 && mi.size() > 1) {
                std::sort(mi.begin(), mi.end(), [&](int a, int b) {
                    const float wa = w_pe[act_e[a]], wb = w_pe[act_e[b]];
                    return (wa != wb) ? (wa < wb) : (act_e[a] < act_e[b]);
                });
            }
            nA = (int)ea.size();
            for (int a : mi) {
                ea.push_back(act_e[a]); oa.push_back(act_o[a]); na.push_back(act_n[a]); ba.push_back(nullptr); sa.push_back(-1);
            }
            for (int a = 0; a < nact; ++a) {
                act_e[a] = ea[a]; act_o[a] = oa[a]; act_n[a] = na[a]; act_body[a] = ba[a]; act_slot[a] = sa[a];
            }
            cold_job = true;
            wave_finished = false;
            cur_a0 = cur_a1 = 0;
            next_miss = nA;
            nxt_a0 = nxt_a1 = nA;
            batch_open = false;
            cold_lr = lr;
            cold_layer = J.layer;
            cold_issue();  // start reading the first cold wave now: overlaps the activation quantization and wave 0
        }
    }
    // fill sorted pair table
    std::vector<int>& fill = cnt;  // reuse: becomes the running write position
    for (int e = 0; e < E; ++e) fill[e] = aoff[e];
    for (size_t q = 0; q < P; ++q) posof[q] = -1;
    for (int t = 0; t < Tc; ++t)
        for (int s = 0; s < TOPK; ++s) {
            const size_t q = (size_t)(t0 + t) * TOPK + s;
            const bool want = J.use ? J.use[q] != 0 : true;
            if (!want) continue;
            const int32_t e = J.idx[q];
            if (e < 0 || e >= E || !(lr[e].body || lr[e].cold)) continue;
            const int p = fill[e]++;
            pair_tok[p] = t;
            posof[(size_t)t * TOPK + s] = p;
        }
    qctr.v.store(0);
    guctr.v.store(0);
    dnctr.v.store(0);
    cbctr.v.store(0);
    fctr.v.store(0);
    for (int a = 0; a < nact; ++a) gdone[a].v.store(0, std::memory_order_relaxed);
    for (int c = 0; c < NUNIT; ++c) dn_cc[c].v.store(0, std::memory_order_relaxed);
    st.bytes += bytes;
    st.pairs += (uint64_t)off;
    st.skipped_unresident += skipped;
}

// one gate/up unit: output indices [b*64, b*64+64) of h for all tokens of active expert a -> h2 blocks
void CpuExperts::Impl::gu_unit(int tid, const Job& J, int a, int b) {
    TScratch& S = ts[tid];
    const Rec& r = recs[(size_t)J.layer * E + act_e[a]];
    const int p0 = act_o[a], m = act_n[a];
    const SkCode gc = (SkCode)r.gu;
    const size_t row_b = 6ull * r.gu_sb;
    const uint8_t* body = act_body[a];  // arena pointer or cold slot
    const uint8_t* wgu = body + 3072;
    const uint8_t* wg = wgu + (size_t)b * RB * row_b;
    const uint8_t* wu = wgu + (size_t)(D + b * RB) * row_b;
    float sq[RB];
    {
        const uint16_t* h16 = (const uint16_t*)body + b * RB;
        for (int i = 0; i < RB; ++i) sq[i] = fp16_to_fp32(h16[i]);
    }
    const int mgmax = fast ? MGF : MG;
    for (int g0 = 0; g0 < m; g0 += mgmax) {
        const int mg = std::min(mgmax, m - g0);
#if HY4_ANY_FAST
        if (fast) {
            const ActRow* xp[MGF];
#ifdef _MSC_VER
#pragma loop(no_vector)
#endif
            for (int i = 0; i < mg; ++i) xp[i] = tokA.p + (size_t)pair_tok[p0 + g0 + i];
            kern_gemm_dispatch(kern, r.gu, wg, RB, xp, mg, S.tg.p, J.hi);
            kern_gemm_dispatch(kern, r.gu, wu, RB, xp, mg, S.tu.p, J.hi);
        } else
#endif
        {
            const BlockQ8_1_G64* xs;
            if (mg == 1) {
                xs = tokq.p + (size_t)pair_tok[p0 + g0] * NB;
            } else {
                for (int i = 0; i < mg; ++i) std::memcpy(S.xg.p + (size_t)i * NB, tokq.p + (size_t)pair_tok[p0 + g0 + i] * NB, NB * sizeof(BlockQ8_1_G64));
                xs = S.xg.p;
            }
            gemm_sk_q8_1(wg, gc, xs, S.tg.p, RB, D, mg);
            gemm_sk_q8_1(wu, gc, xs, S.tu.p, RB, D, mg);
            if (J.hi) {  // residual term of the activations
                const BlockQ8_1_G64* xs2;
                if (mg == 1) {
                    xs2 = tokq2.p + (size_t)pair_tok[p0 + g0] * NB;
                } else {
                    for (int i = 0; i < mg; ++i) std::memcpy(S.xg2.p + (size_t)i * NB, tokq2.p + (size_t)pair_tok[p0 + g0 + i] * NB, NB * sizeof(BlockQ8_1_G64));
                    xs2 = S.xg2.p;
                }
                gemm_sk_q8_1(wg, gc, xs2, S.tg2.p, RB, D, mg);
                for (int i = 0; i < mg * RB; ++i) S.tg.p[i] += S.tg2.p[i];
                gemm_sk_q8_1(wu, gc, xs2, S.tg2.p, RB, D, mg);
                for (int i = 0; i < mg * RB; ++i) S.tu.p[i] += S.tg2.p[i];
            }
        }
        for (int i = 0; i < mg; ++i) {
            float h[RB];
            const float* g = S.tg.p + i * RB;
            const float* u = S.tu.p + i * RB;
#if HY4_KERNELS
            if (fast && (kern == 1 || kern == 2 || kern == 4)) {
                swiglu_64_avx2(g, u, sq, h);
            } else
#endif
            for (int d = 0; d < RB; ++d) {
                const float gv = g[d];
                const float si = gv / (1.0f + fexpf(-gv));
                const float sc = std::min(si, SWIGLU_LIMIT);
                const float uc = std::max(-SWIGLU_LIMIT, std::min(u[d], SWIGLU_LIMIT));
                h[d] = (sc * uc) / sq[d];
            }
#if HY4_ANY_FAST
            if (fast) {
                BlockQ8_1_G64 blk;
                quantize_q8_1_g64(h, &blk, 64);
                if (J.hi) {
                    BlockQ8_1_G64 blk2;
                    residual_blocks(h, &blk, &blk2, 1);
                    kern_store_hi(kern, h2A.p + (size_t)(p0 + g0 + i), b, blk, blk2);
                } else {
                    kern_store(kern, h2A.p + (size_t)(p0 + g0 + i), b, blk);
                }
            } else
#endif
            {
                BlockQ8_1_G64* hb = h2q.p + (size_t)(p0 + g0 + i) * NB + b;
                quantize_q8_1_g64(h, hb, 64);
                if (J.hi) residual_blocks(h, hb, h2q2.p + (size_t)(p0 + g0 + i) * NB + b, 1);
            }
        }
    }
}

// one down unit: output rows [c*64, c*64+64) of y for all tokens of active expert a
void CpuExperts::Impl::dn_unit(int tid, const Job& J, int a, int c) {
    TScratch& S = ts[tid];
    const Rec& r = recs[(size_t)J.layer * E + act_e[a]];
    const int p0 = act_o[a], m = act_n[a];
    const size_t row_b = 6ull * r.dn_sb;
    const uint8_t* wd = act_body[a] + r.dn_off + (size_t)c * RB * row_b;
    const int mgmax = fast ? MGF : MG;
    for (int g0 = 0; g0 < m; g0 += mgmax) {
        const int mg = std::min(mgmax, m - g0);
#if HY4_ANY_FAST
        if (fast) {
            const ActRow* xp[MGF];
            const ActRow* base = h2A.p + (size_t)(p0 + g0);
#ifdef _MSC_VER
#pragma loop(no_vector)
#endif
            for (int i = 0; i < mg; ++i) xp[i] = base + i;
            kern_gemm_dispatch(kern, r.dn, wd, RB, xp, mg, S.tg.p, J.hi);
        } else
#endif
        {
            gemm_sk_q8_1(wd, (SkCode)r.dn, h2q.p + (size_t)(p0 + g0) * NB, S.tg.p, RB, D, mg);
            if (J.hi) {
                gemm_sk_q8_1(wd, (SkCode)r.dn, h2q2.p + (size_t)(p0 + g0) * NB, S.tg2.p, RB, D, mg);
                for (int i = 0; i < mg * RB; ++i) S.tg.p[i] += S.tg2.p[i];
            }
        }
        for (int i = 0; i < mg; ++i) std::memcpy(Y.p + (size_t)(p0 + g0 + i) * D + c * RB, S.tg.p + i * RB, RB * sizeof(float));
    }
}

// zout[t][c0..c0+256) (+)= sum over slots (ascending) w * y
void CpuExperts::Impl::combine_unit(const Job& J, int t0, int tl, int c0) {
    constexpr int CW = 256;
    float acc[CW];
    for (int i = 0; i < CW; ++i) acc[i] = 0.0f;
    for (int s = 0; s < TOPK; ++s) {
        const int p = posof[(size_t)tl * TOPK + s];
        if (p < 0) continue;
        const float wt = J.w[(size_t)(t0 + tl) * TOPK + s];
        const float* y = Y.p + (size_t)p * D + c0;
        for (int i = 0; i < CW; ++i) acc[i] += wt * y[i];
    }
    float* zo = J.zout + (size_t)(t0 + tl) * D + c0;
    if (J.accum) for (int i = 0; i < CW; ++i) zo[i] += acc[i];
    else for (int i = 0; i < CW; ++i) zo[i] = acc[i];
}

// zout[t][c*64 .. c*64+64) (+)= sum over slots (ascending) w * y, for the Tc tokens of the chunk (same per-element arithmetic as combine_unit)
void CpuExperts::Impl::combine_cols(const Job& J, int t0, int Tc, int c) {
    constexpr int CW = RB;
    for (int tl = 0; tl < Tc; ++tl) {
        float acc[CW];
        for (int i = 0; i < CW; ++i) acc[i] = 0.0f;
        for (int s = 0; s < TOPK; ++s) {
            const int p = posof[(size_t)tl * TOPK + s];
            if (p < 0) continue;
            const float wt = J.w[(size_t)(t0 + tl) * TOPK + s];
            const float* y = Y.p + (size_t)p * D + c * CW;
            for (int i = 0; i < CW; ++i) acc[i] += wt * y[i];
        }
        float* zo = J.zout + (size_t)(t0 + tl) * D + c * CW;
        if (J.accum) for (int i = 0; i < CW; ++i) zo[i] += acc[i];
        else for (int i = 0; i < CW; ++i) zo[i] = acc[i];
    }
}

void CpuExperts::Impl::flow_phase(int tid, const Job& J, int t0, int Tc) {
    if (nact == 0) {  // nothing to compute: still (over)write zout with zeros / leave it unchanged
        for (int c = tid; c < NUNIT; c += nth) combine_cols(J, t0, Tc, c);
        return;
    }
    const int G = nact * NUNIT, total = 2 * G;
    for (;;) {
        const int u = fctr.v.fetch_add(1, std::memory_order_relaxed);
        if (u >= total) break;
        if (u < G) {
            const int a = u / NUNIT;
            gu_unit(tid, J, a, u % NUNIT);
            gdone[a].v.fetch_add(1, std::memory_order_release);
        } else {
            const int v = u - G, a = v / NUNIT, c = v % NUNIT;
            uint32_t spins = 0;
            while (gdone[a].v.load(std::memory_order_acquire) < NUNIT) {
                HY4_RELAX();
                if ((++spins & 0xFFFF) == 0) std::this_thread::yield();
            }
            dn_unit(tid, J, a, c);
            if (dn_cc[c].v.fetch_add(1, std::memory_order_acq_rel) == nact - 1) combine_cols(J, t0, Tc, c);
        }
    }
}

// ---- cold waves (thread 0 only) -------------------------------------------------------------------------------------------
// Start reading the next wave of cold misses (at most nslots/2, only into unpinned slots, LRU first). Never called while reads are pending.
void CpuExperts::Impl::cold_issue() {
    nxt_a0 = nxt_a1 = next_miss;
    if (next_miss >= nact || cs->err.load()) return;
    const int B = std::max(1, cs->nslots / 2);
    int issued = 0;
    uint64_t bytes = 0;
    while (next_miss < nact && issued < B) {
        int best = -1;
        {
            std::lock_guard<std::mutex> lk(cs->slot_mu);
            uint64_t bt = ~0ull;
            for (int s = 0; s < cs->nslots; ++s)
                if (cs->slots[s].pin == 0 && cs->slots[s].io_active == 0 && cs->slots[s].tick < bt) { bt = cs->slots[s].tick; best = s; }
            if (best < 0) break;  // everything pinned by the wave being computed
            const int a = next_miss;
            ColdStore::Slot& sl = cs->slots[best];
            if (sl.key >= 0) cs->where.erase(sl.key);
            sl.key = (int64_t)cold_layer * E + act_e[a];
            sl.tick = ++cs->tick;
            sl.pin = 1;
            sl.io_active++;
            if (auto* stp = cs->sched.slot_state(best)) stp->reset_for_load(sl.key);
            cs->where[sl.key] = best;
            act_slot[a] = best;
            act_body[a] = cs->mem + (size_t)best * cs->slot_bytes;
        }
        const int a = next_miss;
        const Rec& r = cold_lr[act_e[a]];
        if (issued == 0 && !batch_open) {
            batch_start_ns = ColdStore::now_ns();
            batch_open = true;
        }
        cs->enqueue_demand((int64_t)cold_layer * E + act_e[a], cold_layer, best,
                           const_cast<uint8_t*>(act_body[a]), r.file_off, (size_t)r.body_bytes, (size_t)r.dn_off);
        bytes += r.body_bytes;
        ++next_miss;
        ++issued;
    }
    nxt_a1 = next_miss;
    st.cold_reads += (uint64_t)issued;
    st.cold_bytes += bytes;
}

void CpuExperts::Impl::cold_wait() {
    const double t0 = now_s();
    // Wait only for gate+up readiness on the wave being promoted so gu_unit can overlap with down NVMe reads!
    for (int a = nxt_a0; a < nxt_a1; ++a) {
        const int sl = act_slot[a];
        if (sl >= 0) {
            if (auto* stp = cs->sched.slot_state(sl)) stp->wait_gate_up();
        }
    }
    st.cold_wait_s += now_s() - t0;
    if (batch_open) {
        const int64_t end = cs->last_done_ns.load();
        if (end > batch_start_ns) st.cold_io_s += (end - batch_start_ns) * 1e-9;
        batch_open = false;
    }
    if (cs->err.load()) {
        st.cold_errors += 1;
        if (!cold_err_reported) {
            std::fprintf(stderr, "[cpuexp] cold: a read from the .strata file failed; results of this job are invalid\n");
            cold_err_reported = true;
        }
        std::lock_guard<std::mutex> lk(cs->slot_mu);
        for (int a = nxt_a0; a < nxt_a1; ++a)
            if (act_slot[a] >= 0) { cs->where.erase(cs->slots[act_slot[a]].key); cs->slots[act_slot[a]].key = -1; }
        cs->err.store(false);
    }
}

// Prepare wave w: wave 0 = resident + cache hits; wave w > 0 = the cold experts read meanwhile (wait for gate+up, then start reading the next wave)
void CpuExperts::Impl::cold_advance(int w) {
    guctr.v.store(0);
    dnctr.v.store(0);
    if (w == 0) {
        cur_a0 = 0;
        cur_a1 = nA;
        wave_finished = false;
        return;
    }
    {
        std::lock_guard<std::mutex> lk(cs->slot_mu);
        for (int a = cur_a0; a < cur_a1; ++a)  // wave w-1 is done: release its slots
            if (act_slot[a] >= 0) cs->slots[act_slot[a]].pin--;
    }
    if (nxt_a1 == nxt_a0) {  // nothing in flight
        if (next_miss >= nact) {
            cur_a0 = cur_a1 = nact;
            wave_finished = true;
            return;
        }
        cold_issue();  // slots are free now (not overlapped with compute: all slots were pinned before)
    }
    cold_wait();
    cur_a0 = nxt_a0;
    cur_a1 = nxt_a1;
    // Wait for prior wave's down projections ONLY if we need to re-issue; current wave's down overlaps gu_unit!
    cold_issue();  // the wave after this one is read while this one is computed
}

void CpuExperts::Impl::run_job(int tid) {
    const Job J = job;  // copy: the submitter may not touch it, but be safe
    int& sense = ts[tid].sense;
    for (int t0 = 0; t0 < J.T; t0 += TC) {
        const int Tc = std::min(TC, J.T - t0);
        double tp = tid == 0 ? now_s() : 0, tn = 0;
        auto lap = [&](int k) {
            if (tid != 0) return;
            tn = now_s();
            st.phase_s[k] += tn - tp;
            tp = tn;
        };
        if (tid == 0) plan(J, t0, Tc);
        bar.wait(sense);
        lap(0);
        // quantize the z rows of tokens that have a CPU pair (8 blocks of 64 per unit)
        for (;;) {
            const int u = qctr.v.fetch_add(1, std::memory_order_relaxed);
            if (u >= Tc * 3) break;
            const int t = u / 3, part = u % 3;
            if (!tok_used[t]) continue;
#if HY4_ANY_FAST
            if (fast) {
                alignas(64) BlockQ8_1_G64 blk[8];
                const float* zs = J.z + (size_t)(t0 + t) * D + part * 512;
                quantize_q8_1_g64(zs, blk, 512);
                if (J.hi) {
                    alignas(64) BlockQ8_1_G64 blk2[8];
                    residual_blocks(zs, blk, blk2, 8);
                    for (int i = 0; i < 8; ++i) kern_store_hi(kern, tokA.p + t, part * 8 + i, blk[i], blk2[i]);
                } else {
                    for (int i = 0; i < 8; ++i) kern_store(kern, tokA.p + t, part * 8 + i, blk[i]);
                }
                continue;
            }
#endif
            const float* zs = J.z + (size_t)(t0 + t) * D + part * 512;
            quantize_q8_1_g64(zs, tokq.p + (size_t)t * NB + part * 8, 512);
            if (J.hi) residual_blocks(zs, tokq.p + (size_t)t * NB + part * 8, tokq2.p + (size_t)t * NB + part * 8, 8);
        }
        bar.wait(sense);
        lap(1);
        if (cold_job) {
            // waves: [resident + cache hits] then the cold misses as their reads complete; combine once at the end
            for (int w = 0;; ++w) {
                if (tid == 0) cold_advance(w);
                bar.wait(sense);
                if (wave_finished) break;
                const int a0 = cur_a0, na = cur_a1 - cur_a0;
                for (;;) {
                    const int u = guctr.v.fetch_add(1, std::memory_order_relaxed);
                    if (u >= na * NUNIT) break;
                    gu_unit(tid, J, a0 + u / NUNIT, u % NUNIT);
                }
                bar.wait(sense);
                // Ensure each cold expert's down projection finished landing (overlapped with gu_unit above via proj_ready[3]!)
                if (tid == 0) {
                    for (int a = cur_a0; a < cur_a1; ++a) {
                        const int sl = act_slot[a];
                        if (sl >= 0) {
                            if (auto* stp = cs->sched.slot_state(sl)) stp->wait_down();
                        }
                    }
                }
                bar.wait(sense);
                for (;;) {
                    const int u = dnctr.v.fetch_add(1, std::memory_order_relaxed);
                    if (u >= na * NUNIT) break;
                    dn_unit(tid, J, a0 + u / NUNIT, u % NUNIT);
                }
                bar.wait(sense);
            }
            lap(2);
            const int cbc = D / 256;
            for (;;) {
                const int u = cbctr.v.fetch_add(1, std::memory_order_relaxed);
                if (u >= Tc * cbc) break;
                combine_unit(J, t0, u / cbc, (u % cbc) * 256);
            }
            bar.wait(sense);
            lap(4);
            continue;
        }
        if (sched == 1) {
            // dataflow: gate/up units of all experts, then down units (expert-major; a down unit waits only for its own expert's
            // gate/up units), and the last down unit of a column chunk does that chunk's slot-ordered combine. One barrier at the end.
            flow_phase(tid, J, t0, Tc);
            bar.wait(sense);
            lap(2);
            continue;
        }
        for (;;) {
            const int u = guctr.v.fetch_add(1, std::memory_order_relaxed);
            if (u >= nact * NUNIT) break;
            gu_unit(tid, J, u / NUNIT, u % NUNIT);
        }
        bar.wait(sense);
        lap(2);
        for (;;) {
            const int u = dnctr.v.fetch_add(1, std::memory_order_relaxed);
            if (u >= nact * NUNIT) break;
            dn_unit(tid, J, u / NUNIT, u % NUNIT);
        }
        bar.wait(sense);
        lap(3);
        const int cb = D / 256;
        for (;;) {
            const int u = cbctr.v.fetch_add(1, std::memory_order_relaxed);
            if (u >= Tc * cb) break;
            combine_unit(J, t0, u / cb, (u % cb) * 256);
        }
        bar.wait(sense);
        lap(4);
    }
    if (tid == 0) {
        st.wall_s += now_s() - J.t_post;
        st.jobs += 1;
        {
            std::lock_guard<std::mutex> lk(m);
            done_gen.store(gen.load(std::memory_order_relaxed), std::memory_order_release);
        }
        cv_done.notify_all();
    }
}

bool CpuExperts::Impl::set_cold(const std::vector<std::vector<bool>>& cold, std::string& err) {
    cs.reset();  // joins the readers
    for (Rec& r : recs)
        if (r.cold) r = Rec();
    size_t maxsz = 0;
    uint64_t n = 0;
    for (size_t i = 0; i < dir.size(); ++i) {
        const DirE& d = dir[i];
        const int layer = (int)(i / E);
        if (layer >= (int)cold.size() || d.expert_id >= cold[layer].size() || !cold[layer][d.expert_id]) continue;
        Rec& r = recs[(size_t)layer * E + d.expert_id];
        if (r.body) { err = "cold expert is also resident: layer " + std::to_string(layer) + " expert " + std::to_string(d.expert_id); return false; }
        const uint32_t gsb = (uint32_t)superblock_bytes((SkCode)d.gu), dsb = (uint32_t)superblock_bytes((SkCode)d.dn);
        if (!gsb || !dsb || d.dn_off != 3072u + 18432u * gsb || (uint64_t)d.dn_off + 9216ull * dsb > d.body_bytes) { err = "unsupported or inconsistent expert body layout (cold)"; return false; }
        if ((s4_off + d.body_off) % 4096) { err = "cold expert body not 4 KiB aligned in file"; return false; }
        r.cold = true;
        r.file_off = s4_off + d.body_off;
        r.dn_off = d.dn_off;
        r.body_bytes = d.body_bytes;
        r.gu = d.gu;
        r.dn = d.dn;
        r.gu_sb = gsb;
        r.dn_sb = dsb;
        maxsz = std::max(maxsz, ((size_t)d.body_bytes + 4095) & ~(size_t)4095);
        ++n;
    }
    if (n == 0) return true;
    size_t cache_mb = 256;
    if (const char* e = std::getenv("STRATA_CPUEXP_COLD_CACHE_MB")) cache_mb = (size_t)std::max(1, std::atoi(e));
    int readers = 8, mode = 0;
    if (const char* e = std::getenv("STRATA_CPUEXP_COLD_READERS")) readers = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("STRATA_CPUEXP_COLD_IO")) {
        if (!std::strcmp(e, "buffered")) mode = 1;
        else if (!std::strcmp(e, "fadvise")) mode = 2;
    }
    const int nslots = (int)std::max<size_t>(2, (cache_mb << 20) / maxsz);
    cs.reset(new ColdStore);
    if (!cs->open(path, maxsz, nslots, readers, mode, err)) {
        cs.reset();
        for (Rec& r : recs)
            if (r.cold) r = Rec();
        return false;
    }
#ifdef _WIN32
    const char* iostr = cs->io_mode == 0 ? "NO_BUFFERING+OVERLAPPED" : "buffered+OVERLAPPED";
#else
    const char* iostr = cs->io_mode == 0 ? "O_DIRECT" : (cs->io_mode == 1 ? "buffered" : "buffered+fadvise");
#endif
    std::fprintf(stderr, "[cpuexp] cold experts: %llu (slot %.2f MB, %d slots = %.0f MB cache, %d readers, io=%s)\n", (unsigned long long)n, maxsz / 1048576.0, nslots,
                 nslots * (double)maxsz / 1048576.0, readers, iostr);
    return true;
}

// ==================================================================================================== public API
bool CpuExperts::set_cold_experts(const std::vector<std::vector<bool>>& cold, std::string& err) {
    wait();
    return p_->set_cold(cold, err);
}
bool CpuExperts::cold_expert(int layer, int e) const {
    return layer >= 0 && layer < p_->nlayers && e >= 0 && e < E && p_->recs[(size_t)layer * E + e].cold;
}
uint64_t CpuExperts::ram_budget_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullTotalPhys : 0;
#else
    uint64_t total = 0;
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f))
            if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) { total = kb * 1024ull; break; }
        std::fclose(f);
    }
    const auto cg = strata::platform::cgroup_mem::probe_cgroup_room(total);
    if (cg.limited && cg.available_bytes != strata::platform::cgroup_mem::kUnlimited) {
        return (total == 0) ? cg.available_bytes : std::min(total, cg.available_bytes);
    }
    if (cg.limit_bytes != strata::platform::cgroup_mem::kUnlimited) {
        return (total == 0) ? cg.limit_bytes : std::min(total, cg.limit_bytes);
    }
    return total;
#endif
}

uint64_t CpuExperts::ram_available_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullAvailPhys : 0;
#else
    uint64_t avail = 0, mfree = 0, mbuf = 0, mcache = 0;
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) avail = kb * 1024ull;
            else if (std::sscanf(line, "MemFree: %llu kB", &kb) == 1) mfree = kb * 1024ull;
            else if (std::sscanf(line, "Buffers: %llu kB", &kb) == 1) mbuf = kb * 1024ull;
            else if (std::sscanf(line, "Cached: %llu kB", &kb) == 1) mcache = kb * 1024ull;
        }
        std::fclose(f);
    }
    if (avail == 0) avail = mfree + mbuf + mcache;
    // Cap by cgroup v2 headroom: memory.max - (memory.current - inactive_file)
    if (std::FILE* fm = std::fopen("/sys/fs/cgroup/memory.max", "r")) {
        char buf[64] = {0};
        if (std::fgets(buf, sizeof(buf), fm) && buf[0] >= '0' && buf[0] <= '9') {
            const uint64_t lim = std::strtoull(buf, nullptr, 10);
            uint64_t cur = 0, inact_file = 0;
            if (std::FILE* fc = std::fopen("/sys/fs/cgroup/memory.current", "r")) {
                if (std::fscanf(fc, "%llu", (unsigned long long*)&cur) != 1) cur = 0;
                std::fclose(fc);
            }
            if (std::FILE* fs = std::fopen("/sys/fs/cgroup/memory.stat", "r")) {
                char line[256];
                unsigned long long v = 0;
                while (std::fgets(line, sizeof(line), fs))
                    if (std::sscanf(line, "inactive_file %llu", &v) == 1) { inact_file = v; break; }
                std::fclose(fs);
            }
            const uint64_t used = cur > inact_file ? (cur - inact_file) : 0;
            const uint64_t cg_avail = lim > used ? (lim - used) : 0;
            if (lim > 0 && (avail == 0 || cg_avail < avail)) avail = cg_avail;
        }
        std::fclose(fm);
    }
    // Cap by cgroup v1 headroom if present
    if (std::FILE* fm = std::fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r")) {
        char buf[64] = {0};
        if (std::fgets(buf, sizeof(buf), fm) && buf[0] >= '0' && buf[0] <= '9') {
            const uint64_t lim = std::strtoull(buf, nullptr, 10);
            if (lim > 0 && lim < (1ull << 60)) {
                uint64_t cur = 0, inact_file = 0;
                if (std::FILE* fc = std::fopen("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r")) {
                    if (std::fscanf(fc, "%llu", (unsigned long long*)&cur) != 1) cur = 0;
                    std::fclose(fc);
                }
                if (std::FILE* fs = std::fopen("/sys/fs/cgroup/memory/memory.stat", "r")) {
                    char line[256];
                    unsigned long long v = 0;
                    while (std::fgets(line, sizeof(line), fs))
                        if (std::sscanf(line, "total_inactive_file %llu", &v) == 1) { inact_file = v; break; }
                    std::fclose(fs);
                }
                const uint64_t used = cur > inact_file ? (cur - inact_file) : 0;
                const uint64_t cg_avail = lim > used ? (lim - used) : 0;
                if (avail == 0 || cg_avail < avail) avail = cg_avail;
            }
        }
        std::fclose(fm);
    }
    return avail;
#endif
}

int CpuExperts::default_threads() {
    const int host_cpus = total_logical_cpus();
    const int phys_cap = std::max(1, host_cpus / 2);
    const double cg = cgroup_cpu_limit();
    if (cg > 0.0 && cg < (double)host_cpus) return std::max(1, std::min(phys_cap, (int)std::floor(cg) - 2));
    if (phys_cap >= 28) return phys_cap - 6;  // leave headroom on 28+ core Threadripper/EPYC for GPU submission & DMA threads
    if (phys_cap >= 20) return phys_cap - 2;
    return phys_cap;
}

bool CpuExperts::read_rank_table(const std::string& path, std::vector<uint16_t>& expert_of_rank, std::vector<uint16_t>& codes, std::string& err) {
    std::vector<DirE> dir;
    uint64_t s4o, s4l;
    if (!read_dir(path, dir, s4o, s4l, err)) return false;
    expert_of_rank.assign(dir.size(), 0);
    codes.assign(dir.size(), 0);
    const size_t nl = dir.size() / E;
    for (size_t i = 0; i < dir.size(); ++i) {
        if (dir[i].expert_id >= E) { err = "bad expert_id in directory"; return false; }
        expert_of_rank[i] = dir[i].expert_id;
        codes[(i / E) * E + dir[i].expert_id] = (uint16_t)((dir[i].gu << 8) | dir[i].dn);
    }
    (void)nl;
    return true;
}

bool CpuExperts::read_body_bytes(const std::string& path, std::vector<uint32_t>& body_bytes, std::string& err) {
    std::vector<DirE> dir;
    uint64_t s4o, s4l;
    if (!read_dir(path, dir, s4o, s4l, err)) return false;
    body_bytes.assign(dir.size(), 0);
    for (size_t i = 0; i < dir.size(); ++i) body_bytes[i] = dir[i].body_bytes;
    return true;
}

bool CpuExperts::make_mask_rank_range(const std::string& path, const std::vector<int>& first_rank, const std::vector<int>& last_rank,
                                      std::vector<std::vector<bool>>& mask, std::string& err) {
    std::vector<uint16_t> eor, codes;
    if (!read_rank_table(path, eor, codes, err)) return false;
    const int nl = (int)(eor.size() / E);
    mask.assign(nl, std::vector<bool>(E, false));
    for (int l = 0; l < nl; ++l) {
        const int a = l < (int)first_rank.size() ? std::max(0, first_rank[l]) : 0;
        const int b = l < (int)last_rank.size() ? std::min(E, last_rank[l]) : 0;
        for (int r = a; r < b; ++r) mask[l][eor[(size_t)l * E + r]] = true;
    }
    return true;
}

bool CpuExperts::make_mask_rank_range(const std::string& path, int first_rank, int last_rank, std::vector<std::vector<bool>>& mask, std::string& err) {
    std::vector<uint16_t> eor, codes;
    if (!read_rank_table(path, eor, codes, err)) return false;
    const int nl = (int)(eor.size() / E);
    return make_mask_rank_range(path, std::vector<int>(nl, first_rank), std::vector<int>(nl, last_rank), mask, err);
}

// S1.5 (ternary, 5 trits per byte) runs on the library kernel: check gemm_sk_q8_1 vs dequant_sk_row with valid payload bytes (< 243).
static bool selftest_lib_s15(std::string& report) {
    uint64_t s = 0x7654321ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    const size_t sb = superblock_bytes(SkCode::S1_5), rb = 6 * sb;
    const int nrows = 5;
    std::vector<uint8_t> W(rb * nrows);
    for (int r = 0; r < nrows; ++r)
        for (int b = 0; b < 6; ++b) {
            uint8_t* p = &W[r * rb + b * sb];
            for (size_t i = 0; i < sb; ++i) p[i] = (uint8_t)rnd();
            for (size_t i = 6; i < 6 + 52; ++i) p[i] = (uint8_t)(rnd() % 243);
            const uint16_t h = (uint16_t)(0x2800 + (rnd() % 0x1000));
            std::memcpy(p, &h, 2);
        }
    double worst = 0;
    for (int m : {1, 3, 11}) {
        std::vector<float> xd((size_t)m * D), xq((size_t)m * D);
        ABuf<BlockQ8_1_G64> A;
        A.alloc((size_t)m * NB);
        for (int t = 0; t < m; ++t) {
            for (int i = 0; i < D; ++i) xd[(size_t)t * D + i] = (float)((double)(rnd() % 2000001) / 1e6 - 1.0);
            quantize_q8_1_g64(&xd[(size_t)t * D], A.p + (size_t)t * NB, D);
            for (int b = 0; b < NB; ++b)
                for (int i = 0; i < 64; ++i) xq[(size_t)t * D + b * 64 + i] = A.p[(size_t)t * NB + b].d * (float)A.p[(size_t)t * NB + b].qs[i];
        }
        std::vector<float> out((size_t)m * nrows), wf(D);
        gemm_sk_q8_1(W.data(), SkCode::S1_5, A.p, out.data(), nrows, D, m);
        for (int r = 0; r < nrows; ++r) {
            strata::quant::sk::dequant_sk_row(&W[r * rb], SkCode::S1_5, wf.data(), D);
            for (int t = 0; t < m; ++t) {
                double ref = 0, mag = 0;
                for (int i = 0; i < D; ++i) { ref += (double)wf[i] * xq[(size_t)t * D + i]; mag += std::fabs((double)wf[i] * xq[(size_t)t * D + i]); }
                worst = std::max(worst, std::fabs(ref - out[(size_t)t * nrows + r]) / (mag + 1e-30));
            }
        }
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), "library kernel S1.5: worst |err|/sum|w*x| = %.2e (m=1,3,11)\n", worst);
    report += buf;
    return worst < 2e-6;
}

// Random-weight self test of the fast row kernel for every format S1.25, S2..S6 (the real SQ-64 file only contains S2/S3/S4):
// random payload bytes / D / u, real quantized activations; reference = dequant_sk_row (double) . dequantized Q8_1 activations.
bool CpuExperts::selftest_kernels(std::string& report) {
#if HY4_ANY_FAST
    uint64_t s = 0x1234567ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    bool ok = true;
    char buf[200];
    const KernCaps kcs = probe_kernel_caps();
#if HY4_KERNELS
    const int klist[] = {1, 4, 2};
#else
    const int klist[] = {3};
#endif
    for (int kk : klist) {
#if HY4_KERNELS
    if ((kk == 1 && !kcs.avx2) || (kk == 4 && !kcs.avx_vnni) || (kk == 2 && !kcs.avx512)) { report += std::string(kern_name(kk)) + ": not supported by this CPU, skipped\n"; continue; }
#else
    if (!kcs.neon_dp) { report += std::string(kern_name(kk)) + ": not supported by this CPU, skipped\n"; continue; }
#endif
    for (int code : {0, 2, 3, 4, 5, 6, 8}) {
        const size_t sb = superblock_bytes((SkCode)code), rb = 6 * sb;
        const int nrows = 5;
        std::vector<uint8_t> W(rb * nrows);
        for (int r = 0; r < nrows; ++r)
            for (int b = 0; b < 6; ++b) {
                uint8_t* p = &W[r * rb + b * sb];
                for (size_t i = 0; i < sb; ++i) p[i] = (uint8_t)rnd();
                const uint16_t h = (uint16_t)(0x2800 + (rnd() % 0x1000));  // fp16 in ~[0.03, 4)
                std::memcpy(p, &h, 2);
            }
        double worst = 0, worst_h = 0, ex_q8 = 0, ex_hi = 0;
        for (int m : {1, 3, 8, 11}) {
            ABuf<ActRow> A;
            A.alloc(m);
            std::vector<float> xd((size_t)m * D), xq((size_t)m * D);
            const ActRow* xp[16];
            for (int t = 0; t < m; ++t) {
                for (int i = 0; i < D; ++i) xd[(size_t)t * D + i] = (float)((double)(rnd() % 2000001) / 1e6 - 1.0) * (1.0f + (i % 64 == 5 ? 6.0f : 0.0f));
                BlockQ8_1_G64 blk[24];
                quantize_q8_1_g64(&xd[(size_t)t * D], blk, D);
                for (int b = 0; b < 24; ++b) {
                    kern_store(kk, &A.p[t], b, blk[b]);
                    for (int i = 0; i < 64; ++i) xq[(size_t)t * D + b * 64 + i] = blk[b].d * (float)blk[b].qs[i];
                }
                xp[t] = &A.p[t];
            }
            std::vector<float> out((size_t)m * nrows);
            kern_gemm(kk, code, W.data(), nrows, xp, m, out.data(), false);
            std::vector<float> wf(D);
            for (int r = 0; r < nrows; ++r) {
                strata::quant::sk::dequant_sk_row(&W[r * rb], (SkCode)code, wf.data(), D);
                for (int t = 0; t < m; ++t) {
                    double ref = 0, mag = 0;
                    for (int i = 0; i < D; ++i) { ref += (double)wf[i] * xq[(size_t)t * D + i]; mag += std::fabs((double)wf[i] * xq[(size_t)t * D + i]); }
                    worst = std::max(worst, std::fabs(ref - out[(size_t)t * nrows + r]) / (mag + 1e-30));
                    double e8 = 0;
                    for (int i = 0; i < D; ++i) e8 += (double)wf[i] * ((double)xd[(size_t)t * D + i]);
                    ex_q8 = std::max(ex_q8, std::fabs(e8 - out[(size_t)t * nrows + r]) / (mag + 1e-30));
                }
            }
            // hi mode: two-term activations, same weights
            std::vector<float> xq2((size_t)m * D);
            for (int t = 0; t < m; ++t) {
                BlockQ8_1_G64 blk[24], blk2[24];
                quantize_q8_1_g64(&xd[(size_t)t * D], blk, D);
                residual_blocks(&xd[(size_t)t * D], blk, blk2, 24);
                for (int b = 0; b < 24; ++b) {
                    kern_store_hi(kk, &A.p[t], b, blk[b], blk2[b]);
                    for (int i = 0; i < 64; ++i) xq2[(size_t)t * D + b * 64 + i] = blk[b].d * (float)blk[b].qs[i] + blk2[b].d * (float)blk2[b].qs[i];
                }
            }
            std::vector<float> outh((size_t)m * nrows);
            kern_gemm(kk, code, W.data(), nrows, xp, m, outh.data(), true);
            for (int r = 0; r < nrows; ++r) {
                strata::quant::sk::dequant_sk_row(&W[r * rb], (SkCode)code, wf.data(), D);
                for (int t = 0; t < m; ++t) {
                    double ref = 0, mag = 0, rex = 0;
                    for (int i = 0; i < D; ++i) {
                        ref += (double)wf[i] * xq2[(size_t)t * D + i];
                        rex += (double)wf[i] * xd[(size_t)t * D + i];
                        mag += std::fabs((double)wf[i] * xd[(size_t)t * D + i]);
                    }
                    worst_h = std::max(worst_h, std::fabs(ref - outh[(size_t)t * nrows + r]) / (mag + 1e-30));
                    ex_hi = std::max(ex_hi, std::fabs(rex - outh[(size_t)t * nrows + r]) / (mag + 1e-30));
                }
            }
        }
        std::snprintf(buf, sizeof(buf), "%s %s: worst |err|/sum|w*x| = %.2e (m=1,3,8,11); hi mode vs 2-term ref %.2e; vs exact x: q8 %.1e, hi %.1e\n", kern_name(kk),
                      code == 0 ? "S1.25" : (std::string("S") + std::to_string(code)).c_str(), worst, worst_h, ex_q8, ex_hi);
        report += buf;
        if (!(worst < 2e-6) || !(worst_h < 2e-6)) ok = false;
    }
    }
    return selftest_lib_s15(report) && ok;
#else
    report = "fast kernel not compiled (needs AVX2+FMA+F16C or AVX-512 BW/VNNI/VBMI+F16C); nothing to test\n";
    return selftest_lib_s15(report);
#endif
}

CpuExperts::CpuExperts() : p_(new Impl) {}
CpuExperts::~CpuExperts() = default;

bool CpuExperts::init(const std::string& path, int n_threads, const std::vector<std::vector<bool>>* mask, std::string& err) {
    if (n_threads <= 0) n_threads = default_threads();
    return p_->load(path, n_threads, mask, err);
}
bool CpuExperts::resident(int layer, int e) const {
    return layer >= 0 && layer < p_->nlayers && e >= 0 && e < E && p_->recs[(size_t)layer * E + e].body != nullptr;
}
CpuExpertView CpuExperts::view(int layer, int e) const {
    CpuExpertView v;
    if (!resident(layer, e)) return v;
    const Rec& r = p_->recs[(size_t)layer * E + e];
    v.body = r.body;
    v.dn_off = r.dn_off;
    v.gu_code = r.gu;
    v.dn_code = r.dn;
    v.body_bytes = r.body_bytes;
    return v;
}
bool CpuExperts::fast_kernel() const { return p_->fast; }
const char* CpuExperts::kernel_name() const { return kern_name(p_->kern); }
int CpuExperts::n_layers() const { return p_->nlayers; }
uint64_t CpuExperts::arena_bytes() const { return p_->arena_size; }
uint64_t CpuExperts::resident_count() const { return p_->n_resident; }
bool CpuExperts::arena_locked() const { return p_->locked; }
double CpuExperts::load_seconds() const { return p_->t_load; }
int CpuExperts::n_threads() const { return p_->nth; }

void CpuExperts::submit(int layer, int T, const float* z, const int32_t* idx, const float* w, const uint8_t* use, float* zout, bool accumulate) {
    Impl& I = *p_;
    wait();
    if (T <= 0 || layer < 0 || layer >= I.nlayers) return;
    const int Tc = std::min(T, TC);
    if ((size_t)Tc > I.cap_tok) {
        const size_t cap = std::max<size_t>(Tc, std::min<size_t>(TC, I.cap_tok * 2));
        if (I.fast) {
            I.tokA.alloc(cap);
            I.h2A.alloc(cap * TOPK);
        } else {
            I.tokq.alloc(cap * NB);
            I.h2q.alloc(cap * TOPK * NB);
            I.tokq2.alloc(cap * NB);
            I.h2q2.alloc(cap * TOPK * NB);
        }
        I.Y.alloc(cap * TOPK * D);
        I.pair_tok.assign(cap * TOPK, 0);
        I.posof.assign(cap * TOPK, -1);
        I.tok_used.assign(cap, 0);
        I.cap_tok = cap;
    }
    Job j;
    j.layer = layer;
    j.T = T;
    j.z = z;
    j.idx = idx;
    j.w = w;
    j.use = use;
    j.zout = zout;
    j.accum = accumulate;
    j.hi = I.hi;
    j.t_post = now_s();
    {
        std::lock_guard<std::mutex> lk(I.m);
        I.job = j;
        I.posted.store(I.gen.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        I.gen.fetch_add(1, std::memory_order_release);
    }
    I.cv_work.notify_all();
}

bool CpuExperts::done() const { return p_->done_gen.load(std::memory_order_acquire) == p_->posted.load(std::memory_order_relaxed); }

void CpuExperts::wait() {
    Impl& I = *p_;
    const uint64_t want = I.posted.load(std::memory_order_relaxed);
    const double t0 = now_s();
    uint32_t it = 0;
    while (I.done_gen.load(std::memory_order_acquire) != want) {
        HY4_RELAX();
        if ((++it & 255) == 0 && (now_s() - t0) * 1e6 > I.spin_us) {
            std::unique_lock<std::mutex> lk(I.m);
            I.cv_done.wait(lk, [&] { return I.done_gen.load(std::memory_order_acquire) == want; });
            break;
        }
    }
}

void CpuExperts::prefetch_experts(int layer, const int32_t* experts, int n_experts) {
    Impl& I = *p_;
    if (!I.l3_pf_enabled || !experts || n_experts <= 0 || layer < 0 || layer >= I.nlayers) return;
    if (!done()) return;  // never contend with an active compute job
    const Rec* lr = &I.recs[(size_t)layer * E];
    int n = 0;
    for (int i = 0; i < n_experts && n < 64; ++i) {
        const int32_t e = experts[i];
        if (e < 0 || e >= E || !lr[e].body) continue;
        bool dup = false;
        for (int j = 0; j < n; ++j) if (I.pf_experts[j] == e) { dup = true; break; }
        if (!dup) I.pf_experts[n++] = e;
    }
    if (n == 0) return;
    I.pf_layer = layer;
    I.pf_n = n;
    I.pf_cursor.store(0, std::memory_order_relaxed);
    I.pf_gen.fetch_add(1, std::memory_order_release);
}

void CpuExperts::set_hi_activations(bool on) {
    wait();
    p_->hi = on;
}
bool CpuExperts::hi_activations() const { return p_->hi; }

CpuExpertStats CpuExperts::stats() const { return p_->st; }
void CpuExperts::reset_stats() { p_->st = CpuExpertStats(); }

void CpuExperts::prefetch_cold(int layer, const int32_t* experts, int n) {
    Impl& I = *p_;
    if (!I.cs || !experts || n <= 0 || layer < 0 || layer >= I.nlayers) return;
    const Rec* lr = I.recs.data() + (size_t)layer * E;
    for (int i = 0; i < n; ++i) {
        const int e = experts[i];
        if (e < 0 || e >= E) continue;
        const Rec& r = lr[e];
        if (r.body || !r.cold) continue;
        I.cs->enqueue_prefetch((int64_t)layer * E + e, layer, r.file_off, (size_t)r.body_bytes, (size_t)r.dn_off);
    }
}

void CpuExperts::drop_stale_cold(int current_layer) {
    Impl& I = *p_;
    if (!I.cs) return;
    I.cs->sched.drop_stale(current_layer, I.cs->gen);
}

}  // namespace hy4
