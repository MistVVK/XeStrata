// hy4_cpu_experts.cpp - see hy4_cpu_experts.h
#include "hy4_cpu_experts.h"

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
#include "strata/quant/sk_codecs.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HY4_RELAX() _mm_pause()
#else
#define HY4_RELAX() std::this_thread::yield()
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

// The fast kernel needs AVX-512 BW + VNNI + VBMI (Ice Lake / Rocket Lake / Zen 4 and later) and F16C; otherwise the library
// gemm_sk_q8_1 path is used (also selectable with STRATA_CPUEXP_KERNEL=lib, and used when an S8 expert is resident).
#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VBMI__) && defined(__F16C__)
#define HY4_FAST 1
#else
#define HY4_FAST 0
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

struct Rec {
    const uint8_t* body = nullptr;
    uint32_t dn_off = 0, body_bytes = 0;
    uint8_t gu = 0, dn = 0;
    uint32_t gu_sb = 0, dn_sb = 0;  // superblock bytes
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
    double t_post = 0;
};

struct SpinBarrier {
    std::atomic<int> count{0};
    std::atomic<int> sense{0};
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

struct TScratch {
    ABuf<BlockQ8_1_G64> xg;  // gathered z blocks for up to 16 tokens
    ABuf<float> tg, tu;      // [16][64] gate / up (or down) results
    int sense = 0;
};

// ==================================================================================================== fast kernel
// Activation row (K = 1536 = 6 super-blocks of 256) in a "transposed" layout so that the weight planes of every S_K format
// can be unpacked with a handful of in-register ops and dotted with dpbusd, without any per-64-weights horizontal reduction:
//   xj[sb][j][L*16 + k] = int8 x[(sb*4 + L)*64 + 4*k + j]      (j = 0..3, L = 64-block within the super-block, k = 0..15)
//   xd[sb][L*4 + i]     = d of block sb*4+L                     (replicated x4: one float per dword lane)
//   xs4[sb][L*4 + i]    = 0.25 * s of that block                (s = d * sum(qs); the x4 replication is undone by the 0.25)
// Weight plane P_j of one super-block is the byte vector with lane L, byte k = q[64*L + 4*k + j] (0..2^bits-1). Then
//   sum_lanes( cvt(dpbusd chain of P_j . xj) * eff * xd ) - half * sum_lanes( eff * xs4 )  ==  sum_g eff_g (xd_g*isum_g - half*xs_g)
// i.e. exactly the library's dot_sk_q8_1_row, with float accumulation in 16 lanes and a single horizontal sum per row.
struct alignas(64) ActRow {
    int8_t xj[6][4][64];
    float xd[6][16];
    float xs4[6][16];
};
static_assert(sizeof(ActRow) == 2304, "ActRow");

#if HY4_FAST
struct FastConsts {
    alignas(64) uint8_t idx4[4][64];    // vpermi2b index: plane j of the 128-byte nibble array (S4/S5/S6)
    alignas(64) uint8_t ctrl3[4][64];   // vpmultishiftqb control: bit n -> bit 2 of the byte (S3 high plane)
    alignas(64) uint8_t ctrl5[4][64];   // ... -> bit 4 (S5 high plane)
    alignas(64) uint8_t tr[64];         // store_block: dst[16*j+k] = src[4*k+j]
    alignas(64) int32_t effidx[16];
    alignas(64) int64_t qidx[8];
    FastConsts() {
        for (int j = 0; j < 4; ++j)
            for (int b = 0; b < 64; ++b) {
                const int L = b >> 4, k = b & 15, q = k >> 3, kk = k & 7;
                idx4[j][b] = (uint8_t)(32 * L + 2 * k + (j >> 1));
                ctrl3[j][b] = (uint8_t)((32 * q + 4 * kk + j - 2) & 63);
                ctrl5[j][b] = (uint8_t)((32 * q + 4 * kk + j - 4) & 63);
            }
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 16; ++k) tr[16 * j + k] = (uint8_t)(4 * k + j);
        for (int i = 0; i < 16; ++i) effidx[i] = i >> 2;
        for (int i = 0; i < 8; ++i) qidx[i] = i >> 1;
    }
};
static const FastConsts kc;

// block b (0..23) of a row from the library's Q8_1_G64 block
static inline void store_block(ActRow* a, int b, const BlockQ8_1_G64& blk) {
    const int sb = b >> 2, L = b & 3;
    const __m512i R = _mm512_permutexvar_epi8(_mm512_load_si512((const void*)kc.tr), _mm512_load_si512((const void*)blk.qs));
    _mm_store_si128((__m128i*)&a->xj[sb][0][L * 16], _mm512_castsi512_si128(R));
    _mm_store_si128((__m128i*)&a->xj[sb][1][L * 16], _mm512_extracti32x4_epi32(R, 1));
    _mm_store_si128((__m128i*)&a->xj[sb][2][L * 16], _mm512_extracti32x4_epi32(R, 2));
    _mm_store_si128((__m128i*)&a->xj[sb][3][L * 16], _mm512_extracti32x4_epi32(R, 3));
    _mm_store_ps(&a->xd[sb][L * 4], _mm_set1_ps(blk.d));
    _mm_store_ps(&a->xs4[sb][L * 4], _mm_set1_ps(0.25f * blk.s));
}

#define HY4_HI3(J) _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_load_si512((const void*)kc.ctrl3[J]), src), m4)
#define HY4_HI5(J) _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_load_si512((const void*)kc.ctrl5[J]), src), m16)
#define HY4_NIB(J) _mm512_permutex2var_epi8(n0, _mm512_load_si512((const void*)kc.idx4[J]), n1)

// P[j] lane L byte k = q[64*L + 4*k + j] of one super-block payload
template <int FMT>
static inline void unpack_planes(const uint8_t* pl, __m512i (&P)[4]) {
    const __m512i m3 = _mm512_set1_epi8(3), m4 = _mm512_set1_epi8(4), m15 = _mm512_set1_epi8(15), m16 = _mm512_set1_epi8(16),
                  m30 = _mm512_set1_epi8(0x30);
    (void)m3; (void)m4; (void)m16; (void)m30;
    if constexpr (FMT == 2 || FMT == 3) {
        const __m512i w = _mm512_loadu_si512((const void*)pl);
        if constexpr (FMT == 2) {
            P[0] = _mm512_and_si512(w, m3);
            P[1] = _mm512_and_si512(_mm512_srli_epi16(w, 2), m3);
            P[2] = _mm512_and_si512(_mm512_srli_epi16(w, 4), m3);
            P[3] = _mm512_and_si512(_mm512_srli_epi16(w, 6), m3);
        } else {
            const __m512i src = _mm512_permutexvar_epi64(_mm512_load_si512((const void*)kc.qidx), _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(pl + 64))));
            P[0] = _mm512_ternarylogic_epi32(w, HY4_HI3(0), m3, 0xEC);
            P[1] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(w, 2), HY4_HI3(1), m3, 0xEC);
            P[2] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(w, 4), HY4_HI3(2), m3, 0xEC);
            P[3] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(w, 6), HY4_HI3(3), m3, 0xEC);
        }
    } else {
        const __m512i n0 = _mm512_loadu_si512((const void*)pl), n1 = _mm512_loadu_si512((const void*)(pl + 64));
        const __m512i v0 = HY4_NIB(0), v1 = _mm512_srli_epi16(HY4_NIB(1), 4), v2 = HY4_NIB(2), v3 = _mm512_srli_epi16(HY4_NIB(3), 4);
        if constexpr (FMT == 4) {
            P[0] = _mm512_and_si512(v0, m15);
            P[1] = _mm512_and_si512(v1, m15);
            P[2] = _mm512_and_si512(v2, m15);
            P[3] = _mm512_and_si512(v3, m15);
        } else if constexpr (FMT == 5) {
            const __m512i src = _mm512_permutexvar_epi64(_mm512_load_si512((const void*)kc.qidx), _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(pl + 128))));
            P[0] = _mm512_ternarylogic_epi32(v0, HY4_HI5(0), m15, 0xEC);
            P[1] = _mm512_ternarylogic_epi32(v1, HY4_HI5(1), m15, 0xEC);
            P[2] = _mm512_ternarylogic_epi32(v2, HY4_HI5(2), m15, 0xEC);
            P[3] = _mm512_ternarylogic_epi32(v3, HY4_HI5(3), m15, 0xEC);
        } else {  // S6: 2-bit high planes, byte i -> q[4i+j] in bits 2j
            const __m512i w6 = _mm512_loadu_si512((const void*)(pl + 128));
            P[0] = _mm512_ternarylogic_epi32(v0, _mm512_and_si512(_mm512_slli_epi16(w6, 4), m30), m15, 0xEC);
            P[1] = _mm512_ternarylogic_epi32(v1, _mm512_and_si512(_mm512_slli_epi16(w6, 2), m30), m15, 0xEC);
            P[2] = _mm512_ternarylogic_epi32(v2, _mm512_and_si512(w6, m30), m15, 0xEC);
            P[3] = _mm512_ternarylogic_epi32(v3, _mm512_and_si512(_mm512_srli_epi16(w6, 2), m30), m15, 0xEC);
        }
    }
}

template <int FMT>
struct FmtInfo;
template <> struct FmtInfo<2> { static constexpr int SB = 70;  static constexpr float HALF = 1.5f; };
template <> struct FmtInfo<3> { static constexpr int SB = 102; static constexpr float HALF = 3.5f; };
template <> struct FmtInfo<4> { static constexpr int SB = 134; static constexpr float HALF = 7.5f; };
template <> struct FmtInfo<5> { static constexpr int SB = 166; static constexpr float HALF = 15.5f; };
template <> struct FmtInfo<6> { static constexpr int SB = 198; static constexpr float HALF = 31.5f; };

// M tokens x 1 weight row (6 super-blocks of format FMT) -> out[t]
template <int FMT, int M>
static inline void row_dot(const uint8_t* row, const ActRow* const* X, float* out) {
    constexpr int SB = FmtInfo<FMT>::SB;
    __m512 facc[M];
    for (int t = 0; t < M; ++t) facc[t] = _mm512_setzero_ps();
    const __m128 inv255 = _mm_set1_ps(255.0f);
    const __m512 halfv = _mm512_set1_ps(FmtInfo<FMT>::HALF);
    const __m512i effidx = _mm512_load_si512((const void*)kc.effidx);
    for (int s = 0; s < 6; ++s) {
        const uint8_t* blk = row + (size_t)s * SB;
        uint16_t dh;
        uint32_t u32;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&u32, blk + 2, 4);
        // eff_g = D * (u_g / 255) exactly as in sk_codecs (float division, then multiply), replicated x4 per group
        __m128 u4 = _mm_div_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)u32))), inv255);
        u4 = _mm_mul_ps(u4, _mm_set1_ps(_cvtsh_ss(dh)));
        const __m512 effv = _mm512_permutexvar_ps(effidx, _mm512_castps128_ps512(u4));
        const __m512 effh = _mm512_mul_ps(effv, halfv);
        __m512i P[4];
        unpack_planes<FMT>(blk + 6, P);
        for (int t = 0; t < M; ++t) {
            const ActRow* a = X[t];
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), P[0], _mm512_load_si512((const void*)a->xj[s][0]));
            acc = _mm512_dpbusd_epi32(acc, P[1], _mm512_load_si512((const void*)a->xj[s][1]));
            acc = _mm512_dpbusd_epi32(acc, P[2], _mm512_load_si512((const void*)a->xj[s][2]));
            acc = _mm512_dpbusd_epi32(acc, P[3], _mm512_load_si512((const void*)a->xj[s][3]));
            const __m512 sc = _mm512_mul_ps(effv, _mm512_load_ps(a->xd[s]));
            facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), sc, facc[t]);
            facc[t] = _mm512_fnmadd_ps(effh, _mm512_load_ps(a->xs4[s]), facc[t]);
        }
    }
    for (int t = 0; t < M; ++t) out[t] = _mm512_reduce_add_ps(facc[t]);
}

// nrows rows x M tokens; out[i*nrows + r]
template <int FMT, int M>
static void rows_m(const uint8_t* W, int nrows, const ActRow* const* X, float* out) {
    constexpr size_t RBYTES = 6 * (size_t)FmtInfo<FMT>::SB;
    for (int r = 0; r < nrows; ++r) {
        float o[M];
        row_dot<FMT, M>(W + r * RBYTES, X, o);
        for (int i = 0; i < M; ++i) out[(size_t)i * nrows + r] = o[i];
    }
}

template <int FMT>
static void rows_fmt(const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out) {
    for (int g0 = 0; g0 < m; g0 += 8) {
        const int mg = std::min(8, m - g0);
        const ActRow* const* x = X + g0;
        float* o = out + (size_t)g0 * nrows;
        switch (mg) {
            case 1: rows_m<FMT, 1>(W, nrows, x, o); break;
            case 2: rows_m<FMT, 2>(W, nrows, x, o); break;
            case 3: rows_m<FMT, 3>(W, nrows, x, o); break;
            case 4: rows_m<FMT, 4>(W, nrows, x, o); break;
            case 5: rows_m<FMT, 5>(W, nrows, x, o); break;
            case 6: rows_m<FMT, 6>(W, nrows, x, o); break;
            case 7: rows_m<FMT, 7>(W, nrows, x, o); break;
            default: rows_m<FMT, 8>(W, nrows, x, o); break;
        }
    }
}

// W: nrows rows of K=1536 in format `code` (2..6); X[m] activation rows; out[m][nrows]
static void rows_gemm(int code, const uint8_t* W, int nrows, const ActRow* const* X, int m, float* out) {
    switch (code) {
        case 2: rows_fmt<2>(W, nrows, X, m, out); break;
        case 3: rows_fmt<3>(W, nrows, X, m, out); break;
        case 4: rows_fmt<4>(W, nrows, X, m, out); break;
        case 5: rows_fmt<5>(W, nrows, X, m, out); break;
        default: rows_fmt<6>(W, nrows, X, m, out); break;
    }
}
#endif  // HY4_FAST

}  // namespace

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
    bool fast = false;
    ABuf<float> Y;                  // [P][1536]
    size_t cap_tok = 0;
    std::vector<int> cnt, aoff, act_e, act_o, act_n, pair_tok, posof;
    std::vector<uint8_t> tok_used;
    std::atomic<int> qctr{0}, guctr{0}, dnctr{0}, cbctr{0};
    int nact = 0;
    // ---- stats (written by thread 0 only, read after wait())
    CpuExpertStats st;
    std::atomic<uint64_t> posted{0};

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
#ifdef _WIN32
        arena = (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        arena_map_size = bytes;
        if (!arena) { err = "VirtualAlloc failed for expert arena"; return false; }
        locked = VirtualLock(arena, bytes) != 0;
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
        locked = mlock(arena, bytes) == 0;  // best effort (RLIMIT_MEMLOCK)
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
};

// ==================================================================================================== load
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
    std::vector<std::thread> rt;
    for (int i = 0; i < nr; ++i) rt.emplace_back(reader);
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
    t_load = now_s() - t0;
    fast = false;
#if HY4_FAST
    fast = true;
    if (const char* k = std::getenv("STRATA_CPUEXP_KERNEL"))
        if (!std::strcmp(k, "lib")) fast = false;
    for (const Rec& r : recs)
        if (r.body && (r.gu < 2 || r.gu > 6 || r.dn < 2 || r.dn > 6)) {
            if (fast) std::fprintf(stderr, "[cpuexp] S8/other expert code resident: using the library kernel\n");
            fast = false;
        }
#endif
    std::fprintf(stderr, "[cpuexp] %llu experts resident, %.2f GB in %.1f s (%.2f GB/s), %zu reads, mlock=%s, isa=%s, kernel=%s\n",
                 (unsigned long long)n_resident, arena_size / 1e9, t_load, arena_size / 1e9 / std::max(t_load, 1e-9), runs.size(),
                 locked ? "yes" : "no", strata::quant::sk::active_sk_isa(), fast ? "fast(avx512-vbmi)" : "lib");
    start_threads(n_threads);
    return true;
}

// ==================================================================================================== pool
namespace {
// physical-core-first list of logical CPU ids to pin to (empty = do not pin)
std::vector<int> pin_list(int want) {
    std::vector<int> prim, sec;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (len && GetLogicalProcessorInformation(info.data(), &len)) {
        for (auto& i : info) {
            if (i.Relationship != RelationProcessorCore) continue;
            bool first = true;
            for (int b = 0; b < (int)(8 * sizeof(ULONG_PTR)); ++b)
                if ((i.ProcessorMask >> b) & 1) { (first ? prim : sec).push_back(b); first = false; }
        }
    }
#else
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return {};
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
        (first == c ? prim : sec).push_back(c);
    }
#endif
    std::vector<int> all = prim;
    all.insert(all.end(), sec.begin(), sec.end());
    if (all.empty() || want > (int)all.size()) return {};
    return all;
}
}  // namespace

void CpuExperts::Impl::start_threads(int n) {
    nth = std::max(1, n);
    bar.n = nth;
    ts.reset(new TScratch[nth]);
    for (int ti = 0; ti < nth; ++ti) {
        TScratch& s = ts[ti];
        if (!fast) s.xg.alloc((size_t)MG * NB);
        s.tg.alloc((size_t)MGF * RB);
        s.tu.alloc((size_t)MGF * RB);
    }
    if (const char* e = std::getenv("STRATA_CPUEXP_SPIN_US")) spin_us = std::atoi(e);
    std::vector<int> pins;
    const char* pe = std::getenv("STRATA_CPUEXP_PIN");
    if (!(pe && pe[0] == '0')) pins = pin_list(nth);
    cnt.assign(E, 0);
    aoff.assign(E + 1, 0);
    act_e.assign(E, 0);
    act_o.assign(E, 0);
    act_n.assign(E, 0);
    threads.reserve(nth);
    for (int i = 0; i < nth; ++i) {
        threads.emplace_back([this, i]() { worker(i); });
        if (!pins.empty()) {
#ifdef _WIN32
            SetThreadAffinityMask((HANDLE)threads.back().native_handle(), (DWORD_PTR)1 << pins[i]);
#else
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(pins[i], &cs);
            pthread_setaffinity_np(threads.back().native_handle(), sizeof(cs), &cs);
#endif
        }
    }
    std::fprintf(stderr, "[cpuexp] %d pool threads%s, spin %d us\n", nth, pins.empty() ? " (not pinned)" : " (pinned, physical cores first)", spin_us);
}

void CpuExperts::Impl::worker(int tid) {
    uint64_t seen = 0;
    for (;;) {
        const double t_spin = now_s();
        uint32_t it = 0;
        while (gen.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_relaxed)) {
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
            if (e < 0 || e >= E || !lr[e].body) { ++skipped; continue; }
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
            ++nact;
            bytes += lr[e].body_bytes;
        }
        off += cnt[e];
    }
    aoff[E] = off;
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
            if (e < 0 || e >= E || !lr[e].body) continue;
            const int p = fill[e]++;
            pair_tok[p] = t;
            posof[(size_t)t * TOPK + s] = p;
        }
    qctr.store(0);
    guctr.store(0);
    dnctr.store(0);
    cbctr.store(0);
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
    const uint8_t* wgu = r.body + 3072;
    const uint8_t* wg = wgu + (size_t)b * RB * row_b;
    const uint8_t* wu = wgu + (size_t)(D + b * RB) * row_b;
    float sq[RB];
    {
        const uint16_t* h16 = (const uint16_t*)r.body + b * RB;
        for (int i = 0; i < RB; ++i) sq[i] = fp16_to_fp32(h16[i]);
    }
    const int mgmax = fast ? MGF : MG;
    for (int g0 = 0; g0 < m; g0 += mgmax) {
        const int mg = std::min(mgmax, m - g0);
#if HY4_FAST
        if (fast) {
            const ActRow* xp[MGF];
            for (int i = 0; i < mg; ++i) xp[i] = tokA.p + (size_t)pair_tok[p0 + g0 + i];
            rows_gemm(r.gu, wg, RB, xp, mg, S.tg.p);
            rows_gemm(r.gu, wu, RB, xp, mg, S.tu.p);
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
        }
        for (int i = 0; i < mg; ++i) {
            float h[RB];
            const float* g = S.tg.p + i * RB;
            const float* u = S.tu.p + i * RB;
            for (int d = 0; d < RB; ++d) {
                const float gv = g[d];
                const float si = gv / (1.0f + fexpf(-gv));
                const float sc = std::min(si, SWIGLU_LIMIT);
                const float uc = std::max(-SWIGLU_LIMIT, std::min(u[d], SWIGLU_LIMIT));
                h[d] = (sc * uc) / sq[d];
            }
#if HY4_FAST
            if (fast) {
                BlockQ8_1_G64 blk;
                quantize_q8_1_g64(h, &blk, 64);
                store_block(h2A.p + (size_t)(p0 + g0 + i), b, blk);
            } else
#endif
                quantize_q8_1_g64(h, h2q.p + (size_t)(p0 + g0 + i) * NB + b, 64);
        }
    }
}

// one down unit: output rows [c*64, c*64+64) of y for all tokens of active expert a
void CpuExperts::Impl::dn_unit(int tid, const Job& J, int a, int c) {
    TScratch& S = ts[tid];
    const Rec& r = recs[(size_t)J.layer * E + act_e[a]];
    const int p0 = act_o[a], m = act_n[a];
    const size_t row_b = 6ull * r.dn_sb;
    const uint8_t* wd = r.body + r.dn_off + (size_t)c * RB * row_b;
    const int mgmax = fast ? MGF : MG;
    for (int g0 = 0; g0 < m; g0 += mgmax) {
        const int mg = std::min(mgmax, m - g0);
#if HY4_FAST
        if (fast) {
            const ActRow* xp[MGF];
            for (int i = 0; i < mg; ++i) xp[i] = h2A.p + (size_t)(p0 + g0 + i);
            rows_gemm(r.dn, wd, RB, xp, mg, S.tg.p);
        } else
#endif
            gemm_sk_q8_1(wd, (SkCode)r.dn, h2q.p + (size_t)(p0 + g0) * NB, S.tg.p, RB, D, mg);
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
            const int u = qctr.fetch_add(1, std::memory_order_relaxed);
            if (u >= Tc * 3) break;
            const int t = u / 3, part = u % 3;
            if (!tok_used[t]) continue;
#if HY4_FAST
            if (fast) {
                alignas(64) BlockQ8_1_G64 blk[8];
                quantize_q8_1_g64(J.z + (size_t)(t0 + t) * D + part * 512, blk, 512);
                for (int i = 0; i < 8; ++i) store_block(tokA.p + t, part * 8 + i, blk[i]);
                continue;
            }
#endif
            quantize_q8_1_g64(J.z + (size_t)(t0 + t) * D + part * 512, tokq.p + (size_t)t * NB + part * 8, 512);
        }
        bar.wait(sense);
        lap(1);
        for (;;) {
            const int u = guctr.fetch_add(1, std::memory_order_relaxed);
            if (u >= nact * NUNIT) break;
            gu_unit(tid, J, u / NUNIT, u % NUNIT);
        }
        bar.wait(sense);
        lap(2);
        for (;;) {
            const int u = dnctr.fetch_add(1, std::memory_order_relaxed);
            if (u >= nact * NUNIT) break;
            dn_unit(tid, J, u / NUNIT, u % NUNIT);
        }
        bar.wait(sense);
        lap(3);
        const int cb = D / 256;
        for (;;) {
            const int u = cbctr.fetch_add(1, std::memory_order_relaxed);
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

// ==================================================================================================== public API
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

// Random-weight self test of the fast row kernel for every format S2..S6 (the real SQ-64 file only contains S2/S3/S4):
// random payload bytes / D / u, real quantized activations; reference = dequant_sk_row (double) . dequantized Q8_1 activations.
bool CpuExperts::selftest_kernels(std::string& report) {
#if HY4_FAST
    uint64_t s = 0x1234567ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    bool ok = true;
    char buf[200];
    for (int code = 2; code <= 6; ++code) {
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
        double worst = 0;
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
                    store_block(&A.p[t], b, blk[b]);
                    for (int i = 0; i < 64; ++i) xq[(size_t)t * D + b * 64 + i] = blk[b].d * (float)blk[b].qs[i];
                }
                xp[t] = &A.p[t];
            }
            std::vector<float> out((size_t)m * nrows);
            rows_gemm(code, W.data(), nrows, xp, m, out.data());
            std::vector<float> wf(D);
            for (int r = 0; r < nrows; ++r) {
                strata::quant::sk::dequant_sk_row(&W[r * rb], (SkCode)code, wf.data(), D);
                for (int t = 0; t < m; ++t) {
                    double ref = 0, mag = 0;
                    for (int i = 0; i < D; ++i) { ref += (double)wf[i] * xq[(size_t)t * D + i]; mag += std::fabs((double)wf[i] * xq[(size_t)t * D + i]); }
                    worst = std::max(worst, std::fabs(ref - out[(size_t)t * nrows + r]) / (mag + 1e-30));
                }
            }
        }
        std::snprintf(buf, sizeof(buf), "S%d: worst |err|/sum|w*x| = %.2e (m=1,3,8,11)\n", code, worst);
        report += buf;
        if (!(worst < 2e-6)) ok = false;
    }
    return ok;
#else
    report = "fast kernel not compiled (needs AVX-512 BW/VNNI/VBMI + F16C); nothing to test\n";
    return true;
#endif
}

CpuExperts::CpuExperts() : p_(new Impl) {}
CpuExperts::~CpuExperts() = default;

bool CpuExperts::init(const std::string& path, int n_threads, const std::vector<std::vector<bool>>* mask, std::string& err) {
    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency() / 2);
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

CpuExpertStats CpuExperts::stats() const { return p_->st; }
void CpuExperts::reset_stats() { p_->st = CpuExpertStats(); }

}  // namespace hy4
