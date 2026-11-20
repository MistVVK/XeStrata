// hy4s_cpuexp.cpp - CPU routed-expert engine (AVX2) for the Hy4-Flash Intel Arc lane. See hy4s_cpuexp.h.
//
// Kernel idea: the weights of S1.25 / S2_K / S3_K are unpacked into 4 "planes" (weight index 4m+j -> plane j, byte m); the Q8_1 activation of a token is
// stored in the same planar permutation (ActPlan) once per call, so no byte interleave is needed per row: 4 x maddubs per pair of 64-weight groups.
// S1.25 uses PDEP to split the 5-bit codeword indices and PSHUFB lookups of the 32-entry codebook (one table per component, offset +2 so the
// products are unsigned x signed; the offset is removed with the group's sum(qs)).  Integer sums are exact, the float accumulation order is the
// one of dot_sk_q8_1_row (group by group), so the results match the engine's reference kernel.
#include "hy4s_cpuexp.h"

#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/artifact/strata_reader.hpp"

#if !defined(__AVX2__) || !defined(__BMI2__) || !defined(__F16C__)
#error "hy4s_cpuexp needs AVX2 + BMI2 + F16C (compile with -march=native on a Zen3 / Haswell+ CPU)"
#endif

namespace hy4s {
using namespace strata::quant::sk;

namespace {

inline uint16_t ld16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t ld64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline float f16f(uint16_t h) { return _cvtsh_ss(h); }

// exact copy of S125_CODEBOOK (sk_codecs.cpp)
constexpr int8_t kCb[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
    {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
    {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},   {0, 0, 0, -2},
    {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},  {2, 0, 2, 0},
};

struct Tabs {
    alignas(32) uint8_t lo[4][16];
    alignas(32) uint8_t hi[4][16];
    Tabs() {
        for (int c = 0; c < 4; ++c)
            for (int i = 0; i < 16; ++i) {
                lo[c][i] = (uint8_t)(kCb[i][c] + 2);
                hi[c][i] = (uint8_t)(kCb[16 + i][c] + 2);
            }
    }
};
const Tabs& tabs() {
    static const Tabs t;
    return t;
}

// the two float divisions of the effective scale, tabulated with the very same expressions (so the values are identical)
struct Luts {
    float u255[256];  // (float)u / 255.0f
    float fine[16];   // 0.5f + (float)n / 15.0f
    Luts() {
        for (int i = 0; i < 256; ++i) u255[i] = (float)i / 255.0f;
        for (int i = 0; i < 16; ++i) fine[i] = 0.5f + (float)i / 15.0f;
    }
};
const Luts& luts() {
    static const Luts l;
    return l;
}

// position of group g's integer sum in the vector layout of the horizontal-add tree (quad q = pairs 4q..4q+3: lane i of half gg = pair 4q+i, group 2p+gg)
struct Perm {
    int pos[24];
    constexpr Perm() : pos() {
        for (int g = 0; g < 24; ++g) {
            const int p = g >> 1, gg = g & 1, q = p >> 2, i = p & 3;
            pos[g] = q * 8 + gg * 4 + i;
        }
    }
};
constexpr Perm kPerm;

// 4 rows at a time x M tokens (M <= 16).  Per quad of pairs (= 2 super-blocks) the weights of 4 pairs are unpacked into 16 planes once and reused for every token.
template <int CODE>
void rows_kernel(const uint8_t* W, int nrows, const ActPlan* const* acts, int M, float* out, size_t ldo) {
    static_assert(CODE == 0 || CODE == 2 || CODE == 3 || CODE == 4, "code");
    constexpr size_t SB = CODE == 0 ? 48 : CODE == 2 ? 70 : CODE == 3 ? 102 : 134;
    constexpr size_t RB = SB * 6;
    constexpr float HALF = CODE == 2 ? 1.5f : CODE == 3 ? 3.5f : 7.5f;
    const __m256i ones = _mm256_set1_epi16(1);
    const Luts& LU = luts();
    __m256i Tlo[4], Thi[4];
    if constexpr (CODE == 0) {
        const Tabs& T = tabs();
        for (int c = 0; c < 4; ++c) {
            Tlo[c] = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)T.lo[c]));
            Thi[c] = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)T.hi[c]));
        }
    }
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i hctl = _mm256_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    const __m256i hM0 = _mm256_set1_epi16(0x0001), hM1 = _mm256_set1_epi16(0x0100);
    for (int r0 = 0; r0 < nrows; r0 += 4) {
        const int nr = std::min(4, nrows - r0);
        float eff[4][24];
        alignas(32) int32_t isum[4][16][24];
        for (int rr = 0; rr < 4; ++rr) {
            const uint8_t* row = W + (size_t)(r0 + std::min(rr, nr - 1)) * RB;
            for (int sb = 0; sb < 6; ++sb) {
                const uint8_t* blk = row + sb * SB;
                const float D = f16f(ld16(blk));
                const uint8_t* u = blk + 2;
                if constexpr (CODE == 0) {
                    const uint8_t f0 = blk[6 + 40], f1 = blk[6 + 41];
                    const float fine[4] = {LU.fine[f0 & 0x0Fu], LU.fine[(f0 >> 4) & 0x0Fu], LU.fine[f1 & 0x0Fu], LU.fine[(f1 >> 4) & 0x0Fu]};
                    for (int g = 0; g < 4; ++g) eff[rr][sb * 4 + g] = D * LU.u255[u[g]] * fine[g];
                } else {
                    for (int g = 0; g < 4; ++g) eff[rr][sb * 4 + g] = D * LU.u255[u[g]];
                }
            }
            for (int q = 0; q < 3; ++q) {
                __m256i w[4][4];
                for (int i = 0; i < 4; ++i) {
                    const int sb = 2 * q + (i >> 1), pp = i & 1;
                    const uint8_t* pl = row + sb * SB + 6;
                    if constexpr (CODE == 2 || CODE == 3) {
                        const __m256i v = _mm256_loadu_si256((const __m256i*)(pl + 32 * pp));
                        w[i][0] = _mm256_and_si256(v, m3);
                        w[i][1] = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
                        w[i][2] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);
                        w[i][3] = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
                        if constexpr (CODE == 3) {
                            const __m128i h = _mm_loadu_si128((const __m128i*)(pl + 64 + 16 * pp));
                            const __m256i hh = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(h), hctl);
                            auto hi = [&](__m256i lo, int j) {
                                const __m256i t = (j == 0) ? hh : (j == 1) ? _mm256_srli_epi16(hh, 1) : (j == 2) ? _mm256_srli_epi16(hh, 2) : _mm256_srli_epi16(hh, 3);
                                const __m256i b = _mm256_or_si256(_mm256_and_si256(t, hM0), _mm256_and_si256(_mm256_srli_epi16(t, 4), hM1));
                                return _mm256_or_si256(lo, _mm256_slli_epi16(b, 2));
                            };
                            w[i][0] = hi(w[i][0], 0);
                            w[i][1] = hi(w[i][1], 1);
                            w[i][2] = hi(w[i][2], 2);
                            w[i][3] = hi(w[i][3], 3);
                        }
                    } else if constexpr (CODE == 4) {
                        // 4-bit: byte b of a group holds weights 2b (low nibble) and 2b+1 (high nibble); plane j wants weight 4m+j = byte 2m+(j>>1), nibble j&1
                        const __m256i A = _mm256_loadu_si256((const __m256i*)(pl + 64 * pp)), B = _mm256_loadu_si256((const __m256i*)(pl + 64 * pp + 32));
                        const __m256i X = _mm256_permute2x128_si256(A, B, 0x20), Y = _mm256_permute2x128_si256(A, B, 0x31);  // low lane = group 0, high lane = group 1
                        const __m256i ev = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
                        const __m256i od = _mm256_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
                        const __m256i E = _mm256_unpacklo_epi64(_mm256_shuffle_epi8(X, ev), _mm256_shuffle_epi8(Y, ev));
                        const __m256i O = _mm256_unpacklo_epi64(_mm256_shuffle_epi8(X, od), _mm256_shuffle_epi8(Y, od));
                        const __m256i m15 = _mm256_set1_epi8(15);
                        w[i][0] = _mm256_and_si256(E, m15);
                        w[i][1] = _mm256_and_si256(_mm256_srli_epi16(E, 4), m15);
                        w[i][2] = _mm256_and_si256(O, m15);
                        w[i][3] = _mm256_and_si256(_mm256_srli_epi16(O, 4), m15);
                    } else {
                        const uint8_t* qp = pl + 20 * pp;
                        const uint64_t K5 = 0x1F1F1F1F1F1F1F1FULL;
                        const __m128i i0 = _mm_set_epi64x((long long)_pdep_u64(ld64(qp + 5), K5), (long long)_pdep_u64(ld64(qp), K5));
                        const __m128i i1 = _mm_set_epi64x((long long)_pdep_u64(ld64(qp + 15), K5), (long long)_pdep_u64(ld64(qp + 10), K5));
                        const __m256i idx = _mm256_set_m128i(i1, i0);
                        const __m256i sel = _mm256_slli_epi16(idx, 3);
                        for (int c = 0; c < 4; ++c) w[i][c] = _mm256_blendv_epi8(_mm256_shuffle_epi8(Tlo[c], idx), _mm256_shuffle_epi8(Thi[c], idx), sel);
                    }
                }
                for (int t = 0; t < M; ++t) {
                    const int8_t* xb = acts[t]->p + q * 512;
                    __m256i v[4];
                    for (int i = 0; i < 4; ++i) {
                        const int8_t* xp = xb + i * 128;
                        const __m256i s = _mm256_add_epi16(
                            _mm256_add_epi16(_mm256_maddubs_epi16(w[i][0], _mm256_loadu_si256((const __m256i*)xp)), _mm256_maddubs_epi16(w[i][1], _mm256_loadu_si256((const __m256i*)(xp + 32)))),
                            _mm256_add_epi16(_mm256_maddubs_epi16(w[i][2], _mm256_loadu_si256((const __m256i*)(xp + 64))), _mm256_maddubs_epi16(w[i][3], _mm256_loadu_si256((const __m256i*)(xp + 96)))));
                        v[i] = _mm256_madd_epi16(s, ones);
                    }
                    const __m256i R = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
                    _mm256_store_si256((__m256i*)&isum[rr][t][q * 8], R);
                }
            }
        }
        // float tails of the 4 rows interleaved (4 independent accumulation chains), group order as dot_sk_q8_1_row
        for (int t = 0; t < M; ++t) {
            const ActPlan& a = *acts[t];
            float total[4] = {0.f, 0.f, 0.f, 0.f};
            for (int g = 0; g < 24; ++g) {
                const int pg = kPerm.pos[g];
                for (int rr = 0; rr < 4; ++rr) {
                    if constexpr (CODE == 0) {
                        const int32_t is = isum[rr][t][pg] - 2 * a.sq[g];
                        total[rr] += eff[rr][g] * a.d[g] * static_cast<float>(is);
                    } else {
                        total[rr] += eff[rr][g] * (a.d[g] * static_cast<float>(isum[rr][t][pg]) - HALF * a.s[g]);
                    }
                }
            }
            for (int rr = 0; rr < nr; ++rr) out[(size_t)t * ldo + r0 + rr] = total[rr];
        }
    }
}

inline void plan_set_block(ActPlan& o, int g, const BlockQ8_1_G64& b) {
    int8_t* dst = o.p + (g >> 1) * 128 + (g & 1) * 16;
    int32_t sum = 0;
    for (int j = 0; j < 4; ++j)
        for (int m = 0; m < 16; ++m) dst[j * 32 + m] = b.qs[4 * m + j];
    for (int i = 0; i < 64; ++i) sum += b.qs[i];
    o.d[g] = b.d;
    o.s[g] = b.s;
    o.sq[g] = sum;
}
inline void plan_get_block(const ActPlan& o, int g, BlockQ8_1_G64& b) {
    const int8_t* src = o.p + (g >> 1) * 128 + (g & 1) * 16;
    for (int j = 0; j < 4; ++j)
        for (int m = 0; m < 16; ++m) b.qs[4 * m + j] = src[j * 32 + m];
    b.d = o.d[g];
    b.s = o.s[g];
}

// any code through the engine's reference dot (slow path, used for codes without a fast kernel)
void rows_generic(int code, const uint8_t* W, int nrows, const ActPlan* const* acts, int M, float* out, size_t ldo) {
    const size_t rb = 6 * superblock_bytes((SkCode)code);
    alignas(64) BlockQ8_1_G64 qb[CX_NG];
    for (int t = 0; t < M; ++t) {
        for (int g = 0; g < CX_NG; ++g) plan_get_block(*acts[t], g, qb[g]);
        for (int r = 0; r < nrows; ++r) out[(size_t)t * ldo + r] = dot_sk_q8_1_row(W + (size_t)r * rb, (SkCode)code, qb, CX_K);
    }
}

inline void rows_any(int code, const uint8_t* W, int nrows, const ActPlan* const* acts, int M, float* out, size_t ldo) {
    if (!cx_rows_q8(code, W, nrows, acts, M, out, ldo)) rows_generic(code, W, nrows, acts, M, out, ldo);
}

inline float dotf(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    return _mm_cvtss_f32(x);
}

inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
void plan_from_blocks(const BlockQ8_1_G64* b, ActPlan& o) {
    for (int g = 0; g < CX_NG; ++g) plan_set_block(o, g, b[g]);
}
void plan_from_floats(const float* x, ActPlan& o) {
    alignas(64) BlockQ8_1_G64 qb[CX_NG];
    quantize_q8_1_g64(x, qb, CX_K);
    plan_from_blocks(qb, o);
}

bool cx_rows_q8(int code, const uint8_t* W, int nrows, const ActPlan* const* acts, int M, float* out, size_t ldo) {
    switch (code) {
        case 0: rows_kernel<0>(W, nrows, acts, M, out, ldo); return true;
        case 2: rows_kernel<2>(W, nrows, acts, M, out, ldo); return true;
        case 3: rows_kernel<3>(W, nrows, acts, M, out, ldo); return true;
        case 4: rows_kernel<4>(W, nrows, acts, M, out, ldo); return true;
        default: return false;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
struct ERef {
    const uint8_t* body = nullptr;
    uint32_t dn_off = 0, body_bytes = 0, touched = 0;
    uint8_t gu = 0, dn = 0;
    bool ok = false;
};
struct TLS {
    alignas(64) float g[16 * 64];
    alignas(64) float u[16 * 64];
    alignas(64) float dq[CX_K];
    alignas(64) float h[64];
    alignas(64) float sqv[64];
    double code_sec[9];
};

struct CpuExperts::Impl {
    strata::artifact::StrataFileReader rd;
    int fd = -1;
    int fdd = -1;  // O_DIRECT descriptor (reads that must not go through the page cache)
    std::atomic<uint64_t> direct_bytes{0}, cached_bytes{0}, miss_bytes{0};
    uint64_t sec4_off = 0;
    void* map = nullptr;
    size_t map_len = 0;
    std::vector<ERef> ex;  // [41*512] by layer*512 + expert id
    int maxT = 512;
    std::vector<ActPlan> zplan, hplan;
    std::vector<float> ye, hf;
    std::vector<TLS> tls;
    std::vector<int32_t> cnt, st, cur, pair_tok, pos;
    std::vector<float> pair_w;
    std::vector<int> act_e;
    bool warmed = false, synthetic = false;
    struct FreeDel { void operator()(uint8_t* p) const { free(p); } };
    std::unique_ptr<uint8_t, FreeDel> synth_pool;
    float limit = 10.0f;
};

CpuExperts::CpuExperts() : im_(new Impl) {}
CpuExperts::~CpuExperts() { close(); }

void CpuExperts::close() {
    if (im_->map) munmap(im_->map, im_->map_len);
    im_->map = nullptr;
    if (im_->fd >= 0) ::close(im_->fd);
    if (im_->fdd >= 0) ::close(im_->fdd);
    im_->fdd = -1;
    im_->fd = -1;
    base_ = nullptr;
}

static double ram_budget_bytes();
bool CpuExperts::open(const std::string& path, int nthreads, std::string& err, int max_tokens) {
    Impl& I = *im_;
    nth_ = std::max(1, nthreads);
    I.maxT = std::max(1, max_tokens);
    if (!I.rd.open(path, err)) return false;
    const auto& h = I.rd.header();
    I.sec4_off = h.sections[4].file_offset;
    sec4_bytes_ = h.sections[4].byte_length;
    {
        const double f = std::min(1.0, ram_budget_bytes() / (double)sec4_bytes_);
        streaming_ = f < 1.0;  // some experts will come from the SSD: prefetch the routed bodies in decode
        bypass_ = f < 0.85;    // clearly larger than the RAM: prefill reads bypass the page cache (O_DIRECT) so the decode's hot set stays cached
        if (const char* e = std::getenv("HY4S_DIRECT")) bypass_ = e[0] == '1';
    }
    const auto& dir = I.rd.expert_directory();
    if (dir.size() != 41 * 512) {
        err = "expert directory size " + std::to_string(dir.size()) + " != 20992";
        return false;
    }
    if (I.sec4_off % 4096) {
        err = "Section 4 not page aligned";
        return false;
    }
    I.fd = ::open(path.c_str(), O_RDONLY);
    I.fdd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (I.fd < 0) {
        err = "open failed";
        return false;
    }
    I.map_len = (sec4_bytes_ + 4095) & ~4095ull;
    I.map = mmap(nullptr, I.map_len, PROT_READ, MAP_SHARED, I.fd, (off_t)I.sec4_off);
    if (I.map == MAP_FAILED) {
        I.map = nullptr;
        err = std::string("mmap failed: ") + strerror(errno);
        return false;
    }
    base_ = (const uint8_t*)I.map;
    I.ex.assign(41 * 512, ERef());
    for (size_t i = 0; i < dir.size(); ++i) {
        const auto& e = dir[i];
        ERef& r = I.ex[(i / 512) * 512 + e.expert_id];
        r.body = base_ + e.body_offset;
        r.dn_off = e.dn_offset;
        r.body_bytes = e.body_bytes;
        r.gu = e.gu_code_id;
        r.dn = e.dn_code_id;
        r.touched = (uint32_t)(3072 + 3072ull * 6 * superblock_bytes((SkCode)e.gu_code_id) + 1536ull * 6 * superblock_bytes((SkCode)e.dn_code_id));
        r.ok = e.body_offset + e.body_bytes <= sec4_bytes_;
    }
    I.zplan.resize(I.maxT);
    I.hplan.resize((size_t)I.maxT * CX_NTOP);
    I.ye.resize((size_t)I.maxT * CX_NTOP * CX_K);
    I.tls.resize(nth_ + 1);
    I.cnt.assign(512, 0);
    I.st.assign(513, 0);
    I.cur.assign(512, 0);
    I.pair_tok.assign((size_t)I.maxT * CX_NTOP, 0);
    I.pos.assign((size_t)I.maxT * CX_NTOP, -1);
    I.pair_w.assign((size_t)I.maxT * CX_NTOP, 0.0f);
    I.act_e.reserve(512);
    return true;
}

bool CpuExperts::open_synthetic(int nthreads, std::string& err, int max_tokens, uint32_t seed) {
    Impl& I = *im_;
    nth_ = std::max(1, nthreads);
    I.maxT = std::max(1, max_tokens);
    // SQ-32-like bodies: ranks 0..31 (gu S2, dn S3), 32..175 (S2, S2), 176..511 (S1.25, S1.25); all 41 layers alias the same 512 bodies
    size_t total = 0;
    uint64_t off[512];
    uint8_t gu[512], dn[512];
    uint32_t dno[512], bytes[512];
    for (int e = 0; e < 512; ++e) {
        gu[e] = e < 176 ? 2 : 0;
        dn[e] = e < 32 ? 3 : e < 176 ? 2 : 0;
        dno[e] = (uint32_t)(3072 + 18432ull * superblock_bytes((SkCode)gu[e]));
        bytes[e] = (uint32_t)((dno[e] + 9216ull * superblock_bytes((SkCode)dn[e]) + 4095) & ~4095ull);
        off[e] = total;
        total += bytes[e];
    }
    I.synth_pool.reset((uint8_t*)aligned_alloc(4096, total + 4096));
    if (!I.synth_pool) { err = "alloc"; return false; }
    uint8_t* P = I.synth_pool.get();
    std::memset(P, 0, total + 4096);
    uint64_t s = 0x9E3779B97F4A7C15ull ^ seed;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int e = 0; e < 512; ++e) {
        uint8_t* b = P + off[e];
        for (int i = 0; i < 1536; ++i) {  // sq_xd_dn: fp16 in [0.5, 2)
            uint16_t h = _cvtss_sh(0.5f + 1.5f * (float)(rnd() % 1000) / 1000.0f, 0);
            std::memcpy(b + 2 * i, &h, 2);
        }
        auto fill = [&](uint8_t* w, int code, size_t nsb) {
            const size_t sb = superblock_bytes((SkCode)code);
            for (size_t k = 0; k < nsb; ++k) {
                uint8_t* blk = w + k * sb;
                for (size_t j = 6; j < sb; ++j) blk[j] = (uint8_t)rnd();
                uint16_t h = _cvtss_sh(0.0015f + 0.012f * (float)(rnd() % 1000) / 1000.0f, 0);
                std::memcpy(blk, &h, 2);
                for (int g = 0; g < 4; ++g) blk[2 + g] = (uint8_t)(1 + rnd() % 255);
            }
        };
        fill(b + 3072, gu[e], 18432);
        fill(b + dno[e], dn[e], 9216);
    }
    base_ = P;
    sec4_bytes_ = total;
    I.ex.assign(41 * 512, ERef());
    for (int L = 0; L < 41; ++L)
        for (int e = 0; e < 512; ++e) {
            ERef& r = I.ex[L * 512 + e];
            r.body = P + off[e];
            r.dn_off = dno[e];
            r.body_bytes = bytes[e];
            r.gu = gu[e];
            r.dn = dn[e];
            r.touched = (uint32_t)(3072 + 3072ull * 6 * superblock_bytes((SkCode)gu[e]) + 1536ull * 6 * superblock_bytes((SkCode)dn[e]));
            r.ok = true;
        }
    I.warmed = true;
    I.synthetic = true;
    I.zplan.resize(I.maxT);
    I.hplan.resize((size_t)I.maxT * CX_NTOP);
    I.ye.resize((size_t)I.maxT * CX_NTOP * CX_K);
    I.tls.resize(nth_ + 1);
    I.cnt.assign(512, 0);
    I.st.assign(513, 0);
    I.cur.assign(512, 0);
    I.pair_tok.assign((size_t)I.maxT * CX_NTOP, 0);
    I.pos.assign((size_t)I.maxT * CX_NTOP, -1);
    I.pair_w.assign((size_t)I.maxT * CX_NTOP, 0.0f);
    I.act_e.reserve(512);
    return true;
}

bool CpuExperts::expert_info(int L, int e, const uint8_t** body, int* gu, int* dn, uint32_t* dn_off, uint32_t* bytes) const {
    if (L < 0 || L >= 41 || e < 0 || e >= 512) return false;
    const ERef& r = im_->ex[L * 512 + e];
    if (!r.ok) return false;
    *body = r.body;
    *gu = r.gu;
    *dn = r.dn;
    *dn_off = r.dn_off;
    *bytes = r.body_bytes;
    return true;
}

// RAM this process may use for expert bodies in the page cache: MemAvailable (the file's own cached pages count as available) - 9 GB (process, pinned KV, Engram rows); HY4S_WARM_GB overrides
static double ram_budget_bytes() {
    double budget = 1e30, avail_gb = 0;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char ln[256];
        while (fgets(ln, sizeof ln, f)) {
            unsigned long long kb;
            if (sscanf(ln, "MemAvailable: %llu kB", &kb) == 1) avail_gb = kb * 1024.0 / 1e9;
        }
        fclose(f);
    }
    if (avail_gb > 0) budget = (avail_gb - 9.0) * 1e9;
    if (const char* e = std::getenv("HY4S_WARM_GB")) budget = atof(e) * 1e9;
    return budget;
}

double CpuExperts::warm(int nthreads, double* seconds) {
    Impl& I = *im_;
    if (I.synthetic) return 0.0;
    double t0 = now_s();
    const size_t CH = 8ull << 20;
    const size_t nch = (sec4_bytes_ + CH - 1) / CH;
    // Experts larger than the RAM this box can spare (SQ-64 on a 62 GB machine, SQ-96/128): warm only a share of the file, spread evenly over it (every k-th 8 MiB chunk);
    // the rest is read from the SSD on demand and the page cache keeps whatever the decode touches. Budget = MemAvailable - 9 GB (process, pinned KV, Engram rows), or HY4S_WARM_GB.
    const double budget = ram_budget_bytes();
    const double frac = budget >= (double)sec4_bytes_ ? 1.0 : std::max(0.0, budget / (double)sec4_bytes_);
    auto selected = [&](size_t c) { return frac >= 1.0 || (size_t)((c + 1) * frac) > (size_t)(c * frac); };
    std::atomic_flag fail = ATOMIC_FLAG_INIT;
#pragma omp parallel num_threads(std::max(1, nthreads))
    {
        void* buf = nullptr;
        if (posix_memalign(&buf, 4096, CH)) fail.test_and_set();
#pragma omp for schedule(dynamic, 1)
        for (long c = 0; c < (long)nch; ++c) {
            if (!buf || !selected((size_t)c)) continue;
            size_t off = (size_t)c * CH, n = std::min(CH, (size_t)sec4_bytes_ - off);
            size_t got = 0;
            while (got < n) {
                ssize_t r = pread(I.fd, buf, n - got, (off_t)(I.sec4_off + off + got));
                if (r <= 0) break;
                got += (size_t)r;
            }
        }
        free(buf);
    }
    double t1 = now_s();
    // map the pages (page-table population)
    size_t popped_fail = 0;
    if (frac < 1.0) {
#pragma omp parallel for num_threads(std::max(1, nthreads)) schedule(dynamic, 8)
        for (long c = 0; c < (long)nch; ++c) {
            if (!selected((size_t)c)) continue;
            size_t off = (size_t)c * CH, n = std::min(CH, (size_t)I.map_len - off);
            madvise((char*)I.map + off, n, 22 /*MADV_POPULATE_READ*/);
        }
    }
#pragma omp parallel for num_threads(std::max(1, nthreads)) schedule(dynamic, 1) reduction(+ : popped_fail)
    for (long c = 0; c < (long)(frac < 1.0 ? 0 : (I.map_len + (1ull << 30) - 1) >> 30); ++c) {
        size_t off = (size_t)c << 30, n = std::min((size_t)1 << 30, I.map_len - off);
        if (madvise((char*)I.map + off, n, 22 /*MADV_POPULATE_READ*/) != 0) {
            volatile uint8_t s = 0;
            for (size_t o = 0; o < n; o += 4096) s += ((const volatile uint8_t*)I.map)[off + o];
            (void)s;
            popped_fail++;
        }
    }
    double t2 = now_s();
    I.warmed = true;
    if (seconds) *seconds = t2 - t0;
    const double gb = sec4_bytes_ * std::min(1.0, frac) / 1e9;
    fprintf(stderr, "[cpuexp] warm: read %.2f GB of %.2f GB (%.0f%%) in %.1f s (%.2f GB/s), map %.1f s%s\n", gb, sec4_bytes_ / 1e9, 100.0 * std::min(1.0, frac), t1 - t0, gb / (t1 - t0), t2 - t1,
            popped_fail ? " (touch fallback)" : "");
    return gb / (t1 - t0);
}

void CpuExperts::drop_cache() {
    Impl& I = *im_;
    if (I.synthetic) return;
    madvise(I.map, I.map_len, MADV_DONTNEED);
    posix_fadvise(I.fd, (off_t)I.sec4_off, (off_t)sec4_bytes_, POSIX_FADV_DONTNEED);
    I.warmed = false;
}

// Copies one expert body (a pointer into the mapping) to dst. Resident in the page cache: memcpy. Not resident: one large pread (buffered: fills the cache, so later chunks hit it;
// 4 KiB page faults through the mapping ran at 0.9 GB/s, large reads at 3+ GB/s), or, when the experts are clearly larger than the RAM (bypass_), an O_DIRECT read that leaves the cache alone.
void CpuExperts::copy_body(const uint8_t* body, uint32_t bytes, void* dst) const {
    Impl& I = *im_;
    const uint64_t off = (uint64_t)(body - base_);
    if (!I.synthetic && I.fdd >= 0 && (off % 4096) == 0 && (bytes % 4096) == 0 && ((uintptr_t)dst % 4096) == 0) {
        const size_t npg = bytes / 4096;
        bool resident = true;
        unsigned char vec[1024];
        for (size_t p0 = 0; p0 < npg && resident; p0 += sizeof vec) {
            const size_t n = std::min(sizeof vec, npg - p0);
            if (mincore((void*)(body + p0 * 4096), n * 4096, vec) != 0) break;
            for (size_t i = 0; i < n; ++i)
                if (!(vec[i] & 1)) { resident = false; break; }
        }
        if (!resident) {
            size_t got = 0;
            if (bypass_) {
                // O_DIRECT into an ordinary page-aligned bounce buffer (pinned USM host memory is not accepted as a direct-I/O target), then one memcpy into the ring slot
                struct Bounce {
                    uint8_t* p = nullptr;
                    size_t cap = 0;
                    ~Bounce() { free(p); }
                };
                static thread_local Bounce bb;
                if (bb.cap < bytes) {
                    free(bb.p);
                    bb.p = nullptr;
                    bb.cap = 0;
                    if (posix_memalign((void**)&bb.p, 4096, (size_t)bytes + (1u << 20)) == 0) bb.cap = (size_t)bytes + (1u << 20);
                }
                if (bb.cap >= bytes) {
                    while (got < bytes) {
                        ssize_t r = pread(I.fdd, bb.p + got, bytes - got, (off_t)(I.sec4_off + off + got));
                        if (r <= 0) break;
                        got += (size_t)r;
                    }
                    if (got == bytes) {
                        std::memcpy(dst, bb.p, bytes);
                        I.direct_bytes += bytes;
                        return;
                    }
                }
            } else {
                while (got < bytes) {
                    ssize_t r = pread(I.fd, (char*)dst + got, bytes - got, (off_t)(I.sec4_off + off + got));
                    if (r <= 0) break;
                    got += (size_t)r;
                }
                if (got == bytes) {
                    I.miss_bytes += bytes;
                    return;
                }
            }
        }
    }
    I.cached_bytes += bytes;
    std::memcpy(dst, body, bytes);
}
uint64_t CpuExperts::miss_read_bytes() const { return im_->miss_bytes.load(); }
uint64_t CpuExperts::direct_read_bytes() const { return im_->direct_bytes.load(); }
uint64_t CpuExperts::cached_read_bytes() const { return im_->cached_bytes.load(); }

void CpuExperts::prefetch_experts(int L, const int32_t* idx, int n) {
    Impl& I = *im_;
    for (int i = 0; i < n; ++i) {
        int e = idx[i];
        if ((unsigned)e >= 512) continue;
        const ERef& r = I.ex[L * 512 + e];
        if (!r.ok) continue;
        posix_fadvise(I.fd, (off_t)(I.sec4_off + (r.body - base_)), (off_t)r.body_bytes, POSIX_FADV_WILLNEED);
    }
}
void CpuExperts::prefetch_layer(int L) {
    Impl& I = *im_;
    // the layer's bodies are not contiguous in id order but ranks are: directory order = rank order, contiguous per layer
    uint64_t lo = ~0ull, hi = 0;
    for (int e = 0; e < 512; ++e) {
        const ERef& r = I.ex[L * 512 + e];
        if (!r.ok) continue;
        uint64_t o = (uint64_t)(r.body - base_);
        lo = std::min(lo, o);
        hi = std::max(hi, o + r.body_bytes);
    }
    if (hi > lo) posix_fadvise(I.fd, (off_t)(I.sec4_off + lo), (off_t)(hi - lo), POSIX_FADV_WILLNEED);
}

void CpuExperts::run(int L, int T, const float* z, const int32_t* idx, const float* w, float* zout, int act_mode) {
    Impl& I = *im_;
    if (T > I.maxT) {
        for (int t0 = 0; t0 < T; t0 += I.maxT) {
            int n = std::min(I.maxT, T - t0);
            run(L, n, z + (size_t)t0 * CX_K, idx + (size_t)t0 * CX_NTOP, w + (size_t)t0 * CX_NTOP, zout + (size_t)t0 * CX_K, act_mode);
        }
        return;
    }
    const double tstart = now_s();
    // ---- routing: counting sort of the (token, k) pairs by expert
    std::fill(I.cnt.begin(), I.cnt.end(), 0);
    const int npair = T * CX_NTOP;
    for (int i = 0; i < npair; ++i) {
        int e = idx[i];
        if ((unsigned)e < 512 && I.ex[L * 512 + e].ok) I.cnt[e]++;
    }
    I.act_e.clear();
    int acc = 0;
    for (int e = 0; e < 512; ++e) {
        I.st[e] = acc;
        I.cur[e] = acc;
        if (I.cnt[e]) I.act_e.push_back(e);
        acc += I.cnt[e];
    }
    I.st[512] = acc;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < CX_NTOP; ++k) {
            int e = idx[t * CX_NTOP + k];
            if ((unsigned)e < 512 && I.ex[L * 512 + e].ok) {
                int p = I.cur[e]++;
                I.pair_tok[p] = t;
                I.pair_w[p] = w[t * CX_NTOP + k];
                I.pos[t * CX_NTOP + k] = p;
            } else {
                I.pos[t * CX_NTOP + k] = -1;
            }
        }
    if (act_mode == 0 && I.hf.size() < (size_t)I.maxT * CX_NTOP * CX_K) I.hf.resize((size_t)I.maxT * CX_NTOP * CX_K);
    const int nact = (int)I.act_e.size();
    for (auto& t : I.tls) std::memset(t.code_sec, 0, sizeof t.code_sec);
    uint64_t bytes = 0;
    for (int e : I.act_e) bytes += I.ex[L * 512 + e].touched;
    if (!I.warmed) prefetch_experts(L, idx, npair);

    double tp[5] = {tstart, 0, 0, 0, 0};
    ActPlan* zplan = I.zplan.data();
    ActPlan* hplan = I.hplan.data();
    float* ye = I.ye.data();
    float* hf = I.hf.data();
    const float lim = I.limit;
    const ERef* exL = I.ex.data() + L * 512;
    const int* act_e = I.act_e.data();
    const int32_t* cnt = I.cnt.data();
    const int32_t* st = I.st.data();
    const int32_t* ptok = I.pair_tok.data();
    const float* pw = I.pair_w.data();
    const int32_t* pos = I.pos.data();
#pragma omp parallel num_threads(nth_)
    {
        TLS& tl = I.tls[omp_get_thread_num()];
        // ---- phase 0: activations of the T tokens
        if (act_mode == 1) {
#pragma omp for schedule(static)
            for (int t = 0; t < T; ++t) plan_from_floats(z + (size_t)t * CX_K, zplan[t]);
        }
        if (omp_get_thread_num() == 0) tp[1] = now_s();
        // ---- phase A: gate/up rows -> h (per expert, per 64-wide block of h)
#pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < nact * CX_NG; ++task) {
            const double ta = now_s();
            const int ei = task / CX_NG, b = task % CX_NG;
            const int e = act_e[ei];
            const ERef& E = exL[e];
            const int M = cnt[e], p0 = st[e];
            const size_t rb = 6 * superblock_bytes((SkCode)E.gu);
            const uint8_t* guW = E.body + 3072;
            const uint16_t* sq16 = (const uint16_t*)E.body;
            const int d0 = b * 64;
            for (int r = 0; r < 64; ++r) tl.sqv[r] = f16f(sq16[d0 + r]);
            for (int c0 = 0; c0 < M; c0 += 16) {
                const int mc = std::min(16, M - c0);
                if (act_mode == 1) {
                    const ActPlan* ap[16];
                    for (int i = 0; i < mc; ++i) ap[i] = &zplan[ptok[p0 + c0 + i]];
                    rows_any(E.gu, guW + (size_t)d0 * rb, 64, ap, mc, tl.g, 64);
                    rows_any(E.gu, guW + (size_t)(CX_K + d0) * rb, 64, ap, mc, tl.u, 64);
                } else {
                    for (int r = 0; r < 64; ++r) {
                        dequant_sk_row(guW + (size_t)(d0 + r) * rb, (SkCode)E.gu, tl.dq, CX_K);
                        for (int i = 0; i < mc; ++i) tl.g[i * 64 + r] = dotf(tl.dq, z + (size_t)ptok[p0 + c0 + i] * CX_K, CX_K);
                        dequant_sk_row(guW + (size_t)(CX_K + d0 + r) * rb, (SkCode)E.gu, tl.dq, CX_K);
                        for (int i = 0; i < mc; ++i) tl.u[i * 64 + r] = dotf(tl.dq, z + (size_t)ptok[p0 + c0 + i] * CX_K, CX_K);
                    }
                }
                for (int i = 0; i < mc; ++i) {
                    for (int r = 0; r < 64; ++r) {
                        const float g = tl.g[i * 64 + r], u = tl.u[i * 64 + r];
                        const float s = std::min(siluf(g), lim);
                        const float uc = std::max(-lim, std::min(u, lim));
                        tl.h[r] = (s * uc) / tl.sqv[r];
                    }
                    const int pr = p0 + c0 + i;
                    if (act_mode == 1) {
                        alignas(64) BlockQ8_1_G64 qb;
                        quantize_q8_1_g64(tl.h, &qb, 64);
                        plan_set_block(hplan[pr], b, qb);
                    } else {
                        std::memcpy(hf + (size_t)pr * CX_K + d0, tl.h, 64 * sizeof(float));
                    }
                }
            }
            tl.code_sec[E.gu] += now_s() - ta;
        }
        if (omp_get_thread_num() == 0) tp[2] = now_s();
        // ---- phase B: down rows
#pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < nact * CX_NG; ++task) {
            const double ta = now_s();
            const int ei = task / CX_NG, b = task % CX_NG;
            const int e = act_e[ei];
            const ERef& E = exL[e];
            const int M = cnt[e], p0 = st[e];
            const size_t rb = 6 * superblock_bytes((SkCode)E.dn);
            const uint8_t* dnW = E.body + E.dn_off;
            const int r0 = b * 64;
            for (int c0 = 0; c0 < M; c0 += 16) {
                const int mc = std::min(16, M - c0);
                float* o = ye + (size_t)(p0 + c0) * CX_K + r0;
                if (act_mode == 1) {
                    const ActPlan* ap[16];
                    for (int i = 0; i < mc; ++i) ap[i] = &hplan[p0 + c0 + i];
                    rows_any(E.dn, dnW + (size_t)r0 * rb, 64, ap, mc, o, CX_K);
                } else {
                    for (int r = 0; r < 64; ++r) {
                        dequant_sk_row(dnW + (size_t)(r0 + r) * rb, (SkCode)E.dn, tl.dq, CX_K);
                        for (int i = 0; i < mc; ++i) o[(size_t)i * CX_K + r] = dotf(tl.dq, hf + (size_t)(p0 + c0 + i) * CX_K, CX_K);
                    }
                }
            }
            tl.code_sec[E.dn] += now_s() - ta;
        }
        if (omp_get_thread_num() == 0) tp[3] = now_s();
        // ---- phase C: zout[t] = sum_k w * ye, experts in increasing id order (as the oracle)
#pragma omp for schedule(static)
        for (int t = 0; t < T; ++t) {
            int ks[CX_NTOP], n = 0;
            for (int k = 0; k < CX_NTOP; ++k)
                if (pos[t * CX_NTOP + k] >= 0) {
                    int j = n++;
                    while (j > 0 && idx[t * CX_NTOP + ks[j - 1]] > idx[t * CX_NTOP + k]) {
                        ks[j] = ks[j - 1];
                        --j;
                    }
                    ks[j] = k;
                }
            float* zo = zout + (size_t)t * CX_K;
            for (int d = 0; d < CX_K; ++d) zo[d] = 0.0f;
            for (int a = 0; a < n; ++a) {
                const int p = pos[t * CX_NTOP + ks[a]];
                const float wv = pw[p];
                const float* y = ye + (size_t)p * CX_K;
                for (int d = 0; d < CX_K; ++d) zo[d] += y[d] * wv;
            }
        }
    }
    tp[4] = now_s();
    for (auto& t : I.tls)
        for (int c = 0; c < 9; ++c) st_.code_sec[c] += t.code_sec[c];
    st_.seconds += tp[4] - tstart;
    st_.t_prep += tp[1] - tp[0];
    st_.t_gu += tp[2] - tp[1];
    st_.t_dn += tp[3] - tp[2];
    st_.t_red += tp[4] - tp[3];
    st_.calls++;
    st_.bytes += bytes;
    st_.pairs += (uint64_t)acc;
}

}  // namespace hy4s
