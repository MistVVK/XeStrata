// sy_gpuexp.cpp - see sy_gpuexp.h
#include "sy_gpuexp.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace hy4s {

namespace xm = sycl::ext::oneapi::experimental::matrix;

namespace {

constexpr int K = 1536;
constexpr int NTOP = 16;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Exact S125_CODEBOOK of engine/src/quant/sk_codecs.cpp
constexpr int8_t kS125[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
    {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
    {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},    {0, 0, 0, -2},
    {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},   {2, 0, 2, 0},
};

struct Item {  // one (expert, 64-token tile) of a group
    uint32_t body_off;  // byte offset of the body inside the device slot
    uint32_t dn_off;    // offset of the down matrix inside the body
    uint32_t pbase;     // first pair row of the tile
    uint32_t ntok;      // valid tokens of the tile (1..64)
    int32_t gu, dn;     // codes
    int32_t pad0, pad1;
};

inline int code_sb_bytes(int code) { return code == 0 ? 48 : code == 2 ? 70 : code == 3 ? 102 : 134; }

// store two adjacent k values (one packed row, 4-byte aligned) with a single 32-bit write
inline void put2(half* p, float a, float b) { *reinterpret_cast<sycl::vec<half, 2>*>(p) = sycl::vec<half, 2>((half)a, (half)b); }

// dequantise the 256 weights of super-block `blk` of weight row `n` of the B tile into the VNNI-packed local tile: element (k, n) at [(k>>1)*128 + n*2 + (k&1)]
template <int CODE>
inline void dq_sb(const uint8_t* blk, half* sb_ptr, int n) {
    const float D = h2f_bits(ld16(blk));
    const uint8_t* pl = blk + 6;
    if constexpr (CODE == 0) {  // S1.25: 8 groups x 8 indices x 5 bits, 4 weights per index, fine multiplier per 64-weight sub-group
        const float f0 = 0.5f + (float)(pl[40] & 15) / 15.0f, f1 = 0.5f + (float)(pl[40] >> 4) / 15.0f;
        const float f2 = 0.5f + (float)(pl[41] & 15) / 15.0f, f3 = 0.5f + (float)(pl[41] >> 4) / 15.0f;
        const float fine[4] = {f0, f1, f2, f3};
        float eff[4];
        for (int g = 0; g < 4; ++g) eff[g] = D * ((float)blk[2 + g] / 255.0f) * fine[g];
        for (int grp = 0; grp < 8; ++grp) {
            const uint8_t* p = pl + grp * 5;
            const uint64_t bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
            const float e = eff[grp >> 1];
            for (int j = 0; j < 8; ++j) {
                const uint32_t idx = (uint32_t)((bits >> (5 * j)) & 0x1Fu);
                const int k0 = (grp * 8 + j) * 4;  // k0 is a multiple of 4: weights k0,k0+1 and k0+2,k0+3 share a packed row each
                half* d0 = sb_ptr + (k0 >> 1) * 128 + n * 2;
                put2(d0, (float)kS125[idx][0] * e, (float)kS125[idx][1] * e);
                put2(d0 + 128, (float)kS125[idx][2] * e, (float)kS125[idx][3] * e);
            }
        }
    } else if constexpr (CODE == 2) {  // S2_K: w = (q - 1.5) * D * u/255, 4 weights per payload byte
        for (int g = 0; g < 4; ++g) {
            const float eff = D * ((float)blk[2 + g] / 255.0f);
            for (int b = 0; b < 16; ++b) {
                const uint32_t by = pl[g * 16 + b];
                const int k0 = g * 64 + b * 4;
                half* d0 = sb_ptr + (k0 >> 1) * 128 + n * 2;
                put2(d0, ((float)(by & 3) - 1.5f) * eff, ((float)((by >> 2) & 3) - 1.5f) * eff);
                put2(d0 + 128, ((float)((by >> 4) & 3) - 1.5f) * eff, ((float)(by >> 6) - 1.5f) * eff);
            }
        }
    } else if constexpr (CODE == 3) {  // S3_K: q = low 2 bits (4 per byte) | high bit (8 per byte at +64) << 2;  w = (q - 3.5) * D * u/255
        for (int g = 0; g < 4; ++g) {
            const float eff = D * ((float)blk[2 + g] / 255.0f);
            for (int u8 = 0; u8 < 8; ++u8) {  // 8 weights per unit
                const int unit = g * 8 + u8;
                const uint32_t hi = pl[64 + unit];
                const uint32_t lo = (uint32_t)pl[2 * unit] | ((uint32_t)pl[2 * unit + 1] << 8);
                for (int jp = 0; jp < 4; ++jp) {  // weight pair (2jp, 2jp+1)
                    const uint32_t q0 = ((lo >> (4 * jp)) & 3) | (((hi >> (2 * jp)) & 1) << 2);
                    const uint32_t q1 = ((lo >> (4 * jp + 2)) & 3) | (((hi >> (2 * jp + 1)) & 1) << 2);
                    const int k0 = unit * 8 + 2 * jp;
                    put2(sb_ptr + (k0 >> 1) * 128 + n * 2, ((float)q0 - 3.5f) * eff, ((float)q1 - 3.5f) * eff);
                }
            }
        }
    } else {  // S4_K: weights 2b (low nibble) and 2b+1 (high nibble) of payload byte b; w = (q - 7.5) * D * u/255
        for (int g = 0; g < 4; ++g) {
            const float eff = D * ((float)blk[2 + g] / 255.0f);
            for (int b = 0; b < 32; ++b) {
                const uint32_t by = pl[g * 32 + b];
                put2(sb_ptr + ((g * 64 + 2 * b) >> 1) * 128 + n * 2, ((float)(by & 15) - 7.5f) * eff, ((float)(by >> 4) - 7.5f) * eff);
            }
        }
    }
}

// MODE 0: GEMM1 + SwiGLU epilogue -> hb (half [P][1536]);  MODE 1: GEMM2 -> ye (float [P][1536])
template <int MODE>
sycl::event launch_gemm(sycl::queue& q, const sycl::event* dep, const uint8_t* dslot, const Item* items, int nitems, const half* X, void* out) {
    constexpr int NT = MODE == 0 ? 48 : 24;
    return q.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        sycl::local_accessor<half, 1> sB(64 * 256, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nitems * NT * 64, 64), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0);
            const Item item = items[gid / NT];
            const int wt = gid % NT;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3, lane = tid & 7;
            const uint8_t* body = dslot + item.body_off;
            const int code = MODE == 0 ? item.gu : item.dn;
            const uint8_t* W = MODE == 0 ? body + 3072 : body + item.dn_off;
            const int row = MODE == 0 ? (tid < 32 ? 32 * wt + tid : K + 32 * wt + (tid - 32)) : 64 * wt + tid;
            const int sby = code_sb_bytes(code);
            const uint8_t* rp = W + (size_t)row * 6 * sby;
            const int m0 = (int)item.pbase + sgi * 8;
            const bool active = sgi * 8 < (int)item.ntok;
            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[8];
            for (int j = 0; j < 8; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);
            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);
            for (int sbi = 0; sbi < 6; ++sbi) {
                const uint8_t* blk = rp + (size_t)sbi * sby;
#ifdef GX_NODQ  // experiment only (results are garbage): measures the matrix-engine + staging part without the dequantisation
                if (sbi == 0) sb_ptr[tid] = (half)0.5f;
#else
                if (code == 0) dq_sb<0>(blk, sb_ptr, tid);
                else if (code == 2) dq_sb<2>(blk, sb_ptr, tid);
                else if (code == 3) dq_sb<3>(blk, sb_ptr, tid);
                else dq_sb<4>(blk, sb_ptr, tid);
#endif
                sycl::group_barrier(it.get_group());
                if (active) {  // sub-group sgi = 8 tokens [8*sgi, 8*sgi+8) x all 64 columns (8 accumulators); A fragments come straight from global memory (L1/L2)
                    const half* xrow = X + (size_t)m0 * K + (size_t)sbi * 256;
                    for (int s = 0; s < 16; ++s) {
                        auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(const_cast<half*>(xrow + s * 16));
                        xm::joint_matrix_load(sg, ma, xp, (size_t)K);
                        for (int j = 0; j < 8; ++j) {
                            xm::joint_matrix_load(sg, mb, sB_mp + (size_t)(s * 8) * 128 + j * 16, (size_t)128);
                            xm::joint_matrix_mad(sg, mc[j], ma, mb, mc[j]);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            {
                // epilogue through local memory: the B tile is dead after the K loop; scratch = float [64 tokens][64 columns]
                float* sc = reinterpret_cast<float*>(sb_ptr);
                auto sc_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sc);
                if (active)
                    for (int j = 0; j < 8; ++j) xm::joint_matrix_store(sg, mc[j], sc_mp + sgi * 8 * 64 + j * 8, (size_t)64, xm::layout::row_major);
                sycl::group_barrier(it.get_group());
                const int ntok = (int)item.ntok;
                if constexpr (MODE == 0) {
                    for (int idx = tid; idx < ntok * 32; idx += 64) {
                        const int t = idx >> 5, i = idx & 31;
                        const float g = sc[t * 64 + i], u = sc[t * 64 + 32 + i];
                        const float sq = h2f_bits(ld16(body + 2 * (32 * wt + i)));
                        const float s = sycl::fmin(g / (1.0f + sycl::exp(-g)), 10.0f);
                        const float uc = sycl::fmax(-10.0f, sycl::fmin(u, 10.0f));
                        ((half*)out)[(size_t)(item.pbase + t) * K + 32 * wt + i] = (half)((s * uc) / sq);
                    }
                } else {
                    for (int idx = tid; idx < ntok * 64; idx += 64) {
                        const int t = idx >> 6, i = idx & 63;
                        ((float*)out)[(size_t)(item.pbase + t) * K + 64 * wt + i] = sc[t * 64 + i];
                    }
                }
            }
        });
    });
}

sycl::event launch_gather(sycl::queue& q, const float* z, const int32_t* ptok, half* zg, int P) {
    const size_t n = (size_t)P * K;
    return q.parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < n) zg[i] = (half)z[(size_t)ptok[i / K] * K + i % K];
    });
}

sycl::event launch_reduce(sycl::queue& q, const float* ye, const int32_t* red, const float* pw, float* zout, int T) {
    return q.parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) {
        const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
        for (int d = tid; d < K; d += 256) {
            float acc = 0.0f;
            for (int j = 0; j < NTOP; ++j) {
                const int p = red[t * NTOP + j];
                if (p < 0) break;
                acc += ye[(size_t)p * K + d] * pw[p];
            }
            zout[(size_t)t * K + d] = acc;
        }
    });
}

}  // namespace

struct GpuExperts::Impl {
    const CpuExperts* cx = nullptr;
    sycl::queue cq;           // copy queue (profiling enabled, in-order)
    sycl::queue* comp = nullptr;
    bool profile = false;
    size_t slot_bytes = 0;
    int max_tokens = 0, copy_threads = 6;
    uint8_t* h_slot[3] = {};
    uint8_t* d_slot[2] = {};
    Item* h_items[3] = {};
    Item* d_items[2] = {};
    half *zg = nullptr, *hb = nullptr;
    float* ye = nullptr;
    int32_t *d_ptok = nullptr, *d_red = nullptr, *h_ptok = nullptr, *h_red = nullptr;
    float *d_pw = nullptr, *h_pw = nullptr;
    size_t dev_bytes = 0;
    sycl::event cp_ev[3], comp_ev[2], last_ev;
    bool cp_valid[3] = {false, false, false}, comp_valid[2] = {false, false}, last_valid = false;
    int group_ctr = 0;
    // per-layer planning
    std::vector<int32_t> cnt, st, cur, pair_tok, pos;
    std::vector<float> pair_w;
    std::vector<int> act_e;
    std::vector<std::pair<sycl::event, int>> prof;  // kind: 0 gemm1, 1 gemm2, 2 other, 3 h2d
    GpuExpertStats stats;
    bool inited = false;
    static constexpr int MAXITEMS = 1024;

    template <class T> T* dev(size_t n) {
        T* p = sycl::malloc_device<T>(n, cq);
        if (!p) {
            fprintf(stderr, "gpuexp: out of device memory (%zu bytes)\n", n * sizeof(T));
            exit(3);
        }
        dev_bytes += n * sizeof(T);
        return p;
    }
};

GpuExperts::GpuExperts() : im_(new Impl) {}
GpuExperts::~GpuExperts() {
    if (!im_->inited) return;
    im_->cq.wait();
    if (im_->comp) im_->comp->wait();
    for (auto* p : im_->h_slot) sycl::free(p, im_->cq);
    for (auto* p : im_->d_slot) sycl::free(p, im_->cq);
    for (auto* p : im_->h_items) sycl::free(p, im_->cq);
    for (auto* p : im_->d_items) sycl::free(p, im_->cq);
    sycl::free(im_->zg, im_->cq);
    sycl::free(im_->hb, im_->cq);
    sycl::free(im_->ye, im_->cq);
    sycl::free(im_->d_ptok, im_->cq);
    sycl::free(im_->d_red, im_->cq);
    sycl::free(im_->d_pw, im_->cq);
    sycl::free(im_->h_ptok, im_->cq);
    sycl::free(im_->h_red, im_->cq);
    sycl::free(im_->h_pw, im_->cq);
}

bool GpuExperts::init(const CpuExperts& cx, size_t stage_bytes, int max_tokens, std::string& err, int copy_threads) {
    Impl& I = *im_;
    I.cx = &cx;
    I.cq = sycl::queue(Q().get_context(), Q().get_device(), sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    I.comp = &Q();
    I.max_tokens = std::max(1, max_tokens);
    I.copy_threads = std::max(1, copy_threads);
    I.slot_bytes = (stage_bytes / 3) & ~(size_t)4095;
    if (I.slot_bytes < (8u << 20)) {
        err = "stage_bytes too small";
        return false;
    }
    // every expert must use a supported code
    for (int L = 0; L < 41; ++L)
        for (int e = 0; e < 512; ++e) {
            const uint8_t* body;
            int gu, dn;
            uint32_t dno, bytes;
            if (!cx.expert_info(L, e, &body, &gu, &dn, &dno, &bytes)) continue;
            auto okc = [](int c) { return c == 0 || c == 2 || c == 3 || c == 4; };
            if (!okc(gu) || !okc(dn)) {
                err = "expert code not supported by the GPU path (only S1.25 / S2_K / S3_K / S4_K): layer " + std::to_string(L) + " expert " + std::to_string(e);
                return false;
            }
            if (bytes > I.slot_bytes) {
                err = "expert body larger than a ring slot";
                return false;
            }
        }
    const size_t maxP = (size_t)I.max_tokens * NTOP;
    for (int i = 0; i < 3; ++i) {
        I.h_slot[i] = sycl::aligned_alloc_host<uint8_t>(4096, I.slot_bytes, I.cq);
        I.h_items[i] = sycl::malloc_host<Item>(Impl::MAXITEMS, I.cq);
        if (!I.h_slot[i] || !I.h_items[i]) {
            err = "pinned host allocation failed";
            return false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        I.d_slot[i] = I.dev<uint8_t>(I.slot_bytes);
        I.d_items[i] = I.dev<Item>(Impl::MAXITEMS);
    }
    I.zg = I.dev<half>((maxP + 64) * K);
    I.hb = I.dev<half>((maxP + 64) * K);
    I.ye = I.dev<float>(maxP * K);
    I.d_ptok = I.dev<int32_t>(maxP);
    I.d_red = I.dev<int32_t>(maxP);
    I.d_pw = I.dev<float>(maxP);
    I.h_ptok = sycl::malloc_host<int32_t>(maxP, I.cq);
    I.h_red = sycl::malloc_host<int32_t>(maxP, I.cq);
    I.h_pw = sycl::malloc_host<float>(maxP, I.cq);
    I.cq.memset(I.zg, 0, (maxP + 64) * K * sizeof(half));
    I.cq.memset(I.hb, 0, (maxP + 64) * K * sizeof(half));
    I.cq.wait();
    I.cnt.assign(512, 0);
    I.st.assign(513, 0);
    I.cur.assign(512, 0);
    I.pair_tok.assign(maxP, 0);
    I.pair_w.assign(maxP, 0.0f);
    I.pos.assign(maxP, -1);
    I.inited = true;
    return true;
}

void GpuExperts::set_profile_queue(sycl::queue* compute) {
    im_->comp = compute ? compute : &Q();
    im_->profile = compute != nullptr;
}

size_t GpuExperts::device_bytes() const { return im_->dev_bytes; }
const GpuExpertStats& GpuExperts::stats() const { return im_->stats; }
void GpuExperts::reset_stats() { im_->stats = GpuExpertStats(); }

void GpuExperts::wait_all() {
    Impl& I = *im_;
    I.cq.wait_and_throw();
    I.comp->wait_and_throw();
    for (auto& pe : I.prof) {
        const double ms = (double)(pe.first.get_profiling_info<sycl::info::event_profiling::command_end>() - pe.first.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
        if (pe.second == 0) I.stats.gpu_gemm1_ms += ms;
        else if (pe.second == 1) I.stats.gpu_gemm2_ms += ms;
        else if (pe.second == 3) I.stats.gpu_h2d_ms += ms;
        else I.stats.gpu_other_ms += ms;
    }
    I.prof.clear();
}

void GpuExperts::run(int L, int T, const float* z_dev, const int32_t* idx, const float* w, float* zout_dev, const uint8_t* only) {
    Impl& I = *im_;
    if (T > I.max_tokens) {
        for (int t0 = 0; t0 < T; t0 += I.max_tokens) {
            const int n = std::min(I.max_tokens, T - t0);
            run(L, n, z_dev + (size_t)t0 * K, idx + (size_t)t0 * NTOP, w + (size_t)t0 * NTOP, zout_dev + (size_t)t0 * K, only);
        }
        return;
    }
    const double t_start = now_s();
    sycl::queue& comp = *I.comp;
    if (I.last_valid) { const double tw0 = now_s(); I.last_ev.wait(); I.stats.t_wait_prev += now_s() - tw0; }  // the pinned planning arrays are reused below
    // ---- plan: counting sort of the (token, slot) pairs by expert id
    const CpuExperts& cx = *I.cx;
    struct EInfo { const uint8_t* body; int gu, dn; uint32_t dn_off, bytes; };
    std::fill(I.cnt.begin(), I.cnt.end(), 0);
    std::vector<EInfo> einfo(512);
    std::vector<char> okv(512, 0);
    for (int e = 0; e < 512; ++e) {
        EInfo& x = einfo[e];
        okv[e] = cx.expert_info(L, e, &x.body, &x.gu, &x.dn, &x.dn_off, &x.bytes) && (!only || only[e]);
    }
    const int npair = T * NTOP;
    for (int i = 0; i < npair; ++i) {
        const int e = idx[i];
        if ((unsigned)e < 512 && okv[e]) I.cnt[e]++;
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
    const int P = acc;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < NTOP; ++k) {
            const int e = idx[t * NTOP + k];
            if ((unsigned)e < 512 && okv[e]) {
                const int p = I.cur[e]++;
                I.pair_tok[p] = t;
                I.pair_w[p] = w[t * NTOP + k];
                I.pos[t * NTOP + k] = p;
            } else {
                I.pos[t * NTOP + k] = -1;
            }
        }
    // reduce plan: per token the pair indices in increasing expert id
    for (int t = 0; t < T; ++t) {
        int ks[NTOP], n = 0;
        for (int k = 0; k < NTOP; ++k)
            if (I.pos[t * NTOP + k] >= 0) {
                int j = n++;
                while (j > 0 && idx[t * NTOP + ks[j - 1]] > idx[t * NTOP + k]) {
                    ks[j] = ks[j - 1];
                    --j;
                }
                ks[j] = k;
            }
        for (int a = 0; a < NTOP; ++a) I.h_red[t * NTOP + a] = a < n ? I.pos[t * NTOP + ks[a]] : -1;
    }
    std::memcpy(I.h_ptok, I.pair_tok.data(), (size_t)std::max(P, 1) * sizeof(int32_t));
    std::memcpy(I.h_pw, I.pair_w.data(), (size_t)std::max(P, 1) * sizeof(float));
    sycl::event e1 = comp.memcpy(I.d_red, I.h_red, (size_t)T * NTOP * sizeof(int32_t));
    sycl::event e2 = comp.memcpy(I.d_ptok, I.h_ptok, (size_t)std::max(P, 1) * sizeof(int32_t));
    sycl::event e3 = comp.memcpy(I.d_pw, I.h_pw, (size_t)std::max(P, 1) * sizeof(float));
    if (I.profile) { I.prof.push_back({e1, 2}); I.prof.push_back({e2, 2}); I.prof.push_back({e3, 2}); }
    if (P > 0) {
        sycl::event eg = launch_gather(comp, z_dev, I.d_ptok, I.zg, P);
        if (I.profile) I.prof.push_back({eg, 2});
    }
    I.stats.t_plan += now_s() - t_start;

    // ---- groups of experts that fit a ring slot
    std::vector<int> grp_begin;  // indices into act_e
    {
        size_t used = 0;
        grp_begin.push_back(0);
        for (size_t i = 0; i < I.act_e.size(); ++i) {
            const size_t b = einfo[I.act_e[i]].bytes;
            if (used + b > I.slot_bytes) {
                grp_begin.push_back((int)i);
                used = 0;
            }
            used += b;
        }
        grp_begin.push_back((int)I.act_e.size());
    }
    const int ng = I.act_e.empty() ? 0 : (int)grp_begin.size() - 1;
    for (int g = 0; g < ng; ++g) {
        const int hs = I.group_ctr % 3, ds = I.group_ctr % 2;
        I.group_ctr++;
        double tw = now_s();
        if (I.cp_valid[hs]) I.cp_ev[hs].wait();  // the pinned slot / item array must be free
        I.stats.t_wait_host += now_s() - tw;
        // lay out the group: bodies 4 KiB aligned (body_bytes already are), items per 64-token tile
        const int a = grp_begin[g], b = grp_begin[g + 1];
        std::vector<uint32_t> off(b - a);
        size_t used = 0;
        int nitems = 0;
        for (int i = a; i < b; ++i) {
            const int e = I.act_e[i];
            off[i - a] = (uint32_t)used;
            for (int tt = 0; tt < I.cnt[e]; tt += 64) {
                Item& it = I.h_items[hs][nitems++];
                it.body_off = (uint32_t)used;
                it.dn_off = einfo[e].dn_off;
                it.pbase = (uint32_t)(I.st[e] + tt);
                it.ntok = (uint32_t)std::min(64, I.cnt[e] - tt);
                it.gu = einfo[e].gu;
                it.dn = einfo[e].dn;
                it.pad0 = it.pad1 = 0;
            }
            used += einfo[e].bytes;
        }
        if (nitems > Impl::MAXITEMS) {
            fprintf(stderr, "gpuexp: too many items in a group (%d)\n", nitems);
            exit(3);
        }
        double tc = now_s();
#pragma omp parallel for num_threads(I.copy_threads) schedule(dynamic, 1)
        for (int i = a; i < b; ++i) cx.copy_body(einfo[I.act_e[i]].body, einfo[I.act_e[i]].bytes, I.h_slot[hs] + off[i - a]);
        I.stats.t_copy_host += now_s() - tc;
        // copy queue: wait for the compute that last used this device slot, then items + bodies
        sycl::event ci = I.cq.submit([&](sycl::handler& h) {
            if (I.comp_valid[ds]) h.depends_on(I.comp_ev[ds]);
            h.memcpy(I.d_items[ds], I.h_items[hs], (size_t)nitems * sizeof(Item));
        });
        sycl::event cb = I.cq.memcpy(I.d_slot[ds], I.h_slot[hs], used);
        I.cp_ev[hs] = cb;
        I.cp_valid[hs] = true;
        // compute queue
        sycl::event k1 = launch_gemm<0>(comp, &cb, I.d_slot[ds], I.d_items[ds], nitems, I.zg, I.hb);
        sycl::event k2 = launch_gemm<1>(comp, nullptr, I.d_slot[ds], I.d_items[ds], nitems, I.hb, I.ye);
        I.comp_ev[ds] = k2;
        I.comp_valid[ds] = true;
        if (I.profile) {
            I.prof.push_back({ci, 3});
            I.prof.push_back({cb, 3});
            I.prof.push_back({k1, 0});
            I.prof.push_back({k2, 1});
        }
        I.stats.bytes += used;
        I.stats.groups++;
        I.stats.experts += b - a;
    }
    sycl::event er = launch_reduce(comp, I.ye, I.d_red, I.d_pw, zout_dev, T);
    if (I.profile) I.prof.push_back({er, 2});
    I.last_ev = er;
    I.last_valid = true;
    I.stats.pairs += P;
    I.stats.calls++;
    I.stats.seconds += now_s() - t_start;
}

}  // namespace hy4s
