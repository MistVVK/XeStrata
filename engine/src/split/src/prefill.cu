// fnx prompt path: a prompt's rows in chunks of up to FNX_PF_ROWS (default 4096), as llama.cpp's CUDA backend runs a batch on a V100: the
// dense matrices dequantized to f16 and multiplied with cuBLAS on the tensor cores (f16 x f16, fp32 accumulation; FNX_PF_F16ACC=1
// accumulates in f16 as llama.cpp's COMPUTE_16F does), the f32 router and the bf16 indexer projections as f32 sgemm, the
// routed experts through the decode kernels' arithmetic in jobs of up to PF_JOB rows per expert. The recurrent core and the attention
// run per group of MAXT rows through the decode kernels (each group with its own position and tag, so the recurrent
// history's parity steps as it does window by window). The sums across the cards use a ring sized for a chunk.
// What the chunk leaves is the state a run of windows over the same rows leaves: K/V, recurrent states and
// histories, the PLE history, the QSA keys, and with the draft layer attached its K/V and QSA keys too (the draft layer
// over the chunk's final residuals in windows of MAXT rows).
#include "pf_common.cuh"
#include "engine.hpp"
#include "engine_impl.hpp"
#include "env.hpp"
#include "pf_ring_plan.hpp"
#if FNX_HIP
#include "hip/rccl.h"
#endif

#include <cublas_v2.h>
#if !FNX_HIP
#include <cublasLt.h>
#endif
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fnx {

namespace {

constexpr size_t PF_WH = (size_t) 16 << 20;                      // halves: a dense matrix's f16 staging (larger ones go in slices)

// buffers carved from a block (aligned to 256 bytes); with base null it only sums the sizes
struct Arena {
    char* base;
    size_t size, off = 0;
    bool fits(size_t bytes) const { return ((off + 255) & ~(size_t) 255) + bytes <= size; }
    template <typename T> T* take(size_t n) {
        off = (off + 255) & ~(size_t) 255;
        if (off + n * sizeof(T) > size) throw std::runtime_error("prefill: an arena overflowed");
        T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
        off += n * sizeof(T);
        return p;
    }
};

__device__ __forceinline__ float pf_sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float pf_silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float pf_wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float pf_wmax(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float pf_block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = pf_wsum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float r = 0.f;
    for (int w = 0; w < nw; ++w) r += sh[w];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------- conversions
__global__ void k_f16_f32(const __half* __restrict__ x, int rows, int cols, float* __restrict__ y, int ldy) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) rows * cols; e += (size_t) gridDim.x * blockDim.x)
        y[(e / cols) * ldy + e % cols] = __half2float(x[e]);
}
__global__ void k_bf16_f32(const uint16_t* __restrict__ x, size_t n, float* __restrict__ y) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        y[e] = __uint_as_float((unsigned) x[e] << 16);
}

// ---------------------------------------------------------------- hyper-connection read, batched
// row t: R' = R + bo * 2 sigmoid(inj / HC) per stream (written back), rs[t][c] = 1 / sqrt(mean R'^2 + eps), and
// xn = R' * w_norm * rs as f16 (the down mix's input); with w_inj (bf16 rows of D) also inject_out[t][r] = w_inj[r] . xn
// from the same f32 xn while it is in registers (k_hcb_mixinj then needs no second pass over R)
constexpr int HCB_THREADS = 256, HCB_PER = N / HCB_THREADS;
static_assert(N % HCB_THREADS == 0, "stream elements per thread");
__global__ void __launch_bounds__(HCB_THREADS) k_hcb_prep(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                                          const float* __restrict__ wn, float* __restrict__ rs, __half* __restrict__ xh,
                                                          float eps, const uint16_t* __restrict__ w_inj, float* __restrict__ inject_out) {
    const int t = blockIdx.x, i0 = threadIdx.x;
    __shared__ float sh[32];
    __shared__ float shi[HC][HCB_THREADS / 32];
    float acc[HC] = {};
    for (int c = 0; c < HC; ++c) {
        float v[HCB_PER];
        float ss = 0.f;
        const float g = bo ? 2.0f * pf_sigm(inj[t * HC + c] / (float) HC) : 0.f;
#pragma unroll
        for (int k = 0; k < HCB_PER; ++k) {
            const int i = i0 + k * HCB_THREADS;
            float r = R[(size_t) t * D + c * N + i];
            if (bo) {
                r = fmaf(bo[(size_t) t * N + i], g, r);
                R[(size_t) t * D + c * N + i] = r;
            }
            v[k] = r;
            ss += r * r;
        }
        const float r_s = rsqrtf(pf_block_sum(ss, sh) / (float) N + eps);
        if (i0 == 0) rs[t * HC + c] = r_s;
#pragma unroll
        for (int k = 0; k < HCB_PER; ++k) {
            const int i = i0 + k * HCB_THREADS;
            const float xn = v[k] * wn[c * N + i] * r_s;
            xh[(size_t) t * D + c * N + i] = __float2half(xn);
            if (w_inj)
#pragma unroll
                for (int r = 0; r < HC; ++r) acc[r] = fmaf(__uint_as_float((unsigned) w_inj[(size_t) r * D + c * N + i] << 16), xn, acc[r]);
        }
    }
    if (!w_inj) return;
    const int lane = i0 & 31, warp = i0 >> 5;
#pragma unroll
    for (int r = 0; r < HC; ++r) {
        const float s = pf_wsum(acc[r]);
        if (lane == 0) shi[r][warp] = s;
    }
    __syncthreads();
    if (i0 < HC) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < HCB_THREADS / 32; ++w) s += shi[i0][w];
        inject_out[t * HC + i0] = s;
    }
}
// lo = silu(down / HC) as f16
__global__ void k_hcb_lo(const float* __restrict__ down, int n, __half* __restrict__ lo) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) lo[e] = __float2half(pf_silu(down[e] / (float) HC));
}
// per row t: mixed[t][i] = mean_c xn_c[i] * sigmoid(up_c[i]), then (with w_inj: bf16 rows of D) inject[t][r] =
// w_inj[r] . xn[t]: warps 4 r .. 4 r + 3 take a quarter of D each, their sums added in order. xn = R wn rs,
// recomputed with k_hcb_prep's operations (so its values) rather than kept as a copy
constexpr int HCM_THREADS = 512, HCM_WPR = HCM_THREADS / 32 / HC;   // warps per inject row
__global__ void __launch_bounds__(HCM_THREADS) k_hcb_mixinj(const float* __restrict__ R, const float* __restrict__ wn,
                                                            const float* __restrict__ rs, const float* __restrict__ up,
                                                            float* __restrict__ mixed, __half* __restrict__ mixh,
                                                            const uint16_t* __restrict__ w_inj,
                                                            float* __restrict__ inj) {
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* Rt = R + (size_t) t * D;
    float r_s[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) r_s[c] = rs[t * HC + c];
    for (int i = tid; i < N; i += HCM_THREADS) {
        float g[HC];
#pragma unroll
        for (int c = 0; c < HC; ++c) g[c] = pf_sigm(up[(size_t) t * D + c * N + i]) * (Rt[c * N + i] * wn[c * N + i] * r_s[c]);
        const float m = (g[0] + g[1] + g[2] + g[3]) / (float) HC;
        mixed[(size_t) t * N + i] = m;
        mixh[(size_t) t * N + i] = __float2half(m);
    }
    if (!w_inj) return;
    __shared__ float part[HC][HCM_WPR];
    const int r = warp / HCM_WPR, q = warp % HCM_WPR;
    constexpr int QLEN = D / HCM_WPR;
    float acc = 0.f;
    for (int i = q * QLEN + lane; i < (q + 1) * QLEN; i += 32)
        acc += __uint_as_float((unsigned) w_inj[(size_t) r * D + i] << 16) * (Rt[i] * wn[i] * r_s[i / N]);
    acc = pf_wsum(acc);
    if (lane == 0) part[r][q] = acc;
    __syncthreads();
    if (tid < HC) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < HCM_WPR; ++k) s += part[tid][k];
        inj[t * HC + tid] = s;
    }
}

// ---------------------------------------------------------------- routing
// row t (a warp): top 10 of the NEXP logits (the lower index on a tie), weights the selected softmax renormalized
__global__ void k_pf_top10(const float* __restrict__ logits, int C, int* __restrict__ topi, float* __restrict__ topw) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= C) return;
    float v[NEXP / 32];
#pragma unroll
    for (int i = 0; i < NEXP / 32; ++i) v[i] = logits[(size_t) t * (NEXP + 1) + lane + 32 * i];
    unsigned taken = 0;
    float top0 = 0.f, z = 0.f, wk = 0.f;
#pragma unroll 1
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = NEXP;
#pragma unroll
        for (int i = 0; i < NEXP / 32; ++i)
            if (!(taken >> i & 1u) && v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
        if (k == 0) top0 = bv;
        const float ek = __expf(bv - top0);
        z += ek;
        if (lane == k) {
            wk = ek;
            topi[t * TOPK + k] = bi;
        }
    }
    if (lane < TOPK) topw[t * TOPK + lane] = wk / z;
}
__global__ void k_pf_count(const int* __restrict__ topi, int P, int* __restrict__ cnt) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) atomicAdd(&cnt[topi[p]], 1);
}
// one block of NEXP threads: each expert's first job (ceil(count / PF_JOB) jobs each), the job count
__global__ void k_pf_jobbase(const int* __restrict__ cnt, int* __restrict__ jbase, int* __restrict__ njobs) {
    __shared__ int s[NEXP];
    const int e = threadIdx.x;
    const int nj = (cnt[e] + PF_JOB - 1) / PF_JOB;
    s[e] = nj;
    __syncthreads();
    for (int o = 1; o < NEXP; o <<= 1) {
        const int a = e >= o ? s[e - o] : 0;
        __syncthreads();
        s[e] += a;
        __syncthreads();
    }
    jbase[e] = s[e] - nj;
    if (e == NEXP - 1) njobs[0] = s[e];
}
// pair p into the next slot of its expert's jobs (the order inside an expert does not matter: each pair's row is
// computed on its own)
// in index order, so a pair always lands in the same job: the expert kernel splits the k-range of small jobs, so which
// job a pair's row is in decides its summation order. A block per expert scans the pairs, a ballot prefix sum per pass
constexpr int PF_FILL_THREADS = 256;
__global__ void __launch_bounds__(PF_FILL_THREADS) k_pf_fill(const int* __restrict__ topi, int P, const int* __restrict__ cnt,
                                                             const int* __restrict__ jbase, BigJob* __restrict__ gu, BigJob* __restrict__ dn) {
    const int e = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (cnt[e] == 0) return;
    __shared__ int wsum[PF_FILL_THREADS / 32];
    int base = 0;
    for (int p0 = 0; p0 < P; p0 += PF_FILL_THREADS) {
        const int p = p0 + tid;
        const bool mine = p < P && topi[p] == e;
        const unsigned m = __ballot_sync(0xffffffffu, mine);
        if (lane == 0) wsum[warp] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
#pragma unroll
        for (int w = 0; w < PF_FILL_THREADS / 32; ++w) {
            before += w < warp ? wsum[w] : 0;
            total += wsum[w];
        }
        if (mine) {
            const int s = base + before + __popc(m & ((1u << lane) - 1u)), j = jbase[e] + s / PF_JOB, k = s % PF_JOB;
            gu[j].y[k] = p / TOPK;
            gu[j].out[k] = p;
            dn[j].y[k] = p;
            dn[j].out[k] = p;
            if (k == 0) {
                gu[j].w = dn[j].w = e;
                gu[j].n = dn[j].n = min(PF_JOB, cnt[e] - (s / PF_JOB) * PF_JOB);
            }
        }
        base += total;
        __syncthreads();
    }
}
// h = silu(gate) * up as f16 (the shared expert's down input)
__global__ void k_pf_swiglu(const float* __restrict__ gu, int ffs, int C, __half* __restrict__ h) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < C * ffs; e += gridDim.x * blockDim.x) {
        const int t = e / ffs, i = e % ffs;
        h[e] = __float2half(pf_silu(gu[(size_t) t * 2 * ffs + i]) * gu[(size_t) t * 2 * ffs + ffs + i]);
    }
}

}  // namespace

// ---------------------------------------------------------------- per-card buffers
// the chunk sums on the copy engines: f16 rows (parts of the chunk's rows, one a card), two sets by the sum's parity,
// a receive slot per reduce hop, flags [2 sets][PF_RING_STEPS] set by the previous card's stream; n*: the next card's;
// plan: this card's steps (pf_ring_plan.hpp)
struct Engine::PfRing {
    __half *send, *in[NCARD - 1][2], *ag[2];   // in, ag: host memory, written by the previous card
    __half *din[NCARD - 1][2], *dag[2];        // this card's own copies in device memory (the add and the output read these)
    __half *nin[NCARD - 1][2], *nag[2];
    unsigned *flags, *nflags;
    __half* pag[NCARD][2];   // every card's gathered rows and flags (the direct gather writes to all of them)
    unsigned* pflags[NCARD];
    int next_dev = 0;
    bool direct = false;     // the plan is pf_ring_plan_direct
    cudaEvent_t fence_ev = nullptr;   // FNX_PF_DMA_BARRIER=1
    bool send_kernel = false;  // FNX_PF_SEND=kernel
    bool dma_check = false;  // FNX_PF_DMA_CHECK=1: compare each DMA copy with its host source by a kernel
    bool dma = true;         // receive and gather by DMA into device memory (FNX_PF_DMA=0: kernels read host memory)
    std::vector<PfRingOp> plan;
};

constexpr int ATT_PIECES_MAX = 8;   // the attention exchange's pieces at most (FNX_PF_ATT_PIECES)
// rows of a card's attention-exchange flags (aflags [row][piece], row k set by card + k + 1): one per other card, with
// a floor of two, so two cards' flag allocation is three cards'. A flag count only: the receive blocks (arecv) and
// every loop over the senders take NCARD - 1
constexpr int ATT_FLAG_ROWS = NCARD - 1 > 2 ? NCARD - 1 : 2;

struct Engine::Pf {
    cublasHandle_t blas = nullptr;
    __half* wh;                       // a dense matrix as f16
    float* widx;                      // the indexer projections as f32 [640][N]
    __half* xh;                       // f16 activations, up to a chunk's rows of D
    float *R, *xf, *rs, *lo32, *up32, *mixed, *bo_a, *bo_f, *inj_a, *inj_f;
    __half* lo16;
    void* xq;                         // mixed as q8_1 (with a twin), for the experts through the decode kernels
    uint16_t* h16;                    // the experts' swiglu rows as f16, for the tensor-core down product
    bool moe_q8 = false;              // FNX_PF_MOE_Q8=1: the experts through the decode kernels' q8_1 arithmetic
    float *logits, *topw, *gu, *part, *shg;
    __half* shh;
    int *topi, *cnt, *jbase, *njobs;
    BigJob *jobs_gu, *jobs_dn;
    float *qkvz, *ab, *y32;
    float *gdn_xc, *gdn_sq;           // the recurrent core's conv outputs (rows of ch) and output square sums
    float* gdn_ch;                    // its sub-chunks' state-free parts (gdn_prompt_work_floats)
    // the chunk sums on the copy engines (FNX_PF_RING_SM=1: the SM ring instead): f16 rows in parts of the chunk,
    // two sets by the sum's parity; flags [2 sets][PF_RING_STEPS], set by the previous card's stream (seq: the sum's number)
    bool ring_sm = false;
    PfRing ring_ce;                   // the chunk sums on the copy engines
    // FNX_PF_RCCL=1 (AMD): the chunk sums through RCCL in place on ring_ce.dag[set] (rc_api: RcclApi*, rc_comm: this card's communicator). RCCL launches a group's collectives
    // at ncclGroupEnd, so what must follow a sum on its stream (the f32 conversion, the `done` event) waits in post until the prompt loop's per-card pass is closed
    bool rccl = false;
    bool rc_f32 = false;              // FNX_PF_RCCL_F32=1: the sums in f32
    const void* rc_api = nullptr;
    void* rc_comm = nullptr;
    std::vector<std::function<void()>> post;
    int dev = 0;
    unsigned seq = 0;
#if FNX_HIP
    // FNX_PF_PEER small-chunk sum (pf_common.cuh k_pfa_*): this card's mailbox [2 sets][NCARD senders][a2a_rows * N] f32 and flags [NCARD], the peers' pointers, the push kernel's counter
    int a2a_rows = 0;
    float4* a2a_box = nullptr;
    unsigned* a2a_flag = nullptr;
    unsigned* a2a_cnt = nullptr;
    float4** a2a_peer_box_d = nullptr;     // device arrays of the peers' pointers (kernel arguments)
    unsigned** a2a_peer_flag_d = nullptr;
#endif
    // attention partials (attn_prompt's blocks per owner card) and those received from the other cards; each card's
    // receive slots and flags (slot k: from card + k + 1). FNX_PF_ATTN_DEC=1: the decode attention
    bool attn_dec = false;
    // two sets by the chunk's parity: with the overlap a chunk's merge waits until the next chunk's hc read is issued
    // (its exchange meanwhile on the links), so the next chunk's exchange must not land in the same buffers
    float *apart[2] = {}, *arecv[2] = {};
    unsigned* aflags[2] = {};         // [slot][piece]
    // the exchange in att_pieces pieces of the chunk's rows (FNX_PF_ATT_PIECES): a piece's partials go while the next
    // piece is computed, so only the last one's copy waits; ev_piece: a piece computed, ev_j: a set's sends on st2 done
    int att_pieces = 4;
    cudaEvent_t ev_piece[ATT_PIECES_MAX], ev_j[2];
    unsigned att_seq[2] = {};         // a set's exchange's sequence number
    int att_pend = -1, att_pend_rows = 0;   // the chunk whose merge waits (-1: none), its rows
    float* parecv[2][NCARD];
    unsigned* paflags[2][NCARD];
    // FNX_PFX_ATTN_DMA (default on): the other cards' partials are copied from the host receive buffer into device memory by the copy engine on st3, piece by
    // piece as their flags arrive, so the merge reads device memory (reading the mapped host buffer from the merge kernel stalls its SMs at PCIe speed)
    bool attn_dma = false;
    float* arecv_dev[2] = {};
    cudaStream_t st3 = nullptr;
    cudaEvent_t ev_r[2] = {}, ev_m[2] = {};   // ev_r: a set's copies done; ev_m: its merge done (the next copies into the buffer wait for it)
    int pdev[NCARD];
    float *qg, *k, *v, *att, *yidx, *qidx;
    int *cells, *ncells;
    float* scores;                    // QSA scores of a group's rows
    int qsa_pf_rows = 256;            // a group's rows for QSA's block selection (FNX_PF_QSA_ROWS, a multiple of MAXT)
    float *ple_emb, *ple_key, *ple_val, *ple_tmp;
    int* ctl;                         // [1 + chunk rows / MAXT][2]: the chunk's (position, tag), then each group's
    Ring ring;
    bool f16acc = false;
    // a pass's chunks: what a chunk carries from one layer, or one half, to the next (R, ctl, bo_f, inj_f, bo_a, inj_a
    // above point at the chunk's while it runs; [0] are those)
    int pass = 1;                     // chunks per pass: FNX_PF_PASS (default 2; 4 with an ExpertCache)
    float *Rk[Engine::PF_PASS], *bofk[Engine::PF_PASS], *injfk[Engine::PF_PASS], *boak[Engine::PF_PASS],
        *injak[Engine::PF_PASS];
    int* ctlk[Engine::PF_PASS];
    // FNX_PF_OVERLAP (default on): each chunk's sums go round the ring on ring_st while the next chunk of the pass computes;
    // ev_go: a half's output ready for its sum, ev_ra / ev_rf: its attention / FFN sum done
    bool overlap = false;
    cudaStream_t ring_st = nullptr;
    cudaEvent_t ev_go[Engine::PF_PASS], ev_ra[Engine::PF_PASS], ev_rf[Engine::PF_PASS];
    // FNX_PF_HCSPLIT: the hyper-connection read of a chunk is done on one card's part of the rows each (the part its ring sum finishes),
    // and the cards swap the resulting `mixed` rows (f32) through host memory on the copy engines, in hg_pieces pieces. A chunk keeps its own
    // `mixed` (mixedk): the reads of the pass's chunks come before their heavy halves. hg_in: the landing place of the other card's rows (two
    // sets by the swap's number), hg_flags[set][piece] set by the other card's stream (the swap's number); rg_*: the same for the residual
    // streams R at the end of the pass (the draft layer reads every row of R).
    bool down_lt = false;                // the cublasLt algorithm was found (NVIDIA only; HIP never sets it, but the hyper-connection checks read it)
#if !FNX_HIP
    // the hyper-connection down GEMM (320 x 10240, K = 10240) through cublasLt with ONE algorithm whose rows do not depend on the number of rows
    // (found at start by pf_lt_calibrate: equal to cublasGemmEx's choice at 4096 rows, and the same rows at 2048); FNX_PF_HCSPLIT only. cublasGemmEx
    // picks split-K 5 at 4096 rows and a k16 kernel at 2048, which rounds differently
    struct LtShape {
        int n = 0;
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
        bool ok = false;
    };
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulAlgo_t down_algo;
    void* lt_ws = nullptr;
    size_t lt_ws_bytes = 0;
    std::vector<LtShape> lt_shapes;
#endif
    bool hcs = false;
    int hcs_mode = 0;                    // FNX_PF_HCSPLIT=1: only passes of whole 4096-row chunks are split (bit-identical to the unsplit path when the calibration found an algorithm); 2: any chunk of 512 rows or more (the last bits differ)
    int hg_pieces = 4;
    float* mixedk[Engine::PF_PASS] = {};
    float* hg_in[2] = {};
    unsigned* hg_flags[2] = {};
    float* nhg_in[2] = {};
    unsigned* nhg_flags[2] = {};
    float* rg_in[2] = {};
    unsigned* rg_flags = nullptr;   // [set][8]
    float* nrg_in[2] = {};
    unsigned* nrg_flags = nullptr;
    unsigned hg_seq = 0, rg_seq = 0;
    cudaEvent_t ev_hc[Engine::PF_PASS] = {}, ev_ag[Engine::PF_PASS] = {}, ev_rg[Engine::PF_PASS] = {};
    cudaStream_t ag_st = nullptr;        // the swaps' stream (not ring_st: a swap must not queue behind a sum that waits for the other card)
    bool hf16 = false;                   // FNX_PF_HCSPLIT_F16=1: the rows go as f16 (half the bytes); both cards then use f16-rounded `mixed` for every row
    __half* mixhk[Engine::PF_PASS] = {};
    // with an ExpertCache: a pass's expert tables (device, [NL][NEXP]; pinned copies to build them in) and the copies
    // into stage[l & 1] (on cs) of layer l's experts not in a slot, as many as fit; the rest are read in place
    char* stage[2] = {};
    size_t stage_bytes = 0;
    size_t alloc_bytes = 0;           // what the prompt path holds on this card (palloc's tally; the stages added later)
    int stage_rows = 0;               // FNX_PF_STAGE_ROWS (default 2048): shorter reads take the experts in place
    const char **ep_gu = nullptr, **ep_dn = nullptr, **ep_hgu = nullptr, **ep_hdn = nullptr;
    int* slot_h = nullptr;            // pinned: the ExpertCache's slot_of at the pass's start
    struct Copy {
        const char* src;
        char* dst;
        size_t n;
    };
    std::vector<Copy> copies[NL];
    size_t staged = 0, in_place = 0;  // the pass's expert bytes copied in, and left in RAM
    cudaStream_t cs = nullptr;
    cudaEvent_t ev_ready[2], ev_done[2];   // stage[k] filled; its layer's experts done with
};

namespace {
// out[t * ldo + r] = W[r] . X[t] for t < C: W f16 [rows][cols], X f16 [C][cols]
void pf_gemm(Engine::Pf& p, const __half* W, int rows, int cols, const __half* X, int C, float* out, int ldo, cudaStream_t s) {
    if (std::getenv("FNX_PF_GEMM_LOG")) {   // each shape once (for tools/gemm_bench.cu)
        static std::set<std::tuple<int, int, int, int>> seen;
        if (seen.insert({rows, cols, C, ldo}).second) std::fprintf(stderr, "pf_gemm: rows %d cols %d C %d ldo %d\n", rows, cols, C, ldo);
    }
    if (p.f16acc) {
        // f16 accumulation and output (llama.cpp's COMPUTE_16F on a V100), then to f32
        const __half one = __float2half(1.0f), zero = __float2half(0.0f);
        __half* tmp = reinterpret_cast<__half*>(p.up32);   // up32 is free outside the hc read
        ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, tmp,
                         CUDA_R_16F, rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm f16");
        k_f16_f32<<<grid_for((size_t) rows * C), 256, 0, s>>>(tmp, C, rows, out, ldo);
    } else {
        const float one = 1.0f, zero = 0.0f;
        ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, out,
                         CUDA_R_32F, ldo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm");
    }
}
// out = W . xh for a split-layout Q8_0 matrix (dequantized to f16), xh: C rows of cols as f16 already in p.xh
void pf_dense_xh(Engine::Pf& p, const Q8S& w, int rows, int cols, int C, float* out, int ldo, cudaStream_t s) {
    if (!w.q) throw std::runtime_error("prefill: a dense matrix without the split Q8_0 layout");
    if (cols % 32 != 0 || (size_t) cols > PF_WH) throw std::runtime_error("prefill: a dense matrix's rows don't fit the f16 staging");
    // in slices of rows that fit the staging buffer (each slice: its rows of out)
    const int slice = (int) std::min<size_t>(rows, PF_WH / cols / 128 * 128);   // whole 128-row tiles
    for (int r0 = 0; r0 < rows; r0 += slice) {
        const int nr = std::min(slice, rows - r0);
        k_q8s_f16<<<grid_for((size_t) nr * cols), 256, 0, s>>>(w.q + (size_t) r0 * cols, w.d + (size_t) r0 * cols / 32, (size_t) nr * cols, p.wh);
        pf_gemm(p, p.wh, nr, cols, p.xh, C, out + r0, ldo, s);
    }
}
// the same from x (f32, C rows of cols), converted into p.xh first
void pf_dense(Engine::Pf& p, const Q8S& w, int rows, int cols, const float* x, int C, float* out, int ldo, cudaStream_t s) {
    k_f32_f16<<<grid_for((size_t) C * cols), 256, 0, s>>>(x, (size_t) C * cols, p.xh);
    pf_dense_xh(p, w, rows, cols, C, out, ldo, s);
}
// out = W . x, f32 weights [rows][cols] and activations
void pf_sgemm(Engine::Pf& p, const float* W, int rows, int cols, const float* x, int C, float* out, int ldo) {
    const float one = 1.0f, zero = 0.0f;
    ckb(cublasSgemm(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, cols, x, cols, &zero, out, ldo), "sgemm");
}
#if !FNX_HIP
// the Lt problem for n rows of the down GEMM (cached): out[t][r] = W[r] . X[t], W f16 [LR][D], X f16 [n][D], out f32 [n][LR]
Engine::Pf::LtShape& pf_lt_shape(Engine::Pf& p, int n) {
    for (Engine::Pf::LtShape& s : p.lt_shapes)
        if (s.n == n) return s;
    p.lt_shapes.emplace_back();
    Engine::Pf::LtShape& s = p.lt_shapes.back();
    s.n = n;
    cublasLtMatmulDescCreate(&s.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(s.desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
    cublasLtMatmulDescSetAttribute(s.desc, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn);
    cublasLtMatrixLayoutCreate(&s.a, CUDA_R_16F, D, LR, D);
    cublasLtMatrixLayoutCreate(&s.b, CUDA_R_16F, D, n, D);
    cublasLtMatrixLayoutCreate(&s.c, CUDA_R_32F, LR, n, LR);
    cublasLtMatmulHeuristicResult_t chk;
    s.ok = cublasLtMatmulAlgoCheck(p.lt, s.desc, s.a, s.b, s.c, s.c, &p.down_algo, &chk) == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= p.lt_ws_bytes;
    return s;
}
// the down GEMM with the chosen algorithm (the caller checked p.down_lt); false: this shape does not take it (the plain call is made instead)
bool pf_down_lt(Engine::Pf& p, const __half* W, const __half* X, int n, float* out, cudaStream_t s) {
    if (n != 4096 && n != 2048) return false;   // the sizes the algorithm was calibrated at (the plain call picks its own for any other)
    Engine::Pf::LtShape& sh = pf_lt_shape(p, n);
    if (!sh.ok) return false;
    const float one = 1.0f, zero = 0.0f;
    ckb(cublasLtMatmul(p.lt, sh.desc, &one, W, sh.a, X, sh.b, &zero, out, sh.c, out, sh.c, &p.down_algo, p.lt_ws, p.lt_ws_bytes, s), "lt gemm");
    return true;
}
// at start: the algorithm for the down GEMM. Candidates come from cublasLt's heuristic at 4096 rows; one is taken when its rows equal cublasGemmEx's at 4096
// rows bit for bit, and its first and second half of the rows at 2048 equal its own at 4096. None: down_lt stays false (the split's results then differ in
// the last bits from the unsplit path's)
void pf_lt_calibrate(Engine::Pf& p, cudaStream_t s) {
    const int rows = LR, cols = D, C0 = 4096;
    p.lt_ws_bytes = 4 << 20;
    ckc(cudaMalloc(&p.lt_ws, p.lt_ws_bytes), "lt workspace");
    own_dev(p.lt_ws);
    ckb(cublasLtCreate(&p.lt), "lt create");
    {
        cublasLtHandle_t h = p.lt;
        const int d = current_card();
        own([d, h] { cudaSetDevice(d); cublasLtDestroy(h); });
    }
    std::vector<__half> hw((size_t) rows * cols), hx((size_t) C0 * cols);
    uint32_t st = 12345u;
    auto rnd = [&] { st = st * 1664525u + 1013904223u; return (float) ((st >> 9) & 0xFFFF) / 65536.f - 0.5f; };
    for (__half& v : hw) v = __float2half(rnd() * 0.2f);
    for (__half& v : hx) v = __float2half(rnd() * 6.f);
    __half *dw = nullptr, *dx = nullptr;
    float* dy = nullptr;
    ckc(cudaMalloc(&dw, hw.size() * 2), "calibrate");
    ckc(cudaMalloc(&dx, hx.size() * 2), "calibrate");
    ckc(cudaMalloc(&dy, (size_t) C0 * rows * 4), "calibrate");
    ckc(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice), "calibrate");
    ckc(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "calibrate");
    auto fetch = [&](size_t n) {
        std::vector<float> v(n);
        ckc(cudaMemcpy(v.data(), dy, n * 4, cudaMemcpyDeviceToHost), "calibrate");
        return v;
    };
    const float one = 1.0f, zero = 0.0f;
    ckc(cudaMemset(dy, 0xFF, (size_t) C0 * rows * 4), "calibrate");
    ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, C0, cols, &one, dw, CUDA_R_16F, cols, dx, CUDA_R_16F, cols, &zero, dy, CUDA_R_32F, rows,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "calibrate gemm");
    ckc(cudaStreamSynchronize(s), "calibrate");
    const std::vector<float> ref = fetch((size_t) C0 * rows);
    Engine::Pf::LtShape sh4, sh2;
    auto mk = [&](Engine::Pf::LtShape& sh, int n) {
        sh.n = n;
        cublasLtMatmulDescCreate(&sh.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        const cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(sh.desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
        cublasLtMatmulDescSetAttribute(sh.desc, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn);
        cublasLtMatrixLayoutCreate(&sh.a, CUDA_R_16F, cols, rows, cols);
        cublasLtMatrixLayoutCreate(&sh.b, CUDA_R_16F, cols, n, cols);
        cublasLtMatrixLayoutCreate(&sh.c, CUDA_R_32F, rows, n, rows);
    };
    mk(sh4, C0), mk(sh2, C0 / 2);
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.lt_ws_bytes, sizeof p.lt_ws_bytes);
    std::vector<cublasLtMatmulHeuristicResult_t> cand(32);
    int nc = 0;
    if (cublasLtMatmulAlgoGetHeuristic(p.lt, sh4.desc, sh4.a, sh4.b, sh4.c, sh4.c, pref, 32, cand.data(), &nc) != CUBLAS_STATUS_SUCCESS) nc = 0;
    for (int i = 0; i < nc && !p.down_lt; ++i) {
        cublasLtMatmulAlgo_t al = cand[(size_t) i].algo;
        cublasLtMatmulHeuristicResult_t chk;
        if (cand[(size_t) i].workspaceSize > p.lt_ws_bytes) continue;
        if (cublasLtMatmulAlgoCheck(p.lt, sh2.desc, sh2.a, sh2.b, sh2.c, sh2.c, &al, &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > p.lt_ws_bytes) continue;
        ckc(cudaMemset(dy, 0xFF, (size_t) C0 * rows * 4), "calibrate");
        if (cublasLtMatmul(p.lt, sh4.desc, &one, dw, sh4.a, dx, sh4.b, &zero, dy, sh4.c, dy, sh4.c, &al, p.lt_ws, p.lt_ws_bytes, s) != CUBLAS_STATUS_SUCCESS) continue;
        ckc(cudaStreamSynchronize(s), "calibrate");
        const std::vector<float> o4 = fetch((size_t) C0 * rows);
        if (std::memcmp(o4.data(), ref.data(), o4.size() * 4) != 0) continue;
        bool halves = true;
        for (int h = 0; h < 2 && halves; ++h) {
            ckc(cudaMemset(dy, 0xFF, (size_t) C0 * rows * 4), "calibrate");
            if (cublasLtMatmul(p.lt, sh2.desc, &one, dw, sh2.a, dx + (size_t) h * (C0 / 2) * cols, sh2.b, &zero, dy + (size_t) h * (C0 / 2) * rows, sh2.c,
                               dy + (size_t) h * (C0 / 2) * rows, sh2.c, &al, p.lt_ws, p.lt_ws_bytes, s) != CUBLAS_STATUS_SUCCESS) {
                halves = false;
                break;
            }
            ckc(cudaStreamSynchronize(s), "calibrate");
            const std::vector<float> o2 = fetch((size_t) C0 * rows);
            if (std::memcmp(o2.data() + (size_t) h * (C0 / 2) * rows, o4.data() + (size_t) h * (C0 / 2) * rows, (size_t) (C0 / 2) * rows * 4) != 0) halves = false;
        }
        if (!halves) continue;
        p.down_algo = al;
        p.down_lt = true;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    for (Engine::Pf::LtShape* sh : {&sh4, &sh2}) {
        cublasLtMatmulDescDestroy(sh->desc), cublasLtMatrixLayoutDestroy(sh->a), cublasLtMatrixLayoutDestroy(sh->b), cublasLtMatrixLayoutDestroy(sh->c);
    }
    cudaFree(dw), cudaFree(dx), cudaFree(dy);
    std::fprintf(stderr, "fnx: prompt path card %d: hyper-connection down GEMM %s\n", p.dev,
                 p.down_lt ? "through cublasLt with an algorithm that gives the same rows at 2048 and 4096 rows (and cublasGemmEx's at 4096)"
                           : "has no row-count-independent algorithm: the split's results differ in the last bits from the unsplit path's");
}
#endif

// the hyper-connection read for C rows (see hc.cuh for what it computes)
void pf_hc_read(Engine::Pf& p, float* R, const float* bo, const float* inj_prev, const float* w_norm, const Q8S& w_down, const Q8S& w_up,
                const uint16_t* w_inj, float eps, float* mixed, float* inject, int C, cudaStream_t s, __half* mixh = nullptr) {
    k_hcb_prep<<<C, HCB_THREADS, 0, s>>>(R, bo, inj_prev, w_norm, p.rs, p.xh, eps, w_inj, inject);   // the inject rows here
    k_q8s_f16<<<grid_for((size_t) LR * D), 256, 0, s>>>(w_down.q, w_down.d, (size_t) LR * D, p.wh);
#if !FNX_HIP
    if (!(p.down_lt && !p.f16acc && pf_down_lt(p, p.wh, p.xh, C, p.lo32, s)))
#endif
        pf_gemm(p, p.wh, LR, D, p.xh, C, p.lo32, LR, s);
    k_hcb_lo<<<grid_for((size_t) C * LR), 256, 0, s>>>(p.lo32, C * LR, p.lo16);
    k_q8s_f16<<<grid_for((size_t) D * LR), 256, 0, s>>>(w_up.q, w_up.d, (size_t) D * LR, p.wh);
    if (p.f16acc) {
        // pf_gemm's f16 path stages through up32: here up32 is the result, so stage through xh (free once lo is made)
        const __half one = __float2half(1.0f), zero = __float2half(0.0f);
        ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, D, C, LR, &one, p.wh, CUDA_R_16F, LR, p.lo16, CUDA_R_16F, LR, &zero, p.xh,
                         CUDA_R_16F, D, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm f16 (hc up)");
        k_f16_f32<<<grid_for((size_t) D * C), 256, 0, s>>>(p.xh, C, D, p.up32, D);
    } else {
        const float one = 1.0f, zero = 0.0f;
        ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, D, C, LR, &one, p.wh, CUDA_R_16F, LR, p.lo16, CUDA_R_16F, LR, &zero, p.up32,
                         CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "gemm (hc up)");
    }
    k_hcb_mixinj<<<C, HCM_THREADS, 0, s>>>(R, w_norm, p.rs, p.up32, mixed, mixh ? mixh : p.xh, nullptr, nullptr);   // prep made the inject rows
    // FNX_PF_HC_CHECK=1 (debugging): the inject rows the way k_hcb_mixinj made them, against prep's
    static const bool check = std::getenv("FNX_PF_HC_CHECK") != nullptr;
    if (check && w_inj) {
        float* ref = nullptr;
        float* scratch = nullptr;
        ckc(cudaMalloc(&ref, (size_t) C * HC * 4), "hc check");
        ckc(cudaMalloc(&scratch, (size_t) C * N * 4 * 2), "hc check");
        k_hcb_mixinj<<<C, HCM_THREADS, 0, s>>>(R, w_norm, p.rs, p.up32, scratch, reinterpret_cast<__half*>(scratch + (size_t) C * N), w_inj, ref);
        std::vector<float> a((size_t) C * HC), b(a.size());
        ckc(cudaMemcpyAsync(a.data(), ref, a.size() * 4, cudaMemcpyDeviceToHost, s), "hc check");
        ckc(cudaMemcpyAsync(b.data(), inject, b.size() * 4, cudaMemcpyDeviceToHost, s), "hc check");
        ckc(cudaStreamSynchronize(s), "hc check");
        double md = 0, mv = 0;
        for (size_t i = 0; i < a.size(); ++i) md = std::max(md, (double) std::fabs(a[i] - b[i])), mv = std::max(mv, (double) std::fabs(a[i]));
        std::fprintf(stderr, "hc check: inject largest difference %.3g of largest value %.3g\n", md, mv);
        cudaFree(ref), cudaFree(scratch);
    }
    ckc(cudaGetLastError(), "hc read");
}

// FNX_PF_HCSPLIT: pf_hc_read on rows [r0, r0 + nr) of a chunk: every per-row pointer moved to the rows, the scratch (xh, rs, lo, up) used from its start
void pf_hc_read_rows(Engine::Pf& p, float* R, const float* bo, const float* inj_prev, const float* w_norm, const Q8S& w_down, const Q8S& w_up,
                     const uint16_t* w_inj, float eps, float* mixed, float* inject, int r0, int nr, cudaStream_t s, __half* mixh = nullptr) {
    pf_hc_read(p, R + (size_t) r0 * D, bo ? bo + (size_t) r0 * N : nullptr, inj_prev ? inj_prev + (size_t) r0 * HC : nullptr, w_norm, w_down, w_up, w_inj,
               eps, mixed + (size_t) r0 * N, inject ? inject + (size_t) r0 * HC : nullptr, nr, s, mixh ? mixh + (size_t) r0 * N : nullptr);
}

// FNX_PF_HCSPLIT: the cards swap the rows they own of M (C rows of row_elems f32; the part card g finishes, (g + 1) % NCARD, is its own) through host
// memory, in `pieces` pieces on stream s: this card's piece i to the other card's landing place `nin` (flag nflags[i] = seq after it), then the other card's
// piece i from `in` into M once flags[i] reaches seq. Two cards only. A piece's copy is queued behind the stream's earlier work, so s must have waited for
// the event that says M's own rows are final.
void pf_swap_rows(void* Mv, size_t row_bytes, int C, int g, int pieces, const void* inv, const unsigned* flags, void* ninv, unsigned* nflags, unsigned seq,
                  cudaStream_t s) {
    char* M = (char*) Mv;
    const char* in = (const char*) inv;
    char* nin = (char*) ninv;
    const int own = (g + 1) % NCARD, peer = (g + NCARD - 1) % NCARD, peer_part = (peer + 1) % NCARD;
    auto b0 = [&](int part) { return part * C / NCARD; };
    const int ob = b0(own), oe = b0(own + 1), pb = b0(peer_part), pe = b0(peer_part + 1);
    for (int i = 0; i < pieces; ++i) {
        const int a = ob + (int) ((long) (oe - ob) * i / pieces), e = ob + (int) ((long) (oe - ob) * (i + 1) / pieces);
        if (e > a)
            ckc(cudaMemcpyAsync(nin + (size_t) (a - ob) * row_bytes, M + (size_t) a * row_bytes, (size_t) (e - a) * row_bytes, cudaMemcpyDefault, s),
                "swap send");
        cud(cuStreamWriteValue32((CUstream) s, (CUdeviceptr) (nflags + i), seq, 0), "swap flag");
    }
    for (int i = 0; i < pieces; ++i) {
        const int a = pb + (int) ((long) (pe - pb) * i / pieces), e = pb + (int) ((long) (pe - pb) * (i + 1) / pieces);
        cud(cuStreamWaitValue32((CUstream) s, (CUdeviceptr) (flags + i), seq, CU_STREAM_WAIT_VALUE_GEQ), "swap wait");
        if (e > a)
            ckc(cudaMemcpyAsync(M + (size_t) a * row_bytes, in + (size_t) (a - pb) * row_bytes, (size_t) (e - a) * row_bytes, cudaMemcpyDefault, s),
                "swap recv");
    }
}

// R = the embedding row in every stream
__global__ void k_pf_streams(const float* __restrict__ emb, float* __restrict__ R, int C) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) C * D; e += (size_t) gridDim.x * blockDim.x)
        R[e] = emb[e / D * N + e % N];
}

// ---- the chunk sum on the copy engines
// this card's addend into x: the routed experts' weighted rows plus the gated shared row (as ring_x makes it)
__global__ void k_pfr_addend(float* __restrict__ x, const float* __restrict__ part, const float* __restrict__ topw,
                             const float* __restrict__ logits, int C) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) C * N; e += (size_t) gridDim.x * blockDim.x) {
        const int t = (int) (e / N), i = (int) (e % N);
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) s = fmaf(topw[t * TOPK + k], part[((size_t) t * TOPK + k) * N + i], s);
        x[e] = fmaf(pf_sigm(logits[(size_t) t * (NEXP + 1) + NEXP]), part[((size_t) C * TOPK + t) * N + i], s);
    }
}


// the sum of x (C rows of N) over the cards on card g, one way round (each card's link is busy both ways already: it
// sends to g + 1 while it takes from g - 1; measured, the other direction at once halves both)
void pf_ring(Engine::Pf& p, int g, float* x, int C, const RingMoe* m, unsigned* err, cudaStream_t s, bool own_out = false) {
    const unsigned seq = ++p.seq;
    if (m) k_pfr_addend<<<grid_for((size_t) C * N), 256, 0, s>>>(x, m->part, m->topw, m->logits, C);
#if FNX_HIP
    if (p.rccl) {
        const RcclApi& api = *(const RcclApi*) p.rc_api;
        const size_t n = (size_t) C * N;
        if (p.rc_f32) {   // FNX_PF_RCCL_F32=1: the f32 rows summed in place (twice the bytes of the f16 sum, no rounding of the cards' parts)
            const int r = api.AllReduce(x, x, n, RCCL_FLOAT32, RCCL_SUM, p.rc_comm, (void*) s);
            if (r != RCCL_SUCCESS) throw std::runtime_error(std::string("prefill: ncclAllReduce: ") + api.ErrorString(r));
            return;
        }
        __half* buf = p.ring_ce.dag[seq & 1u];
        k_pfr_pack<<<grid_for(n / 2), 256, 0, s>>>((const float2*) x, (int) (n / 2), (__half2*) buf, err);
        const int r = api.AllReduce(buf, buf, n, RCCL_FLOAT16, RCCL_SUM, p.rc_comm, (void*) s);
        if (r != RCCL_SUCCESS) throw std::runtime_error(std::string("prefill: ncclAllReduce: ") + api.ErrorString(r));
        p.post.push_back([&p, buf, x, n, err, s] { k_pfr_out_chk<<<grid_for(n / 2), 256, 0, s>>>((const __half2*) buf, (float2*) x, (int) (n / 2), err); });
        ckc(cudaGetLastError(), "ring");
        return;
    }
#endif
#if FNX_HIP
    if (p.a2a_rows > 0 && C <= p.a2a_rows && !own_out) {   // a small chunk: all-to-all and a local sum, one cross-card latency (bit-identical to the ring)
        const size_t slot4 = (size_t) p.a2a_rows * N / 4, set_off4 = (size_t) (seq & 1u) * NCARD * slot4;
        const int n4 = C * N / 4;
        k_pfa_push<<<grid_for((size_t) n4), 256, 0, s>>>((const float4*) x, n4, p.a2a_peer_box_d, p.a2a_peer_flag_d, p.a2a_cnt, g, seq, set_off4, slot4);
        k_pfa_sum<<<grid_for((size_t) n4), 256, 0, s>>>(x, C, p.a2a_box, p.a2a_flag, g, seq, set_off4, slot4, err);
        ckc(cudaGetLastError(), "ring (small chunk)");
        return;
    }
#endif
    pf_ring_run(p.ring_ce, p.dev, g, x, C, seq, err, s, own_out);
    ckc(cudaGetLastError(), "ring");
}

// the attention partials to their heads' cards: block k of apart to card k (into its slot for this card), then wait
// for the other cards' blocks of this card's heads. One receive set: a card's next exchange comes after a chunk sum that
// needs every card past its merge
// rows [r0, r0 + nr) of the partials to their heads' cards (piece i of the exchange, flag [slot][i] there)
void pf_attn_send(Engine::Pf& p, int set, int g, int C, int r0, int nr, int i, unsigned seq, cudaStream_t s) {
    const size_t blk = attn_prompt_block_floats(C), off = attn_prompt_block_floats(r0);
    for (int d = 1; d < NCARD; ++d) {
        const int k = (g + d) % NCARD, slot = (g - k + NCARD) % NCARD - 1;
        ckc(cudaMemcpyAsync(p.parecv[set][k] + slot * blk + off, p.apart[set] + k * blk + off,
                            attn_prompt_block_floats(nr) * sizeof(float), cudaMemcpyDefault, s), "attn copy");   // device -> host buffer
        cud(cuStreamWriteValue32((CUstream) s, (CUdeviceptr) (p.paflags[set][k] + slot * ATT_PIECES_MAX + i), seq, 0), "attn flag");
    }
}

// a prompt chunk's attention and the sends of its exchange (buffer set `set`): each piece of the rows' partials goes to
// its heads' cards (on st2) while the next piece is computed
void pf_attn_start(Engine::Pf& p, CardBuf& b, uint16_t* kc, uint16_t* vc, int g, int C, int set, cudaStream_t s) {
    const unsigned seq = p.att_seq[set] = ++p.seq;
    const int n = p.att_pieces;
    for (int i = 0; i < n; ++i) {
        const int r0 = (int) ((long) C * i / n), r1 = (int) ((long) C * (i + 1) / n);
        attn_prompt(p.qg, kc, vc, b.pg, p.ctl, p.cells, p.ncells, g, C, r0, r1 - r0, p.apart[set], s);
        ckc(cudaEventRecord(p.ev_piece[i], s), "attn event");
        ckc(cudaStreamWaitEvent(b.st2, p.ev_piece[i], 0), "attn wait piece");
        pf_attn_send(p, set, g, C, r0, r1 - r0, i, seq, b.st2);
    }
    ckc(cudaEventRecord(p.ev_j[set], b.st2), "attn event");
    if (p.attn_dma) {
        const size_t blk = attn_prompt_block_floats(C);
        ckc(cudaStreamWaitEvent(p.st3, p.ev_m[set], 0), "attn recv reuse");
        for (int i = 0; i < n; ++i) {
            const int r0 = (int) ((long) C * i / n), r1 = (int) ((long) C * (i + 1) / n);
            const size_t off = attn_prompt_block_floats(r0);
            for (int slot = 0; slot < NCARD - 1; ++slot) {
                cud(cuStreamWaitValue32((CUstream) p.st3, (CUdeviceptr) (p.aflags[set] + slot * ATT_PIECES_MAX + i), seq, CU_STREAM_WAIT_VALUE_GEQ), "attn wait (dma)");
                ckc(cudaMemcpyAsync(p.arecv_dev[set] + slot * blk + off, p.arecv[set] + slot * blk + off, attn_prompt_block_floats(r1 - r0) * sizeof(float),
                                    cudaMemcpyDefault, p.st3), "attn recv copy");   // host buffer -> device, on the copy engine
            }
        }
        ckc(cudaEventRecord(p.ev_r[set], p.st3), "attn event");
    }
}
// its merge into att once the other cards' pieces are here and this card's sends are done; qg must still be the
// chunk's (the merge reads its gates)
void pf_attn_finish(Engine::Pf& p, int g, int C, int set, cudaStream_t s) {
    if (p.attn_dma) {
        ckc(cudaStreamWaitEvent(s, p.ev_r[set], 0), "attn recv join");
        ckc(cudaStreamWaitEvent(s, p.ev_j[set], 0), "attn join");
        attn_prompt_merge(p.apart[set] + (size_t) g * attn_prompt_block_floats(C), p.arecv_dev[set], p.qg, p.att, C, g, s);
        ckc(cudaEventRecord(p.ev_m[set], s), "attn merge event");
        return;
    }
    for (int slot = 0; slot < NCARD - 1; ++slot)
        for (int i = 0; i < p.att_pieces; ++i)
            cud(cuStreamWaitValue32((CUstream) s, (CUdeviceptr) (p.aflags[set] + slot * ATT_PIECES_MAX + i), p.att_seq[set],
                                    CU_STREAM_WAIT_VALUE_GEQ),
                "attn wait");
    ckc(cudaStreamWaitEvent(s, p.ev_j[set], 0), "attn join");
    attn_prompt_merge(p.apart[set] + (size_t) g * attn_prompt_block_floats(C), p.arecv[set], p.qg, p.att, C, g, s);
}
void pf_attention(Engine::Pf& p, CardBuf& b, uint16_t* kc, uint16_t* vc, int g, int C, cudaStream_t s) {
    pf_attn_start(p, b, kc, vc, g, C, 0, s);
    pf_attn_finish(p, g, C, 0, s);
}
}  // namespace

void Engine::pf_clear_ring(int c) {
    const Pf& p = *pf_[(size_t) c];
    if (!p.ring_sm) return;   // the decode ring's buffers otherwise
    ckc(cudaMemset(p.ring.in, 0, ring_in_words(pf_c_) * sizeof(unsigned long long)), "prompt ring clear");
    ckc(cudaMemset(p.ring.ag, 0, ring_ag_words(pf_c_) * sizeof(unsigned long long)), "prompt ring clear");
    ckc(cudaMemset(p.ring.flags, 0, ring_flag_words() * sizeof(unsigned)), "prompt ring clear");
}

void Engine::pf_seq_reset(unsigned start) {
    // every card idle first: a card's stream writes its numbers into the next cards' flags
    for (CardBuf* b : cb_) { ckc(cudaSetDevice(b->dev), "set"); ckc(cudaDeviceSynchronize(), "prompt sequence reset sync"); }
    for (int c = 0; c < NCARD; ++c) {
        Pf& p = *pf_[(size_t) c];
        if (p.att_pend >= 0) throw std::runtime_error("prefill: sequence numbers restarted with an attention merge pending");
        ckc(cudaSetDevice(p.dev), "set");
        ckc(cudaMemset(p.ring_ce.flags, 0, 2 * PF_RING_STEPS * sizeof(unsigned)), "prompt ring flags");
#if FNX_HIP
        if (p.a2a_flag) ckc(cudaMemset(p.a2a_flag, 0, NCARD * sizeof(unsigned)), "small-chunk sum flags");
#endif
        for (unsigned* f : p.aflags) ckc(cudaMemset(f, 0, ATT_FLAG_ROWS * ATT_PIECES_MAX * sizeof(unsigned)), "attention exchange flags");
        if (p.hcs) {
            for (unsigned* f : p.hg_flags) std::memset(f, 0, 8 * sizeof(unsigned));
            std::memset(p.rg_flags, 0, 16 * sizeof(unsigned));
            p.hg_seq = p.rg_seq = 0;
        }
        p.seq = p.att_seq[0] = p.att_seq[1] = start;
    }
    for (CardBuf* b : cb_) { ckc(cudaSetDevice(b->dev), "set"); ckc(cudaDeviceSynchronize(), "prompt sequence reset sync"); }
}

void Engine::prefill_alloc() {
    if (!pf_.empty()) throw std::runtime_error("prefill_alloc: the prompt path's buffers exist already");
    pf_.resize(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ckc(cudaSetDevice(b.dev), "set");
        size_t free0 = 0, tot0 = 0;
        ckc(cudaMemGetInfo(&free0, &tot0), "mem info");
        pf_card_bytes = 0;
        Pf* p = new Pf;
        ckb(cublasCreate(&p->blas), "create");
        {
            const int d = current_card();
            cublasHandle_t h = p->blas;
            own([d, h] { cudaSetDevice(d); cublasDestroy(h); });
            own([p] { delete p; });
        }
        ckb(cublasSetStream(p->blas, b.st), "stream");
        ckb(cublasSetMathMode(p->blas, CUBLAS_TENSOR_OP_MATH), "math mode");
        p->f16acc = env_flag("FNX_PF_F16ACC");
        const int ffs = w.ff1 - w.ff0, rs_qkvz = w.conv_ch() + w.nv() * HS;
        // the borrowed block (the recurrent snapshots): the phase region, then whole-chunk scratch; the rest on its own
        Arena big{(char*) b.snap_block, b.snap_block_bytes};
        // one region for the buffers of each phase of a layer: they never live at once (the FFN half's are read last
        // by the chunk sum's addend, on the stream before the next half starts)
        auto ffn_set = [&](Arena& a) {
            p->gu = a.take<float>((size_t) pf_c_ * TOPK * 2 * ffs);
            p->part = a.take<float>((size_t) (pf_c_ * TOPK + pf_c_) * N);
            p->shg = a.take<float>((size_t) pf_c_ * 2 * ffs);
            p->shh = a.take<__half>((size_t) pf_c_ * ffs);
            p->h16 = a.take<uint16_t>((size_t) pf_c_ * TOPK * ffs);
        };
        auto attn_set = [&](Arena& a) {
            p->qg = a.take<float>((size_t) pf_c_ * NQH * 2 * HD);
            p->k = a.take<float>((size_t) pf_c_ * NKVH * HD);
            p->v = a.take<float>((size_t) pf_c_ * NKVH * HD);
            p->att = a.take<float>((size_t) pf_c_ * w.nq * HD);
            p->yidx = a.take<float>((size_t) pf_c_ * qsa_rows());
            p->qidx = a.take<float>((size_t) pf_c_ * 512);
            p->cells = a.take<int>((size_t) pf_c_ * qsa_maxsel());
            p->ncells = a.take<int>(pf_c_);
            if (const char* qr = std::getenv("FNX_PF_QSA_ROWS")) {
                char* end = nullptr;
                const long v = std::strtol(qr, &end, 10);
                if (end == qr || *end || v < MAXT || v % MAXT)
                    throw std::runtime_error("FNX_PF_QSA_ROWS: a positive multiple of " + std::to_string(MAXT) + ", not '" + qr + "'");
                p->qsa_pf_rows = (int) v;
            }
            p->scores = a.take<float>((size_t) std::max(attn_rows(), p->qsa_pf_rows) * (max_ctx_ / 4));
            p->apart[0] = a.take<float>(NCARD * attn_prompt_block_floats(pf_c_));
        };
        auto rec_set = [&](Arena& a) {
            p->qkvz = a.take<float>((size_t) pf_c_ * rs_qkvz);
            p->ab = a.take<float>((size_t) pf_c_ * 2 * w.nv());
            p->y32 = a.take<float>((size_t) pf_c_ * w.nv() * HS);
            p->gdn_xc = a.take<float>((size_t) pf_c_ * w.conv_ch());
            p->gdn_sq = a.take<float>((size_t) pf_c_ * w.nv() * 4);
            p->gdn_ch = a.take<float>(gdn_prompt_work_floats(pf_c_, w.nv()));
        };
        auto ple_set = [&](Arena& a) {
            p->ple_key = a.take<float>((size_t) pf_c_ * D);
            p->ple_val = a.take<float>((size_t) pf_c_ * N);
            p->ple_tmp = a.take<float>((size_t) 2 * MAXT * D);
        };
        size_t phase = 0;
        {
            Arena dry{nullptr, SIZE_MAX};
            ffn_set(dry), phase = std::max(phase, dry.off), dry.off = 0;
            attn_set(dry), phase = std::max(phase, dry.off), dry.off = 0;
            rec_set(dry), phase = std::max(phase, dry.off), dry.off = 0;
            ple_set(dry), phase = std::max(phase, dry.off);
        }
        char* region = big.fits(phase) ? big.take<char>(phase) : palloc<char>(phase);
        for (auto set : {0, 1, 2, 3}) {
            Arena a{region, phase};
            if (set == 0) ffn_set(a);
            else if (set == 1) attn_set(a);
            else if (set == 2) rec_set(a);
            else ple_set(a);
        }
        auto from_big = [&](size_t floats) { return big.fits(floats * sizeof(float)) ? big.take<float>(floats) : palloc<float>(floats); };
        p->up32 = from_big(std::max<size_t>((size_t) pf_c_ * D, (size_t) pf_c_ * NQH * 2 * HD / 2 + 16));
        p->xf = from_big((size_t) pf_c_ * D);
        p->bo_a = from_big((size_t) pf_c_ * N);
        p->widx = from_big((size_t) qsa_rows() * N);
        p->wh = palloc<__half>(PF_WH);
        p->xh = palloc<__half>((size_t) pf_c_ * D);
        p->R = palloc<float>((size_t) pf_c_ * D);
        p->rs = palloc<float>((size_t) pf_c_ * HC);
        p->lo32 = palloc<float>((size_t) pf_c_ * LR);
        p->lo16 = palloc<__half>((size_t) pf_c_ * LR);
        p->mixed = palloc<float>((size_t) pf_c_ * N);
        p->bo_f = palloc<float>((size_t) pf_c_ * N);
        p->inj_a = palloc<float>((size_t) pf_c_ * HC);
        p->inj_f = palloc<float>((size_t) pf_c_ * HC);
        p->moe_q8 = env_flag("FNX_PF_MOE_Q8");
        if (p->moe_q8) {
            p->xq = palloc<uint8_t>((size_t) pf_c_ * q8_1_bytes(N));
            q8_twin(p->xq, (size_t) pf_c_ * q8_1_bytes(N), palloc<int8_t>((size_t) pf_c_ * N), palloc<float>((size_t) pf_c_ * N / 32));
            const void* q = p->xq;
            own([q] { q8_untwin(q); });
        }
        p->logits = palloc<float>((size_t) pf_c_ * (NEXP + 1));
        p->topw = palloc<float>((size_t) pf_c_ * TOPK);
        p->topi = palloc<int>((size_t) pf_c_ * TOPK);
        p->cnt = palloc<int>(NEXP);
        p->jbase = palloc<int>(NEXP);
        p->njobs = palloc<int>(2);
        p->jobs_gu = palloc<BigJob>(pf_jobs_);
        p->jobs_dn = palloc<BigJob>(pf_jobs_);
        p->ple_emb = palloc<float>((size_t) pf_c_ * N);
        p->ctl = palloc<int>((size_t) (1 + (pf_c_ / MAXT)) * 2);
        p->ring_sm = env_int("FNX_PF_RING_SM", 0) != 0;   // strata-split: the chunk sums go through host memory on the copy engines; FNX_PF_RING_SM=1 is the GPU-thread ring (slower, kept as the fallback path)
        p->ring = b.ring;
        if (p->ring_sm) {   // ~140 MB a card at 1,024 rows a chunk
            p->ring.f32 = false;   // HOSTRING: the prompt path's sums are fp16 on every count; FNX_RING_F32 is the decode ring's
            p->ring.rows = pf_c_;
            p->ring.in = hpalloc<unsigned long long>(ring_in_words(pf_c_));
            p->ring.ag = hpalloc<unsigned long long>(ring_ag_words(pf_c_));
            p->ring.flags = hpalloc<unsigned>(ring_flag_words());
        }
        {
            PfRing& r = p->ring_ce;
            const size_t share = (size_t) (pf_c_ + NCARD - 1) / NCARD * N;   // a card's share of the rows
            r.send = palloc<__half>(share);
            for (int k = 0; k < 2; ++k) {
#if FNX_HIP
                if (pf_peer_) {   // device memory on this card: the previous card's kernels store into it
                    for (int h = 0; h < NCARD - 1; ++h) r.in[h][k] = ppalloc<__half>(share);
                    r.ag[k] = ppalloc<__half>((size_t) pf_c_ * N);
                } else
#endif
                {
                    for (int h = 0; h < NCARD - 1; ++h) r.in[h][k] = hpalloc<__half>(share);   // a slot per reduce hop
                    r.ag[k] = hpalloc<__half>((size_t) pf_c_ * N);
                }
                for (int h = 0; h < NCARD - 1; ++h) r.din[h][k] = palloc<__half>(share);
                r.dag[k] = palloc<__half>((size_t) pf_c_ * N);
            }
#if FNX_HIP
            if (pf_peer_) r.flags = ppalloc<unsigned>(2 * PF_RING_STEPS);
            else
#endif
            r.flags = hpalloc<unsigned>(2 * PF_RING_STEPS);
#if FNX_HIP
            if (pf_peer_) {   // the small-chunk sum: FNX_PF_A2A_ROWS rows at most (0: always the ring)
                const long long ar = env_int("FNX_PF_A2A_ROWS", 256);
                p->a2a_rows = (int) std::min<long long>(std::max<long long>(ar, 0), pf_c_);
                if (p->a2a_rows > 0) {
                    p->a2a_box = ppalloc<float4>((size_t) 2 * NCARD * p->a2a_rows * N / 4);
                    p->a2a_flag = ppalloc<unsigned>(NCARD);
                    p->a2a_cnt = palloc<unsigned>(1);
                    p->a2a_peer_box_d = palloc<float4*>(NCARD);
                    p->a2a_peer_flag_d = palloc<unsigned*>(NCARD);
                }
            }
#endif
        }
        p->dev = b.dev;
#if FNX_HIP
        if (rccl_on_) {
            p->rccl = true;
            p->rc_f32 = env_int("FNX_PF_RCCL_F32", 0) != 0;
            p->rc_api = &static_cast<RcclState*>(rccl_.get())->api;
            p->rc_comm = static_cast<RcclState*>(rccl_.get())->comm[(size_t) c];
        }
#endif
        p->attn_dec = env_flag("FNX_PF_ATTN_DEC");
        p->apart[1] = palloc<float>(NCARD * attn_prompt_block_floats(pf_c_));   // set 0 is in the attention phase's region
        for (int s = 0; s < 2; ++s) {
            p->arecv[s] = hpalloc<float>((NCARD - 1) * attn_prompt_block_floats(pf_c_));
            p->aflags[s] = hpalloc<unsigned>(ATT_FLAG_ROWS * ATT_PIECES_MAX);   // [slot][piece]
        }
        if (const char* ap = std::getenv("FNX_PF_ATT_PIECES")) {
            char* end = nullptr;
            const long v = std::strtol(ap, &end, 10);
            if (end == ap || *end || v < 1 || v > ATT_PIECES_MAX)
                throw std::runtime_error("FNX_PF_ATT_PIECES: 1 to " + std::to_string(ATT_PIECES_MAX) + ", not '" + ap + "'");
            p->att_pieces = (int) v;
        }
        for (cudaEvent_t& e : p->ev_piece) {
            ckc(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
            own_event(e);
        }
        for (cudaEvent_t& e : p->ev_j) {
            ckc(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
            own_event(e);
        }
#if FNX_HIP
        // HIP: the copy-engine receive is opt-in (FNX_PF_DMA=1, see r.dma above: a DMA read of a peer's host stores was not always complete on gfx1201), so the attention
        // partials' copy is opt-in too: FNX_PFX_ATTN_DMA=1 turns it on
        p->attn_dma = env_int("FNX_PFX_ATTN_DMA", 0) != 0;
#else
        p->attn_dma = env_int("FNX_PFX_ATTN_DMA", 1) != 0;   // on by default (3.3-5.6% on 32K and 62.9K IQ3_S prompts, bit-identical logits); FNX_PFX_ATTN_DMA=0 turns it off
#endif
        if (p->attn_dma) {
            for (int s2 = 0; s2 < 2; ++s2) {
                p->arecv_dev[s2] = palloc<float>((NCARD - 1) * attn_prompt_block_floats(pf_c_));
                ckc(cudaEventCreateWithFlags(&p->ev_r[s2], cudaEventDisableTiming), "event");
                own_event(p->ev_r[s2]);
                ckc(cudaEventCreateWithFlags(&p->ev_m[s2], cudaEventDisableTiming), "event");
                own_event(p->ev_m[s2]);
            }
            ckc(cudaStreamCreateWithFlags(&p->st3, cudaStreamNonBlocking), "stream");
            own_stream(p->st3);
        }
        p->Rk[0] = p->R, p->bofk[0] = p->bo_f, p->injfk[0] = p->inj_f, p->ctlk[0] = p->ctl, p->boak[0] = p->bo_a,
        p->injak[0] = p->inj_a;
        {
            // the hyper-connection split (below) wants 4 chunks to a pass: a chunk's swap and sum then hide behind the other chunks' work (2: no gain)
            bool hcs_wanted = false;
#if !FNX_HIP
            hcs_wanted = NCARD == 2 && !p->ring_sm && env_int("FNX_PF_OVERLAP", 1) != 0 && env_int("FNX_PF_HCSPLIT", 1) != 0;
#endif
            const long long pass = env_int("FNX_PF_PASS", pf_pass_pick_ > 0 ? pf_pass_pick_ : (w.ec.n_slots || hcs_wanted ? 4 : 2));   // with an ExpertCache, staging amortizes over more
            if (pass < 1 || pass > PF_PASS) throw std::runtime_error("FNX_PF_PASS: 1 to " + std::to_string(PF_PASS) + ", got " + std::to_string(pass));
            p->pass = (int) pass;
            for (int k = 1; k < p->pass; ++k) {
                p->Rk[k] = palloc<float>((size_t) pf_c_ * D);
                p->bofk[k] = palloc<float>((size_t) pf_c_ * N);
                p->injfk[k] = palloc<float>((size_t) pf_c_ * HC);
                p->ctlk[k] = palloc<int>((size_t) (1 + (pf_c_ / MAXT)) * 2);
                p->boak[k] = palloc<float>((size_t) pf_c_ * N);
                p->injak[k] = palloc<float>((size_t) pf_c_ * HC);
            }
            p->overlap = !p->ring_sm && env_int("FNX_PF_OVERLAP", 1) != 0;
            if (p->overlap) {
                ckc(cudaStreamCreateWithFlags(&p->ring_st, cudaStreamNonBlocking), "ring stream");
                own_stream(p->ring_st);
                for (int k = 0; k < PF_PASS; ++k)
                    for (cudaEvent_t* e : {&p->ev_go[k], &p->ev_ra[k], &p->ev_rf[k]}) {
                        ckc(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
                        own_event(*e);
                    }
            }
            // FNX_PF_HCSPLIT (NVIDIA, two cards, the copy-engine ring): see Pf
#if !FNX_HIP
            p->hcs_mode = (int) env_int("FNX_PF_HCSPLIT", 2);   // on by default (mode 2 since owner 11-03; mode 1 is bit-identical to the unsplit path): used only if the calibration finds the algorithm
            p->hcs = NCARD == 2 && p->overlap && !p->ring_sm && p->hcs_mode != 0;
#endif
            if (p->hcs) {
                const long long np = env_int("FNX_PF_HCSPLIT_PIECES", 4);
                if (np < 1 || np > 8) throw std::runtime_error("FNX_PF_HCSPLIT_PIECES: 1 to 8, got " + std::to_string(np));
                p->hg_pieces = (int) np;
                p->hf16 = env_int("FNX_PF_HCSPLIT_F16", 0) != 0;
#if !FNX_HIP
                if (env_int("FNX_PF_HCSPLIT_LT", 1) != 0) pf_lt_calibrate(*p, b.st);
#endif
                ckc(cudaStreamCreateWithFlags(&p->ag_st, cudaStreamNonBlocking), "swap stream");
                own_stream(p->ag_st);
                if (p->hf16)
                    for (int k = 0; k < p->pass; ++k) p->mixhk[k] = palloc<__half>((size_t) pf_c_ * N);
                const size_t share = (size_t) (pf_c_ + NCARD - 1) / NCARD;
                p->mixedk[0] = p->mixed;
                for (int k = 1; k < p->pass; ++k) p->mixedk[k] = palloc<float>((size_t) pf_c_ * N);
                for (int k = 0; k < 2; ++k) {
                    p->hg_in[k] = hpalloc<float>(share * N);
                    p->hg_flags[k] = hpalloc<unsigned>(8);
                }
                for (int k = 0; k < 2; ++k) p->rg_in[k] = hpalloc<float>(share * D);
                p->rg_flags = hpalloc<unsigned>(16);
                for (int k = 0; k < PF_PASS; ++k)
                    for (cudaEvent_t* e : {&p->ev_hc[k], &p->ev_ag[k], &p->ev_rg[k]}) {
                        ckc(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
                        own_event(*e);
                    }
            }
        }
        if (w.ec.n_slots) {   // the stages themselves: prefill_stages, once the experts' places are final
            const long long stage_rows = env_int("FNX_PF_STAGE_ROWS", 2048);
            if (stage_rows < 0 || stage_rows > INT32_MAX)
                throw std::runtime_error("FNX_PF_STAGE_ROWS: want tokens >= 0, got " + std::to_string(stage_rows));
            p->stage_rows = (int) stage_rows;
            p->ep_gu = (const char**) palloc<char*>((size_t) NL * NEXP);
            p->ep_dn = (const char**) palloc<char*>((size_t) NL * NEXP);
            ckc(cudaMallocHost(&p->ep_hgu, (size_t) NL * NEXP * sizeof(char*)), "pinned tables");
            own_pinned(p->ep_hgu);
            ckc(cudaMallocHost(&p->ep_hdn, (size_t) NL * NEXP * sizeof(char*)), "pinned tables");
            own_pinned(p->ep_hdn);
            ckc(cudaMallocHost(&p->slot_h, (size_t) NL * NEXP * sizeof(int)), "pinned slots");
            own_pinned(p->slot_h);
            ckc(cudaStreamCreateWithFlags(&p->cs, cudaStreamNonBlocking), "copy stream");
            own_stream(p->cs);
            for (int k = 0; k < 2; ++k)
                for (cudaEvent_t* e : {&p->ev_ready[k], &p->ev_done[k]}) {
                    ckc(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
                    own_event(*e);
                }
        }
        p->alloc_bytes = pf_card_bytes;
        pf_[c] = p;
        if (env_flag("FNX_PF_TIMING"))
            std::fprintf(stderr, "prefill buffers on card %d: %zu MiB (%zu MiB free before)\n", c, pf_card_bytes >> 20, free0 >> 20);
    }
    {   // the hyper-connection split runs on every card or on none; mode 1 needs the calibrated algorithm on every card
        bool all = true, asked = false;
        for (int c = 0; c < NCARD; ++c) {
            asked = asked || pf_[c]->hcs;
            all = all && pf_[c]->hcs && pf_[c]->down_lt;
        }
        for (int c = 0; c < NCARD; ++c) pf_[c]->hcs = all;
        if (asked && !all)
            std::fprintf(stderr, "fnx: prompt path: the hyper-connection read stays on both cards (no cublasLt algorithm that gives the same rows at 2048 and 4096 rows on this card)\n");
        else if (all)
            std::fprintf(stderr, "fnx: prompt path: the hyper-connection read is split between the cards (%s)\n", pf_[0]->hcs_mode == 1 ? "same results as the unsplit read" : "results differ in the last bits");
    }
    for (int c = 0; c < NCARD; ++c)
        for (int k = 0; k < NCARD; ++k) {
            for (int s = 0; s < 2; ++s) {
                pf_[c]->parecv[s][k] = pf_[k]->arecv[s];
                pf_[c]->paflags[s][k] = pf_[k]->aflags[s];
            }
            pf_[c]->pdev[k] = pf_[k]->dev;
        }
    for (int c = 0; c < NCARD; ++c) {
        Pf& p = *pf_[c];
        const Pf& n = *pf_[(c + 1) % NCARD];
        p.ring.next_in = n.ring.in;
        p.ring.next_ag = n.ring.ag;
        p.ring.next_flags = n.ring.flags;
        PfRing& r = p.ring_ce;
        for (int k = 0; k < 2; ++k) {
            for (int h = 0; h < NCARD - 1; ++h) r.nin[h][k] = n.ring_ce.in[h][k];
            r.nag[k] = n.ring_ce.ag[k];
        }
        r.nflags = n.ring_ce.flags;
        r.next_dev = n.dev;
#if FNX_HIP
        if (p.a2a_rows > 0) {
            ckc(cudaSetDevice(p.dev), "set");
            float4* boxes[NCARD];
            unsigned* flags[NCARD];
            for (int d = 0; d < NCARD; ++d) boxes[d] = pf_[d]->a2a_box, flags[d] = pf_[d]->a2a_flag;
            ckc(cudaMemcpy(p.a2a_peer_box_d, boxes, sizeof boxes, cudaMemcpyHostToDevice), "a2a boxes");
            ckc(cudaMemcpy(p.a2a_peer_flag_d, flags, sizeof flags, cudaMemcpyHostToDevice), "a2a flags");
        }
#endif
        if (p.hcs) {
            for (int k = 0; k < 2; ++k) p.nhg_in[k] = n.hg_in[k], p.nhg_flags[k] = n.hg_flags[k];
            for (int k = 0; k < 2; ++k) p.nrg_in[k] = n.rg_in[k];
            p.nrg_flags = n.rg_flags;
        }
        for (int d = 0; d < NCARD; ++d) {
            r.pflags[d] = pf_[d]->ring_ce.flags;
            for (int k = 0; k < 2; ++k) r.pag[d][k] = pf_[d]->ring_ce.ag[k];
        }
        // the all-gather: HIP defaults to the direct gather (a ring of host-to-host forwards costs 1/3 of the prompt time on four cards),
        // CUDA to the ring; FNX_PF_GATHER=ring|direct chooses
        const char* gather = std::getenv("FNX_PF_GATHER");
#if FNX_HIP
        r.direct = !(gather && std::strcmp(gather, "ring") == 0);
#else
        r.direct = gather && std::strcmp(gather, "direct") == 0;
#endif
        // HIP: the copy-engine receive/gather is opt-in (FNX_PF_DMA=1). On gfx1201 a DMA read of a peer's host stores was not always complete when the flag
        // said so (FNX_PF_DMA_CHECK=1 finds it; answers varied from start to start), and it was not faster; the kernel reads are repeatable
#if FNX_HIP
        r.dma = env_int("FNX_PF_DMA", 0) != 0;
#else
        r.dma = env_int("FNX_PF_DMA", 1) != 0;   // (FNX_HIP is defined only by the HIP build: an expression use of it does not compile on NVIDIA)
#endif
        r.dma_check = env_int("FNX_PF_DMA_CHECK", 0) != 0;
        if (env_int("FNX_PF_DMA_BARRIER", 0) != 0) {
            ckc(cudaSetDevice(p.dev), "set");
            ckc(cudaEventCreateWithFlags(&r.fence_ev, cudaEventDisableTiming), "event");
            own_event(r.fence_ev);
        }
        { const char* sk = std::getenv("FNX_PF_SEND"); r.send_kernel = sk && std::strcmp(sk, "kernel") == 0; }
        r.plan = r.direct ? pf_ring_plan_direct(NCARD, c) : pf_ring_plan(NCARD, c);
    }
    // the flags as cudaMalloc left them could hold anything, and a wait passes on any flag >= its number
    const long long seq0 = env_int("FNX_PF_SEQ_START", 0);
    if (seq0 < 0 || seq0 > (long long) UINT32_MAX)
        throw std::runtime_error("FNX_PF_SEQ_START: want 0.." + std::to_string(UINT32_MAX) + ", got " + std::to_string(seq0));
    pf_seq_reset((unsigned) seq0);
}

size_t Engine::pf_stage_bytes(int c) const {
    const CardW& w = m_.cards[c];
    if (!w.ec.n_slots) return 0;
    // a stage holds the largest layer's experts not in a slot at the start, FNX_PF_STAGE_MIB at most
    size_t most = 0;
    for (int l = 0; l < NL; ++l) {
        size_t n = 0;
        for (int e = 0; e < NEXP; ++e) n += m_.resident[(size_t) l][(size_t) e] ? 0 : 1;
        most = std::max(most, n * (w.L[(size_t) l].gu_e_stride + w.L[(size_t) l].down_e_stride));
    }
    const long long stage_mib = env_int("FNX_PF_STAGE_MIB", 384);
    if (stage_mib < 0) throw std::runtime_error("FNX_PF_STAGE_MIB: want MiB >= 0, got " + std::to_string(stage_mib));
    return std::min(most, (size_t) stage_mib << 20);
}

void Engine::prefill_stages() {
    for (int c = 0; c < NCARD; ++c) {
        Pf& p = *pf_[c];
        if (p.stage[0]) throw std::runtime_error("prefill_stages: card " + std::to_string(c) + " has its stages already");
        p.stage_bytes = pf_stage_bytes(c);
        if (!p.stage_bytes) continue;
        ckc(cudaSetDevice(p.dev), "set");
        pf_card_bytes = p.alloc_bytes;   // palloc's out-of-memory message says what this card's prompt path holds
        for (char*& st : p.stage) st = palloc<char>(p.stage_bytes);
        p.alloc_bytes = pf_card_bytes;
        if (env_flag("FNX_PF_TIMING"))
            std::fprintf(stderr, "prefill on card %d: passes of %d chunks, stages of %zu MiB\n", c, p.pass, p.stage_bytes >> 20);
    }
}

// the draft layer over a chunk's rows, for its state only (its K/V and QSA keys; its outputs over a prompt are never
// read): row t's source is the chunk's final residual (p.R, as the last layer's FFN-half read leaves it), paired with
// the token after the row. As mtp_enqueue up to its attention, every card computing every row (no sums across them)
void Engine::prefill_draft(int C, int set) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        Pf& p = *pf_[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        cudaStream_t s = b.st;
        ckc(cudaSetDevice(b.dev), "set");
        // the next tokens' embeddings: rows 1.. of this chunk's, then the token after it (ple_emb is free past the PLE layer)
        if (C > 1) ckc(cudaMemcpyAsync(p.ple_emb, pf_emb_[set] + N, (size_t) (C - 1) * N * 4, cudaMemcpyHostToDevice, s), "draft emb");
        ckc(cudaMemcpyAsync(p.ple_emb + (size_t) (C - 1) * N, pf_after_[set], (size_t) N * 4, cudaMemcpyHostToDevice, s), "draft after");
        rms_norm_rows(p.ple_emb, mc.norm_emb, p.mixed, C, N, eps, s);   // mixed: free until the hc read below
        pf_dense(p, mc.fc_emb.s, N, N, p.mixed, C, p.bo_f, N, s);
        // the source: one RMS over the four streams, then fc_hidden per stream (HC * C rows of N)
        rms_norm_rows(p.R, mc.norm_hid, p.xf, C, D, eps, s);
        pf_dense(p, mc.fc_hid.s, N, N, p.xf, HC * C, p.up32, N, s);
        add_streams(p.up32, p.bo_f, p.R, C, s);
        pf_hc_read(p, p.R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], nullptr, eps, p.mixed, nullptr, C, s);
        if (L.idx_wq) {
            k_bf16_f32<<<grid_for((size_t) 512 * N), 256, 0, s>>>(L.idx_wq, (size_t) 512 * N, p.widx);
            k_bf16_f32<<<grid_for((size_t) 128 * N), 256, 0, s>>>(L.idx_wk, (size_t) 128 * N, p.widx + (size_t) 512 * N);
            pf_sgemm(p, p.widx, qsa_rows(), N, p.mixed, C, p.yidx, qsa_rows());
        } else {
            pf_dense(p, L.idx_qk.s, qsa_rows(), N, p.mixed, C, p.yidx, qsa_rows(), s);
        }
        qsa_index(p.yidx, L.idx_qn, L.idx_kn, b.mtp_qsa_ring, b.mtp_qsa_pooled, b.pg, p.qidx, p.ctl, C, eps, rope_of(b.mrope), s);
        pf_dense(p, L.wk.s, NKVH * HD, N, p.mixed, C, p.k, NKVH * HD, s);
        pf_dense(p, L.wv.s, NKVH * HD, N, p.mixed, C, p.v, NKVH * HD, s);
        attn_prep(nullptr, p.k, L.q_norm, L.k_norm, 0, NKVH, p.ctl, C, eps, rope_of(b.mrope), s);
        kv_append(p.k, p.v, b.mtp_kc, b.mtp_vc, b.pg, b.att_ring.card, C, p.ctl, s);
        ckc(cudaGetLastError(), "draft layer (prompt)");
    }
}

void Engine::prefill(const std::vector<int>& tokens, int after) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("prefill: a window is not committed");
    if (mtp_ && after < 0) throw std::runtime_error("prefill: the draft layer needs the token after the last row");
    if (pf_[0]->seq >= PF_SEQ_RESET) {
        std::fprintf(stderr, "fnx: prompt ring sequence numbers at %u: flags zeroed, numbers from 0 again\n", pf_[0]->seq);
        pf_seq_reset(0);
    }
    // a prompt shorter than FNX_PF_STAGE_ROWS reads the experts in RAM in place, a chunk at a time
    const bool staging = pf_[0]->stage[0] && (int) tokens.size() >= pf_[0]->stage_rows;
    int K = PF_PASS;
    for (const Pf* p : pf_) K = std::min(K, p->pass);
    if (pass_rows_ > 0) K = std::max(1, std::min(K, pass_rows_ / pf_c_));
    for (int k = 0; k < 2 * K; ++k)
        if (!pf_emb_[k]) {
            ckc(cudaMallocHost(&pf_emb_[k], (size_t) pf_c_ * N * sizeof(float)), "pinned emb");
            own_pinned(pf_emb_[k]);
            ckc(cudaMallocHost(&pf_ple_[k], (size_t) pf_c_ * N * sizeof(float)), "pinned ple");
            own_pinned(pf_ple_[k]);
            ckc(cudaMallocHost(&pf_ctl_[k], (size_t) (1 + (pf_c_ / MAXT)) * 2 * sizeof(int)), "pinned ctl");
            own_pinned(pf_ctl_[k]);
            ckc(cudaMallocHost(&pf_after_[k], (size_t) N * sizeof(float)), "pinned after");
            own_pinned(pf_after_[k]);
        }
    // every pass's rows embedded on the host while the cards run the pass before; passes alternate between the two
    // halves of the sets
    const int h0 = (int) history_.size();
    const auto tok_at = [&](int q) -> int { return q < h0 ? history_[q] : tokens[q - h0]; };
    const size_t nt = tokens.size();
#if FNX_DF2
    if (df2_) {   // DFlash2: this read's last kRing positions are tapped into the lane's host ring (the drafter takes them at its first round)
        Df2Lane& L = df2_lane_[(size_t) cur_];
        if (!L.ring) {
            ckc(cudaHostAlloc((void**) &L.ring, (size_t) df2::kRing * df2::kTaps * df2::kNEmbd * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "df2 tap ring");
            own_pinned(L.ring);
        }
        df2_seg_end_ = h0 + (int) nt;
    }
#endif
    // a pass takes K chunks; with the hyper-connection split a short last chunk (not a whole pf_c_ rows) is a pass of its own, so it does not keep the whole chunks
    // before it from being split
    const bool lone_tail = pf_[0]->hcs;
    const int lone_min = (int) env_int("FNX_PF_HCSPLIT_MIN", 3);   // fewer whole chunks than this: the tail stays with them (one pass, the chunks hide each other's sums)
    auto chunks_of = [&](size_t i0, int half) {
        std::vector<PfChunk> ch;
        for (size_t i = i0; i < nt && (int) ch.size() < K; i += pf_c_) {
            const size_t e = std::min(nt, i + pf_c_);
            if (lone_tail && (int) ch.size() >= lone_min && e - i != (size_t) pf_c_) break;
            ch.push_back({std::vector<int>(tokens.begin() + i, tokens.begin() + e), e < nt ? tokens[e] : after, half * K + (int) ch.size()});
        }
        return ch;
    };
    auto embed = [&](size_t i0, const std::vector<PfChunk>& ch) {
        size_t i = i0;
        for (const PfChunk& c : ch) {
            embed_rows(h0 + (int) i, (int) c.tokens.size(), tok_at, pf_emb_[c.set], pf_ple_[c.set]);
            i += c.tokens.size();
        }
    };
    std::vector<PfChunk> ch = chunks_of(0, 0);
    embed(0, ch);
    for (size_t i = 0, k = 0; i < nt; ++k) {
        size_t cov = 0;
        for (const PfChunk& c : ch) cov += c.tokens.size();
        std::vector<PfChunk> next = chunks_of(i + cov, (int) ((k + 1) & 1));
        prefill_pass(ch, staging, [&] { embed(i + cov, next); });
        size_t e = i;
        for (const PfChunk& c : ch) {
            e += c.tokens.size();
            if (on_prefill_chunk) on_prefill_chunk((int) e);
        }
        i += cov;
        ch = std::move(next);
    }
}

#if FNX_DF2
// DFlash2: rows [row, row + n) of a chunk (absolute position pos_first for the first) on card b: R's streams with the pending FFN write folded in, their mean, into
// the lane's host ring slots (position % kRing; a run that wraps is two launches)
void Engine::df2_prompt_tap(int slot, int pos_first, int row, int n, CardBuf& b, const float* R, const float* bo, const float* inj) {
    Df2Lane& L = df2_lane_[(size_t) cur_];
    constexpr int64_t rowf = (int64_t) df2::kTaps * df2::kNEmbd;
    for (int done = 0; done < n;) {
        const int pos = pos_first + done, at = pos % df2::kRing, k = std::min(n - done, df2::kRing - at);
        df2::Bridge::tap(R + (size_t) (row + done) * D, bo + (size_t) (row + done) * N, inj + (size_t) (row + done) * HC, k, slot, L.ring + (size_t) at * rowf, rowf, b.st);
        done += k;
    }
}
#else
void Engine::df2_prompt_tap(int, int, int, int, CardBuf&, const float*, const float*, const float*) {}
#endif

void Engine::prefill_pass(const std::vector<PfChunk>& chunks, bool staging, const std::function<void()>& between) {
    const auto t_pass = std::chrono::steady_clock::now();
    const int nk = (int) chunks.size();
    int rows = 0;
    std::vector<int> pos0((size_t) nk);
    for (int j = 0; j < nk; ++j) pos0[(size_t) j] = pos_ + rows, rows += (int) chunks[(size_t) j].tokens.size();
    if (pos_ + rows > max_ctx_) throw Refused("prefill: context full");
    ensure_pages(pos_ + rows);
    const float eps = m_.eps;
    // each chunk's (position, tag) and each group's: a tag per group, so the recurrent history's parity steps per group
    for (int j = 0; j < nk; ++j) {
        const int Cj = (int) chunks[(size_t) j].tokens.size(), ngj = (Cj + MAXT - 1) / MAXT;
        int* ctl = pf_ctl_[chunks[(size_t) j].set];
        std::fill(ctl, ctl + (1 + (pf_c_ / MAXT)) * 2, 0);
        ctl[0] = pos0[(size_t) j];
        const unsigned tag0 = take_tags(ngj);
        ctl[1] = (int) tag0;
        for (int g = 0; g < ngj; ++g) {
            ctl[2 + 2 * g] = pos0[(size_t) j] + g * MAXT;
            ctl[3 + 2 * g] = (int) (tag0 + (unsigned) g);
        }
    }
    auto each = [&](auto&& f) {
#if FNX_HIP
        const RcclApi* rc = rccl_on_ ? &static_cast<RcclState*>(rccl_.get())->api : nullptr;
        if (rc) rc->GroupStart();
#endif
        for (int c = 0; c < NCARD; ++c) {
            ckc(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c], *pf_[c]);
        }
#if FNX_HIP
        if (rc) {   // the group's collectives launch now; what follows each sum on its stream is enqueued after them
            const int r = rc->GroupEnd();
            if (r != RCCL_SUCCESS) throw std::runtime_error(std::string("prefill: ncclGroupEnd: ") + rc->ErrorString(r));
            for (int c = 0; c < NCARD; ++c) {
                ckc(cudaSetDevice(cb_[c]->dev), "set");
                for (auto& fn : pf_[c]->post) fn();
                pf_[c]->post.clear();
            }
        }
#endif
    };
    // the chunk that runs: its rows, groups and buffers
    int C = 0, ng = 0, set = 0;
    // FNX_PF_HCSPLIT: this pass's hyper-connection reads are split between the cards (every chunk of the pass long enough to be worth it)
    bool hs = pf_[0]->hcs;
    static const int hs_full = (int) env_int("FNX_PF_HCSPLIT_FULL", 0);   // 1: every card reads all rows in one call; 2: all rows in two halves (the split's GEMM shapes)
      // debugging: every card reads all the rows (the swap still runs): the answers must equal the unsplit path's
    // fewer than hcs_min chunks in the pass (default 3): the swap and the sum have nothing to hide behind, and the split read is slower than the plain one (6.4K prompt: 1.8 -> 2.3 s)
    static const int hcs_min = (int) env_int("FNX_PF_HCSPLIT_MIN", 3);
    if ((int) chunks.size() < hcs_min) hs = false;
    for (const PfChunk& ck : chunks)
        if ((int) ck.tokens.size() < 512 || (pf_[0]->hcs_mode == 1 && (!pf_[0]->down_lt || (int) ck.tokens.size() != 4096))) hs = false;
    auto use = [&](int j) {
        C = (int) chunks[(size_t) j].tokens.size(), ng = (C + MAXT - 1) / MAXT, set = chunks[(size_t) j].set;
        for (Pf* p : pf_) {
            p->R = p->Rk[j], p->bo_f = p->bofk[j], p->inj_f = p->injfk[j], p->ctl = p->ctlk[j], p->bo_a = p->boak[j], p->inj_a = p->injak[j];
            if (p->hcs) p->mixed = p->mixedk[hs ? j : 0];
        }
    };
    // with an ExpertCache: the pass's expert tables, each layer's experts not in a slot at stage[l & 1] (as many as
    // fit, in expert order; the rest in place), the others in their slots
    if (staging)
        each([&](int, CardBuf& b, const CardW& w, Pf& p) {
            const ExpertCache& ec = w.ec;
            ckc(cudaMemcpyAsync(p.slot_h, ec.slot_of, (size_t) NL * NEXP * sizeof(int), cudaMemcpyDeviceToHost, b.st), "slots");
            ckc(cudaStreamSynchronize(b.st), "slots");
            p.staged = p.in_place = 0;
            for (int l = 0; l < NL; ++l) {
                const size_t gs = w.L[(size_t) l].gu_e_stride, ds = w.L[(size_t) l].down_e_stride;
                int n_ram = 0;
                for (int e = 0; e < NEXP; ++e) n_ram += p.slot_h[l * NEXP + e] < 0 ? 1 : 0;
                const int fit = std::min(n_ram, (int) (p.stage_bytes / (gs + ds)));
                char* gu_at = p.stage[l & 1];
                char* dn_at = gu_at + (size_t) fit * gs;
                std::vector<Pf::Copy>& cp = p.copies[l];
                cp.clear();
                auto add = [&](const char* src, char* dst, size_t n) {
                    if (!cp.empty() && cp.back().src + cp.back().n == src && cp.back().dst + cp.back().n == dst) cp.back().n += n;
                    else cp.push_back({src, dst, n});
                };
                int k = 0;
                for (int e = 0; e < NEXP; ++e) {
                    const size_t i = (size_t) l * NEXP + e;
                    const int s = p.slot_h[i];
                    if (s >= 0) {
                        const int q = ec.pool_of[l];
                        const char* at = ec.pool[q] + (size_t) (s - ec.s0[q]) * ec.slot_bytes[q];
                        p.ep_hgu[i] = at, p.ep_hdn[i] = at + ec.dn_off[q];
                    } else if (k < fit) {
                        p.ep_hgu[i] = gu_at + (size_t) k * gs, p.ep_hdn[i] = dn_at + (size_t) k * ds;
                        add(w.ec_host_gu[i], gu_at + (size_t) k * gs, gs);
                        add(w.ec_host_dn[i], dn_at + (size_t) k * ds, ds);
                        ++k;
                        p.staged += gs + ds;
                    } else {
                        p.ep_hgu[i] = w.ec_host_gu[i], p.ep_hdn[i] = w.ec_host_dn[i];
                        p.in_place += gs + ds;
                    }
                }
            }
            ckc(cudaMemcpyAsync(p.ep_gu, p.ep_hgu, (size_t) NL * NEXP * sizeof(char*), cudaMemcpyHostToDevice, b.st), "tables");
            ckc(cudaMemcpyAsync(p.ep_dn, p.ep_hdn, (size_t) NL * NEXP * sizeof(char*), cudaMemcpyHostToDevice, b.st), "tables");
        });
    // layer l's experts into stage[l & 1] once the layer two before is done with it
    auto stage_in = [&](int l) {
        each([&](int, CardBuf&, const CardW&, Pf& p) {
            ckc(cudaStreamWaitEvent(p.cs, p.ev_done[l & 1], 0), "stage wait");
            for (const Pf::Copy& c : p.copies[l]) ckc(cudaMemcpyAsync(c.dst, c.src, c.n, cudaMemcpyDefault, p.cs), "stage copy");
            ckc(cudaEventRecord(p.ev_ready[l & 1], p.cs), "stage ready");
        });
    };
    for (int j = 0; j < nk; ++j) {
        use(j);
        each([&](int, CardBuf& b, const CardW&, Pf& p) {
            // the embedding rows into xf (free until the first hc read), then into every stream of R
            ckc(cudaMemcpyAsync(p.xf, pf_emb_[set], (size_t) C * N * 4, cudaMemcpyHostToDevice, b.st), "emb");
            ckc(cudaMemcpyAsync(p.ctl, pf_ctl_[set], (size_t) (1 + (pf_c_ / MAXT)) * 2 * 4, cudaMemcpyHostToDevice, b.st), "ctl");
            k_pf_streams<<<grid_for((size_t) C * D), 256, 0, b.st>>>(p.xf, p.R, C);
        });
    }
    if (staging) stage_in(0);
    const auto t_up = std::chrono::steady_clock::now();
    // FNX_PF_OVERLAP: a half's sum goes round the ring on ring_st once its output is ready (and the routed experts' addend,
    // made here: the next chunk's experts reuse their buffers), while the pass's next chunk computes; `done` marks it
    // finished, and the chunk's next half waits for that alone. A layer runs every chunk's attention half, then every
    // chunk's FFN half, so there is a next chunk to compute meanwhile
    auto ring_async = [&](int c, CardBuf& b, Pf& p, float* x, const RingMoe* m, int j, cudaEvent_t done, int rows, bool own = false) {
        if (m) k_pfr_addend<<<grid_for((size_t) rows * N), 256, 0, b.st>>>(x, m->part, m->topw, m->logits, rows);
        ckc(cudaEventRecord(p.ev_go[j], b.st), "ring go");
        ckc(cudaStreamWaitEvent(p.ring_st, p.ev_go[j], 0), "ring go");
        pf_ring(p, c, x, rows, nullptr, b.ring.err, p.ring_st, own);
        if (p.rccl) p.post.push_back([&p, done] { ckc(cudaEventRecord(done, p.ring_st), "ring done"); });   // after the sum's conversion, once the group is closed
        else ckc(cudaEventRecord(done, p.ring_st), "ring done");
    };
    // with the overlap an attention chunk's merge (then wo and its sum) waits until the next chunk's hc read is issued,
    // its exchange crossing the links meanwhile; the layer's last chunk finishes after the layer's chunks
    auto finish_att = [&](int c, CardBuf& b, const CardW& w, Pf& p, int l) {
        const int j = p.att_pend, rows = p.att_pend_rows;
        if (j < 0) return;
        p.att_pend = -1;
        pf_attn_finish(p, c, rows, j & 1, b.st);
        pf_dense(p, w.L[l].wo.s, N, w.nq * HD, p.att, rows, p.boak[j], N, b.st);
        if (l < NL - 1 || mtp_) ring_async(c, b, p, p.boak[j], nullptr, j, p.ev_ra[j], rows, hs && l >= PLE_LAYER);   // the last layer's sum: unused
    };
    // FNX_PF_HCSPLIT: layer l's attention half (half 0) or FFN half (1) reads its hyper-connection on this card's rows of chunk j (use(j) done) and starts the
    // swap of the resulting `mixed` rows; the half's heavy part waits for the swap (ev_ag). The sum that feeds a split read is a reduce-scatter only (ring_async's
    // `own`): the attention half's output from layer PLE_LAYER on, the FFN half's from layer PLE_LAYER on (it feeds the next layer's read, which is split from
    // PLE_LAYER + 1 on: the PLE layer's own attention half writes every row of R first)
    auto hc_phase = [&](int l, int half, int j) {
        each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
            const LayerW& L = w.L[l];
            cudaStream_t s = b.st;
            const int own = (c + 1) % NCARD, r0 = hs_full ? 0 : own * C / NCARD, nr = hs_full ? C : (own + 1) * C / NCARD - r0;
            const int nq = hs_full == 2 ? 2 : 1;
            auto Q0 = [&](int q) { return hs_full == 2 ? q * (C / 2) : r0; };
            auto QN = [&](int q) { return hs_full == 2 ? (q == 0 ? C / 2 : C - C / 2) : nr; };
            __half* mh = p.hf16 ? p.mixhk[j] : nullptr;
            if (half == 0) {
                ckc(cudaStreamWaitEvent(s, p.ev_rf[j], 0), "ffn sum");
                for (int q = 0; q < nq; ++q) pf_hc_read_rows(p, p.R, p.bo_f, p.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, p.mixed, p.inj_a, Q0(q), QN(q), s, mh);
            } else {
                ckc(cudaStreamWaitEvent(s, p.ev_ra[j], 0), "attention sum");
                for (int q = 0; q < nq; ++q) pf_hc_read_rows(p, p.R, p.bo_a, p.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], eps, p.mixed, p.inj_f, Q0(q), QN(q), s, mh);
            }
            ckc(cudaEventRecord(p.ev_hc[j], s), "hc event");
            ckc(cudaStreamWaitEvent(p.ag_st, p.ev_hc[j], 0), "hc wait");
            const unsigned sq = ++p.hg_seq;
            const int hset = (int) (sq & 1u);
            if (p.hf16) pf_swap_rows(p.mixhk[j], N * sizeof(__half), C, c, p.hg_pieces, p.hg_in[hset], p.hg_flags[hset], p.nhg_in[hset], p.nhg_flags[hset], sq, p.ag_st);
            else pf_swap_rows(p.mixed, N * sizeof(float), C, c, p.hg_pieces, p.hg_in[hset], p.hg_flags[hset], p.nhg_in[hset], p.nhg_flags[hset], sq, p.ag_st);
            ckc(cudaEventRecord(p.ev_ag[j], p.ag_st), "swap done");
        });
    };
    // the heavy part's start after a split read: the swapped rows are here, and xh = f16 of mixed as the read leaves it (k_f32_f16 rounds as the read's
    // k_hcb_mixinj does)
    // the pass's last read, the draft layer's source: the attention sum folded into R on this card's rows, then the rows of R swapped (the draft layer reads all)
    auto draft_r_phase = [&](int l, int j) {
        each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
            const LayerW& L = w.L[l];
            cudaStream_t s = b.st;
            const int own = (c + 1) % NCARD, r0 = hs_full ? 0 : own * C / NCARD, nr = hs_full ? C : (own + 1) * C / NCARD - r0;
            const int nq = hs_full == 2 ? 2 : 1;
            auto Q0 = [&](int q) { return hs_full == 2 ? q * (C / 2) : r0; };
            auto QN = [&](int q) { return hs_full == 2 ? (q == 0 ? C / 2 : C - C / 2) : nr; };
            ckc(cudaStreamWaitEvent(s, p.ev_ra[j], 0), "attention sum");
            for (int q = 0; q < nq; ++q) pf_hc_read_rows(p, p.R, p.bo_a, p.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], eps, p.mixed, p.inj_f, Q0(q), QN(q), s);
            ckc(cudaEventRecord(p.ev_hc[j], s), "hc event");
            ckc(cudaStreamWaitEvent(p.ag_st, p.ev_hc[j], 0), "hc wait");
            const unsigned sq = ++p.rg_seq;
            const int rset = (int) (sq & 1u);
            pf_swap_rows(p.R, D * sizeof(float), C, c, p.hg_pieces, p.rg_in[rset], p.rg_flags + rset * 8, p.nrg_in[rset], p.nrg_flags + rset * 8, sq, p.ag_st);
            ckc(cudaEventRecord(p.ev_rg[j], p.ag_st), "swap done");
        });
    };
    auto swap_join = [&](CardBuf& b, Pf& p, int j) {
        ckc(cudaStreamWaitEvent(b.st, p.ev_ag[j], 0), "swap join");
        if (p.hf16) {   // every row from the f16 values (own rows too: both cards must hold the same numbers)
            k_f16_f32<<<grid_for((size_t) C * N), 256, 0, b.st>>>(p.mixhk[j], C, N, p.mixed, N);
            ckc(cudaMemcpyAsync(p.xh, p.mixhk[j], (size_t) C * N * sizeof(__half), cudaMemcpyDeviceToDevice, b.st), "mixh");
        } else k_f32_f16<<<grid_for((size_t) C * N), 256, 0, b.st>>>(p.mixed, (size_t) C * N, p.xh);
    };
    for (int l = 0; l < NL; ++l) {
        if (staging && l + 1 < NL - 1) stage_in(l + 1);   // the last layer runs no experts
#if FNX_DF2
        // DFlash2: layer l - 1's completed output (its FFN half's sum folded into R as the next read would) is the tap of the rows of the last kRing positions of
        // this read. Before this layer's reads fold the sum into R; in the split read (HCSPLIT) a card holds the sums of its own rows only and taps those
        if (df2_ && l > 0 && df2_slot_[(size_t) (l - 1)] >= 0) {
            const int slot = df2_slot_[(size_t) (l - 1)];
            for (int j2 = 0; j2 < nk; ++j2) {
                use(j2);
                const int lo = std::max(0, df2_seg_end_ - df2::kRing - pos0[(size_t) j2]);
                if (lo >= C) continue;
                const bool own_rows = hs && l - 1 >= PLE_LAYER;
                each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                    int a = lo, z = C;
                    if (own_rows) {
                        const int own = (c + 1) % NCARD, r0 = hs_full ? 0 : own * C / NCARD, nr = hs_full ? C : (own + 1) * C / NCARD - r0;
                        a = std::max(a, r0), z = std::min(z, r0 + nr);
                    } else if (c != 0) {
                        return;   // every card holds all the sums: the first taps
                    }
                    if (a >= z) return;
                    if (p.overlap) ckc(cudaStreamWaitEvent(b.st, p.ev_rf[j2], 0), "ffn sum");
                    df2_prompt_tap(slot, pos0[(size_t) j2] + a, a, z - a, b, p.R, p.bo_f, p.inj_f);
                });
            }
        }
#endif
        for (int j = 0; j < nk; ++j) {
            use(j);
            // ---- attention half: the last layer's FFN sum (pending in bo_f) first
            const bool pending = l > 0;
            const bool sa = hs && l > PLE_LAYER;   // this half's read is split (the PLE layer's own attention half is not: it writes every row of R)
            if (sa && j == 0)
                for (int j2 = 0; j2 < nk; ++j2) use(j2), hc_phase(l, 0, j2);
            if (sa) use(j);
            each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                const LayerW& L = w.L[l];
                cudaStream_t s = b.st;
                if (sa) swap_join(b, p, j);
                else if (pending && p.overlap) ckc(cudaStreamWaitEvent(s, p.ev_rf[j], 0), "ffn sum");
                bool apply = pending;
                if (!sa && l == PLE_LAYER) {
                    ckc(cudaMemcpyAsync(p.ple_emb, pf_ple_[set], (size_t) C * N * 4, cudaMemcpyHostToDevice, s), "ple");
                    hc_write(p.R, p.bo_f, p.inj_f, C, s);
                    apply = false;
                    pf_dense(p, w.ple_key.s, w.ple_key.n_out, w.ple_key.n_in, p.ple_emb, C, p.ple_key, w.ple_key.n_out, s);
                    pf_dense(p, w.ple_value.s, w.ple_value.n_out, w.ple_value.n_in, p.ple_emb, C, p.ple_val, w.ple_value.n_out, s);
                    // per group (the conv history runs row by row); ple_tmp holds a group's gated and normed rows
                    for (int g = 0; g < ng; ++g) {
                        const int r0 = g * MAXT, Tg = std::min(MAXT, C - r0);
                        ple_post(p.R + (size_t) r0 * D, p.ple_key + (size_t) r0 * D, p.ple_val + (size_t) r0 * N, w.ple_norm_key,
                                 w.ple_norm_query, w.ple_norm_conv, w.ple_conv, b.ple_hist, b.ple_hsnap, Tg, eps, s, p.ple_tmp);
                    }
                }
                if (!sa)
                    pf_hc_read(p, p.R, apply ? p.bo_f : nullptr, p.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, p.mixed,
                               p.inj_a, C, s);
                if (!is_attn(l)) {
                    const int nv = w.nv(), ch = w.conv_ch(), rs_qkvz = ch + nv * HS;
                    pf_dense_xh(p, L.qkvz.s, rs_qkvz, N, C, p.qkvz, rs_qkvz, s);   // xh: the hc read's f16 of mixed
                    pf_sgemm(p, L.ab, 2 * nv, N, p.mixed, C, p.ab, 2 * nv);
                    gdn_prompt(p.qkvz, p.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], p.ctl, ng, w.nk, nv, w.vmap, C, eps,
                               p.gdn_xc, p.gdn_sq, p.y32, p.gdn_ch, s);
                    pf_dense(p, L.ssm_out.s, N, nv * HS, p.y32, C, p.bo_a, N, s);
                } else {
                    finish_att(c, b, w, p, l);   // the previous chunk's, before qg is this chunk's
                    // mixed to f16 again (finish_att's wo used xh), then k and v from that
                    pf_dense(p, L.wq.s, NQH * 2 * HD, N, p.mixed, C, p.qg, NQH * 2 * HD, s);
                    pf_dense_xh(p, L.wk.s, NKVH * HD, N, C, p.k, NKVH * HD, s);
                    pf_dense_xh(p, L.wv.s, NKVH * HD, N, C, p.v, NKVH * HD, s);
                    attn_prep(p.qg, p.k, L.q_norm, L.k_norm, NQH, NKVH, p.ctl, C, eps, rope_of(b.mrope), s);
                    kv_append(p.k, p.v, b.kc[l], b.vc[l], b.pg, b.att_ring.card, C, p.ctl, s);
                    // the indexer: bf16 projections as f32 (llama.cpp's choice past 8 rows on a V100)
                    k_bf16_f32<<<grid_for((size_t) 512 * N), 256, 0, s>>>(L.idx_wq, (size_t) 512 * N, p.widx);
                    k_bf16_f32<<<grid_for((size_t) 128 * N), 256, 0, s>>>(L.idx_wk, (size_t) 128 * N, p.widx + (size_t) 512 * N);
                    pf_sgemm(p, p.widx, qsa_rows(), N, p.mixed, C, p.yidx, qsa_rows());
                    qsa_index(p.yidx, L.idx_qn, L.idx_kn, b.qsa_ring[l], b.qsa_pooled[l], b.pg, p.qidx, p.ctl, C, eps, rope_of(b.mrope), s);
                    // groups of rows (each with the position and tag of its first MAXT group) for QSA's block selection:
                    // qsa_pf_rows (FNX_PF_QSA_ROWS), or attn_rows() when the decode attention takes the groups too
                    const int AR = p.attn_dec ? attn_rows() : p.qsa_pf_rows;
                    for (int g = 0; g * AR < C; ++g) {
                        const int r0 = g * AR, Tg = std::min(AR, C - r0);
                        const int* gctl = p.ctl + 2 + 2 * (r0 / MAXT);
                        qsa_select(p.qidx + (size_t) r0 * 512, b.qsa_pooled[l], b.pg, p.scores, p.cells + (size_t) r0 * qsa_maxsel(),
                                   p.ncells + r0, max_ctx_ / 4, gctl, Tg, b.att_ring.card, s);
                        if (!p.attn_dec) continue;
                        attn(p.qg + (size_t) r0 * NQH * 2 * HD, b.kc[l], b.vc[l], b.pg, p.att + (size_t) r0 * w.nq * HD, b.att_ring,
                             (const unsigned*) (gctl + 1), 128 + l, gctl, Tg, b.att_part, s, p.cells + (size_t) r0 * qsa_maxsel(),
                             p.ncells + r0);
                    }
                    if (!p.attn_dec && p.overlap) {
                        pf_attn_start(p, b, b.kc[l], b.vc[l], c, C, j & 1, s);   // merged by finish_att
                        p.att_pend = j, p.att_pend_rows = C;
                    } else {
                        if (!p.attn_dec) pf_attention(p, b, b.kc[l], b.vc[l], c, C, s);
                        pf_dense(p, L.wo.s, N, w.nq * HD, p.att, C, p.bo_a, N, s);
                    }
                }
            });
            if (l == NL - 1 && !mtp_) continue;   // the last layer's FFN half changes no state
            each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                if (p.ring_sm) ring_sum(p.ring, p.bo_a, C, (const unsigned*) (p.ctl + 1), 2 * l, b.st);
                else if (p.overlap) {
                    if (!is_attn(l) || p.attn_dec) ring_async(c, b, p, p.bo_a, nullptr, j, p.ev_ra[j], C, hs && l >= PLE_LAYER);   // else finish_att's
                } else pf_ring(p, c, p.bo_a, C, nullptr, b.ring.err, b.st);
            });
        }
        if (is_attn(l)) each([&](int c, CardBuf& b, const CardW& w, Pf& p) { finish_att(c, b, w, p, l); });   // the last chunk
        if (l == NL - 1 && !mtp_) continue;
        for (int j = 0; j < nk; ++j) {
            use(j);
            each([&](int, CardBuf& b, const CardW&, Pf& p) {
                if (p.overlap) ckc(cudaStreamWaitEvent(b.st, p.ev_ra[j], 0), "attention sum");
            });
            const bool sf = hs && l >= PLE_LAYER;   // this half's read is split
            if (sf && j == 0) {
                for (int j2 = 0; j2 < nk; ++j2) {
                    use(j2);
                    if (l == NL - 1) draft_r_phase(l, j2);
                    else hc_phase(l, 1, j2);
                }
                use(j);
            }
            if (l == NL - 1 && sf) continue;   // read and swap issued above
            if (l == NL - 1) {
                // the draft layer's source: R with the last attention write folded in, as a window's FFN-half read leaves it
                each([&](int, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    pf_hc_read(p, p.R, p.bo_a, p.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], eps, p.mixed, p.inj_f, C, b.st);
                });
                continue;
            }
            // ---- FFN half
            each([&](int, CardBuf& b, const CardW& w, Pf& p) {
                const LayerW& L = w.L[l];
                cudaStream_t s = b.st;
                const int P = C * TOPK, ffs = w.ff1 - w.ff0;
                DW gu_e = L.gu_e, down_e = L.down_e;
                if (staging) {
                    gu_e.ep = p.ep_gu + (size_t) l * NEXP, down_e.ep = p.ep_dn + (size_t) l * NEXP;
                    if (j == 0) ckc(cudaStreamWaitEvent(s, p.ev_ready[l & 1], 0), "stage in");
                }
                if (sf) swap_join(b, p, j);
                else pf_hc_read(p, p.R, p.bo_a, p.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], eps, p.mixed, p.inj_f, C, s);
                if (p.moe_q8) quantize_q8_1(p.mixed, p.xq, N, C, s);   // else xh holds the hc read's f16 of mixed
                pf_sgemm(p, L.router, NEXP + 1, N, p.mixed, C, p.logits, NEXP + 1);
                k_pf_top10<<<(C + 7) / 8, 256, 0, s>>>(p.logits, C, p.topi, p.topw);
                ckc(cudaMemsetAsync(p.cnt, 0, NEXP * 4, s), "cnt");
                k_pf_count<<<(P + 255) / 256, 256, 0, s>>>(p.topi, P, p.cnt);
                if (b.route_cnt) route_count(p.topi, P, b.route_cnt + (size_t) (NL + l) * NEXP, s);
                k_pf_jobbase<<<1, NEXP, 0, s>>>(p.cnt, p.jbase, p.njobs);
                k_pf_fill<<<NEXP, PF_FILL_THREADS, 0, s>>>(p.topi, P, p.cnt, p.jbase, p.jobs_gu, p.jobs_dn);
                if (p.moe_q8) {
                    moe_gate_up_big(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, p.xq, p.gu, s);
                    moe_down_big(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.gu, p.part, s);
                } else {
                    moe_gate_up_mma(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, reinterpret_cast<const uint16_t*>(p.xh), p.gu, s);
                    swiglu_h16(p.gu, ffs, P, p.h16, s);
                    moe_down_mma(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.h16, p.part, s);
                }
                if (staging && j == nk - 1) ckc(cudaEventRecord(p.ev_done[l & 1], s), "stage done");
                // the shared expert: rows P + t of part
                pf_dense_xh(p, L.sh_gu.s, 2 * ffs, N, C, p.shg, 2 * ffs, s);   // xh: still the f16 of mixed
                k_pf_swiglu<<<grid_for((size_t) C * ffs), 256, 0, s>>>(p.shg, ffs, C, p.shh);
                k_q8s_f16<<<grid_for((size_t) N * ffs), 256, 0, s>>>(L.sh_down.s.q, L.sh_down.s.d, (size_t) N * ffs, p.wh);
                pf_gemm(p, p.wh, N, ffs, p.shh, C, p.part + (size_t) P * N, N, s);
                ckc(cudaGetLastError(), "ffn");
            });
            each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                RingMoe m;
                m.part = p.part;
                m.topw = p.topw;
                m.logits = p.logits;
                if (p.ring_sm) ring_sum(p.ring, p.bo_f, C, (const unsigned*) (p.ctl + 1), 2 * l + 1, b.st, m);
                else if (p.overlap) ring_async(c, b, p, p.bo_f, &m, j, p.ev_rf[j], C, hs && l >= PLE_LAYER);
                else pf_ring(p, c, p.bo_f, C, &m, b.ring.err, b.st);
            });
        }
    }
    static const bool draft8 = env_flag("FNX_PF_DRAFT8");
    if (mtp_ && !draft8)
        for (int j = 0; j < nk; ++j) {
            use(j);
            embed_token(chunks[(size_t) j].after, pos0[(size_t) j] + C, pf_after_[set]);
            if (hs)
                each([&](int, CardBuf& b, const CardW&, Pf& p) { ckc(cudaStreamWaitEvent(b.st, p.ev_rg[j], 0), "R swap"); });
            prefill_draft(C, set);
        }
    if (hs && mtp_ && draft8)
        for (int j = 0; j < nk; ++j)
            each([&](int, CardBuf& b, const CardW&, Pf& p) { ckc(cudaStreamWaitEvent(b.st, p.ev_rg[j], 0), "R swap"); });
    const auto t_enq = std::chrono::steady_clock::now();
    between();
    unsigned err = 0;
    // bounded: a copy-engine ring step waits on its flag without a timeout of its own
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    each([&](int, CardBuf& b, const CardW&, Pf&) {
        for (;;) {
            const cudaError_t q = cudaStreamQuery(b.st);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) ckc(q, "prefill sync");
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("prefill: a pass did not finish in 120 s (a ring step's flag never came?)");
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        unsigned e = 0;
        ckc(cudaMemcpy(&e, b.ring.err, 4, cudaMemcpyDeviceToHost), "err");
        err |= e;
    });
    if (err) throw std::runtime_error("prefill: ring error " + std::to_string(err));
    const auto t_main = std::chrono::steady_clock::now();
    if (mtp_ && draft8)
        for (int j = 0; j < nk; ++j) {
            use(j);
            const std::vector<int>& tokens = chunks[(size_t) j].tokens;
            for (int r0 = 0; r0 < C; r0 += MAXT) {
                const int T = std::min(MAXT, C - r0);
                std::vector<const float*> src;
                for (Pf* p : pf_) src.push_back(p->R + (size_t) r0 * D);
                std::vector<int> next(tokens.begin() + r0 + 1, tokens.begin() + r0 + T);
                next.push_back(r0 + T < C ? tokens[(size_t) (r0 + T)] : chunks[(size_t) j].after);
                draft_rows(src, T, next, pos0[(size_t) j] + r0);
            }
        }
    use(0);
    for (const PfChunk& c : chunks) history_.insert(history_.end(), c.tokens.begin(), c.tokens.end());
    pos_ += rows;
#if FNX_DF2
    if (df2_) {   // the positions this pass tapped: [a, pos_) with a the later of the pass's start and the read's last kRing positions
        Df2Lane& L = df2_lane_[(size_t) cur_];
        const long a = std::max<long>(pos_ - rows, (long) df2_seg_end_ - df2::kRing), e = pos_;
        if (a < e) {
            if (L.hi == a && L.hi > L.lo) L.hi = e, L.lo = std::max<long>(L.lo, e - df2::kRing);
            else L.lo = a, L.hi = e;
        }
    }
#endif
    static const bool timing = env_flag("FNX_PF_TIMING");
    if (timing) {
        std::fprintf(stderr, "prefill pass at %d: %d chunks, %d rows in %.1f ms (embed + upload %.1f ms, enqueued in %.1f ms, draft layer %.1f ms)",
                     pos_ - rows, nk, rows, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pass).count(),
                     std::chrono::duration<double, std::milli>(t_up - t_pass).count(),
                     std::chrono::duration<double, std::milli>(t_enq - t_pass).count(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_main).count());
        if (staging)
            for (int c = 0; c < NCARD; ++c)
                std::fprintf(stderr, "; card %d experts %zu MiB copied in, %zu MiB read in place", c, pf_[c]->staged >> 20, pf_[c]->in_place >> 20);
        std::fprintf(stderr, "\n");
    }
}

}  // namespace fnx
