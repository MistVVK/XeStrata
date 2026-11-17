// Expert gate|up for Q4_K weights (included by kernels.cu after moe.cuh). A lane takes whole 64-value groups of a row
// (32 bytes of nibbles: low = sub-block 2g, high = 2g+1, plus the block's 16-byte header), three 16-byte loads per
// group; ggml's dot reads the same data in 2- and 4-byte pieces. Same arithmetic as vec_dot_q4_K_q8_1: per sub-block
// d8 * (sc * dot(q, a)) and d8 * (m * sum(a)) on the quantized activations, then d * . - dmin * .
namespace {

constexpr int Q4K_THREADS = 256, Q4K_LPR = 8, Q4K_ROWS = Q4K_THREADS / Q4K_LPR;

// byte k (0..11) of the 12 scale bytes held in the header words y, z, w
__device__ __forceinline__ unsigned q4k_sbyte(const uint4& h, int k) {
    const unsigned word = k < 4 ? h.y : k < 8 ? h.z : h.w;
    return (word >> (8 * (k & 3))) & 0xFFu;
}
// 6-bit scale and min of sub-block j (get_scale_min_k4)
__device__ __forceinline__ void q4k_scale_min(const uint4& h, int j, float& sc, float& m) {
    if (j < 4) {
        sc = (float) (q4k_sbyte(h, j) & 63);
        m = (float) (q4k_sbyte(h, j + 4) & 63);
    } else {
        sc = (float) ((q4k_sbyte(h, j + 4) & 0xF) | ((q4k_sbyte(h, j - 4) >> 6) << 4));
        m = (float) ((q4k_sbyte(h, j + 4) >> 4) | ((q4k_sbyte(h, j) >> 6) << 4));
    }
}

// rows of N columns: N / 64 groups, Q4K_GPL per lane
constexpr int Q4K_GPL = N / 64 / Q4K_LPR;
static_assert(N % (64 * Q4K_LPR) == 0, "groups per lane");

// the lane's groups [I0, I1): their weights loaded together, then each token's dots in group order
template <int I0, int I1>
__device__ __forceinline__ void q4k_groups(const char* wrow, int lane8, int n, const block_q8_1* y, const int* ys, int y_stride,
                                           float* tmp) {
    constexpr int G = I1 - I0;
    uint4 hdr[G], q0[G], q1[G];
#pragma unroll
    for (int ii = 0; ii < G; ++ii) {
        const int g = lane8 + (I0 + ii) * Q4K_LPR, b = g >> 2, jg = g & 3;
        const char* blk = wrow + (size_t) b * 144;
        hdr[ii] = __ldg(reinterpret_cast<const uint4*>(blk));
        q0[ii] = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 32 * jg));
        q1[ii] = __ldg(reinterpret_cast<const uint4*>(blk + 32 + 32 * jg));
    }
#pragma unroll
    for (int ii = 0; ii < G; ++ii) {
        const int i = I0 + ii;
        const int g = lane8 + i * Q4K_LPR, b = g >> 2, jg = g & 3;
        const float d = __half2float(__ushort_as_half((uint16_t) (hdr[ii].x & 0xffff)));
        const float dmin = __half2float(__ushort_as_half((uint16_t) (hdr[ii].x >> 16)));
        float sc0, m0, sc1, m1;
        q4k_scale_min(hdr[ii], 2 * jg, sc0, m0);
        q4k_scale_min(hdr[ii], 2 * jg + 1, sc1, m1);
        const unsigned qw[8] = {q0[ii].x, q0[ii].y, q0[ii].z, q0[ii].w, q1[ii].x, q1[ii].y, q1[ii].z, q1[ii].w};
        // rolled over the job's tokens: unrolled, the body repeats per token and the kernel no longer fits the
        // instruction cache it is fetched into cold at every launch
#pragma unroll 1
        for (int j = 0; j < n; ++j) {
            const block_q8_1* yb = y + (size_t) ys[j] * y_stride + b * 8 + jg * 2;
            // q8_1 blocks are 36 bytes with qs at offset 4: 4-byte loads
            const int* a0 = reinterpret_cast<const int*>(yb[0].qs);
            const int* a1 = reinterpret_cast<const int*>(yb[1].qs);
            int dl = 0, dh = 0, sl = 0, sh = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const unsigned q = qw[w];
                const int al = a0[w], ah = a1[w];
                dl = __dp4a((int) (q & 0x0F0F0F0Fu), al, dl);
                dh = __dp4a((int) ((q >> 4) & 0x0F0F0F0Fu), ah, dh);
                sl = __dp4a(0x01010101, al, sl);
                sh = __dp4a(0x01010101, ah, sh);
            }
            const float d8l = __low2float(yb[0].ds), d8h = __low2float(yb[1].ds);
            tmp[j] += d * (d8l * (sc0 * (float) dl) + d8h * (sc1 * (float) dh)) - dmin * (d8l * (m0 * (float) sl) + d8h * (m1 * (float) sh));
        }
    }
}

// the lane's groups in two passes, the second's loads after the first's dots (a compiler barrier between): all five
// held at once took the registers of 3 blocks an SM. Each token's sum takes the groups in the same order
template <int TT>
__device__ __forceinline__ void q4k_row(const char* wrow, int lane8, int n, const block_q8_1* y, const int* ys, int y_stride,
                                        float* tmp) {
    constexpr int SPLIT = (Q4K_GPL + 1) / 2;
    q4k_groups<0, SPLIT>(wrow, lane8, n, y, ys, y_stride, tmp);
    asm volatile("" ::: "memory");
    q4k_groups<SPLIT, Q4K_GPL>(wrow, lane8, n, y, ys, y_stride, tmp);
}

// sd.rows: the launch's first sd_rows rows of blocks make the shared expert's swiglu and down, a q8s block or two each
// (one row of 14 blocks doing ~23 each took the launch 34 -> 61 us); first, so they start first (TT: the window's rows)
template <int TT>
__global__ void __launch_bounds__(Q4K_THREADS) k_moe_gu_q4k(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                          const block_q8_1* __restrict__ y, float* __restrict__ out, const ShexpDown sd,
                                                          int sd_rows) {
    static_assert(Q4K_THREADS == Q8S_THREADS, "the shared expert's q8s blocks in the gate|up launch");
    if ((int) blockIdx.y < sd_rows) {
        shexp_down_blocks<TT>(sd, blockIdx.y * gridDim.x + blockIdx.x, sd_rows * gridDim.x);
        return;
    }
    const int ji = blockIdx.y - sd_rows;
    if (ji >= *njobs) return;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    const int row = blockIdx.x * Q4K_ROWS + threadIdx.x / Q4K_LPR, lane8 = threadIdx.x % Q4K_LPR;
    const bool live = row < w.nrows;
    float tmp[TT] = {};
    if (live) q4k_row<TT>(moe_expert(w, jb.w) + (size_t) row * w.row_a, lane8, jb.n, y, jb.y, N / 32, tmp);
    job_store<Q4K_LPR, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

// k_moe_gu_q4k for BigJobs: the job's pairs MAXT at a time through the row (re-reads of the row hit the caches)
__global__ void __launch_bounds__(Q4K_THREADS) k_moe_gu_q4k_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                              const block_q8_1* __restrict__ y, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    load_bigjob(jb, jobs + ji);
    const int row = blockIdx.x * Q4K_ROWS + threadIdx.x / Q4K_LPR, lane8 = threadIdx.x % Q4K_LPR;
    const bool live = row < w.nrows;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int nb = min(MAXT, jb.n - b0);
        float tmp[MAXT] = {};
        if (live) q4k_row<MAXT>(moe_expert(w, jb.w) + (size_t) row * w.row_a, lane8, nb, y, jb.y + b0, N / 32, tmp);
        job_store<Q4K_LPR>(tmp, nb, live, row, jb.out + b0, out, w.nrows);
    }
}

void gu_q4k_launch(int T, const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const block_q8_1* y, float* out,
                   cudaStream_t s, const ShexpDown& sd = ShexpDown{}) {
    if (w.ncols != N || w.row_a != (size_t) N / 256 * 144) throw std::runtime_error("moe gate|up q4_K: row layout");
    const int gx = (w.nrows + Q4K_ROWS - 1) / Q4K_ROWS, sd_blocks = (sd.rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
    const int sd_rows = sd.rows ? (sd_blocks + gx - 1) / gx : 0;
    const dim3 grid(gx, njobs_max + sd_rows);
    switch (T) {
        case 1: k_moe_gu_q4k<1><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 2: k_moe_gu_q4k<2><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 3: k_moe_gu_q4k<3><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 4: k_moe_gu_q4k<4><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 5: k_moe_gu_q4k<5><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 6: k_moe_gu_q4k<6><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 7: k_moe_gu_q4k<7><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 8: k_moe_gu_q4k<8><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 9: k_moe_gu_q4k<9><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 10: k_moe_gu_q4k<10><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 11: k_moe_gu_q4k<11><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 12: k_moe_gu_q4k<12><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 13: k_moe_gu_q4k<13><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 14: k_moe_gu_q4k<14><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 15: k_moe_gu_q4k<15><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 16: k_moe_gu_q4k<16><<<grid, Q4K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        default: throw std::runtime_error("moe gate|up q4_K: T out of range");
    }
}

}  // namespace
