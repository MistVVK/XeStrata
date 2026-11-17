// Attention over a prompt chunk on INT8 tensor cores (included by kernels.cu after attn_prompt.cuh; FNX_PFX_ATTN_I8, on by default on NVIDIA).
//
// Strata's prompt_attn_i8v3_kernel (src/kernels/cuda/qsa_prompt_attn.cu) adapted to the split's K/V pages and to the cross-card merge: one block of
// four warps per (query row, KV head); a warp owns 64 of the 256 head dimensions; the K and V codes of 32 cells at a time come through cp.async into
// the warp's own stage (one stage each, so the warp needs no block barrier to reuse it); q.k and p.v are INT8 tensor-core products. The query (and,
// for p.v, the probabilities times the V scale) are split into three 8-bit parts per row (h, l: signed bytes, m: unsigned) so that the product with
// the exact int8 K/V codes carries 24 bits, as in Strata's kernel; the parts' exact int32 sums are joined in f32.
//
// Differences from Strata's kernel:
//  - the split's int8 K/V rows carry an fp16 scale per 32 values (KV8_GROUP), Strata's per 64: a warp's 64 dimensions have TWO scales a cell. q.k adds the
//    two 32-dimension k-steps' joined sums scaled each by its group's K scale; p.v runs its three-part product once with p scaled by the first group's V
//    scale (the columns of dimensions 0..31 take that result) and once with the second group's (dimensions 32..63): twice the p.v products.
//  - the keys of a row are the card's own: the QSA cell list of this card (cells + t * QSA_MAXSEL, n_cells[t]) or, dense, its rows 0 .. kv_owned - 1,
//    through the page table of KvPages.
//  - the output is the card's partial (acc, max, sum) per head for k_pf_attn_merge (natural-log max), not the normalised attention.
//  - q is the split's qg rows (query then gate, 2 * HD floats a head).
#if !FNX_HIP && FNX_ARCH >= 80
namespace {

constexpr int AI_CH = 32;          // cells a chunk
constexpr int AI_THREADS = 128;    // four warps, 64 dimensions each
constexpr int AI_G = ATT_QPK;      // query heads of a KV head
static_assert(HD == 256 && AI_G <= 16 && KV8_GROUP == 32, "k_pf_attn_i8: four 64-dimension warps, two 32-value scale groups each");

__device__ __forceinline__ void ai_imma(int* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// the low part's product: unsigned A (0..255), signed B
__device__ __forceinline__ void ai_imma_us(int* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// x (|x| <= 127) as X = rint(x * 65536) = h * 65536 + l * 256 + m exactly: h, l signed bytes, m an unsigned one
__device__ __forceinline__ void ai_split3(float x, int& h, int& l, int& m) {
    const int X = __float2int_rn(x * 65536.0f);
    h = (X + 32768) >> 16;
    const int r = X - (h << 16);
    l = r >> 8;
    m = r & 255;
}
__device__ __forceinline__ uint32_t ai_pack4(int a, int b, int c, int d) {
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (d & 0xff) << 24);
}
__device__ __forceinline__ float ai_join3(int h, int l, int m) { return (float) h + (float) ((l << 8) + m) * (1.0f / 65536.0f); }
// four words of four bytes each (word i = cell i) -> four words of the same byte position across the cells
__device__ __forceinline__ void ai_tr4x4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t* o) {
    const uint32_t t0 = __byte_perm(w0, w1, 0x5140), t1 = __byte_perm(w0, w1, 0x7362);
    const uint32_t t2 = __byte_perm(w2, w3, 0x5140), t3 = __byte_perm(w2, w3, 0x7362);
    o[0] = __byte_perm(t0, t2, 0x5410);
    o[1] = __byte_perm(t0, t2, 0x7632);
    o[2] = __byte_perm(t1, t3, 0x5410);
    o[3] = __byte_perm(t1, t3, 0x7632);
}
__device__ __forceinline__ int ai_swz(int cell, int byte) {   // byte offset of (cell, byte) in a warp's stage slice of 64 bytes a cell
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__device__ __forceinline__ void ai_cp16(void* smem, const void* gmem, bool valid) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ai_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void ai_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

struct AiSmem {
    int8_t kv[4][2][AI_CH][64];     // warp, K/V, cell, the warp's 64 dimensions in 16-byte pieces XOR-swizzled by the cell
    float sc[4][2][AI_CH][2];       // warp, K/V, cell, the two 32-value groups' scales
    float part[4][AI_G][AI_CH + 1]; // q.k of each warp's dimensions (the real head rows only)
    float p[AI_G][AI_CH + 1];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__global__ void __launch_bounds__(AI_THREADS, 3) k_pf_attn_i8(const float* __restrict__ qg, const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                              KvPages g, const int* __restrict__ pos0p, const int* __restrict__ cells,
                                                              const int* __restrict__ n_cells, int card, int C, int r0, float* __restrict__ part) {
    extern __shared__ __align__(16) unsigned char ai_raw[];
    AiSmem& S = *reinterpret_cast<AiSmem*>(ai_raw);
    const int tq = r0 + blockIdx.x, kvh = blockIdx.y;
    const int* list = cells && n_cells[tq] >= 0 ? cells + (size_t) tq * QSA_MAXSEL : nullptr;
    const int n = list ? n_cells[tq] : kv_owned(*pos0p + tq, card);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;
    const float* q = qg + ((size_t) tq * NQH + (size_t) kvh * AI_G) * 2 * HD;
    if (t < 16) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
        S.alpha[t] = 0.0f;
    }

    // q: rows gid / gid + 8 (heads; 12..15 are padding), this warp's 64 dimensions as the A fragments of two k-steps of 32: three 8-bit parts with a
    // scale per row
    uint32_t qa[3][2][4];
    float qinv[2];
    {
        float x[2][2][2][4];   // [row half][k-step][k half][4 dimensions]
        float mx[2] = {0.0f, 0.0f};
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            const int row = gid + rh * 8;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (row < AI_G) v = *reinterpret_cast<const float4*>(q + (size_t) row * 2 * HD + dim0 + ks * 32 + kh * 16 + tig * 4);
                    x[rh][ks][kh][0] = v.x; x[rh][ks][kh][1] = v.y; x[rh][ks][kh][2] = v.z; x[rh][ks][kh][3] = v.w;
                    mx[rh] = fmaxf(mx[rh], fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], o));
        }
        float up[2];
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            up[rh] = mx[rh] > 0.0f ? 127.0f / mx[rh] : 0.0f;
            qinv[rh] = mx[rh] * (1.0f / 127.0f);
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int r = 0; r < 4; ++r) {   // a0: row gid, k 0..15 half; a1: row gid + 8; a2: row gid, k 16..31; a3
                const int rh = r & 1, kh = r >> 1;
                int hh[4], ll[4], mm[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) ai_split3(x[rh][ks][kh][e] * up[rh], hh[e], ll[e], mm[e]);
                qa[0][ks][r] = ai_pack4(hh[0], hh[1], hh[2], hh[3]);
                qa[1][ks][r] = ai_pack4(ll[0], ll[1], ll[2], ll[3]);
                qa[2][ks][r] = ai_pack4(mm[0], mm[1], mm[2], mm[3]);
            }
    }

    const int n_chunks = (n + AI_CH - 1) / AI_CH;
    const int8_t* kb = reinterpret_cast<const int8_t*>(kc);
    const int8_t* vb = reinterpret_cast<const int8_t*>(vc);
    auto cell_of = [&](int c) -> int { return c < n ? (list ? __ldg(list + c) : c) : -1; };
    // a cell's first code byte (from the layer's K or V pointer) and first scale (halves from it); -1: no cell
    auto rows_of = [&](int cell, long long& rb, long long& rs) {
        if (cell < 0) {
            rb = rs = -1;
            return;
        }
        rb = (long long) kvq_row(g, cell) + (long long) kvh * HD;
        rs = (long long) kvq_scale(g, cell) + (long long) kvh * (HD / KV8_GROUP);
    };
    // this warp's 64-dimension slice of K (kv = 0) or V (1) of the chunk whose cells the lanes hold, and the cells' two scales
    auto issue = [&](long long rb, long long rs, int kv) {
        const int8_t* pool = kv == 0 ? kb : vb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, rb, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr + dim0 + pc * 16 : 0;
            ai_cp16(&S.kv[warp][kv][0][0] + ai_swz(cell, pc * 16), pool + off, ok);
        }
        const __half* sc = kv == 0 ? kc : vc;
        S.sc[warp][kv][lane][0] = rb >= 0 ? __half2float(sc[rs + warp * 2]) : 0.0f;
        S.sc[warp][kv][lane][1] = rb >= 0 ? __half2float(sc[rs + warp * 2 + 1]) : 0.0f;
    };
    long long rb_nxt, rs_nxt;
    rows_of(cell_of(lane), rb_nxt, rs_nxt);   // chunk 0
    issue(rb_nxt, rs_nxt, 0);
    ai_commit();
    issue(rb_nxt, rs_nxt, 1);
    ai_commit();
    rows_of(cell_of(AI_CH + lane), rb_nxt, rs_nxt);   // chunk 1
    int cell_nn = cell_of(2 * AI_CH + lane);          // chunk 2
    __syncthreads();

    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    const int8_t* K = &S.kv[warp][0][0][0];
    const int8_t* V = &S.kv[warp][1][0][0];

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * AI_CH;
        const bool more = ci + 1 < n_chunks;
        ai_wait1();   // K(ci) is in (pending: V(ci))
        __syncwarp();
        // q.k over this warp's 64 dimensions: n-tile nt holds cells nt * 8 .. + 7 (column n = cell nt * 8 + n); the two k-steps are the two 32-value groups
#pragma unroll
        for (int nt = 0; nt < AI_CH / 8; ++nt) {
            const int cell = nt * 8 + gid;
            const int c = nt * 8 + 2 * tig;
            float sa[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0}, cm[4] = {0, 0, 0, 0};
                uint32_t b[2];
                b[0] = *reinterpret_cast<const uint32_t*>(K + ai_swz(cell, ks * 32 + tig * 4));
                b[1] = *reinterpret_cast<const uint32_t*>(K + ai_swz(cell, ks * 32 + 16 + tig * 4));
                ai_imma(ch, qa[0][ks], b);
                ai_imma(cl, qa[1][ks], b);
                ai_imma_us(cm, qa[2][ks], b);
                const float s0 = S.sc[warp][0][c][ks], s1 = S.sc[warp][0][c + 1][ks];
                sa[0] += ai_join3(ch[0], cl[0], cm[0]) * s0;
                sa[1] += ai_join3(ch[1], cl[1], cm[1]) * s1;
                sa[2] += ai_join3(ch[2], cl[2], cm[2]) * s0;
                sa[3] += ai_join3(ch[3], cl[3], cm[3]) * s1;
            }
            S.part[warp][gid][c] = sa[0] * qinv[0];
            S.part[warp][gid][c + 1] = sa[1] * qinv[0];
            if (gid + 8 < AI_G) {
                S.part[warp][gid + 8][c] = sa[2] * qinv[1];
                S.part[warp][gid + 8][c + 1] = sa[3] * qinv[1];
            }
        }
        __syncwarp();   // every lane has read K(ci) and its scales
        if (more) issue(rb_nxt, rs_nxt, 0);
        ai_commit();
        __syncthreads();
        // online softmax over the four warps' sums (fixed order): rows 0..11, 8 threads a row
        if (t < AI_G * 8) {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * scale_log2 : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        ai_wait1();   // V(ci) is in (pending: K(ci + 1))
        __syncwarp();
        // p.v over this warp's 64 dimensions: A = p * v_scale (rows gid / gid + 8, k = the chunk's 32 cells) in three parts scaled per row, once for each
        // 32-value group's V scale; B = V (k = cells, n-tile j column n = dimension n * 8 + j): columns 0..3 are the first group's dimensions, 4..7 the second's
        {
            uint32_t pa[2][3][4];
            float pinv[2][2];   // [group][row half]
#pragma unroll
            for (int gq = 0; gq < 2; ++gq) {
                float a[4][4], mx[2] = {0.0f, 0.0f};
#pragma unroll
                for (int r = 0; r < 4; ++r) {   // a0: row gid, cells tig * 4 ..; a1: row gid + 8; a2 / a3: cells 16 + ...
                    const int row = gid + (r & 1) * 8, cb = (r >> 1) * 16 + tig * 4;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        a[r][e] = row < AI_G ? S.p[row][cb + e] * S.sc[warp][1][cb + e][gq] : 0.0f;
                        mx[r & 1] = fmaxf(mx[r & 1], a[r][e]);
                    }
                }
                float up[2];
#pragma unroll
                for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
                    for (int o = 1; o < 4; o <<= 1) mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], o));
                    up[rh] = mx[rh] > 0.0f ? 127.0f / mx[rh] : 0.0f;
                    pinv[gq][rh] = mx[rh] * (1.0f / 127.0f);
                }
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int hh[4], ll[4], mm[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) ai_split3(a[r][e] * up[r & 1], hh[e], ll[e], mm[e]);
                    pa[gq][0][r] = ai_pack4(hh[0], hh[1], hh[2], hh[3]);
                    pa[gq][1][r] = ai_pack4(ll[0], ll[1], ll[2], ll[3]);
                    pa[gq][2][r] = ai_pack4(mm[0], mm[1], mm[2], mm[3]);
                }
            }
            // B: this thread's 8 dimensions (gid * 8 .. + 7) of cells tig * 4 .. + 3 (b0) and 16 + tig * 4 .. + 3 (b1)
            uint32_t bv[2][8];   // [k half][dimension j]
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                uint32_t lo[4], hi[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint2 w = *reinterpret_cast<const uint2*>(V + ai_swz(kh * 16 + tig * 4 + e, gid * 8));
                    lo[e] = w.x;
                    hi[e] = w.y;
                }
                ai_tr4x4(lo[0], lo[1], lo[2], lo[3], &bv[kh][0]);
                ai_tr4x4(hi[0], hi[1], hi[2], hi[3], &bv[kh][4]);
            }
            __syncwarp();   // every lane has read V(ci) and its scales
            if (more) issue(rb_nxt, rs_nxt, 1);
            ai_commit();
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
            const int gsel = tig >> 1;   // this lane's output columns 2 tig, 2 tig + 1 are dimensions of group tig / 2
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int ch[2][4], cl[2][4], cm[2][4];
                const uint32_t b[2] = {bv[0][j], bv[1][j]};
#pragma unroll
                for (int gq = 0; gq < 2; ++gq) {
                    ch[gq][0] = ch[gq][1] = ch[gq][2] = ch[gq][3] = 0;
                    cl[gq][0] = cl[gq][1] = cl[gq][2] = cl[gq][3] = 0;
                    cm[gq][0] = cm[gq][1] = cm[gq][2] = cm[gq][3] = 0;
                    ai_imma(ch[gq], pa[gq][0], b);
                    ai_imma(cl[gq], pa[gq][1], b);
                    ai_imma_us(cm[gq], pa[gq][2], b);
                }
                const float v0 = (gsel ? ai_join3(ch[1][0], cl[1][0], cm[1][0]) * pinv[1][0] : ai_join3(ch[0][0], cl[0][0], cm[0][0]) * pinv[0][0]);
                const float v1 = (gsel ? ai_join3(ch[1][1], cl[1][1], cm[1][1]) * pinv[1][0] : ai_join3(ch[0][1], cl[0][1], cm[0][1]) * pinv[0][0]);
                const float v2 = (gsel ? ai_join3(ch[1][2], cl[1][2], cm[1][2]) * pinv[1][1] : ai_join3(ch[0][2], cl[0][2], cm[0][2]) * pinv[0][1]);
                const float v3 = (gsel ? ai_join3(ch[1][3], cl[1][3], cm[1][3]) * pinv[1][1] : ai_join3(ch[0][3], cl[0][3], cm[0][3]) * pinv[0][1]);
                acc[j][0] = fmaf(acc[j][0], a0, v0);
                acc[j][1] = fmaf(acc[j][1], a0, v1);
                acc[j][2] = fmaf(acc[j][2], a1, v2);
                acc[j][3] = fmaf(acc[j][3], a1, v3);
            }
        }
        rows_of(cell_nn, rb_nxt, rs_nxt);
        cell_nn = cell_of((ci + 3) * AI_CH + lane);
    }
    __syncthreads();
    // this card's partial for the heads of the KV head: acc, then the running max (natural log) and sum
#pragma unroll
    for (int j = 0; j < 8; ++j) {   // column 2 * tig is dimension 16 * tig + j, column 2 * tig + 1 dimension 16 * tig + 8 + j
        const int d0 = dim0 + 16 * tig + j, d1 = d0 + 8;
        {
            const int h = kvh * AI_G + gid, k = h / ATT_OWN, hl = h % ATT_OWN;
            float* dst = part + (((size_t) k * C + tq) * ATT_OWN + hl) * AP_W;
            dst[d0] = acc[j][0];
            dst[d1] = acc[j][1];
        }
        if (gid + 8 < AI_G) {
            const int h = kvh * AI_G + gid + 8, k = h / ATT_OWN, hl = h % ATT_OWN;
            float* dst = part + (((size_t) k * C + tq) * ATT_OWN + hl) * AP_W;
            dst[d0] = acc[j][2];
            dst[d1] = acc[j][3];
        }
    }
    if (warp == 0 && tig == 0) {
        {
            const int h = kvh * AI_G + gid, k = h / ATT_OWN, hl = h % ATT_OWN;
            float* dst = part + (((size_t) k * C + tq) * ATT_OWN + hl) * AP_W;
            dst[HD] = S.mrow[gid] * 0.6931471805599453f;   // the running max is in log2 units; the merge takes natural logs
            dst[HD + 1] = S.lsum[gid];
        }
        if (gid + 8 < AI_G) {
            const int h = kvh * AI_G + gid + 8, k = h / ATT_OWN, hl = h % ATT_OWN;
            float* dst = part + (((size_t) k * C + tq) * ATT_OWN + hl) * AP_W;
            dst[HD] = S.mrow[gid + 8] * 0.6931471805599453f;
            dst[HD + 1] = S.lsum[gid + 8];
        }
    }
}

}  // namespace
#endif
