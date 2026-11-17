// metal/src/engine/msl/dflash2.metal - the small kernels of the DFlash2 drafter on Metal (the ones between the Q8_0 GEMVs, which are the
// engine's own gemv_8_* over the DF2P pack): the same math as src/kernels/cuda/dflash2.cu, see include/strata/kernels/dflash2.hpp.
//
//   df2_rms         y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w            rows of width n (w: the plain weight)
//   df2_conv        the grouped dynamic 2-tap convolution over the block's rows (+ residual)
//   df2_norm_rope   per (row, head): rope(rmsnorm(x) * gamma), written to slot(r) (the K/V ring or the block's own rows)
//   df2_copy_rows   the V half of the ring write
//   df2_swiglu      silu(gate) * up
//   df2_attn_part / df2_attn_merge   the 8 query rows over the ring + the block's keys, windowed, not causal inside the block
//   df2_topk_part / df2_topk_merge   the head's top-16 per row (value-descending, ties to the lower index)
//   df2_select      the candidate selector's left-to-right walk
//   df2_tap         the verify window's / prompt chunk's tap: the mean over the hc streams of a layer's completed output
//
// Compiled standalone (no quant.metal): the GEMVs and xprep come from the engine's main library.

#include <metal_stdlib>
using namespace metal;

constant int HD = 128;

struct RmsArgs { uint n; float eps; };

// grid (rows), 256 threads
kernel void df2_rms(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
                    constant RmsArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float* xr = x + tg.x * a.n;
    float ss = 0.0f;
    for (uint d = tid; d < a.n; d += 256) ss = fma(xr[d], xr[d], ss);
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = rsqrt(tot / float(a.n) + a.eps);
    device float* yr = y + tg.x * a.n;
    for (uint d = tid; d < a.n; d += 256) yr[d] = xr[d] * rs * w[d];
}

struct ConvArgs { uint rows; uint H; uint group; uint dyn_stride; uint has_resid; };

kernel void df2_conv(device const float* x [[buffer(0)]], device const float* dyn [[buffer(1)]], device const float* base [[buffer(2)]],
                     device const float* resid [[buffer(3)]], device float* out [[buffer(4)]], constant ConvArgs& a [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.H) return;
    const uint b = i / a.H, c = i % a.H;
    const uint groups = a.H / a.group;
    float acc = 0.0f;
    for (uint o = 0; o < 2; ++o) {
        if (b < o) continue;
        const float k = base[o * a.H + c] + dyn[b * a.dyn_stride + o * groups + c / a.group];
        acc = fma(k, x[(b - o) * a.H + c], acc);
    }
    out[i] = a.has_resid ? resid[i] + acc : acc;
}

struct NrArgs { uint in_stride; uint slot_mod; uint heads; float eps; uint has_rope; };

// grid (rows, heads), 128 threads
kernel void df2_norm_rope(device const float* in [[buffer(0)]], device const float* gamma [[buffer(1)]], device const int* pos [[buffer(2)]],
                          device const float* table [[buffer(3)]], device float* out [[buffer(4)]], constant NrArgs& a [[buffer(5)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    threadgroup float y[HD];
    const uint r = tg.x, h = tg.y;
    const float x = in[r * a.in_stride + h * HD + d];
    float ss = simd_sum(x * x);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = red[0] + red[1] + red[2] + red[3];
    const float v = x * rsqrt(tot / float(HD) + a.eps) * gamma[d];
    float res = v;
    if (a.has_rope) {
        y[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int i = d & 63;
        const float c = table[(r * 2 + 0) * 64 + i], s = table[(r * 2 + 1) * 64 + i];
        res = d < 64 ? v * c - y[d + 64] * s : v * c + y[d - 64] * s;
    }
    const uint slot = a.slot_mod > 0 ? uint(pos[r]) % a.slot_mod : r;
    out[(slot * a.heads + h) * HD + d] = res;
}

struct CopyArgs { uint slot_mod; uint rows; uint cols; };

kernel void df2_copy_rows(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], device const int* pos [[buffer(2)]],
                          constant CopyArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.cols) return;
    const uint r = i / a.cols, c = i % a.cols;
    const uint slot = a.slot_mod > 0 ? uint(pos[r]) % a.slot_mod : r;
    out[slot * a.cols + c] = in[i];
}

struct SwArgs { uint n; };

kernel void df2_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* y [[buffer(2)]],
                       constant SwArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float gv = g[i];
    y[i] = gv / (1.0f + exp(-gv)) * u[i];
}

// ---- attention.  Grid (8 K/V heads, chunks of 64 keys), 64 threads: thread t scores key t of the chunk against the 32 queries
// (4 heads x 8 rows) of its K/V head, the 32 threads below 32 then take each one query's softmax statistics over the chunk, and
// thread t sums V for the dims t and t + 64.  part[chunk][kvh][qi] = (m, l, O[128]).
struct AttnArgs { uint ring; uint window; };
constant int AK = 64;     // keys per chunk
constant int AROWS = 8;   // the block's rows (and block keys)

kernel void df2_attn_part(device const float* q [[buffer(0)]], device const float* kring [[buffer(1)]], device const float* vring [[buffer(2)]],
                          device const float* kblk [[buffer(3)]], device const float* vblk [[buffer(4)]],
                          device const int* anchor_pos [[buffer(5)]], device float* part [[buffer(6)]], constant AttnArgs& a [[buffer(7)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float Qs[32 * HD];
    threadgroup float Ss[32 * AK];
    threadgroup float Mx[32];
    threadgroup float Ls[32];
    const uint kvh = tg.x, chunk = tg.y;
    const int s = *anchor_pos;
    const int lo = max(max(0, s - (int(a.window) - 1)), anchor_pos[1]);   // anchor_pos[1]: first position whose ring slot belongs to this request's lineage
    const int nctx = s - lo;
    const int nkeys = nctx + AROWS;
    const int k_begin = int(chunk) * AK;
    device float* pbase = part + ((chunk * 8 + kvh) * 32) * 130;
    if (k_begin >= nkeys) {
        for (int qi = 0; qi < 32; ++qi) {
            if (tid == 0) { pbase[qi * 130] = -INFINITY; pbase[qi * 130 + 1] = 0.0f; }
            pbase[qi * 130 + 2 + tid] = 0.0f;
            pbase[qi * 130 + 2 + tid + 64] = 0.0f;
        }
        return;
    }
    const float scale = rsqrt(float(HD));
    for (int i = tid; i < 32 * HD; i += AK) {
        const int qa = i / HD, d = i % HD;
        const int row = qa >> 2, g = qa & 3;
        Qs[qa * HD + d] = q[(row * 32 + kvh * 4 + g) * HD + d] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // phase 1: scores
    const int k = k_begin + int(tid);
    const bool valid = k < nkeys;
    float sc[32];
    for (int qi = 0; qi < 32; ++qi) sc[qi] = 0.0f;
    if (valid) {
        device const float4* kp = (device const float4*)(k < nctx ? kring + ((uint(lo + k) % a.ring) * 8 + kvh) * HD
                                                                   : kblk + (uint(k - nctx) * 8 + kvh) * HD);
        for (int d4 = 0; d4 < HD / 4; ++d4) {
            const float4 kv = kp[d4];
            for (int qi = 0; qi < 32; ++qi) sc[qi] += dot(kv, *(threadgroup const float4*)(Qs + qi * HD + d4 * 4));
        }
    }
    for (int qi = 0; qi < 32; ++qi) {
        const int row = qi >> 2;
        bool ok = valid;
        if (ok && k < nctx) ok = (lo + k) > (s + row - int(a.window));   // s + row - j < window
        Ss[qi * AK + tid] = ok ? sc[qi] : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // phase 2: softmax statistics per query
    if (tid < 32) {
        float m = -INFINITY;
        for (int j = 0; j < AK; ++j) m = max(m, Ss[tid * AK + j]);
        float l = 0.0f;
        for (int j = 0; j < AK; ++j) {
            const float sv = Ss[tid * AK + j];
            const float p = sv == -INFINITY ? 0.0f : exp(sv - m);
            Ss[tid * AK + j] = p;
            l += p;
        }
        Mx[tid] = m;
        Ls[tid] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // phase 3: P V
    float acc0[32], acc1[32];
    for (int qi = 0; qi < 32; ++qi) { acc0[qi] = 0.0f; acc1[qi] = 0.0f; }
    const int kend = min(AK, nkeys - k_begin);
    for (int j = 0; j < kend; ++j) {
        const int kk = k_begin + j;
        device const float* vp = kk < nctx ? vring + ((uint(lo + kk) % a.ring) * 8 + kvh) * HD : vblk + (uint(kk - nctx) * 8 + kvh) * HD;
        const float v0 = vp[tid], v1 = vp[tid + 64];
        for (int qi = 0; qi < 32; ++qi) {
            const float p = Ss[qi * AK + j];
            acc0[qi] = fma(p, v0, acc0[qi]);
            acc1[qi] = fma(p, v1, acc1[qi]);
        }
    }
    for (int qi = 0; qi < 32; ++qi) {
        if (tid == 0) { pbase[qi * 130] = Mx[qi]; pbase[qi * 130 + 1] = Ls[qi]; }
        pbase[qi * 130 + 2 + tid] = acc0[qi];
        pbase[qi * 130 + 2 + tid + 64] = acc1[qi];
    }
}

// grid (8 * 32), 128 threads: out[row][32 heads][128]
kernel void df2_attn_merge(device const float* part [[buffer(0)]], device const int* anchor_pos [[buffer(1)]], device float* out [[buffer(2)]],
                           constant AttnArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]]) {
    const uint kvh = tg.x >> 5, qi = tg.x & 31;
    const int s = *anchor_pos;
    const int lo = max(max(0, s - (int(a.window) - 1)), anchor_pos[1]);   // anchor_pos[1]: first position whose ring slot belongs to this request's lineage
    const int nchunks = ((s - lo) + AROWS + AK - 1) / AK;
    float M = -INFINITY;
    for (int c = 0; c < nchunks; ++c) M = max(M, part[(((uint(c) * 8 + kvh) * 32) + qi) * 130]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < nchunks; ++c) {
        device const float* p = part + (((uint(c) * 8 + kvh) * 32) + qi) * 130;
        const float mc = p[0];
        const float w = mc == -INFINITY ? 0.0f : exp(mc - M);
        L += p[1] * w;
        o = fma(p[2 + d], w, o);
    }
    const uint row = qi >> 2, g = qi & 3;
    out[(row * 32 + kvh * 4 + g) * HD + d] = L > 0.0f ? o / L : 0.0f;
}

// ---- top-16
struct TkArgs { int n; };
struct TkMArgs { int use_map; };
constant int TOPB = 64;
constant int TOPK = 16;

static inline bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

// grid (rows, TOPB), 256 threads
kernel void df2_topk_part(device const float* logits [[buffer(0)]], device float* cval [[buffer(1)]], device int* cidx [[buffer(2)]],
                          constant TkArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float vals[4096];
    threadgroup float wv[8];
    threadgroup int wi[8];
    const int row = int(tg.x), blk = int(tg.y), n = a.n;
    const int cs = (n + TOPB - 1) / TOPB;
    const int c0 = blk * cs, c1 = min(c0 + cs, n);
    for (int i = tid; i < cs; i += 256) vals[i] = (c0 + i < c1) ? logits[row * n + c0 + i] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = tid; i < cs; i += 256)
            if (better(vals[i], i, bv, bi)) { bv = vals[i]; bi = i; }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { wv[sg] = bv; wi[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; }
            const int o = (row * TOPB + blk) * TOPK + k;
            cval[o] = v;
            cidx[o] = v == -INFINITY ? -1 : c0 + ix;
            if (v != -INFINITY) vals[ix] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// grid (rows), 256 threads
kernel void df2_topk_merge(device const float* cval [[buffer(0)]], device const int* cidx [[buffer(1)]], device const int* idx_map [[buffer(2)]],
                           device float* val [[buffer(3)]], device int* tok [[buffer(4)]], constant TkMArgs& a [[buffer(5)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NC = TOPB * TOPK;
    threadgroup float sv[NC];
    threadgroup int si[NC];
    threadgroup float wv[8];
    threadgroup int wi[8], wp[8];
    const int row = int(tg.x);
    for (int i = tid; i < NC; i += 256) { sv[i] = cval[row * NC + i]; si[i] = cidx[row * NC + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bp = -1;
        for (int i = tid; i < NC; i += 256)
            if (si[i] >= 0 && better(sv[i], si[i], bv, bi)) { bv = sv[i]; bi = si[i]; bp = i; }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            const int op = simd_shuffle_xor(bp, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bp = op; }
        }
        if (lane == 0) { wv[sg] = bv; wi[sg] = bi; wp[sg] = bp; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0], p = wp[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; p = wp[w]; }
            if (p >= 0) si[p] = -1;
            val[row * TOPK + k] = v;
            tok[row * TOPK + k] = p >= 0 ? (a.use_map != 0 ? idx_map[ix] : ix) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- the selector's walk: one threadgroup of 256.  Scores: unary[i][j] + dot(succ[tok[i][j]], pred[prev] * hp[i]) over rank 256.
struct SelArgs { uint rank; uint rows; };

static inline float bf16f(ushort h) { return as_type<float>(uint(h) << 16); }

kernel void df2_select(device const float* hp [[buffer(0)]], device const float* unary [[buffer(1)]], device const int* tok [[buffer(2)]],
                       device const ushort* pred_cb [[buffer(3)]], device const ushort* succ_cb [[buffer(4)]],
                       device const int* anchor_tok [[buffer(5)]], device int* out_tok [[buffer(6)]], device float* out_prob [[buffer(7)]],
                       constant SelArgs& a [[buffer(8)]], ushort tid [[thread_index_in_threadgroup]],
                       ushort w [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float vp[256];
    threadgroup float sc[TOPK];
    threadgroup int prev;
    if (tid == 0) prev = *anchor_tok;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < a.rows; ++i) {
        vp[tid] = bf16f(pred_cb[uint(prev) * a.rank + tid]) * hp[i * a.rank + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = w * 2; j < w * 2 + 2; ++j) {
            const int t = tok[i * TOPK + j];
            const uint4 raw = *(device const uint4*)(succ_cb + uint(t) * a.rank + lane * 8);
            const uint u[4] = {raw.x, raw.y, raw.z, raw.w};
            float acc = 0.0f;
            for (int e = 0; e < 4; ++e) {
                acc = fma(as_type<float>(u[e] << 16), vp[lane * 8 + e * 2], acc);
                acc = fma(as_type<float>(u[e] & 0xffff0000u), vp[lane * 8 + e * 2 + 1], acc);
            }
            acc = simd_sum(acc);
            if (lane == 0) sc[j] = unary[i * TOPK + j] + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            int bj = 0;
            float bv = sc[0];
            for (int j = 1; j < TOPK; ++j)
                if (sc[j] > bv) { bv = sc[j]; bj = j; }
            float z = 0.0f;
            for (int j = 0; j < TOPK; ++j) z += exp(sc[j] - bv);
            out_tok[i] = tok[i * TOPK + bj];
            out_prob[i] = 1.0f / z;
            prev = out_tok[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The same walk over Q4_0 selector tables (the community GGUF's): a row is 8 ggml Q4_0 blocks of 18 bytes (half d, 16 bytes: byte j = element j low nibble,
// element j + 16 high nibble); a value is (q - 8) * d, in fp32 as ggml dequantizes it.
static inline float q4_0_val(device const uchar* row, uint e) {
    device const uchar* b = row + (e >> 5) * 18;
    const float d = float(*(device const half*)b);
    const uint j = e & 31;
    const uint q = j < 16 ? (b[2 + j] & 15u) : (b[2 + j - 16] >> 4);
    return d * (float(q) - 8.0f);
}

kernel void df2_select_q4(device const float* hp [[buffer(0)]], device const float* unary [[buffer(1)]], device const int* tok [[buffer(2)]],
                          device const uchar* pred_cb [[buffer(3)]], device const uchar* succ_cb [[buffer(4)]],
                          device const int* anchor_tok [[buffer(5)]], device int* out_tok [[buffer(6)]], device float* out_prob [[buffer(7)]],
                          constant SelArgs& a [[buffer(8)]], ushort tid [[thread_index_in_threadgroup]],
                          ushort w [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float vp[256];
    threadgroup float sc[TOPK];
    threadgroup int prev;
    const uint rowb = (a.rank / 32) * 18;   // 144 bytes a row
    if (tid == 0) prev = *anchor_tok;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < a.rows; ++i) {
        vp[tid] = q4_0_val(pred_cb + uint(prev) * rowb, tid) * hp[i * a.rank + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = w * 2; j < w * 2 + 2; ++j) {
            const int t = tok[i * TOPK + j];
            device const uchar* row = succ_cb + uint(t) * rowb;
            float acc = 0.0f;
            for (int e = 0; e < 8; ++e) acc = fma(q4_0_val(row, lane * 8 + e), vp[lane * 8 + e], acc);
            acc = simd_sum(acc);
            if (lane == 0) sc[j] = unary[i * TOPK + j] + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            int bj = 0;
            float bv = sc[0];
            for (int j = 1; j < TOPK; ++j)
                if (sc[j] > bv) { bv = sc[j]; bj = j; }
            float z = 0.0f;
            for (int j = 0; j < TOPK; ++j) z += exp(sc[j] - bv);
            out_tok[i] = tok[i * TOPK + bj];
            out_prob[i] = 1.0f / z;
            prev = out_tok[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- the tap: out[r][out_off + i] = (sum_c R[r][c][i] + bo[r][i] * sum_c 2 sigmoid(inj[r][c] / hc)) / hc.  Grid (n / 256, rows), 256 threads.
struct TapArgs { uint n; uint hc; uint R_stride; uint bo_stride; uint inj_stride; uint out_stride; uint out_off; };

kernel void df2_tap(device const float* R [[buffer(0)]], device const float* bo [[buffer(1)]], device const float* inj [[buffer(2)]],
                    device float* out [[buffer(3)]], constant TapArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, r = gid.y;
    if (i >= a.n) return;
    float wsum = 0.0f, rs = 0.0f;
    for (uint c = 0; c < a.hc; ++c) {
        const float x = inj[r * a.inj_stride + c] / float(a.hc);
        wsum += 2.0f / (1.0f + exp(-x));
        rs += R[r * a.R_stride + c * a.n + i];
    }
    out[r * a.out_stride + a.out_off + i] = (rs + bo[r * a.bo_stride + i] * wsum) / float(a.hc);
}

// ---- Q8_0 matrix x up to 8 float columns, for the repacked pack (dflash2.inc): qs = int8 [rows][K], sc = half [rows][K / 32].
// A threadgroup = 4 simdgroups x R rows each.  The columns' K chunk of 512 values sits in threadgroup memory; lane l reads 16 int8 of
// every row of its simdgroup (one 16-byte load per row, a simdgroup covers 512 contiguous bytes of a row per step), converts them to
// float four at a time, scales them with the block's d and takes the dot products with the NCOL columns.  Every weight byte is read
// once; K must be a multiple of 512.
constant int NCOL [[function_constant(0)]];

struct MmArgs { uint K; uint rows; uint y_stride; uint x_stride; };

template <int R, int SG>
kernel void df2_mm(device const char* qs [[buffer(0)]], device const half* sc [[buffer(1)]], device const float* x [[buffer(2)]],
                   device float* y [[buffer(3)]], constant MmArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]], ushort sgi [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 xs[8 * 128];   // [col][s * 32 + lane]: float4 number lane * 4 + s of the chunk (a conflict-free read for the lanes)
    const uint row0 = (tg.x * SG + sgi) * R;
    const uint nb = a.K / 32;
    device const char* wr[R];
    device const half* sr[R];
    for (int r = 0; r < R; ++r) {
        const uint row = min(row0 + uint(r), a.rows - 1);
        wr[r] = qs + ulong(row) * a.K;
        sr[r] = sc + ulong(row) * nb;
    }
    float acc[R][8];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < 8; ++c) acc[r][c] = 0.0f;
    const uint kk = uint(lane) * 16;
    uint4 raw[R];
    float d[R];
    for (int r = 0; r < R; ++r) {
        raw[r] = *(device const uint4*)(wr[r] + kk);
        d[r] = float(sr[r][kk / 32]);
    }
    for (uint k0 = 0; k0 < a.K; k0 += 512) {
        // the next chunk's weights are in flight while this one is staged and used
        uint4 rawn[R];
        float dn[R];
        const bool more = k0 + 512 < a.K;
        if (more)
            for (int r = 0; r < R; ++r) {
                rawn[r] = *(device const uint4*)(wr[r] + k0 + 512 + kk);
                dn[r] = float(sr[r][(k0 + 512 + kk) / 32]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < uint(NCOL) * 128; i += SG * 32) {
            const uint c = i / 128, j = i % 128;
            xs[c * 128 + (j & 3) * 32 + (j >> 2)] = ((device const float4*)(x + ulong(c) * a.x_stride + k0))[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int sub = 0; sub < 4; ++sub) {
            float4 xv[8];
            for (int c = 0; c < NCOL; ++c) xv[c] = xs[c * 128 + sub * 32 + lane];
            for (int r = 0; r < R; ++r) {
                const uint word = sub == 0 ? raw[r].x : sub == 1 ? raw[r].y : sub == 2 ? raw[r].z : raw[r].w;
                const float4 w = float4(as_type<char4>(word)) * d[r];
                for (int c = 0; c < NCOL; ++c) acc[r][c] += dot(w, xv[c]);
            }
        }
        if (more)
            for (int r = 0; r < R; ++r) { raw[r] = rawn[r]; d[r] = dn[r]; }
    }
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < NCOL; ++c) {
            const float v = simd_sum(acc[r][c]);
            if (lane == 0 && row0 + uint(r) < a.rows) y[ulong(c) * a.y_stride + row0 + r] = v;
        }
}

#define DF2MM(R, S) template [[host_name("df2_mm_" #R "_" #S)]] kernel decltype(df2_mm<R, S>) df2_mm<R, S>;
DF2MM(1, 4) DF2MM(1, 8) DF2MM(1, 16) DF2MM(2, 4) DF2MM(2, 8) DF2MM(2, 16) DF2MM(4, 4) DF2MM(4, 8) DF2MM(4, 16) DF2MM(8, 4) DF2MM(8, 8)

// ---- Q4_0 matrix x up to 8 float columns, for the pack repacked at load (DF2-METAL micro-bench): qs = uchar [rows][K / 32][16] (the 16 bytes of a
// Q4_0 block, byte j = element j in the low nibble, element j + 16 in the high one), sc = half [rows][K / 32].  Same shape as df2_mm: a
// threadgroup = SG simdgroups x R rows each; the columns' K chunk of 512 values sits in threadgroup memory; lane l reads 8 bytes (half a block:
// block l / 2 of the chunk, bytes 8 * (l & 1) ..) of every row of its simdgroup, so the 32 lanes cover 16 blocks = 512 values per step.
// K must be a multiple of 512.  The block's scale multiplies the dot product of its nibbles (minus 8) with the columns.
struct Mm4Args { uint K; uint rows; uint y_stride; uint x_stride; };

template <int R, int SG>
kernel void df2_mm4(device const uchar* qs [[buffer(0)]], device const half* sc [[buffer(1)]], device const float* x [[buffer(2)]],
                    device float* y [[buffer(3)]], constant Mm4Args& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sgi [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 xs[8 * 128];   // [col][128 float4 of the chunk]
    const uint row0 = (tg.x * SG + sgi) * R;
    const uint nb = a.K / 32;
    const uint blk = uint(lane) >> 1, hh = uint(lane) & 1;
    device const uchar* wr[R];
    device const half* sr[R];
    for (int r = 0; r < R; ++r) {
        const uint row = min(row0 + uint(r), a.rows - 1);
        wr[r] = qs + (ulong(row) * nb + blk) * 16 + hh * 8;
        sr[r] = sc + ulong(row) * nb + blk;
    }
    float acc[R][8];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < 8; ++c) acc[r][c] = 0.0f;
    uint2 raw[R];
    float d[R];
    for (int r = 0; r < R; ++r) { raw[r] = *(device const uint2*)(wr[r]); d[r] = float(sr[r][0]); }
    for (uint k0 = 0; k0 < a.K; k0 += 512) {
        uint2 rawn[R];
        float dn[R];
        const bool more = k0 + 512 < a.K;
        if (more)
            for (int r = 0; r < R; ++r) { rawn[r] = *(device const uint2*)(wr[r] + (k0 + 512) / 2); dn[r] = float(sr[r][(k0 + 512) / 32]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < uint(NCOL) * 128; i += SG * 32) {
            const uint c = i / 128, j = i % 128;
            xs[c * 128 + j] = ((device const float4*)(x + ulong(c) * a.x_stride + k0))[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // this lane's 16 elements: low nibbles = chunk positions blk * 32 + hh * 8 + 0..7, high nibbles = +16
        float4 xl0[8], xl1[8], xh0[8], xh1[8];
        const uint pl = (blk * 32 + hh * 8) / 4;
        for (int c = 0; c < NCOL; ++c) {
            xl0[c] = xs[c * 128 + pl]; xl1[c] = xs[c * 128 + pl + 1];
            xh0[c] = xs[c * 128 + pl + 4]; xh1[c] = xs[c * 128 + pl + 5];
        }
        for (int r = 0; r < R; ++r) {
            const uint w0 = raw[r].x, w1 = raw[r].y;
            const float4 lo0 = float4(int4((w0 >> 0) & 15, (w0 >> 8) & 15, (w0 >> 16) & 15, (w0 >> 24) & 15)) - 8.0f;
            const float4 lo1 = float4(int4((w1 >> 0) & 15, (w1 >> 8) & 15, (w1 >> 16) & 15, (w1 >> 24) & 15)) - 8.0f;
            const float4 hi0 = float4(int4((w0 >> 4) & 15, (w0 >> 12) & 15, (w0 >> 20) & 15, (w0 >> 28) & 15)) - 8.0f;
            const float4 hi1 = float4(int4((w1 >> 4) & 15, (w1 >> 12) & 15, (w1 >> 20) & 15, (w1 >> 28) & 15)) - 8.0f;
            for (int c = 0; c < NCOL; ++c)
                acc[r][c] = fma(d[r], dot(lo0, xl0[c]) + dot(lo1, xl1[c]) + dot(hi0, xh0[c]) + dot(hi1, xh1[c]), acc[r][c]);
        }
        if (more)
            for (int r = 0; r < R; ++r) { raw[r] = rawn[r]; d[r] = dn[r]; }
    }
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < NCOL; ++c) {
            const float v = simd_sum(acc[r][c]);
            if (lane == 0 && row0 + uint(r) < a.rows) y[ulong(c) * a.y_stride + row0 + r] = v;
        }
}

#define DF2MM4(R, S) template [[host_name("df2_mm4_" #R "_" #S)]] kernel decltype(df2_mm4<R, S>) df2_mm4<R, S>;
DF2MM4(1, 4) DF2MM4(1, 8) DF2MM4(2, 4) DF2MM4(2, 8) DF2MM4(2, 16) DF2MM4(4, 4) DF2MM4(4, 8) DF2MM4(4, 16) DF2MM4(8, 4) DF2MM4(8, 8)

// ---- Q4_0 matrix x 8 float columns on the simdgroup matrix units (df2_mma4).  The pack is repacked at load into FRAGMENT ORDER: for a tile of 8 rows
// and one Q4_0 block (32 values) the 32 lanes read 32 consecutive uint32; lane l holds the eight 4-bit codes of the weights W[row fm(l)][8 s + fn(l) + e]
// for s = 0..3 (the four k steps of the block) and e = 0..1 (the two columns of its fragment), code number 2 s + e at bits 4 (2 s + e).  fm / fn are the
// fragment coordinates of simdgroup_matrix 8x8 (row fm, columns fn and fn + 1 of lane l).  The scales: half [row tile][block][8 rows].
// y[t][row] = sum over blocks of d(row, block) * sum_k (code - 8) * x[t][k], the integer part on the matrix units in half operands with float
// accumulation (codes - 8 are exact in half; the activations are rounded to half).
struct MmaQArgs { uint K; uint rows; uint y_stride; uint x_stride; };

template <int RT, int SG>
kernel void df2_mma4(device const uint* qw [[buffer(0)]], device const half* sc [[buffer(1)]], device const half* xh [[buffer(2)]],
                     device float* y [[buffer(3)]], constant MmaQArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                     ushort sgi [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint nb = a.K / 32;
    const uint nrt_total = (a.rows + 7) / 8;
    const uint rt0 = (tg.x * SG + sgi) * RT;
    if (rt0 >= nrt_total) return;
    const uint qid = uint(lane) >> 2;
    const uint fm = (qid & 4) + ((uint(lane) >> 1) & 3);
    const uint fn = (qid & 2) * 2 + (uint(lane) & 1) * 2;
    float acc0[RT], acc1[RT];
    for (int r = 0; r < RT; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
    device const uint* wp[RT];
    device const half* sp[RT];
    for (int r = 0; r < RT; ++r) {
        const uint rt = min(rt0 + uint(r), nrt_total - 1);
        wp[r] = qw + (ulong(rt) * nb) * 32 + lane;
        sp[r] = sc + (ulong(rt) * nb) * 8 + fm;
    }
    uint wcur[RT]; half dcur[RT];
    for (int r = 0; r < RT; ++r) { wcur[r] = wp[r][0]; dcur[r] = sp[r][0]; }
    for (uint b = 0; b < nb; ++b) {
        uint wn[RT]; half dn[RT];
        const bool more = b + 1 < nb;
        if (more)
            for (int r = 0; r < RT; ++r) { wn[r] = wp[r][(b + 1) * 32]; dn[r] = sp[r][(b + 1) * 8]; }
        simdgroup_half8x8 B0, B1, B2, B3;
        simdgroup_load(B0, xh + b * 32 + 0, a.x_stride, ulong2(0, 0), true);
        simdgroup_load(B1, xh + b * 32 + 8, a.x_stride, ulong2(0, 0), true);
        simdgroup_load(B2, xh + b * 32 + 16, a.x_stride, ulong2(0, 0), true);
        simdgroup_load(B3, xh + b * 32 + 24, a.x_stride, ulong2(0, 0), true);
        for (int r = 0; r < RT; ++r) {
            const uint w = wcur[r];
            simdgroup_half8x8 A0, A1, A2, A3;
            A0.thread_elements()[0] = half(int((w >> 0) & 15) - 8);  A0.thread_elements()[1] = half(int((w >> 4) & 15) - 8);
            A1.thread_elements()[0] = half(int((w >> 8) & 15) - 8);  A1.thread_elements()[1] = half(int((w >> 12) & 15) - 8);
            A2.thread_elements()[0] = half(int((w >> 16) & 15) - 8); A2.thread_elements()[1] = half(int((w >> 20) & 15) - 8);
            A3.thread_elements()[0] = half(int((w >> 24) & 15) - 8); A3.thread_elements()[1] = half(int((w >> 28) & 15) - 8);
            simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_multiply_accumulate(C, A0, B0, C);
            simdgroup_multiply_accumulate(C, A1, B1, C);
            simdgroup_multiply_accumulate(C, A2, B2, C);
            simdgroup_multiply_accumulate(C, A3, B3, C);
            const float d = float(dcur[r]);
            acc0[r] = fma(d, C.thread_elements()[0], acc0[r]);
            acc1[r] = fma(d, C.thread_elements()[1], acc1[r]);
        }
        if (more)
            for (int r = 0; r < RT; ++r) { wcur[r] = wn[r]; dcur[r] = dn[r]; }
    }
    for (int r = 0; r < RT; ++r) {
        const uint row = (rt0 + uint(r)) * 8 + fm;
        if (rt0 + uint(r) < nrt_total && row < a.rows) {
            y[ulong(fn) * a.y_stride + row] = acc0[r];
            y[ulong(fn + 1) * a.y_stride + row] = acc1[r];
        }
    }
}

#define DF2MMA4(R, S) template [[host_name("df2_mma4_" #R "_" #S)]] kernel decltype(df2_mma4<R, S>) df2_mma4<R, S>;
DF2MMA4(1, 2) DF2MMA4(1, 4) DF2MMA4(1, 8) DF2MMA4(2, 2) DF2MMA4(2, 4) DF2MMA4(4, 2) DF2MMA4(4, 4)

kernel void df2_f16(device const float* x [[buffer(0)]], device half* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = half(x[i]);
}

// split-K variant of df2_mma4: the SG simdgroups of a threadgroup work on the SAME RT row tiles, each on its own range of blocks (the weights stream in
// parallel from SG places, so a small matrix still fills the GPU), then add their partial sums through threadgroup memory.  Block loop unrolled by 2.
template <int RT, int SG>
kernel void df2_mma4s(device const uint* qw [[buffer(0)]], device const half* sc [[buffer(1)]], device const half* xh [[buffer(2)]],
                      device float* y [[buffer(3)]], constant MmaQArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                      ushort sgi [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[SG][RT][2][32];
    const uint nb = a.K / 32;
    const uint nrt_total = (a.rows + 7) / 8;
    const uint rt0 = tg.x * RT;
    const uint qid = uint(lane) >> 2;
    const uint fm = (qid & 4) + ((uint(lane) >> 1) & 3);
    const uint fn = (qid & 2) * 2 + (uint(lane) & 1) * 2;
    const uint b0 = nb * uint(sgi) / SG, b1 = nb * (uint(sgi) + 1) / SG;
    float acc0[RT], acc1[RT];
    for (int r = 0; r < RT; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
    device const uint* wp[RT];
    device const half* sp[RT];
    for (int r = 0; r < RT; ++r) {
        const uint rt = min(rt0 + uint(r), nrt_total - 1);
        wp[r] = qw + (ulong(rt) * nb) * 32 + lane;
        sp[r] = sc + (ulong(rt) * nb) * 8 + fm;
    }
    for (uint b = b0; b < b1; b += 2) {
        const bool two = b + 1 < b1;
        uint w0[RT], w1[RT]; half d0[RT], d1[RT];
        for (int r = 0; r < RT; ++r) {
            w0[r] = wp[r][b * 32]; d0[r] = sp[r][b * 8];
            w1[r] = two ? wp[r][(b + 1) * 32] : 0u; d1[r] = two ? sp[r][(b + 1) * 8] : half(0);
        }
        for (int h = 0; h < (two ? 2 : 1); ++h) {
            const uint bb = b + uint(h);
            simdgroup_half8x8 B0, B1, B2, B3;
            simdgroup_load(B0, xh + bb * 32 + 0, a.x_stride, ulong2(0, 0), true);
            simdgroup_load(B1, xh + bb * 32 + 8, a.x_stride, ulong2(0, 0), true);
            simdgroup_load(B2, xh + bb * 32 + 16, a.x_stride, ulong2(0, 0), true);
            simdgroup_load(B3, xh + bb * 32 + 24, a.x_stride, ulong2(0, 0), true);
            for (int r = 0; r < RT; ++r) {
                const uint w = h == 0 ? w0[r] : w1[r];
                simdgroup_half8x8 A0, A1, A2, A3;
                A0.thread_elements()[0] = half(int((w >> 0) & 15) - 8);  A0.thread_elements()[1] = half(int((w >> 4) & 15) - 8);
                A1.thread_elements()[0] = half(int((w >> 8) & 15) - 8);  A1.thread_elements()[1] = half(int((w >> 12) & 15) - 8);
                A2.thread_elements()[0] = half(int((w >> 16) & 15) - 8); A2.thread_elements()[1] = half(int((w >> 20) & 15) - 8);
                A3.thread_elements()[0] = half(int((w >> 24) & 15) - 8); A3.thread_elements()[1] = half(int((w >> 28) & 15) - 8);
                simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_multiply_accumulate(C, A0, B0, C);
                simdgroup_multiply_accumulate(C, A1, B1, C);
                simdgroup_multiply_accumulate(C, A2, B2, C);
                simdgroup_multiply_accumulate(C, A3, B3, C);
                const float d = float(h == 0 ? d0[r] : d1[r]);
                acc0[r] = fma(d, C.thread_elements()[0], acc0[r]);
                acc1[r] = fma(d, C.thread_elements()[1], acc1[r]);
            }
        }
    }
    for (int r = 0; r < RT; ++r) { part[sgi][r][0][lane] = acc0[r]; part[sgi][r][1][lane] = acc1[r]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0) {
        for (int r = 0; r < RT; ++r) {
            float t0 = 0.0f, t1 = 0.0f;
            for (int g = 0; g < SG; ++g) { t0 += part[g][r][0][lane]; t1 += part[g][r][1][lane]; }
            const uint row = (rt0 + uint(r)) * 8 + fm;
            if (rt0 + uint(r) < nrt_total && row < a.rows) {
                y[ulong(fn) * a.y_stride + row] = t0;
                y[ulong(fn + 1) * a.y_stride + row] = t1;
            }
        }
    }
}
#define DF2MMA4S(R, S) template [[host_name("df2_mma4s_" #R "_" #S)]] kernel decltype(df2_mma4s<R, S>) df2_mma4s<R, S>;
DF2MMA4S(1, 2) DF2MMA4S(1, 4) DF2MMA4S(1, 8) DF2MMA4S(2, 4) DF2MMA4S(2, 8)

// ---- Q8_0 rows on the matrix units for the draft head: the same fragment order, eight int8 per lane per block as two uint32 (word 0: k steps 0 and 1, word 1: 2 and 3).
template <int RT, int SG>
kernel void df2_mma8(device const uint2* qw [[buffer(0)]], device const half* sc [[buffer(1)]], device const half* xh [[buffer(2)]],
                     device float* y [[buffer(3)]], constant MmaQArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                     ushort sgi [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint nb = a.K / 32;
    const uint nrt_total = (a.rows + 7) / 8;
    const uint rt0 = (tg.x * SG + sgi) * RT;
    if (rt0 >= nrt_total) return;
    const uint qid = uint(lane) >> 2;
    const uint fm = (qid & 4) + ((uint(lane) >> 1) & 3);
    const uint fn = (qid & 2) * 2 + (uint(lane) & 1) * 2;
    float acc0[RT], acc1[RT];
    for (int r = 0; r < RT; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
    device const uint2* wp[RT];
    device const half* sp[RT];
    for (int r = 0; r < RT; ++r) {
        const uint rt = min(rt0 + uint(r), nrt_total - 1);
        wp[r] = qw + (ulong(rt) * nb) * 32 + lane;
        sp[r] = sc + (ulong(rt) * nb) * 8 + fm;
    }
    uint2 wcur[RT]; half dcur[RT];
    for (int r = 0; r < RT; ++r) { wcur[r] = wp[r][0]; dcur[r] = sp[r][0]; }
    for (uint b = 0; b < nb; ++b) {
        uint2 wn[RT]; half dn[RT];
        const bool more = b + 1 < nb;
        if (more)
            for (int r = 0; r < RT; ++r) { wn[r] = wp[r][(b + 1) * 32]; dn[r] = sp[r][(b + 1) * 8]; }
        simdgroup_half8x8 B0, B1, B2, B3;
        simdgroup_load(B0, xh + b * 32 + 0, a.x_stride, ulong2(0, 0), true);
        simdgroup_load(B1, xh + b * 32 + 8, a.x_stride, ulong2(0, 0), true);
        simdgroup_load(B2, xh + b * 32 + 16, a.x_stride, ulong2(0, 0), true);
        simdgroup_load(B3, xh + b * 32 + 24, a.x_stride, ulong2(0, 0), true);
        for (int r = 0; r < RT; ++r) {
            const char4 c0 = as_type<char4>(wcur[r].x), c1 = as_type<char4>(wcur[r].y);
            simdgroup_half8x8 A0, A1, A2, A3;
            A0.thread_elements()[0] = half(c0.x); A0.thread_elements()[1] = half(c0.y);
            A1.thread_elements()[0] = half(c0.z); A1.thread_elements()[1] = half(c0.w);
            A2.thread_elements()[0] = half(c1.x); A2.thread_elements()[1] = half(c1.y);
            A3.thread_elements()[0] = half(c1.z); A3.thread_elements()[1] = half(c1.w);
            simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_multiply_accumulate(C, A0, B0, C);
            simdgroup_multiply_accumulate(C, A1, B1, C);
            simdgroup_multiply_accumulate(C, A2, B2, C);
            simdgroup_multiply_accumulate(C, A3, B3, C);
            const float d = float(dcur[r]);
            acc0[r] = fma(d, C.thread_elements()[0], acc0[r]);
            acc1[r] = fma(d, C.thread_elements()[1], acc1[r]);
        }
        if (more)
            for (int r = 0; r < RT; ++r) { wcur[r] = wn[r]; dcur[r] = dn[r]; }
    }
    for (int r = 0; r < RT; ++r) {
        const uint row = (rt0 + uint(r)) * 8 + fm;
        if (rt0 + uint(r) < nrt_total && row < a.rows) {
            y[ulong(fn) * a.y_stride + row] = acc0[r];
            y[ulong(fn + 1) * a.y_stride + row] = acc1[r];
        }
    }
}
#define DF2MMA8(R, S) template [[host_name("df2_mma8_" #R "_" #S)]] kernel decltype(df2_mma8<R, S>) df2_mma8<R, S>;
DF2MMA8(1, 4) DF2MMA8(1, 8) DF2MMA8(2, 4) DF2MMA8(2, 8) DF2MMA8(4, 4)

// ---- the same kernels with a second output: the result also rounded to half, the matrix-unit GEMMs' activation operand (replaces the df2_f16 dispatch
// before each GEMM; the half value is exactly what df2_f16 made from the stored float)
kernel void df2_rms_h(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
                      constant RmsArgs& a [[buffer(3)]], device half* yh [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float* xr = x + tg.x * a.n;
    float ss = 0.0f;
    for (uint d = tid; d < a.n; d += 256) ss = fma(xr[d], xr[d], ss);
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = rsqrt(tot / float(a.n) + a.eps);
    device float* yr = y + tg.x * a.n;
    device half* yhr = yh + tg.x * a.n;
    for (uint d = tid; d < a.n; d += 256) { const float v = xr[d] * rs * w[d]; yr[d] = v; yhr[d] = half(v); }
}

kernel void df2_conv_h(device const float* x [[buffer(0)]], device const float* dyn [[buffer(1)]], device const float* base [[buffer(2)]],
                       device const float* resid [[buffer(3)]], device float* out [[buffer(4)]], constant ConvArgs& a [[buffer(5)]],
                       device half* outh [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.H) return;
    const uint b = i / a.H, c = i % a.H;
    const uint groups = a.H / a.group;
    float acc = 0.0f;
    for (uint o = 0; o < 2; ++o) {
        if (b < o) continue;
        const float k = base[o * a.H + c] + dyn[b * a.dyn_stride + o * groups + c / a.group];
        acc = fma(k, x[(b - o) * a.H + c], acc);
    }
    const float v = a.has_resid ? resid[i] + acc : acc;
    out[i] = v;
    outh[i] = half(v);
}

kernel void df2_swiglu_h(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* y [[buffer(2)]],
                         constant SwArgs& a [[buffer(3)]], device half* yh [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float gv = g[i];
    const float v = gv / (1.0f + exp(-gv)) * u[i];
    y[i] = v;
    yh[i] = half(v);
}

kernel void df2_attn_merge_h(device const float* part [[buffer(0)]], device const int* anchor_pos [[buffer(1)]], device float* out [[buffer(2)]],
                             constant AttnArgs& a [[buffer(3)]], device half* outh [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                             ushort d [[thread_index_in_threadgroup]]) {
    const uint kvh = tg.x >> 5, qi = tg.x & 31;
    const int s = *anchor_pos;
    const int lo = max(max(0, s - (int(a.window) - 1)), anchor_pos[1]);
    const int nchunks = ((s - lo) + AROWS + AK - 1) / AK;
    float M = -INFINITY;
    for (int c = 0; c < nchunks; ++c) M = max(M, part[(((uint(c) * 8 + kvh) * 32) + qi) * 130]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < nchunks; ++c) {
        device const float* p = part + (((uint(c) * 8 + kvh) * 32) + qi) * 130;
        const float mc = p[0];
        const float w = mc == -INFINITY ? 0.0f : exp(mc - M);
        L += p[1] * w;
        o = fma(p[2 + d], w, o);
    }
    const uint row = qi >> 2, g = qi & 3;
    const float v = L > 0.0f ? o / L : 0.0f;
    out[(row * 32 + kvh * 4 + g) * HD + d] = v;
    outh[(row * 32 + kvh * 4 + g) * HD + d] = half(v);
}

// ---- the row log (lineage rebuild): the half-rounded hidden_norm rows of the context update, kept per position, are the exact operand of every layer's
// k/v projection; a restored conversation's ring is rebuilt from them (same kernels, same bytes).
kernel void df2_log_rows(device const half* in [[buffer(0)]], device half* log [[buffer(1)]], device const int* pos [[buffer(2)]],
                         constant CopyArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.cols) return;
    const uint r = i / a.cols, c = i % a.cols;
    log[(uint(pos[r]) % a.slot_mod) * a.cols + c] = in[i];
}

kernel void df2_gather_rows(device const half* log [[buffer(0)]], device half* out [[buffer(1)]], device const int* pos [[buffer(2)]],
                            constant CopyArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.cols) return;
    const uint r = i / a.cols, c = i % a.cols;
    out[i] = log[(uint(pos[r]) % a.slot_mod) * a.cols + c];
}
