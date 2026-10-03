// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/s_gemv.cpp - the Xe port of Strata's src/kernels/cuda/s_gemv.cu, s2_gemv_quads.cu and s2_gemv_fast.cu:
// the S-family GEMV (see the header).  Work-group shapes, per-thread quads and summation orders are the CUDA
// kernels'; a warp is a sub-group of 32, and lane 0 of its xor butterfly adds in the order of CUDA's shuffle-down
// tree.
//
// The codebook is staged in local memory once per work-group, as in CUDA.  The warp-per-row kernel returned before
// that barrier on the CUDA side; here every work-item reaches the barrier first, since a SYCL barrier that some
// work-items of the group never reach is undefined.
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"

#include <array>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;

constexpr signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                    1,    13,   25,  38,  53,  69,  89,  113};

constexpr int Q8K_BLOCK_BYTES = 292;
constexpr int Q8K_BLOCK_ELEMS = 256;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_0_BLOCK_ELEMS = 32;
constexpr int QK_S2 = 64;
constexpr int MAX_SHARED_HALVES = 4096;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}
void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

[[noreturn]] void fail(const std::string& msg) { throw core::DeviceError(msg); }

inline float decode_tbl(int code, int bias, int codebook, const signed char* tbl) {
    if (codebook == (int) Codebook::Iq4Nl) return (float) tbl[code & 0x0F];
    return (float) (code + bias);       // the bias is applied to the CODE, in the integer domain
}

inline float q8k_at(const uint8_t* x, long long i) {
    const uint8_t* blk = x + (i / Q8K_BLOCK_ELEMS) * Q8K_BLOCK_BYTES;
    const float d = *(const float*) blk;
    const int8_t q = ((const int8_t*) (blk + 4))[i % Q8K_BLOCK_ELEMS];
    return d * (float) q;
}

inline float q8_0_at(const uint8_t* x, long long i) {
    const uint8_t* blk = x + (i / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    const float d = f32_from_f16(*(const uint16_t*) blk);
    const int8_t q = ((const int8_t*) (blk + 2))[i % Q8_0_BLOCK_ELEMS];
    return d * (float) q;
}

template <typename Item>
inline void stage_codebook(const Item& it, signed char* tbl, int tid) {
    if (tid < 16) tbl[tid] = kIq4Nl[tid];
    sycl::group_barrier(it.get_group());
}

int shift_of(int group_elems) {
    int s = 0;
    while ((1 << s) < group_elems) ++s;
    return s;
}

// One work-item per output row.  X_Q8K selects the activation: fp16 (false) or block_q8_K (true).
template <int CODE_BITS, bool X_Q8K>
sycl::event launch_naive(sycl::queue& q, const void* xv, const uint8_t* codes, const float* scales,
                         const float* offset, float* y, long long n_in, long long n_out, int bias, int codebook,
                         int group_elems, int has_offset) {
    constexpr int THREADS = 128;
    const size_t groups = (size_t) ((n_out + THREADS - 1) / THREADS);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<signed char, 1> tbl(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            signed char* t = &tbl[0];
            stage_codebook(it, t, tid);
            const long long o = (long long) it.get_global_id(0);
            if (o >= n_out) return;
            constexpr int PER_BYTE = 8 / CODE_BITS;
            const long long n_groups = n_in / group_elems;
            const uint8_t* c = codes + o * (n_in / PER_BYTE);
            const float* s = scales + o * n_groups;
            const float* off = has_offset ? offset + o * n_groups : nullptr;
            float acc = 0.0f;
            for (long long g = 0; g < n_groups; ++g) {
                const float d = s[g];
                const float b = off ? off[g] : 0.0f;
                const long long base = g * (long long) group_elems;
                for (int j = 0; j < group_elems; ++j) {
                    const long long i = base + j;
                    const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
                    // the offset belongs to the WEIGHT and is applied before the activation multiply
                    const float w = decode_tbl(code, bias, codebook, t) * d + b;
                    if constexpr (X_Q8K) acc += w * q8k_at((const uint8_t*) xv, i);
                    else acc += w * f32_from_f16(((const uint16_t*) xv)[i]);
                }
            }
            y[o] = acc;
        });
    });
}

// One work-group of threads_per_row work-items per output row, four consecutive elements per work-item, reduced
// by a fixed tree over local memory.
template <int CODE_BITS>
sycl::event launch_split(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                         const float* offset, float* y, long long n_in, long long n_out, int bias, int codebook,
                         int group_elems, int group_shift, int has_offset, int threads_per_row) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t) threads_per_row), h);
        sycl::local_accessor<signed char, 1> tbl(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * threads_per_row, threads_per_row),
                       [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            signed char* t = &tbl[0];
            stage_codebook(it, t, tid);
            const long long o = (long long) it.get_group(0);
            constexpr int PER_BYTE = 8 / CODE_BITS;
            const long long n_groups = n_in / group_elems;
            const uint8_t* c = codes + o * (n_in / PER_BYTE);
            const float* s = scales + o * n_groups;
            const float* off = has_offset ? offset + o * n_groups : nullptr;
            constexpr int QE = 4;
            constexpr int QB = QE * CODE_BITS / 8;             // 1 for S2, 2 for S4, 4 for S8
            constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            long long i = (long long) tid * QE;
            for (; i + QE <= n_in; i += (long long) threads_per_row * QE) {
                const long long g = i >> group_shift;
                const float d = s[g];
                const float b = off ? off[g] : 0.0f;
                const uint8_t* cp = c + i / PER_BYTE;
                unsigned v;
                if constexpr (QB == 1) v = cp[0];
                else if constexpr (QB == 2) v = *(const uint16_t*) cp;
                else v = *(const uint32_t*) cp;
                const float f0 = f32_from_f16(x[i]), f1 = f32_from_f16(x[i + 1]);
                const float f2 = f32_from_f16(x[i + 2]), f3 = f32_from_f16(x[i + 3]);
                acc0 += (decode_tbl((int) (v & MASK), bias, codebook, t) * d + b) * f0;
                acc1 += (decode_tbl((int) ((v >> CODE_BITS) & MASK), bias, codebook, t) * d + b) * f1;
                acc2 += (decode_tbl((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook, t) * d + b) * f2;
                acc3 += (decode_tbl((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook, t) * d + b) * f3;
            }
            // the last, partial quad - at most one per work-item
            for (; i < n_in; i += (long long) threads_per_row * QE) {
                for (int k = 0; k < QE && i + k < n_in; ++k) {
                    const long long e = i + k;
                    const long long g = e >> group_shift;
                    const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
                    acc0 += (decode_tbl(code, bias, codebook, t) * s[g] + (off ? off[g] : 0.0f)) * f32_from_f16(x[e]);
                }
            }
            partial[tid] = (acc0 + acc1) + (acc2 + acc3);
            sycl::group_barrier(it.get_group());
            for (int step = threads_per_row / 2; step > 0; step >>= 1) {
                if (tid < step) partial[tid] += partial[tid + step];
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) y[o] = partial[0];
        });
    });
}

// One sub-group per output row over a Q8_K (Q8K) or Q8_0 activation, sixteen consecutive elements per lane.
template <int CODE_BITS, bool Q8K>
sycl::event launch_q8_split(sycl::queue& q, const uint8_t* x, const uint8_t* codes, const float* scales,
                            const float* offset, float* y, long long n_in, long long n_out, int bias, int codebook,
                            int group_shift, int has_offset) {
    constexpr int THREADS = 256;
    constexpr int WARPS = THREADS / WARP;
    const size_t groups = (size_t) ((n_out + WARPS - 1) / WARPS);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<signed char, 1> tbl(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0);
            signed char* t = &tbl[0];
            stage_codebook(it, t, tid);
            const long long o = (long long) it.get_group(0) * WARPS + (long long) sg.get_group_linear_id();
            if (o >= n_out) return;   // uniform over the sub-group
            const int lane = (int) sg.get_local_linear_id();
            constexpr int PER_BYTE = 8 / CODE_BITS;
            const long long n_groups = n_in >> group_shift;
            const uint8_t* c = codes + o * (n_in / PER_BYTE);
            const float* s = scales + o * n_groups;
            const float* off = has_offset ? offset + o * n_groups : nullptr;

            float acc[16];
            for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
            constexpr int QE = 16;
            constexpr int QW = 4 * CODE_BITS / 8;              // bytes per four-code word: 1, 2 or 4
            constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
            long long i = (long long) lane * QE;
            for (; i + QE <= n_in; i += WARP * QE) {
                const long long g = i >> group_shift;
                const float d = s[g];
                const float b = off ? off[g] : 0.0f;
                const uint8_t* cp = c + i / PER_BYTE;
                unsigned vw[4];
                for (int k = 0; k < 4; ++k) {
                    if constexpr (QW == 1) vw[k] = cp[k];
                    else if constexpr (QW == 2) vw[k] = *(const uint16_t*) (cp + 2 * k);
                    else vw[k] = *(const uint32_t*) (cp + 4 * k);
                }
                // all sixteen elements lie in one activation block (see the CUDA source)
                constexpr int blk_elems = Q8K ? Q8K_BLOCK_ELEMS : Q8_0_BLOCK_ELEMS;
                const uint8_t* xb = x + (i / blk_elems) * (Q8K ? Q8K_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
                const int xi = (int) (i % blk_elems);
                float xd;
                if constexpr (Q8K) xd = *(const float*) xb;
                else xd = f32_from_f16(*(const uint16_t*) xb);
                const int8_t* xq = (const int8_t*) (xb + (Q8K ? 4 : 2)) + xi;
                for (int k = 0; k < 16; ++k) {
                    const unsigned v = vw[k >> 2];
                    const float w = decode_tbl((int) ((v >> ((k & 3) * CODE_BITS)) & MASK), bias, codebook, t) * d + b;
                    acc[k] += w * (xd * (float) xq[k]);
                }
            }
            // the last, partial run - at most one per lane
            for (; i < n_in; i += WARP * QE) {
                for (int k = 0; k < QE && i + k < n_in; ++k) {
                    const long long e = i + k;
                    const long long g = e >> group_shift;
                    const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
                    acc[0] += (decode_tbl(code, bias, codebook, t) * s[g] + (off ? off[g] : 0.0f)) *
                              (Q8K ? q8k_at(x, e) : q8_0_at(x, e));
                }
            }
            float a = (((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))) +
                      (((acc[8] + acc[9]) + (acc[10] + acc[11])) + ((acc[12] + acc[13]) + (acc[14] + acc[15])));
            for (int step = 16; step > 0; step >>= 1) a += sycl::permute_group_by_xor(sg, a, step);
            if (lane == 0) y[o] = a;
        });
    });
}

// byte -> the four S2 code values with the -1 bias applied, in element order (bits 0, 2, 4, 6)
constexpr std::array<float, 1024> make_s2_lut() {
    std::array<float, 1024> t{};
    for (int b = 0; b < 256; ++b)
        for (int k = 0; k < 4; ++k) t[b * 4 + k] = (float) (((b >> (2 * k)) & 3) - 1);
    return t;
}
constexpr std::array<float, 1024> kS2Lut = make_s2_lut();

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0)
        fail("s_gemv: n_in " + std::to_string(n_in) + " is not a multiple of group_elems " +
             std::to_string(form.group_elems));
    if (form.has_offset && offset == nullptr) fail("s_gemv: form says has_offset but offset is null");
    auto& q = queue_for(nullptr);
    const int cb = (int) form.codebook, ho = form.has_offset ? 1 : 0;
    sycl::event e;
    switch (form.code_bits) {
        case 2: e = launch_naive<2, false>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho); break;
        case 4: e = launch_naive<4, false>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho); break;
        case 8: e = launch_naive<8, false>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho); break;
        default: fail("s_gemv: unsupported code_bits " + std::to_string(form.code_bits));
    }
    finish(nullptr, e, "s_gemv");
}

static void s_gemv_split_impl(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                              float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                              void* stream, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 || threads_per_row > 1024)
        fail("s_gemv_split: threads_per_row must be a power of two in 1..1024, got " + std::to_string(threads_per_row));
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0)
        fail("s_gemv_split: group_elems must be a power of two, got " + std::to_string(form.group_elems));
    if (form.group_elems % 4 != 0)
        fail("s_gemv_split: group_elems " + std::to_string(form.group_elems) + " is not a multiple of 4");
    const int gs = shift_of(form.group_elems);
    auto& q = queue_for(stream);
    const int cb = (int) form.codebook, ho = form.has_offset ? 1 : 0;
    sycl::event e;
    switch (form.code_bits) {
        case 2: e = launch_split<2>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, gs, ho, threads_per_row); break;
        case 4: e = launch_split<4>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, gs, ho, threads_per_row); break;
        case 8: e = launch_split<8>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, gs, ho, threads_per_row); break;
        default: fail("s_gemv_split: unsupported code_bits " + std::to_string(form.code_bits));
    }
    if (sync) core::Runtime::get().wait(e, "s_gemv_split");
}

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, nullptr, true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, stream, false);
}

namespace {

void q8k_form_check(const SForm& form, int64_t n_in, const char* who) {
    if (form.group_elems <= 0 || n_in % form.group_elems != 0)
        fail(std::string(who) + ": n_in " + std::to_string(n_in) + " is not a multiple of group_elems " +
             std::to_string(form.group_elems));
    if (n_in % Q8K_BLOCK_ELEMS != 0)
        fail(std::string(who) + ": n_in " + std::to_string(n_in) + " is not a multiple of the Q8_K block 256");
}

}  // namespace

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    q8k_form_check(form, n_in, "s_gemv_q8k");
    auto& q = queue_for(stream);
    const int cb = (int) form.codebook, ho = form.has_offset ? 1 : 0;
    sycl::event e;
    switch (form.code_bits) {
        case 4: e = launch_naive<4, true>(q, x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho); break;
        case 8: e = launch_naive<8, true>(q, x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems, ho); break;
        default:
            // 2-bit S2 never has a Q8_K activation: its vec_dot_type is Q8_0
            fail("s_gemv_q8k: code_bits " + std::to_string(form.code_bits) +
                 " has no Q8_K contract (S2 uses Q8_0; see docs/activation-contract.md)");
    }
    finish(stream, e, "s_gemv_q8k");
}

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    q8k_form_check(form, n_in, "s_gemv_q8k_split");
    const int gs = shift_of(form.group_elems);
    if ((1 << gs) != form.group_elems)
        fail("s_gemv_q8k_split: group_elems " + std::to_string(form.group_elems) + " is not a power of two");
    if (form.group_elems % 16 != 0)
        fail("s_gemv_q8k_split: group_elems " + std::to_string(form.group_elems) + " is not a multiple of 16");
    auto& q = queue_for(stream);
    const int cb = (int) form.codebook, ho = form.has_offset ? 1 : 0;
    sycl::event e;
    switch (form.code_bits) {
        case 4: e = launch_q8_split<4, true>(q, x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, gs, ho); break;
        case 8: e = launch_q8_split<8, true>(q, x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, gs, ho); break;
        default: fail("s_gemv_q8k_split: code_bits " + std::to_string(form.code_bits) + " has no Q8_K contract");
    }
    finish(stream, e, "s_gemv_q8k_split");
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % Q8_0_BLOCK_ELEMS != 0)
        fail("s_gemv_q8_0_split: n_in " + std::to_string(n_in) + " is not a multiple of 32");
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0)
        fail("s_gemv_q8_0_split: group_elems must be a power of two, got " + std::to_string(form.group_elems));
    // sixteen, not four: a lane-iteration takes QE = 16 consecutive elements under one scale, so a group of 4 or 8
    // would read the wrong scale for most of them (upstream 7ed0e90)
    if (form.group_elems % 16 != 0)
        fail("s_gemv_q8_0_split: group_elems " + std::to_string(form.group_elems) + " is not a multiple of 16");
    const int gs = shift_of(form.group_elems);
    auto& q = queue_for(stream);
    const int cb = (int) form.codebook, ho = form.has_offset ? 1 : 0;
    sycl::event e;
    switch (form.code_bits) {
        case 4: e = launch_q8_split<4, false>(q, x_q8_0, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, gs, ho); break;
        case 8: e = launch_q8_split<8, false>(q, x_q8_0, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, gs, ho); break;
        default: fail("s_gemv_q8_0_split: code_bits " + std::to_string(form.code_bits) + " has no Q8_0 contract");
    }
    finish(stream, e, "s_gemv_q8_0_split");
}

// ---- s2_gemv_quads.cu and s2_gemv_fast.cu -----------------------------------------------------------------

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                   int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) fail("s2_gemv_quads: n_in " + std::to_string(n_in) + " is not a multiple of 4");
    auto& q = queue_for(nullptr);
    const long long ni = n_in, no = n_out;
    const int tpr = threads_per_row;
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t) tpr), h);
        h.parallel_for(sycl::nd_range<1>((size_t) no * tpr, tpr), [=](sycl::nd_item<1> it) {
            const long long o = (long long) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const long long n_quads = ni / 4;
            const uint8_t* c = codes + o * n_quads;             // exactly one code byte per quad
            const float* s = scales + o * (ni / QK_S2);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (long long qd = tid; qd < n_quads; qd += tpr) {
                const uint8_t byte = c[qd];
                const float d = s[qd >> 4];                     // (q*4) >> 6
                const uint16_t* xq = x + qd * 4;
                const float w0 = (float) ((int) (byte & 3) - 1) * d;
                const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
                const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
                const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
                a0 += w0 * f32_from_f16(xq[0]);
                a1 += w1 * f32_from_f16(xq[1]);
                a2 += w2 * f32_from_f16(xq[2]);
                a3 += w3 * f32_from_f16(xq[3]);
            }
            partial[tid] = (a0 + a1) + (a2 + a3);
            sycl::group_barrier(it.get_group());
            for (int step = tpr / 2; step > 0; step >>= 1) {
                if (tid < step) partial[tid] += partial[tid + step];
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) y[o] = partial[0];
        });
    });
    finish(nullptr, e, "s2_gemv_quads");
}

// CUDA's measured-neutral experiment, kept for its parity: the code table (a __constant__ array there, a constexpr
// table here) and, with stage_x, x staged in local memory.
void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                  int64_t n_out, int threads_per_row, bool stage_x) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0)
        fail("s2_gemv_fast: n_in " + std::to_string(n_in) + " must be a multiple of 64");
    if (stage_x && n_in > MAX_SHARED_HALVES)
        fail("s2_gemv_fast: n_in " + std::to_string(n_in) + " exceeds the 4096-half local staging limit");
    auto& q = queue_for(nullptr);
    const long long ni = n_in, no = n_out;
    const int tpr = threads_per_row;
    const bool stage = stage_x;
    const auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> sx(sycl::range<1>(stage ? MAX_SHARED_HALVES : 1), h);
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t) tpr), h);
        h.parallel_for(sycl::nd_range<1>((size_t) no * tpr, tpr), [=](sycl::nd_item<1> it) {
            const long long o = (long long) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            if (stage) {
                for (long long i = tid; i < ni; i += tpr) sx[i] = x[i];
                sycl::group_barrier(it.get_group());
            }
            const long long n_quads = ni / 4;
            const uint8_t* c = codes + o * n_quads;
            const float* s = scales + o * (ni / QK_S2);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (long long qd = tid; qd < n_quads; qd += tpr) {
                const uint8_t byte = c[qd];
                const float d = s[qd >> 4];
                const float* cv = &kS2Lut[byte * 4];
                const float h0 = f32_from_f16(stage ? sx[qd * 4] : x[qd * 4]);
                const float h1 = f32_from_f16(stage ? sx[qd * 4 + 1] : x[qd * 4 + 1]);
                const float h2 = f32_from_f16(stage ? sx[qd * 4 + 2] : x[qd * 4 + 2]);
                const float h3 = f32_from_f16(stage ? sx[qd * 4 + 3] : x[qd * 4 + 3]);
                a0 += cv[0] * d * h0;
                a1 += cv[1] * d * h1;
                a2 += cv[2] * d * h2;
                a3 += cv[3] * d * h3;
            }
            partial[tid] = (a0 + a1) + (a2 + a3);
            sycl::group_barrier(it.get_group());
            for (int step = tpr / 2; step > 0; step >>= 1) {
                if (tid < step) partial[tid] += partial[tid + step];
                sycl::group_barrier(it.get_group());
            }
            if (tid == 0) y[o] = partial[0];
        });
    });
    finish(nullptr, e, "s2_gemv_fast");
}

}  // namespace strata::kernels
