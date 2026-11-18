// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/rows_i8.hpp - the W8A8 products' int8 rows: a row of values rotated by Sylvester's Hadamard
// matrix in groups of kI8Rotate and quantized with one scale for the row, the activations' tokens and the weights'
// rows alike (the image DiTs' products and the LLMs' dense prompt products, prefill::Gemm).  Moved here from the
// image engine (qi_kernels.hpp's qi_rows_i8, dit_rows.hpp's RowsI8Of) so the prompt paths reach the same kernels.
#pragma once

#include "strata/core/runtime.hpp"
#include "strata/kernels/q8_act.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {

/// The group a W8A8 product's operands are rotated in (rows_i8): K must be a multiple of it.
constexpr int64_t kI8Rotate = 256;

// A W8A8 product's int8 input (rows_i8 below) straight from a row-wise op: kI8Rotate values at a
// time through Sylvester's Hadamard butterflies, times 1 / 16, then one scale a row (q8_act.hpp's Mul rule), without
// FP16 rows and a second pass between: on the RX 9060 XT those two passes were a fifth of MOSS-Image's step
// (2026-10-29).  A work-group a row, a sub-group of 32 a group of kI8Rotate values (8 a lane, in registers), up to G
// groups each.  A lane holds its 8 values W side by side (value j at (j / W) 32 W + lane W + j % W of the group), so a
// bit of a value's position in the group is a bit of j or of lane by W.  The butterflies go over the position's bits
// in one order whatever W is (bit_at), within the lane for j's and across the lanes for lane's, and the row's sum of
// the squares is a tree over the same bits: the int8 values and the scales are the same bits for 1, 4 and 8.  Each
// sum in the lane's own order before (W 8 its bits 0 to 7, W 1 5, 6, 7, then 0 to 4; the squares a lane at a time)
// made the images depend on the width rows_i8_width timed (the B70, 2026-11-10: SeedVR2 7B's FP16 file, MOSS-Image
// and Mage-Flow each two or three images from start to start).  Op: raw(t, c), the value before the row's statistic (keep(t, c, v) writes it back
// where kKeep); where kRms, inv(sum of the squares) of the row; out(t, c, v, inv), the value given it (none: v).  Or
// W values at once: raw_w, keep_w, out_w; or with the sub-group, load_sg (a lane's 8 values of a GGUF block, W 8).
// row_scale(t), where the op has it: the row's own factor on its scale.  An op with kRegular rotates by the regular
// Hadamard matrix instead (kron(H4, H4, H4, H4) / 16, H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] / 2: the
// basis Comfy-Org's int8_convrot weights are saved in, hunyuanimage3_file.hpp), which is Sylvester's, a sign that flips
// for each of the position's base-4 digits that is 3, and Sylvester's again: H4 = S D S with D = diag(1, 1, 1, -1).
template<class Op> constexpr bool regular_rotation() {
    if constexpr (requires { Op::kRegular; }) return Op::kRegular;
    else return false;
}
template<class Op, int W, int G = 4> struct RowsI8Of {
    static constexpr int SG = 32, V = kI8Rotate / SG, kMaxG = G;
    static_assert(W == 1 || W == 4 || W == V);
    static_assert(G == 1 || G == 2 || G == 4);
    static constexpr bool kSg = requires(const Op& o, const sycl::sub_group& sg, float* p) {
        o.template load_sg<W>(sg, int64_t{}, int64_t{}, p);
    };
    // an op's W values side by side in one access (raw_w, keep_w where it keeps them; out_w): RmsI8 on the RTX 4070
    // through scalar reads W values apart kept the L1 busy at 44% of the memory's bandwidth (2026-11-05)
    static constexpr bool kVec = W > 1 && requires(const Op& o, float* p) {
        o.template raw_w<W>(int64_t{}, int64_t{}, p);
    } && (!Op::kKeep || requires(const Op& o, float* p) { o.template keep_w<W>(int64_t{}, int64_t{}, p); });
    static constexpr bool kVecOut = W > 1 && requires(const Op& o, float* p) {
        o.template out_w<W>(int64_t{}, int64_t{}, p, 1.f);
    };
    static constexpr bool kRegular = regular_rotation<Op>();
    // the position's bits in the order the sums take them (W 4's bits of j, then of lane, as before: W 4 kept its
    // rotation), and where each bit is: a mask of j's, else of lane's
    static constexpr int kLogW = W == 1 ? 0 : W == 4 ? 2 : 3;
    static constexpr int bit_at(int st) {
        constexpr int order[8] = {0, 1, 7, 2, 3, 4, 5, 6};
        return order[st];
    }
    static constexpr int j_mask(int b) { return b < kLogW ? 1 << b : b >= kLogW + 5 ? 1 << (b - 5) : 0; }
    static constexpr int lane_mask(int b) { return j_mask(b) != 0 || b < kLogW ? 0 : 1 << (b - kLogW); }
    // every position's bit in j's mask or in the sub-group's 32 lanes'
    static constexpr bool masks_cover() {
        for (int st = 0; st < 8; ++st) {
            const int b = bit_at(st);
            if (j_mask(b) == 0 && (b < kLogW || b - kLogW >= 5)) return false;
        }
        return true;
    }
    static_assert(masks_cover());
    static_assert(V == 8 && SG == 32 && kI8Rotate == 256);
    Op op;
    int64_t K, gpr, nsg;
    int8_t* q;
    float* s;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const auto r = static_cast<int64_t>(it.get_group_linear_id());
        const auto si = static_cast<int64_t>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        auto off = [](int j) { return (int64_t) (j / W) * SG * W + j % W; };   // value j past the lane's first
        // every loop unrolled, the groups a sub-group skips by a condition: rolled (the breaks they had), v went to
        // local memory on NVPTX (qi_kernels.cpp's RowsI8: the L2's writes 35 times its output, RTX 4070, 2026-11-05)
        float v[kMaxG][V];
        float ss = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxG; ++c) {
            const int64_t g = si + c * nsg;
            if (g >= gpr) continue;   // the whole sub-group
            const int64_t k0 = g * kI8Rotate + (int64_t) lane * W;
            if constexpr (kSg) {
                op.template load_sg<W>(sg, r, k0, v[c]);
            } else if constexpr (kVec) {   // W values a load: one access each, not W the lanes' accesses W values apart
#pragma unroll
                for (int h = 0; h < V / W; ++h) {
                    const int j = h * W;
                    op.template raw_w<W>(r, k0 + (int64_t) h * SG * W, &v[c][j]);
                    if constexpr (Op::kKeep) op.template keep_w<W>(r, k0 + (int64_t) h * SG * W, &v[c][j]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    v[c][j] = op.raw(r, k0 + off(j));
                    if constexpr (Op::kKeep) op.keep(r, k0 + off(j), v[c][j]);
                }
            }
            if constexpr (Op::kRms) {   // the group's sum of the squares, every lane the same tree (a + b == b + a)
                float sq[V];
#pragma unroll
                for (int j = 0; j < V; ++j) sq[j] = v[c][j] * v[c][j];
                int done = 0;   // j's bits summed: the sums left at the j that have none of them
#pragma unroll
                for (int st = 0; st < 8; ++st) {
                    const int b = bit_at(st), m = j_mask(b);
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        if ((j & done) != 0 || (j & m) != 0) continue;
                        sq[j] = m != 0 ? sq[j] + sq[j + m] : sq[j] + sycl::permute_group_by_xor(sg, sq[j], lane_mask(b));
                    }
                    done |= m;
                }
                ss += sq[0];
            }
        }
        float inv = 1.f;
        // the sub-groups' sums (each lane's the same) once each: the work-group's tree over them whatever W is
        if constexpr (Op::kRms)
            inv = op.inv(sycl::reduce_over_group(it.get_group(), lane == 0 ? ss : 0.f, sycl::plus<float>()));
        float amax = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxG; ++c) {
            const int64_t g = si + c * nsg;
            if (g >= gpr) continue;
            const int64_t k0 = g * kI8Rotate + (int64_t) lane * W;
            if constexpr (kVecOut) {
#pragma unroll
                for (int h = 0; h < V / W; ++h) {
                    const int j = h * W;
                    op.template out_w<W>(r, k0 + (int64_t) h * SG * W, &v[c][j], inv);
                }
            } else if constexpr (requires { op.out(r, k0, 1.f, inv); }) {
#pragma unroll
                for (int j = 0; j < V; ++j) v[c][j] = op.out(r, k0 + off(j), v[c][j], inv);
            }
            // Sylvester's butterflies over the position's bits in bit_at's order: within the lane for j's, across the
            // lanes for lane's
            auto sylvester = [&] {
#pragma unroll
                for (int st = 0; st < 8; ++st) {
                    const int b = bit_at(st), m = j_mask(b), l = lane_mask(b);
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        if (m != 0) {
                            if ((j & m) == 0) {
                                const float a = v[c][j], e = v[c][j + m];
                                v[c][j] = a + e;
                                v[c][j + m] = a - e;
                            }
                        } else {
                            const float o = sycl::permute_group_by_xor(sg, v[c][j], l);
                            v[c][j] = (lane & l) != 0 ? o - v[c][j] : v[c][j] + o;
                        }
                    }
                }
            };
            sylvester();
            float norm = 1.f / 16.f;
            if constexpr (kRegular) {   // D between two of Sylvester's: the value at position p of the group
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    const int p = (int) off(j) + lane * W;
                    int threes = 0;
#pragma unroll
                    for (int d = 0; d < 8; d += 2) threes += ((p >> d) & 3) == 3;
                    if (threes & 1) v[c][j] = -v[c][j];
                }
                sylvester();
                norm = 1.f / 256.f;
            }
#pragma unroll
            for (int j = 0; j < V; ++j) {
                v[c][j] *= norm;
                amax = sycl::fmax(amax, sycl::fabs(v[c][j]));
            }
        }
        amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float d = kernels::q8act::Mul::scale(amax), id = kernels::q8act::Mul::inv(d);
#pragma unroll
        for (int c = 0; c < kMaxG; ++c) {
            const int64_t g = si + c * nsg;
            if (g >= gpr) continue;
            int8_t* qg = q + r * K + g * kI8Rotate + (int64_t) lane * W;
#pragma unroll
            for (int h = 0; h < V / W; ++h) {
                auto q8 = [&](int p) { return (uint32_t) (uint8_t) kernels::q8act::Mul::value(v[c][h * W + p], d, id, amax); };
                if constexpr (W == 1) {
                    qg[(int64_t) h * SG] = (int8_t) q8(0);
                } else {   // packed into 32-bit words and stored as one: a sycl::vec<int8_t, W> store is split into
                           // its bytes on NVPTX
                    uint32_t o[W / 4] = {};
#pragma unroll
                    for (int p = 0; p < W; ++p) o[p / 4] |= q8(p) << (8 * (p % 4));
                    if constexpr (W == 4) {
                        *reinterpret_cast<uint32_t*>(qg + (int64_t) h * SG * W) = o[0];
                    } else {
                        using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
                        *reinterpret_cast<u32x2*>(qg + (int64_t) h * SG * W) = u32x2{o[0], o[1]};
                    }
                }
            }
        }
        if (si == 0 && lane == 0) {
            if constexpr (requires { op.row_scale(r); }) s[r] = d * op.row_scale(r);
            else s[r] = d;
        }
    }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const { return sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::sub_group_size<SG>}; }
};

// G groups a sub-group: ceil(K / kI8Rotate / G) sub-groups a row
template<class Op, int W, int G = 4>
inline sycl::event rows_i8_as(const Op& op, int64_t T, int64_t K, int8_t* q, float* s, sycl::queue& qu) {
    constexpr int64_t SG = RowsI8Of<Op, W, G>::SG;
    const int64_t gpr = K / kI8Rotate;
    const auto max_sg =
        static_cast<int64_t>(qu.get_device().get_info<sycl::info::device::max_work_group_size>()) / SG;
    if (K % kI8Rotate != 0 || gpr > max_sg * G)
        throw std::runtime_error("a W8A8 product takes rows of a multiple of 256 values, up to " +
                                 std::to_string(max_sg * G * kI8Rotate));
    const int64_t nsg = (gpr + G - 1) / G;
    return qu.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * nsg * SG)), sycl::range<1>((size_t) (nsg * SG))),
        RowsI8Of<Op, W, G>{op, K, gpr, nsg, q, s});
}

// rows_i8's launcher, 8 values side by side a lane (a GGUF block's 256 values a sub-group) and as few groups a
// sub-group as the work-group allows: the registers of 4 held the RTX 4070's occupancy to a third where 1 was enough
// (2026-11-05).  Src: RowsI8Of's op.
template<class Src>
void rows_i8_of(const Src& src, int64_t rows, int64_t K, int8_t* q, float* s, void* stream) {
    if (rows <= 0) return;
    sycl::queue& qu = core::Runtime::get().stream(stream);
    const int64_t gpr = K / kI8Rotate;
    const auto max_sg = static_cast<int64_t>(qu.get_device().get_info<sycl::info::device::max_work_group_size>()) / 32;
    const int64_t per = (gpr + max_sg - 1) / max_sg;
    const auto e = per <= 1 ? rows_i8_as<Src, 8, 1>(src, rows, K, q, s, qu)
                 : per == 2 ? rows_i8_as<Src, 8, 2>(src, rows, K, q, s, qu)
                            : rows_i8_as<Src, 8, 4>(src, rows, K, q, s, qu);
    if (!stream) core::Runtime::get().wait(e, "rows_i8");
}

/// The operands of a W8A8 product (prefill::Gemm::w8a8): rows of K FP16 (or with bf16 BF16) values (K up to 32768 on a
/// GPU of 1024-item work-groups), each group of kI8Rotate times the normalized Hadamard matrix (orthogonal, so the
/// product of two rotated operands is that of the originals; it spreads a channel's outliers over the group), then as
/// int8 with a scale for the row: q [rows][K], s [rows], times rs [rows] where given (rows the caller scaled down).
/// The weights' rows and the activations' tokens alike.
void rows_i8(const uint16_t* src, bool bf16, int64_t rows, int64_t K, int8_t* q, float* s, const float* rs,
             void* stream);

/// rows_i8 of a matrix of GGUF blocks (`type`: Q4_K, Q5_K or Q6_K; K a multiple of 256) straight from the blocks,
/// the int8 rows rows_i8 makes of kernels::dequant_f16's output; false for another type (nothing is written).
bool rows_i8_blocks(int type, const void* blocks, int64_t rows, int64_t K, int8_t* q, float* s, void* stream);

/// y [T][N] (row stride ldy) holds a W8A8 product's int32 sums (their bits): y = sum * sx[t] * sw[n] + bias[n], in
/// place (bias null: none).  col[n] (null: 1) is a DoRA's factor of the weight's row n: the product's part of y only,
/// the bias not scaled.
void i8_scale(float* y, int64_t T, int64_t N, int64_t ldy, const float* sx, const float* sw, const float* bias,
              const float* col, void* stream);

}  // namespace strata::kernels
