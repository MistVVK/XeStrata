#pragma once
#include <sycl/sycl.hpp>
#include "strata/kprof.hpp"
namespace strata::prefill::bmg_gdn_residual {
constexpr int N = 2560, HC = 4;
inline uint16_t bf(float v) {
    auto u = sycl::bit_cast<uint32_t>(v);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
// Each work-item handles two original channels. Fold the same eight warp sums
// in the same FP32 order as gr_write_norm_rs_kernel.
inline void write_norm(sycl::queue &q, float *r, const float *bo, const float *inj, int stride, const float *weight,
                       float eps, float *rs, uint16_t *hi, uint16_t *lo, int rows, int image_stride) {
    constexpr int F = 2, SG = 16, threads = 256 / F;
    strata::kprof::kp(q, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sums(8, cgh);
        cgh.parallel_for(sycl::nd_range<1>{size_t(rows) * HC * threads, threads},
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                             const int row = it.get_group(0), t = row / HC, c = row % HC, id = it.get_local_id(0);
                             auto sg = it.get_sub_group();
                             float sc = 2.f * (1.f / (1.f + sycl::native::exp(-inj[t * stride + c] / float(HC))));
                             using Vec = sycl::vec<float, F>;
                             Vec v[10];
                             Vec ss(0.f);
#pragma unroll
                             for (int k = 0; k < 10; ++k) {
                                 int d = 256 * k + id * F;
                                 Vec b = *reinterpret_cast<const Vec *>(bo + t * N + d);
                                 auto *dest = reinterpret_cast<Vec *>(r + row * N + d);
                                 Vec x = sycl::fma(b, Vec(sc), *dest);
                                 *dest = x;
                                 v[k] = x;
#pragma unroll
                                 for (int j = 0; j < F; ++j)
                                     ss[j] += x[j] * x[j];
                             }
#pragma unroll
                             for (int off = 16 / F; off; off /= 2)
#pragma unroll
                                 for (int j = 0; j < F; ++j)
                                     ss[j] += sycl::permute_group_by_xor(sg, float(ss[j]), off);
#pragma unroll
                             for (int off = F / 2; off; off /= 2)
#pragma unroll
                                 for (int j = 0; j < off; ++j)
                                     ss[j] += ss[j + off];
                             if ((id * F) % 32 == 0)
                                 sums[id * F / 32] = ss[0];
                             it.barrier(sycl::access::fence_space::local_space);
                             int lane = sg.get_local_linear_id();
                             float part = lane < 8 ? sums[lane] : 0.f;
#pragma unroll
                             for (int off = 4; off; off /= 2)
                                 part += sycl::permute_group_by_xor(sg, part, off);
                             float inv = sycl::rsqrt(sycl::group_broadcast(sg, part, 0) / float(N) + eps);
                             if (id == 0)
                                 rs[row] = inv;
#pragma unroll
                             for (int k = 0; k < 10; ++k) {
                                 int d = k * 256 + id * F;
                                 Vec x = v[k] * inv * (*reinterpret_cast<const Vec *>(weight + c * N + d));
                                 sycl::vec<uint16_t, F> upper, lower;
#pragma unroll
                                 for (int j = 0; j < F; ++j) {
                                     upper[j] = bf(x[j]);
                                     if (lo)
                                         lower[j] = bf(x[j] - sycl::bit_cast<float>(uint32_t(upper[j]) << 16));
                                 }
                                 size_t o = size_t(t) * image_stride + c * N + d;
                                 *reinterpret_cast<sycl::vec<uint16_t, F> *>(hi + o) = upper;
                                 if (lo)
                                     *reinterpret_cast<sycl::vec<uint16_t, F> *>(lo + o) = lower;
                             }
                         });
    });
}
} // namespace strata::prefill::bmg_gdn_residual
