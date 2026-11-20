// sy_gemm_xmx.h - prefill GEMM of the trunk on the Arc XMX engines (joint_matrix, DG2 shape 8x8x16 fp16 -> fp32), S_K weights dequantised into local memory.
//   Y[m][n] = sum_k W[n][k] X[m][k];  W = S_K [N][K] (K % 256 == 0, N % 64 == 0) in the raw wire format or in the RP layout (sy_rp.h), X half [M rows, K], Y float [M][N], M % 8 == 0.
// Work-group = 8 sub-groups of 8 lanes (64 threads): tile 64 tokens x 64 output rows; one super-block (256 k) of the 64 weight rows is dequantised per step into a VNNI-packed
// B tile in local memory (32 KiB), every sub-group owns 8 token rows and 8 accumulators (8x64 outputs).
#pragma once
#include "sy_common.h"
#include "sy_linear.h"
#include "sy_rp.h"

namespace hy4s {

namespace xm = sycl::ext::oneapi::experimental::matrix;

template <int FMT, bool RP>
inline void k_gemm_xmx(const uint8_t* W, int N, int K, const half* X, float* Y, int M) {
    constexpr int SB = sk_sb_bytes(FMT);
    constexpr int PB = FMT == FMT_S4 ? 128 : FMT == FMT_S5 ? 160 : 256;
    const int nbm = (M + 63) / 64, nbn = N / 64;
    const size_t pay_off = RP ? rp_pay_off(N, K >> 8) : 0;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sB(64 * 256, h);
        h.parallel_for(sycl::nd_range<1>((size_t)nbm * nbn * 64, 64), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
            const int gid = (int)it.get_group(0);
            const int bn = gid % nbn, bm = gid / nbn;
            const int tid = (int)it.get_local_id(0);
            auto sg = it.get_sub_group();
            const int sgi = tid >> 3;
            const int nsb = K >> 8;
            const int n_row = bn * 64 + tid;  // weight row dequantised by this thread
            const int m0 = bm * 64 + sgi * 8;
            const bool active = m0 + 8 <= M;  // M must be a multiple of 8 (the caller handles the tail rows)
            xm::joint_matrix<sycl::sub_group, half, xm::use::a, 8, 16, xm::layout::row_major> ma;
            xm::joint_matrix<sycl::sub_group, half, xm::use::b, 16, 8, xm::layout::ext_intel_packed> mb;
            xm::joint_matrix<sycl::sub_group, float, xm::use::accumulator, 8, 8> mc[8];
            for (int j = 0; j < 8; ++j) xm::joint_matrix_fill(sg, mc[j], 0.0f);
            half* sb_ptr = &sB[0];
            auto sB_mp = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sb_ptr);
            for (int sbi = 0; sbi < nsb; ++sbi) {
                // dequantise 256 weights of row n_row (tid) -> packed B tile: element (k, n) at [(k>>1)*128 + n*2 + (k&1)]
                if constexpr (RP) {
                    const uint8_t* hd = W + ((size_t)n_row * nsb + sbi) * 8;
                    const uint8_t* pl = W + pay_off + ((size_t)n_row * nsb + sbi) * PB;
                    const float D = h2f_bits(ld16(hd));
                    for (int g = 0; g < 4; ++g) {
                        const float eff = D * ((float)hd[2 + g] / 255.0f);
                        if constexpr (FMT == FMT_S4) {
                            for (int i = 0; i < 32; ++i) {  // 32 bytes = 64 weights of group g; weights k and k+1 share the k>>1 row of the packed tile
                                const uint8_t b = pl[g * 32 + i];
                                const int k = g * 64 + 2 * i;
                                sb_ptr[(k >> 1) * 128 + tid * 2 + 0] = (half)(((float)(b & 15) - 7.5f) * eff);
                                sb_ptr[(k >> 1) * 128 + tid * 2 + 1] = (half)(((float)(b >> 4) - 7.5f) * eff);
                            }
                        } else if constexpr (FMT == FMT_S5) {
                            for (int i = 0; i < 32; ++i) {  // low nibbles pl[0..128), high bits pl[128..160): weight k has bit k&7 of byte 128 + (k>>3)
                                const uint8_t b = pl[g * 32 + i];
                                const int k = g * 64 + 2 * i;
                                const uint32_t hbits = pl[128 + (k >> 3)] >> (k & 7);
                                sb_ptr[(k >> 1) * 128 + tid * 2 + 0] = (half)(((float)((b & 15) | ((hbits & 1u) << 4)) - 15.5f) * eff);
                                sb_ptr[(k >> 1) * 128 + tid * 2 + 1] = (half)(((float)((b >> 4) | (((hbits >> 1) & 1u) << 4)) - 15.5f) * eff);
                            }
                        } else {
                            for (int i = 0; i < 64; ++i) {
                                const int k = g * 64 + i;
                                sb_ptr[(k >> 1) * 128 + tid * 2 + (k & 1)] = (half)((float)(int8_t)pl[k] * eff);
                            }
                        }
                    }
                } else {
                    const uint8_t* blk = W + (size_t)n_row * nsb * SB + (size_t)sbi * SB;
                    const float D = h2f_bits(ld16(blk));
                    for (int g = 0; g < 4; ++g) {
                        const float eff = D * ((float)blk[2 + g] / 255.0f);
                        for (int i = 0; i < 64; ++i) {
                            const int k = g * 64 + i;
                            sb_ptr[(k >> 1) * 128 + tid * 2 + (k & 1)] = (half)(((float)sk_q<FMT>(blk + 6, k) - sk_half<FMT>()) * eff);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                if (active) {
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
            if (active) {
                for (int j = 0; j < 8; ++j) {
                    auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(Y + (size_t)m0 * N + bn * 64 + j * 8);
                    xm::joint_matrix_store(sg, mc[j], yp, (size_t)N, xm::layout::row_major);
                }
            }
        });
    });
}

inline bool gemm_xmx(int fmt, const void* Wv, int N, int K, const half* X, float* Y, int M, bool rp = false) {
    const uint8_t* W = (const uint8_t*)Wv;
    if ((N & 63) || (K & 255) || (M & 7)) return false;
    if (rp) {
        if (fmt == FMT_S4) { k_gemm_xmx<FMT_S4, true>(W, N, K, X, Y, M); return true; }
        if (fmt == FMT_S5) { k_gemm_xmx<FMT_S5, true>(W, N, K, X, Y, M); return true; }
        if (fmt == FMT_S8) { k_gemm_xmx<FMT_S8, true>(W, N, K, X, Y, M); return true; }
        return false;
    }
    switch (fmt) {
        case FMT_S4: k_gemm_xmx<FMT_S4, false>(W, N, K, X, Y, M); return true;
        case FMT_S5: k_gemm_xmx<FMT_S5, false>(W, N, K, X, Y, M); return true;
        case FMT_S6: k_gemm_xmx<FMT_S6, false>(W, N, K, X, Y, M); return true;
        case FMT_S8: k_gemm_xmx<FMT_S8, false>(W, N, K, X, Y, M); return true;
        case FMT_S2: k_gemm_xmx<FMT_S2, false>(W, N, K, X, Y, M); return true;
        case FMT_S3: k_gemm_xmx<FMT_S3, false>(W, N, K, X, Y, M); return true;
        default: return false;
    }
}

// production entry. rp: the weight is in the RP layout (S4/S8 repacked at load time). M <= 8: GEMV; M >= 16: XMX GEMM for the first (M & ~7) rows, the rest by the GEMV/simple kernel.
inline void linear_auto(int fmt, const void* W, int N, int K, const half* X, float* Y, int M, bool rp = false) {
    int Mx = 0;
    if (M >= 16 && !(N & 63) && !(K & 255) && fmt != FMT_Q8ROW && fmt != FMT_F32) {
        Mx = M & ~7;
        if (!gemm_xmx(fmt, W, N, K, X, Y, Mx, rp)) Mx = 0;
    }
    for (int m0 = Mx; m0 < M; m0 += 8) {
        const int mm = std::min(8, M - m0);
        if (rp) {
            if (!gemv_rp(fmt, W, N, K, X + (size_t)m0 * K, Y + (size_t)m0 * N, mm)) { fprintf(stderr, "linear_auto: rp fmt %d unsupported\n", fmt); abort(); }
        } else {
            linear_h(fmt, W, N, K, X + (size_t)m0 * K, Y + (size_t)m0 * N, mm);
        }
    }
}

}  // namespace hy4s
