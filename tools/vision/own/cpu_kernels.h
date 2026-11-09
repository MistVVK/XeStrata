// tools/vision/own/cpu_kernels.h - the instruction-set specific kernels of the CPU tower (VitCpu), one table per ISA tier.
//
// Every tier lives in its own translation unit with its own compile flags (cpu_kernels_scalar.cpp, _avx2.cpp, _avx512.cpp, all from cpu_kernels_impl.h), so
// the portable build runs on any CPU and the dispatcher (cpu_ops()) picks the best tier the CPU has (quant::best_isa(), STRATA_FORCE_ISA caps it).
#pragma once

#include "cpu_caps.h"

#include <cstddef>
#include <cstdint>

namespace strata_vit {

struct CpuOps {
    const char* name;
    int MR, NR;        // the register tile of the product: MR rows of A against NR columns of B

    // Packs the weight rows n0 .. n0 + nvalid (nvalid <= NR; the rest of the NR columns are zero) of W[N][K] (row-major, row stride ldw elements), columns
    // k0 .. k0 + kc, into dst[kc][NR] float32.  kind: 0 float32, 1 BF16, 2 F16 (W points at elements of that type).
    void (*pack_w)(const void* W, int kind, size_t ldw, int n0, int nvalid, int k0, int kc, float* dst);

    // n contiguous 16-bit weights (kind 1 BF16, 2 F16) to float32 (bit exact).  The tower keeps its matrices as 16-bit panels [tile][K][NR] made once at load,
    // and a product task converts the panel it needs into a float buffer: no transposition in the hot path.
    void (*conv)(const void* src, int kind, size_t n, float* dst);

    // C[M][N] = init-mode + A[M][K] * B[K][N]; A rows lda apart, B rows ldb apart (B must be readable up to the next multiple of NR columns), C rows ldc apart.
    // init 0: C = bias (or zero) + A*B; 1: C += A*B; 2: C = C + bias + A*B.  bias: N floats or null.
    void (*gemm)(const float* A, size_t lda, int M, const float* B, size_t ldb, int K, int N, float* C, size_t ldc, int init, const float* bias);

    // x = gelu_tanh(x) on rows x n of a matrix with row stride ldx
    void (*gelu)(float* x, size_t ldx, int rows, int n);
    // y = LayerNorm(x) * w + b for one row of D
    void (*layernorm)(const float* x, float* y, int D, const float* w, const float* b, float eps);
    // x[i] += y[i], i < n
    void (*add)(float* x, const float* y, size_t n);
    // p[i] rounded to the 16-bit type and back (kind 1 BF16, 2 F16; round to nearest even): STRATA_VIT_REF_ROUND, the inputs of the products as ggml's CPU path hands them
    void (*round16)(float* p, size_t n, int kind);
    // One key tile of the online softmax for qrows query rows: S[qrows][nkeys] (row stride lds, already scaled) becomes P = exp(S - m_new) in place, the running
    // max m[r] and sum l[r] are updated and the context rows O[r][0..hd) (row stride ldo) are rescaled by exp(m_old - m_new) (O's P*V is added afterwards).
    void (*softmax_update)(float* S, size_t lds, int qrows, int nkeys, float* m, float* l, float* O, size_t ldo, int hd);
};

const CpuOps& cpu_ops_scalar();
const CpuOps& cpu_ops_avx2();      // defined only on x86-64 (cpu_ops_avx2 returns the scalar table elsewhere)
const CpuOps& cpu_ops_avx512();
const CpuOps& cpu_ops(quant::Isa isa);

}  // namespace strata_vit
