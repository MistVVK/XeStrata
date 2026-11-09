// src/quant/act_ref.cpp - the reference activation quantizers (Generic contract) and the Q4_0 requantizer.
//
// Derived from ggml's `quantize_row_q8_0_ref`, `quantize_row_q8_1_ref`, `quantize_row_q8_K_ref` and `quantize_row_q4_0_ref` (llama.cpp
// ggml/src/ggml-quants.c at 3cf03257f219afbe7334045ff7c6a06ac68c627d, Copyright (c) 2023-2026 The ggml authors, MIT).  Q8_0 / Q8_1 round half
// away from zero (roundf) here; ggml's AVX / AVX2 quantizers round half to even (x86/act_avx*.cpp).  Q8_K is ggml's scalar code on every x86
// flavour, and its scale is -127/max, not -128/max (the comment in the loop says why): do not "fix" that.
#include "strata/quant/act.hpp"

#include <cassert>

#include "internal.hpp"
#include "registry.hpp"

#define GGML_UNUSED(x) (void) (x)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace strata::quant {
namespace {

void q8_0_ref(const float* STRATA_RESTRICT x, block_q8_0* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void q8_1_ref(const float* STRATA_RESTRICT x, block_q8_1* STRATA_RESTRICT y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = fp32_to_fp16(sum*d);
    }
}

void q8_K_ref(const float* STRATA_RESTRICT x, block_q8_K* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void q4_0_ref(const float* STRATA_RESTRICT x, block_q4_0* STRATA_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

}  // namespace

void quantize_q8_0_ref(const float* x, void* y, int64_t n) { q8_0_ref(x, (block_q8_0*) y, n); }
void quantize_q8_1_ref(const float* x, void* y, int64_t n) { q8_1_ref(x, (block_q8_1*) y, n); }
void quantize_q8_K_ref(const float* x, void* y, int64_t n) { q8_K_ref(x, (block_q8_K*) y, n); }
void quantize_q4_0_ref(const float* x, void* y, int64_t n) { q4_0_ref(x, (block_q4_0*) y, n); }

namespace {
const ActKernel kGenericAct[] = {
    {quantize_q8_0_ref, Type::Q8_0, Arith::Generic, Isa::Scalar, "generic_q8_0"},
    {quantize_q8_1_ref, Type::Q8_1, Arith::Generic, Isa::Scalar, "generic_q8_1"},
    {quantize_q8_K_ref, Type::Q8_K, Arith::Generic, Isa::Scalar, "generic_q8_K"},
};
}  // namespace

ActTable generic_act_table() noexcept { return {kGenericAct, 3}; }

}  // namespace strata::quant
