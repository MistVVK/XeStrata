/* tqp_ref_c.c - the fork's CPU TQP5/6/7 dequantiser as a tiny standalone harness.
 *
 * The decoder is verbatim from agentionai's llama.cpp fork:
 *   ggml/src/ggml-common.h  block_tqp5/6/7, tq2t_lut_f16, tq2t_hyb_index, tqp_state
 *   ggml/src/ggml-quants.c  dequantize_row_tqp_impl, GGML_FP16_TO_FP32
 * No ggml / llama.cpp build: this file compiles alone (cc -O2 tqp_ref_c.c).
 *
 * Usage: tqp_ref_c <k> <in.bin> <out.f32>   (k = 5, 6 or 7; in.bin = whole blocks;
 * out.f32 = one float32 per weight, little-endian)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef uint16_t ggml_half;

/* ggml-common.h: the TQ2_T "Hyb" codebook, 4096 fp16 bit patterns (2048 2-D points).
 * Generated from the fork's header by ref/tqp/gen_table.py - must stay bit-identical. */
#include "tq2t_lut_table.h"

#define TQ_STATE_BITS 15
#define TQ_STATE_MASK ((1u << TQ_STATE_BITS) - 1u)
#define QK_TQP 128

/* ggml-quants.c: GGML_FP16_TO_FP32 via the soft-float path (no fp16 hardware dependency) */
static const uint32_t fp32_from_bits[] = {
    0x00000000u, 0x38000000u, 0x38800000u, 0x39000000u, 0x39800000u, 0x3a000000u, 0x3a800000u, 0x3b000000u,
    0x3b800000u, 0x3c000000u, 0x3c800000u, 0x3d000000u, 0x3d800000u, 0x3e000000u, 0x3e800000u, 0x3f000000u,
    0x3f800000u, 0x40000000u, 0x40800000u, 0x41000000u, 0x41800000u, 0x42000000u, 0x42800000u, 0x43000000u,
    0x43800000u, 0x44000000u, 0x44800000u, 0x45000000u, 0x45800000u, 0x46000000u, 0x46800000u, 0x47000000u,
    0x47800000u, 0x48000000u, 0x48800000u, 0x49000000u, 0x49800000u, 0x4a000000u, 0x4a800000u, 0x4b000000u,
    0x4b800000u, 0x4c000000u, 0x4c800000u, 0x4d000000u, 0x4d800000u, 0x4e000000u, 0x4e800000u, 0x4f000000u,
    0x4f800000u, 0x50000000u, 0x50800000u, 0x51000000u, 0x51800000u, 0x52000000u, 0x52800000u, 0x53000000u,
    0x53800000u, 0x54000000u, 0x54800000u, 0x55000000u, 0x55800000u, 0x56000000u, 0x56800000u, 0x00000000u};

static float ggml_fp16_to_fp32(ggml_half x) {
    const uint32_t sign = (uint32_t) x & 0x8000u;
    const uint32_t e = ((uint32_t) x >> 10) & 0x1fu;
    const uint32_t m = (uint32_t) x & 0x3ffu;
    uint32_t f;
    if (e == 0) {                       /* subnormal / zero */
        if (m == 0) {
            f = sign << 16;
        } else {
            uint32_t mm = m;
            int exp = -14;
            while (!(mm & 0x400u)) { mm <<= 1; exp--; }
            mm &= 0x3ffu;
            f = (uint32_t) (exp + 127) << 23 | mm << 13 | sign << 16;
        }
    } else if (e == 0x1fu) {            /* inf / nan */
        f = 0x7f800000u | m << 13 | sign << 16;
    } else {
        f = (e + 112u) << 23 | m << 13 | sign << 16;
    }
    (void) fp32_from_bits;
    return *(const float *) &f;
}

/* ggml-common.h: tq2t_hyb_index */
static inline uint32_t tq2t_hyb_index(uint32_t e) {
    const uint32_t x = ((e >> 1) & TQ_STATE_MASK) * 0x9e3779b1u;
    return (e & 1) ? ((x >> 10) & 2047u) : (x >> 21);
}

/* ggml-common.h: tqp_state */
static inline uint32_t tqp_state(const uint8_t * qs, int k, int t) {
    const uint32_t nbytes = 8u * (uint32_t) k;
    const uint32_t off    = (uint32_t) (63 - t) * (uint32_t) k;
    const uint32_t i0     = off >> 3;
    const uint32_t i1     = (i0 + 1) % nbytes;
    const uint32_t i2     = (i0 + 2) % nbytes;
    const uint32_t w      = (uint32_t) qs[i0] | ((uint32_t) qs[i1] << 8) | ((uint32_t) qs[i2] << 16);
    return (w >> (off & 7)) & 0xffffu;
}

static inline float tq2t_lut_val(uint32_t i) {
    return ggml_fp16_to_fp32(tq2t_lut_f16[i]);
}

/* ggml-quants.c: dequantize_row_tqp_impl */
static void dequantize_row_tqp_impl(const uint8_t * x, float * y, int64_t n, int kb) {
    const int64_t nb = n / QK_TQP;
    const int nsteps = QK_TQP / 2;
    const size_t bsize = sizeof(ggml_half) + 8 * (size_t) kb;
    for (int64_t i = 0; i < nb; ++i, x += bsize) {
        const float d = ggml_fp16_to_fp32(*(const ggml_half *) x);
        const uint8_t * qs = x + sizeof(ggml_half);
        for (int t = 0; t < nsteps; ++t) {
            const uint32_t idx = tq2t_hyb_index(2 * tqp_state(qs, kb, t));
            *y++ = d * tq2t_lut_val(2 * idx + 0);
            *y++ = d * tq2t_lut_val(2 * idx + 1);
        }
    }
}

#ifdef TQP_REF_LIB
/* the entry points metal/tests/tqp_parity.cpp calls, when this file is linked into it (no main) */
void ggml_ref_dequantize_row_tqp5(const void * x, float * y, int64_t k) { dequantize_row_tqp_impl((const uint8_t *) x, y, k, 5); }
void ggml_ref_dequantize_row_tqp6(const void * x, float * y, int64_t k) { dequantize_row_tqp_impl((const uint8_t *) x, y, k, 6); }
void ggml_ref_dequantize_row_tqp7(const void * x, float * y, int64_t k) { dequantize_row_tqp_impl((const uint8_t *) x, y, k, 7); }
#else
int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <5|6|7> <in.bin> <out.f32>\n", argv[0]);
        return 2;
    }
    const int k = atoi(argv[1]);
    if (k < 5 || k > 7) { fprintf(stderr, "k must be 5, 6 or 7\n"); return 2; }
    FILE * in = fopen(argv[2], "rb");
    if (!in) { perror(argv[2]); return 1; }
    fseek(in, 0, SEEK_END);
    const long size = ftell(in);
    fseek(in, 0, SEEK_SET);
    const size_t bsize = 2 + 8 * (size_t) k;
    if ((size_t) size % bsize) { fprintf(stderr, "%s: %ld B not a whole number of %zu B blocks\n", argv[2], size, bsize); return 1; }
    uint8_t * buf = malloc((size_t) size);
    if (fread(buf, 1, (size_t) size, in) != (size_t) size) { fprintf(stderr, "short read\n"); return 1; }
    fclose(in);
    const int64_t n = (size_t) size / bsize * QK_TQP;
    float * out = malloc((size_t) n * sizeof(float));
    dequantize_row_tqp_impl(buf, out, n, k);
    FILE * f = fopen(argv[3], "wb");
    if (!f) { perror(argv[3]); return 1; }
    if (fwrite(out, sizeof(float), (size_t) n, f) != (size_t) n) { fprintf(stderr, "short write\n"); return 1; }
    fclose(f);
    fprintf(stderr, "tqp%d: %ld blocks -> %lld floats\n", k, (long) (size / (long) bsize), (long long) n);
    return 0;
}
#endif
