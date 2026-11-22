// Copyright (c) 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
//
// arm_act.inc - ARM64 NEON activation quantizers for Q8_0, Q8_1, and Q8_K.
// Q8_0 / Q8_1 round half to even (vcvtnq_s32_f32) with vectorized narrow-and-store.
// Q8_K matches quantize_q8_K_ref byte-for-byte on both GCC (-ffp-contract=fast) and Clang (unfused mul+add)
// using 4-way unrolled max-abs, exact first-occurrence sign scan, narrow-store, and single-instruction vaddlvq_s8 bsums.

void quantize_q8_0(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = (int) (k / QK8_0);

    block_q8_0* STRATA_RESTRICT y = (block_q8_0*) vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
        amaxv[0] = vmaxq_f32(amaxv[0], amaxv[4]);

        const float amax = vmaxvq_f32(amaxv[0]);
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f2h(d);

        int16x8_t n[4];
        for (int j = 0; j < 4; j++) {
            const int32x4_t vi0 = vcvtnq_s32_f32(vmulq_n_f32(srcv[2 * j + 0], id));
            const int32x4_t vi1 = vcvtnq_s32_f32(vmulq_n_f32(srcv[2 * j + 1], id));
            n[j] = vcombine_s16(vmovn_s32(vi0), vmovn_s32(vi1));
        }
        vst1q_s8(y[i].qs +  0, vcombine_s8(vmovn_s16(n[0]), vmovn_s16(n[1])));
        vst1q_s8(y[i].qs + 16, vcombine_s8(vmovn_s16(n[2]), vmovn_s16(n[3])));
    }
#else
    UNUSED(nb);
    quantize_q8_0_ref(x, y, k);
#endif
}

void quantize_q8_1(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = (int) (k / QK8_1);

    block_q8_1* STRATA_RESTRICT y = (block_q8_1*) vy;
#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
        amaxv[0] = vmaxq_f32(amaxv[0], amaxv[4]);

        const float amax = vmaxvq_f32(amaxv[0]);
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f2h(d);

        int32x4_t accv = vdupq_n_s32(0);
        int16x8_t n[4];
        for (int j = 0; j < 4; j++) {
            const int32x4_t vi0 = vcvtnq_s32_f32(vmulq_n_f32(srcv[2 * j + 0], id));
            const int32x4_t vi1 = vcvtnq_s32_f32(vmulq_n_f32(srcv[2 * j + 1], id));
            accv = vaddq_s32(accv, vaddq_s32(vi0, vi1));
            n[j] = vcombine_s16(vmovn_s32(vi0), vmovn_s32(vi1));
        }
        vst1q_s8(y[i].qs +  0, vcombine_s8(vmovn_s16(n[0]), vmovn_s16(n[1])));
        vst1q_s8(y[i].qs + 16, vcombine_s8(vmovn_s16(n[2]), vmovn_s16(n[3])));
        y[i].s = f2h(d * vaddvq_s32(accv));
    }
#else
    UNUSED(nb);
    quantize_q8_1_ref(x, y, k);
#endif
}

void quantize_q8_K(const float* STRATA_RESTRICT x, void* STRATA_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    block_q8_K* STRATA_RESTRICT y = (block_q8_K*) vy;

#if defined(__ARM_NEON)
    const float32x4_t magic = vdupq_n_f32(12582912.f);
    const int32x4_t mant = vdupq_n_s32(0x007fffff);
    const int32x4_t off = vdupq_n_s32(0x00400000);
    const int32x4_t c127 = vdupq_n_s32(127);

    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        if (i + 1 < nb) {
            __builtin_prefetch(x + QK_K, 0, 0);
            __builtin_prefetch(x + QK_K + 64, 0, 0);
        }

        // 4-way unrolled max(abs(x)) across all 256 floats
        float32x4_t m0 = vdupq_n_f32(0.0f);
        float32x4_t m1 = vdupq_n_f32(0.0f);
        float32x4_t m2 = vdupq_n_f32(0.0f);
        float32x4_t m3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK_K; j += 16) {
            m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + j +  0)));
            m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x + j +  4)));
            m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(x + j +  8)));
            m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(x + j + 12)));
        }
        const float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
        if (!amax) {
            y[i].d = 0.0f;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }

        // Exact first-occurrence sign scan matching q8_K_ref (`if (ax > amax)`)
        float mx = 0.0f;
        const float32x4_t vamax = vdupq_n_f32(amax);
        for (int j = 0; j < QK_K; j += 4) {
            const uint32x4_t eq = vceqq_f32(vabsq_f32(vld1q_f32(x + j)), vamax);
            if (vmaxvq_u32(eq)) {
                if      (vgetq_lane_u32(eq, 0)) mx = x[j + 0];
                else if (vgetq_lane_u32(eq, 1)) mx = x[j + 1];
                else if (vgetq_lane_u32(eq, 2)) mx = x[j + 2];
                else                            mx = x[j + 3];
                break;
            }
        }

        const float iscale = -127.f / mx;
        const float32x4_t vis = vdupq_n_f32(iscale);

        // Quantize 16 floats per iteration (1 bsums sub-block)
        for (int j = 0; j < QK_K; j += 16) {
            int32x4_t v[4];
            for (int q = 0; q < 4; ++q) {
                const float32x4_t vx = vld1q_f32(x + j + 4 * q);
#if defined(__GNUC__) && !defined(__clang__)
                // Matches GCC ggml-base (-ffp-contract=fast), which fuses nearest_int(iscale * x) into FMA
                const float32x4_t t = vfmaq_f32(magic, vis, vx);
#else
                // Matches Clang ggml-base (-ffp-contract=on), which rounds iscale * x before adding magic
                float32x4_t p = vmulq_f32(vis, vx);
#if defined(__clang__)
                __asm__("" : "+w"(p));
#endif
                const float32x4_t t = vaddq_f32(p, magic);
#endif
                v[q] = vminq_s32(vsubq_s32(vandq_s32(vreinterpretq_s32_f32(t), mant), off), c127);
            }
            const int16x8_t n01 = vcombine_s16(vmovn_s32(v[0]), vmovn_s32(v[1]));
            const int16x8_t n23 = vcombine_s16(vmovn_s32(v[2]), vmovn_s32(v[3]));
            const int8x16_t b = vcombine_s8(vmovn_s16(n01), vmovn_s16(n23));
            vst1q_s8(y[i].qs + j, b);
            y[i].bsums[j / 16] = (int16_t) vaddlvq_s8(b);
        }
        y[i].d = 1.0f / iscale;
    }
#else
    UNUSED(nb);
    quantize_q8_K_ref(x, y, k);
#endif
}
