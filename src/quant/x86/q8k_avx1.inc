void quantize_q8_K_avx1(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m128 absm = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 magic = _mm_set1_ps(12582912.f);
    const __m128i mant = _mm_set1_epi32(0x007fffff), off = _mm_set1_epi32(0x00400000);
    const __m128i c127 = _mm_set1_epi32(127);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m128 m = _mm_setzero_ps();
        for (int j = 0; j < QK_K; j += 4) m = _mm_max_ps(_mm_and_ps(_mm_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(m, _mm_movehl_ps(m, m));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m128 va = _mm_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 4) {
            const int msk = _mm_movemask_ps(_mm_cmpeq_ps(_mm_and_ps(_mm_loadu_ps(x + j), absm), va));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m128 vis = _mm_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 16) {
            __m128i v[4];
            for (int q = 0; q < 4; ++q) {
                __m128 p = _mm_mul_ps(vis, _mm_loadu_ps(x + j + 4 * q));
#if defined(__GNUC__) && !defined(__clang__)
                // GCC fuses a multiply and an add into an FMA by default (-ffp-contract=fast); the reference
                // rounds the product before the add, so the product must stay a separate value (no FMA under
                // -mavx, but the barrier keeps the contract if the file is ever built with a newer floor)
                __asm__("" : "+x"(p));
#endif
                const __m128 t = _mm_add_ps(p, magic);
                v[q] = _mm_min_epi32(_mm_sub_epi32(_mm_and_si128(_mm_castps_si128(t), mant), off), c127);
                // the reference stores MIN(127, v) into an int8 (keeps the low byte), and its bsums add those
                // bytes: the same here, so even a degenerate block (iscale overflowing to inf) gives the same bytes
                v[q] = _mm_srai_epi32(_mm_slli_epi32(v[q], 24), 24);
            }
            __m128i s = _mm_add_epi32(_mm_add_epi32(v[0], v[1]), _mm_add_epi32(v[2], v[3]));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            y[i].bsums[j / 16] = (int16_t) _mm_cvtsi128_si32(s);
            _mm_storeu_si128((__m128i*) (y[i].qs + j),
                             _mm_packs_epi16(_mm_packs_epi32(v[0], v[1]), _mm_packs_epi32(v[2], v[3])));
        }
        y[i].d = 1 / iscale;
    }
}
