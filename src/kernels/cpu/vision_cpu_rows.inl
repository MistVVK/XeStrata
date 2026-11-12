// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/vision_cpu_rows.inl - the vision encoder's row loops that are plain C++ (vision_cpu.hpp): included
// by each instruction set's file inside its own namespace, so each is compiled, and vectorized, with that file's flags.

// The heads of rows [t0, t1) out of the QKV product's rows [T][q | k | v][heads][hd]: q and k turned by the 2-D RoPE's
// table (pairs j, j + hd / 2) into [heads][T][hd], q scaled; the values transposed into [heads][hd][T].
void qkv_rope_rows(const float* qkv, int64_t t0, int64_t t1, int64_t T, int64_t heads, int64_t hd, const float* cos,
                   const float* sin, float q_scale, float* q, float* k, float* vt) {
    const int64_t C = heads * hd, half = hd / 2;
    for (int64_t t = t0; t < t1; ++t) {
        const float* c = cos + t * half;
        const float* s = sin + t * half;
        for (int64_t h = 0; h < heads; ++h) {
            const float* qi = qkv + t * 3 * C + h * hd;
            const float* ki = qi + C;
            const float* vi = qi + 2 * C;
            float* qo = q + (h * T + t) * hd;
            float* ko = k + (h * T + t) * hd;
            for (int64_t j = 0; j < half; ++j) {
                const float qa = qi[j], qb = qi[j + half], ka = ki[j], kb = ki[j + half];
                qo[j] = (qa * c[j] - qb * s[j]) * q_scale;
                qo[j + half] = (qa * s[j] + qb * c[j]) * q_scale;
                ko[j] = ka * c[j] - kb * s[j];
                ko[j + half] = ka * s[j] + kb * c[j];
            }
            float* vo = vt + h * hd * T + t;
            for (int64_t d = 0; d < hd; ++d) vo[d * T] = vi[d];
        }
    }
}
