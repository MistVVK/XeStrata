// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/cpu/resize_rows.cpp - see include/strata/kernels/cpu/resize_rows.hpp: the choice of the copy, on the
// baseline x86-64 (the copies are resize_rows_avx2.cpp's).

#include "strata/kernels/cpu/resize_rows.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif

namespace strata::kernels::cpu {

void resize_rows_avx2(const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out, const ResizeRowsTaps& t,
                      int bits, uint8_t* dst);
#if STRATA_AVXVNNI
void resize_rows_avxvnni(const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out, const ResizeRowsTaps& t,
                         int bits, uint8_t* dst);
#endif
void resize_rows_avx512(const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out, const ResizeRowsTaps& t,
                        int bits, uint8_t* dst);

ResizeRowsIsa resize_rows_isa() {
    static const ResizeRowsIsa isa = [] {
        if (!cpu_avx2_ok()) return ResizeRowsIsa::scalar;
        if (STRATA_AVXVNNI && cpu_avxvnni_ok()) return ResizeRowsIsa::avxvnni;
        if (cpu_avx512_ok()) return ResizeRowsIsa::avx512;
        return ResizeRowsIsa::avx2;
    }();
    return isa;
}

bool resize_rows_u8(ResizeRowsIsa isa, const uint8_t* src, int64_t lines, int64_t n_in, int64_t n_out,
                    const ResizeRowsTaps& t, int bits, uint8_t* dst) {
    if (isa == ResizeRowsIsa::scalar) return false;
    // the VNNI copies' low halves: up to 64 taps of 255 x 32767 stay within int32
    if (t.n_max > 64) isa = ResizeRowsIsa::avx2;
#if STRATA_AVXVNNI
    if (isa == ResizeRowsIsa::avxvnni) {
        resize_rows_avxvnni(src, lines, n_in, n_out, t, bits, dst);
        return true;
    }
#endif
    if (isa == ResizeRowsIsa::avx512) {
        resize_rows_avx512(src, lines, n_in, n_out, t, bits, dst);
        return true;
    }
    resize_rows_avx2(src, lines, n_in, n_out, t, bits, dst);
    return true;
}

}  // namespace strata::kernels::cpu
