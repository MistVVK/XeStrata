// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/i8_means_test.cpp - I8Means (the int8 attention's K means kept from a step to the next): a slot's
// statistics, the kept-context cap (a slot past the kept ones gives no mean) and the new-image reset, against the
// meaning the DiTs rely on.  Host only, no GPU.
#include "strata/image/i8_means.hpp"

#include <cstdio>
#include <vector>

int main() {
    using strata::image::I8Means;
    std::vector<float> buf(size_t{6} * 4, 0.f);   // 6 slots of C = 4
    I8Means m;
    m.init(buf.data(), 6, 4);
    int fails = 0;
    auto ck = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("FAIL %s\n", what);
            ++fails;
        }
    };
    // fresh: every slot's mean not ready, its buffer given, no V scales
    for (int64_t s = 0; s < 6; ++s) {
        const auto st = m.stats(s);
        ck(st.kmean == buf.data() + s * 4 && !st.kmean_ready && st.sv == nullptr, "fresh slot");
    }
    // mark a slot: ready, the same buffer; the others stay fresh
    m.mark(2);
    ck(m.stats(2).kmean_ready, "marked ready");
    ck(!m.stats(3).kmean_ready, "other slot still fresh");
    // the V scales pass through
    float sv[4] = {1.f, 2.f, 3.f, 4.f};
    ck(m.stats(2, sv).sv == sv, "sv passed");
    // a slot past the kept ones (a third context): no mean, and marking it changes nothing
    const auto past = m.stats(6);
    ck(past.kmean == nullptr && !past.kmean_ready, "past slot no mean");
    m.mark(6);
    ck(m.stats(2).kmean_ready, "past mark a no-op");
    // reset (a new image): all fresh again, the buffer kept
    m.reset();
    ck(!m.stats(2).kmean_ready && m.stats(2).kmean == buf.data() + ptrdiff_t{2} * 4, "reset");
    // clear
    m.clear();
    ck(m.buf == nullptr && m.slots == 0 && m.ready.empty(), "clear");
    std::printf("%s\n", fails ? "i8_means: FAILED" : "i8_means: ok");
    return fails ? 1 : 0;
}