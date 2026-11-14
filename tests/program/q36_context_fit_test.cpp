// strata-q36's one-session context check (GitHub #5715, include/strata/qwen35/context_fit.hpp): the reporter's RTX 5070 12 GB starts from its log,
// a card where every expert fits, and a card too small for any sentence.  CPU only.
#include "strata/qwen35/context_fit.hpp"

#include <cstdio>
#include <string>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++fails;
    }
}
constexpr size_t MiB = (size_t) 1 << 20;
constexpr double GiB = (double) (1ull << 30);
constexpr size_t KV = 11 * 2 * 256 * 2 * 4, SHARED = 2 * 256 * 2 * 2;
constexpr size_t STATE = 30 * 32 * 128 * 128 * 4;     // the GDN layers' state (~0.06 GiB)

// UD-Q4_K_XL on the reporter's RTX 5070: at 64K the log says 0.45 GiB free after the session and 1256 MiB needed by the hybrid buffers.
// fr0 (before the session, after the fp16 copy at this context) = 0.45 GiB + the session; the fp16 copy grows with the context.
strata::qwen35::OneSessionInput rtx5070(size_t ctx) {
    const size_t fr0_64k = (size_t) (0.45 * GiB) + KV * 65536 + STATE;
    const size_t fr0 = fr0_64k + SHARED * 65536 - SHARED * ctx;
    const size_t outputs = 2048 * 8 * 2048 * 4 + 2048 * 8 * 4;
    return {ctx, KV, SHARED, STATE, fr0, (size_t) (19.0 * GiB), 0, (size_t) (48 << 20) * 6, 1256 * MiB - outputs, outputs};
}

// UD-Q4_K_XL on an RTX 3060 Ti 8 GB, prompt chunk 512: the build with MMQ said 1.50 GiB needed, 0.41 GiB free, 7168 tokens fit at 32K; the build
// without MMQ went on and ended in place_experts with 1159 MiB needed (the staging pair, ~1127 MiB, + the CPU part's outputs) and 114 MiB free.
// staging_counted: whether the engine passes the staging pair (one_session_staging_counted), as the engine does at q36_engine.cu's fit check.
strata::qwen35::OneSessionInput rtx3060ti(size_t ctx, bool staging_counted) {
    const size_t outputs = 512 * 8 * 2048 * 4 + 512 * 8 * 4;
    const size_t staging = 1127 * MiB;
    const size_t fr0_32k = (size_t) (0.41 * GiB) + staging + outputs - SHARED * 32768;
    const size_t fr0 = fr0_32k + SHARED * 32768 - SHARED * ctx;
    return {ctx, KV, SHARED, STATE, fr0, (size_t) (19.0 * GiB), 0, (size_t) (48 << 20) * 6, staging_counted ? staging : 0, outputs};
}
}  // namespace

int main() {
    using strata::qwen35::one_session_fit;
    using strata::qwen35::one_session_fit_line;
    // 32K ran on that card: no refusal
    check(!one_session_fit(rtx5070(32768)).refuse, "32K on the 12 GB card is not refused");
    // 64K ended with the hybrid buffers' text, 128K with the bare out-of-memory line: now the owner's sentence, with the size that fits
    for (size_t ctx : {(size_t) 65536, (size_t) 131072}) {
        const auto f = one_session_fit(rtx5070(ctx));
        check(f.refuse, "64K / 128K on the 12 GB card are refused in words");
        check(f.fit >= 32768 && f.fit < 49152 && f.fit % 1024 == 0, "the size said to fit is between 32K and 48K, in steps of 1024");
        check(f.x_gib > f.y_gib, "the memory it needs is more than the memory said to be free");
        std::printf("%s\n", one_session_fit_line((long long) ctx, f).c_str());
    }
    const auto f64 = one_session_fit(rtx5070(65536));
    check(one_session_fit_line(65536, f64) ==
              "a context of 65536 tokens does not fit on this graphics card: its conversation memory needs 2.93 GiB and only 2.16 GiB is free "
              "after the model. Start it again with a shorter context (47104 tokens fit here).",
          "the 64K line, word for word");
    check(!one_session_fit(rtx5070((size_t) f64.fit)).refuse, "the size it names is not refused");
    // a card where every main layer's experts still fit after the session: no hybrid minimum asked, only the session itself
    {
        strata::qwen35::OneSessionInput in{131072, KV, SHARED, STATE, (size_t) (24.0 * GiB), (size_t) (18.0 * GiB), 0, 0, 1100 * MiB, 128 * MiB};
        check(!one_session_fit(in).refuse, "a 32 GB card at 128K with every expert fitting is not refused");
        in.fr0 = (size_t) (5.0 * GiB);   // the session alone (5.4 GiB) does not fit
        check(one_session_fit(in).refuse, "a session larger than the free VRAM is refused");
    }
    // a card where not even 1024 tokens fit: nothing said here, the engine goes on as before
    check(!one_session_fit(strata::qwen35::OneSessionInput{32768, KV, SHARED, STATE, 1100 * MiB, (size_t) (19.0 * GiB), 0, 0, 1000 * MiB, 128 * MiB}).refuse,
          "no sentence when nothing fits");
    // the staging pair in the floor: the build with MMQ keeps its rule (MMQ or our int8 kernels on a layer), the build without MMQ counts it on the
    // streamed prompt path too (the fused kernels take the layers there: use_mmq false, n_i8 0), as place_experts allocates it
    {
        using strata::qwen35::one_session_staging_counted;
        check(one_session_staging_counted(true, false, true, true), "MMQ build, a layer on MMQ: counted (as before)");
        check(one_session_staging_counted(false, true, true, true), "MMQ build, a layer on our int8 kernels: counted (as before)");
        check(!one_session_staging_counted(false, false, true, true), "MMQ build, every layer fused: not counted (as before)");
        check(!one_session_staging_counted(false, false, false, true), "MMQ build, no streamed prompt path: not counted (as before)");
        check(one_session_staging_counted(false, false, true, false), "no-MMQ build, every layer fused: counted (new)");
        check(one_session_staging_counted(false, true, true, false), "no-MMQ build, a layer on our int8 kernels: counted (as before)");
        check(!one_session_staging_counted(false, false, false, false), "no-MMQ build, no streamed prompt path: not counted (place_experts has no staging)");
    }
    // the RTX 3060 Ti at 32K: without the staging pair (the no-MMQ build before) no sentence; with it the MMQ build's line, word for word
    {
        using strata::qwen35::one_session_staging_counted;
        check(!one_session_fit(rtx3060ti(32768, false)).refuse, "3060 Ti 32K, staging not counted: not refused (the old no-MMQ gap)");
        const auto f = one_session_fit(rtx3060ti(32768, one_session_staging_counted(false, false, true, false)));
        check(f.refuse, "3060 Ti 32K, no-MMQ build, fused layers: refused in words");
        check(one_session_fit_line(32768, f) ==
                  "a context of 32768 tokens does not fit on this graphics card: its conversation memory needs 1.50 GiB and only 0.41 GiB is free "
                  "after the model. Start it again with a shorter context (7168 tokens fit here).",
              "the 3060 Ti 32K line, word for word");
        std::printf("%s\n", one_session_fit_line(32768, f).c_str());
        check(!one_session_fit(rtx3060ti((size_t) f.fit, true)).refuse, "3060 Ti: the size it names is not refused");
        check(!one_session_fit(rtx3060ti(7168, true)).refuse, "3060 Ti 7168: not refused (starts as before)");
    }
    if (fails) return 1;
    std::printf("q36_context_fit_test: ok\n");
    return 0;
}
