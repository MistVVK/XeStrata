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
    if (fails) return 1;
    std::printf("q36_context_fit_test: ok\n");
    return 0;
}
