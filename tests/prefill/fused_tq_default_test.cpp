// the default rule of the CUDA fused trellis (Gyro) prompt kernels (include/strata/prefill/fused_tq_default.hpp): unset = on for a compute capability 8.6 card with 80+ SMs,
// off for every other class; =1 forces on, any other value off.  No GPU, no model.
#include <cstdio>

#include "strata/prefill/fused_tq_default.hpp"

using strata::prefill::fused_tq::gyro_m_on;
using strata::prefill::fused_tq::on;
using strata::prefill::fused_tq::parse_env;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the variable's three readings
    CHECK(parse_env(nullptr) == -1 && parse_env("") == -1);
    CHECK(parse_env("1") == 1 && parse_env("1x") == 1);
    CHECK(parse_env("0") == 0 && parse_env("2") == 0 && parse_env("off") == 0);

    // unset: the sm_86 cards with 80+ SMs take the trellis kernels (A6000 / A40 84, 3090 Ti 84, 3090 82, 3080 Ti 80) ...
    const int unset = parse_env(nullptr);
    CHECK(on(unset, true, 84) && on(unset, true, 82) && on(unset, true, 80));
    // ... and nothing else does: RTX 3080 (68), A10 (72), A4000 (48) on sm_86 below 80 SMs; sm_120 cards (5070 48, 5080 84, PRO 5000 110, PRO 6000 188), GB10, sm_80 / 89 / 90
    // (s86 false) whatever their SM count
    CHECK(!on(unset, true, 79) && !on(unset, true, 72) && !on(unset, true, 68) && !on(unset, true, 48));
    CHECK(!on(unset, false, 48) && !on(unset, false, 84) && !on(unset, false, 110) && !on(unset, false, 188) && !on(unset, false, 108) && !on(unset, false, 128));
    // STRATA_PF_FUSED_SM86=0 makes DevInfo::s86 false: the A6000 class is back on MMQ
    CHECK(!on(unset, false, 84));

    // =1: the opt-in of before, on any card class (the kernels still have to set up on it)
    CHECK(on(1, true, 84) && on(1, true, 48) && on(1, false, 48) && on(1, false, 110) && on(1, false, 188));
    // =0 (or any other value): off, also on the A6000 class
    CHECK(!on(0, true, 84) && !on(0, true, 80) && !on(0, false, 110));
    CHECK(!on(parse_env("0"), true, 84) && !on(parse_env("false"), true, 84) && on(parse_env("1"), false, 48));

    // Gyro-M (TQ2_T / TQ2_T) has no gate series: never by default, only with =1
    CHECK(!gyro_m_on(parse_env(nullptr)) && !gyro_m_on(parse_env("")) && !gyro_m_on(parse_env("0")) && gyro_m_on(parse_env("1")));

    if (failed) {
        std::fprintf(stderr, "fused_tq_default_test: %d failed\n", failed);
        return 1;
    }
    std::printf("fused_tq_default_test: all checks passed\n");
    return 0;
}
