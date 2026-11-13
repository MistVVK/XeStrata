// own_only_switches (include/strata/prefill/own_only_switches.hpp): the opt-in fused-kernel switches a build without MMQ turns on by default.  No GPU, no model.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strata/prefill/own_only_switches.hpp"

using strata::prefill::own_only_default_switches;
using strata::prefill::own_only_switch_on;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // CUDA: the list the build had before the Q2_0 kernel joined, in its order, then the Q2_0 / Q2_0 switch last
    const std::vector<std::string> cuda_before = {"STRATA_PF_FUSED", "STRATA_PF_FUSED_Q50", "STRATA_PF_FUSED_Q8D", "STRATA_PF_FUSED_Q8GU",
                                                  "STRATA_PF_FUSED_Q6K", "STRATA_PF_FUSED_KQ", "STRATA_PF_FUSED_TQ", "STRATA_PF_FUSED_SM75"};
    const auto cuda = own_only_default_switches(false);
    CHECK(cuda.size() == cuda_before.size() + 1);
    for (size_t i = 0; i < cuda_before.size() && i < cuda.size(); ++i) CHECK(cuda_before[i] == cuda[i]);
    CHECK(own_only_switch_on("STRATA_PF_FUSED_Q2N", false));
    // HIP: unchanged (the gfx12 Q2_0 kernel is a default of its own there); no Turing switch, no Q2N
    const std::vector<std::string> hip_before = {"STRATA_PF_FUSED", "STRATA_PF_FUSED_Q50", "STRATA_PF_FUSED_Q8D", "STRATA_PF_FUSED_Q8GU",
                                                 "STRATA_PF_FUSED_Q6K", "STRATA_PF_FUSED_KQ", "STRATA_PF_FUSED_TQ"};
    const auto hip = own_only_default_switches(true);
    CHECK(hip.size() == hip_before.size());
    for (size_t i = 0; i < hip_before.size() && i < hip.size(); ++i) CHECK(hip_before[i] == hip[i]);
    CHECK(!own_only_switch_on("STRATA_PF_FUSED_Q2N", true));
    CHECK(!own_only_switch_on("STRATA_PF_FUSED_SM75", true));
    CHECK(own_only_switch_on("STRATA_PF_FUSED_SM75", false));
    // names that are not in the list stay off (the other opt-ins of the kernels are not defaults)
    CHECK(!own_only_switch_on("STRATA_PF_FUSED_SM86", false));
    CHECK(!own_only_switch_on("STRATA_PF_FUSED_IQ2S", false));
    CHECK(!own_only_switch_on("", false));
    // no name twice
    for (size_t i = 0; i < cuda.size(); ++i)
        for (size_t j = i + 1; j < cuda.size(); ++j) CHECK(std::strcmp(cuda[i], cuda[j]) != 0);
    if (failed) {
        std::fprintf(stderr, "own_only_switches_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("own_only_switches_test: OK\n");
    return 0;
}
