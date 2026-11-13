// own_only_switches (include/strata/prefill/own_only_switches.hpp): the opt-in fused-kernel switches a build without MMQ turns on by default.  No GPU, no model.
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
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
    // the recorder: a fake environment; a name that was there before the initialiser ran is the user's and is not recorded, the others are set and recorded in list order
    {
        std::vector<std::pair<std::string, std::string>> env;   // name -> value
        const auto get = [&](const char* k) -> const char* {
            for (const auto& kv : env)
                if (kv.first == k) return kv.second.c_str();
            return nullptr;
        };
        const auto set = [&](const char* k) { env.emplace_back(k, "1"); };
        std::vector<std::string> rec;
        env.emplace_back("STRATA_PF_FUSED_Q8D", "0");   // the user named one of them (their way back)
        strata::prefill::own_only_apply_defaults(false, get, set, rec);
        CHECK(rec.size() == cuda.size() - 1);
        CHECK(!strata::prefill::own_only_set_by_build("STRATA_PF_FUSED_Q8D", rec));
        CHECK(std::string(get("STRATA_PF_FUSED_Q8D")) == "0");        // kept
        CHECK(strata::prefill::own_only_set_by_build("STRATA_PF_FUSED", rec));
        CHECK(std::string(get("STRATA_PF_FUSED")) == "1");
        CHECK(strata::prefill::own_only_set_by_build("STRATA_PF_FUSED_Q2N", rec));
        // "named by the user" (what the small-card rule reads): the build's own STRATA_PF_FUSED=1 is not, the same text from the user is, "" is neither
        CHECK(!strata::prefill::own_only_user_named("STRATA_PF_FUSED", get("STRATA_PF_FUSED"), rec));
        CHECK(strata::prefill::own_only_user_named("STRATA_PF_FUSED_Q8D", get("STRATA_PF_FUSED_Q8D"), rec));
        CHECK(!strata::prefill::own_only_user_named("STRATA_PF_FUSED", nullptr, rec));
        CHECK(!strata::prefill::own_only_user_named("STRATA_PF_FUSED_Q8D", "", rec));
        // a user who set STRATA_PF_FUSED itself (either value) before the initialiser: not recorded, so still the user's
        for (const char* v : {"1", "0"}) {
            env.clear();
            rec.clear();
            env.emplace_back("STRATA_PF_FUSED", v);
            strata::prefill::own_only_apply_defaults(false, get, set, rec);
            CHECK(!strata::prefill::own_only_set_by_build("STRATA_PF_FUSED", rec));
            CHECK(strata::prefill::own_only_user_named("STRATA_PF_FUSED", get("STRATA_PF_FUSED"), rec));
            CHECK(std::string(get("STRATA_PF_FUSED")) == v);
        }
        // nothing set: every name of the list is the build's; the HIP list is shorter
        env.clear();
        rec.clear();
        strata::prefill::own_only_apply_defaults(true, get, set, rec);
        CHECK(rec.size() == hip.size());
        CHECK(strata::prefill::own_only_set_by_build("STRATA_PF_FUSED", rec));
        CHECK(!strata::prefill::own_only_set_by_build("STRATA_PF_FUSED_SM75", rec));
        // the process-wide list is empty until the engine's initialiser (a binary that does not link prefill.cpp) runs
        CHECK(strata::prefill::own_only_recorded().empty());
        CHECK(!strata::prefill::own_only_user_named("STRATA_PF_FUSED", nullptr));
        CHECK(strata::prefill::own_only_user_named("STRATA_PF_FUSED", "1"));   // no record: the user's
    }
    if (failed) {
        std::fprintf(stderr, "own_only_switches_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("own_only_switches_test: OK\n");
    return 0;
}
