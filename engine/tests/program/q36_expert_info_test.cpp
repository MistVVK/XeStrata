// #5012: the qwen35moe engine's INFO line for the Monitor's "Expert cache: <expert_slots> / <experts>" (strata/qwen35/expert_info.hpp).
// Before #5012 that engine printed no expert figures at all, and no engine printed the total (`experts`): the Monitor showed nothing,
// or "N / 0".  No GPU, no model: fake layers with the two fields place_experts leaves (n_slots, stride).
#include "strata/qwen35/expert_info.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
struct FakeLayer { int n_slots; size_t stride; };
}  // namespace

int main() {
    using strata::qwen35::expert_info;
    using strata::qwen35::expert_info_line;
    // Qwen3.6-35B-A3B: 40 layers of 256 routed experts; an expert of the Q8_0 set is ~3.2 MB
    const size_t stride = 3342336;
    std::vector<FakeLayer> all(40, FakeLayer{256, stride});
    auto a = expert_info(all, 256);
    check(a.slots == 10240 && a.experts == 10240, "every expert fits: slots = experts = 40 x 256");
    check(a.cache_mib == (long long) ((10240ull * stride) >> 20), "their VRAM in MiB");
    check(expert_info_line(a) == "INFO expert_slots=10240 experts=10240 expert_cache_mib=" + std::to_string(a.cache_mib) + "\n",
          "the INFO line: expert_slots, experts (the total) and expert_cache_mib");
    // hybrid: 64 of 256 per layer in VRAM (the slot count), the rest on the CPU pool
    std::vector<FakeLayer> hyb(40, FakeLayer{64, stride});
    auto h = expert_info(hyb, 256);
    check(h.slots == 2560 && h.experts == 10240, "hybrid: 2560 of 10240 resident, the total unchanged");
    check(expert_info_line(h).find(" experts=10240 ") != std::string::npos, "the total is in the line, so the Monitor shows a share");
    // the line carries no vram_elastic key (the server takes that key to mean the VRAM command exists; this engine has none)
    check(expert_info_line(h).find("vram_elastic") == std::string::npos, "no vram_elastic key");
    check(expert_info_line(h).rfind("INFO ", 0) == 0 && expert_info_line(h).back() == '\n', "an INFO line, one line");
    // no layers: nothing to report (the engine prints the line only when experts > 0)
    check(expert_info(std::vector<FakeLayer>{}, 256).experts == 0, "no layers: experts 0");
    std::printf("q36_expert_info_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
