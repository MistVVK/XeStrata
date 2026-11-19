// lm_walk_rule.hpp: the experts the layer-major prompt walk counts and streams (#5104, #5354, #5966, #6018, #6134).
// A 256-expert pack on an engine compiled for 512: the walk must not ask for experts 256.., and the resident-slot table has the MODEL's stride.  No GPU, no model.
#include <cstdio>
#include <vector>

#include "strata/prefill/lm_walk_rule.hpp"

using strata::prefill::lm_expert_count;
using strata::prefill::lm_streamed_experts;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the count: the model's, never above the compiled one
    CHECK(lm_expert_count(256, 512) == 256);   // GSQ-RCO Coder on the default (flash_next) geometry
    CHECK(lm_expert_count(512, 512) == 512);   // Flash-Next: as before
    CHECK(lm_expert_count(512, 256) == 256);   // never past what the tables hold
    CHECK(lm_expert_count(0, 512) == 0);

    // a 256-expert pack, 48 layers, nothing resident, nothing pinned (the small-card case of #5104): the prefetch lists 0..255 and not one id more
    const int32_t ne = lm_expert_count(256, 512);
    auto none = [](int64_t, int32_t) { return false; };
    for (int64_t l : {0, 1, 24, 47}) {
        const std::vector<int32_t> v = lm_streamed_experts(l, ne, nullptr, none);
        CHECK(v.size() == 256);
        CHECK(!v.empty() && v.front() == 0 && v.back() == 255);   // with the compiled count the list went to 511: "could not copy experts 256.. of layer 0"
    }

    // the resident-slot table is n_layers x 256: layer 1's entries sit at 256.., so layer 1 expert 5 is slot_of[1 * 256 + 5], not slot_of[1 * 512 + 5]
    std::vector<int32_t> host_res(48 * 256, -1);
    host_res[(size_t) 1 * 256 + 5] = 7;       // layer 1, expert 5 resident
    host_res[(size_t) 1 * 256 + 255] = 9;     // layer 1, the last expert resident
    host_res[(size_t) 0 * 256 + 5] = 3;       // layer 0, expert 5 resident
    {
        const std::vector<int32_t> v = lm_streamed_experts(1, ne, host_res.data(), none);
        CHECK(v.size() == 254);
        bool has5 = false, has255 = false, has6 = false;
        for (int32_t e : v) { has5 |= e == 5; has255 |= e == 255; has6 |= e == 6; }
        CHECK(!has5 && !has255 && has6);
    }
    {   // the last layer reads inside the table (a 512 stride read past its end from layer 24 on)
        const std::vector<int32_t> v = lm_streamed_experts(47, ne, host_res.data(), none);
        CHECK(v.size() == 256);
    }
    // a pinned expert is not streamed
    {
        auto pin3 = [](int64_t l, int32_t e) { return l == 2 && e == 3; };
        const std::vector<int32_t> v = lm_streamed_experts(2, ne, nullptr, pin3);
        CHECK(v.size() == 255);
        bool has3 = false;
        for (int32_t e : v) has3 |= e == 3;
        CHECK(!has3);
    }
    // the old behaviour (the compiled count as the loop bound) is the bug: it lists 512 and goes past the model's table
    CHECK(lm_streamed_experts(0, 512, nullptr, none).size() == 512);

    if (failed) { std::fprintf(stderr, "lm_walk_rule_test: %d failed\n", failed); return 1; }
    std::printf("lm_walk_rule_test: ok\n");
    return 0;
}
