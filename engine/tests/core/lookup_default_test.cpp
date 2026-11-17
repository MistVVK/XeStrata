// (asserts stay on in a Release build: the test is its own check)
#undef NDEBUG
#include "strata/core/lookup_default.hpp"
#include <cassert>
#include <iostream>

using strata::core::LookupDefaultIn;
using strata::core::default_lookup_max_t;

int main() {
    LookupDefaultIn cuda;   // the 8-row CUDA serve on one card, the case the owner decided
    cuda.narrow_window = cuda.serve = cuda.suffix_draft = cuda.spec_at_least_2 = true;
    assert(default_lookup_max_t(cuda) == 12);

    LookupDefaultIn metal = cuda;   // Metal also has 8-row windows, but no 12-row window: the option stays off
    metal.metal_build = true;
    assert(default_lookup_max_t(metal) == 0);

    LookupDefaultIn wide = cuda;    // HIP (16 rows) keeps its old windows
    wide.narrow_window = false;
    assert(default_lookup_max_t(wide) == 0);

    // every other condition turns it off on its own
    bool LookupDefaultIn::*off_when_true[] = {&LookupDefaultIn::env_off, &LookupDefaultIn::multi_gpu, &LookupDefaultIn::layer_split,
                                              &LookupDefaultIn::batch, &LookupDefaultIn::dflash2, &LookupDefaultIn::lookup_chain};
    for (auto m : off_when_true) {
        LookupDefaultIn x = cuda;
        x.*m = true;
        assert(default_lookup_max_t(x) == 0);
    }
    bool LookupDefaultIn::*off_when_false[] = {&LookupDefaultIn::narrow_window, &LookupDefaultIn::serve, &LookupDefaultIn::suffix_draft,
                                               &LookupDefaultIn::spec_at_least_2};
    for (auto m : off_when_false) {
        LookupDefaultIn x = cuda;
        x.*m = false;
        assert(default_lookup_max_t(x) == 0);
    }
    std::cout << "lookup_default_test: ok\n";
    return 0;
}
