// The residency entries past what the segmented expert cache still maps (strata/core/residency_trim.hpp): after a shrink, also one
// the driver refused part way, every entry at or past the live slots becomes not resident and is listed for the next grow.  No GPU.
#include "strata/core/residency_trim.hpp"

#include <cstdio>
#include <utility>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    // 8 entries: slots 0..5, two not resident; 4 slots stay mapped
    std::vector<int32_t> res = {0, 5, kNotResident, 3, 4, kNotResident, 1, 2};
    std::vector<std::pair<int32_t, int32_t>> ev;
    CHECK(evict_slots_past(res, 4, ev) == 2);
    CHECK((res == std::vector<int32_t>{0, kNotResident, kNotResident, 3, kNotResident, kNotResident, 1, 2}));
    CHECK(ev.size() == 2 && ev[0] == std::make_pair(1, 5) && ev[1] == std::make_pair(4, 4));
    // a second shrink appends after what an earlier one listed, in table order
    CHECK(evict_slots_past(res, 2, ev) == 2);
    CHECK(ev.size() == 4 && ev[2] == std::make_pair(3, 3) && ev[3] == std::make_pair(7, 2));
    CHECK(res[0] == 0 && res[6] == 1 && res[3] == kNotResident && res[7] == kNotResident);
    // nothing past the live slots (a refusal before anything was given back): nothing changes
    const std::vector<int32_t> before = res;
    CHECK(evict_slots_past(res, 2, ev) == 0 && res == before && ev.size() == 4);
    // no slot left: every resident entry goes
    CHECK(evict_slots_past(res, 0, ev) == 2 && ev.size() == 6);
    for (const int32_t s : res) CHECK(s == kNotResident);
    if (failed) {
        std::fprintf(stderr, "residency_trim_test: %d failed\n", failed);
        return 1;
    }
    std::puts("residency_trim_test: the entries past the mapped slots are taken, in table order");
    return 0;
}
