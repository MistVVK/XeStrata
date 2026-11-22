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

    // #533 range overload: under a layer split each card trims only its own layers [lo, hi); slot numbers overlap across cards.
    std::vector<int32_t> t = {5, 0, 7, kNotResident, 9, 4, 6, 1};   // n_expert=2 -> layer=i/2: L0{0,1} L1{2,3} L2{4,5} L3{6,7}
    std::vector<std::pair<int32_t, int32_t>> er;
    CHECK(evict_slots_past(t, 3, 1, 3, 2, er) == 3);   // card owns layers [1,3): in-range idx{2,4,5} sit past live=3 -> taken (idx3 not-resident stays)
    CHECK(er.size() == 3 && er[0] == std::make_pair(2, 7) && er[1] == std::make_pair(4, 9) && er[2] == std::make_pair(5, 4));   // table order within the range
    CHECK(t[0] == 5 && t[6] == 6);                     // another card's past-slot experts (idx0 L0, idx6 L3) are left alone
    CHECK(t[1] == 0 && t[2] == kNotResident && t[4] == kNotResident && t[5] == kNotResident && t[7] == 1);   // in-range taken; below-live and not-resident kept

    std::vector<int32_t> g = {1, 9, 2, 8};             // n_expert=1 -> layer=i
    std::vector<std::pair<int32_t, int32_t>> eb;
    CHECK(evict_slots_past(g, 5, 0, 2, 1, eb) == 1);   // [0,2): only idx1 (9>=live) taken; hi is exclusive so idx3 stays even though >=live
    CHECK(eb.size() == 1 && eb[0] == std::make_pair(1, 9));
    CHECK(g[1] == kNotResident && g[3] == 8);          // boundary: the index at hi (and past it) is untouched

    std::vector<std::pair<int32_t, int32_t>> ef;       // the whole range reproduces the one-GPU behaviour
    CHECK(evict_slots_past(g, 5, 0, INT64_MAX, 1, ef) == 1 && ef.size() == 1);   // takes what the bounded call left: idx3
    if (failed) {
        std::fprintf(stderr, "residency_trim_test: %d failed\n", failed);
        return 1;
    }
    std::puts("residency_trim_test: the entries past the mapped slots are taken, in table order");
    return 0;
}
