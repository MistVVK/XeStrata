// The first-request VRAM note (item 17, engine 0.1.63): said only when the free memory fell since the start-up line or is under 512 MiB.
#undef NDEBUG
#include "strata/core/vram_request_notice.hpp"
#include <cassert>
#include <iostream>

using strata::core::vram_request_note_due;
using strata::core::vram_request_reserve_mib;

int main() {
    constexpr int64_t kWin = 1024, kOther = 512;

    // the case of the report: 442 MiB free at the first request, the same at the start-up line (Windows safe 1024): it is under the
    // 512 MiB floor, so it is still said (a card with that little room can page)
    assert(vram_request_note_due(442, 442, kWin));
    // 700 MiB free at start-up and at the request: above the floor, not fallen: silent (before 0.1.63 this warned at every start)
    assert(!vram_request_note_due(700, 700, kWin));
    assert(!vram_request_note_due(1000, 1000, kWin));
    // a few MiB of movement is not a fall
    assert(!vram_request_note_due(690, 700, kWin));
    assert(!vram_request_note_due(637, 700, kWin));
    // 64 MiB or more lower than at start-up: said
    assert(vram_request_note_due(636, 700, kWin));
    assert(vram_request_note_due(300 + 512, 1000, kWin));
    // higher than at start-up: silent
    assert(!vram_request_note_due(900, 700, kWin));
    // at or above the safe figure: never said, even after a fall
    assert(!vram_request_note_due(1024, 3000, kWin));
    assert(!vram_request_note_due(5000, 9000, kWin));
    // under 512 MiB: said whether or not it fell
    assert(vram_request_note_due(511, 400, kWin));
    assert(vram_request_note_due(100, 100, kWin));
    assert(vram_request_note_due(511, 511, kOther));   // other systems: safe figure 512, 511 is under it and under the floor
    assert(!vram_request_note_due(512, 512, kOther));
    // start-up figure unknown: the old rule (below the safe figure: said)
    assert(vram_request_note_due(900, -1, kWin));
    assert(!vram_request_note_due(1100, -1, kWin));

    // the suggested reserve: the missing MiB to the safe figure on top of the current reserve, rounded up to 64
    assert(vram_request_reserve_mib(0, 442, kWin) == 640);       // 582 -> 640
    assert(vram_request_reserve_mib(700, 442, kWin) == 1344);    // the example of the approved wording
    assert(vram_request_reserve_mib(1344, 1024, kWin) == 1344);

    std::cout << "vram_request_notice_test: ok\n";
    return 0;
}
