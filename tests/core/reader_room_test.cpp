// The picture reader's room in the resident RAM mode (strata/core/reader_room.hpp): only an engine that serves pictures, was asked for
// the shrinking cache and runs the resident RAM mode keeps the reader's 2,048 MiB in its reserve; every other start keeps its reserve.
#include "strata/core/reader_room.hpp"

#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    CHECK(kReaderRoomMib == 2048);
    // the case: setup's 700 MiB, pictures on, the shrinking cache asked for, the resident RAM mode -> the reader's room is kept
    CHECK(reader_room_reserve_mib(700, true, true, true) == 2048);
    CHECK(reader_room_reserve_mib(0, true, true, true) == 2048);
    // a larger reserve (the person's, the app's, Windows' own) is never lowered
    CHECK(reader_room_reserve_mib(2048, true, true, true) == 2048);
    CHECK(reader_room_reserve_mib(3000, true, true, true) == 3000);
    // any one of the three missing: the reserve as asked (the arena mode shrinks; no pictures; no on-demand reader)
    for (int v = 0; v < 2; ++v)
        for (int e = 0; e < 2; ++e)
            for (int r = 0; r < 2; ++r)
                if (!(v && e && r)) CHECK(reader_room_reserve_mib(700, v, e, r) == 700);
    if (failed) {
        std::fprintf(stderr, "reader_room_test: %d failed\n", failed);
        return 1;
    }
    std::printf("reader_room_test: ok\n");
    return 0;
}
