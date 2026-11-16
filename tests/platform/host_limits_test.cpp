// Windows HIP auto-chunk cap (#1630), the commit-limit verdict (#1607) and the commit with the page files' room to grow
// (#5742): no GPU, no Windows needed.
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "strata/platform/commit.hpp"
#include "strata/platform/host_limits.hpp"

using namespace strata::platform::host_limits;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // a bare auto on Windows HIP is capped below the 8192 that crashed; every other build keeps it
    CHECK(auto_chunk_cap(true, true) == 6144);
    CHECK(auto_chunk_cap(true, true) < 8192);
    CHECK(auto_chunk_cap(false, true) == 0);
    CHECK(auto_chunk_cap(true, false) == 0);    // auto:N / STRATA_PREFILL_AUTO_MAX: the operator's pick
    CHECK(auto_chunk_cap(false, false) == 0);
    // a chunk the user names stays, with a warning from 8192 up (7500 was stable)
    CHECK(warn_user_chunk(true, false, 8192));
    CHECK(warn_user_chunk(true, false, 16384));
    CHECK(!warn_user_chunk(true, false, 7500));
    CHECK(!warn_user_chunk(true, true, 8192));  // the bare auto is capped instead
    CHECK(!warn_user_chunk(false, false, 8192));
    // commit: 188 GiB limit with 90 GiB left and a 10 GiB plan: quiet; 95 GiB limit with 6 GiB left: warns
    const uint64_t G = 1ull << 30;
    CHECK(!commit_verdict(90 * G, 188 * G, 10 * G, 4 * G).warn);
    CHECK(!commit_verdict(14 * G, 96 * G, 10 * G, 4 * G).warn);       // exactly enough
    const CommitVerdict v = commit_verdict(6 * G, 96 * G, 10 * G, 4 * G);
    CHECK(v.warn);
    CHECK(v.text.find("1607") != std::string::npos);
    CHECK(!commit_verdict(0, 0, 10 * G, 4 * G).warn);                 // unknown limit (not Windows)
    // #5742: the commit left plus the room the page files may still grow by.  The reporter: 21.67 GiB left of a
    // 47.9 GiB limit with a page file at 8 of its 32 GB (24 GB of room), 32.25 GiB of RAM free
    {
        using strata::platform::commit_available;
        const uint64_t left = 21670ull * G / 1000, room = 24 * G, phys = 32250ull * G / 1000;
        CHECK(commit_available(left, 0) == left);                       // no room (STRATA_COMMIT_GROWTH=0, off Windows)
        CHECK(commit_available(left, room) == left + room);
        CHECK(std::min(phys, commit_available(left, room)) == phys);    // RAM, no longer the page file's size now, bounds it
        CHECK(std::min(phys, commit_available(left, 0)) == left);       // as before: the 21.67 GiB
        CHECK(commit_available(UINT64_MAX - 1, 5) == UINT64_MAX);       // saturates
        CHECK(commit_available(UINT64_MAX, UINT64_MAX) == UINT64_MAX);
        // the planned-allocation warning reads the same sum: 4 GiB left of 47.9 with 24 GiB of room and a 3.5 GiB plan
        CHECK(commit_verdict(4 * G, 48 * G, 3584ull << 20, 4 * G).warn);
        CHECK(!commit_verdict(commit_available(4 * G, room), commit_available(48 * G, room), 3584ull << 20, 4 * G).warn);
    }
#if defined(_WIN32)
    // this machine: the reading the engine now takes (it can only grow the commit, never shrink it)
    {
        MEMORYSTATUSEX s{};
        s.dwLength = sizeof s;
        CHECK(GlobalMemoryStatusEx(&s) != 0);
        const uint64_t grow = strata::platform::commit_growth((uint64_t) s.ullTotalPhys);
        CHECK(strata::platform::commit_available((uint64_t) s.ullAvailPageFile, grow) >= (uint64_t) s.ullAvailPageFile);
        std::printf("host_limits_test: this machine: %.1f GiB of commit left of %.1f, the page files may grow by %.1f GiB\n",
                    (double) s.ullAvailPageFile / G, (double) s.ullTotalPageFile / G, (double) grow / G);
    }
#endif
    if (failed == 0) std::printf("host_limits_test: ok\n");
    return failed ? 1 : 0;
}
