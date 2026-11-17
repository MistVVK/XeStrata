// home_cpus_test: the widest CPU set a thread of the engine takes (include/strata/platform/home_cpus.hpp).  Without a home
// set a thread that widens gets every CPU the process may use (as before); with one it stays inside it, also when it
// started on a single CPU (the host's, inherited).  No GPU.
#include "strata/platform/home_cpus.hpp"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

#if defined(__linux__)
static std::vector<int> mine() {
    std::vector<int> out;
    cpu_set_t m;
    CPU_ZERO(&m);
    if (sched_getaffinity(0, sizeof m, &m) == 0)
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &m)) out.push_back(c);
    return out;
}

static void pin_one(int cpu) {
    cpu_set_t m;
    CPU_ZERO(&m);
    CPU_SET(cpu, &m);
    (void) sched_setaffinity(0, sizeof m, &m);
}
#endif

int main() {
#if defined(__linux__)
    const std::vector<int> start = mine();
    if (start.size() < 2) {
        std::printf("home_cpus_test: skipped (this process may run on %zu CPU)\n", start.size());
        return 0;
    }
    {   // no home: the old behaviour, every CPU the process may use
        const cpu_set_t h = strata::platform::home_mask();
        expect(CPU_COUNT(&h) == CPU_SETSIZE, "no home: the mask is every CPU number");
        std::vector<int> got;
        std::thread t([&] {
            pin_one(start.front());
            strata::platform::widen_to_home();
            got = mine();
        });
        t.join();
        expect(got == start, "no home: a pinned thread widens back to all the process's CPUs");
    }
    const std::vector<int> half(start.begin(), start.begin() + (long) (start.size() / 2));
    strata::platform::set_home_cpus({});   // empty: nothing changes
    {
        const cpu_set_t h = strata::platform::home_mask();
        expect(CPU_COUNT(&h) == CPU_SETSIZE, "empty home list: still every CPU number");
    }
    strata::platform::set_home_cpus(half);
    {
        std::vector<int> got;
        std::thread t([&] {
            pin_one(start.front());
            strata::platform::widen_to_home();
            got = mine();
        });
        t.join();
        expect(got == half, "home set: a pinned thread widens to the home CPUs only");
        std::vector<int> got2;
        std::thread u([&] {   // a thread that runs on every CPU of the process widens DOWN to the home set too
            strata::platform::widen_to_home();
            got2 = mine();
        });
        u.join();
        expect(got2 == half, "home set: a wide thread ends on the home CPUs");
    }
    expect(mine() == start, "the calling thread is not moved by set_home_cpus");
#else
    strata::platform::set_home_cpus({0});
    strata::platform::widen_to_home();
    std::printf("home_cpus_test: not Linux, nothing to check\n");
#endif
    if (g_fail != 0) {
        std::fprintf(stderr, "home_cpus_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("home_cpus_test: all checks passed\n");
    return 0;
}
