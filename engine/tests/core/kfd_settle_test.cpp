// the wait before READY on an integrated AMD GPU under Linux (include/strata/core/kfd_settle.hpp).  No GPU, no sleeping: the
// counter reader and the clock are faked; the folder reader is tried on a temporary folder.  Cases: file missing, unreadable, garbage,
// counter moving, counter quiet, cap reached.  (The device rule is checked in src/core/arch_defaults_test.cpp.)
#include "strata/core/kfd_settle.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace strata::core;
using namespace strata::core::kfd_settle;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

struct Fake {
    double t = 0.0;
    int sleeps = 0;
    std::function<Reading(int)> reading;   // the reading at poll number i (0 = the first, before any sleep)
    Result run(double quiet, double cap, double poll = 0.2) {
        return settle([&] { return reading(sleeps); }, [&] { return t; }, [&](double s) { t += s; ++sleeps; }, quiet, cap, poll);
    }
};
Reading ok(int64_t ms) { Reading r; r.kind = Reading::Ok; r.ms = ms; return r; }
Reading kind(Reading::Kind k) { Reading r; r.kind = k; return r; }
bool near(double a, double b, double tol) { return a > b - tol && a < b + tol; }

#if defined(__linux__)
void put(const std::string& path, const std::string& text) {
    if (std::FILE* f = std::fopen(path.c_str(), "w")) { std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); }
}
#endif
}  // namespace

int main() {
    // 1. the counter file is missing: nothing is waited, nothing is slept
    {
        Fake f;
        f.reading = [](int) { return kind(Reading::Missing); };
        const Result r = f.run(2.0, 10.0);
        CHECK(r.why == Result::NoCounter);
        CHECK(r.waited_s == 0.0);
        CHECK(f.sleeps == 0);
    }
    // 2. unreadable: the same at the first reading; a counter that stops being readable while waiting ends the wait
    {
        Fake f;
        f.reading = [](int) { return kind(Reading::Unreadable); };
        const Result r = f.run(2.0, 10.0);
        CHECK(r.why == Result::NoCounter && r.waited_s == 0.0 && f.sleeps == 0);
        Fake g;
        g.reading = [](int i) { return i < 3 ? ok(100 + i) : kind(Reading::Unreadable); };   // moving, then gone
        const Result s = g.run(2.0, 10.0);
        CHECK(s.why == Result::NoCounter);
        CHECK(near(s.waited_s, 0.6, 0.05));
    }
    // 3. garbage in the file
    {
        int64_t ms = -1;
        CHECK(!parse_evicted_ms("", ms));
        CHECK(!parse_evicted_ms("   \n", ms));
        CHECK(!parse_evicted_ms("abc", ms));
        CHECK(!parse_evicted_ms("12x", ms));
        CHECK(!parse_evicted_ms("-5", ms));
        CHECK(!parse_evicted_ms("+5", ms));
        CHECK(!parse_evicted_ms("1 2", ms));
        CHECK(!parse_evicted_ms("1.5", ms));
        CHECK(!parse_evicted_ms("1234567890123456789", ms));   // 19 digits
        CHECK(ms == -1);                                       // untouched on failure
        CHECK(parse_evicted_ms("0", ms) && ms == 0);
        CHECK(parse_evicted_ms("12\n", ms) && ms == 12);
        CHECK(parse_evicted_ms("  7 \t\n", ms) && ms == 7);
        CHECK(parse_evicted_ms("123456789012345678", ms) && ms == 123456789012345678LL);   // 18 digits
    }
    // 4. the counter moving: every poll for 1 s, then still: the wait ends two quiet seconds after the last move
    {
        Fake f;
        f.reading = [](int i) { return ok(i <= 5 ? 1000 + 10 * i : 1050); };   // moves at polls 1..5 (t = 0.2 .. 1.0)
        const Result r = f.run(2.0, 10.0);
        CHECK(r.why == Result::Quiet);
        CHECK(near(r.waited_s, 3.0, 0.25));    // last move at 1.0 s + 2.0 s quiet
        CHECK(r.ms == 1050);
    }
    // 5. the counter quiet from the start: the least wait is the quiet time
    {
        Fake f;
        f.reading = [](int) { return ok(4242); };
        const Result r = f.run(2.0, 10.0);
        CHECK(r.why == Result::Quiet);
        CHECK(near(r.waited_s, 2.0, 0.21));
        CHECK(r.ms == 4242);
        Fake g;
        g.reading = [](int) { return ok(0); };   // a process that was never evicted
        CHECK(g.run(2.0, 10.0).why == Result::Quiet);
    }
    // 6. the cap: a counter that never stops ends the wait at the cap
    {
        Fake f;
        f.reading = [](int i) { return ok(i * 5); };
        const Result r = f.run(2.0, 10.0);
        CHECK(r.why == Result::Cap);
        CHECK(r.waited_s >= 10.0 && r.waited_s < 10.3);
        // quiet is judged before the cap: quiet exactly at the cap counts as quiet
        Fake g;
        g.reading = [](int i) { return ok(i < 40 ? i : 40); };   // moves until 8.0 s, quiet from then: 10.0 s
        const Result q = g.run(2.0, 10.0);
        CHECK(q.why == Result::Quiet);
    }
#if defined(__linux__)
    // the folder reader: sum over stats_*, missing, unreadable
    {
        char tmpl[] = "/tmp/kfd_settle_test_XXXXXX";
        const char* dir = mkdtemp(tmpl);
        CHECK(dir != nullptr);
        if (dir != nullptr) {
            const std::string d = dir;
            CHECK(read_own_evicted_ms(d + "/no_such_folder").kind == Reading::Missing);
            CHECK(read_own_evicted_ms(d).kind == Reading::Missing);                          // an empty folder
            mkdir((d + "/stats_1").c_str(), 0700);
            CHECK(read_own_evicted_ms(d).kind == Reading::Missing);                          // a GPU folder without the file
            put(d + "/stats_1/evicted_ms", "12\n");
            Reading r = read_own_evicted_ms(d);
            CHECK(r.kind == Reading::Ok && r.ms == 12);
            mkdir((d + "/stats_2").c_str(), 0700);
            put(d + "/stats_2/evicted_ms", "30");
            r = read_own_evicted_ms(d);
            CHECK(r.kind == Reading::Ok && r.ms == 42);                                      // two GPUs: the sum
            put(d + "/stats_2/evicted_ms", "not a number\n");
            CHECK(read_own_evicted_ms(d).kind == Reading::Unreadable);
            put(d + "/stats_2/evicted_ms", "");
            CHECK(read_own_evicted_ms(d).kind == Reading::Unreadable);
            std::remove((d + "/stats_1/evicted_ms").c_str());
            std::remove((d + "/stats_2/evicted_ms").c_str());
            rmdir((d + "/stats_1").c_str());
            rmdir((d + "/stats_2").c_str());
            rmdir(d.c_str());
        }
        // this process's own folder: whatever the box has, the call returns (it is Missing on a box without KFD)
        (void) read_own_evicted_ms(own_proc_dir());
    }
#endif
    // the engine's call: with the flag down (the default) it returns at once
    {
        CHECK(!kfd_settle_flag().load());
        const auto t0 = std::chrono::steady_clock::now();
        settle_before_ready();
        CHECK(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5);
    }
    if (failed == 0) std::printf("kfd_settle_test: ok\n");
    return failed == 0 ? 0 : 1;
}
