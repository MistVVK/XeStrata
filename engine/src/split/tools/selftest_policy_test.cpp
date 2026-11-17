// The kernel self-test rules (src/hip/selftest_policy.hpp): gfx1201 (run) goes without a word and without the test; any other architecture prints one experimental line with the
// report link, runs the test, and says so when it passed; a failed (or forced-failed) test stops the engine with one plain line that the server turns into the layer split. CPU only.
#include <cstdio>
#include <string>

#include "../src/hip/selftest_policy.hpp"

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main() {
    using namespace fnx;
    int calls = 0;
    auto pass = [&](std::string& covered) { ++calls; covered = "card 0: wmma, expert tile against a scalar reference"; return std::string(); };
    auto fail = [&](std::string& covered) { ++calls; covered.clear(); return std::string("card 1: wmma A row, B col (Q K^T): 37 of 512 values wrong"); };

    // the tested architecture: nothing, no test, whatever the name carries after the colon
    SelftestDecision d = selftest_decide("gfx1201:sramecc-:xnack-", nullptr, nullptr, pass);
    CHECK(d.ok && !d.run && d.lines.empty() && calls == 0);
    // ... unless FNX_SELFTEST=1 asks for it (no experimental line there)
    d = selftest_decide("gfx1201:sramecc-:xnack-", "1", nullptr, pass);
    CHECK(d.ok && d.run && calls == 1 && d.lines.size() == 1 && d.lines[0].rfind("The split engine's self-test passed on gfx1201 (", 0) == 0);
    // an untested architecture: the experimental line with the link, then the test, then the pass line
    calls = 0;
    d = selftest_decide("gfx1100", nullptr, nullptr, pass);
    CHECK(d.ok && d.run && calls == 1 && d.lines.size() == 2);
    CHECK(d.lines[0] == std::string("The split engine is experimental on this card (gfx1100): Strata has not run it on this architecture. Please report whether it works: ") + SPLIT_REPORT_URL);
    CHECK(d.lines[1] == "The split engine's self-test passed on gfx1100 (card 0: wmma, expert tile against a scalar reference).");
    for (const std::string& l : d.lines) CHECK(l.find('\n') == std::string::npos);
    // a failure: not ok, the second line says why in one plain sentence and what happens next
    calls = 0;
    d = selftest_decide("gfx1103", nullptr, nullptr, fail);
    CHECK(!d.ok && d.run && calls == 1 && d.lines.size() == 2);
    CHECK(d.lines[1] == "The split engine's self-test failed on gfx1103 (card 1: wmma A row, B col (Q K^T): 37 of 512 values wrong), so it does not run on this card: the normal layer split starts instead.");
    // the forced failure (the tester hook) does not even call the test, on any architecture
    calls = 0;
    d = selftest_decide("gfx1201", "1", "1", pass);
    CHECK(!d.ok && calls == 0 && d.lines.size() == 1 && d.lines[0].find("forced failure (FNX_SELFTEST_FAIL=1)") != std::string::npos);
    d = selftest_decide("gfx1200", nullptr, "1", pass);
    CHECK(!d.ok && calls == 0 && d.lines.size() == 2);
    // FNX_SELFTEST_FAIL on a tested architecture without FNX_SELFTEST=1 changes nothing
    d = selftest_decide("gfx1201", nullptr, "1", pass);
    CHECK(d.ok && !d.run && d.lines.empty());
    std::printf(bad ? "selftest_policy_test: %d FAILED\n" : "selftest_policy_test: ok\n", bad);
    return bad ? 1 : 0;
}
