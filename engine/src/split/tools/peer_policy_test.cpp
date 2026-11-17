// The peer-push decision (src/hip/peer_policy.hpp): FNX_PEER=0 forces the host exchange, a failing probe leaves the host exchange for the whole run with one plain
// line, a forced probe failure (FNX_PEER_PROBE_FAIL=1) does the same, and nothing runs unless the policy says so. CPU only.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/hip/peer_policy.hpp"

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main() {
    using namespace fnx;
    int calls = 0;
    auto pass = [&](std::string& why) { ++calls; why = "all pairs ok"; return true; };
    auto fail = [&](std::string& why) { ++calls; why = "card 0 -> card 2: 3 of 1048576 words arrived wrong"; return false; };
    unsetenv("FNX_PEER_PROBE_FAIL");

    // FNX_PEER=0: the host exchange, no probe, no line, whatever the default
    calls = 0;
    PeerDecision d = peer_decide("0", true, pass);
    CHECK(!d.use && d.line.empty() && calls == 0);
    // unset and the default off: nothing, no probe
    d = peer_decide(nullptr, false, pass);
    CHECK(!d.use && d.line.empty() && calls == 0);
    // unset and the default on: the probe runs and decides
    d = peer_decide(nullptr, true, pass);
    CHECK(d.use && calls == 1 && d.line.find("Peer push between the cards is on") == 0);
    // FNX_PEER=1: the probe runs even if the default is off
    calls = 0;
    d = peer_decide("1", false, pass);
    CHECK(d.use && calls == 1);
    // a failing probe: the host exchange for the whole run and one plain line naming the reason
    calls = 0;
    d = peer_decide("1", false, fail);
    CHECK(!d.use && calls == 1);
    CHECK(d.line == "The cards cannot push into each other's memory here (card 0 -> card 2: 3 of 1048576 words arrived wrong): using the host exchange for this run.");
    CHECK(d.line.find('\n') == std::string::npos);
    // the same through the default
    d = peer_decide(nullptr, true, fail);
    CHECK(!d.use);
    // a forced probe failure (the tester hook): the probe is not even called, the line says why
    setenv("FNX_PEER_PROBE_FAIL", "1", 1);
    calls = 0;
    d = peer_decide("1", false, pass);
    CHECK(!d.use && calls == 0 && d.line.find("forced failure (FNX_PEER_PROBE_FAIL=1)") != std::string::npos && d.line.find("using the host exchange for this run.") != std::string::npos);
    d = peer_decide(nullptr, true, pass);
    CHECK(!d.use && calls == 0);
    // but FNX_PEER=0 stays silent and off
    d = peer_decide("0", true, pass);
    CHECK(!d.use && d.line.empty());
    unsetenv("FNX_PEER_PROBE_FAIL");
    std::printf(bad ? "peer_policy_test: %d FAILED\n" : "peer_policy_test: ok\n", bad);
    return bad ? 1 : 0;
}
