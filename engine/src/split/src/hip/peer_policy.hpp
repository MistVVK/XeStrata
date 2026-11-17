// Whether the cards exchange by pushing into each other's memory (AMD, hip/peer.h) or through host memory, decided once at engine start. CPU only, so
// tools/peer_policy_test.cpp holds it to its rules without a card. The rules (owner, 2026-11-02: "Peer access on R9700 is OK if faster, with fallback"):
//   FNX_PEER=0          the host exchange, whatever else (testers; no probe, no line)
//   FNX_PEER=1          try the peer push: probe every directed pair; use it only if every probe passes
//   FNX_PEER unset      the same as 1 if the build's default says the push wins (PEER_DEFAULT_ON, set from the screens), else the host exchange
//   any probe failure   the host exchange for the WHOLE run and one plain line saying why (never a mix of the two)
//   FNX_PEER_PROBE_FAIL=1  makes the probe fail on purpose (tests of the fallback)
#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace fnx {

// The default when FNX_PEER is unset: off until the screens (RESULTS-HIP.md) say the peer push beats the host exchange on the 8-user round and the 103K prompt.
constexpr bool PEER_DEFAULT_ON = true;

struct PeerDecision {
    bool use = false;
    std::string line;   // one plain line for the log ("" when nothing needs saying)
};

// probe(why): true if every pair passed; false with the reason in why
inline PeerDecision peer_decide(const char* env, bool default_on, const std::function<bool(std::string&)>& probe) {
    PeerDecision d;
    const bool forced_off = env && std::strcmp(env, "0") == 0;
    const bool wanted = env ? std::strcmp(env, "0") != 0 : default_on;
    if (forced_off || !wanted) return d;
    std::string why;
    const char* fail = std::getenv("FNX_PEER_PROBE_FAIL");
    const bool ok = (fail && std::strcmp(fail, "1") == 0) ? (why = "forced failure (FNX_PEER_PROBE_FAIL=1)", false) : probe(why);
    if (!ok) {
        d.line = "The cards cannot push into each other's memory here (" + why + "): using the host exchange for this run.";
        return d;
    }
    d.use = true;
    d.line = "Peer push between the cards is on (" + why + ").";
    return d;
}

}  // namespace fnx
