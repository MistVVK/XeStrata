#pragma once
// The links between the cards tried before the engine uses them (src/peer_probe.cu), and what is said about them. For
// every directed pair, a buffer on the receiving card is written from the other, by a kernel's stores as the ring
// writes (timed) and once by the copy engines as the prompt path's exchange copies, and watched for word by word with
// the receiving card's own reads; what that does not prove: docs/config.md, Boot requirement. A pair whose words do
// not all arrive throws, naming it; when a card is left running or faulted, the message goes to stderr and the process
// ends, as nothing on that card can be cleaned up safely. A slow pair is reported, not refused. Run after the static
// checks (Model::Model, check_cards) and peer access, before the window graphs. The report and the failure text below
// need no CUDA, so tools/card_check_test.cpp tests them on the CPU
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "card_check.hpp"

namespace fnx {

// PEER_PROBE_BYTES stored from one card into another, the median time until the receiver saw every word. Under
// PEER_SLOW_GBS a link works but far below what one should: the rates measured on prod's V100s and a tester's RTX
// 3090s are in docs/config.md (Boot requirement), several GB/s on both. Slow is said, not refused:
// the tokens are the same, only slower (the ring's sums cross each card's link to the next every layer, the prompt
// path's attention exchange every pair)
constexpr size_t PEER_PROBE_BYTES = 512 << 10;
constexpr double PEER_SLOW_GBS = 1.0;
struct PeerRate {
    int from, to;
    double us;   // median of a kernel's stores of PEER_PROBE_BYTES
};
inline double peer_gbs(double us) { return PEER_PROBE_BYTES / (us * 1e3); }

// devs: the cards' CUDA devices in card order, peer access enabled between every pair. A rate for each of the
// n (n - 1) directed pairs; throws, or ends the process, as above
std::vector<PeerRate> probe_peer_stores(const std::vector<int>& devs);

// one line for the whole probe, then a warning for each slow pair, each line ready for stderr. ncard: the cards probed,
// so a run that left a pair out is refused rather than reported as healthy
inline std::vector<std::string> peer_report(const std::vector<PeerRate>& r, int ncard) {
    if (ncard < 2 || r.size() != (size_t) ncard * (size_t) (ncard - 1))
        throw std::logic_error("peer_report: " + std::to_string(r.size()) + " pairs probed for " + std::to_string(ncard) + " cards");
    // n (n - 1) entries, each a directed pair of distinct cards in range and none twice, are every pair once
    std::vector<bool> seen((size_t) ncard * (size_t) ncard);
    for (const PeerRate& p : r) {
        if (p.from < 0 || p.to < 0 || p.from >= ncard || p.to >= ncard || p.from == p.to ||
            seen[(size_t) p.from * (size_t) ncard + (size_t) p.to])
            throw std::logic_error("peer_report: card " + std::to_string(p.from) + " -> " + std::to_string(p.to) + " is not a new pair of " +
                                   std::to_string(ncard) + " cards");
        seen[(size_t) p.from * (size_t) ncard + (size_t) p.to] = true;
    }
    double lo = 1e300, hi = 0;
    for (const PeerRate& p : r) {
        if (!(p.us > 0)) throw std::logic_error("peer_report: card " + std::to_string(p.from) + " -> " + std::to_string(p.to) + " took no time");
        lo = std::min(lo, peer_gbs(p.us));
        hi = std::max(hi, peer_gbs(p.us));
    }
    char line[512];
    // a line cut short by the buffer would say less than it should: refuse it rather than print it
    const auto fits = [&](int n) {
        if (n < 0 || n >= (int) sizeof line) throw std::logic_error("peer_report: a line past " + std::to_string(sizeof line) + " bytes");
    };
    fits(std::snprintf(line, sizeof line, "fnx: kernel stores between the cards: %zu pairs, %.1f to %.1f GB/s (%zu KiB a pair)", r.size(), lo,
                       hi, PEER_PROBE_BYTES >> 10));
    std::vector<std::string> out{line};
    for (const PeerRate& p : r)
        if (peer_gbs(p.us) < PEER_SLOW_GBS) {
            fits(std::snprintf(line, sizeof line,
                               "fnx: warning: card %d -> card %d stores at %.2f GB/s (%zu KiB in %.0f us), under %.0f GB/s: the "
                               "engine runs, but every sum that crosses this link waits on it. Check the link "
                               "(nvidia-smi topo -p2p r; lspci -vv, LnkSta)",
                               p.from, p.to, peer_gbs(p.us), PEER_PROBE_BYTES >> 10, p.us, PEER_SLOW_GBS));
            out.push_back(line);
        }
    return out;
}

// a pair whose words did not all arrive within the timeout though CUDA allows peer access
struct PeerFailure {
    int from, to;
    int ncard;                       // the cards probed, for the remedy
    bool copy;                       // the copy engines' run, else a kernel's stores
    unsigned missing;                // words not seen
    size_t words;                    // words written
    unsigned long long timeout_ms;   // how long the receiver watched
};
// lost, misplaced, or on a link slower than words in timeout_ms, which the probe cannot tell from lost
inline std::string peer_probe_failure(const PeerFailure& f) {
    if (f.timeout_ms == 0 || f.missing == 0 || f.missing > f.words)
        throw std::logic_error("peer_probe_failure: nothing to report (" + std::to_string(f.missing) + " of " + std::to_string(f.words) +
                               " words missing within " + std::to_string(f.timeout_ms) + " ms)");
    const double floor_mbs = (double) f.words * 8 / ((double) f.timeout_ms * 1e3);
    char rate[32];
    std::snprintf(rate, sizeof rate, "%.1f", floor_mbs);
    return "card " + std::to_string(f.from) + " -> card " + std::to_string(f.to) + ": " + std::to_string(f.missing) + " of " +
           std::to_string(f.words) + " words " + (f.copy ? "copied by the copy engines" : "stored by a kernel") + " from card " +
           std::to_string(f.from) + " were not seen on card " + std::to_string(f.to) + " within " + std::to_string(f.timeout_ms) +
           " ms, though CUDA allows peer access between them: lost, misplaced, or on a link under " + rate +
           " MB/s; the engine would hang on it. Check BAR1 (nvidia-smi -q, BAR1 Memory), the P2P driver if these are "
           "GeForce cards, and dmesg (either card can be the one at fault), or " +
           leave_card_out("card " + std::to_string(f.from) + " or card " + std::to_string(f.to), f.ncard);
}

}  // namespace fnx
