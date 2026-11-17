// The prompt path's copy-engine ring plan (src/pf_ring_plan.hpp), on the CPU alone:
//   1. two and three cards: the plan is exactly the step sequence pf_ring_run (pf_common.cuh) runs for them, written
//      out below, for every card;
//   2. every count card_split.hpp allows: the n cards' plans run against each other for several sums in a row, over
//      the two buffer sets and the sequence-numbered flags the streams use, in many random interleavings: every sum's
//      parts hold every card's addend once, nothing is read on a card before the flag that covers it for that sum, no
//      slot is written twice in a sum or over a previous sum's contents before they were read, no deadlock, and no
//      more flags than PF_RING_STEPS holds.
// Exit 0 when all pass; each failure printed.
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/model.hpp"   // card_split.hpp: the supported counts
#include "../src/pf_ring_plan.hpp"

using fnx::PfRingOp;
using K = PfRingOp::Kind;

static int fails = 0;
static void fail(const std::string& m) {
    std::printf("FAIL: %s\n", m.c_str());
    ++fails;
}

static PfRingOp op(K k, int part = -1, int hop = -1, int step = -1, bool fin = false) {
    PfRingOp o{k, part, hop, step};
    o.fin = fin;
    return o;
}

// the ring as pf_common.cuh's pf_ring_run was written to run it for three cards (a = g, b = g - 1, c = g + 1) and two
// (c = b = the other)
static std::vector<PfRingOp> written(int n, int g) {
    const int a = g, b = (g + n - 1) % n, c = (g + 1) % n;
    std::vector<PfRingOp> v = {op(K::PACK, a), op(K::SEND, a, 0), op(K::SIGNAL, -1, -1, 0)};
    if (n == 2) {
        for (const PfRingOp& o : {op(K::WAIT, -1, -1, 0), op(K::ADD, c, 0, -1, true), op(K::PASS, c), op(K::SIGNAL, -1, -1, 1),
                                  op(K::WAIT, -1, -1, 1), op(K::OUT)})
            v.push_back(o);
        return v;
    }
    for (const PfRingOp& o : {op(K::WAIT, -1, -1, 0), op(K::ADD, b, 0), op(K::SEND, b, 1), op(K::SIGNAL, -1, -1, 1),
                              op(K::WAIT, -1, -1, 1), op(K::ADD, c, 1, -1, true), op(K::PASS, c), op(K::SIGNAL, -1, -1, 2),
                              op(K::WAIT, -1, -1, 2), op(K::PASS, a), op(K::SIGNAL, -1, -1, 3), op(K::WAIT, -1, -1, 3), op(K::OUT)})
        v.push_back(o);
    return v;
}

// a buffer's contents in the simulation: the sum it belongs to (its sequence number), which part, whose addends (bit
// per card), the flag that publishes it on the card holding it (-1: written there by the card itself, -2: copied in,
// its flag not yet written), and whether this card is through reading it (a later sum may then write over it)
struct Slot {
    bool full = false;
    unsigned seq = 0;
    int part = -1;
    unsigned addends = 0;
    int step = -1;
    bool consumed = false;
};

// `sums` sums one after another on every card, as pf_ring runs them: sum seq (from 1, after pf_seq_reset's 0) uses buffer
// set seq & 1, a signal writes seq into the flag (cuStreamWriteValue32) and a wait passes once the flag is >= seq
// (CU_STREAM_WAIT_VALUE_GEQ). Each card's steps are one stream (in order); the cards interleave as the random order
// picks, so a card may run sums ahead of a slower one as far as the flags let it. Returns false after a failure.
static bool simulate(int n, int sums, unsigned seed) {
    const std::string tag = std::to_string(n) + " cards, order " + std::to_string(seed) + ": ";
    const unsigned all = (1u << n) - 1;
    std::vector<std::vector<PfRingOp>> plan(n);
    for (int g = 0; g < n; ++g) plan[g] = fnx::pf_ring_plan(n, g);
    const int steps = fnx::pf_ring_steps(n);
    struct Copied {
        bool ag;
        int i;
    };
    struct Card {
        size_t pc = 0;
        unsigned seq = 1;
        std::vector<unsigned> flag, waited;   // [set][step]: the flag's value; the sum that last waited on it
        std::vector<Slot> in, ag;             // [set][hop], [set][part]
        Slot send;
        std::vector<Copied> pending;          // copied into the next card since this card's last signal (this sum's set)
    };
    std::vector<Card> cd(n);
    for (Card& c : cd) {
        c.flag.assign(2 * steps, 0);
        c.waited.assign(2 * steps, 0);
        c.in.assign(2 * (n - 1), {});
        c.ag.assign(2 * n, {});
    }
    std::mt19937 rng(seed);
    auto readable = [&](const Card& c, const Slot& s, int set) {
        return s.full && s.seq == c.seq && (s.step == -1 || (s.step >= 0 && c.waited[(size_t) (set * steps + s.step)] == c.seq));
    };
    // a write into a slot: the sum before on this set must have been read out of it, and this sum not yet written
    auto writable = [&](const Slot& s, unsigned seq, const std::string& at, const char* what) {
        if (s.full && s.seq == seq) { fail(at + what + " written twice in one sum"); return false; }
        if (s.full && !s.consumed) {
            fail(at + what + " of sum " + std::to_string(s.seq) + " written over by sum " + std::to_string(seq) + " before it was read");
            return false;
        }
        return true;
    };
    for (;;) {
        std::vector<int> ready;
        bool left = false;
        for (int g = 0; g < n; ++g) {
            const Card& c = cd[g];
            if (c.seq > (unsigned) sums) continue;
            left = true;
            const PfRingOp& o = plan[g][c.pc];
            if (o.k == K::WAIT) {
                if (o.step < 0 || o.step >= steps) { fail(tag + "a flag past PF_RING_STEPS"); return false; }
                if (c.flag[(size_t) ((c.seq & 1) * steps + o.step)] < c.seq) continue;
            }
            ready.push_back(g);
        }
        if (!left) break;
        if (ready.empty()) {
            for (int g = 0; g < n; ++g)
                if (cd[g].seq <= (unsigned) sums)
                    fail(tag + "card " + std::to_string(g) + " stuck in sum " + std::to_string(cd[g].seq) + " at op " + std::to_string(cd[g].pc));
            return false;
        }
        const int g = ready[rng() % ready.size()];
        Card& c = cd[g];
        Card& nx = cd[(g + 1) % n];
        const int set = (int) (c.seq & 1u);
        const PfRingOp& o = plan[g][c.pc];
        const std::string at = tag + "card " + std::to_string(g) + " sum " + std::to_string(c.seq) + " op " + std::to_string(c.pc) + ": ";
        if ((o.k == K::ADD || o.k == K::SEND) && (o.hop < 0 || o.hop >= n - 1)) { fail(at + "a receive slot past n - 1 hops"); return false; }
        auto in = [&](Card& k, int hop) -> Slot& { return k.in[(size_t) (set * (n - 1) + hop)]; };
        auto ag = [&](Card& k, int part) -> Slot& { return k.ag[(size_t) (set * n + part)]; };
        if (o.k == K::WAIT) {
            c.waited[(size_t) (set * steps + o.step)] = c.seq;
        } else if (o.k == K::PACK) {
            c.send = {true, c.seq, o.part, 1u << g, -1, false};
        } else if (o.k == K::ADD) {
            Slot& r = in(c, o.hop);
            if (!readable(c, r, set)) { fail(at + "adds a receive slot not yet covered by a flag it waited on"); return false; }
            if (r.part != o.part) fail(at + "the receive slot holds another part");
            if (r.addends & (1u << g)) fail(at + "adds its own addend twice");
            r.consumed = true;
            const Slot s{true, c.seq, o.part, r.addends | (1u << g), -1, false};
            if (o.fin) {
                if (s.addends != all) fail(at + "finishes a part without every card's addend");
                if (!writable(ag(c, o.part), c.seq, at, "a gathered part")) return false;
                ag(c, o.part) = s;
            } else {
                c.send = s;
            }
        } else if (o.k == K::SEND) {
            if (!c.send.full || c.send.seq != c.seq || c.send.part != o.part) { fail(at + "sends what the send buffer does not hold"); return false; }
            if (!writable(in(nx, o.hop), c.seq, at, "the next card's receive slot")) return false;
            Slot s = c.send;
            s.step = -2;
            in(nx, o.hop) = s;
            c.pending.push_back({false, o.hop});
        } else if (o.k == K::PASS) {
            const Slot& f = ag(c, o.part);
            if (!readable(c, f, set)) { fail(at + "passes a part not here yet"); return false; }
            if (f.addends != all) fail(at + "passes an unfinished part");
            if (!writable(ag(nx, o.part), c.seq, at, "the next card's gathered part")) return false;
            Slot s = f;
            s.step = -2;
            s.consumed = false;
            ag(nx, o.part) = s;
            c.pending.push_back({true, o.part});
        } else if (o.k == K::SIGNAL) {
            if (o.step < 0 || o.step >= steps) { fail(at + "a flag past PF_RING_STEPS"); return false; }
            nx.flag[(size_t) (set * steps + o.step)] = c.seq;
            for (const Copied& p : c.pending) (p.ag ? ag(nx, p.i) : in(nx, p.i)).step = o.step;
            c.pending.clear();
        } else if (o.k == K::OUT) {
            for (int k = 0; k < n; ++k) {
                Slot& s = ag(c, k);
                if (!readable(c, s, set) || s.addends != all) {
                    fail(at + "the sum read out without part " + std::to_string(k) + " finished and covered");
                    return false;
                }
                s.consumed = true;
            }
            if (!c.pending.empty()) { fail(at + "copied after its last flag"); return false; }
        } else {
            fail(at + "a step of unknown kind " + std::to_string((int) o.k));
            return false;
        }
        if (++c.pc == plan[g].size()) {
            c.pc = 0;
            ++c.seq;
        }
    }
    return true;
}

int main() {
    for (int n : {2, 3})
        for (int g = 0; g < n; ++g)
            if (fnx::pf_ring_plan(n, g) != written(n, g))
                fail(std::to_string(n) + " cards, card " + std::to_string(g) + ": the plan is not the written sequence");
    for (const auto& [n, g] : std::vector<std::pair<int, int>>{{1, 0}, {0, 0}, {3, 3}, {3, -1}}) {
        try {
            fnx::pf_ring_plan(n, g);
            fail("pf_ring_plan(" + std::to_string(n) + ", " + std::to_string(g) + ") made a plan");
        } catch (const std::invalid_argument&) {
        }
    }
    constexpr int SUMS = 8, ORDERS = 500;
    int counts = 0;
    for (int n = 2; n <= fnx::NCARD_MAX; ++n)
        if (fnx::ncard_ok(n)) {
            for (unsigned seed = 0; seed < ORDERS; ++seed)
                if (!simulate(n, SUMS, seed)) break;
            ++counts;
        }
    if (fails) {
        std::printf("%d failures\n", fails);
        return 1;
    }
    std::printf("pf ring plan: two and three cards as written; %d counts (2 to %d): %d sums in a row over the two buffer "
                "sets complete, %d random orders each\n",
                counts, fnx::NCARD_MAX, SUMS, ORDERS);
    return 0;
}
