// routed_stream_test: STRATA_PREFILL_ROUTED_STREAM's issuer walk and route gate (include/strata/prefill/routed_stream.hpp).
// CPU only.  A simulated chunk: an in-order link that lands one copy every few ticks, a consumer that walks the layers as
// prefill.cpp's streamed walk does (publish a layer's routing, give back unrouted entries, read a routed entry's slot
// once its copy has landed), and the issuer walk itself, each idle() of which advances the simulation by one tick.
#include "strata/prefill/routed_stream.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace rs = strata::prefill::routed_stream;

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

struct Entry { int32_t l, e; int job; };

struct Case {
    int layers = 6, n_expert = 64, resident_every = 3, ring = 48, depth = 4;
    double unrouted = 0.4;   // share of the streamed experts no token routes to
    int link_ticks = 3;      // ticks per copy
    int publish_delay = 5;   // ticks from the end of layer l-1 to layer l's routing
    int compute_ticks = 1;   // ticks per routed entry
    bool publish = true;     // false: no layer is ever published (the fused path, or the switch off)
    int staged_every = 0;    // every n-th entry has a stager job (never dropped); 0: none
    unsigned seed = 1;
};

struct Sim {
    Case c;
    std::vector<Entry> seq;
    std::vector<size_t> seq_start;
    std::vector<std::vector<int32_t>> cnt;   // [layer][expert] tokens
    rs::RouteGate gate;
    std::atomic<size_t> a_issued{0}, a_consumed{0};
    // the link: copies in issue order, each lands link_ticks after it starts
    struct Copy { size_t k; int slot; };
    std::deque<Copy> queue;
    int link_left = 0;
    std::vector<long> slot_content;          // the entry whose bytes a ring slot holds (-1: none)
    std::vector<long> slot_last_issue;       // the entry last issued into a slot (its `copied` event)
    std::vector<char> landed_k;              // per entry: its copy has landed
    std::vector<char> issued_k, dropped_k, pub_at_drop;
    size_t max_out = 0, outstanding = 0;
    bool ring_violation = false, depth_violation = false, wrong_bytes = false, read_before_issue = false;
    // the consumer
    int64_t cl = 0;            // layer being walked
    size_t ck = 0;             // next entry of the walk
    int phase = 0;             // 0: waiting to publish, 1: walking the layer
    int wait = 0;
    int64_t tick = 0;

    explicit Sim(const Case& cc) : c(cc) {
        std::mt19937 rng(c.seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        seq_start.assign((size_t) c.layers + 1, 0);
        cnt.assign((size_t) c.layers, std::vector<int32_t>((size_t) c.n_expert, 0));
        for (int l = 0; l < c.layers; ++l) {
            seq_start[(size_t) l] = seq.size();
            for (int e = 0; e < c.n_expert; ++e) {
                cnt[(size_t) l][(size_t) e] = u(rng) < c.unrouted ? 0 : 1 + (int) (u(rng) * 7);
                if (c.resident_every > 0 && e % c.resident_every == 0) continue;   // resident: not in the stream
                const int job = c.staged_every > 0 && seq.size() % (size_t) c.staged_every == 0 ? (int) seq.size() : -1;
                seq.push_back({l, e, job});
            }
        }
        seq_start[(size_t) c.layers] = seq.size();
        gate.reset(c.layers, c.n_expert);
        slot_content.assign((size_t) c.ring, -1);
        slot_last_issue.assign((size_t) c.ring, -1);
        landed_k.assign(seq.size(), 0);
        issued_k.assign(seq.size(), 0);
        dropped_k.assign(seq.size(), 0);
        pub_at_drop.assign(seq.size(), 0);
        wait = c.publish_delay;
    }

    void step_link() {
        if (queue.empty()) return;
        if (--link_left > 0) return;
        const Copy cp = queue.front();
        queue.pop_front();
        slot_content[(size_t) cp.slot] = (long) cp.k;   // the bytes land
        landed_k[cp.k] = 1;
        --outstanding;
        link_left = c.link_ticks;
    }
    void give_back(size_t upto) { a_consumed.store(upto, std::memory_order_release); }
    void step_consumer() {
        if (cl >= c.layers) return;
        const size_t kend = seq_start[(size_t) cl + 1];
        if (phase == 0) {   // the layer's attention and router: then the routing is known
            if (--wait > 0) return;
            if (c.publish) gate.publish(cl, cnt[(size_t) cl].data());
            ck = seq_start[(size_t) cl];
            phase = 1;
            wait = 0;
            return;
        }
        if (wait > 0) { --wait; return; }
        if (ck < kend) {
            const Entry& en = seq[ck];
            if (cnt[(size_t) cl][(size_t) en.e] == 0) {   // release_to: an unrouted entry only gives its slot back
                give_back(++ck);
                return;
            }
            // a routed entry: wait_issued, then the copy's event, then compute from the slot
            if (a_issued.load(std::memory_order_acquire) <= ck) return;
            if (!issued_k[ck]) { read_before_issue = true; give_back(++ck); return; }
            if (!landed_k[ck]) return;
            if (slot_content[ck % (size_t) c.ring] != (long) ck) wrong_bytes = true;
            wait = c.compute_ticks;
            give_back(++ck);
            return;
        }
        ++cl;   // the next layer's attention and router
        phase = 0;
        wait = c.publish_delay;
    }
    void tick_world() {
        ++tick;
        step_link();
        step_consumer();
    }
};

struct SimOps {
    Sim& s;
    size_t consumed() { return s.a_consumed.load(std::memory_order_acquire); }
    bool stop() { return s.tick > 50000000; }
    bool landed(int slot) {
        const long k = s.slot_last_issue[(size_t) slot];
        return k < 0 || s.landed_k[(size_t) k];
    }
    bool droppable(const Entry& en) { return en.job < 0; }
    void issue(size_t k, int sl) {
        if (k >= consumed() + (size_t) s.c.ring) s.ring_violation = true;
        // the copy waits for the slot's previous entry to be released (its `used` event): the ring rule above
        s.slot_last_issue[(size_t) sl] = (long) k;
        s.queue.push_back({k, sl});
        if (s.queue.size() == 1) s.link_left = s.c.link_ticks;
        ++s.outstanding;
        s.max_out = std::max(s.max_out, s.outstanding);
        if (s.outstanding > (size_t) s.c.depth) s.depth_violation = true;
        s.issued_k[k] = 1;
    }
    void idle() { s.tick_world(); }
};

// drops are recorded through the gate as seen at the walk's end; check each dropped entry was unrouted and published
static void run_case(const Case& c, const char* name, size_t* dropped_out = nullptr, size_t* unrouted_out = nullptr) {
    Sim s(c);
    // the walk ticks the world only while it waits; ticks between its steps (the issuer's own work) advance it too
    struct TickingOps : SimOps {
        void issue(size_t k, int sl) { SimOps::issue(k, sl); s.tick_world(); }
    } tops{{s}};
    const rs::WalkResult r = rs::issue_walk(s.seq, (size_t) c.ring, c.depth, s.gate, s.a_issued, tops);
    for (size_t k = 0; k < s.seq.size(); ++k)
        if (!s.issued_k[k]) s.dropped_k[k] = 1;
    // let the consumer finish the chunk
    while (s.cl < c.layers && s.tick < 50000000) s.tick_world();
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s: walk not stopped", name);
    expect(!r.stopped, buf);
    std::snprintf(buf, sizeof buf, "%s: issued + dropped == entries (%zu + %zu vs %zu)", name, r.issued, r.dropped, s.seq.size());
    expect(r.issued + r.dropped == s.seq.size(), buf);
    std::snprintf(buf, sizeof buf, "%s: a_issued covers every entry", name);
    expect(s.a_issued.load() == s.seq.size(), buf);
    std::snprintf(buf, sizeof buf, "%s: the consumer walked every layer", name);
    expect(s.cl == c.layers, buf);
    std::snprintf(buf, sizeof buf, "%s: no copy beyond the ring", name);
    expect(!s.ring_violation, buf);
    std::snprintf(buf, sizeof buf, "%s: at most depth copies queued (max %zu, depth %d)", name, s.max_out, c.depth);
    expect(!s.depth_violation, buf);
    std::snprintf(buf, sizeof buf, "%s: every routed entry read its own bytes from slot k %% ring", name);
    expect(!s.wrong_bytes && !s.read_before_issue, buf);
    size_t unrouted = 0, bad_drop = 0;
    for (size_t k = 0; k < s.seq.size(); ++k) {
        const Entry& en = s.seq[k];
        const bool unr = s.cnt[(size_t) en.l][(size_t) en.e] == 0;
        if (unr && en.job < 0) ++unrouted;
        if (s.dropped_k[k] && (!unr || en.job >= 0 || !c.publish)) ++bad_drop;
    }
    std::snprintf(buf, sizeof buf, "%s: only unrouted, arena-copied entries of published layers dropped (%zu bad)", name, bad_drop);
    expect(bad_drop == 0, buf);
    std::printf("  %-34s entries %5zu  unrouted %5zu  dropped %5zu (%.0f%% of unrouted)  max queued %zu  ticks %lld\n", name,
                s.seq.size(), unrouted, r.dropped, unrouted ? 100.0 * (double) r.dropped / (double) unrouted : 0.0,
                s.max_out, (long long) s.tick);
    if (dropped_out) *dropped_out = r.dropped;
    if (unrouted_out) *unrouted_out = unrouted;
}

// The same walk on three threads (issuer, consumer, link), real atomics, no simulated clock: the bookkeeping holds
// whatever the interleaving.
static void threaded_case(unsigned seed, int depth) {
    Case c;
    c.layers = 8; c.n_expert = 128; c.ring = 96; c.depth = depth; c.seed = seed; c.unrouted = 0.45;
    Sim s(c);
    std::atomic<bool> done{false};
    std::vector<std::atomic<int>> landed(s.seq.size());
    for (auto& x : landed) x.store(0);
    std::vector<std::atomic<long>> content((size_t) c.ring);
    for (auto& x : content) x.store(-1);
    std::vector<std::atomic<long>> last((size_t) c.ring);
    for (auto& x : last) x.store(-1);
    std::vector<std::atomic<int>> issued(s.seq.size());
    for (auto& x : issued) x.store(0);
    std::mutex qm;
    std::deque<std::pair<size_t, int>> q;
    std::atomic<size_t> out{0};
    std::atomic<bool> depth_bad{false}, ring_bad{false}, bytes_bad{false};
    struct ThreadOps {
        Sim& s; std::mutex& qm; std::deque<std::pair<size_t, int>>& q; std::vector<std::atomic<long>>& last;
        std::vector<std::atomic<int>>& landed_v; std::vector<std::atomic<int>>& issued; std::atomic<size_t>& out;
        std::atomic<bool>& depth_bad; std::atomic<bool>& ring_bad; int depth;
        size_t consumed() { return s.a_consumed.load(std::memory_order_acquire); }
        bool stop() { return false; }
        bool landed(int slot) { const long k = last[(size_t) slot].load(); return k < 0 || landed_v[(size_t) k].load(); }
        bool droppable(const Entry& en) { return en.job < 0; }
        void issue(size_t k, int sl) {
            if (k >= consumed() + (size_t) s.c.ring) ring_bad = true;
            last[(size_t) sl].store((long) k);
            issued[k].store(1);
            if (out.fetch_add(1) + 1 > (size_t) depth) depth_bad = true;
            std::lock_guard<std::mutex> g(qm);
            q.emplace_back(k, sl);
        }
        void idle() { std::this_thread::yield(); }
    } ops{s, qm, q, last, landed, issued, out, depth_bad, ring_bad, depth};
    std::thread link([&] {
        while (!done.load()) {
            std::pair<size_t, int> cp;
            {
                std::lock_guard<std::mutex> g(qm);
                if (q.empty()) { cp.second = -1; } else { cp = q.front(); q.pop_front(); }
            }
            if (cp.second < 0) { std::this_thread::yield(); continue; }
            for (int i = 0; i < 50; ++i) std::this_thread::yield();   // the copy's time
            content[(size_t) cp.second].store((long) cp.first);
            out.fetch_sub(1);   // before `landed`: the issuer counts a queued copy until it sees this one land
            landed[cp.first].store(1);
        }
    });
    std::thread consumer([&] {
        for (int l = 0; l < c.layers; ++l) {
            for (int i = 0; i < 200; ++i) std::this_thread::yield();   // attention and router
            s.gate.publish(l, s.cnt[(size_t) l].data());
            for (size_t k = s.seq_start[(size_t) l]; k < s.seq_start[(size_t) l + 1]; ++k) {
                if (s.cnt[(size_t) l][(size_t) s.seq[k].e] == 0) { s.a_consumed.store(k + 1, std::memory_order_release); continue; }
                while (s.a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
                while (!landed[k].load()) std::this_thread::yield();
                if (!issued[k].load() || content[k % (size_t) c.ring].load() != (long) k) bytes_bad = true;
                s.a_consumed.store(k + 1, std::memory_order_release);
            }
        }
    });
    const rs::WalkResult r = rs::issue_walk(s.seq, (size_t) c.ring, depth, s.gate, s.a_issued, ops);
    consumer.join();
    done = true;
    link.join();
    char buf[200];
    std::snprintf(buf, sizeof buf, "threaded seed %u depth %d: bookkeeping (issued %zu dropped %zu of %zu)", seed, depth,
                  r.issued, r.dropped, s.seq.size());
    expect(r.issued + r.dropped == s.seq.size() && !depth_bad && !ring_bad && !bytes_bad, buf);
    size_t bad = 0;
    for (size_t k = 0; k < s.seq.size(); ++k)
        if (!issued[k].load() && s.cnt[(size_t) s.seq[k].l][(size_t) s.seq[k].e] != 0) ++bad;
    std::snprintf(buf, sizeof buf, "threaded seed %u depth %d: every routed entry issued", seed, depth);
    expect(bad == 0, buf);
    std::printf("  threaded seed %u depth %-3d        entries %5zu  dropped %5zu\n", seed, depth, s.seq.size(), r.dropped);
}

int main() {
    // the switch and its depth
    expect(!rs::enabled_from(nullptr), "unset: off");
    expect(!rs::enabled_from(""), "empty: off");
    expect(!rs::enabled_from("0"), "0: off");
    expect(rs::enabled_from("1"), "1: on");
    expect(rs::depth_from(nullptr) == rs::kDefaultDepth, "depth unset: default");
    expect(rs::depth_from("8") == 8, "depth 8");
    expect(rs::depth_from("0") == 1 && rs::depth_from("-4") == 1, "depth clamps to 1");
    expect(rs::depth_from("100000") == rs::kMaxDepth, "depth clamps to the maximum");

    // Which chunks take the routed walk: the copy policies of STRATA_PREFILL_COPY_EVENT_EVERY > 1 and
    // STRATA_PREFILL_COPY_WAIT_DEDUPE=1 keep the default issuer.  Otherwise a sparse consumer's wait_copied would spin
    // on an a_rec the routed walk never publishes, and the requested release-wait deduplication would be bypassed.
    expect(rs::eligible(true, true, false, 1, false), "streamed threaded single GPU: routed walk eligible");
    expect(!rs::eligible(false, true, false, 1, false), "short prompt: default walk");
    expect(!rs::eligible(true, false, false, 1, false), "inline issuer: default walk");
    expect(!rs::eligible(true, true, true, 1, false), "peer prompt ring: default walk");
    for (int every : {2, 16, 64}) {
        expect(!rs::eligible(true, true, false, every, false), "sparse copied events: default walk");
        expect(!rs::eligible(true, true, false, every, true), "sparse events and dedupe: default walk");
    }
    expect(!rs::eligible(true, true, false, 1, true), "release-wait dedupe: default walk");
    // the reason the engine prints once when the switch is set and a streamed chunk keeps the default walk
    expect(rs::held_off(true, false, 1, false) == nullptr, "eligible: no reason");
    for (bool iss : {false, true})
        for (bool peer : {false, true})
            for (int every : {1, 2, 16})
                for (bool dedupe : {false, true}) {
                    const char* why = rs::held_off(iss, peer, every, dedupe);
                    expect((why == nullptr) == rs::eligible(true, iss, peer, every, dedupe), "a reason exactly when held off");
                }
    auto names = [](const char* why, const char* what) { return why != nullptr && std::strstr(why, what) != nullptr; };
    expect(names(rs::held_off(false, false, 1, false), "STRATA_PREFILL_ISSUER=0"), "inline issuer named");
    expect(names(rs::held_off(true, true, 1, false), "--peer-device"), "peer ring named");
    expect(names(rs::held_off(true, false, 4, false), "STRATA_PREFILL_COPY_EVENT_EVERY") &&
               names(rs::held_off(true, false, 1, true), "STRATA_PREFILL_COPY_WAIT_DEDUPE"),
           "copy policies named");

    // the gate
    {
        rs::RouteGate g;
        g.reset(3, 8);
        const int32_t c1[8] = {0, 2, 0, 1, 0, 0, 5, 0};
        expect(!g.published(1) && !g.unrouted(1, 0), "nothing published: nothing unrouted");
        g.publish(1, c1);
        expect(g.published(1) && !g.published(0) && !g.published(2), "one layer published");
        bool ok = true;
        for (int e = 0; e < 8; ++e) ok = ok && g.unrouted(1, e) == (c1[e] == 0);
        expect(ok, "unrouted exactly where the count is 0");
        expect(!g.unrouted(0, 0) && !g.unrouted(2, 0), "other layers keep every entry");
        expect(!g.unrouted(1, -1) && !g.unrouted(1, 8) && !g.unrouted(-1, 0) && !g.unrouted(3, 0), "out of range: kept");
        g.reset(3, 8);
        expect(!g.published(1) && !g.unrouted(1, 0), "reset: nothing published");
    }

    std::printf("routed_stream_test: simulated chunks\n");
    {
        Case c;   // routing never published (the default walk's behaviour): every entry copied, in order
        c.publish = false;
        size_t d = 1;
        run_case(c, "never published", &d);
        expect(d == 0, "never published: nothing dropped");
    }
    {
        Case c;   // the routing of every layer known before the walk could reach it: every unrouted entry dropped
        c.publish_delay = 1; c.link_ticks = 40; c.ring = 20;
        size_t d = 0, u = 0;
        run_case(c, "routing long before the link", &d, &u);
        expect(d > 0 && d >= u * 8 / 10, "early routing: most unrouted entries dropped");
    }
    {
        Case c;   // a link-bound chunk: some of each layer is issued before its routing lands
        c.publish_delay = 30; c.link_ticks = 4;
        size_t d = 0;
        run_case(c, "link-bound", &d);
        expect(d > 0, "link-bound: some entries dropped");
    }
    {
        Case c;   // stager jobs are never dropped
        c.staged_every = 3; c.publish_delay = 1; c.link_ticks = 20;
        run_case(c, "with stager jobs");
    }
    {
        Case c;   // a ring smaller than a layer, depth larger than the ring
        c.ring = 10; c.depth = 64; c.link_ticks = 2;
        run_case(c, "ring < layer, depth > ring");
    }
    {
        Case c;   // depth 1: one copy in flight at a time
        c.depth = 1; c.link_ticks = 2;
        run_case(c, "depth 1");
    }
    {
        Case c;   // every expert routed: nothing to drop
        c.unrouted = 0.0;
        size_t d = 1;
        run_case(c, "all routed", &d);
        expect(d == 0, "all routed: nothing dropped");
    }
    {
        Case c;   // nothing routed in the stream (all unrouted), fast compute
        c.unrouted = 1.0; c.link_ticks = 5;
        run_case(c, "none routed");
    }
    for (unsigned seed = 2; seed < 40; ++seed) {
        Case c;
        c.seed = seed;
        c.layers = 3 + (int) (seed % 5);
        c.n_expert = 32 + (int) (seed * 7 % 97);
        c.ring = 4 + (int) (seed * 13 % 120);
        c.depth = 1 + (int) (seed * 5 % 40);
        c.unrouted = (seed % 10) / 10.0;
        c.link_ticks = 1 + (int) (seed % 6);
        c.publish_delay = 1 + (int) (seed * 3 % 60);
        c.compute_ticks = (int) (seed % 4);
        c.staged_every = seed % 3 == 0 ? 4 : 0;
        char name[64];
        std::snprintf(name, sizeof name, "random seed %u", seed);
        run_case(c, name);
    }
    std::printf("routed_stream_test: threaded walks\n");
    for (unsigned seed = 1; seed <= 6; ++seed) threaded_case(seed, seed % 2 ? 4 : 16);

    if (g_fail) { std::fprintf(stderr, "routed_stream_test: %d failure(s)\n", g_fail); return 1; }
    std::printf("routed_stream_test: OK\n");
    return 0;
}
