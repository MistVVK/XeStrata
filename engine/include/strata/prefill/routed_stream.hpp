#pragma once
// STRATA_PREFILL_ROUTED_STREAM (opt-in, off by default): the streamed prompt walk drops the ring entries no token of
// the chunk routes to, when their layer's routing is known before the entry is issued.
//
// A chunk of stream_all_min() tokens or more streams every non-resident expert of every layer through the ring
// (prefill.cpp, step 3), and its issuer thread enqueues each copy as soon as the ring has room: with a ring longer
// than a layer's entries, a layer's copies are all on the copy stream before its router has run, so an expert no
// token routes to is copied anyway (measured on an RTX 3090, IQ3_XXS: 47% of the streamed experts at 1,045 tokens,
// 35% at 2K, 15% at 8,192-token chunks).  With the switch on:
//   - the issuer keeps at most `depth` copies queued ahead of the link (it waits until the copy issued `depth`
//     copies earlier has landed), so the copies run back to back but are not committed far ahead of the link;
//     how far the stream runs ahead of the compute (the ring) is unchanged;
//   - after a layer's router, the compute thread publishes which experts at least one token routes to (RouteGate);
//   - the issuer drops an entry not yet issued whose layer is published and whose expert no token routes to.
//     Entries already issued stay; every routed entry is issued, into the ring slot it has today (k % ring), and
//     the consumer's walk is unchanged (it never reads an unrouted entry's slot), so the prompt's products are the
//     same bits.  Only entries copied straight from the pinned arena are dropped: an entry staged through the
//     Stager's pinned buffers keeps its copy (its job queue is consumed in order).
// Header-only so the walk and its bookkeeping are tested on the CPU (tests/prefill/routed_stream_test.cpp).
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace strata::prefill::routed_stream {

constexpr int kDefaultDepth = 16;   // about 28 MB of 1.7 MB blobs, 2.4 ms of a PCIe 3.0 x16 link
constexpr int kMaxDepth = 256;

// STRATA_PREFILL_ROUTED_STREAM: unset, empty or 0 is off; any other number is on.
inline bool enabled_from(const char* v) { return v != nullptr && v[0] != '\0' && std::atoi(v) != 0; }

// STRATA_PREFILL_ROUTED_STREAM_DEPTH: the copies kept queued ahead of the link (default 16, clamped to 1..256).
inline int depth_from(const char* v) {
    if (v == nullptr || v[0] == '\0') return kDefaultDepth;
    return std::clamp(std::atoi(v), 1, kMaxDepth);
}

// Whether a chunk takes the routed walk.  The routed walk records a copied event for every copy it issues, but it does
// not publish the entries a sparse consumer waits on (STRATA_PREFILL_COPY_EVENT_EVERY > 1: a_rec / rec_of) and does not
// deduplicate release waits (STRATA_PREFILL_COPY_WAIT_DEDUPE=1).  With either set the chunk keeps the default issuer:
// otherwise the consumer would wait on a record that never comes, or the requested wait policy would be bypassed.
// held_off: why a streamed chunk keeps the default walk with the switch set (null when it takes the routed walk); the
// engine prints it once, so a set switch does not read as active where it cannot run.
inline const char* held_off(bool issuer_on, bool peer, int copy_event_every, bool copy_wait_dedupe) {
    if (!issuer_on) return "the inline issuer (STRATA_PREFILL_ISSUER=0) copies every expert";
    if (peer) return "the peer prompt ring (--peer-device) keeps its own walk";
    if (copy_event_every != 1 || copy_wait_dedupe)
        return "STRATA_PREFILL_COPY_EVENT_EVERY > 1 and STRATA_PREFILL_COPY_WAIT_DEDUPE=1 keep the default issuer, whose "
               "copy records the routed walk does not write";
    return nullptr;
}
inline bool eligible(bool stream_all, bool issuer_on, bool peer, int copy_event_every, bool copy_wait_dedupe) {
    return stream_all && held_off(issuer_on, peer, copy_event_every, copy_wait_dedupe) == nullptr;
}

// Which experts each layer's tokens route to, published by the compute thread once the layer's router has run and
// read by the issuer thread.  One writer per layer per chunk: the routed bytes are written before the layer's flag
// is set (release), and the reader reads them only after it sees the flag (acquire).
class RouteGate {
public:
    void reset(int64_t n_layers, int32_t n_expert) {
        n_layers_ = n_layers > 0 ? n_layers : 0;
        n_expert_ = n_expert > 0 ? n_expert : 0;
        routed_.assign((size_t) n_layers_ * (size_t) n_expert_, 0);
        pub_.reset(new std::atomic<uint8_t>[(size_t) (n_layers_ > 0 ? n_layers_ : 1)]);
        for (int64_t l = 0; l < n_layers_; ++l) pub_[(size_t) l].store(0, std::memory_order_relaxed);
    }
    // `cnt[e]`: the tokens of the chunk that route to expert e in layer l (the host grouping's counts)
    void publish(int64_t l, const int32_t* cnt) {
        if (l < 0 || l >= n_layers_ || cnt == nullptr) return;
        uint8_t* row = routed_.data() + (size_t) l * (size_t) n_expert_;
        for (int32_t e = 0; e < n_expert_; ++e) row[e] = cnt[e] > 0 ? 1 : 0;
        pub_[(size_t) l].store(1, std::memory_order_release);
    }
    bool published(int64_t l) const {
        return l >= 0 && l < n_layers_ && pub_[(size_t) l].load(std::memory_order_acquire) != 0;
    }
    // true only when layer l is published and no token of the chunk routes to expert e
    bool unrouted(int64_t l, int32_t e) const {
        if (e < 0 || e >= n_expert_ || !published(l)) return false;
        return routed_[(size_t) l * (size_t) n_expert_ + (size_t) e] == 0;
    }

private:
    int64_t n_layers_ = 0;
    int32_t n_expert_ = 0;
    std::vector<uint8_t> routed_;
    std::unique_ptr<std::atomic<uint8_t>[]> pub_;
};

// The walk's operations as six callables (prefill.cpp passes lambdas; a test may pass any type with these members).
template <class C, class S, class L, class D, class I, class Y>
struct FnOps {
    C consumed;
    S stop;
    L landed;
    D droppable;
    I issue;
    Y idle;
};
template <class C, class S, class L, class D, class I, class Y>
FnOps<C, S, L, D, I, Y> make_ops(C c, S s, L l, D d, I i, Y y) {
    return FnOps<C, S, L, D, I, Y>{c, s, l, d, i, y};
}

struct WalkResult {
    bool stopped = false;   // the walk was told to stop before its end
    size_t issued = 0;      // entries copied
    size_t dropped = 0;     // entries not copied: published, unrouted and droppable
};

// The issuer's walk over a chunk's entries `seq` (each with members `l` and `e`), entry k going to ring slot
// k % ring as without the switch.  `a_issued` is advanced past every entry, copied or dropped (the consumer's
// wait_issued reads it).  Ops:
//   size_t consumed()            the consumer's give-back count (acquire load of a_consumed)
//   bool stop()                  the chunk is ending early
//   bool landed(int slot)        the copy last issued into `slot` has completed (a query, never a wait)
//   bool droppable(const E&)     the entry may be dropped (its blob is copied straight from the pinned arena)
//   void issue(size_t k, int sl) enqueue entry k's copy into slot sl exactly as the default walk does
//   void idle()                  nothing to do yet (yield)
// An entry is copied only when (1) the ring has room for it (k < consumed() + ring, as without the switch) and
// (2) fewer than `depth` copies are queued (the copy issued `depth` copies before it has landed).  While it waits for
// either, its layer's routing may be published; then an unrouted entry is dropped instead.
template <class Entry, class Ops>
WalkResult issue_walk(const std::vector<Entry>& seq, size_t ring, int depth, const RouteGate& gate,
                      std::atomic<size_t>& a_issued, Ops& ops) {
    WalkResult r;
    if (ring == 0) return r;
    const size_t q = (size_t) std::clamp(depth, 1, kMaxDepth);
    int win[kMaxDepth];   // the slots of the last q copies, the n-th copy at n % q
    size_t n_out = 0;
    for (size_t k = 0; k < seq.size(); ++k) {
        const Entry& en = seq[k];
        const bool may_drop = ops.droppable(en);
        bool drop = false;
        for (;;) {
            if (may_drop && gate.unrouted(en.l, en.e)) { drop = true; break; }
            if (k < ops.consumed() + ring && (n_out < q || ops.landed(win[n_out % q]))) break;
            if (ops.stop()) { r.stopped = true; return r; }
            ops.idle();
        }
        if (drop) {
            ++r.dropped;
        } else {
            const int sl = (int) (k % ring);
            ops.issue(k, sl);
            win[n_out % q] = sl;
            ++n_out;
            ++r.issued;
        }
        a_issued.store(k + 1, std::memory_order_release);
    }
    return r;
}

}  // namespace strata::prefill::routed_stream
