// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace strata::core {

/// upstream 8c71c492 (#5697): hands the memory this process freed back to the system where the C library keeps it in
/// its heap (glibc: malloc_trim, every arena; freed pages inside the heap are released, not only its top).  MemAvailable,
/// which the RAM gates read, counts only what was handed back.  False where there is no such call (Windows, macOS, musl:
/// their allocators return large blocks on free).  Costs a walk over the heap's free chunks: call it when RAM is short.
inline bool release_free_heap() {
#if defined(__GLIBC__)
    ::malloc_trim(0);
    return true;
#else
    return false;
#endif
}

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // STRATA_CKPT_VRAM_SLOTS: the tag of the device copy of `gdn` (0: none).  Set by conversation_checkpoint_save (and
    // the staged save), the only writers of a checkpoint's GDN bytes besides a whole copy, which keeps the tag.
    uint64_t dev_tag = 0;
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // --reply-checkpoint-every: taken in a reply, between windows (a run-time mark too): the newest replaces the last.
    bool reply = false;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    // STRATA_PARK_VRAM_MIB: the tag this layer's parts were recorded under in the park's VRAM slot (0: none).  Set by
    // conversation_kv_save; a restore that finds the slot copies from it instead of from the buffers above.
    uint64_t dev_tag = 0;
    // STRATA_PARK_VRAM_MIB: set while a background copy fills the buffers above from the slot; its deleter waits for
    // that copy and releases the slot.  This layer's own token (not shared): the layers outlive their image when they
    // are retained for the next park (ConversationCache::retain), and go in any order.  Last member: it goes before
    // any buffer.
    std::shared_ptr<void> pending;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // The conversation the image was captured from: a park reuses it only for the prefix it shares with the
    // conversation then in the main session (a batch slot may have replaced that session's K/V since; upstream #1503).
    std::vector<int32_t> ids;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + ids.capacity() * sizeof(int32_t);
        for (const auto& layer : kv) n += layer.bytes();
        return n;
    }
};

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from (a session file records it; restore requires the same).
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // STRATA_PARK_ASYNC: set while a background copy fills this image's K/V; its deleter waits for that copy, and as
    // one of the last members it runs before any buffer is freed (whichever copy of the image goes last)
    std::shared_ptr<void> pending;
    // STRATA_CKPT_ASYNC: the same for the background copy that fills `live.gdn` (the park's running state)
    std::shared_ptr<void> live_pending;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j] == image)) return 0;
        ++j;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

// #342: whether a conversation's live tokens or checkpoint chain hold the checkpoint `c` (tokens and images)
inline bool conversation_chain_holds(const ConversationCheckpoint& c, const std::vector<int32_t>& ids,
                                     const std::vector<ConversationImageKey>& images,
                                     const std::vector<ConversationCheckpoint>& checkpoints) {
    if (c.ids == ids && c.imgs == images) return true;
    for (const auto& k : checkpoints)
        if (k.ids == c.ids && k.imgs == c.imgs) return true;
    return false;
}

/// `c` holds the state `ids` (with `images`) describes, or a continuation of it: its tokens start with `ids` and its
/// pictures below ids.size() are exactly `images` (upstream 01e404c14: an idle batch slot need not park it again).
inline bool conversation_covers(const ConversationCheckpoint& c, const std::vector<int32_t>& ids,
                                const std::vector<ConversationImageKey>& images) {
    const size_t n = ids.size();
    if (n == 0 || c.ids.size() < n || !std::equal(ids.begin(), ids.end(), c.ids.begin())) return false;
    size_t j = 0;
    for (const auto& image : c.imgs) {
        if (image.start >= (int64_t) n) continue;
        if (j == images.size() || !(images[j] == image)) return false;
        ++j;
    }
    return j == images.size();
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
        uint64_t serial = 0;   // the entry's identity; its index shifts when entries before it go
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `ids`: the conversation the K/V belongs to (take_reuse_for refuses it for any other; none: never reused).
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens, std::vector<int32_t> ids = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, std::move(ids)};
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    // The retained K/V for parking `live`: limited to the prefix `live` shares with the conversation it was captured
    // from, and nothing when they share none.  Without it a park after a batch slot gave its conversation back to the
    // main session wrote the restored conversation's K/V into the new one's image (upstream f8ff3420d, #1503).
    ConversationKvReuse take_reuse_for(const std::vector<int32_t>& live) {
        ConversationKvReuse r = std::exchange(reuse_, {});
        const size_t n = std::min(r.ids.size(), live.size());
        size_t same = 0;
        while (same < n && r.ids[same] == live[same]) ++same;
        r.unchanged_tokens = std::min(r.unchanged_tokens, (int64_t) same);
        if (r.unchanged_tokens <= 0) return {};
        return r;
    }
    /// A parked conversation holds this state already (its live tokens or one of its checkpoints start with `ids`,
    /// same pictures, same steering).  Read-only; no LRU effect.
    bool holds(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const {
        for (const auto& e : entries_) {
            if (e.cvec != cvec) continue;
            if (conversation_covers(e.live, ids, images)) return true;
            for (const auto& c : e.checkpoints)
                if (conversation_covers(c, ids, images)) return true;
        }
        return false;
    }
    int64_t retained_tokens() const { return reuse_.kv.empty() ? 0 : reuse_.unchanged_tokens; }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live, serials_[i]};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        erase(index);
        return out;
    }

    // A checkpoint inside a parked conversation that a NEW conversation starts with - a sibling subagent sharing the
    // system prompt and tool list, a compacted history - is borrowed: restored from the entry, which stays parked.
    // take() would move the whole conversation into the session and the new request would overwrite everything after
    // that checkpoint, so the other conversation's next turn would read its history again.  A match on the entry's
    // live state or on its deepest checkpoint is the same conversation going on (the deepest one: its client
    // re-rendered the last reply), whose tail is stale: that is still taken.  So is a match that reaches a pinned
    // shared prefix (pin=N): a borrow copies the checkpoints up to the match, and taking it instead keeps one
    // conversation holding the pin, as pin=N expects.
    bool borrows(const Match& m) const {
        if (m.tokens <= 0 || m.live || m.index >= entries_.size() || serials_[m.index] != m.serial) return false;
        size_t deepest = 0;
        for (const auto& c : entries_[m.index].checkpoints) {
            if (c.pinned && (int64_t) c.ids.size() <= m.tokens) return false;
            deepest = std::max(deepest, c.ids.size());
        }
        return (size_t) m.tokens < deepest;
    }
    // The entry with this identity: its index (SIZE_MAX: gone), or the entry itself (nullptr: gone; the pointer is
    // valid until the cache changes).
    size_t index_of(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (serials_[i] == serial) return i;
        return SIZE_MAX;
    }
    const SavedConversation* find(uint64_t serial) const {
        const size_t i = index_of(serial);
        return i == SIZE_MAX ? nullptr : &entries_[i];
    }
    // While a borrowed entry (the donor) is being restored, parking the outgoing conversation must not evict or drop
    // it: guard() it.  When the outgoing image fits only without it, make_room refuses before evicting anything and
    // guard_blocked() says so: the caller then takes the entry as before, so the conversation that was active a
    // moment ago keeps its place.
    void guard(uint64_t serial) { guarded_ = serial; guard_blocked_ = false; }
    void unguard() { guarded_ = 0; }
    bool guard_blocked() const { return guard_blocked_; }
    size_t borrowed() const { return borrowed_; }
    void count_borrow() { ++borrowed_; }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    // `evicted`: receives the conversations pushed out (--conversation-save writes them, then frees them).
    bool make_room(size_t incoming, size_t held = 0, std::vector<SavedConversation>* evicted = nullptr) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        auto stays = [&](size_t i) { return entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_); };
        if (guarded_ != 0 && index_of(guarded_) != SIZE_MAX) {
            // with everything else gone, would it fit beside the entries that stay (the guarded donor and any pinned
            // shared prefix)?  If not, evict nothing
            size_t kept = 0, kept_bytes = 0;
            for (size_t i = 0; i < entries_.size(); ++i)
                if (stays(i)) { ++kept; kept_bytes += entries_[i].bytes(); }
            if (kept >= slots_ || kept_bytes > budget_ - held - incoming) { guard_blocked_ = true; return false; }
        }
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the caller skips parking it - the pinned prefix is what the queries come back to)
            if (!drop_oldest(evicted)) return false;
        }
        return true;
    }

    // The parked conversation that has gone unused the longest, for the callers that free RAM one entry at a time and
    // read MemAvailable after each (the physical-RAM admission gate in generate.cpp).  upstream 8c71c492 (#5697): the
    // buffers' segments can come from the C library's heap, and free() hands back only the heap's top - an entry below a
    // newer one stayed resident and the gate read no change - so the freed memory is handed back here too
    // (release_free_heap).  make_room() drops entries without it: its room is for the image allocated right after, which
    // reuses that memory.  False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    // `evicted`: as make_room's.
    bool evict_oldest(std::vector<SavedConversation>* evicted = nullptr) {
        if (!drop_oldest(evicted)) return false;
        release_free_heap();
        return true;
    }

    // #879 (upstream f332a855): every parked conversation (pinned ones too) and the retained K/V go - after
    // non-finite logits any of them may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        entries_.clear();
        serials_.clear();
        bytes_ = 0;
        reuse_ = {};
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() &&
                conversation_chain_holds(*deepest, ids, images, checkpoints) &&
                !(guarded_ != 0 && serials_[i] == guarded_)) {
                bytes_ -= e.bytes();
                erase(i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    // upstream c7a0cef13 (#4420): a batch slot's chain holds one checkpoint (the one its admission copied in), while the main
    // session's parked copy of the same conversation keeps the whole chain, the root at the end of an agent's system prompt among
    // them.  Parking the slot would drop that copy as superseded (above) and the root with it, and every sibling would read the
    // shared prefix again.  True when drop_superseded(ids, images, checkpoints, cvec) would drop an entry that has a checkpoint the
    // given chain does not hold.  Nothing is changed.
    bool supersedes_richer(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec != cvec || !deepest || deepest->ids.empty() ||
                !conversation_chain_holds(*deepest, ids, images, checkpoints) || (guarded_ != 0 && serials_[i] == guarded_))
                continue;
            for (const auto& c : e.checkpoints)
                if (!conversation_chain_holds(c, ids, images, checkpoints)) return true;
        }
        return false;
    }
    size_t superseded() const { return superseded_; }
    const std::deque<SavedConversation>& entries() const { return entries_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none.  It is the entry drop_superseded() would remove, so the caller can tell whether a
    /// park now would only rewrite what a turn from now drops: the prefix such an entry holds is this conversation's
    /// prefix, and the conversation only appends, so the copy stays usable for the reuse it was made for.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        size_t longest = 0;
        for (const auto& e : entries_) {
            if (e.cvec != cvec) continue;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (!deepest || deepest->ids.empty()) continue;
            bool holds = deepest->ids == ids && deepest->imgs == images;
            for (const auto& k : checkpoints)
                if (!holds && k.ids == deepest->ids && k.imgs == deepest->imgs) holds = true;
            if (holds) longest = std::max(longest, deepest->ids.size());
        }
        return longest;
    }

    bool put(SavedConversation&& image, size_t held = 0, std::vector<SavedConversation>* evicted = nullptr) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held, evicted)) return false;
        entries_.push_back(std::move(image));
        serials_.push_back(next_serial_++);
        bytes_ += n;
        return true;
    }

private:
    // evict_oldest() without handing the memory back (make_room's loop)
    bool drop_oldest(std::vector<SavedConversation>* evicted = nullptr) {
        // the oldest entry that holds no pinned shared prefix and is not the guarded (borrowed) donor
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            bytes_ -= entries_[i].bytes();
            if (evicted) evicted->push_back(std::move(entries_[i]));
            erase(i);
            ++evictions_;
            return true;
        }
        return false;
    }
    void erase(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) index);
    }
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, borrowed_ = 0;
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    std::deque<SavedConversation> entries_; // least recently active first
    std::deque<uint64_t> serials_;          // entries_[i]'s identity, never reused
    uint64_t next_serial_ = 1, guarded_ = 0;
    bool guard_blocked_ = false;
    ConversationKvReuse reuse_;
};

} // namespace strata::core
