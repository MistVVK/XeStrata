#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

// Count or fail individual deque allocations without adding production hooks.
static long fail_after = -1;
static bool count_allocations = false;
static size_t allocation_count = 0;
void* operator new(std::size_t n) {
    if (count_allocations) ++allocation_count;
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}

struct ThrowingPool : SegmentPoolBudget {
    size_t idle = 0, reclaims = 0, expected_size = 0;
    uint64_t old_serial = 0, new_serial = 0;
    ConversationCache* cache = nullptr;
    bool fail_next = true, saw_staged = false;
    std::vector<ConversationCheckpoint> released;

    size_t idle_bytes() const override { return idle; }
    size_t shrink(size_t want) override {
        ++reclaims;
        saw_staged = cache && cache->size() == expected_size && cache->find(old_serial) &&
                     cache->index_of(new_serial) == expected_size - 1;
        if (fail_next) {
            fail_next = false;
            fail_after = 0;
            try { released.emplace_back(); }
            catch (...) { fail_after = -1; throw; }
            fail_after = -1;
        }
        const size_t given = std::min(idle, want);
        idle -= given;
        return given;
    }
};

void pool_reclaim_allocation_failure() {
    constexpr size_t budget = 1u << 20, held = 256;
    ThrowingPool pool;
    ConversationCache cache(budget, 4);
    ConversationCheckpoint checkpoint;
    checkpoint.ids = {10, 11}; checkpoint.gdn = {1, 2, 3};
    auto old = image({10, 11, 12});
    old.checkpoints.push_back(checkpoint);
    old.kv.emplace_back(); old.kv.back().k.resize(8, 0x35);
    check(cache.put(std::move(old)) && cache.put(image({99, 98, 97})), "park entries before pool failure");
    const uint64_t old_serial = cache.best(std::vector<int32_t>{10, 11, 12, 90}, {}, true).serial;
    const uint64_t other_serial = cache.best(std::vector<int32_t>{99, 98, 97, 90}, {}, true).serial;
    const size_t before_bytes = cache.bytes(), old_bytes = cache.find(old_serial)->bytes();
    auto replacement = image({10, 11, 20});
    replacement.checkpoints.push_back(checkpoint);
    replacement.kv.emplace_back(); replacement.kv.back().k.resize(8, 0x5a);
    const size_t replacement_bytes = replacement.bytes();
    pool.idle = budget; pool.cache = &cache; pool.expected_size = 3;
    pool.old_serial = old_serial; pool.new_serial = other_serial + 1;
    cache.set_pool(&pool);
    check(cache.can_fit(replacement_bytes, held), "exchange fits before pool reclaim");
    bool threw = false;
    try { (void) cache.put_exchanged(std::move(replacement), held); }
    catch (const std::bad_alloc&) { threw = true; }
    check(threw && pool.reclaims == 1 && pool.saw_staged,
          "pool allocation fails after both queue insertions and before old removal");
    const auto* preserved = cache.find(old_serial);
    uint8_t old_k = 0, new_k = 0;
    check(cache.size() == 2 && cache.index_of(old_serial) == 0 && cache.index_of(other_serial) == 1 &&
          cache.index_of(pool.new_serial) == SIZE_MAX && cache.bytes() == before_bytes &&
          cache.superseded() == 0 && cache.evictions() == 0 && preserved && preserved->bytes() == old_bytes &&
          preserved->live.ids == std::vector<int32_t>({10, 11, 12}) &&
          preserved->checkpoints.size() == 1 && preserved->checkpoints[0].ids == checkpoint.ids &&
          preserved->checkpoints[0].gdn == checkpoint.gdn && preserved->kv.size() == 1 &&
          preserved->kv[0].k.read(&old_k, 0, 1) && old_k == 0x35,
          "pool failure preserves old entries, order, serials, bytes and counters");
    check(replacement.bytes() == replacement_bytes && replacement.live.ids == std::vector<int32_t>({10, 11, 20}) &&
          replacement.checkpoints.size() == 1 && replacement.checkpoints[0].ids == checkpoint.ids &&
          replacement.checkpoints[0].gdn == checkpoint.gdn && replacement.kv.size() == 1 &&
          replacement.kv[0].k.read(&new_k, 0, 1) && new_k == 0x5a,
          "pool failure returns the intact replacement for retry");
    check(cache.put_exchanged(std::move(replacement), held), "retry after pool failure succeeds");
    const auto inserted = cache.best(std::vector<int32_t>{10, 11, 20, 90}, {}, true);
    const auto* stored = cache.find(inserted.serial);
    new_k = 0;
    check(pool.reclaims == 2 && pool.saw_staged && cache.size() == 2 && cache.find(old_serial) == nullptr &&
          cache.index_of(other_serial) == 0 && inserted.serial == pool.new_serial &&
          cache.index_of(inserted.serial) == 1 && cache.bytes() == before_bytes - old_bytes + replacement_bytes &&
          cache.superseded() == 1 && cache.evictions() == 0 && stored && stored->kv.size() == 1 &&
          stored->kv[0].k.read(&new_k, 0, 1) && new_k == 0x5a,
          "pool retry removes only the intended old entry and assigns its serial once");
}

void publication_allocation_failures() {
    // Find a real deque boundary where each push allocates. This avoids assuming
    // a particular standard library's block size or adding a production hook.
    size_t seed_count = 0, serial_allocations = 0;
    {
        std::deque<uint64_t> serials;
        std::deque<SavedConversation> entries;
        for (size_t n = 0; n < 4096; ++n) {
            allocation_count = 0; count_allocations = true;
            serials.push_back(n);
            count_allocations = false;
            const size_t serial_calls = allocation_count;
            allocation_count = 0; count_allocations = true;
            entries.emplace_back();
            count_allocations = false;
            if (n >= 2 && serial_calls && allocation_count) {
                seed_count = n;
                serial_allocations = serial_calls;
                break;
            }
        }
    }
    check(seed_count != 0, "found simultaneous deque allocation boundary");
    for (int failed_queue = 0; failed_queue < 2; ++failed_queue) {
        ConversationCache cache(size_t(64) << 20, seed_count + 2);
        ConversationCheckpoint checkpoint;
        checkpoint.ids = {10, 11}; checkpoint.gdn = {1, 2, 3};
        SavedConversation old = image({10, 11, 12});
        old.checkpoints.push_back(checkpoint);
        check(cache.put(std::move(old)), "park replaceable old entry");
        std::vector<uint64_t> serials;
        serials.push_back(cache.best(std::vector<int32_t>{10, 11, 12, 99}, {}, true).serial);
        for (size_t i = 1; i < seed_count; ++i) {
            const int32_t id = 10000 + static_cast<int32_t>(i);
            auto other = image({id, 2, 3});
            check(cache.put(std::move(other)), "park unrelated entry");
            serials.push_back(cache.best(std::vector<int32_t>{id, 2, 3, 99}, {}, true).serial);
        }
        const uint64_t old_serial = serials.front();
        const size_t old_bytes = cache.find(old_serial)->bytes();
        const size_t before_bytes = cache.bytes();
        auto replacement = image({10, 11, 20});
        replacement.checkpoints.push_back(checkpoint);
        const size_t replacement_bytes = replacement.bytes();
        bool threw = false;
        fail_after = failed_queue == 0 ? 0 : static_cast<long>(serial_allocations);
        try { (void) cache.put_exchanged(std::move(replacement)); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1;
        check(threw, "injected allocation failure at selected queue");
        const auto* preserved = cache.find(old_serial);
        check(cache.size() == seed_count && cache.bytes() == before_bytes && cache.superseded() == 0 &&
              cache.evictions() == 0 && preserved && preserved->bytes() == old_bytes &&
              preserved->live.ids == std::vector<int32_t>({10, 11, 12}) &&
              preserved->checkpoints.size() == 1 && preserved->checkpoints[0].ids == checkpoint.ids &&
              preserved->checkpoints[0].gdn == checkpoint.gdn,
              "failed publication preserves old payload, size and accounting");
        for (size_t i = 0; i < serials.size(); ++i)
            check(serials[i] != 0 && cache.index_of(serials[i]) == i, "failed publication preserves every serial and order");
        check(replacement.live.ids == std::vector<int32_t>({10, 11, 20}) &&
              replacement.checkpoints.size() == 1, "failed queue insertion leaves the replacement intact");
        const bool published = cache.put_exchanged(std::move(replacement));
        const auto inserted = cache.best(std::vector<int32_t>{10, 11, 20, 99}, {}, true);
        check(published && cache.size() == seed_count && cache.superseded() == 1 && cache.evictions() == 0 &&
              cache.bytes() == before_bytes - old_bytes + replacement_bytes && cache.find(old_serial) == nullptr &&
              inserted.serial == serials.back() + 1 && cache.find(inserted.serial) != nullptr,
              "successful replacement removes only the old entry and advances its serial once");
        for (size_t i = 1; i < serials.size(); ++i)
            check(cache.index_of(serials[i]) == i - 1, "successful replacement keeps unrelated entries in order");
    }
    {
        // Ordinary put keeps the baseline order: drop superseded, then refuse
        // before queue insertion when a pinned entry leaves too little room.
        ConversationCheckpoint checkpoint;
        checkpoint.ids = {10, 11}; checkpoint.gdn = {1, 2, 3};
        auto old = image({10, 11, 12});
        old.checkpoints.push_back(checkpoint);
        auto fixed = image({30000, 2, 3});
        fixed.live.gdn.resize(1u << 20, 7);
        ConversationCheckpoint pin;
        pin.ids = {30000, 2}; pin.pinned = true;
        fixed.checkpoints.push_back(pin);
        const size_t old_bytes = old.bytes(), fixed_bytes = fixed.bytes();
        const size_t other_bytes = image({10000, 2, 3}).bytes();
        const size_t budget = old_bytes + fixed_bytes + (seed_count - 2) * other_bytes;
        auto replacement = image({10, 11, 20});
        replacement.checkpoints.push_back(checkpoint);
        replacement.live.gdn.resize(old_bytes + seed_count * other_bytes + 1024, 7);
        const size_t replacement_bytes = replacement.bytes();
        check(replacement_bytes <= budget && fixed_bytes + replacement_bytes > budget,
              "ordinary refusal fixture fits alone but not beside the pinned entry");
        ConversationCache cache(budget, seed_count);
        std::vector<uint64_t> serials;
        check(cache.put(std::move(old)), "ordinary refusal: park superseded entry");
        serials.push_back(cache.best(std::vector<int32_t>{10, 11, 12, 90}, {}, true).serial);
        for (size_t i = 1; i + 1 < seed_count; ++i) {
            const int32_t id = 10000 + static_cast<int32_t>(i);
            check(cache.put(image({id, 2, 3})), "ordinary refusal: park unrelated entry");
            serials.push_back(cache.best(std::vector<int32_t>{id, 2, 3, 90}, {}, true).serial);
        }
        check(cache.put(std::move(fixed)), "ordinary refusal: park pinned entry");
        serials.push_back(cache.best(std::vector<int32_t>{30000, 2, 3, 90}, {}, true).serial);
        const size_t before_bytes = cache.bytes();
        check(cache.size() == seed_count && before_bytes == budget, "ordinary refusal: cache is full");
        bool accepted = false, threw = false;
        allocation_count = 0; count_allocations = true; fail_after = 0;
        try { accepted = cache.put(std::move(replacement)); }
        catch (const std::bad_alloc&) { threw = true; }
        fail_after = -1; count_allocations = false;
        check(!threw && !accepted && allocation_count == 0 && cache.size() == seed_count - 1 &&
              cache.find(serials.front()) == nullptr && cache.bytes() == before_bytes - old_bytes &&
              cache.superseded() == 1 && cache.evictions() == 0 && replacement.bytes() == replacement_bytes,
              "ordinary refusal drops the superseded entry before refusing without queue allocations");
        for (size_t i = 1; i < serials.size(); ++i)
            check(cache.index_of(serials[i]) == i - 1, "ordinary refusal preserves the other serials in order");
        (void) cache.take(cache.index_of(serials.back()));
        check(cache.can_fit(replacement_bytes) && cache.put(std::move(replacement)),
              "ordinary replacement can be retried after the pinned entry is taken");
        const auto inserted = cache.best(std::vector<int32_t>{10, 11, 20, 90}, {}, true);
        check(inserted.serial == serials.back() + 1, "ordinary refusal did not consume a serial");
    }
}
}

int main() {
    publication_allocation_failures();
    pool_reclaim_allocation_failure();
    {
        ConversationCache cache(1u << 20, 2);
        const std::vector<int32_t> live = {1, 2, 3, 4, 9};
        check(!cache.has_reuse_for(live), "no retained K/V does not block exchange");
        std::vector<ConversationKv> retained(1);
        retained[0].k.resize(64, 0x5a);
        cache.retain(std::move(retained), 5, {}, {1, 2, 3, 4, 5});
        cache.limit_reuse(4);
        const size_t before_bytes = cache.retained_bytes();
        check(cache.has_reuse_for(live) && cache.has_reuse_for(std::vector<int32_t>{1, 99}) &&
              !cache.has_reuse_for(std::vector<int32_t>{9, 1, 2}) &&
              !cache.has_reuse_for(std::vector<int32_t>{}),
              "only a positive common retained prefix blocks exchange");
        bool attempted = false;
        // Use the same gate as generate.cpp: the old park keeps its reusable prefix.
        if (!cache.has_reuse_for(live)) {
            attempted = true;
            (void) cache.can_fit(SIZE_MAX);
        }
        check(!attempted && !cache.can_fit(SIZE_MAX) && cache.retained_bytes() == before_bytes,
              "existing retained prefix bypasses exchange and survives budget refusal");
        auto reuse = cache.take_reuse_for(live);
        uint8_t first = 0, last = 0;
        check(reuse.unchanged_tokens == 4 && reuse.kv.size() == 1 && reuse.kv[0].k.size() == 64 &&
              reuse.kv[0].k.read(&first, 0, 1) && reuse.kv[0].k.read(&last, 63, 1) && first == 0x5a && last == 0x5a,
              "old capture path still takes the original reusable prefix and bytes");
        cache.retain(std::move(reuse.kv), 5, {}, {1, 2, 3, 4, 5});
        const std::vector<int32_t> one_prefix = {1, 99};
        check(cache.has_reuse_for(one_prefix), "one shared token blocks exchange");
        reuse = cache.take_reuse_for(one_prefix);
        check(reuse.unchanged_tokens == 1 && !reuse.kv.empty(), "one shared token is reusable by the old path");
        cache.retain(std::move(reuse.kv), 5, {}, {1, 2, 3, 4, 5});
        const std::vector<int32_t> unrelated = {9, 1, 2};
        check(!cache.has_reuse_for(unrelated), "unrelated retained K/V does not block exchange");
        reuse = cache.take_reuse_for(unrelated);
        check(reuse.unchanged_tokens == 0 && reuse.kv.empty(), "the old path also refuses an unrelated prefix");
        std::vector<ConversationKv> no_identity(1);
        no_identity[0].k.resize(64, 0x5a);
        cache.retain(std::move(no_identity), 5);
        check(cache.retained_tokens() > 0 && !cache.has_reuse_for(live),
              "retained K/V without identity does not block exchange");
    }
    {
        int mib = conversation_cache_swap_default_block_mib;
        check(mib == 0 && conversation_cache_swap_block_bytes(mib) == 0,
              "host exchange is disabled by default");
        for (int value = 0; value <= 256; ++value) {
            const std::string text = std::to_string(value);
            check(parse_conversation_cache_swap_block_mib(text, mib) && mib == value &&
                  conversation_cache_swap_block_bytes(mib) == (size_t(value) << 20),
                  "exchange block accepts every in-range decimal");
        }
        for (const auto& text : {"", "257", "-0", "-1", "+1", "1x", "1,2", "1.0", " 1", "1 ", "99999999999999999999999"}) {
            mib = 17;
            check(!parse_conversation_cache_swap_block_mib(text, mib) && mib == 17,
                  "exchange block rejects invalid or partial numbers without changing the option");
        }
    }
    {
        auto pattern_a = [](size_t i) { return uint8_t((i * 17 + 31) % 251); };
        auto pattern_b = [](size_t i) { return uint8_t((i * 29 + 87) % 251); };
        const size_t segment = ConversationBuffer::segment_bytes;
        for (const auto& [a_size, b_size, block] : std::vector<std::array<size_t,3>>{
                 {73,73,17}, {101,63,17}, {63,101,17}, {segment+53,segment+19,3*1024*1024+7},
                 {segment+19,segment+53,segment+5}}) {
            ConversationBuffer parked;
            parked.resize_for_overwrite(b_size);
            check(parked.visit(0,b_size,[&](uint8_t* p,size_t n,size_t at) {
                for (size_t i=0;i<n;++i) p[i]=pattern_b(at+i);
                return true;
            }), "initialize parked exchange image");
            uint8_t* original = nullptr;
            parked.visit(0,1,[&](uint8_t* p,size_t,size_t) { original=p; return true; });
            const size_t original_bytes = parked.bytes();
            const size_t peak = parked.allocation_peak(std::max(a_size,b_size));
            parked.resize_for_overwrite(std::max(a_size,b_size));
            check(parked.bytes() <= peak, "exchange growth stays within admitted capacity");
            if (a_size <= b_size) check(parked.bytes() == original_bytes, "exchange allocates no new K/V capacity when B is long enough");
            std::vector<uint8_t> active(std::max(a_size,b_size), 0);
            for (size_t i=0;i<a_size;++i) active[i]=pattern_a(i);
            std::vector<uint8_t> scratch(block);
            bool written = false;
            check(!conversation_buffer_exchange(parked,active.data(),a_size,b_size,nullptr,block,written) && !written,
                  "missing scratch fails before a write");
            check(conversation_buffer_exchange(parked,active.data(),a_size,b_size,scratch.data(),block,written) && written,
                  "exchange succeeds with one scratch block");
            parked.visit(0,1,[&](uint8_t* p,size_t,size_t) { check(p==original,"exchange reuses B's first allocation"); return true; });
            for (size_t i=0;i<b_size;++i) if (active[i]!=pattern_b(i)) check(false,"active bytes equal B");
            check(parked.visit(0,a_size,[&](const uint8_t* p,size_t n,size_t at) {
                for (size_t i=0;i<n;++i) if (p[i]!=pattern_a(at+i)) return false;
                return true;
            }), "parked bytes equal A across block and segment boundaries");
            written = false;
            check(conversation_buffer_exchange(parked,active.data(),b_size,a_size,scratch.data(),block,written) && written,
                  "reverse exchange succeeds");
            for (size_t i=0;i<a_size;++i) if (active[i]!=pattern_a(i)) check(false,"active bytes return to A");
            check(parked.visit(0,b_size,[&](const uint8_t* p,size_t n,size_t at) {
                for (size_t i=0;i<n;++i) if (p[i]!=pattern_b(at+i)) return false;
                return true;
            }), "parked bytes return to B");
        }
    }
    {
        // The second chunk crosses a deliberately small internal segment, and
        // later chunks start far beyond the five-byte scratch allocation.
        constexpr size_t first = 7, total = 83, block = 5;
        ConversationBuffer parked;
        parked.resize_for_overwrite(first);
        parked.resize_for_overwrite(total);
        std::vector<uint8_t> active(total), expected_a(total), expected_b(total), scratch(block);
        for (size_t i = 0; i < total; ++i) {
            expected_a[i] = uint8_t((i * 17 + 31) % 251);
            expected_b[i] = uint8_t((i * 29 + 87) % 251);
        }
        active = expected_a;
        size_t spans = 0;
        check(parked.visit(0, total, [&](uint8_t* p, size_t n, size_t at) {
            ++spans;
            std::memcpy(p, expected_b.data() + at, n);
            return true;
        }) && spans == 2, "small exchange crosses an internal segment");
        bool written = false;
        check(conversation_buffer_exchange(parked, active.data(), total, total, scratch.data(), block, written) && written,
              "small scratch exchanges at least three chunks across a segment");
        std::vector<uint8_t> actual(total);
        check(active == expected_b && parked.read(actual.data(), 0, total) && actual == expected_a,
              "small scratch keeps the offsets and contents of both images");
        std::array<uint8_t, 3> boundary{};
        check(parked.read(boundary.data(), first - 1, boundary.size()) &&
              std::equal(boundary.begin(), boundary.end(), expected_a.begin() + first - 1),
              "short read spans the internal segment boundary");
        std::array<uint8_t, 3> tail{};
        check(parked.read(tail.data(), total - tail.size(), tail.size()) &&
              std::equal(tail.begin(), tail.end(), expected_a.end() - tail.size()),
              "short read at a late segment offset uses a bounded destination");
        written = false;
        check(conversation_buffer_exchange(parked, active.data(), total, total, scratch.data(), block, written) && written &&
              active == expected_a && parked.read(actual.data(), 0, total) && actual == expected_b,
              "small scratch exchange round trip restores both images");
    }
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        // a parked conversation's K/V buffer (16 MiB, as captured) grows by one turn: the reserve for later turns is
        // an eighth of the buffer at most, not another 16 MiB segment
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 3);
        const size_t captured = bytes.bytes();
        check(captured - bytes.size() < 1024, "a fresh capture reserves nothing");
        const size_t turn = 1 << 20;
        const size_t peak = bytes.allocation_peak(bytes.size() + turn);
        bytes.resize(bytes.size() + turn, 4);
        check(bytes.bytes() <= peak, "a turn's growth stays within the predicted allocation peak");
        check(bytes.bytes() - captured <= turn + ConversationBuffer::segment_bytes / 8 + 1024,
              "the reserve for later turns is at most an eighth of the buffer");
        const size_t reserved = bytes.bytes();
        bytes.resize(bytes.size() + turn / 2, 5);
        check(bytes.bytes() == reserved, "the next small turn fills the reserve instead of allocating");
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments == 2, "the reserve extends the turn's segment");
    }
    {
        // many turns of 1 MiB onto a 16 MiB buffer: the reserve grows with the buffer, so the segments do not
        // multiply one per turn, and the reserve never exceeds an eighth of the buffer
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 1);
        for (int k = 0; k < 256; ++k) {
            const size_t before = bytes.size();
            const size_t peak = bytes.allocation_peak(before + (1 << 20));
            bytes.resize(before + (1 << 20), 2);
            check(bytes.bytes() <= peak, "each turn stays within the predicted allocation peak");
            check(bytes.bytes() - bytes.size() <= before / 8 + 4096, "the reserve stays within an eighth");
        }
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments <= 64, "256 turns of 1 MiB take far fewer than 256 segments");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    {
        ConversationCache cache(4096, 1);
        check(cache.put(image({1,2,3})), "park an unrelated conversation before exchange publication");
        auto outgoing = image({4,5,6});
        check(!cache.put_exchanged(std::move(outgoing)) && cache.size() == 1 && cache.evictions() == 0 &&
              outgoing.live.ids == std::vector<int32_t>({4,5,6}),
              "exchange publication refuses rather than evicting and keeps the outgoing image");
        (void) cache.take(0);
        check(cache.put_exchanged(std::move(outgoing)) && cache.size() == 1 && cache.evictions() == 0,
              "exchange publication stores the image when it fits");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true).tokens == 0 && cache.best(b, {}, true).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        const size_t pinned_bytes = cache.bytes();
        cache.put(image({9, 8, 7}));
        // #2253: what the RAM gate may count on: everything parked but the pinned conversation
        check(cache.evictable_bytes() == cache.bytes() - pinned_bytes && cache.evictable_bytes() == image({9, 8, 7}).bytes(),
              "#2253: evictable_bytes leaves out the conversation that holds a pinned prefix");
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(cache.evictable_bytes() == 0, "#2253: only a pinned one is left: nothing may go");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // holds(): what an idle batch slot holds is parked already (or a continuation of it is)
        auto s = image({1, 2, 3, 4, 5});
        ConversationCheckpoint cp;
        cp.ids = {1, 2, 3};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        check(cache.holds(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true), "the parked live state itself");
        check(cache.holds(std::vector<int32_t>{1, 2, 3, 4}, {}, true), "a prefix of the parked live state");
        check(cache.holds(std::vector<int32_t>{1, 2}, {}, true), "a prefix of a parked checkpoint");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3, 4, 5, 6}, {}, true), "a longer state is not held");
        check(!cache.holds(std::vector<int32_t>{1, 2, 9}, {}, true), "a diverging state is not held");
        check(!cache.holds(std::vector<int32_t>{}, {}, true), "an empty state is never held");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3}, {}, false), "steering mode is isolated");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3}, {{1, 77}}, true), "a slot with pictures is not held by a text image");
        auto p = image({1, 2, 3, 4});
        p.live.imgs = {{1, 77}, {3, 78}};
        cache.put(std::move(p));
        check(cache.holds(std::vector<int32_t>{1, 2, 3}, {{1, 77}}, true), "pictures below the state must match");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3, 4}, {{1, 77}}, true), "a picture inside the state that the slot lacks");
    }
    {
        // The 2026-10-14 four-conversation measurement replayed at policy level: two batch slots, four parked
        // entries, conversations A..D.  Each has a prompt P (8 tokens; the read's periodic checkpoint at 4), a first
        // reply R1 (3 tokens; the slot holds P + 2 of them), the next turn as the client renders it (P + R1 + end +
        // header: 14 tokens, the read's turn checkpoint at 12), a second reply R2 (4 tokens; the slot holds 17) and
        // a third turn (21 tokens).  The engine parks the MAIN session (the prompt as read, plus the admission's
        // token) whenever another conversation displaces it; the replies are decoded in the slots.  Measured:
        // A's third turn found 199584 of 199839 held tokens (its prompt-read state), B found nothing.
        //   slot_parks:   an idle slot parks what it holds before an admission overwrites it
        //   checks_back:  a slot that gives "all it holds" back to the main session gives its checkpoints with it,
        //                 so the chain still supersedes (#342) the parked copies a turn back
        auto replay = [&](bool slot_parks, bool checks_back) {
            std::vector<std::vector<int32_t>> full(4);
            for (int c = 0; c < 4; ++c) {
                for (int32_t t = 1; t <= 8; ++t) full[(size_t) c].push_back(c * 100 + t);
                for (int32_t t : {21, 22, 23, 9, 31, 32, 41, 42, 43, 44, 9, 51, 52}) full[(size_t) c].push_back(c * 100 + t);
            }
            auto st = [&](int c, size_t n) { return std::vector<int32_t>(full[(size_t) c].begin(), full[(size_t) c].begin() + (std::ptrdiff_t) n); };
            auto cp = [&](int c, size_t n) { ConversationCheckpoint k; k.ids = st(c, n); return k; };
            auto park = [&](ConversationCache& cache, int c, size_t n, const std::vector<size_t>& cks) {
                SavedConversation s = image({});
                s.live.ids = st(c, n);
                for (size_t k : cks) s.checkpoints.push_back(cp(c, k));
                check(cache.put(std::move(s)), "the replay's parks all fit the budget");
            };
            auto restore = [&](ConversationCache& cache, int c, size_t prompt) -> int64_t {
                const auto m = cache.best(st(c, prompt), {}, true);
                if (m.tokens > 0) cache.take(m.index);
                return m.tokens;
            };
            // the main chain after a slot gave "all it holds" back: its checkpoints, or none
            auto back = [&](const std::vector<size_t>& slot_cks) { return checks_back ? slot_cks : std::vector<size_t>{}; };
            auto slot_park = [&](ConversationCache& cache, int c, size_t n, const std::vector<size_t>& cks) {
                if (slot_parks && !cache.holds(st(c, n), {}, true)) park(cache, c, n, cks);
            };
            enum { A, B, C, D };
            ConversationCache cache(1 << 20, 4);
            // initial-AB: B solo then admitted (its prompt + the first token parked when A's request came), A
            // admitted directly, then alone: solo again from its slot (A's prompt-read state parked first)
            park(cache, B, 9, {4});
            park(cache, A, 8, {4});
            std::vector<size_t> chain_a = back({4});           // slot 1 gave back A's 10 tokens
            // resident1-AB: B's request displaces A's solo state; B's slot gives back 10; B admitted with the turn
            // checkpoint; A's request restores A's solo state from RAM and is admitted beside B
            park(cache, A, 12, chain_a);
            std::vector<size_t> chain_b = back({4});
            chain_b.push_back(12);
            // (a request takes its parked conversation before it parks the one it displaces)
            check(restore(cache, A, 14) == 12, "A's second turn resumes from its parked solo state");
            park(cache, B, 15, chain_b);
            chain_a.push_back(12);
            // both decode: slot 0 holds B's 17, slot 1 holds A's 17
            // initial-CD: C's request displaces A's main state; D admitted into slot 0 (B's reply), C into slot 1
            park(cache, A, 14, chain_a);
            slot_park(cache, B, 17, {12});
            park(cache, D, 9, {4});
            slot_park(cache, A, 17, {12});
            park(cache, C, 8, {4});                              // C alone: solo again from its slot
            std::vector<size_t> chain_c = back({4});
            // resident1-CD: C continues from the main session (no park), D's request displaces it (slot 1 held C's
            // own earlier state: covered, no slot park), D's slot gives back 10 (covered as well)
            chain_c.push_back(12);
            park(cache, C, 15, chain_c);
            std::vector<size_t> chain_d = back({4});
            chain_d.push_back(12);
            // both decode: slot 0 holds D's 17, slot 1 holds C's 17
            // exchange1-AB: A's request takes A's parked state, displaces D's main state, is admitted into slot 1
            // (C's reply); B's request displaces A's state
            const int64_t a_found = restore(cache, A, 21);
            park(cache, D, 14, chain_d);
            slot_park(cache, C, 17, {12});
            park(cache, A, 22, {12, 20});
            const int64_t b_found = restore(cache, B, 21);
            return std::pair<int64_t, int64_t>(a_found, b_found);
        };
        const auto baseline = replay(false, false);
        check(baseline.first == 14 && baseline.second == 0, "baseline: A returns to its prompt-read state, B to nothing (as measured)");
        const auto parks_only = replay(true, false);
        check(parks_only.first == 17 && parks_only.second == 0, "slot parks alone: A whole, B's copy evicted by a stale initial image");
        const auto both = replay(true, true);
        check(both.first == 17 && both.second == 17, "slot parks + checkpoints back: both return whole");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        // --conversation-cache-park-gain N: parked_prefix_tokens reports what a park now would supersede, so a
        // conversation that advanced fewer than N tokens since that copy keeps it instead of writing the whole
        // snapshot again for the next turn to drop.  It must name exactly the entries drop_superseded() would remove.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        check(cache.parked_prefix_tokens({1, 2, 3}, {}, {}, true) == 0, "an empty cache supersedes nothing");

        const std::vector<int32_t> live_ids = {1, 2, 3, 4, 60, 70};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        SavedConversation same = image({1, 2, 3, 4, 60, 70});
        same.checkpoints = chain;
        check(cache.put(std::move(same)), "park a conversation with a checkpoint chain");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, true) == 5,
              "the entry the outgoing chain supersedes is reported by its deepest checkpoint");

        const std::vector<ConversationCheckpoint> other_chain = {cp({1, 2, 3})};
        check(cache.parked_prefix_tokens({1, 2, 3, 40, 50}, {}, other_chain, true) == 0,
              "a conversation sharing only the root supersedes nothing");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, false) == 0,
              "the other steering mode is a different entry");
        check(cache.parked_prefix_tokens({1, 2, 3, 4, 60, 70, 71}, {}, chain, true) == 5,
              "an appended tail keeps the reported depth at the parked one");
        SavedConversation no_cp = image({1, 2, 3, 9});
        check(cache.put(std::move(no_cp)) && cache.parked_prefix_tokens({1, 2, 3, 9}, {}, {}, true) == 0,
              "an entry without checkpoints is never reported");
    }
    {
        // borrowing: a NEW conversation that starts with a checkpoint inside a parked one (a sibling subagent with
        // the same system prompt and tools) restores from it and leaves it parked; the same conversation going on
        // (its live state, or its deepest checkpoint after the client re-rendered the last reply) is still taken
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({});
        a.live.ids = {1, 2, 3, 4, 10, 11, 12, 13};
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        ConversationCache cache(1 << 20, 2);
        check(cache.put(std::move(a)), "park subagent A");
        const auto sib = cache.best(std::vector<int64_t>{1, 2, 3, 4, 20, 21}, {}, true);
        check(sib.tokens == 4 && !sib.live && sib.serial != 0 && cache.borrows(sib), "a sibling on A's root borrows");
        const auto again = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 30}, {}, true);
        check(again.tokens == 6 && !again.live && !cache.borrows(again),
              "A re-rendered at its deepest checkpoint is taken");
        const auto cont = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true);
        check(cont.live && !cache.borrows(cont), "A going on is taken");
        check(!cache.borrows(cache.best(std::vector<int64_t>{5, 6, 7}, {}, true)), "no match, nothing to borrow");
        // while the outgoing conversations are parked, the guarded donor is neither evicted nor dropped
        cache.guard(sib.serial);
        check(cache.put(image({7, 7, 7})) && cache.put(image({8, 8, 8})), "two outgoing conversations park");
        check(cache.find(sib.serial) != nullptr && cache.size() == 2 && cache.evictions() == 1 && !cache.guard_blocked(),
              "the slot limit evicts the other entry, not the guarded donor");
        check(cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true).tokens == 0, "the other one went");
        check(cache.evictable_bytes() == cache.bytes() - cache.find(sib.serial)->bytes(),
              "#2253: evictable_bytes leaves out the guarded donor");
        SavedConversation rewritten = image({1, 2, 3, 4, 10, 11, 40});   // holds A's deepest checkpoint: supersedes A
        rewritten.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        check(cache.put(std::move(rewritten)) && cache.find(sib.serial) != nullptr && cache.superseded() == 0,
              "a guarded donor is not dropped as superseded");
        cache.unguard();
        // identity survives index shifts: take an older entry, the donor's serial still finds it
        const auto donor_index = cache.index_of(sib.serial);
        check(donor_index != SIZE_MAX, "donor indexed by its serial");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true).serial == sib.serial,
              "matches report the entry's serial");
    }
    {
        // the outgoing conversation fits only in the guarded donor's room: make_room refuses before evicting anything
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({1, 2, 3, 4, 10, 11});
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        const size_t a_bytes = a.bytes();
        const size_t out_bytes = image({9, 9, 9}).bytes();
        ConversationCache tight(a_bytes + out_bytes - 1, 4);
        check(tight.put(std::move(a)), "park the donor");
        const auto m = tight.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true);
        check(tight.borrows(m), "a sibling would borrow");
        tight.guard(m.serial);
        check(!tight.put(image({9, 9, 9})) && tight.guard_blocked(), "no room beside the guarded donor: refused");
        check(tight.size() == 1 && tight.evictions() == 0, "nothing evicted by the refusal");
        tight.unguard();
        auto taken = tight.take(tight.index_of(m.serial));   // what the caller does then: take it, as before
        check(taken.live.ids.size() == 6 && taken.checkpoints.size() == 2 && tight.size() == 0,
              "the taken donor is whole");
        ConversationCache one_slot(1 << 20, 1);
        SavedConversation b = image({1, 2, 3, 4, 10, 11});
        b.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        check(one_slot.put(std::move(b)), "park in a single slot");
        const auto mb = one_slot.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true);
        one_slot.guard(mb.serial);
        check(!one_slot.put(image({9, 9, 9})) && one_slot.guard_blocked() && one_slot.size() == 1,
              "one slot: the guarded donor blocks, nothing evicted");
        one_slot.unguard();
    }
    {
        // pin=N beside a borrow: the guarded donor and a conversation holding a pinned shared prefix both stay; the
        // oldest other entry is the one evicted (the case midhatn tested on #1164)
        auto cp = [](std::vector<int32_t> ids, bool pinned = false) {
            ConversationCheckpoint c; c.ids = std::move(ids); c.pinned = pinned; return c;
        };
        SavedConversation doc = image({1, 2, 3, 4, 5, 6, 30, 31});
        doc.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        SavedConversation donor = image({1, 2, 10, 11, 12});
        donor.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        ConversationCache cache(1 << 20, 3);
        check(cache.put(std::move(doc)) && cache.put(std::move(donor)) && cache.put(image({7, 7, 7})),
              "pinned document, donor and an ordinary conversation parked");
        const auto m = cache.best(std::vector<int64_t>{1, 2, 40, 41}, {}, true);
        check(cache.borrows(m), "a sibling on the donor's root borrows");
        cache.guard(m.serial);
        check(cache.put(image({8, 8, 8})) && !cache.guard_blocked() && cache.size() == 3 && cache.evictions() == 1,
              "the outgoing conversation parks");
        check(cache.find(m.serial) != nullptr &&
              cache.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 9}, {}, true).tokens == 6 &&
              cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true).tokens == 0,
              "only the ordinary entry was evicted: the donor and the pinned document stay");
        check(cache.put(image({9, 9, 9})) && cache.size() == 3 && cache.evictions() == 2 &&
              cache.find(m.serial) != nullptr && cache.best(std::vector<int64_t>{8, 8, 8, 1}, {}, true).tokens == 0,
              "the next outgoing one evicts the previous outgoing one, still not the donor or the document");
        cache.unguard();
        ConversationCache two(1 << 20, 2);
        SavedConversation doc2 = image({1, 2, 3, 4, 5, 6, 30});
        doc2.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true)};
        SavedConversation donor2 = image({1, 2, 10, 11, 12});
        donor2.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        check(two.put(std::move(doc2)) && two.put(std::move(donor2)), "pinned document and donor fill two slots");
        const auto m2 = two.best(std::vector<int64_t>{1, 2, 40}, {}, true);
        two.guard(m2.serial);
        check(!two.put(image({9, 9, 9})) && two.guard_blocked() && two.size() == 2 && two.evictions() == 0,
              "no slot beside the donor and the pinned document: refused before evicting anything");
        two.unguard();
        // a match that reaches the pinned prefix is taken, not borrowed: one conversation keeps holding the pin
        const auto on_pin = two.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 77}, {}, true);
        check(on_pin.tokens == 6 && !on_pin.live && !two.borrows(on_pin), "a pinned prefix is taken, not borrowed");
        ConversationCache three(1 << 20, 3);
        SavedConversation doc3 = image({1, 2, 3, 4, 5, 6, 30, 31, 32});
        doc3.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30}), cp({1, 2, 3, 4, 5, 6, 30, 31})};
        check(three.put(std::move(doc3)), "a conversation past its pinned prefix is parked");
        const auto past_pin = three.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 30, 99}, {}, true);
        check(past_pin.tokens == 7 && !three.borrows(past_pin),
              "a match past the pinned prefix (the copy would carry the pin) is taken, not borrowed");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                p.dev_tag = uint64_t(salt) * 100 + k + 1;
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.parts[0][0].dev_tag == 301 && split.parts[1][1].dev_tag == 702,
              "split: each stage part keeps the tag of its device copy");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn &&
                       back[j].stage_parts[k].dev_tag == original[j].stage_parts[k].dev_tag;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    {
        // #1503, the cross-conversation leak: the retained K/V of a restored conversation must never be reused to
        // park ANOTHER conversation the main session later holds (a batch slot gave its conversation back)
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<std::vector<ConversationKv>> st;
        st.push_back(kv(100));
        cache.retain(kv(500), 16, std::move(st), a);
        std::vector<int32_t> b(16, 99);
        auto other = cache.take_reuse_for(b);
        check(other.kv.empty() && other.stages.empty() && other.unchanged_tokens == 0,
              "another conversation never gets the retained K/V");
        check(cache.retained_bytes() == 0, "the retained K/V is dropped once refused");
        cache.retain(kv(500), 16, {}, a);
        check(cache.retained_tokens() == 16, "the retained extent is reported");
        std::vector<int32_t> cont(a.begin(), a.begin() + 10);
        cont.push_back(42);
        cont.push_back(43);
        auto partly = cache.take_reuse_for(cont);
        check(partly.unchanged_tokens == 10 && !partly.kv.empty(), "a conversation that diverges reuses only the shared prefix");
        std::vector<std::vector<ConversationKv>> st2;
        st2.push_back(kv(100));
        cache.retain(kv(500), 16, std::move(st2), a);
        cache.limit_reuse(9);
        auto same = cache.take_reuse_for(a);
        check(same.unchanged_tokens == 9 && same.stages.size() == 1 && same.stages[0].unchanged_tokens == 9,
              "the same conversation keeps the first rewrite's limit, on every stage");
        cache.retain(kv(500), 16);
        check(cache.take_reuse_for(a).kv.empty(), "retained K/V of unknown identity is never reused");

        // the sequence of the report: P restored (retained), a slot hands Q to the main session, Q is parked.  Q shares
        // only the system prompt with P (4 tokens): with the slot's drop, nothing is reused; without it (the ids check
        // alone), at most the shared system prompt
        const std::vector<int32_t> p = {7, 7, 7, 7, 100, 101, 102, 103, 104, 105};
        const std::vector<int32_t> q = {7, 7, 7, 7, 200, 201, 202, 203, 204, 205, 206, 207};
        std::vector<std::vector<ConversationKv>> stp;
        stp.push_back(kv(100));
        cache.retain(kv(500), 10, std::move(stp), p);
        (void) cache.take_reuse();   // copy_from_slot
        check(cache.take_reuse_for(q).kv.empty(), "a slot that gave its conversation back drops the retained K/V");
        std::vector<std::vector<ConversationKv>> stp2;
        stp2.push_back(kv(100));
        cache.retain(kv(500), 10, std::move(stp2), p);
        auto sys_only = cache.take_reuse_for(q);
        check(sys_only.unchanged_tokens == 4 && sys_only.stages.size() == 1 && sys_only.stages[0].unchanged_tokens == 4,
              "without the drop, only the shared system prompt is reused, on every stage");
        check(cache.retained_bytes() == 0, "taken once");
        // a conversation that is a strict extension of the retained one keeps the whole extent
        cache.retain(kv(500), 10, {}, p);
        std::vector<int32_t> p_more = p;
        p_more.push_back(300);
        check(cache.take_reuse_for(p_more).unchanged_tokens == 10, "the conversation's own continuation reuses all of it");
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "a rewrite from the start drops every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
        // #5697: an image that cannot fit beside the pinned entry even with every other entry gone is refused BEFORE
        // anything is evicted: the two ordinary conversations stay parked (before, both were evicted, then refused)
        auto p3 = image({1,2,3});
        p3.checkpoints.push_back(ConversationCheckpoint{});
        p3.checkpoints.back().ids = {1,2};
        p3.checkpoints.back().pinned = true;
        p3.checkpoints.back().gdn.resize(64, 1);
        ConversationCache keep(pinned_bytes + one * 2, 4);
        check(keep.put(std::move(p3)) && keep.put(image({7,8,9})) && keep.put(image({4,5,6})) && keep.size() == 3,
              "a pinned conversation and two ordinary ones are parked");
        check(!keep.make_room(one * 2 + 1) && keep.size() == 3 && keep.evictions() == 0,
              "a park too big for the room beside the pin is refused with nothing evicted");
        check(keep.best(std::vector<int32_t>{7,8,9,1}, {}, true).tokens == 3 && keep.best(std::vector<int32_t>{4,5,6,1}, {}, true).tokens == 3,
              "both ordinary conversations can still be restored");
        check(keep.make_room(one * 2) && keep.size() == 1 && keep.evictions() == 2,
              "a park that fits once both go evicts both (as before)");
        ConversationCache slots(pinned_bytes * 4 + one * 4, 1);
        auto p4 = image({1,2,3});
        p4.checkpoints.push_back(ConversationCheckpoint{});
        p4.checkpoints.back().ids = {1,2};
        p4.checkpoints.back().pinned = true;
        p4.checkpoints.back().gdn.resize(64, 1);
        check(slots.put(std::move(p4)) && !slots.make_room(one) && slots.size() == 1 && slots.evictions() == 0,
              "one slot, held by a pin: refused, nothing evicted");
    }
    {   // P2: an evicted conversation goes to the sink (the SSD tier), whole; superseded copies and clear() do not
        const size_t one = image({1,2,3}).bytes();
        ConversationCache c(one * 2, 4);
        std::vector<SavedConversation> sunk;
        c.set_evict_sink([&](SavedConversation&& s) { sunk.push_back(std::move(s)); });
        check(c.put(image({1,2,3})) && c.put(image({4,5,6})), "two parked");
        check(c.put(image({7,8,9})) && sunk.size() == 1 && sunk[0].live.ids == std::vector<int32_t>({1,2,3}) &&
              sunk[0].live.gdn == std::vector<uint8_t>(64, 7), "the budget's eviction hands the oldest over whole");
        check(c.evict_oldest() && sunk.size() == 2 && sunk[1].live.ids == std::vector<int32_t>({4,5,6}),
              "the RAM gate's eviction too");
        check(c.size() == 1 && c.evictions() == 2, "the cache keeps its own count as before");
        auto chained = image({7,8,9,10});
        chained.checkpoints.push_back(ConversationCheckpoint{});
        chained.checkpoints.back().ids = {7,8,9};
        check(c.drop_superseded(chained.live.ids, chained.live.imgs, chained.checkpoints, true) == 0 && sunk.size() == 2,
              "(a parked entry with no checkpoint is never superseded)");
        c.put(image({11,12,13}));
        const size_t before = sunk.size();
        check(c.clear() > 0 && sunk.size() == before, "#879's clear() hands nothing over");
        ConversationCache plain(one, 4);
        check(plain.put(image({1,2,3})) && plain.put(image({4,5,6})) && plain.size() == 1, "without a sink: evicted as before");
    }
    {
        // #4420: a batch slot's chain holds one checkpoint (its admission's): parking it must not drop (#342) the main session's copy of the
        // same conversation that also holds the root a sibling borrows from (an agent's system prompt and tools) - supersedes_richer()
        // names that case, so the slot is not parked
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation main_copy = image({1, 2, 3, 4, 10, 11, 12});
        main_copy.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        ConversationCache cache(1 << 20, 4);
        check(cache.put(std::move(main_copy)), "park the main session's copy with its root and turn checkpoints");
        const std::vector<int32_t> slot_ids{1, 2, 3, 4, 10, 11, 12, 20, 21};          // the slot decoded the reply
        const std::vector<ConversationCheckpoint> slot_chain{cp({1, 2, 3, 4, 10, 11})};   // the admission's only one
        check(cache.supersedes_richer(slot_ids, {}, slot_chain, true), "the slot's park would drop the root checkpoint");
        const std::vector<ConversationCheckpoint> full_chain{cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        check(!cache.supersedes_richer(slot_ids, {}, full_chain, true), "a chain that holds both loses nothing");
        check(!cache.supersedes_richer(slot_ids, {}, slot_chain, false), "steering mode is isolated");
        check(!cache.supersedes_richer(std::vector<int32_t>{1, 2, 3, 4, 30, 31}, {},
                                       std::vector<ConversationCheckpoint>{cp({1, 2, 3, 4, 30})}, true),
              "another conversation on the same root supersedes nothing");
        check(cache.size() == 1 && cache.superseded() == 0, "the predicate changes nothing");
        const auto sib = cache.best(std::vector<int64_t>{1, 2, 3, 4, 40}, {}, true);
        check(sib.tokens == 4 && cache.borrows(sib), "a sibling still borrows the root");
        // what drop_superseded does with the slot's chain (the park this rule prevents): the root goes
        check(cache.drop_superseded(slot_ids, {}, slot_chain, true) == 1 && cache.size() == 0,
              "the slot's chain supersedes the richer copy");
    }
    {
        // STRATA_PARK_SEGPOOL_MIB: the segment pool's idle bytes count against the budget beside the images, are given back
        // first, and never change what the cache evicts, refuses or parks
        struct FakePool final : SegmentPoolBudget {
            size_t idle = 0, shrinks = 0, shrunk = 0;
            size_t idle_bytes() const override { return idle; }
            size_t shrink(size_t want) override { ++shrinks; const size_t g = std::min(want, idle); idle -= g; shrunk += g; return g; }
        };
        auto sized = [](int32_t id, size_t bytes) { SavedConversation s = image({id}); s.live.gdn.resize(bytes, 7); return s; };
        {
            FakePool pool;
            ConversationCache cache(10000, 8);
            cache.set_pool(&pool);
            check(cache.put(sized(1, 4000)) && cache.put(sized(2, 4000)), "two images of about 4000 bytes");
            const size_t images = cache.bytes();
            pool.idle = 1800;
            check(cache.bytes() == images && cache.pool_idle_bytes() == 1800 && cache.resident_bytes() == images + 1800,
                  "bytes() is the images'; resident_bytes() adds the pool's idle bytes");
            check(images + 1800 <= 10000, "(the images and the idle bytes fit the budget)");
            const size_t room = 10000 - images - 400;   // an incoming image that fits beside the images only if idle bytes go
            check(cache.make_room(room) && cache.size() == 2 && cache.evictions() == 0, "an image that fits beside the images needs no eviction, the idle bytes are the ones that go");
            check(pool.shrinks >= 1 && pool.shrunk == 1800 - 400 && pool.idle == 400, "the pool gave back exactly the excess");
            check(cache.resident_bytes() + room <= 10000, "images + idle + the incoming image fit the budget");
            pool.idle = 9000;   // far more idle than the budget leaves: all of it is shrinkable, nothing else is touched
            check(cache.make_room(500) && cache.size() == 2 && cache.evictions() == 0 && cache.resident_bytes() + 500 <= 10000,
                  "a pool holding more than the budget leaves is trimmed to fit, no image goes");
            check(cache.make_room(5500) && cache.size() == 1 && cache.evictions() == 1,
                  "an incoming image that does not fit beside the images evicts the oldest, as without a pool");
        }
        {
            // the same random workload against a cache with no pool and one whose pool holds a different idle amount before every step:
            // every answer, size, eviction count and byte count is the same, and the pool's bytes never push the total past the budget
            struct Lcg { uint64_t s; uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 33; } };
            Lcg rng{12345};
            const size_t budget = 24000;
            ConversationCache plain(budget, 5), pooled(budget, 5);
            FakePool pool;
            pooled.set_pool(&pool);
            int32_t id = 100;
            size_t refused = 0, evicted_any = 0;
            for (int step = 0; step < 4000; ++step) {
                pool.idle = size_t(rng.next() % (budget / 2));
                const uint64_t op = rng.next() % 4;
                if (op <= 1) {
                    const size_t n = 300 + size_t(rng.next() % 9000);
                    const bool a = plain.put(sized(id, n)), b = pooled.put(sized(id, n));
                    ++id;
                    check(a == b, "put answers the same with and without a pool");
                    refused += !a;
                } else if (op == 2) {
                    const size_t incoming = size_t(rng.next() % (budget + 3000)), held = size_t(rng.next() % 3) == 0 ? size_t(rng.next() % 4000) : 0;
                    const bool a = plain.make_room(incoming, held), b = pooled.make_room(incoming, held);
                    check(a == b, "make_room answers the same with and without a pool");
                    refused += !a;
                    if (b) check(pooled.resident_bytes() + incoming + held <= budget, "a successful make_room leaves images + idle + incoming inside the budget");
                } else if (plain.size() > 0) {
                    const size_t at = size_t(rng.next() % plain.size());
                    SavedConversation x = plain.take(at), y = pooled.take(at);
                    check(x.live.ids == y.live.ids, "take returns the same entry");
                }
                check(plain.size() == pooled.size() && plain.evictions() == pooled.evictions() && plain.bytes() == pooled.bytes(),
                      "entries, evictions and bytes are the same with and without a pool");
                evicted_any += plain.evictions() > 0;
            }
            check(refused > 0 && evicted_any > 0 && pool.shrinks > 0, "the workload refused, evicted and trimmed the pool at least once");
        }
    }
    {
        // #2518: a slot carries every checkpoint that prefixes the prompt, not only the longest one.
        auto chain = [](int32_t n) {
            ConversationCheckpoint c;
            c.ids.resize((size_t) n);
            for (int32_t i = 0; i < n; ++i) c.ids[(size_t) i] = 100 + i;
            c.gdn.assign(3, (uint8_t) n);        // distinguishable, so a copy is recognisable
            return c;
        };
        std::vector<ConversationCheckpoint> checks{chain(4), chain(8), chain(12)};
        std::vector<int32_t> live = chain(12).ids;
        live.push_back(112);                     // the prompt goes on past the deepest checkpoint

        std::vector<ConversationCheckpoint> kept = conversation_checkpoints_prefixing(checks, live, false);
        check(kept.size() == 3, "#2518: every checkpoint that prefixes the prompt is carried, not only the longest");
        check(kept[0].ids.size() == 4 && kept[1].ids.size() == 8 && kept[2].ids.size() == 12,
              "#2518: the carried checkpoints keep the order and the lengths they had");
        check(kept[2].gdn == checks[2].gdn && kept[2].gdn.data() != checks[2].gdn.data(),
              "#2518: a carried checkpoint is a copy of the state, not a reference to it");

        std::vector<int32_t> ends_at_eight = chain(8).ids;
        std::vector<ConversationCheckpoint> shorter = conversation_checkpoints_prefixing(checks, ends_at_eight, false);
        check(shorter.size() == 1 && shorter[0].ids.size() == 4,
              "#2518: a checkpoint as long as the prompt is not a prefix of it");

        ConversationCheckpoint diverging = chain(8);
        diverging.ids[6] = -7;
        std::vector<ConversationCheckpoint> mixed{chain(4), diverging};
        check(conversation_checkpoints_prefixing(mixed, live, false).size() == 1,
              "#2518: a checkpoint that diverges from the prompt is not carried");

        check(conversation_checkpoints_prefixing(checks, live, true).empty(),
              "#2518: an image prompt carries no checkpoints (its tokens do not match again)");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
