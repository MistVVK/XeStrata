// The segment pool (STRATA_PARK_SEGPOOL_MIB): the arena logic behind the OS interface, tested with a fake OS that counts its calls.
#include "strata/core/conversation_segment_pool.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr size_t MiB = size_t(1) << 20;

// Real memory from the C library, 2 MiB aligned; release wipes the range (a released range loses its content, like madvise)
struct FakeOs final : SegmentPoolOs {
    size_t reserved = 0, commits = 0, populates = 0, populated_bytes = 0, releases = 0, released_bytes = 0;
    bool refuse_commit = false, refuse_populate = false, refuse_reserve = false;
    void* mem = nullptr;
    void* reserve(size_t bytes) override {
        if (refuse_reserve) return nullptr;
        mem = std::aligned_alloc(size_t(2) << 20, bytes);
        if (mem) { std::memset(mem, 0, bytes); reserved = bytes; }
        return mem;
    }
    void unreserve(void* base, size_t) override { std::free(base); }
    bool commit(void*, size_t) override { ++commits; return !refuse_commit; }
    bool populate(void*, size_t n) override { if (refuse_populate) return false; ++populates; populated_bytes += n; return true; }
    void release(void* p, size_t n) override { ++releases; released_bytes += n; std::memset(p, 0, n); }
};
}

int main() {
    // an inert pool: too small to reserve, or the OS refuses
    {
        FakeOs os;
        SegmentPool tiny(os, 1 * MiB);
        check(!tiny.active() && tiny.capacity() == 0, "a pool under 2 MiB is inert");
        check(tiny.allocate(MiB) == nullptr, "an inert pool serves nothing");
        int x = 0;
        check(!tiny.deallocate(&x, 16), "an inert pool owns no pointer");
        FakeOs refusing;
        refusing.refuse_reserve = true;
        SegmentPool none(refusing, 64 * MiB);
        check(!none.active(), "a refused reserve is an inert pool");
    }
    // take / give: sizes, the minimum, the capacity rounding, the same range again
    {
        FakeOs os;
        SegmentPool pool(os, 64 * MiB);
        check(pool.active() && pool.capacity() == 64 * MiB, "capacity is the reserved bytes");
        check(pool.allocate(SegmentPool::min_alloc - 1) == nullptr, "below 512 KiB is not served");
        void* a = pool.allocate(SegmentPool::min_alloc);
        check(a != nullptr && pool.contains(a), "512 KiB is served from the region");
        check(pool.live_requested() == SegmentPool::min_alloc, "live bytes are the bytes asked for");
        void* b = pool.allocate(14 * MiB + 1);
        check(b != nullptr && static_cast<char*>(b) >= static_cast<char*>(a) + SegmentPool::min_alloc, "a second range does not overlap the first");
        check(pool.deallocate(a, SegmentPool::min_alloc), "a pool range is given back to the pool");
        void* a2 = pool.allocate(SegmentPool::min_alloc);
        check(a2 == a, "a freed range is the next best fit");
        check(pool.deallocate(a2, SegmentPool::min_alloc) && pool.deallocate(b, 14 * MiB + 1), "all given back");
        check(pool.live_requested() == 0, "nothing live after every range is given back");
        int outside = 0;
        check(!pool.deallocate(&outside, 16), "a foreign pointer is not the pool's");
    }
    // a request larger than the pool, or than any free run, is not served (the caller falls back to the heap) and is counted
    {
        FakeOs os;
        SegmentPool pool(os, 8 * MiB);
        check(pool.allocate(16 * MiB) == nullptr, "a request larger than the pool is a miss");
        void* a = pool.allocate(6 * MiB);
        check(a != nullptr, "6 MiB of an 8 MiB pool");
        check(pool.allocate(4 * MiB) == nullptr, "4 MiB does not fit beside it");
        check(pool.misses() == 2 && pool.hits() == 1, "misses and hits are counted");
        check(pool.allocate(2 * MiB) != nullptr, "the 2 MiB that is left");
    }
    // best fit and coalescing
    {
        FakeOs os;
        SegmentPool pool(os, 32 * MiB);
        void* s[8];
        for (int i = 0; i < 8; ++i) s[i] = pool.allocate(4 * MiB);
        check(s[7] != nullptr, "eight 4 MiB ranges fill the pool");
        check(pool.allocate(SegmentPool::min_alloc) == nullptr, "full");
        pool.deallocate(s[1], 4 * MiB);          // a 4 MiB hole
        pool.deallocate(s[3], 4 * MiB);          // 4 MiB hole, then s[4] and s[5] freed make it 12 MiB
        pool.deallocate(s[4], 4 * MiB);
        pool.deallocate(s[5], 4 * MiB);
        void* small = pool.allocate(3 * MiB);    // best fit: the 4 MiB hole at s[1], not the 12 MiB one
        check(small == s[1], "best fit takes the smallest hole that fits");
        void* big = pool.allocate(12 * MiB);
        check(big == s[3], "freed neighbours coalesce into one 12 MiB run");
        pool.deallocate(small, 3 * MiB);
        pool.deallocate(big, 12 * MiB);
    }
    // bytes written to a range survive the neighbours' allocation and release; ranges never overlap
    {
        FakeOs os;
        SegmentPool pool(os, 64 * MiB);
        std::vector<std::pair<uint8_t*, size_t>> held;
        for (int i = 0; i < 12; ++i) {
            const size_t n = size_t(600 + 311 * i) * 1024;
            uint8_t* p = static_cast<uint8_t*>(pool.allocate(n));
            check(p != nullptr, "ranges of mixed sizes");
            std::memset(p, 0x10 + i, n);
            held.push_back({p, n});
        }
        for (int i = 0; i < 12; i += 2) { pool.deallocate(held[i].first, held[i].second); }
        for (int i = 1; i < 12; i += 2)
            for (size_t k = 0; k < held[i].second; k += 4099)
                check(held[i].first[k] == uint8_t(0x10 + i), "a live range keeps its bytes while its neighbours are freed");
        std::set<std::pair<const uint8_t*, const uint8_t*>> spans;
        for (int i = 1; i < 12; i += 2) spans.insert({held[i].first, held[i].first + held[i].second});
        const uint8_t* end = nullptr;
        for (const auto& s : spans) { check(end == nullptr || s.first >= end, "live ranges do not overlap"); end = s.second; }
    }
    // residency accounting: populate, idle, shrink; idle + live = resident
    {
        FakeOs os;
        SegmentPool pool(os, 64 * MiB);
        check(pool.resident_bytes() == 0 && pool.idle_bytes() == 0 && pool.cold_free_bytes() == 64 * MiB, "a new pool is cold");
        check(pool.populate_step(100 * MiB) == 32 * MiB, "a populate step is at most 32 MiB");
        check(pool.populate_step(1 * MiB) == 1 * MiB && pool.resident_bytes() == 33 * MiB, "a smaller step is honoured");
        check(pool.idle_bytes() == 33 * MiB, "everything resident is idle while nothing is live");
        void* a = pool.allocate(10 * MiB);
        check(a != nullptr && pool.live_requested() == 10 * MiB, "a live range");
        check(pool.idle_bytes() == pool.resident_bytes() - 10 * MiB, "idle = resident minus the live bytes asked for");
        void* far = pool.allocate(40 * MiB);   // needs the cold granules too: counted resident once used
        check(far != nullptr && os.commits >= 1, "a range over cold granules commits them");
        check(pool.resident_bytes() >= 50 * MiB && pool.idle_bytes() == pool.resident_bytes() - 50 * MiB, "used cold granules count as resident");
        pool.deallocate(far, 40 * MiB);
        const size_t idle = pool.idle_bytes();
        const size_t gave = pool.shrink(8 * MiB);
        check(gave >= 8 * MiB && gave < 8 * MiB + SegmentPool::granule, "shrink gives back what it was asked for, to the granule");
        check(pool.idle_bytes() == idle - gave && os.released_bytes == gave, "idle falls by what was released, the OS saw the release");
        check(pool.shrink(1000 * MiB) > 0 && pool.idle_bytes() == 0 && pool.live_requested() == 10 * MiB, "shrinking everything leaves only the live range");
        check(pool.resident_bytes() >= 10 * MiB, "the live range stays resident");
        pool.deallocate(a, 10 * MiB);
        check(pool.idle_bytes() == pool.resident_bytes(), "after the last range is given back all that is resident is idle");
    }
    // a refused commit or populate changes nothing
    {
        FakeOs os;
        SegmentPool pool(os, 16 * MiB);
        os.refuse_commit = true;
        check(pool.allocate(4 * MiB) == nullptr && pool.live_requested() == 0 && pool.resident_bytes() == 0, "a refused commit is a miss with no state change");
        os.refuse_commit = false;
        os.refuse_populate = true;
        check(pool.populate_step(4 * MiB) == 0 && pool.populate_failed() && pool.resident_bytes() == 0, "a refused populate makes nothing resident");
        check(pool.allocate(4 * MiB) != nullptr, "the granules stayed usable after a refused populate");
    }
    // the background filler: tops the pool up to what permit allows, nothing when it allows nothing, stops at destruction
    {
        FakeOs os;
        SegmentPool pool(os, 128 * MiB);
        std::atomic<size_t> cap{40 * MiB};   // the most the pool may hold resident
        std::atomic<int> started{0};
        {
            SegmentPoolFiller filler(pool, [&](size_t step) {
                const size_t have = pool.resident_bytes();
                return cap.load() > have ? std::min(step, cap.load() - have) : size_t(0);
            }, [&] { ++started; });
            auto settle = [&] {
                for (int i = 0; i < 5000 && filler.busy(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                check(!filler.busy(), "the filler finished its pass");
            };
            filler.kick();
            settle();
            check(pool.resident_bytes() == 40 * MiB, "the filler stops at what is permitted");
            check(started.load() == 1, "the thread's start hook ran once");
            cap = 128 * MiB;
            filler.kick();
            settle();
            check(pool.resident_bytes() == 128 * MiB, "with the whole pool permitted the filler reaches it");
            pool.shrink(64 * MiB);
            cap = 0;
            filler.kick();
            settle();
            check(pool.resident_bytes() == 64 * MiB, "a zero permit populates nothing");
            cap = 100 * MiB;
            filler.kick();
            settle();
            check(pool.resident_bytes() == 100 * MiB, "after a shrink and a larger permit the pool is topped up to it");
        }
    }
    // the process-wide hook: installed once
    {
        FakeOs os;
        SegmentPool pool(os, 8 * MiB);
        check(segment_pool() == nullptr, "no process-wide pool at first");
        check(segment_pool_install(&pool) && segment_pool() == &pool, "installed");
        SegmentPool other(os, 8 * MiB);
        check(!segment_pool_install(&other) && segment_pool() == &pool, "a second install is refused");
        segment_pool_uninstall_for_test();
        check(segment_pool() == nullptr, "uninstalled by the test");
    }
    std::printf("conversation_segment_pool_test: %d checks passed\n", checks);
    return 0;
}
