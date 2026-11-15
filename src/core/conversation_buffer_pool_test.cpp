// ConversationBuffer on the segment pool (STRATA_PARK_SEGPOOL_MIB): a real-sized 2K and 64K K/V image alternating, a pool smaller than
// the request, and the buffers' bytes with the pool on and off.  Host only; the OS side is a fake that counts calls.
#include "strata/core/conversation_buffer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

using namespace strata::core;

namespace {
// 2 MiB aligned memory from the C library: MSVC's C runtime has no std::aligned_alloc (its free cannot release it), so _aligned_malloc there
void* aligned_2m(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, size_t(2) << 20);
#else
    return std::aligned_alloc(size_t(2) << 20, bytes);
#endif
}
void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr size_t MiB = size_t(1) << 20;

struct FakeOs final : SegmentPoolOs {
    void* reserve(size_t bytes) override { return aligned_2m(bytes); }
    void unreserve(void* base, size_t) override { aligned_free(base); }
    bool commit(void*, size_t) override { return true; }
    bool populate(void*, size_t) override { return true; }
    void release(void* p, size_t n) override { std::memset(p, 0, n); }
};

// One conversation's K/V host image at Qwen3.8-Flash-Next's int8 sizes (13 layers: 12 QSA + the draft layer; two K/V heads of 256):
// k and v cells x 512 B, the two scale buffers cells x 32 B, the pooled indexer rows (one 128-float row per 128 tokens)
struct Image {
    std::vector<ConversationBuffer> buffers;   // 5 per layer
    size_t bytes = 0;
    explicit Image(size_t tokens, uint8_t tag, bool fill = true) {
        const size_t cells = (tokens + 63) / 64 * 64;
        const size_t sizes[5] = {cells * 512, cells * 512, cells * 32, cells * 32, (tokens / 128 + 1) * 128 * 4};
        buffers.resize(13 * 5);
        for (size_t layer = 0; layer < 13; ++layer)
            for (size_t part = 0; part < 5; ++part) {
                ConversationBuffer& b = buffers[layer * 5 + part];
                b.resize_for_overwrite(sizes[part]);
                bytes += sizes[part];
                if (fill)
                    b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t at) {
                        std::memset(p, int(tag + layer * 5 + part), n);
                        (void) at;
                        return true;
                    });
            }
    }
    bool intact(uint8_t tag) const {
        for (size_t i = 0; i < buffers.size(); ++i) {
            bool ok = true;
            buffers[i].visit(0, buffers[i].size(), [&](const uint8_t* p, size_t n, size_t) {
                for (size_t k = 0; k < n; k += 997) ok = ok && p[k] == uint8_t(tag + i);
                return ok;
            });
            if (!ok) return false;
        }
        return true;
    }
};
}

int main() {
    FakeOs os;
    // pool off: every segment is a plain allocation (nothing pooled), the sizes are what they were
    {
        check(segment_pool() == nullptr, "no pool is installed");
        Image img(61225, 1);
        check(img.intact(1) && img.bytes > 700 * MiB && img.bytes < 900 * MiB, "a 64K image is about 0.83 GB (13 layers)");
    }
    SegmentPool pool(os, 1024 * MiB);
    check(pool.active() && segment_pool_install(&pool), "a 1 GiB pool is installed");
    // a 2K and a 64K conversation alternating: the second 64K image is served entirely from the pool
    {
        auto a = std::make_unique<Image>(1931, 10);          // 2K
        const uint64_t after_2k = pool.hits();
        check(after_2k > 0 && pool.live_requested() > 0, "the 2K image's large buffers come from the pool");
        check(pool.live_requested() < a->bytes, "its small buffers (scales, pooled rows) stay on the heap");
        auto z = std::make_unique<Image>(61225, 20);         // new 64K
        check(z->intact(20) && a->intact(10), "both images keep their bytes");
        const size_t live_both = pool.live_requested();
        const uint64_t misses_first = pool.misses();
        z.reset();                                           // the 64K image goes (evicted): its ranges are free and warm
        check(pool.live_requested() < live_both && pool.idle_bytes() > 0, "an evicted image's bytes go back to the pool");
        auto w = std::make_unique<Image>(1931, 30);          // another 2K
        check(w->intact(30), "a 2K image takes some of the freed ranges");
        auto z2 = std::make_unique<Image>(61225, 40);        // the 64K conversation again
        check(z2->intact(40) && a->intact(10) && w->intact(30), "the second 64K image and the 2K images are intact");
        check(pool.misses() == misses_first, "the second 64K image took every large buffer from the pool: no new miss");
        z2.reset(); w.reset(); a.reset();
        check(pool.live_requested() == 0, "everything is given back");
    }
    // growth by appending (a continued conversation) and rewind
    {
        ConversationBuffer b;
        b.resize(3 * MiB, 7);
        const size_t live = pool.live_requested();
        b.resize(3 * MiB + 5, 9);   // within the segment's spare capacity or a new small segment
        std::vector<uint8_t> tail(5);
        check(b.read(tail.data(), 3 * MiB, 5) && tail[0] == 9 && tail[4] == 9, "append keeps working on a pooled buffer");
        b.resize(1 * MiB);
        check(b.size() == 1 * MiB && pool.live_requested() >= live, "a shortened buffer keeps its segment");
        ConversationBuffer c = b;   // a copy is its own storage
        check(c == b, "a copy equals its source");
        ConversationBuffer d = std::move(b);
        check(d.size() == 1 * MiB && b.size() == 0, "a move leaves the source empty");
    }
    check(pool.live_requested() == 0, "no pooled byte is left after the buffers are gone");
    segment_pool_uninstall_for_test();
    // a pool smaller than the request: the pool takes what it can, the rest is the heap, and every byte is right
    {
        SegmentPool small(os, 256 * MiB);
        segment_pool_install(&small);
        {
            Image big(61225, 50);
            check(big.intact(50), "a 64K image over a 256 MiB pool keeps its bytes");
            check(small.hits() > 0 && small.misses() > 0, "the pool served part of it and missed the rest");
            check(small.live_requested() <= 256 * MiB, "the pool never holds more than its size");
            Image other(8000, 60);
            check(other.intact(60) && big.intact(50), "an 8K image beside it, both intact");
        }
        check(small.live_requested() == 0, "all pooled bytes given back");
        segment_pool_uninstall_for_test();
    }
    std::printf("conversation_buffer_pool_test: %d checks passed\n", checks);
    return 0;
}
