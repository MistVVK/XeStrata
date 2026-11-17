// #5722: a layer split prices each participant's prompt-path loan (Prefill::bytes_needed) on THAT participant's device.
// bytes_needed answers for the current device (prefill.cpp fused_ring: the fused layout and its smaller MoE buffers are
// decided per card), so a later stage priced while CUDA0 was current got CUDA0's figure: on an RTX 5080 (fused) + RTX
// 4090 (not) split the 4090's loan was 2882 MiB for a chunk whose buffers there take 3014 MiB, and its prompt path fell
// back to 1024-token chunks.  No GPU: the devices are faked (a "current device" and a price per device).  No real
// mixed-card run backs this test.
#include "strata/program/split_ring.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using strata::program::split_lend_slots;
using strata::program::split_part_figure;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr uint64_t MiB = 1ull << 20;
int current = 0;                         // the faked current device

struct FakeOnDevice {                    // strata::core::OnDevice's contract: -1 leaves the device alone
    int previous = -1;
    explicit FakeOnDevice(int dev) { if (dev >= 0 && dev != current) { previous = current; current = dev; } }
    ~FakeOnDevice() { if (previous >= 0) current = previous; }
    FakeOnDevice(const FakeOnDevice&) = delete;
    FakeOnDevice& operator=(const FakeOnDevice&) = delete;
};
auto on_device = [](int dev) { return FakeOnDevice(dev); };

// one card's buffers per chunk token, in bytes: a "fused" card needs fewer (its smaller MoE buffers)
struct Card { const char* name; uint64_t per_token; int64_t slots; };

// bytes_needed as the engine computes it: for the CURRENT device
uint64_t price_now(const std::vector<Card>& cards, int64_t chunk) {
    return (uint64_t) chunk * cards[(size_t) current].per_token;
}

struct Part { int dev; int64_t slots; };   // dev -1: CUDA0's own part

// the lend of one participant: the slots of its cache that cover the price of `chunk` (uniform max_blob slots)
int64_t lend(const std::vector<Card>& cards, const Part& p, int64_t chunk, uint64_t blob) {
    return split_lend_slots(p.slots, p.slots * (int64_t) blob, nullptr, blob,
                            split_part_figure(on_device, p.dev, [&] { return price_now(cards, chunk); }));
}

// what the participant's own init carves, on its own device
uint64_t carved(const std::vector<Card>& cards, const Part& p, int64_t chunk) {
    return (uint64_t) chunk * cards[(size_t) (p.dev < 0 ? 0 : p.dev)].per_token;
}

std::vector<Part> parts_of(const std::vector<Card>& cards) {
    std::vector<Part> parts{{-1, cards[0].slots}};
    for (size_t d = 1; d < cards.size(); ++d) parts.push_back({(int) d, cards[d].slots});
    return parts;
}
}  // namespace

int main() {
    const uint64_t blob = 4 * MiB;
    const int64_t chunks[] = {256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 16384, 32768};

    // 1. the reporter's pair: CUDA0 an RTX 5080 (fused: 2880 MiB at 8192), CUDA1 an RTX 4090 (3014 MiB at 8192)
    {
        const std::vector<Card> cards{{"RTX 5080", 2880 * MiB / 8192, 4000}, {"RTX 4090", 3014 * MiB / 8192, 4000}};
        const auto parts = parts_of(cards);
        current = 0;
        const int64_t k1 = lend(cards, parts[1], 8192, blob);
        const uint64_t lent = (uint64_t) k1 * blob;
        std::printf("split_part_price_test: CUDA1 (%s) chunk 8192: lent %llu MiB, its buffers there %llu MiB\n",
                    cards[1].name, (unsigned long long) (lent / MiB), (unsigned long long) (carved(cards, parts[1], 8192) / MiB));
        // the stage's init then lays its buffers out in that loan, stepping its chunk down until they fit ("trying a
        // 6144-token chunk ..." in the report)
        int64_t reads = 0;
        for (const int64_t c : {8192, 6144, 4096, 3072, 2048, 1024})
            if (carved(cards, parts[1], c) <= lent) { reads = c; break; }
        std::printf("split_part_price_test: CUDA1 reads its prompt in %lld-token chunks (planned 8192)\n", (long long) reads);
        CHECK(reads == 8192);
        CHECK(lent >= carved(cards, parts[1], 8192));                 // the loan covers what the stage's init carves
        CHECK(current == 0);                                          // the device is put back
        const int64_t k0 = lend(cards, parts[0], 8192, blob);
        CHECK((uint64_t) k0 * blob >= carved(cards, parts[0], 8192)); // CUDA0's own part unchanged
        CHECK((uint64_t) (k0 - 1) * blob < carved(cards, parts[0], 8192));
    }

    // 2. like cards: the device switch changes no number.  Two cards (z440's 2x A4000 class) and four cards (the 4x
    //    R9700 class), every chunk of the scan's grid, priced with the switch against priced on the current device
    //    alone (the old rule).  Equal per-device prices are what like cards give: fused_ring answers the same on each.
    for (int n : {2, 4}) {
        const std::vector<Card> cards((size_t) n, Card{"like card", 2950 * MiB / 8192, 3500});
        const auto parts = parts_of(cards);
        for (const int64_t c : chunks)
            for (const Part& p : parts) {
                current = 0;
                const int64_t with_switch = lend(cards, p, c, blob);
                const int64_t old_rule = split_lend_slots(p.slots, p.slots * (int64_t) blob, nullptr, blob, price_now(cards, c));
                CHECK(with_switch == old_rule);
                CHECK(current == 0);
            }
    }

    if (failed) { std::fprintf(stderr, "split_part_price_test: %d failed\n", failed); return 1; }
    std::printf("split_part_price_test: ok\n");
    return 0;
}
