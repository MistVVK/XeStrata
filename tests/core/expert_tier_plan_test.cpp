// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The VRAM tier's plan every engine shares (expert_tier.hpp): plan_tier_room, plan_tier_fill and tier_slots_within
// against the rule as PlainModel::open_experts had it written out, over synthetic memories and layers.  No model,
// no GPU.
#include "strata/core/expert_tier.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace strata::core;

namespace {

constexpr uint64_t MiB = 1ull << 20;
int failed = 0;

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    ++failed;
    std::printf("FAIL %s: %lld, want %lld\n", what, got, want);
}

}  // namespace

int main() {
    using Pairs = std::vector<std::pair<int32_t, int32_t>>;
    // ---- the room: the free memory less the window and the reserve, a layer's count, --expert-cache's cap
    struct {
        uint64_t free, total, window;
        int64_t reserve_mib, max_slots;
        uint64_t per_layer;
        int64_t layers, n_expert;
    } rooms[] = {
        {11000 * MiB, 12288 * MiB, 300 * MiB, -1, -1, 40 * MiB, 40, 128},    // what fits: under every expert
        {30000 * MiB, 32657 * MiB, 300 * MiB, -1, -1, 40 * MiB, 40, 128},    // every expert fits
        {11000 * MiB, 12288 * MiB, 300 * MiB, 2048, -1, 40 * MiB, 40, 128},  // --vram-reserve-mib
        {11000 * MiB, 12288 * MiB, 300 * MiB, -1, 2000, 40 * MiB, 40, 128},  // --expert-cache 2000
        {11000 * MiB, 12288 * MiB, 300 * MiB, -1, 0, 40 * MiB, 40, 128},     // --expert-cache 0
        {400 * MiB, 4096 * MiB, 300 * MiB, -1, -1, 40 * MiB, 40, 128},       // nothing left
        {11000 * MiB, 12288 * MiB, 300 * MiB, -1, -1, 0, 40, 128},           // no layer holds experts
    };
    for (const auto& c : rooms) {
        const TierRoom t = plan_tier_room(c.free, c.total, c.window, c.reserve_mib, c.max_slots, c.per_layer, c.layers,
                                          c.n_expert);
        // (the rule as open_experts had it)
        const uint64_t reserve = c.reserve_mib >= 0 ? (uint64_t) c.reserve_mib << 20
                                                    : vram_room(c.total, /*parts*/ 1, kTierRoomLeast);
        const uint64_t room = c.free > c.window + reserve ? c.free - c.window - reserve : 0;
        int64_t q = c.per_layer > 0 ? (int64_t) (room / c.per_layer) : 0;
        q = std::min<int64_t>(q, c.n_expert);
        if (c.max_slots >= 0) q = std::min<int64_t>(q, c.max_slots / std::max<int64_t>(c.layers, 1));
        check(t.reserve == reserve, "reserve", (long long) t.reserve, (long long) reserve);
        check(t.room == room, "room", (long long) t.room, (long long) room);
        check(t.q == q, "q", (long long) t.q, (long long) q);
        const uint64_t bytes = c.per_layer * (uint64_t) q;
        check(t.bytes == bytes, "bytes", (long long) t.bytes, (long long) bytes);
    }
    check(plan_tier_room(11000 * MiB, 12288 * MiB, 300 * MiB, -1, -1, 40 * MiB, 40, 128).reserve == 384 * MiB,
          "a 12 GiB card's reserve", 0, 384);
    check(tier_window_bytes(8, 8, 2048, 30 * MiB) == 8ull * 8 * 2048 * 12 + 480 * MiB + 64 * MiB, "window", 0, 0);

    // ---- the slots each layer takes: layers of two sizes, the odd ones not this tier's
    const int64_t L = 6, E = 16;
    const auto bytes = [](int64_t l) { return (uint64_t) (l < 3 ? 10 : 7) * MiB + 100; };
    const auto local = [](int64_t l) { return l != 1; };
    uint64_t per_layer = 0;
    for (int64_t l = 0; l < L; ++l)
        if (local(l)) per_layer += bytes(l);
    const TierRoom room = plan_tier_room(700 * MiB, 4096 * MiB, 100 * MiB, -1, -1, per_layer, L, E);
    check(room.q > 0 && room.q < E, "the case's q", room.q, 9);
    {
        // without a profile: every layer of the tier the same count, its first experts, the layers in turn
        const auto f = plan_tier_fill<Pairs>(nullptr, L, E, room, bytes, local, false);
        for (int64_t l = 0; l < L; ++l)
            check(f.quota[(size_t) l] == (local(l) ? room.q : 0), "quota without a profile", f.quota[(size_t) l],
                  room.q);
        check((int64_t) f.fill.size() == room.q * 5, "fill without a profile", (long long) f.fill.size(), room.q * 5);
        bool turn = true;
        for (size_t i = 0; i < f.fill.size(); ++i) {
            static const int32_t order[] = {0, 2, 3, 4, 5};
            turn &= f.fill[i].first == order[i % 5] && f.fill[i].second == (int32_t) (i / 5);
        }
        check(turn, "the layers in turn", 0, 1);
    }
    {
        // with a profile: plan_profile_fill's over the room's bytes, whichever way the layers share
        Pairs prior;
        for (int32_t e = 0; e < E; ++e)
            for (int32_t l = (e % 2 ? 5 : 0); l >= 0 && l < L; l += (e % 2 ? -1 : 1))
                prior.emplace_back(l, (int32_t) (((int64_t) e * 7) % E));
        for (const bool alike : {false, true}) {
            const auto f = plan_tier_fill(&prior, L, E, room, bytes, local, alike);
            const auto want = plan_profile_fill(prior, L, E, per_layer * (uint64_t) room.q, bytes, local, alike);
            check(f.quota == want.quota && f.fill == want.fill, "the profile's fill", alike, alike);
            uint64_t took = 0;
            for (int64_t l = 0; l < L; ++l) took += bytes(l) * (uint64_t) f.quota[(size_t) l];
            check(took <= room.bytes && f.quota[1] == 0, "within the room", (long long) took, (long long) room.bytes);
        }
    }

    // ---- the slots that fit the memory read again: the last ones come off (open_experts' loop)
    std::vector<int64_t> slots;
    slots.reserve(300);
    for (int i = 0; i < 300; ++i) slots.push_back((i % 3 ? 7 : 10) * (int64_t) MiB + 100 + i % 5);
    const uint64_t limits[] = {0, 5 * MiB, 10 * MiB + 256, 1000 * MiB, 2000 * MiB + 77, 1ull << 40};
    for (const uint64_t limit : limits) {
        std::vector<int64_t> left = slots;
        uint64_t need = 0;
        for (const int64_t b : left) need += ((uint64_t) b + 255) / 256 * 256;
        while (!left.empty() && need > limit) {
            need -= ((uint64_t) left.back() + 255) / 256 * 256;
            left.pop_back();
        }
        const size_t got = tier_slots_within(slots, limit);
        check(got == left.size(), "slots within", (long long) got, (long long) left.size());
    }
    std::printf("expert_tier_plan: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
