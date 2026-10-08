// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exchange_storage.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <thread>
#include <stdexcept>
#include <vector>

using strata::core::detail::ExchangeStorage;
constexpr uint64_t absent = ~uint64_t{0};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

void exercise(size_t bytes, size_t rounds) {
    constexpr size_t n = 24, resident = 16, spare_count = 4, guard = 19;
    std::vector<uint8_t> arena(guard + resident * bytes + guard, 0xcd);
    std::vector<uint8_t> spare(guard + spare_count * bytes + guard, 0xef);
    // Distinct alias arrays catch accidentally retaining the old device address.
    std::vector<uint8_t> aliases(resident * bytes), spare_aliases(spare_count * bytes);
    std::vector<std::vector<uint8_t>> reference(n, std::vector<uint8_t>(bytes));
    std::mt19937 random(601);   // NOLINT(bugprone-random-generator-seed): a fixed fixture
    for (auto& row : reference) for (auto& v : row) v = (uint8_t)random();
    std::vector<uint64_t> offsets(n, absent);
    for (size_t i = 0; i < resident; ++i) {
        offsets[i] = i * bytes;
        std::memcpy(arena.data() + guard + offsets[i], reference[i].data(), bytes);
    }
    ExchangeStorage storage;
    std::string error;
    require(storage.initialize(offsets, arena.data() + guard, aliases.data(), resident * bytes,
        spare.data() + guard, spare_aliases.data(), spare_count, bytes, error), error.c_str());
    std::set<uint8_t*> original;
    for (size_t i = 0; i < n; ++i) if (auto p = storage.resident(i).host) original.insert(p);
    for (size_t q = 0; q < spare_count; ++q) original.insert(storage.spare(q).host);
    // A lookahead thread may inspect residency while ownership is committed.
    // It only observes immutable address pairs, never reads mutable expert bytes.
    std::atomic<bool> stop{false}, reader_failed{false};
    std::atomic<uint64_t> observations{0};
    std::vector<ExchangeStorage::View> valid;
    for (size_t i = 0; i < n; ++i) if (storage.resident(i).host) valid.push_back(storage.resident(i));
    for (size_t q = 0; q < spare_count; ++q) valid.push_back(storage.spare(q));
    std::thread reader([&] {
        while (!stop.load()) {
            for (size_t i = 0; i < n; ++i) {
                const auto view = storage.resident(i);
                if (view.host && std::none_of(valid.begin(), valid.end(), [&](auto v) {
                        return v.host == view.host && v.device == view.device; })) reader_failed.store(true);
            }
            ++observations;
        }
    });
    struct Join {
        std::atomic<bool>& stop; std::thread& reader;
        void finish() { stop.store(true); if (reader.joinable()) reader.join(); }
        ~Join() { finish(); }
    } join{stop, reader};
    while (!observations.load()) std::this_thread::yield();
    uint64_t exchanges = 0;
    for (size_t round = 0; round < rounds; ++round) {
        std::vector<size_t> ins, outs;
        for (size_t i = 0; i < n; ++i) (storage.resident(i).host ? ins : outs).push_back(i);
        std::shuffle(ins.begin(), ins.end(), random);
        std::shuffle(outs.begin(), outs.end(), random);
        ExchangeStorage::View before[spare_count], staged[spare_count];
        for (size_t q = 0; q < spare_count; ++q) {
            before[q] = storage.resident(ins[q]); staged[q] = storage.spare(q);
            // Simulated completed D2H eviction. The old RAM input stays readable
            // until simulated H2D and all readers finish, before commit.
            std::memcpy(staged[q].host, reference[outs[q]].data(), bytes);
            require(!std::memcmp(before[q].host, reference[ins[q]].data(), bytes), "incoming expert overwritten early");
        }
        for (size_t q = 0; q < spare_count; ++q) {
            require(!storage.commit(ins[q], ins[q], q, staged[q].host, bytes), "self swap accepted");
            require(!storage.commit(ins[q], outs[q], q, staged[q].host, bytes + 1), "wrong size accepted");
            require(!storage.commit(ins[q], outs[q], q, nullptr, bytes), "missing staged buffer accepted");
            require(!storage.commit(ins[q], outs[q], spare_count, staged[q].host, bytes), "bad spare accepted");
            require(!storage.commit(n, outs[q], q, staged[q].host, bytes), "bad expert accepted");
            require(storage.commit(ins[q], outs[q], q, staged[q].host, bytes), "valid swap rejected");
            require(!storage.commit(ins[q], outs[q], q, staged[q].host, bytes), "duplicate commit accepted");
            require(storage.resident(outs[q]).host == staged[q].host &&
                    storage.resident(outs[q]).device == staged[q].device, "evicted buffer was not adopted with its alias");
            require(storage.spare(q).host == before[q].host && storage.spare(q).device == before[q].device,
                    "old RAM input was not recycled with its alias");
            // Even the old incoming bytes have not changed: commit copies metadata only.
            require(!std::memcmp(storage.spare(q).host, reference[ins[q]].data(), bytes), "commit changed spare bytes");
            ++exchanges;
        }
        std::set<uint8_t*> live;
        for (size_t i = 0; i < n; ++i) if (auto p = storage.resident(i).host) {
            require(live.insert(p).second, "two experts share storage");
            require(!std::memcmp(p, reference[i].data(), bytes), "resident bytes corrupted");
        }
        for (size_t q = 0; q < spare_count; ++q)
            require(live.insert(storage.spare(q).host).second, "spare aliases live expert");
        require(live == original, "storage leaked or duplicated");
    }
    require(storage.exchanges() == exchanges && storage.avoided_bytes() == exchanges * bytes, "wrong accounting");
    require(std::all_of(arena.begin(), arena.begin() + guard, [](auto x) { return x == 0xcd; }) &&
            std::all_of(arena.end() - guard, arena.end(), [](auto x) { return x == 0xcd; }) &&
            std::all_of(spare.begin(), spare.begin() + guard, [](auto x) { return x == 0xef; }) &&
            std::all_of(spare.end() - guard, spare.end(), [](auto x) { return x == 0xef; }), "guard overwritten");
    require(!storage.initialize(offsets, arena.data() + guard, aliases.data(), resident * bytes,
        spare.data() + guard, spare_aliases.data(), spare_count, bytes, error), "live storage reinitialized");
    join.finish();
    require(!reader_failed.load() && observations.load(), "concurrent residency reader observed invalid alias");
    storage.clear();
    require(!storage.active() && !storage.resident(0).host && !storage.spare(0).host &&
            !storage.exchanges() && !storage.avoided_bytes(), "clear retained state");
}

void invalid_geometry() {
    uint8_t h[64]{}, d[64]{}, x[64]{}, xd[64]{};
    ExchangeStorage s; std::string e;
    require(!s.initialize({0, 3}, h, d, 8, x, xd, 2, 4, e), "unaligned slot accepted");
    require(!s.initialize({0, 0}, h, d, 8, x, xd, 2, 4, e), "duplicate offset accepted");
    require(!s.initialize({0, 8}, h, d, 8, x, xd, 2, 4, e), "out of bounds accepted");
    require(!s.initialize({0, absent}, h, d, 8, x, xd, 2, 4, e), "lost slot accepted");
    require(!s.initialize({0, 4}, h, nullptr, 8, x, xd, 2, 4, e), "unmapped resident accepted");
    require(!s.initialize({0, 4}, h, d, 8, x, nullptr, 2, 4, e), "unmapped spare accepted");
    require(!s.initialize({0, 4}, h, d, 8, x, xd, 2, 0, e), "zero blob accepted");
    require(!s.active(), "failed initialization left active state");
}

int main() {
    try {
        invalid_geometry();
        for (size_t bytes : {1u, 33u, 4097u}) exercise(bytes, 1024);
        exercise(5222400, 4); // actual Q8 expert size
        std::cout << "exchange_storage_test: PASS (byte/alias/ownership checks, 12304 exchanges)\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
