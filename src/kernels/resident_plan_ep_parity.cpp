// SPDX-FileCopyrightText: 2026 recutita, MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/resident_plan_ep_parity.cpp - the two plans of an expert-parallel pair (resident_plan_ep,
// --expert-parallel-device) against a host replay of their contract.
//
// Each expert is resident on GPU 0, on GPU 1, on both or on neither.  Its owner is GPU 0 when GPU 0 holds it, else GPU
// 1.  For each role the plan is resident_plan's layout over exactly the entries that GPU owns: the distinct owned
// experts in routing order, each with its entries ascending, ptr[g] = the expert's slot on that GPU, counts[1] = the
// owned entries.  Together the two plans cover every entry once.  An expert on neither GPU leaves an empty plan and
// sets *plan_err.  With every expert on GPU 0, role 1's plan is resident_plan's word for word.
// n = 1..80 entries, ids from pools of 3..512 experts, with and without a slot table.  GPU, synthetic, no model.
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

namespace {
template <typename T> T* dalloc(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
}  // namespace

int main() try {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(1402);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    const long long blob = 4096;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced
    void* st = P::stream();

    auto* d_ids = dalloc<int32_t>(128);
    auto* d_r0 = dalloc<int32_t>(NE);
    auto* d_r1 = dalloc<int32_t>(NE);
    auto* d_off = dalloc<unsigned long long>(9000);
    auto* d_plan = dalloc<int32_t>(plan_words);
    auto* d_plan2 = dalloc<int32_t>(plan_words);
    auto* d_skip = dalloc<uint32_t>(1);
    auto* h_err = static_cast<volatile uint32_t*>(P::alloc_host_bytes(64));
    std::vector<unsigned long long> off(9000);
    for (size_t s = 0; s < off.size(); ++s) off[s] = (unsigned long long) s * 4096 + (unsigned long long) (s % 7) * 64;
    P::copy_bytes_in(d_off, off.data(), off.size() * 8);
    const std::vector<int32_t> fill(plan_words, (int32_t) 0xABABABABu);
    auto read_plan = [&](const int32_t* d) {
        std::vector<int32_t> h(plan_words);
        P::copy_bytes_out(h.data(), d, plan_words * 4);
        return h;
    };

    int cases = 0, bad = 0;
    for (int pool : {3, 10, 40, 160, 512}) {
        for (int n = 1; n <= 80; n += (n < 12 ? 1 : 7)) {
            for (int rep = 0; rep < 4; ++rep) {
                std::vector<int32_t> ids((size_t) n), r0(NE, -1), r1(NE, -1);
                std::vector<int> pick((size_t) pool);
                for (int& p : pick) p = (int) (rng() % NE);
                for (auto& v : ids) v = pick[rng() % (size_t) pool];
                // mostly on one GPU, some on both (GPU 0 owns those); on neither only in the error case
                for (int e = 0; e < NE; ++e) {
                    const unsigned r = rng() % 100;
                    if (r < 45) r0[(size_t) e] = (int) (rng() % 9000);
                    else if (r < 90) r1[(size_t) e] = (int) (rng() % 9000);
                    else { r0[(size_t) e] = (int) (rng() % 9000); r1[(size_t) e] = (int) (rng() % 9000); }
                }
                const bool missing = rep == 3;
                if (missing) r0[(size_t) ids[0]] = r1[(size_t) ids[0]] = -1;
                const bool use_off = (n + rep) % 2 == 1;
                P::copy_bytes_in(d_ids, ids.data(), (size_t) n * 4);
                P::copy_bytes_in(d_r0, r0.data(), (size_t) NE * 4);
                P::copy_bytes_in(d_r1, r1.data(), (size_t) NE * 4);
                int covered = 0;
                for (int role = 1; role <= 2; ++role) {
                    const auto& own = role == 1 ? r0 : r1;
                    P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
                    h_err[0] = 0;
                    k::resident_plan_ep(d_ids, n, K, role == 1 ? d_r0 : d_r1, role == 1 ? d_r1 : d_r0, role, NE,
                                        cache_base, use_off ? d_off : nullptr, blob, d_plan, capx,
                                        const_cast<uint32_t*>(h_err), st);
                    P::sync();
                    const std::vector<int32_t> h = read_plan(d_plan);
                    const uint32_t err = h_err[0];
                    ++cases;
                    if (missing) {
                        if (err != 1 || h[0] != 0 || h[1] != 0 || h[4] != 0) {
                            std::printf("    *** pool %d n %d role %d: an expert on neither GPU gave err %u groups %d ***\n",
                                        pool, n, role, err, h[0]);
                            ++bad;
                        }
                        continue;
                    }
                    // the host replay of this role's plan
                    std::vector<int> ge;
                    std::vector<std::vector<int>> gi;
                    for (int i = 0; i < n; ++i) {
                        const int e = ids[(size_t) i];
                        if ((r0[(size_t) e] >= 0 ? 1 : 2) != role) continue;
                        size_t g = 0;
                        while (g < ge.size() && ge[g] != e) ++g;
                        if (g == ge.size()) { ge.push_back(e); gi.emplace_back(); }
                        gi[g].push_back(i);
                    }
                    int ents = 0;
                    for (const auto& v : gi) ents += (int) v.size();
                    const int32_t* start = h.data() + 4;
                    const int32_t* dst = start + capx + 1;
                    const int32_t* tok = dst + capx;
                    const auto* ptr = reinterpret_cast<const unsigned long long*>(h.data() + ptr_off);
                    const int32_t* start2 = h.data() + ptr_off + 4 * capx;
                    bool ok = err == 0 && h[0] == (int) ge.size() && h[1] == ents && h[2] == 0 &&
                              start[ge.size()] == ents && start2[0] == ents;
                    int at = 0;
                    for (size_t g = 0; ok && g < ge.size(); ++g) {
                        const int32_t slot = own[(size_t) ge[g]];
                        const unsigned long long want = (unsigned long long) (uintptr_t) cache_base +
                            (use_off ? off[(size_t) slot] : (unsigned long long) slot * (unsigned long long) blob);
                        ok = ptr[g] == want && start[g] == at;
                        for (int i : gi[g]) {
                            ok = ok && dst[at] == i && tok[at] == i / K;
                            ++at;
                        }
                    }
                    covered += ents;
                    if (!ok) {
                        std::printf("    *** pool %d n %d role %d: groups %d (want %zu) entries %d (want %d) err %u ***\n",
                                    pool, n, role, h[0], ge.size(), h[1], ents, err);
                        ++bad;
                    }
                }
                if (!missing && covered != n) {
                    std::printf("    *** pool %d n %d: the two plans cover %d of %d entries ***\n", pool, n, covered, n);
                    ++bad;
                }
                if (missing) continue;
                // every expert on GPU 0: role 1 is resident_plan's plan word for word
                for (int e = 0; e < NE; ++e)
                    if (r0[(size_t) e] < 0) r0[(size_t) e] = (int) (rng() % 9000);
                P::copy_bytes_in(d_r0, r0.data(), (size_t) NE * 4);
                P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
                P::copy_bytes_in(d_plan2, fill.data(), plan_words * 4);
                k::resident_plan_ep(d_ids, n, K, d_r0, d_r1, 1, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan,
                                    capx, const_cast<uint32_t*>(h_err), st);
                k::resident_plan(d_ids, n, K, d_r0, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan2, capx,
                                 d_skip, 7u, st);
                P::sync();
                ++cases;
                if (read_plan(d_plan) != read_plan(d_plan2)) {
                    std::printf("    *** pool %d n %d: role 1 with every expert on GPU 0 differs from resident_plan ***\n",
                                pool, n);
                    ++bad;
                }
            }
        }
    }
    std::printf("resident_plan_ep: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("resident_plan_ep_parity OK\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "resident_plan_ep_parity: %s\n", e.what());
    return 1;
}
