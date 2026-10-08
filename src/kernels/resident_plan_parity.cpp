// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/resident_plan_parity.cpp - the verify window's device-built expert plan (#783 PR-e: the groups numbered
// by a scan) against a host replay of its contract: distinct experts in routing order, each with its entries
// ascending, `ptr[g]` = the expert's slot, counts / start / dst / tok / start2 in the host pool's layout.
//
// n = 1..80 entries (a window of up to kVerifyMaxT tokens x 10 experts), ids drawn from pools of 3..512 experts so
// duplicates run from "every entry the same expert" to "all distinct", with and without a slot table (`slot_off`),
// and the skip word (ring when the plan was built; 0 and the plan untouched when an expert was not resident).
// Every word of the plan the host pool would read is compared with memcmp. GPU, synthetic, no model.
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

namespace {
template <typename T> T* dalloc(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
}  // namespace

int main() {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(783);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    const long long blob = 1337;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced
    void* st = P::stream();

    auto* d_ids = dalloc<int32_t>(128);
    auto* d_res = dalloc<int32_t>(NE);
    auto* d_off = dalloc<unsigned long long>(1024);
    auto* d_plan = dalloc<int32_t>(plan_words);
    auto* d_skip = dalloc<uint32_t>(1);

    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[e] = (e * 37 + 11) % 1000;      // distinct slots, not the identity
    std::vector<unsigned long long> off(1024);
    for (int s = 0; s < 1024; ++s) off[s] = (unsigned long long) s * 4096 + (unsigned long long) (s % 7) * 64;
    P::copy_bytes_in(d_res, res.data(), (size_t) NE * 4);
    P::copy_bytes_in(d_off, off.data(), off.size() * 8);
    const std::vector<int32_t> fill(plan_words, (int32_t) 0xABABABABu);
    const uint32_t zero = 0;

    int cases = 0, bad = 0;
    for (int n = 1; n <= 80; ++n) {
        for (int variant = 0; variant < 4; ++variant) {
            const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = (int32_t) ((rng() % pool) * (512 / pool));
            const bool use_off = (n + variant) % 2 == 1;
            P::copy_bytes_in(d_ids, ids.data(), (size_t) n * 4);

            // host replay
            std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
            std::vector<unsigned long long> ptr;
            std::vector<int> first_seen;     // group -> expert
            for (int i = 0; i < n; ++i) {
                bool seen = false;
                for (int e : first_seen) seen = seen || e == ids[i];
                if (!seen) first_seen.push_back(ids[i]);
            }
            const int groups = (int) first_seen.size();
            counts[0] = groups;
            start.assign(groups + 1, 0);
            int pos = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = pos;
                ptr.push_back((unsigned long long) (uintptr_t) cache_base +
                              (use_off ? off[res[first_seen[g]]] : (unsigned long long) res[first_seen[g]] * (unsigned long long) blob));
                for (int i = 0; i < n; ++i)
                    if (ids[i] == first_seen[g]) { dst[pos] = i; tok[pos] = i / K; ++pos; }
            }
            start[groups] = n;

            P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
            P::copy_bytes_in(d_skip, &zero, 4);
            k::resident_plan(d_ids, n, K, d_res, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx, d_skip,
                             7u, st);
            P::sync();
            std::vector<int32_t> plan(plan_words);
            P::copy_bytes_out(plan.data(), d_plan, plan_words * 4);
            uint32_t skip = 0;
            P::copy_bytes_out(&skip, d_skip, 4);
            bool ok = skip == 7u;
            ok = ok && std::memcmp(plan.data(), counts.data(), counts.size() * 4) == 0;
            ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
            const int32_t* pdst = plan.data() + 4 + capx + 1;
            ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
            ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
            ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t) groups * 8) == 0;
            ok = ok && plan[(size_t) (ptr_off + 4 * capx)] == n;
            ++cases;
            if (!ok) {
                std::printf("    *** n=%d pool=%d %s: the plan differs from the host replay ***\n", n, pool,
                            use_off ? "slot_off" : "blob");
                ++bad;
            }
        }
    }

    // a routed expert that is not resident: the kernel leaves the plan alone and zeroes *skip
    {
        std::vector<int32_t> res2 = res;
        res2[5] = -1;
        P::copy_bytes_in(d_res, res2.data(), (size_t) NE * 4);
        const int32_t ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        P::copy_bytes_in(d_ids, ids, sizeof ids);
        P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
        const uint32_t seven = 7;
        P::copy_bytes_in(d_skip, &seven, 4);
        k::resident_plan(d_ids, 10, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, st);
        P::sync();
        std::vector<int32_t> plan(plan_words);
        P::copy_bytes_out(plan.data(), d_plan, plan_words * 4);
        uint32_t skip = 99;
        P::copy_bytes_out(&skip, d_skip, 4);
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t) w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched) {
            std::printf("    *** a non-resident expert: skip=%u untouched=%d (want 0 / 1) ***\n", skip, (int) untouched);
            ++bad;
        }
    }

    std::printf("resident_plan: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("resident_plan_parity OK\n");
    return 0;
}
