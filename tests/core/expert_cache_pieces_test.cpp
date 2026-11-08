// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// The pieces fallback of ExpertCache::open_sized/open: when the driver refuses the one large allocation (the B70's
// xe driver, 2026-10-22: 25.5 GiB refused with 26.4 GiB reported free, every smaller whole-arena retry refused
// after it), the cache is taken in pieces of an eighth (at least 256 MiB, at most the device's largest
// allocation), keeping the prefix the device gives.  STRATA_TEST_ALLOC_FAIL fakes the single allocation's refusal
// and counts down, STRATA_TEST_PIECE_FAIL the pieces' from the second on, so the cases run in this one process; a want beyond the
// card takes the pre-check's refusal.
// Needs a GPU (exits 77 without one).
#include "strata/core/expert_cache.hpp"

#include "strata/core/runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// the environment as the engine reads it (std::getenv); Windows has no setenv, and an empty value removes the variable
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (value[0] == '\0') ::unsetenv(name);
    else ::setenv(name, value, 1);
#endif
}

int open_and_check(const char* name, const std::vector<int64_t>& slot_bytes, int64_t nl, int64_t ne,
                   int64_t want_slots) {
    std::string err;
    strata::core::ExpertCache cache;
    if (!cache.open_sized(slot_bytes, nl, ne, err)) {
        if (want_slots >= 0) {
            std::fprintf(stderr, "%s: refused, expected %lld slots: %s\n", name, (long long) want_slots,
                         err.c_str());
            return 1;
        }
        // the give-up: the refusal is said in full, naming the allocation or the card's free VRAM
        if (std::strstr(err.c_str(), "MiB failed") == nullptr &&
            std::strstr(err.c_str(), "of VRAM is free") == nullptr) {
            std::fprintf(stderr, "%s: the refusal is not said in full: %s\n", name, err.c_str());
            return 1;
        }
        std::printf("%s: refused cleanly\n", name);
        return 0;
    }
    if (want_slots < 0) {
        std::fprintf(stderr, "%s: opened with %lld slots, expected a refusal\n", name, (long long) cache.slots());
        return 1;
    }
    if (cache.slots() != want_slots) {
        std::fprintf(stderr, "%s: %lld slots, expected %lld\n", name, (long long) cache.slots(),
                     (long long) want_slots);
        return 1;
    }
    if (cache.slot_offsets() == nullptr || cache.last_piece_first() < 0 ||
        cache.last_piece_first() >= cache.slots()) {
        std::fprintf(stderr, "%s: offsets %s, last piece at %lld of %lld\n", name,
                     cache.slot_offsets() ? "set" : "NULL", (long long) cache.last_piece_first(),
                     (long long) cache.slots());
        return 1;
    }
    // every slot the cache reports must have an address: the pieces' wrap-around arithmetic
    for (int64_t s = 0; s < cache.slots(); ++s)
        if (cache.device_slot((int32_t) s) == nullptr) {
            std::fprintf(stderr, "%s: slot %lld has no address\n", name, (long long) s);
            return 1;
        }
    std::printf("%s: %lld slots (last piece at %lld)\n", name, (long long) cache.slots(),
                (long long) cache.last_piece_first());
    return 0;
}

}  // namespace

int main() {
    try { strata::core::Runtime::get(); } catch (const std::exception& e) {
        std::puts(e.what()); return 77;
    }
    // 6 slots of 256 MiB: one allocation of 1.5 GiB, the fallback's pieces 256 MiB each
    const std::vector<int64_t> slots(6, 256 << 20);
    int bad = 0;
    // plain: one allocation, no pieces
    bad += open_and_check("plain", slots, 3, 4, 6);
    // the one allocation refused: every piece taken, the whole cache in pieces
    set_env("XESTRATA_TEST_ALLOC_FAIL", "1");
    bad += open_and_check("fallback", slots, 3, 4, 6);
    // the same refusal, with the pieces from the second on refused: only the first piece's slots stay
    set_env("XESTRATA_TEST_ALLOC_FAIL", "1");
    set_env("XESTRATA_TEST_PIECE_FAIL", "5");
    bad += open_and_check("prefix", slots, 3, 4, 1);
    set_env("XESTRATA_TEST_PIECE_FAIL", "");
    set_env("XESTRATA_TEST_ALLOC_FAIL", "");
    // a want beyond any card here: the pre-check's refusal, said in full (Lower --expert-cache)
    bad += open_and_check("toobig", std::vector<int64_t>(6, 64ll << 30), 3, 4, -1);
    return bad == 0 ? 0 : 1;
}
