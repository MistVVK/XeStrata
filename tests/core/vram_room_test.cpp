// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// core::vram_room - the one rule every VRAM margin is sized by: a 32nd of the memory the device reports,
// held between the caller's measured bounds, of which a stage keeps a part.  Synthetic memories; no model, no GPU.
#include "strata/core/device.hpp"

#include <cstdint>
#include <cstdio>

int main() {
    using strata::core::vram_room;
    constexpr uint64_t MiB = 1ull << 20;
    struct {
        const char* what;
        uint64_t total, parts, least, most, want;
    } cases[] = {
        // the image pipelines' and SeedVR2's defaults: the 512 MiB they measured, up against the 768 MiB ceiling
        {"16 GiB card, engine room", 16304 * MiB, 1, 512 * MiB, 768 * MiB, 512 * MiB},
        {"12 GiB card, engine room", 12288 * MiB, 1, 512 * MiB, 768 * MiB, 512 * MiB},
        {"6 GiB card, engine room", 5783 * MiB, 1, 512 * MiB, 768 * MiB, 512 * MiB},
        {"31 GiB card, engine room", 31023 * MiB, 1, 512 * MiB, 768 * MiB, 768 * MiB},
        // the plain engine's expert tier: its own floor, so the device's share speaks on the small cards.  On a
        // 16304 MiB card the share is 509.5 MiB, what the rule returns whole; the image engine's floor of 512
        // MiB takes that instead, and the stage's parts halve, quarter and eighth it
        {"16 GiB card, plain tier", 16304 * MiB, 1, 192 * MiB, 768 * MiB, 16304 * MiB / 32},
        {"8 GiB card, plain tier", 8192 * MiB, 1, 192 * MiB, 768 * MiB, 256 * MiB},
        {"4 GiB card, plain tier", 4096 * MiB, 1, 192 * MiB, 768 * MiB, 192 * MiB},
        // Qwen3.8's reserve: never under the room its graphs, hit scratch and head measured
        {"16 GiB card, Qwen3.8 reserve", 16304 * MiB, 1, 700 * MiB, 768 * MiB, 700 * MiB},
        {"31 GiB card, Qwen3.8 reserve", 31023 * MiB, 1, 700 * MiB, 768 * MiB, 768 * MiB},
        // a stage's part of the room: SeedVR2's VAE, MOSS-Image's decode, MOSS-Image's encoders
        {"16 GiB card, stage half", 16304 * MiB, 2, 512 * MiB, 768 * MiB, 512 * MiB / 2},
        {"16 GiB card, stage quarter", 16304 * MiB, 4, 512 * MiB, 768 * MiB, 512 * MiB / 4},
        {"16 GiB card, stage eighth", 16304 * MiB, 8, 512 * MiB, 768 * MiB, 512 * MiB / 8},
        {"31 GiB card, stage eighth", 31023 * MiB, 8, 512 * MiB, 768 * MiB, 96 * MiB},
        // the device's memory unknown: the caller's own floor
        {"unknown device, engine room", 0, 1, 512 * MiB, 768 * MiB, 512 * MiB},
    };
    int failed = 0;
    for (const auto& c : cases) {
        const uint64_t got = vram_room(c.total, c.parts, c.least, c.most);
        const bool ok = got == c.want;
        if (!ok) ++failed;
        std::printf("%-28s %7llu MiB -> %7llu MiB (want %7llu) %s\n", c.what, (unsigned long long) (c.total / MiB),
                    (unsigned long long) (got / MiB), (unsigned long long) (c.want / MiB), ok ? "ok" : "FAIL");
    }
    // the rule over the memories between them: one expression, never under the floor, never over the ceiling
    for (uint64_t mib = 512; mib <= 196608; mib *= 2) {
        const uint64_t got = vram_room(mib * MiB, 1, 192 * MiB, 768 * MiB);
        const uint64_t want = (mib / 32 < 192 ? 192 : (mib / 32 > 768 ? 768 : mib / 32)) * MiB;
        if (got != want) {
            std::printf("FAIL rule at %llu MiB: %llu MiB, want %llu\n", (unsigned long long) mib,
                        (unsigned long long) (got / MiB), (unsigned long long) (want / MiB));
            ++failed;
        }
    }
    std::printf("vram_room: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
