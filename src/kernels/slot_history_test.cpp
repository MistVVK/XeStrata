// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// --batch slots keep each request's repeat / frequency / presence penalties: the per-row histories of a batch window.
// Header-only, CPU, no model (upstream 2678c9732, the SYCL copy 8a6afd5f1).
//
// A slot's row must be the one the same request would get alone: its reply tail (the tokens its session consumed
// since the reply began) followed by the tokens this window fed to it up to the row, and a second row of the same
// slot (--batch-mtp's draft) counts the first.
#include "strata/kernels/sampler.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::kernels::penalty_rows;
using strata::kernels::slot_penalty_rows;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr int kH = 8;   // a small window, so a row is readable in the test

// One slot's row of a batch window, as `penalty_rows` gives the same slot alone: the tail then the slot's tokens.
std::vector<int32_t> solo_row(const std::vector<int32_t>& tail, const std::vector<int32_t>& slot_tokens, int t) {
    std::vector<int32_t> window(slot_tokens.begin(), slot_tokens.begin() + t + 1);
    std::vector<int32_t> rows((size_t) window.size() * kH);
    penalty_rows(tail.data(), (int64_t) tail.size(), window.data(), (int) window.size(), kH, rows.data());
    return std::vector<int32_t>(rows.begin() + (std::ptrdiff_t) t * kH, rows.begin() + (std::ptrdiff_t) (t + 1) * kH);
}

std::vector<int32_t> batch_row(const int* rows, const int32_t* tokens, int S,
                               const std::vector<std::vector<int32_t>>& tails, int t) {
    std::vector<const int32_t*> p;
    std::vector<int64_t> n;
    for (const auto& tail : tails) { p.push_back(tail.empty() ? nullptr : tail.data()); n.push_back((int64_t) tail.size()); }
    std::vector<int32_t> out((size_t) S * kH, -7);
    slot_penalty_rows(rows, tokens, S, p.data(), n.data(), kH, out.data());
    return std::vector<int32_t>(out.begin() + (std::ptrdiff_t) t * kH, out.begin() + (std::ptrdiff_t) (t + 1) * kH);
}

std::vector<int32_t> of(std::initializer_list<int32_t> v) { return std::vector<int32_t>(v); }
}  // namespace

int main() {
    // ---- one slot, one row, a tail longer than the window: the same row the solo path stages
    {
        std::vector<int32_t> tail;
        for (int32_t i = 1; i <= 20; ++i) tail.push_back(100 + i);
        const std::vector<int32_t> fed = {900, 901};
        const int rows[2] = {0, 0};                       // the slot twice: a fed token and a draft
        const std::vector<std::vector<int32_t>> tails = {tail};
        const std::vector<int32_t> r1 = batch_row(rows, fed.data(), 2, tails, 1);
        check(r1 == solo_row(tail, fed, 1), "one slot: row 1 is the solo path's row");
        check(r1[kH - 1] == 901 && r1[kH - 2] == 900, "one slot: the fed tokens are the last two, most recent last");
        check(r1[0] == 100 + 15, "one slot: the window slid over the tail's last tokens");
    }

    // ---- two slots, interleaved rows (slot 1, slot 0, slot 1): each row is that slot's own history
    {
        const std::vector<std::vector<int32_t>> tails = {of({11, 12, 13}), of({21, 22})};
        const int rows[3] = {1, 0, 1};
        const std::vector<int32_t> fed = {500, 600, 501};
        const std::vector<int32_t> a1 = batch_row(rows, fed.data(), 3, tails, 0);   // slot 1, its first row
        const std::vector<int32_t> b0 = batch_row(rows, fed.data(), 3, tails, 1);   // slot 0
        const std::vector<int32_t> a2 = batch_row(rows, fed.data(), 3, tails, 2);   // slot 1, its second row
        check(a1 == solo_row(tails[1], of({500, 501}), 0), "slot 1's first row: its own tail and fed token");
        check(b0 == solo_row(tails[0], of({600}), 0), "slot 0's row: its own tail, not slot 1's");
        check(a2 == solo_row(tails[1], of({500, 501}), 1), "slot 1's second row: the first row's token counted");
        bool own = true;
        for (int32_t v : b0) own = own && v != 22;
        check(own, "slot 0's row never holds slot 1's tail");
    }

    // ---- a slot twice in a row (a draft that was the token after it): the draft's row counts the row before it
    {
        const std::vector<std::vector<int32_t>> tails = {of({70})};
        const int rows[2] = {0, 0};
        const std::vector<int32_t> fed = {800, 801};
        const std::vector<int32_t> r0 = batch_row(rows, fed.data(), 2, tails, 0);
        const std::vector<int32_t> r1 = batch_row(rows, fed.data(), 2, tails, 1);
        check(r0[kH - 1] == 800 && r0[kH - 2] == 70, "draft slot row 0: the tail then its token");
        check(r1[kH - 1] == 801 && r1[kH - 2] == 800 && r1[kH - 3] == 70, "draft slot row 1: both its tokens");
        check(r1 == solo_row(tails[0], fed, 1), "draft slot: the window is the slot's alone");
    }

    // ---- a slot with no tail yet (its first pick): only the fed token
    {
        const std::vector<std::vector<int32_t>> tails = {of({}), of({42})};
        const int rows[2] = {0, 1};
        const std::vector<int32_t> fed = {300, 400};
        const std::vector<int32_t> r0 = batch_row(rows, fed.data(), 2, tails, 0);
        int filled = 0;
        for (int32_t v : r0) filled += v >= 0;
        check(filled == 1 && r0[kH - 1] == 300, "no tail: the first pick counts only its fed token");
    }

    // ---- a window wider than the tail and the tokens so far: -1 in front, never a stale token
    {
        const std::vector<std::vector<int32_t>> tails = {of({5, 6})};
        const int rows[1] = {0};
        const std::vector<int32_t> fed = {7};
        const std::vector<int32_t> r = batch_row(rows, fed.data(), 1, tails, 0);
        check(r[0] == -1 && r[1] == -1 && r[kH - 3] == 5 && r[kH - 2] == 6 && r[kH - 1] == 7, "front slots are -1");
    }

    std::printf("slot_history_test: %d checks passed\n", checks);
    return 0;
}
