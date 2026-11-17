// How one layer divides over n cards (included by model.hpp, after the model's shapes): the recurrent key and value
// heads, the experts' FF width, and which counts the shapes allow. Functions of the count, so the static_asserts below
// check every supported count in every build, not only the one compiled.
#pragma once

namespace fnx {

// the counts the shapes allow: the query heads (NQH) split evenly, so do the value heads (NVH), every card gets at
// least one 32-value FF block, and every card is a CUDA peer of every other (the prompt path's attention exchange
// writes into every card; CUDA allows eight peer connections a card on PCIe, so nine cards at most, and nine fails the
// head split: NCARD_MAX is the largest count that passes both). 2, 3, 4, 6, 8
constexpr int NCARD_MAX = 8;
constexpr bool ncard_ok(int n) { return n >= 2 && n <= NCARD_MAX && NQH % n == 0 && NVH % n == 0 && FF / 32 >= n; }

// Value heads in key-head order (index i = k * (NVH / NKH) + s for value head s * NKH + k: a key head's value heads
// side by side), dealt in n equal runs: card c takes i in [c NVH / n, (c + 1) NVH / n). Its key heads are the ones
// those value heads belong to, so a run that starts or ends inside a key head's value heads puts that key head (its
// q and k) on both cards. Three cards: 16 value heads each, key heads 0-5, 5-10, 10-15
constexpr int VPK = NVH / NKH;   // value heads per key head
constexpr int vhead_card(int n, int v) { return ((v % NKH) * VPK + v / NKH) / (NVH / n); }
constexpr int khead_first(int n, int c) { return c * (NVH / n) / VPK; }
constexpr int khead_count(int n, int c) { return ((c + 1) * (NVH / n) - 1) / VPK - khead_first(n, c) + 1; }
// The FF width in whole 32-value blocks, the first cards one block more when they do not divide evenly: card c's
// slice [ff_lo, ff_hi). Three cards: 224, 224, 192
constexpr int ff_lo(int n, int c) { return 32 * (c * (FF / 32 / n) + (c < FF / 32 % n ? c : FF / 32 % n)); }
constexpr int ff_hi(int n, int c) { return ff_lo(n, c + 1); }

namespace split_check {
// the two- and three-card splits the lab's bases (and prod's logits) are made with, written out: the rule above must
// give exactly these, value heads in this order
constexpr int K3[3][2] = {{0, 6}, {5, 6}, {10, 6}};
constexpr int V3[3][16] = {{0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 32, 33, 34, 35, 36},
                           {6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41},
                           {11, 12, 13, 14, 15, 27, 28, 29, 30, 31, 42, 43, 44, 45, 46, 47}};
constexpr int F3[3][2] = {{0, 224}, {224, 448}, {448, 640}};
constexpr int K2[2][2] = {{0, 8}, {8, 8}};
constexpr int V2[2][24] = {{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39},
                           {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47}};
constexpr int F2[2][2] = {{0, 320}, {320, 640}};

// card c's value heads as the loader lists them (ascending) against a written-out list
template <int M>
constexpr bool vheads_are(int n, int c, const int (&want)[M]) {
    int j = 0;
    for (int v = 0; v < NVH; ++v)
        if (vhead_card(n, v) == c) {
            if (j >= M || want[j] != v) return false;
            ++j;
        }
    return j == M;
}
template <int NC, int M>
constexpr bool matches(const int (&k)[NC][2], const int (&vh)[NC][M], const int (&f)[NC][2]) {
    for (int c = 0; c < NC; ++c)
        if (khead_first(NC, c) != k[c][0] || khead_count(NC, c) != k[c][1] || !vheads_are(NC, c, vh[c]) ||
            ff_lo(NC, c) != f[c][0] || ff_hi(NC, c) != f[c][1])
            return false;
    return true;
}
// any count: every value head on one card beside its key head, the cards' value heads equal in number, key heads
// contiguous and covering all, FF slices whole blocks, none empty, covering the width in order
constexpr bool sound(int n) {
    for (int v = 0; v < NVH; ++v) {
        const int c = vhead_card(n, v), k = v % NKH;
        if (c < 0 || c >= n || k < khead_first(n, c) || k >= khead_first(n, c) + khead_count(n, c)) return false;
    }
    for (int c = 0; c < n; ++c) {
        int nv = 0;
        for (int v = 0; v < NVH; ++v) nv += vhead_card(n, v) == c;
        if (nv != NVH / n || khead_count(n, c) < 1) return false;
        const int prev_end = c ? khead_first(n, c - 1) + khead_count(n, c - 1) : 0;   // past the last card's key heads
        if (c && (khead_first(n, c) < prev_end - 1 || khead_first(n, c) > prev_end)) return false;
        if (ff_lo(n, c) % 32 || ff_hi(n, c) <= ff_lo(n, c)) return false;
    }
    return khead_first(n, 0) == 0 && khead_first(n, n - 1) + khead_count(n, n - 1) == NKH && ff_lo(n, 0) == 0 && ff_hi(n, n - 1) == FF;
}
constexpr bool all_sound() {
    for (int n = 2; n <= NCARD_MAX; ++n)
        if (ncard_ok(n) && !sound(n)) return false;
    return true;
}
}  // namespace split_check
static_assert(split_check::matches(split_check::K3, split_check::V3, split_check::F3), "the split rule moves three cards' split");
static_assert(split_check::matches(split_check::K2, split_check::V2, split_check::F2), "the split rule moves two cards' split");
static_assert(split_check::all_sound(), "the split rule breaks for a supported count");
static_assert(ncard_ok(2) && ncard_ok(3) && ncard_ok(4) && ncard_ok(6) && ncard_ok(8) && !ncard_ok(5) && !ncard_ok(7) && !ncard_ok(9),
              "the supported counts are 2, 3, 4, 6 and 8 (the Makefile's list)");

}  // namespace fnx
