// tests/program/split_walk_test.cpp - the layer split's spill walk (include/strata/program/split_walk.hpp): who is short, who is not, and what is given up in which order.
#include "strata/program/split_walk.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

using namespace strata::program::split_walk;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
// IQ3_S on Qwen3.8-Flash-Next, 128K, int8 KV, window 32768, MTP: the figures the 8 GB start log gave (3485 MiB dense set, 835 MiB draft layer, 410 MiB head)
Model iq3s() {
    Model m;
    m.dense_mib = 1466 + 2019;
    m.session_mib = 400;
    m.head_mib = 410;
    m.drafter_mib = 1000;
    m.slot_mib = 3;
    m.reserve_mib = 796;
    return m;
}
}  // namespace

int main() {
    const Model m = iq3s();
    // 1. the trim: 8 GB cards are short, 12 GB and bigger are not (a big card never reaches the walk)
    check(trim_needed(7635, m, true), "an 8 GB card holding the head and the drafter is short");
    check(trim_needed(7635, m, false), "an 8 GB card without them is short too");
    check(!trim_needed(12288, m, true), "a 12 GB card is not short");
    check(!trim_needed(16384, m, true), "a 16 GB card is not short");
    check(!trim_needed(97000, m, true), "a 96 GB card is never short");
    check(!trim_needed(0, m, true), "an unreadable card (0 free) decides nothing");
    check(untrimmed_need_mib(m, true) > untrimmed_need_mib(m, false), "the last stage needs more");
    // monotone in the card: the threshold is one number
    int64_t edge = 0;
    for (int64_t f = 1000; f < 40000; ++f)
        if (!trim_needed(f, m, true)) { edge = f; break; }
    check(edge == untrimmed_need_mib(m, true), "the threshold is the need");
    for (int64_t f = edge; f < edge + 5000; f += 97) check(!trim_needed(f, m, true), "never short above the threshold");
    // a model with a smaller dense set (a smaller quant) is short later
    Model small = m;
    small.dense_mib = 1500;
    check(!trim_needed(7635, small, false), "a small dense set fits an 8 GB card untrimmed");
    // 2. fewer cards
    check(!card_usable(2048), "a card with 2 GB free leaves the split");
    check(card_usable(4250), "a 6 GB card (4.15 GiB free) stays: the other steps make it fit");
    check(card_usable(7635), "an 8 GB card stays");
    check(card_usable(kMinCardMib) && !card_usable(kMinCardMib - 1), "the edge");
    // 3. the draft layer: the single card's arithmetic, unchanged
    const uint64_t layer = 835ull << 20, head = 410ull << 20;
    check(draft_layer_need_bytes(layer, head) == layer + head + (220ull << 20) + (1550ull << 20), "the single card's need");
    check(draft_layer_short(1830ull << 20, layer, head), "1.8 GiB free: the layer goes");
    check(!draft_layer_short(3640ull << 20, layer, head), "3.6 GiB free (trimmed 8 GB card): the layer stays");
    check(!draft_layer_short(0, layer, head), "0 free is an unreadable figure");
    check(dflash2_need_bytes(715ull << 20, 213ull << 20) < draft_layer_need_bytes(layer, head), "DFlash2 needs less than the draft layer");
    // 4. the KV window walks 32768 -> 20480 -> 8192 -> 4096 -> stop, and never goes past the context or the engine's minimum
    check(next_kv_window(32768, 131072, 4096) == 20480, "32768 -> 20480");
    check(next_kv_window(20480, 131072, 4096) == 8192, "20480 -> 8192");
    check(next_kv_window(8192, 131072, 4096) == 4096, "8192 -> 4096");
    check(next_kv_window(4096, 131072, 4096) == 0, "4096 is the last");
    check(next_kv_window(0, 131072, 4096) == 20480, "the whole context in VRAM goes to 20480");
    check(next_kv_window(0, 16384, 4096) == 8192, "a context of 16K has no 20480 step");
    check(next_kv_window(0, 4096, 4096) == 0, "a context at the minimum has none");
    check(next_kv_window(8192, 131072, 8192) == 0, "a larger engine minimum stops the walk");
    // 5. the shares of the layers
    int64_t fm[2] = {5000, 5000}, out[2] = {0, 0};
    proportional_layers(fm, 2, 48, out);
    check(out[0] == 24 && out[1] == 24, "two equal rooms share 48 layers 24/24");
    int64_t fm1[2] = {5285, 3875};
    proportional_layers(fm1, 2, 48, out);
    check(out[0] + out[1] == 48 && out[0] == 28 && out[1] == 20, "the card that carries the head and the drafter takes fewer layers (28/20)");
    int64_t fm2[2] = {9000, 5000};
    proportional_layers(fm2, 2, 48, out);
    check(out[0] + out[1] == 48 && out[0] > out[1], "the bigger room takes more layers");
    int64_t fm3[3] = {5000, 5000, 5000}, out3[3];
    proportional_layers(fm3, 3, 48, out3);
    check(out3[0] == 16 && out3[1] == 16 && out3[2] == 16, "three equal rooms 16/16/16");
    int64_t fm4[2] = {90000, 100}, out4[2];
    proportional_layers(fm4, 2, 48, out4);
    check(out4[0] + out4[1] == 48 && out4[0] >= 2 && out4[1] >= 0, "a tiny room leaves the sum right");
    int64_t fm5[2] = {0, 0}, out5[2];
    proportional_layers(fm5, 2, 48, out5);
    check(out5[0] + out5[1] == 48, "rooms of 0 still sum to the layers");
    // 6. fewer cards: feasible when the rooms plus the relief pay for every layer
    int64_t r8[2] = {5200, 5200};
    check(split_feasible(r8, 2, 48, 90, 1410), "two 8 GB cards (5.2 GiB of room each) hold the model");
    int64_t r6[2] = {1744, 1552};
    check(split_feasible(r6, 2, 48, 87, 1410), "two 6 GB cards hold it only with the drafter and the head moved off");
    check(!split_feasible(r6, 2, 48, 87, 0), "...and not without that relief");
    int64_t r3[2] = {2000, 100};
    check(!split_feasible(r3, 2, 48, 90, 1410), "two cards with 2 GB and 0.1 GB of room do not: fewer cards");
    if (g_fail == 0) std::printf("split_walk_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
