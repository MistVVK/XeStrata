// q36_prompt_route (include/strata/kernels/q36_prompt_route.hpp): which kernels carry the prompt experts of one Qwen3.6 layer: fused > int8 (switch set to 1) > MMQ > int8 > grouped.
// No GPU, no model.
#include <cstdio>

#include "strata/kernels/q36_prompt_route.hpp"

using namespace strata::kernels;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the build WITH MMQ, nothing set: fused where it covers, else MMQ; a layer MMQ has no tile for goes to int8 when it covers it, else grouped
    CHECK(q36_layer_route(true, -1, false, true, true) == Q36Route::Fused);
    CHECK(q36_layer_route(false, -1, false, true, true) == Q36Route::Mmq);
    CHECK(q36_layer_route(false, -1, false, true, false) == Q36Route::Mmq);
    CHECK(q36_layer_route(false, -1, false, false, true) == Q36Route::Int8);
    CHECK(q36_layer_route(false, -1, false, false, false) == Q36Route::Grouped);
    // the build WITH MMQ, STRATA_EXPERT_I8=1: int8 before MMQ where it covers the layer (the A/B arm), MMQ where it does not; the fused kernels still come first
    CHECK(q36_layer_route(false, 1, false, true, true) == Q36Route::Int8);
    CHECK(q36_layer_route(false, 1, false, true, false) == Q36Route::Mmq);
    CHECK(q36_layer_route(true, 1, false, true, true) == Q36Route::Fused);
    // the build WITH MMQ, STRATA_EXPERT_I8=0: never int8
    CHECK(q36_layer_route(false, 0, false, true, true) == Q36Route::Mmq);
    CHECK(q36_layer_route(false, 0, false, false, true) == Q36Route::Grouped);
    // STRATA_Q36_NO_MMQ in a build with MMQ (mmq_takes false), nothing set: int8 where it covers (it used to be the grouped kernels), grouped otherwise
    CHECK(q36_layer_route(false, -1, false, false, true) == Q36Route::Int8);
    // the build WITHOUT MMQ, nothing set (reads as 1): fused > int8 > grouped, whatever the card
    CHECK(q36_layer_route(true, -1, true, false, true) == Q36Route::Fused);
    CHECK(q36_layer_route(false, -1, true, false, true) == Q36Route::Int8);
    CHECK(q36_layer_route(false, -1, true, false, false) == Q36Route::Grouped);
    // the build WITHOUT MMQ, STRATA_EXPERT_I8=0: the way back to the grouped kernels; =1 is the same as unset
    CHECK(q36_layer_route(false, 0, true, false, true) == Q36Route::Grouped);
    CHECK(q36_layer_route(false, 1, true, false, true) == Q36Route::Int8);
    CHECK(q36_layer_route(true, 0, true, false, true) == Q36Route::Fused);   // STRATA_PF_FUSED decides the fused kernels, not this switch
    // one layer on the grouped kernels sends the whole prompt path there
    CHECK(q36_prompt_path_ok(41, 0));
    CHECK(!q36_prompt_path_ok(41, 1));
    CHECK(!q36_prompt_path_ok(0, 0));
    // the few-rows route: with the limit 7 and 256 experts a chunk takes int8 from 7 * 256 = 1,792 routed entries (224 tokens of 8 picks), below that the grouped kernels
    CHECK(!q36_i8_chunk_takes(1791, 256, 7));
    CHECK(q36_i8_chunk_takes(1792, 256, 7));
    CHECK(q36_i8_chunk_takes(16384, 256, 7));    // 2,048 tokens
    CHECK(q36_i8_chunk_takes(100, 256, 0));      // no route: always int8
    CHECK(q36_i8_chunk_takes(1791, 256, 6));   // limit 6 (a Tesla P100): 6 * 256 = 1,536 <= 1,791
    CHECK(!q36_i8_chunk_takes(1535, 256, 6));
    if (failed) { std::fprintf(stderr, "q36_prompt_route_test: %d failed\n", failed); return 1; }
    std::printf("q36_prompt_route_test: all ok\n");
    return 0;
}
