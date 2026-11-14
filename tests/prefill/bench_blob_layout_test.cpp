// bench_blob_layout (tests/prefill/bench_blob_layout.hpp): the byte prefill_expert_paths_bench changes per expert is never a scale byte, for every
// format in the table.  No GPU, no model, no ggml.
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "bench_blob_layout.hpp"

using strata::bench_blob::BlockLayout;
using strata::bench_blob::is_scale;
using strata::bench_blob::layout_of;
using strata::bench_blob::perturb_offset;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const int types[] = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 40, 42};
    for (int t : types) {
        BlockLayout L;
        CHECK(layout_of(t, L));
        CHECK(L.type == t && L.bytes > 0 && L.values > 0);
        int data = 0;
        for (int b = 0; b < L.bytes; ++b) data += !is_scale(L, b);
        CHECK(data > 0 && data < L.bytes);   // every format has a scale and quant bytes
        for (const auto& s : L.scale) CHECK(s.begin >= 0 && s.begin < s.end && s.end <= L.bytes);

        // a base blob of 40 blocks (the bench's gate row of 2,560 values is 10 to 80 blocks), every expert of a 512-expert layer
        const size_t n = (size_t) L.bytes * 40;
        std::vector<uint8_t> base(n);
        uint32_t x = 2463534242u + (uint32_t) t;
        for (uint8_t& v : base) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; v = (uint8_t) x; }
        std::set<size_t> offs;
        for (int e = 0; e < 512; ++e) {
            const size_t o = perturb_offset(L, e);
            CHECK(o >= 4 && o < n);
            CHECK(!is_scale(L, (int) (o % (size_t) L.bytes)));
            offs.insert(o);
            std::vector<uint8_t> b = base;
            b[o] ^= (uint8_t) e;
            int differ = 0, scale_differ = 0;
            for (size_t i = 0; i < n; ++i) {
                if (b[i] == base[i]) continue;
                ++differ;
                scale_differ += is_scale(L, (int) (i % (size_t) L.bytes));
            }
            CHECK(scale_differ == 0);                              // the scales stay bit-equal to the base
            CHECK(differ == ((uint8_t) e != 0 ? 1 : 0));           // one quant byte, or none where e is a multiple of 256
        }
        CHECK(offs.size() == 16);                                  // sixteen distinct places, as the fixed offsets had
    }

    // Q2_0: the old offsets 18 and 19 were block 1's scale; the walk takes 4..17, then 20 and 21
    BlockLayout q;
    CHECK(layout_of(42, q));
    CHECK(is_scale(q, 18 % q.bytes) && is_scale(q, 19 % q.bytes));
    CHECK(perturb_offset(q, 0) == 4 && perturb_offset(q, 13) == 17 && perturb_offset(q, 14) == 20 && perturb_offset(q, 15) == 21);
    CHECK(perturb_offset(q, 31) == 21 && perturb_offset(q, 16) == 4);
    // IQ4_NL as a gate/up type had the same fault (18-byte blocks)
    BlockLayout nl;
    CHECK(layout_of(20, nl) && perturb_offset(nl, 14) == 20);
    // formats whose bytes 4..19 hold no scale bits keep the old offsets (and so the blobs of earlier runs)
    for (int t : {6, 7, 8, 14, 17, 18, 21, 22, 29}) {
        BlockLayout L;
        CHECK(layout_of(t, L));
        for (int e = 0; e < 16; ++e) CHECK(perturb_offset(L, e) == (size_t) (4 + e));
    }
    // IQ2_XXS: byte 9 and 17 carry a group's scale nibble, so the walk steps over them
    BlockLayout xx;
    CHECK(layout_of(16, xx));
    CHECK(is_scale(xx, 9) && is_scale(xx, 17) && perturb_offset(xx, 4) == 8 && perturb_offset(xx, 5) == 10);
    // IQ4_XS: scales_l sits at 4..7, so the walk starts at 8
    BlockLayout x4;
    CHECK(layout_of(23, x4) && perturb_offset(x4, 0) == 8);
    // a type the table does not know is refused, not guessed
    BlockLayout u;
    CHECK(!layout_of(24, u) && !layout_of(144, u) && !layout_of(-1, u));

    if (failed) {
        std::fprintf(stderr, "bench_blob_layout_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("bench_blob_layout_test: OK\n");
    return 0;
}
