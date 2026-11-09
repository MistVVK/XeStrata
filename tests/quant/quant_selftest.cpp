// tests/quant/quant_selftest.cpp - the layer without ggml: type table, probe, and every kernel against the reference (selfcheck.hpp).
#include <cstdio>
#include <cstring>

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/selfcheck.hpp"
#include "strata/quant/types.hpp"

namespace sq = strata::quant;

static int g_fail = 0;
#define CHECK(c, ...)                                  \
    do {                                               \
        if (!(c)) {                                    \
            ++g_fail;                                  \
            std::printf("FAIL line %d: ", __LINE__);   \
            std::printf(__VA_ARGS__);                  \
            std::printf("\n");                         \
        }                                              \
    } while (0)

int main() {
    const sq::Caps& c = sq::caps();
    std::printf("quant_selftest: best isa %s, default contract %s (avx=%d avx2=%d avx512=%d vnni=%d neon=%d)\n", sq::isa_name(sq::best_isa()),
                sq::arith_name(sq::default_arith()), c.avx, c.avx2, c.avx512, c.avx512_vnni, c.neon);

    // the table: sizes the GGUF format fixes, unknown ids answered nullptr
    CHECK(sq::row_bytes(12, 256) == 144, "q4_K row of 256 is 144 bytes");
    CHECK(sq::row_bytes(15, 512) == 584, "q8_K row of 512");
    CHECK(sq::row_bytes(8, 64) == 68, "q8_0 row of 64");
    CHECK(sq::row_bytes(42, 128) == 36, "q2_0 row of 128");
    CHECK(sq::row_bytes(12, 100) == 0, "not whole blocks");
    CHECK(sq::type_info(43) == nullptr && sq::type_info(-1) == nullptr && sq::type_info(100) == nullptr && sq::type_info(4) == nullptr, "unknown ids");
    CHECK(sq::type_info(sq::Type::IQ4_NL)->act == sq::Type::Q8_0, "iq4_nl takes q8_0");
    CHECK(sq::type_info(sq::Type::Q5_1)->act == sq::Type::Q8_1, "q5_1 takes q8_1");
    CHECK(sq::type_info(sq::Type::IQ2_S)->act == sq::Type::Q8_K, "iq2_s takes q8_K");

    // every dot kernel the layer has, that this CPU can run, against the Generic reference on exact data
    int dots = 0, acts = 0;
    for (int t = 0; t < 150; ++t) {
        const sq::TypeInfo* ti = sq::type_info(t);
        if (ti == nullptr || !ti->has_dot) continue;
        for (sq::Arith a : {sq::Arith::Generic, sq::Arith::Avx, sq::Arith::Avx2, sq::Arith::Neon}) {
            const sq::DotKernel* ks[16];
            const int n = sq::dot_kernels(ti->type, a, ks, 16);
            for (int i = 0; i < n; ++i) {
                if (!sq::isa_ok(ks[i]->isa)) continue;
                std::string why;
                const bool ok = sq::selfcheck(*ks[i], &why);
                CHECK(ok, "%s", why.c_str());
                ++dots;
            }
        }
        CHECK(sq::dot_kernel(ti->type, sq::Arith::Generic) != nullptr, "%s has no Generic dot", ti->name);
        CHECK(sq::checked_dot_kernel(ti->type, sq::Arith::Generic) != nullptr, "%s has no checked Generic kernel", ti->name);
        CHECK(sq::checked_dot_kernel(ti->type, sq::default_arith()) != nullptr, "%s has no checked kernel of the default contract", ti->name);
    }
    for (sq::Type t : {sq::Type::Q8_0, sq::Type::Q8_1, sq::Type::Q8_K})
        for (sq::Arith a : {sq::Arith::Generic, sq::Arith::Avx, sq::Arith::Avx2, sq::Arith::Neon}) {
            const sq::ActKernel* k = sq::act_kernel(t, a);
            if (k == nullptr || k->arith != a) continue;
            std::string why;
            CHECK(sq::selfcheck(*k, &why), "%s", why.c_str());
            ++acts;
        }
    for (sq::Type t : {sq::Type::Q8_0, sq::Type::Q8_1, sq::Type::Q8_K}) { CHECK(sq::checked_act_kernel(t, sq::Arith::Generic) != nullptr, "no checked Generic activation kernel"); CHECK(sq::checked_act_kernel(t, sq::default_arith()) != nullptr, "no checked activation kernel of the default contract"); }

    // the Q4_0 requantizer: a ramp of 32 values 0..31 -> d = max/-8 with max = 31 -> d = -3.875, ids from the ggml rule
    float ramp[32];
    for (int i = 0; i < 32; ++i) ramp[i] = (float) i;
    unsigned char blk[18];
    sq::quantize_q4_0_ref(ramp, blk, 32);
    CHECK(blk[0] == 0xC0 && blk[1] == 0xC3, "q4_0 scale of the ramp is fp16 -3.875 (0xC3C0), got %02x%02x", blk[1], blk[0]);
    CHECK(blk[2] == 0x48, "q4_0 first byte of the ramp is 0x48 (x=0 -> 8, x=16 -> 4), got %02x", blk[2]);

    std::printf("quant_selftest: %d dot kernels, %d activation kernels checked, %d failures -> %s\n", dots, acts, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
