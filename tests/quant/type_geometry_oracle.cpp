// tests/quant/type_geometry_oracle.cpp - Strata's type table (strata::quant::type_info) against ggml's own table, id by id.
//
// The engine's prompt buffers size an expert matrix from this table (src/prefill/moe_helpers.cu, mmq::matrix_bytes), which used to ask
// ggml (ggml_blck_size / ggml_type_size).  ggml is the oracle here only: the test is built where ggml-base is (STRATA_BUILD_TESTS).
#include <cstdio>

#include "ggml.h"
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
    int same = 0, ggml_only = 0;
    // every id of ggml's table: where Strata's table knows the id, the block geometry is ggml's
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const int64_t gb = ggml_blck_size((ggml_type) t);
        const size_t gs = ggml_type_size((ggml_type) t);
        const sq::TypeInfo* ti = sq::type_info(t);
        if (ti == nullptr) {
            if (gb > 0) {
                ++ggml_only;
                std::printf("  id %d (%s): in ggml's table only (%lld values, %zu bytes)\n", t, ggml_type_name((ggml_type) t), (long long) gb, gs);
            }
            continue;
        }
        CHECK(gb == ti->blck && gs == ti->bytes, "id %d (%s): Strata %d values / %zu bytes, ggml %lld / %zu", t, ti->name, ti->blck, ti->bytes,
              (long long) gb, gs);
        ++same;
    }
    // the trellis types, which ggml's table does not have: the geometry moe_helpers.cu had as constants
    const int tq_ids[] = {144, 145, 146, 147, 148, 149};
    const size_t tq_bytes[] = {34, 26, 30, 42, 50, 58};
    for (int i = 0; i < 6; ++i) {
        const sq::TypeInfo* ti = sq::type_info(tq_ids[i]);
        CHECK(ti != nullptr && ti->blck == 128 && ti->bytes == tq_bytes[i], "trellis id %d: 128 values / %zu bytes", tq_ids[i], tq_bytes[i]);
    }
    // ids in neither table are unknown, never read out of range
    const int unknown[] = {-1, 4, 5, (int) GGML_TYPE_COUNT, 143, 150, 1000};
    for (int t : unknown) CHECK(sq::type_info(t) == nullptr, "id %d unknown", t);
    std::printf("type_geometry_oracle: %d ids equal to ggml's, %d in ggml's table only, %d failures\n", same, ggml_only, g_fail);
    return g_fail == 0 ? 0 : 1;
}
