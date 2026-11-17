// src/quant/types.cpp - the quant type table.
#include "strata/quant/types.hpp"

#include "internal.hpp"

namespace strata::quant {
namespace {

constexpr Type kQ80 = Type::Q8_0, kQ81 = Type::Q8_1, kQ8K = Type::Q8_K;

// id order; the table is searched by id (sparse ids: 0..42 dense, 144..149 trellis)
const TypeInfo kTable[] = {
    {Type::F32,     "f32",     1,   4,                         Type::F32,     false, false},
    {Type::F16,     "f16",     1,   2,                         Type::F16,     false, false},
    {Type::Q4_0,    "q4_0",    QK4_0, sizeof(block_q4_0),      kQ80,          true,  false},
    {Type::Q4_1,    "q4_1",    QK4_1, sizeof(block_q4_1),      kQ81,          true,  false},
    {Type::Q5_0,    "q5_0",    QK5_0, sizeof(block_q5_0),      kQ80,          true,  false},
    {Type::Q5_1,    "q5_1",    QK5_1, sizeof(block_q5_1),      kQ81,          true,  false},
    {Type::Q8_0,    "q8_0",    QK8_0, sizeof(block_q8_0),      kQ80,          true,  true},
    {Type::Q8_1,    "q8_1",    QK8_1, sizeof(block_q8_1),      kQ81,          false, true},
    {Type::Q2_K,    "q2_K",    QK_K, sizeof(block_q2_K),       kQ8K,          true,  false},
    {Type::Q3_K,    "q3_K",    QK_K, sizeof(block_q3_K),       kQ8K,          true,  false},
    {Type::Q4_K,    "q4_K",    QK_K, sizeof(block_q4_K),       kQ8K,          true,  false},
    {Type::Q5_K,    "q5_K",    QK_K, sizeof(block_q5_K),       kQ8K,          true,  false},
    {Type::Q6_K,    "q6_K",    QK_K, sizeof(block_q6_K),       kQ8K,          true,  false},
    {Type::Q8_K,    "q8_K",    QK_K, sizeof(block_q8_K),       kQ8K,          false, true},
    {Type::IQ2_XXS, "iq2_xxs", QK_K, sizeof(block_iq2_xxs),    kQ8K,          true,  false},
    {Type::IQ2_XS,  "iq2_xs",  QK_K, sizeof(block_iq2_xs),     kQ8K,          true,  false},
    {Type::IQ3_XXS, "iq3_xxs", QK_K, sizeof(block_iq3_xxs),    kQ8K,          true,  false},
    {Type::IQ1_S,   "iq1_s",   QK_K, sizeof(block_iq1_s),      kQ8K,          true,  false},
    {Type::IQ4_NL,  "iq4_nl",  QK4_NL, sizeof(block_iq4_nl),   kQ80,          true,  false},
    {Type::IQ3_S,   "iq3_s",   QK_K, sizeof(block_iq3_s),      kQ8K,          true,  false},
    {Type::IQ2_S,   "iq2_s",   QK_K, sizeof(block_iq2_s),      kQ8K,          true,  false},
    {Type::IQ4_XS,  "iq4_xs",  QK_K, sizeof(block_iq4_xs),     kQ8K,          true,  false},
    {Type::IQ1_M,   "iq1_m",   QK_K, sizeof(block_iq1_m),      kQ8K,          true,  false},
    {Type::BF16,    "bf16",    1,   2,                         Type::BF16,    false, false},
    {Type::MXFP4,   "mxfp4",   QK_MXFP4, sizeof(block_mxfp4),  kQ80,          true,  false},
    {Type::NVFP4,   "nvfp4",   QK_NVFP4, sizeof(block_nvfp4),  kQ80,          true,  false},
    {Type::Q1_0,    "q1_0",    QK1_0, sizeof(block_q1_0),      kQ80,          false, false},
    {Type::Q2_0,    "q2_0",    QK2_0, sizeof(block_q2_0),      kQ80,          true,  false},
    // the trellis types: 128 values a block, 2 + 4K bytes (2 + 8K for the pair layout); their rows are tq_cpu.hpp's
    {Type::TQ2_T,   "tq2_t",   128, 34,                        kQ80,          false, false},
    {Type::TQK6,    "tqk6",    128, 26,                        kQ80,          false, false},
    {Type::TQK7,    "tqk7",    128, 30,                        kQ80,          false, false},
    {Type::TQP5,    "tqp5",    128, 42,                        kQ80,          false, false},
    {Type::TQP6,    "tqp6",    128, 50,                        kQ80,          false, false},
    {Type::TQP7,    "tqp7",    128, 58,                        kQ80,          false, false},
};

}  // namespace

const TypeInfo* type_info(int type) noexcept {
    for (const TypeInfo& t : kTable)
        if ((int) t.type == type) return &t;
    return nullptr;
}

size_t row_bytes(int type, int64_t n) noexcept {
    const TypeInfo* t = type_info(type);
    if (t == nullptr || n < 0 || n % t->blck != 0) return 0;
    return (size_t) (n / t->blck) * t->bytes;
}

}  // namespace strata::quant
