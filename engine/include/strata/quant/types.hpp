// include/strata/quant/types.hpp - the quant type table (own-stack phase 2): ggml's type ids, block geometry, activation type.
//
// The ids are the GGUF file format's (ggml_type), so a GGUF's type field and a pack's type field are valid here unchanged.  An id this
// table does not know is answered with nullptr, never read out of range (an upstream ggml indexes its table by the id).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::quant {

enum class Type : int32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    IQ1_M = 29, BF16 = 30, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41, Q2_0 = 42,
    // Strata's trellis types (Agention ids): their rows are tq_cpu.hpp's, the table only knows their geometry
    TQ2_T = 144, TQK6 = 145, TQK7 = 146, TQP5 = 147, TQP6 = 148, TQP7 = 149,
};

struct TypeInfo {
    Type type;
    const char* name;     ///< ggml's name ("q4_K", "iq3_s", ...)
    int blck;             ///< elements per block (QK)
    size_t bytes;         ///< bytes per block
    Type act;             ///< the activation format its dot takes (ggml's vec_dot_type); itself for activation formats and floats
    bool has_dot;         ///< the layer has a dot product with this type as the weights
    bool is_act;          ///< Q8_0, Q8_1, Q8_K
};

/// nullptr for an id the table does not know.
const TypeInfo* type_info(int type) noexcept;
inline const TypeInfo* type_info(Type t) noexcept { return type_info((int) t); }

/// Bytes of a row of n elements (0 when the type is unknown or n is not a whole number of blocks).
size_t row_bytes(int type, int64_t n) noexcept;

/// The activation formats.
constexpr int kBlockQ8_0 = 34, kBlockQ8_1 = 36, kBlockQ8_K = 292;

}  // namespace strata::quant
