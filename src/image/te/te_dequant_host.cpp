// src/image/te/te_dequant_host.cpp - the host side of include/strata/image/te_dequant.hpp: the row sizes and the host
// reference decoder (include/strata/artifact/dequant.hpp's f32 values, then f16_bits.hpp's round to nearest even).
//
// Build this file with contraction off (-ffp-contract=off; GCC on aarch64 fuses "d1 * q - m1" by default, MSVC and
// x86 GCC without -march do not): dequant.hpp rounds every product and difference on its own, like ggml.
#include "strata/image/te_dequant.hpp"

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <vector>

namespace strata::image {

namespace {
struct Geo { int elems, bytes; };
bool geo(uint32_t type, Geo& g) {
    switch (type) {
    case TE_F32: g = {1, 4}; return true;
    case TE_F16: g = {1, 2}; return true;
    case TE_BF16: g = {1, 2}; return true;
    case TE_Q8_0: g = {32, 34}; return true;
    case TE_Q4_K: g = {256, 144}; return true;
    case TE_Q5_K: g = {256, 176}; return true;
    case TE_Q6_K: g = {256, 210}; return true;
    case TE_IQ4_XS: g = {256, 136}; return true;
    default: return false;
    }
}
}  // namespace

bool te_type_supported(uint32_t type) {
    Geo g;
    return geo(type, g);
}

size_t te_row_bytes(uint32_t type, int64_t ne0) {
    Geo g;
    if (!geo(type, g) || ne0 <= 0 || ne0 % 32 != 0 || ne0 % g.elems != 0) return 0;   // the device path decodes groups of 32
    return (size_t) (ne0 / g.elems) * (size_t) g.bytes;
}

std::string te_dequant_f32_host(uint32_t type, const uint8_t* src, int64_t ne0, int64_t rows, float* dst) {
    const size_t rb = te_row_bytes(type, ne0);
    if (rb == 0) return "te_dequant: type " + std::to_string(type) + " with rows of " + std::to_string(ne0) + " values is not supported";
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row = src + (size_t) r * rb;
        float* y = dst + r * ne0;
        switch (type) {
        case TE_Q4_K: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_q4_K(row + b * 144, y + b * 256); break;
        case TE_Q5_K: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_q5_K(row + b * 176, y + b * 256); break;
        case TE_Q6_K: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_q6_K(row + b * 210, y + b * 256); break;
        case TE_IQ4_XS: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_iq4_xs(row + b * 136, y + b * 256); break;
        case TE_Q8_0: for (int64_t b = 0; b < ne0 / 32; ++b) strata::dequantize_q8_0(row + b * 34, y + b * 32); break;
        case TE_F32: strata::dequantize_f32(row, y, (int) ne0); break;
        case TE_BF16: strata::dequantize_bf16(row, y, (int) ne0); break;
        case TE_F16: strata::dequantize_f16(row, y, (int) ne0); break;
        default: return "te_dequant: type not supported";
        }
    }
    return std::string();
}

std::string te_dequant_f16_host(uint32_t type, const uint8_t* src, int64_t ne0, int64_t rows, uint16_t* dst) {
    const size_t rb = te_row_bytes(type, ne0);
    if (rb == 0) return "te_dequant: type " + std::to_string(type) + " with rows of " + std::to_string(ne0) + " values is not supported";
    std::vector<float> f((size_t) ne0);
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row = src + (size_t) r * rb;
        float* y = f.data();
        switch (type) {
        case TE_Q4_K: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_q4_K(row + b * 144, y + b * 256); break;
        case TE_Q5_K: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_q5_K(row + b * 176, y + b * 256); break;
        case TE_Q6_K: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_q6_K(row + b * 210, y + b * 256); break;
        case TE_IQ4_XS: for (int64_t b = 0; b < ne0 / 256; ++b) strata::dequantize_iq4_xs(row + b * 136, y + b * 256); break;
        case TE_Q8_0: for (int64_t b = 0; b < ne0 / 32; ++b) strata::dequantize_q8_0(row + b * 34, y + b * 32); break;
        case TE_F32: strata::dequantize_f32(row, y, (int) ne0); break;
        case TE_BF16: strata::dequantize_bf16(row, y, (int) ne0); break;
        case TE_F16:
            for (int64_t i = 0; i < ne0; ++i) dst[r * ne0 + i] = (uint16_t) (row[2 * i] | (row[2 * i + 1] << 8));
            continue;
        default: return "te_dequant: type not supported";
        }
        for (int64_t i = 0; i < ne0; ++i) dst[r * ne0 + i] = strata::kernels::f16_from_f32(y[i]);
    }
    return std::string();
}

}  // namespace strata::image
