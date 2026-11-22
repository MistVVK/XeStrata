// Compare the fused expert dequantizer to the independent GU and down kernels.
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
int check_format(sycl::queue& q, int type, size_t block_bytes) {
    constexpr int nf = 640, ne = 2560;
    constexpr size_t elements = nf * ne, blocks = elements / 256;
    const size_t gu_bytes = blocks * block_bytes;
    constexpr size_t down_bytes = (elements / 64) * 18;
    constexpr size_t guard = 128;
    constexpr uint16_t sentinel = 0x7d55;

    std::mt19937 rng(4321);
    std::vector<uint8_t> bytes(2 * gu_bytes + down_bytes);
    for (auto& byte : bytes) byte = uint8_t(rng());
    // Keep the scale finite while varying the codebook indices, signs and scales.
    auto set_scales = [&](size_t offset, size_t end, size_t stride) {
        for (size_t i = offset; i < end; i += stride) {
            bytes[i] = uint8_t(rng());
            bytes[i + 1] = uint8_t(0x20 + rng() % 0x20);
        }
    };
    if (type == 29) {
        // IQ1_M distributes the FP16 scale across four high nibbles. The
        // fourth 16-bit word supplies bits 12..15 of the assembled scale.
        for (size_t i = block_bytes - 1; i < 2 * gu_bytes; i += block_bytes)
            bytes[i] = uint8_t((bytes[i] & 0x0f) | 0x30);
    } else {
        set_scales(0, 2 * gu_bytes, block_bytes);
    }
    set_scales(2 * gu_bytes, bytes.size(), 18);

    auto* blob = sycl::malloc_device<uint8_t>(bytes.size(), q);
    q.memcpy(blob, bytes.data(), bytes.size()).wait_and_throw();
    const size_t total = 3 * elements + 3 * guard;
    auto* fused = sycl::malloc_device<uint16_t>(total, q);
    auto* reference = sycl::malloc_device<uint16_t>(total, q);
    q.fill(fused, sentinel, total);
    q.fill(reference, sentinel, total);
    const size_t down = 2 * elements + 2 * guard;
    strata::kernels::iq_dequant_gu_f16(type, blob, blob + gu_bytes, nf, ne, reference + guard, &q);
    strata::kernels::iq_dequant_f16(42, blob + 2 * gu_bytes, elements, reference + down, &q);
    strata::kernels::iq_dequant_expert_f16(type, 42, blob, blob + gu_bytes, blob + 2 * gu_bytes,
                                           nf, ne, fused + guard, fused + down, &q);
    std::vector<uint16_t> actual(total), expected(total);
    q.memcpy(actual.data(), fused, total * sizeof(uint16_t));
    q.memcpy(expected.data(), reference, total * sizeof(uint16_t));
    q.wait_and_throw();
    sycl::free(reference, q);
    sycl::free(fused, q);
    sycl::free(blob, q);
    for (size_t i = 0; i < total; ++i) {
        const bool padding = i < guard || (i >= down - guard && i < down) || i >= down + elements;
        if (actual[i] != expected[i] || (padding && actual[i] != sentinel)) {
            std::fprintf(stderr, "expert type %d mismatch at half %zu: %04x vs %04x\n", type, i, actual[i], expected[i]);
            return 1;
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    auto& queue = dpct::get_in_order_queue();
    if (argc == 1) return check_format(queue, 16, 66);
    if (argc != 2 || std::strcmp(argv[1], "--native") != 0) return 2;
    const auto device = queue.get_device();
    if (queue.get_backend() != sycl::backend::ext_oneapi_level_zero ||
        strata::intel_gpu_gen(device) != strata::IntelGpuGen::Battlemage) return 77;
    if (check_format(queue, 22, 82) != 0) return 1;
    return check_format(queue, 29, 56);
}
