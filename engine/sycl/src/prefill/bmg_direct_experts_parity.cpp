#include "bmg_direct_experts.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/bmg_gate_up.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace direct = strata::prefill::bmg_direct;
constexpr int width = 2560, hidden = 640, rows = 32, guard = 64;

std::vector<uint8_t> packed_matrix(int type, size_t elements, std::mt19937& random) {
    const size_t block_size = type == 42 ? 18 : type == 16 ? 66 : type == 22 ? 82 : 56;
    std::vector<uint8_t> bytes(elements / (type == 42 ? 64 : 256) * block_size);
    for (auto& value : bytes) value = uint8_t(random());
    for (size_t i = 0; i < bytes.size(); i += block_size) {
        // Finite scales, including signed zero. IQ1_M splits its FP16 scale across four words.
        const uint16_t scale = uint16_t((random() & 1 ? 0x8000u : 0u) | ((random() % 4) * 0x0c00u));
        if (type != 29) {
            std::memcpy(bytes.data() + i, &scale, sizeof(scale));
        } else {
            for (unsigned j = 0; j < 4; ++j) {
                auto& high = bytes[i + 49 + j * 2];
                high = uint8_t((high & 15u) | (((scale >> (4*j)) & 15u) << 4));
            }
        }
    }
    return bytes;
}

bool compare(sycl::queue& q, strata::prefill::Gemm& gemm, int type, std::mt19937& random) {
    constexpr int gu_elems = 2 * hidden * width, d_elems = hidden * width;
    constexpr int hidden_stride = rows * hidden + 2 * guard, output_stride = rows * width + 2 * guard;
    const std::array<int, 4> counts{1, 7, 16, 32};
    constexpr size_t count = counts.size();
    auto* weights_gu = sycl::malloc_device<uint16_t>(gu_elems, q);
    auto* weights_d = sycl::malloc_device<uint16_t>(d_elems, q);
    auto* projected = sycl::malloc_device<float>(rows * 2 * hidden, q);
    auto* x = sycl::malloc_device<uint16_t>(count * rows * width, q);
    std::vector<uint16_t> input(count * rows * width);
    for (auto& item : input) {
        item = sycl::bit_cast<uint16_t>(sycl::half(float(int(random() % 4097) - 2048) / 256.0f));
    }
    q.memcpy(x, input.data(), input.size() * sizeof(input[0]));
    auto* h = sycl::malloc_device<uint16_t>(count * hidden_stride, q);
    auto* y = sycl::malloc_device<float>(count * output_stride, q);
    auto* href = sycl::malloc_device<uint16_t>(count * hidden_stride, q);
    auto* yref = sycl::malloc_device<float>(count * output_stride, q);
    q.fill(href, uint16_t(0x7d55), count * hidden_stride);
    q.fill(yref, -371.5f, count * output_stride);
    direct::Batch batch{};
    batch.count = count;
    std::vector<uint8_t*> allocations;
    auto upload = [&](int format, size_t elements) {
        auto bytes = packed_matrix(format, elements, random);
        uint8_t* ptr = sycl::malloc_device<uint8_t>(bytes.size(), q);
        q.memcpy(ptr, bytes.data(), bytes.size()).wait_and_throw();
        allocations.push_back(ptr);
        return ptr;
    };
    for (int i = 0; i < int(count); ++i) {
        auto& e = batch.experts[i];
        e = {x + i * rows * width, upload(type, d_elems), upload(type, d_elems), upload(42, d_elems),
             h + i * hidden_stride + guard, y + i * output_stride + guard, counts[i]};
        auto* hr = href + i * hidden_stride + guard;
        strata::kernels::iq_dequant_expert_f16(type, 42, e.gate, e.up, e.down, hidden, width, weights_gu, weights_d, &q);
        gemm.f16(e.x, weights_gu, projected, e.rows, 2 * hidden, width);
        strata::prefill::swiglu_interleaved(projected, hr, e.rows, &q);
        gemm.f16(hr, weights_d, yref + i * output_stride + guard, e.rows, width, hidden);
    }
    std::vector<uint16_t> expected_h(count * hidden_stride), actual_h(expected_h.size());
    std::vector<float> expected_y(count * output_stride), actual_y(expected_y.size());
    q.memcpy(expected_h.data(), href, expected_h.size() * sizeof(*href));
    q.memcpy(expected_y.data(), yref, expected_y.size() * sizeof(*yref)).wait_and_throw();
    bool ok = true;
    for (int layout : {0, 1, 3}) {
        q.fill(h, uint16_t(0x7d55), count * hidden_stride);
        q.fill(y, -371.5f, count * output_stride);
        direct::gate(q, batch, type, layout);
        direct::down(q, batch, layout);
        q.memcpy(actual_h.data(), h, actual_h.size() * sizeof(*h));
        q.memcpy(actual_y.data(), y, actual_y.size() * sizeof(*y)).wait_and_throw();
        if (std::memcmp(actual_h.data(), expected_h.data(), count * hidden_stride * sizeof(*h)) ||
            std::memcmp(actual_y.data(), expected_y.data(), count * output_stride * sizeof(*y))) {
            std::fprintf(stderr, "direct expert parity failed at format %d, layout %d\n", type, layout);
            ok = false;
        }
    }
    for (auto ptr : allocations) sycl::free(ptr, q);
    for (void* ptr : std::array<void*, 8>{weights_gu, weights_d, projected, x, h, y, href, yref}) sycl::free(ptr, q);
    return ok;
}
} // namespace

int main() {
    auto& q = dpct::get_in_order_queue();
    if (!strata::prefill::bmg_gate_up::supported(q)) return 77;
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init(&q, 0, error)) return 1;
    std::mt19937 random(17);
    for (int type : {16, 22, 29}) if (!compare(q, gemm, type, random)) return 1;
    return 0;
}
