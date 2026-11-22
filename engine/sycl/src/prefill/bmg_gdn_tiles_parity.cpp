#include "bmg_gdn_tiles.hpp"
#include "strata/prefill/kernels.hpp"
#include "bmg_gdn_test_timing.hpp"
#include <cstring>
#include <random>
#include <vector>

namespace {
template <class T> T* operand(sycl::queue& q, size_t count, std::mt19937& rng) {
    std::vector<T> v(count);
    for (auto& item : v) {
        const float f = float(int(rng() % 8193) - 4096) / 997.0f;
        if constexpr (sizeof(T) == sizeof(float))
            item = f;
        else
            item = uint16_t(sycl::bit_cast<uint32_t>(f) >> 16);
    }
    auto* out = sycl::malloc_device<T>(count, q);
    q.memcpy(out, v.data(), count * sizeof(T)).wait_and_throw();
    return out;
}
// Check raw bits including the untouched half of the strided projection and the convolution history.
bool equal(sycl::queue& q, const float* a, const float* b, size_t count) {
    constexpr size_t block = 256 * 1024;
    std::vector<float> left(block), right(block);
    for (size_t i = 0; i < count; i += block) {
        const size_t bytes = std::min(block, count - i) * sizeof(float);
        q.memcpy(left.data(), a + i, bytes);
        q.memcpy(right.data(), b + i, bytes).wait_and_throw();
        if (std::memcmp(left.data(), right.data(), bytes))
            return false;
    }
    return true;
}
bool projection(sycl::queue& q, int rows, std::mt19937& rng, bool timing) {
    namespace det = strata::prefill::strata_bf16_det;
    constexpr int k = 2560, n = 48, stride = 96;
    auto* x = operand<uint16_t>(q, rows * k, rng);
    auto* w = operand<uint16_t>(q, n * k, rng);
    auto* reference = operand<float>(q, rows * stride, rng);
    auto* result = sycl::malloc_device<float>(rows * stride, q);
    q.memcpy(result, reference, rows * stride * sizeof(float));
    auto previous = [&] { det::gemm(q, x, k, w, reference + n, rows, n, k, stride, 0.0f); };
    auto tiled = [&] { det::bmg_gdn_projection(q, x, k, w, result + n, rows, stride); };
    previous();
    tiled();
    bool ok = equal(q, result, reference, rows * stride);
    if (ok && timing)
        gdn_test::measure(q, "projection", rows, previous, tiled);
    sycl::free(x, q);
    sycl::free(w, q);
    sycl::free(result, q);
    sycl::free(reference, q);
    return ok;
}
using strata::prefill::GdnConvPath;
const char* path_name(GdnConvPath path) {
    switch (path) {
        case GdnConvPath::Serial: return "serial";
        case GdnConvPath::SeparateL2: return "separate L2";
        case GdnConvPath::FusedL2: return "fused L2";
        case GdnConvPath::Battlemage: return "Battlemage tiles";
    }
    return "unknown";
}

bool expect_path(const sycl::queue& q, int rows, GdnConvPath expected) {
    const auto actual = strata::prefill::select_gdn_conv_path(q, rows);
    if (actual == expected) return true;
    std::fprintf(stderr, "GDN convolution dispatch at %d rows: expected %s, selected %s\n",
                 rows, path_name(expected), path_name(actual));
    return false;
}

bool convolution(sycl::queue& q, int rows, std::mt19937& rng, bool timing,
                 GdnConvPath reference_path = GdnConvPath::SeparateL2, bool use_dispatch = false) {
    if (use_dispatch && !expect_path(q, rows, reference_path)) return false;
    constexpr int c = 10240;
    auto* x = operand<float>(q, size_t(rows) * c, rng);
    auto* w = operand<float>(q, 4 * c, rng);
    auto* history = operand<float>(q, 3 * c, rng);
    auto* hist = sycl::malloc_device<float>(3 * c, q);
    auto* result = sycl::malloc_device<float>(size_t(rows) * c, q);
    auto* reference = sycl::malloc_device<float>(size_t(rows) * c, q);
    q.memcpy(hist, history, 3 * c * sizeof(float));
    auto previous = [&] {
        strata::prefill::gdn_conv_impl(history, x, w, reference, rows, 1e-6f, &q,
                                       reference_path);
    };
    auto tiled = [&] {
        if (use_dispatch) strata::prefill::gdn_conv(hist, x, w, result, rows, 1e-6f, &q);
        else strata::prefill::gdn_conv_impl(hist, x, w, result, rows, 1e-6f, &q, GdnConvPath::Battlemage);
    };
    previous();
    tiled();
    bool ok = equal(q, result, reference, size_t(rows) * c) && equal(q, hist, history, 3 * c);
    if (ok && timing)
        gdn_test::measure(q, "conv", rows, previous, tiled);
    for (auto* ptr : {x, w, history, hist, result, reference})
        sycl::free(ptr, q);
    return ok;
}
} // namespace
int main(int argc, char** argv) {
    auto& q = dpct::get_in_order_queue();
    if (!strata::prefill::use_bmg_gdn_tiles(q, 8192))
        return 77;
    const bool timing = argc == 2 && std::string_view(argv[1]) == "--timing";
    if (strata::prefill::use_bmg_gdn_tiles(q, 511) || !strata::prefill::use_bmg_gdn_tiles(q, 512))
        return 1;
    std::mt19937 rng(981);
    if (argc == 2 && !timing) {
        const std::string_view mode(argv[1]);
        if (mode == "--auto")
            return expect_path(q, 64, GdnConvPath::Serial) && expect_path(q, 511, GdnConvPath::SeparateL2) &&
                   convolution(q, 512, rng, false, GdnConvPath::Battlemage, true) ? 0 : 1;
        if (mode == "--serial") return convolution(q, 512, rng, false, GdnConvPath::Serial, true) ? 0 : 1;
        if (mode == "--fused") return convolution(q, 512, rng, false, GdnConvPath::FusedL2, true) ? 0 : 1;
        if (mode == "--separate") return convolution(q, 512, rng, false, GdnConvPath::SeparateL2, true) ? 0 : 1;
        return 1;
    }
    for (int rows : {512, 1024, 8192}) {
        if (!projection(q, rows, rng, timing) || !convolution(q, rows, rng, timing)) {
            std::fprintf(stderr, "GDN projection or convolution differs at %d rows\n", rows);
            return 1;
        }
    }
    return convolution(q, 1025, rng, false) ? 0 : 1;
}
