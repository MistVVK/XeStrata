// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Compare a few SeedVR2 FP16 rows quantized by the engine with the int8_convrot file.
#include "strata/image/safetensors.hpp"
#include "strata/image/train_ops.hpp"
#include "strata/platform/env.hpp"
#include "../kernels/parity_device.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace si = strata::image;
namespace p = strata::parity;

template<class T> std::vector<T> rows(const std::string& path, const si::SafeTensors& st,
                                      const std::string& name, const char* dtype, size_t count) {
    const auto* tensor = st.find(name);
    if (!tensor || tensor->dtype != dtype || tensor->end - tensor->begin < count * sizeof(T))
        throw std::runtime_error("bad tensor " + name);
    std::ifstream file(path, std::ios::binary);
    file.seekg((std::streamoff) (st.data_base() + tensor->begin));
    std::vector<T> data(count);
    file.read(reinterpret_cast<char*>(data.data()), (std::streamsize) (count * sizeof(T)));
    if (!file) throw std::runtime_error("short tensor " + name);
    return data;
}

namespace {

// the test proper; main reports what it throws
int run() {
    const char* dir = strata::env_get("SEEDVR2_TEST_DIR");
    if (!dir) return 77;
    const std::string fp = std::string(dir) + "/seedvr2_7b_fp16.safetensors";
    const std::string i8 = std::string(dir) + "/seedvr2_7b_int8_convrot.safetensors";
    if (!std::filesystem::is_regular_file(fp) || !std::filesystem::is_regular_file(i8) ||
        sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    si::SafeTensors f(fp), g(i8);
    constexpr int N = 8;
    for (const char* base : {"blocks.0.attn.proj_qkv.vid", "blocks.10.mlp.txt.proj_out"}) {
        const std::string name = std::string(base) + ".weight";
        const auto* ft = f.find(name);
        const auto* gt = g.find(name);
        if (!ft || !gt || ft->shape.size() != 2 || ft->shape != gt->shape || ft->shape[0] < N || ft->shape[1] % 256)
            throw std::runtime_error("bad shape " + name);
        const int64_t K = ft->shape[1];
        auto input = rows<uint16_t>(fp, f, name, "F16", N * K);
        auto expected = rows<int8_t>(i8, g, name, "I8", N * K);
        auto expected_scale = rows<float>(i8, g, std::string(base) + ".weight_scale", "F32", N);
        std::vector<int8_t> actual(N * K);
        std::vector<float> scale(N);
        p::Device dev;
        auto* x = dev.upload(input);
        auto* q = dev.alloc<int8_t>(actual.size());
        auto* s = dev.alloc<float>(N);
        si::train::w8_quantize(x, N, K, q, s, p::stream());
        p::copy_out(actual.data(), q, actual.size());
        p::copy_out(scale.data(), s, scale.size());
        for (int r = 0; r < N; ++r) {
            if (!(scale[r] > 0) || std::abs(scale[r] / expected_scale[r] - 1.f) > 1.f / 127.f)
                throw std::runtime_error("scale mismatch " + name);
            for (int64_t c = 0; c < K; ++c) {
                const int diff = (int) actual[r * K + c] - (int) expected[r * K + c];
                if (std::abs(diff) > 1) {
                    std::fprintf(stderr, "%s row %d col %lld: difference %d\n", name.c_str(), r, (long long) c, diff);
                    return 1;
                }
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "seedvr2_convrot_parity: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "seedvr2_convrot_parity: an unknown exception\n");
    }
    return 1;
}
