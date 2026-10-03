// Shared helpers for the native-kernel probes in this directory: a CPU ggml context whose graphs are computed by
// ggml-cpu at the pinned llama.cpp commit, and comparison statistics.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "parity_device.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace probe {

struct Ggml {
    ggml_context* ctx = nullptr;
    explicit Ggml(size_t bytes) {
        ggml_init_params p{bytes, nullptr, false};
        ctx = ggml_init(p);
        if (!ctx) throw std::runtime_error("ggml_init failed");
    }
    ~Ggml() { ggml_free(ctx); }
    ggml_tensor* f32(std::initializer_list<int64_t> ne, const std::vector<float>& data) {
        int64_t d[4] = {1, 1, 1, 1};
        int i = 0;
        for (int64_t n : ne) d[i++] = n;
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, i, d);
        if ((size_t) ggml_nelements(t) != data.size()) throw std::runtime_error("tensor size mismatch");
        std::memcpy(t->data, data.data(), data.size() * sizeof(float));
        return t;
    }
    void compute(ggml_tensor* out, int threads = 8) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(gf, out);
        if (ggml_graph_compute_with_ctx(ctx, gf, threads) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml graph compute failed");
    }
    static std::vector<float> read(const ggml_tensor* t, size_t offset_elems, size_t n) {
        std::vector<float> v(n);
        std::memcpy(v.data(), (const float*) t->data + offset_elems, n * sizeof(float));
        return v;
    }
};

struct Diff {
    size_t n = 0, bit_diff = 0;
    double max_abs = 0, max_rel = 0, ref_scale = 0;
};

inline Diff compare(const std::vector<float>& got, const std::vector<float>& ref) {
    Diff d;
    d.n = ref.size();
    for (double r : ref) d.ref_scale = std::fmax(d.ref_scale, std::fabs(r));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) ++d.bit_diff;
        const double a = std::fabs((double) got[i] - (double) ref[i]);
        if (!(a <= d.max_abs)) d.max_abs = a;   // NaN propagates
    }
    d.max_rel = d.max_abs / (d.ref_scale > 0 ? d.ref_scale : 1.0);
    return d;
}

inline void print(const char* what, const Diff& d) {
    std::printf("  %-44s %zu/%zu bits differ, max |d| %.3e, rel to scale %.3e (scale %.3e)\n", what, d.bit_diff,
                d.n, d.max_abs, d.max_rel, d.ref_scale);
}

inline std::vector<float> normal(size_t n, uint32_t seed, float sd = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

template <class T> T* upload(const std::vector<T>& h) {
    T* d = static_cast<T*>(strata::parity::alloc_bytes(h.size() * sizeof(T) + 64));
    strata::parity::copy_bytes_in(d, h.data(), h.size() * sizeof(T));
    return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    strata::parity::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}

}  // namespace probe
