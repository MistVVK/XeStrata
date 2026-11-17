// src/image/pipeline/qi21_glue.cpp - see include/strata/image/qi21_image.hpp.  Contraction off for the file: every
// product and sum rounds on its own, as sd.cpp's tensor operations do (one pass per operation).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "strata/image/qi21_image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::image {

std::vector<float> qi21_initial_latent(const std::vector<float>& noise, float sigma0) {
    const size_t n = noise.size();
    std::vector<float> a(n, 0.0f), b(n);
    const float keep = 1.0f - sigma0;
    for (size_t i = 0; i < n; ++i) a[i] = a[i] * keep;          // latent * (1 - sigma), latent = zeros
    for (size_t i = 0; i < n; ++i) b[i] = noise[i] * sigma0;    // noise * sigma
    for (size_t i = 0; i < n; ++i) a[i] = a[i] + b[i];          // the sum
    return a;
}

namespace {

float tensor_mean(const std::vector<float>& v) {   // sd::Tensor<float>::mean
    if (v.empty()) return 0.0f;
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x);
    return static_cast<float>(s / static_cast<double>(v.size()));
}

}  // namespace

std::vector<float> qi21_condition(std::vector<float> hidden, int n, int hidden_dim, const std::vector<float>& weights,
                                  int n_system) {
    bool all_one = true;
    for (float w : weights)
        if (w != 1.0f) {
            all_one = false;
            break;
        }
    if (!all_one && !hidden.empty()) {
        const float m0 = tensor_mean(hidden);
        for (int t = 0; t < n; ++t) {
            const float w = weights[(size_t) t];
            float* row = hidden.data() + (size_t) t * hidden_dim;
            for (int c = 0; c < hidden_dim; ++c) row[c] = row[c] * w;
        }
        const float m1 = tensor_mean(hidden);
        if (std::isfinite(m0) && std::isfinite(m1) && m1 != 0.0f) {
            const float s = m0 / m1;
            for (float& x : hidden) x = x * s;
        }
    }
    if (n_system <= 0) return hidden;
    return std::vector<float>(hidden.begin() + (size_t) n_system * hidden_dim, hidden.end());
}

void qi21_picture_to_unit(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        volatile float s = x[i] + 1.0f;   // its own rounding
        const float v = s * 0.5f;
        x[i] = std::max(0.0f, std::min(1.0f, v));
    }
}

std::string qi21_parameters_text(const std::string& prompt, int steps, uint64_t seed, int width, int height,
                                 const std::string& model_name) {
    char line[256];
    std::snprintf(line, sizeof line, "\nSteps: %d, CFG scale: 1, Seed: %llu, Size: %dx%d, Model: %s, Sampler: euler, "
                  "Generator: strata-image", steps, (unsigned long long) seed, width, height, model_name.c_str());
    return prompt + line;
}

}  // namespace strata::image
