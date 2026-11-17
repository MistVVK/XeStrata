// src/image/dit/qi21_rope.cpp - see include/strata/image/qi21_rope.hpp.  Written as sd.cpp writes it, operation by
// operation (float arithmetic, the float overloads of std::cos / std::sin, ::powf), so the bits are the same.
#include "strata/image/qi21_rope.hpp"

#include <cmath>

namespace strata::image {

namespace {
const int kAxes[3] = {16, 56, 56};

std::vector<float> rope_frequencies(int dim, float theta) {
    const int half = dim / 2;
    // linspace<float>(0.f, (dim * 1.f - 2) / dim, half)
    const float start = 0.f, end = (dim * 1.f - 2) / dim;
    std::vector<float> scale(half);
    if (half == 1) {
        scale[0] = start;
    } else {
        const float step = (end - start) / (half - 1);
        for (int i = 0; i < half; ++i) scale[i] = start + i * step;
    }
    std::vector<float> omega(half);
    for (int i = 0; i < half; ++i) omega[i] = 1.0f / ::powf(1.f * theta, scale[i]);
    return omega;
}
}  // namespace

std::vector<float> qi21_positions(int64_t text_len, int64_t H, int64_t W) {
    std::vector<float> p;
    p.reserve((size_t) (text_len + H * W) * 3);
    int64_t position = 0;
    for (int64_t j = 0; j < text_len; ++j, ++position) {
        const float f = static_cast<float>(position);
        p.push_back(f), p.push_back(f), p.push_back(f);
    }
    for (int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            p.push_back(static_cast<float>(position));
            p.push_back(static_cast<float>(h - (H - H / 2)));
            p.push_back(static_cast<float>(w - (W - W / 2)));
        }
    return p;
}

std::vector<float> qi21_rope_table(const std::vector<float>& positions) {
    const size_t n = positions.size() / 3;
    std::vector<float> out(n * 256);
    size_t offset = 0;
    for (int a = 0; a < 3; ++a) {
        const std::vector<float> omega = rope_frequencies(kAxes[a], 10000.0f);
        for (size_t i = 0; i < n; ++i) {
            float* row = out.data() + i * 256 + offset;
            for (size_t j = 0; j < omega.size(); ++j) {
                const float angle = positions[3 * i + a] * omega[j];
                row[4 * j] = std::cos(angle);
                row[4 * j + 1] = -std::sin(angle);
                row[4 * j + 2] = std::sin(angle);
                row[4 * j + 3] = std::cos(angle);
            }
        }
        offset += omega.size() * 4;
    }
    return out;
}

}  // namespace strata::image
