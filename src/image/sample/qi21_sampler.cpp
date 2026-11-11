// src/image/sample/qi21_sampler.cpp - see include/strata/image/qi21_sampler.hpp.
//
// No contraction into fused multiply-adds: sd.cpp computes every line as its own pass over a tensor, so each operation
// rounds on its own.  The passes are kept separate here too (a temporary per line), and contraction is switched off
// for this file for every compiler that would otherwise fuse (GCC defaults to -ffp-contract=fast; on aarch64 an FMA
// is always there).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "strata/image/qi21_sampler.hpp"

#include <cerrno>
#include <cstdlib>
#include <sstream>

namespace strata::image {

const char* const QI21_TURBO_SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0";

std::vector<float> qi21_parse_sigmas(const std::string& text) {
    std::string s = text;
    if (!s.empty() && s.front() == '[') s.erase(0, 1);
    if (!s.empty() && s.back() == ']') s.pop_back();
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    const char* ws = " \t\n\r\f\v";
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(ws));
        const size_t last = item.find_last_not_of(ws);
        item.erase(last == std::string::npos ? 0 : last + 1);
        if (item.empty()) continue;
        errno = 0;
        char* end = nullptr;
        const float f = std::strtof(item.c_str(), &end);   // what std::stof calls
        if (end == item.c_str() || errno == ERANGE) return {};
        out.push_back(f);
    }
    return out;
}

float qi21_timestep(float sigma) { return sigma * 1000.f; }

void qi21_euler_step(float* x, const float* v, size_t n, float sigma, float sigma_next) {
    if (sigma == sigma_next) return;
    const float c_skip = 1.0f, c_out = -sigma;
    const float dt = sigma_next - sigma;
    std::vector<float> den(n), d(n);
    for (size_t i = 0; i < n; ++i) den[i] = v[i] * c_out;           // guided.pred * c_out
    for (size_t i = 0; i < n; ++i) d[i] = x[i] * c_skip;             // x * c_skip
    for (size_t i = 0; i < n; ++i) den[i] = den[i] + d[i];           // denoised
    for (size_t i = 0; i < n; ++i) d[i] = x[i] - den[i];             // x - denoised
    for (size_t i = 0; i < n; ++i) d[i] = d[i] / sigma;              // ... / sigma
    for (size_t i = 0; i < n; ++i) d[i] = d[i] * dt;                 // d * (sigmas[i + 1] - sigma)
    for (size_t i = 0; i < n; ++i) x[i] = x[i] + d[i];               // x += ...
}

}  // namespace strata::image
