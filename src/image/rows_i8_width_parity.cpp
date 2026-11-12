// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The W8A8 inputs' rows (dit_rows.hpp's RowsI8Of) at each width rows_i8_width can take, 1, 4 and 8 values side by side
// a lane: the same int8 bytes and the same scales, bit for bit, on synthetic rows.  The width is timed at the start,
// so a width whose rows differed made the images differ from start to start.
#include "dit_rows.hpp"
#include "../kernels/parity_device.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

namespace p = strata::parity;
namespace d = strata::image::dit_rows;

struct Rows {
    std::vector<int8_t> q;
    std::vector<float> s;
};

// one op's rows at width W; `reset` puts back what an op that keeps its sums (kKeep) wrote
template<class Op, int W>
Rows run(const Op& op, int64_t T, int64_t K, int8_t* q, float* s, sycl::queue& qu, const std::function<void()>& reset) {
    reset();
    d::rows_i8_as<Op, W>(op, T, K, q, s, qu).wait();
    Rows r{std::vector<int8_t>((size_t) (T * K)), std::vector<float>((size_t) T)};
    p::copy_out(r.q.data(), q, r.q.size());
    p::copy_out(r.s.data(), s, r.s.size());
    return r;
}

template<class Op>
bool same_at_widths(const char* name, const Op& op, int64_t T, int64_t K, int8_t* q, float* s, sycl::queue& qu,
                    const std::function<void()>& reset = [] {}) {
    const Rows r8 = run<Op, 8>(op, T, K, q, s, qu, reset);
    const Rows r4 = run<Op, 4>(op, T, K, q, s, qu, reset);
    const Rows r1 = run<Op, 1>(op, T, K, q, s, qu, reset);
    bool ok = true;
    for (const auto& [w, r] : {std::pair<int, const Rows*>{4, &r4}, {1, &r1}}) {
        if (std::memcmp(r->q.data(), r8.q.data(), r8.q.size()) != 0) {
            std::fprintf(stderr, "%s, K %lld: the int8 values at width %d differ from width 8's\n", name,
                         (long long) K, w);
            ok = false;
        }
        if (std::memcmp(r->s.data(), r8.s.data(), r8.s.size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "%s, K %lld: the scales at width %d differ from width 8's\n", name, (long long) K, w);
            ok = false;
        }
    }
    return ok;
}

// the test proper; main reports what it throws
int run() {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    sycl::queue& qu = d::q_of(p::stream());
    p::Device dev;
    std::mt19937 rng(1110);  // NOLINT(bugprone-random-generator-seed): reproducible inputs.
    std::normal_distribution<float> dist(0.f, 1.f);
    bool ok = true;
    // 3 groups of 256 a row (one sub-group, three of its four), and 12 (three sub-groups); a few large values a row
    for (const int64_t K : {768, 3072}) {
        constexpr int64_t T = 5;
        std::vector<float> x((size_t) (T * K)), aff((size_t) (2 * K));
        std::vector<uint16_t> h((size_t) (T * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = dist(rng) * (i % 97 == 0 ? 40.f : 1.f);
        for (auto& v : aff) v = 1.f + 0.25f * dist(rng);
        for (size_t i = 0; i < h.size(); ++i) h[i] = strata::kernels::f16_from_f32(dist(rng) * 3.f);
        float* xd = dev.upload(x);
        float* ad = dev.upload(aff);
        uint16_t* hd = dev.upload(h);
        auto* q = dev.alloc<int8_t>((size_t) (T * K));
        auto* s = dev.alloc<float>((size_t) T);
        ok &= same_at_widths("RMS norm, Sylvester's", d::RmsAffineI8<false>{xd, ad, ad + K, K, 1e-6f}, T, K, q, s, qu);
        ok &= same_at_widths("RMS norm, regular", d::RmsAffineI8<true>{xd, ad, ad + K, K, 1e-6f}, T, K, q, s, qu);
        ok &= same_at_widths("FP16 rows, regular", d::HalfI8<true>{hd, K}, T, K, q, s, qu);
        ok &= same_at_widths("FP32 rows, Sylvester's", d::FloatI8<false>{xd, K}, T, K, q, s, qu);
        // an op that writes its sums back (kKeep) and norms them: x += gate y, y int32 sums with their scales
        std::vector<int32_t> y((size_t) (T * K));
        for (auto& v : y) v = (int32_t) (dist(rng) * 3000.f);
        std::vector<float> sx((size_t) T), gw((size_t) (3 * K));
        for (auto& v : sx) v = 1e-3f * (1.f + std::abs(dist(rng)));
        for (auto& v : gw) v = 0.5f + 0.1f * dist(rng);
        int32_t* yd = dev.upload(y);
        float* sxd = dev.upload(sx);
        float* gwd = dev.upload(gw);
        float* kept = dev.alloc<float>((size_t) (T * K));
        strata::image::DitOut yo;
        yo.i = yd;
        yo.sx = sxd;
        yo.sw = gwd;
        yo.C = K;
        ok &= same_at_widths("gate add and RMS norm, kept", d::TimedI8{kept, yo, gwd, K}, T, K, q, s, qu,
                             [&] { p::copy_in(kept, x.data(), x.size()); });
    }
    if (ok) std::printf("rows_i8_width_parity: the same bits at widths 1, 4 and 8\n");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rows_i8_width_parity: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "rows_i8_width_parity: an unknown exception\n");
    }
    return 1;
}
