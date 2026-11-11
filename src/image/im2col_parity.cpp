// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The image engine's one unfold (qi_im2col16) against a host loop, at the shapes its users take: the 8-channel form,
// the one-value form a picture's 3-channel patch embedding is left with, a dilated convolution's tap spacing (the line
// art's), a row wider than the layer's own channels (RealESRGAN's dense blocks) and the 3D VAEs' time taps.  The aux
// nets unfolded with a kernel of their own until this one took them; a form that placed a value elsewhere would move
// a control map's or a VAE's bytes.
#include "strata/image/qi_kernels.hpp"
#include "../kernels/parity_device.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdio>
#include <random>
#include <vector>

namespace {

namespace p = strata::parity;
using strata::image::qi_im2col16;

struct Case {
    const char* what = "";
    int64_t T = 1, H = 16, W = 16, C = 8, ldx = 0;
    int kt = 1, k = 3, s = 1, pad = 1, dil = 1;
    int64_t OH = 16, OW = 16, p0 = 0, rows = 256;
};

// col[r][(dt k k + dy k + dx) C + c] = in[t + dt dil - 1][oy s + dy dil - pad][ox s + dx dil - pad][c], 0 outside
std::vector<uint16_t> unfold(const std::vector<uint16_t>& in, const Case& x) {
    const int64_t kk2 = (int64_t) x.k * x.k, taps = (int64_t) x.kt * kk2, stride = x.ldx != 0 ? x.ldx : x.C;
    std::vector<uint16_t> col((size_t) (x.rows * taps * x.C), 0);
    for (int64_t r = 0; r < x.rows; ++r) {
        const int64_t pq = x.p0 + r, ox = pq % x.OW, oyt = pq / x.OW, oy = oyt % x.OH, t = oyt / x.OH;
        for (int64_t tap = 0; tap < taps; ++tap) {
            const int64_t kk = tap % kk2, tt = t + (x.kt == 3 ? tap / kk2 - 1 : 0);
            const int64_t y = oy * x.s + (kk / x.k) * x.dil - x.pad, xx = ox * x.s + (kk % x.k) * x.dil - x.pad;
            for (int64_t c = 0; c < x.C; ++c) {
                uint16_t v = 0;
                if (tt >= 0 && tt < x.T && y >= 0 && y < x.H && xx >= 0 && xx < x.W)
                    v = in[(size_t) (((tt * x.H + y) * x.W + xx) * stride + c)];
                col[(size_t) ((r * taps + tap) * x.C + c)] = v;
            }
        }
    }
    return col;
}

bool one(const Case& x, std::mt19937& rng) {
    const int64_t stride = x.ldx != 0 ? x.ldx : x.C;
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<uint16_t> in((size_t) (x.T * x.H * x.W * stride));
    for (auto& v : in) v = strata::kernels::f16_from_f32(dist(rng));
    const std::vector<uint16_t> want = unfold(in, x);
    p::Device dev;
    uint16_t* xin = dev.upload(in);
    uint16_t* col = dev.alloc<uint16_t>(want.size());
    qi_im2col16(xin, x.T, x.H, x.W, x.C, x.kt, x.k, x.s, x.pad, x.OH, x.OW, x.p0, x.rows, col, p::stream(), x.ldx,
                x.dil);
    std::vector<uint16_t> got(want.size());
    p::copy_out(got.data(), col, got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == want[i]) continue;
        std::fprintf(stderr, "%s: value %zu is %04x, the host unfold has %04x\n", x.what, i, got[i], want[i]);
        return false;
    }
    std::printf("%s: %zu values, the same as the host unfold\n", x.what, got.size());
    return true;
}

}  // namespace

int main() {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    std::mt19937 rng(1112);  // NOLINT(bugprone-random-generator-seed): reproducible inputs.
    bool ok = true;
    // 8 channels at a time: the engine's own width, and the aux nets' maps with their channels padded to 8
    ok &= one({.what = "3 x 3, 8 channels"}, rng);
    ok &= one({.what = "1 x 1, 32 channels", .C = 32, .k = 1, .pad = 0}, rng);
    // a stride 2 down-sampling padded only after the edge (the VAEs' down block), a band that starts past row 0
    ok &= one({.what = "3 x 3 stride 2, a band from row 60", .H = 32, .W = 32, .C = 16, .s = 2, .pad = 0, .OH = 16,
               .OW = 16, .p0 = 60, .rows = 196}, rng);
    // a dilated convolution (the line art's block23: k 3, the taps 5 apart, padding 5)
    ok &= one({.what = "3 x 3 dilated 5, 96 channels", .H = 20, .W = 24, .C = 96, .pad = 5, .dil = 5, .OH = 20,
               .OW = 24}, rng);
    // a row wider than the layer's own channels: RealESRGAN's dense blocks read a prefix of them
    ok &= one({.what = "3 x 3 of 8 of a 40-wide row", .H = 12, .W = 12, .C = 8, .ldx = 40, .OH = 12, .OW = 12,
               .rows = 144}, rng);
    // one value at a time: a picture's 3 channels (SAM 3, BiRefNet and Florence-2's patch embeddings) and odd widths
    ok &= one({.what = "4 x 4 stride 4, 3 channels", .H = 32, .W = 32, .C = 3, .k = 4, .s = 4, .pad = 0, .OH = 8,
               .OW = 8, .rows = 64}, rng);
    ok &= one({.what = "3 x 3, 5 channels", .H = 18, .W = 18, .C = 5, .OH = 18, .OW = 18, .rows = 324}, rng);
    // the 3D VAEs: the frames before and after each one, including the taps outside the clip
    ok &= one({.what = "3 x 3 x 3 over 3 frames", .T = 3, .H = 8, .W = 8, .C = 16, .kt = 3, .OH = 8, .OW = 8,
               .rows = 192}, rng);
    ok &= one({.what = "3 x 3, one frame", .H = 8, .W = 8, .C = 16, .OH = 8, .OW = 8, .rows = 64}, rng);
    if (!ok) return 1;
    std::printf("im2col_parity: every form is the host unfold\n");
    return 0;
}
