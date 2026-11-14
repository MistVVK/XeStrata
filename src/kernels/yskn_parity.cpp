// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/kernels/yskn_parity.cpp - the .yskn products on the GPU (yskn.hpp) against the CPU reference
// (artifact/yskn.hpp), on a synthetic container: random codes in YASAKANI's warp layout (written from its repack.cu),
// a 6-bit matrix with an output rotation and a seed past 2^53, an 8-bit head without one.
//
// The reader is checked where the GPU reads through it: the header's fields and the seed, and the device order against
// the words the container was written with.  Then the decode's product at 1, 3 and 8 rows against gemv_reference (its
// integer sums: the same up to an int8 activation rounded the other way), a row the same bits whatever the window's
// rows, the engine's row order (perm_out), and the prompt's pieces: the FP16 rows, the input's and the output's
// rotations.
#include "strata/kernels/yskn.hpp"

#include "parity_device.hpp"
#include "strata/artifact/yskn.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace y = strata::yskn;
namespace k = strata::kernels;
namespace p = strata::parity;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct Mat {
    std::string name, role, rot_in, rot_out;
    int rows, cols, bits;
    std::vector<uint32_t> units;   // [row][unit][word], as written
};

// The container: the JSON header padded to 64 KiB, then each matrix's lut16, scale16 and warp-layout stream
std::vector<uint8_t> container(std::vector<Mat>& mats) {
    std::mt19937 rng(20261114);  // NOLINT(bugprone-random-generator-seed): reproducible codes.
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::vector<uint8_t> pay;
    const size_t base = 0x20 + 65536;
    auto put = [&](const void* src, size_t n) {
        while (pay.size() % 16) pay.push_back(0);
        const size_t off = pay.size();
        pay.insert(pay.end(), (const uint8_t*) src, (const uint8_t*) src + n);
        return base + off;
    };
    std::string tensors;
    for (Mat& m : mats) {
        std::vector<uint16_t> lut(2 * y::kStates), sc((size_t) m.rows);
        for (auto& v : lut) v = k::f16_from_f32(gauss(rng));
        for (auto& v : sc) v = k::f16_from_f32(0.01f + 0.02f * std::fabs(gauss(rng)));
        const int S = m.cols / 256, wps = 4 * m.bits;
        m.units.resize((size_t) m.rows * S * wps);
        for (auto& v : m.units) v = (uint32_t) rng();
        std::vector<uint32_t> stream(m.units.size());
        for (int r = 0; r < m.rows; ++r)
            for (int s = 0; s < S; ++s)
                for (int w = 0; w < wps; ++w) {   // repack.cu: pair q = unit * rows + row, 32 a tile, 16-byte chunks
                    const size_t q = (size_t) s * m.rows + r, tile = q / 32, lane = q % 32;
                    stream[((tile * (wps / 4) + w / 4) * 32 + lane) * 4 + w % 4] = m.units[((size_t) r * S + s) * wps + w];
                }
        const size_t lo = put(lut.data(), lut.size() * 2), so = put(sc.data(), sc.size() * 2),
                     wo = put(stream.data(), stream.size() * 4);
        if (!tensors.empty()) tensors += ",";
        tensors += "{\"name\":\"" + m.name + "\",\"role\":\"" + m.role + "\",\"rows\":" + std::to_string(m.rows) +
                   ",\"cols\":" + std::to_string(m.cols) + ",\"bits\":" + std::to_string(m.bits) +
                   ",\"lut_off\":" + std::to_string(lo) + ",\"scale_off\":" + std::to_string(so) +
                   ",\"stream_off\":" + std::to_string(wo) + ",\"stream_bytes\":" + std::to_string(stream.size() * 4) +
                   ",\"rot_in\":" + m.rot_in + ",\"rot_out\":" + m.rot_out + "}";
    }
    std::string json = "{\"model\":{\"arch\":\"qwen3_5\",\"layers\":1,\"hidden\":1024,\"ffn\":1024,\"heads\":4,"
                       "\"kv_heads\":1,\"head_dim\":256,\"vocab\":64,\"eps\":1e-06,\"rope_theta\":10000000.0},"
                       "\"tensors\":[" + tensors + "],\"vectors\":[]}";
    json.resize(65536, ' ');
    std::vector<uint8_t> f(0x20, 0);
    std::memcpy(f.data(), "YSKN", 4);
    const uint32_t ver = 1, len = (uint32_t) json.size();
    std::memcpy(f.data() + 4, &ver, 4);
    std::memcpy(f.data() + 8, &len, 4);
    f.insert(f.end(), json.begin(), json.end());
    f.insert(f.end(), pay.begin(), pay.end());
    return f;
}

double rel_err(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: yskn_parity [--selftest]\n");
            return 2;
        }
    std::vector<Mat> mats = {
        {"layers.0.mlp.up_proj", "linear",
         "{\"kind\":\"H512\",\"m\":2,\"sign_seed\":18446744073709551557,\"sign_offset\":3,\"rule\":\"splitmix64\"}",
         "{\"kind\":\"H512\",\"m\":2,\"sign_seed\":11,\"sign_offset\":1024,\"rule\":\"splitmix64\"}", 1024, 1024, 6, {}},
        {"lm_head", "head", "{\"kind\":\"H512\",\"m\":1,\"sign_seed\":7,\"sign_offset\":0,\"rule\":\"mulberry32_7919_13\"}",
         "null", 64, 512, 8, {}},
    };
    const std::vector<uint8_t> file = container(mats);
    const y::Container c(file.data(), file.size());

    // ---- the reader, where the GPU reads through it
    const y::TensorDesc* up = c.find_tensor(mats[0].name);
    const y::TensorDesc* head = c.find_tensor(mats[1].name);
    check(up && head && up->bits == 6 && head->bits == 8 && up->physical_rows == 1024 && up->has_rot_out &&
              !head->has_rot_out,
          "the header's tensors");
    check(up->rot_in.sign_seed == 18446744073709551557ull && up->rot_in.sign_offset == 3, "a u64 seed past 2^53");
    for (size_t i = 0; i < mats.size(); ++i) {
        const y::TensorDesc& t = *c.find_tensor(mats[i].name);
        const std::vector<uint32_t> dev = y::device_order(c, t);
        const int N = t.physical_rows, S = t.seqs(), wps = t.words();
        bool ok = dev.size() == (size_t) N * S * wps;
        for (int s = 0; ok && s < S; ++s)
            for (int r = 0; r < N; ++r)
                for (int w = 0; w < wps; ++w)
                    ok = ok && dev[(((size_t) s * (wps / 4) + w / 4) * N + r) * 4 + w % 4] ==
                                   mats[i].units[((size_t) r * S + s) * wps + w];
        check(ok, t.name + ": the device order is the words written");
    }

    // ---- the matrices on the device
    p::Device dv;
    std::mt19937 rng(7);  // NOLINT(bugprone-random-generator-seed): reproducible.
    std::normal_distribution<float> gauss(0.f, 1.f);
    for (const y::TensorDesc* t : {up, head}) {
        const int N = t->physical_rows, K = t->cols;
        std::vector<uint16_t> lut8;
        std::vector<float> tau;
        y::lut8_tau(c, *t, lut8, tau);
        const auto si = y::sign_stream(t->rot_in.rule, t->rot_in.sign_seed, t->rot_in.sign_offset, (size_t) K);
        const auto so = t->has_rot_out
                            ? y::sign_stream(t->rot_out.rule, t->rot_out.sign_seed, t->rot_out.sign_offset, (size_t) N)
                            : std::vector<int8_t>{};
        k::YsknDev d;
        d.words = dv.upload(y::device_order(c, *t));
        d.lut8 = dv.upload(lut8);
        d.lut16 = dv.upload(std::vector<uint16_t>(c.lut16(*t), c.lut16(*t) + 2 * y::kStates));
        d.tau = dv.upload(tau);
        d.scale16 = dv.upload(std::vector<uint16_t>(c.scale16(*t), c.scale16(*t) + N));
        d.sign_in = dv.upload(si);
        d.sign_out = so.empty() ? nullptr : dv.upload(so);
        d.rows = N;
        d.rows_used = t->rows;
        d.cols = K;
        d.bits = t->bits;
        d.n_workers = K / 512;   // every 512-group its own work-item: the partial sums are added by yskn_finish

        // the decode's product: 8 rows at once, then 3, then 1; each row against the reference
        constexpr int T = 8;
        std::vector<float> x((size_t) T * K);
        for (auto& v : x) v = gauss(rng);
        x[5] = 9.f;   // an outlier the rotation spreads
        std::vector<float> want((size_t) T * N);
        for (int r = 0; r < T; ++r) y::gemv_reference(c, *t, x.data() + (size_t) r * K, want.data() + (size_t) r * N);
        float* xd = dv.upload(x);
        auto* scratch = dv.alloc<uint8_t>(k::yskn_scratch_bytes(d, T));
        float* yd = dv.alloc<float>((size_t) T * N);
        std::vector<float> got8((size_t) T * N), got3((size_t) 3 * N), got1(N);
        k::yskn_proj(d, xd, K, T, yd, N, scratch, p::stream());
        p::copy_out(got8.data(), yd, got8.size());
        k::yskn_proj(d, xd, K, 3, yd, N, scratch, p::stream());
        p::copy_out(got3.data(), yd, got3.size());
        k::yskn_proj(d, xd + (size_t) 2 * K, K, 1, yd, N, scratch, p::stream());
        p::copy_out(got1.data(), yd, got1.size());
        double worst = 0;
        for (int r = 0; r < T; ++r)
            worst = std::max(worst, rel_err(got8.data() + (size_t) r * N, want.data() + (size_t) r * N, (size_t) N));
        check(worst < 1e-3, t->name + ": the decode's product against the reference (" + std::to_string(worst) + ")");
        check(std::memcmp(got3.data(), got8.data(), got3.size() * 4) == 0 &&
                  std::memcmp(got1.data(), got8.data() + (size_t) 2 * N, (size_t) N * 4) == 0,
              t->name + ": a row the same bits at 1, 3 and 8 rows");

        // the matrix form's arm (where the device runs it): the same integer sums as the DP4a form (the rows above,
        // untimed: DP4a), against the reference and bit for bit, at 8 and 16 rows (the widest window)
        if (k::yskn_mma_usable(p::stream())) {
            constexpr int TM = 16;
            std::vector<float> xm((size_t) TM * K);
            std::copy(x.begin(), x.end(), xm.begin());
            for (size_t i = x.size(); i < xm.size(); ++i) xm[i] = gauss(rng);
            std::vector<float> wantm((size_t) TM * N);
            for (int r = 0; r < TM; ++r)
                y::gemv_reference(c, *t, xm.data() + (size_t) r * K, wantm.data() + (size_t) r * N);
            float* xmd = dv.upload(xm);
            auto* scratchm = dv.alloc<uint8_t>(k::yskn_scratch_bytes(d, TM));
            float* ymd = dv.alloc<float>((size_t) TM * N);
            std::vector<float> gotm((size_t) TM * N);
            bool same = true;
            double wm = 0;
            for (int rows : {8, 16}) {
                k::yskn_mma_proj(d, xmd, K, rows, ymd, N, scratchm, p::stream());
                p::copy_out(gotm.data(), ymd, (size_t) rows * N);
                for (int r = 0; r < rows; ++r)
                    wm = std::max(wm, rel_err(gotm.data() + (size_t) r * N, wantm.data() + (size_t) r * N, (size_t) N));
                same = same && std::memcmp(gotm.data(), got8.data(), (size_t) std::min(rows, T) * N * 4) == 0;
            }
            check(wm < 1e-3 && same,
                  t->name + ": the matrix form against the reference and the DP4a bits (" + std::to_string(wm) + ")");
        }

        // the engine's row order: row r written at perm[r]
        if (t == up) {
            std::vector<int32_t> perm(N);
            for (int r = 0; r < N; ++r) perm[(size_t) r] = r;
            std::shuffle(perm.begin(), perm.end(), rng);
            k::YsknDev dp = d;
            dp.perm_out = dv.upload(perm);
            k::yskn_proj(dp, xd, K, 1, yd, N, scratch, p::stream());
            std::vector<float> gp(N);
            p::copy_out(gp.data(), yd, gp.size());
            bool ok = true;
            for (int r = 0; r < N; ++r) ok = ok && gp[(size_t) perm[(size_t) r]] == got8[(size_t) r];
            check(ok, t->name + ": perm_out writes row r at perm[r]");
        }

        // the prompt's pieces: the FP16 rows, the input's rotation, the output's
        std::vector<uint16_t> w16((size_t) N * K);
        uint16_t* wd = dv.alloc<uint16_t>(w16.size());
        k::yskn_dequant_f16(d, wd, p::stream());
        p::copy_out(w16.data(), wd, w16.size());
        std::vector<float> row(K);
        bool deq = true;
        for (int r = 0; r < N && deq; ++r) {
            y::row_rotated(c, *t, r, row.data());
            for (int i = 0; i < K && deq; ++i) deq = w16[(size_t) r * K + i] == k::f16_from_f32(row[(size_t) i]);
        }
        check(deq, t->name + ": the FP16 rows are the reference's rows rounded");

        std::vector<uint16_t> x16((size_t) 2 * K), r16((size_t) 2 * K);
        for (size_t i = 0; i < x16.size(); ++i) x16[i] = k::f16_from_f32(x[i]);
        uint16_t* x16d = dv.upload(x16);
        uint16_t* r16d = dv.alloc<uint16_t>(r16.size());
        k::yskn_rotate_in_f16(d, x16d, K, 2, r16d, p::stream());
        p::copy_out(r16.data(), r16d, r16.size());
        bool rot_in_ok = true;
        for (int r = 0; r < 2; ++r) {
            std::vector<float> v(K);
            for (int i = 0; i < K; ++i) v[(size_t) i] = k::f32_from_f16(x16[(size_t) r * K + i]) * (float) si[(size_t) i];
            y::fwht512(v.data(), (size_t) K);
            for (int i = 0; i < K; ++i)
                rot_in_ok = rot_in_ok && r16[(size_t) r * K + i] == k::f16_from_f32(v[(size_t) i]);
        }
        check(rot_in_ok, t->name + ": the input's rotation in FP16");

        std::vector<float> yr((size_t) 2 * N), yo((size_t) 2 * N);
        for (auto& v : yr) v = gauss(rng);
        float* yrd = dv.upload(yr);
        k::yskn_rotate_out(d, yrd, 2, yd, N, p::stream());
        p::copy_out(yo.data(), yd, yo.size());
        std::vector<float> wo = yr;
        if (!so.empty())
            for (int r = 0; r < 2; ++r) {
                y::fwht512(wo.data() + (size_t) r * N, (size_t) N);
                for (int i = 0; i < N; ++i) wo[(size_t) r * N + i] *= (float) so[(size_t) i];
            }
        check(std::memcmp(yo.data(), wo.data(), yo.size() * 4) == 0, t->name + ": the output's rotation");
    }

    std::printf("%s\n", failures == 0 ? "yskn_parity: all passed" : "yskn_parity: FAILED");
    return failures == 0 ? 0 : 1;
}
