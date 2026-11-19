// src/kernels/cpu/kq512_parity.cpp - BW-4: UD-Q4_K_XL's AVX-512 VNNI expert rows (kq_avx512.cpp: Q4_K gate/up, Q5_1 and Q8_0 down) against the build's one-token dot, BIT FOR BIT,
// for 1 to 160 tokens, on random blocks (real-quantized activations and raw random bytes, so every int8 value and every 5-bit / 4-bit code occurs); and the speed of the rows against
// the AVX2 multi-token rows (kq256) and against the dot called once a token, one thread, weights in L1 / L2.
//
//     kq512_parity [--bench]          exits 77 (skipped) on a CPU without AVX-512 VNNI
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/cpu/pool_quant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace pq = strata::kernels::cpu::pq;

namespace {

uint32_t g_seed = 20260119u;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

uint16_t half_bits(float f) {   // a normal f16 for f in [2^-14, 2^15]
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    return (uint16_t) (((u >> 16) & 0x8000) | ((uint32_t) e << 10) | ((u >> 13) & 0x3ff));
}

const char* tname(int t) { return t == 12 ? "Q4_K" : t == 7 ? "Q5_1" : "Q8_0"; }

// random weights of `rows` rows (n values): every byte random, the fp16 scale fields (d, and dmin / m) set to a sane positive range
std::vector<uint8_t> make_weights(int type, int rows, int n, size_t& rb) {
    rb = pq::row_size_of(type, n);
    const size_t bb = pq::type_size_of(type), nb = rb / bb;
    std::vector<uint8_t> w((size_t) rows * rb);
    for (auto& b : w) b = (uint8_t) rnd();
    for (int r = 0; r < rows; ++r)
        for (size_t k = 0; k < nb; ++k) {
            uint8_t* d = &w[(size_t) r * rb + k * bb];
            const uint16_t h0 = half_bits(0.02f + (float) (rnd() % 1000) * 0.001f);
            std::memcpy(d, &h0, 2);
            if (type == 12 || type == 7) {
                const uint16_t h1 = half_bits(0.01f + (float) (rnd() % 500) * 0.001f);
                std::memcpy(d + 2, &h1, 2);
            }
        }
    return w;
}

// the activation rows of the type's dot: `real` = quantized from random floats by the layer's quantizer, else raw random bytes with sane scale fields
std::vector<std::vector<uint8_t>> make_acts(int type, int n, int nt, bool real) {
    const pq::Tr tr = pq::traits(type);
    const int at = tr.vec_dot_type;
    const pq::Tr ta = pq::traits(at);
    const size_t ab = pq::row_size_of(at, n);
    std::vector<std::vector<uint8_t>> acts((size_t) nt, std::vector<uint8_t>(ab));
    for (int t = 0; t < nt; ++t) {
        if (real) {
            std::vector<float> x((size_t) n);
            const float scale = (t % 5 == 0) ? 8.f : 1.f;
            for (auto& v : x) v = (((float) (rnd() % 2001) - 1000.f) * 0.001f) * scale;
            ta.from_float(x.data(), acts[(size_t) t].data(), n);
        } else {
            for (auto& b : acts[(size_t) t]) b = (uint8_t) rnd();
            const size_t bb = pq::type_size_of(at), nbk = ab / bb;
            for (size_t k = 0; k < nbk; ++k) {
                uint8_t* d = acts[(size_t) t].data() + k * bb;
                if (type == 12) {   // block_q8_K: float d, int8 qs[256], int16 bsums[16]
                    const float f = 0.001f + (float) (rnd() % 100) * 0.001f;
                    std::memcpy(d, &f, 4);
                    for (int i = 0; i < 16; ++i) {   // bsums as a quantizer writes them (the sums of 16 values), for the real arithmetic
                        int s = 0;
                        for (int j = 0; j < 16; ++j) s += (int8_t) d[4 + i * 16 + j];
                        const int16_t s16 = (int16_t) s;
                        std::memcpy(d + 260 + 2 * i, &s16, 2);
                    }
                } else {            // block_q8_1: half d, half s, qs[32]  /  block_q8_0: half d, qs[32]
                    const uint16_t h = half_bits(0.001f + (float) (rnd() % 100) * 0.001f);
                    std::memcpy(d, &h, 2);
                    if (type == 7) { const uint16_t s = half_bits(0.01f + (float) (rnd() % 500) * 0.01f); std::memcpy(d + 2, &s, 2); }
                }
            }
        }
    }
    return acts;
}

double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int check(int type, int n, int rows, int& cases) {
    int bad = 0;
    const pq::Tr tr = pq::traits(type);
    size_t rb = 0;
    const std::vector<uint8_t> w = make_weights(type, 2 * rows, n, rb);   // gate rows then up rows (the Q4_K gu path), or 2 x rows of down rows
    for (const bool real : {true, false}) {
        const int NT = 160;
        const auto acts = make_acts(type, n, NT, real);
        const void* ap[160];
        for (int t = 0; t < NT; ++t) ap[t] = acts[(size_t) t].data();
        for (const int nt : {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 23, 32, 33, 64, 100, 160}) {
            std::vector<float> got((size_t) nt * rows), want((size_t) nt * rows);
            std::vector<float*> gp((size_t) nt);
            for (int t = 0; t < nt; ++t) gp[(size_t) t] = &got[(size_t) t * rows];
            cpu::kq512_rows(type, w.data(), rb, n, ap, nt, gp.data(), 0, rows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r) tr.dot(n, &want[(size_t) t * rows + r], w.data() + (size_t) r * rb, ap[t]);
            ++cases;
            if (std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0) {
                ++bad;
                std::printf("  FAIL %s rows n=%d nt=%d %s\n", tname(type), n, nt, real ? "real activations" : "raw bytes");
            }
            // a sub-range of rows (the pool hands a thread a row range)
            if (rows > 8) {
                std::vector<float> g2((size_t) nt * rows, 0.f);
                for (int t = 0; t < nt; ++t) gp[(size_t) t] = &g2[(size_t) t * rows];
                cpu::kq512_rows(type, w.data(), rb, n, ap, nt, gp.data(), 3, rows - 2);
                bool same = true;
                for (int t = 0; t < nt && same; ++t)
                    for (int r = 3; r < rows - 2; ++r)
                        if (std::memcmp(&g2[(size_t) t * rows + r], &want[(size_t) t * rows + r], 4) != 0) { same = false; break; }
                ++cases;
                if (!same) { ++bad; std::printf("  FAIL %s row range n=%d nt=%d\n", tname(type), n, nt); }
            }
            if (type == 12) {   // gate/up + SwiGLU
                std::vector<float> gu((size_t) nt * rows), wu((size_t) nt * rows);
                for (int t = 0; t < nt; ++t) gp[(size_t) t] = &gu[(size_t) t * rows];
                cpu::kq512_gu_rows(type, w.data(), rb, (size_t) rows * rb, n, ap, nt, gp.data(), 0, rows);
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < rows; ++r) {
                        float g = 0.f, u = 0.f;
                        tr.dot(n, &g, w.data() + (size_t) r * rb, ap[t]);
                        tr.dot(n, &u, w.data() + (size_t) rows * rb + (size_t) r * rb, ap[t]);
                        wu[(size_t) t * rows + r] = (g / (1.f + std::exp(-g))) * u;
                    }
                ++cases;
                if (std::memcmp(gu.data(), wu.data(), gu.size() * sizeof(float)) != 0) {
                    ++bad;
                    std::printf("  FAIL Q4_K gate/up n=%d nt=%d %s\n", n, nt, real ? "real activations" : "raw bytes");
                }
            }
        }
    }
    return bad;
}

void bench(int type, int n, int rows) {
    const pq::Tr tr = pq::traits(type);
    size_t rb = 0;
    const std::vector<uint8_t> w = make_weights(type, rows, n, rb);
    std::printf("%s n=%d, %d rows (one thread, weights %.0f KB):\n", tname(type), n, rows, (double) rows * rb / 1024.0);
    for (const int nt : {1, 2, 4, 6, 8, 16, 32, 64, 160}) {
        const auto acts = make_acts(type, n, nt, true);
        std::vector<const void*> ap((size_t) nt);
        for (int t = 0; t < nt; ++t) ap[(size_t) t] = acts[(size_t) t].data();
        std::vector<float> out((size_t) nt * rows);
        std::vector<float*> op((size_t) nt);
        for (int t = 0; t < nt; ++t) op[(size_t) t] = &out[(size_t) t * rows];
        const int it = std::max(3, 3000 / (nt * 4));
        auto run512 = [&] { cpu::kq512_rows(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows); };
        auto run256 = [&] { cpu::kq256_rows(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows); };
        auto rundot = [&] {
            for (int r = 0; r < rows; ++r)
                for (int t = 0; t < nt; ++t) tr.dot(n, &op[(size_t) t][r], w.data() + (size_t) r * rb, ap[(size_t) t]);
        };
        double best[3] = {1e30, 1e30, 1e30};
        for (int rep = 0; rep < 5; ++rep) {
            double t0 = now_us();
            for (int i = 0; i < it; ++i) run512();
            best[0] = std::min(best[0], (now_us() - t0) / it);
            t0 = now_us();
            for (int i = 0; i < it; ++i) run256();
            best[1] = std::min(best[1], (now_us() - t0) / it);
            t0 = now_us();
            for (int i = 0; i < it; ++i) rundot();
            best[2] = std::min(best[2], (now_us() - t0) / it);
        }
        std::printf("  nt %3d: AVX-512 %8.1f us   AVX2 multi-token %8.1f us (%.2fx)   dot per token %8.1f us (%.2fx)\n", nt, best[0], best[1], best[1] / best[0], best[2], best[2] / best[0]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (!cpu::cpu_avx512_ok()) {
        std::printf("kq512_parity: SKIP, this CPU (%s) has no AVX-512 F/BW/VL/DQ/VNNI/VBMI\n", cpu::cpu_name().c_str());
        return 77;
    }
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) do_bench = do_bench || std::string(argv[i]) == "--bench";
    std::printf("kq512_parity: %s\n", cpu::cpu_name().c_str());
    int bad = 0, cases = 0;
    for (const int n : {256, 512, 2560}) bad += check(12, n, 24, cases);       // gate/up: n_embd 2560 as the model has it, and shorter rows
    for (const int n : {32, 96, 640}) bad += check(7, n, 24, cases);           // down: n_ff 640 (20 blocks), an odd 3-block row
    for (const int n : {32, 96, 640}) bad += check(8, n, 24, cases);
    std::printf("kq512_parity: %d cases, %d differ: %s\n", cases, bad, bad == 0 ? "PASS" : "FAIL");
    if (do_bench) {
        bench(12, 2560, 256);
        bench(7, 640, 256);
        bench(8, 640, 256);
    }
    return bad == 0 ? 0 : 1;
}
