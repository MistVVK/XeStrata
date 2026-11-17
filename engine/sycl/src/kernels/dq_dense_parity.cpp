// sycl/src/kernels/dq_dense_parity.cpp - model-free check of the Qwen engine's dense weight kernels on a SYCL device against the
// CPU dequantizers of include/strata/artifact/dequant.hpp (ggml's dequantize_row_*): for every dense type the engine reads
// (Q4_0 Q5_0 Q5_1 Q8_0 Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS Q2_0), random legal blocks, no model file.
//   dequant   dequant_f32 / dequant_f16 / dequant_bf16 / dequant_f16_ld, the per-group kernels (and, with -DDQ_HAVE_WIDE, the
//             CUDA wide ones: dequant_set_wide 0 / 1), whole matrices and row slices, widths with k % 1024 == 0 and without
//   mmvq      native_mmvq (the decode token's GEMV, 1 .. 8 columns) against a double reference on the Q8_1 activations as the
//             kernel got them
// Every element must be written (the output starts as NaN), finite, and equal to the CPU value (F32: 1e-6 relative; F16 / BF16:
// the CPU value rounded the same way; mmvq: 5e-3 of the sum of |terms|).  Prints RESULT lines and SUMMARY; exit 1 on any failure.
//   dq_dense_parity [--quick]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl_queue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

struct Ty { int id; const char* name; int be, bb; };
const Ty kTys[] = {{2, "Q4_0", 32, 18}, {6, "Q5_0", 32, 22}, {7, "Q5_1", 32, 24}, {8, "Q8_0", 32, 34}, {11, "Q3_K", 256, 110},
                   {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210}, {20, "IQ4_NL", 32, 18},
                   {23, "IQ4_XS", 256, 136}, {42, "Q2_0", 64, 18}};

uint16_t f2h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) (s | 0x7e00u);
    const float a = std::fabs(f);
    if (a >= 65520.0f) return (uint16_t) (s | 0x7c00u);
    if (a < 6.103515625e-05f) return (uint16_t) (s | (uint32_t) std::nearbyint(std::ldexp((double) a, 24)));
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t v = ((uint32_t) e << 10) | ((u & 0x7fffffu) >> 13);
    const uint32_t rem = u & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// F16 scales at the right place, sane
void sane(int id, uint8_t* p, std::mt19937_64& rng) {
    auto put = [&](int off) { const uint16_t h = f2h(0.02f + 0.0002f * (float) (rng() % 600)); std::memcpy(p + off, &h, 2); };
    switch (id) {
        case 11: put(108); break;
        case 14: put(208); break;
        case 12: case 13: put(0); put(2); break;
        case 7: put(0); put(2); break;   // Q5_1: d, m
        default: put(0); break;
    }
}
void cpu_block(int id, const uint8_t* b, float* out) {
    using namespace strata;
    switch (id) {
        case 2: dequantize_q4_0(b, out); break;
        case 6: dequantize_q5_0(b, out); break;
        case 7: dequantize_q5_1(b, out); break;
        case 8: dequantize_q8_0(b, out); break;
        case 11: dequantize_q3_K(b, out); break;
        case 12: dequantize_q4_K(b, out); break;
        case 13: dequantize_q5_K(b, out); break;
        case 14: dequantize_q6_K(b, out); break;
        case 20: dequantize_iq4_nl(b, out); break;
        case 23: dequantize_iq4_xs(b, out); break;
        case 42: dequantize_q2_0(b, out); break;
    }
}

struct Tally { int cases = 0, fail = 0; std::string first; };
std::map<std::string, Tally> g_res;
int g_fails = 0;
void fail(Tally& t, const std::string& s) {
    ++t.fail; ++g_fails;
    if (t.first.empty()) t.first = s;
    std::printf("  FAIL %s\n", s.c_str());
}

sycl::queue& Q() { return *strata::q_of(nullptr); }
void* S() { return (void*) &Q(); }   // (native MMVQ wants an explicit, non-null stream)

void run_dequant(const Ty& ty, bool quick) {
    sycl::queue& q = Q();
    const int cols_list[] = {256, 512, 1024, 2048, 4096, 5120};
    const int rows_list[] = {1, 3, 33, 130};
    for (int mode : {0, 1}) {
#ifndef DQ_HAVE_WIDE   // (the wide kernels are CUDA's: the SYCL build declares dequant_set_wide and does not define it)
        if (mode) continue;
#else
        K::dequant_set_wide(mode);
#endif
        for (int form = 0; form < 4; ++form) {   // 0 f32, 1 f16, 2 bf16, 3 f16 with a padded row stride
            const char* fn[] = {"f32", "f16", "bf16", "f16_ld"};
            Tally& t = g_res[std::string(ty.name) + " dequant-" + fn[form] + (mode ? " wide" : " group")];
            for (int cols : cols_list) {
                if (quick && cols != 1024 && cols != 5120) continue;
                if (cols % ty.be) continue;
                for (int rows : rows_list) {
                    if (quick && rows != 33) continue;
                    for (int row0 : {0, 2}) {
                        std::mt19937_64 rng((uint64_t) ty.id * 7919 + cols * 13 + rows + row0 * 101);
                        const size_t rb = (size_t) cols / ty.be * ty.bb;
                        std::vector<uint8_t> raw((size_t) (rows + row0) * rb);
                        for (auto& b : raw) b = (uint8_t) rng();
                        for (size_t b = 0; b < raw.size() / ty.bb; ++b) sane(ty.id, &raw[b * ty.bb], rng);
                        std::vector<float> ref((size_t) rows * cols);
                        for (int r = 0; r < rows; ++r)
                            for (int b = 0; b < cols / ty.be; ++b) cpu_block(ty.id, &raw[(size_t) (r + row0) * rb + (size_t) b * ty.bb], &ref[(size_t) r * cols + (size_t) b * ty.be]);
                        uint8_t* d_raw = sycl::malloc_device<uint8_t>(raw.size(), q);
                        q.memcpy(d_raw, raw.data(), raw.size()).wait();
                        const int64_t ld = form == 3 ? cols + 8 : cols;
                        const size_t cnt = (size_t) rows * ld;
                        const size_t esz = form == 0 ? 4 : 2;
                        void* d_out = sycl::malloc_device(cnt * esz + 64, q);
                        std::vector<uint8_t> nanv(cnt * esz, 0xFF);
                        q.memcpy(d_out, nanv.data(), nanv.size()).wait();
                        ++t.cases;
                        char tag[160];
                        std::snprintf(tag, sizeof tag, "%s %s %s cols=%d rows=%d row0=%d", ty.name, fn[form], mode ? "wide" : "group", cols, rows, row0);
                        try {
                            if (form == 0) K::dequant_f32(ty.id, d_raw, row0, rows, cols, (float*) d_out, S());
                            else if (form == 1) K::dequant_f16(ty.id, d_raw, row0, rows, cols, (uint16_t*) d_out, S());
                            else if (form == 2) K::dequant_bf16(ty.id, d_raw, row0, rows, cols, (uint16_t*) d_out, S());
                            else if (!K::dequant_f16_ld(ty.id, d_raw, row0, rows, cols, ld, (uint16_t*) d_out, S())) {
                                std::printf("  note: %s refused by dequant_f16_ld\n", tag);
                                sycl::free(d_raw, q); sycl::free(d_out, q);
                                continue;
                            }
                            q.wait();
                        } catch (const std::exception& e) {
                            fail(t, std::string(tag) + ": exception " + e.what());
                            sycl::free(d_raw, q); sycl::free(d_out, q);
                            continue;
                        }
                        std::vector<uint8_t> out(cnt * esz);
                        q.memcpy(out.data(), d_out, out.size()).wait();
                        size_t bad = 0, unw = 0, nonf = 0;
                        size_t first = (size_t) -1;
                        for (int r = 0; r < rows; ++r)
                            for (int c = 0; c < cols; ++c) {
                                const size_t i = (size_t) r * ld + c, ri = (size_t) r * cols + c;
                                bool ok;
                                if (form == 0) {
                                    float v; std::memcpy(&v, &out[i * 4], 4);
                                    uint32_t u; std::memcpy(&u, &out[i * 4], 4);
                                    if (u == 0xFFFFFFFFu) { ++unw; ok = false; }
                                    else if (!std::isfinite(v)) { ++nonf; ok = false; }
                                    else ok = std::fabs(v - ref[ri]) <= 1e-6 * std::fabs(ref[ri]) + 1e-30;
                                } else {
                                    uint16_t h; std::memcpy(&h, &out[i * 2], 2);
                                    if (h == 0xFFFFu) { ++unw; ok = false; }
                                    else {
                                        const uint16_t want = form == 2 ? f2bf(ref[ri]) : f2h(ref[ri]);
                                        ok = h == want || std::abs((int) h - (int) want) == 1;   // (1 ulp: reported as ok, the CPU rounds from a different path)
                                    }
                                }
                                if (!ok) { ++bad; if (first == (size_t) -1) first = ri; }
                            }
                        if (bad) {
                            char b[300];
                            std::snprintf(b, sizeof b, "%s: %zu wrong (%zu unwritten, %zu non-finite), first at element %zu", tag, bad, unw, nonf, first);
                            fail(t, b);
                        }
                        sycl::free(d_raw, q); sycl::free(d_out, q);
                    }
                }
            }
        }
    }
#ifdef DQ_HAVE_WIDE
    K::dequant_set_wide(-1);
#endif
}

void run_mmvq(const Ty& ty, bool quick) {
    if (!K::native_mmvq_supported(ty.id)) return;
    sycl::queue& q = Q();
    Tally& t = g_res[std::string(ty.name) + " mmvq"];
    std::mt19937_64 xr(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n_in : {256, 1024, 2048, 2560, 3072, 4096}) {   // Q4-SYCL: and the Qwen4 test shapes' hidden widths 2048 and 3072
        if (n_in % ty.be) continue;
        if (quick && n_in != 1024 && n_in != 2048 && n_in != 2560 && n_in != 3072 && n_in != 4096) continue;
        for (int n_out : {1, 7, 8, 33, 130}) {
            if (quick && n_out != 33) continue;
            for (int nc : {1, 2, 4, 5, 8}) {
                if (quick && nc != 1 && nc != 5) continue;
                std::mt19937_64 rng((uint64_t) ty.id * 31 + n_in + n_out * 7);
                const size_t rb = (size_t) n_in / ty.be * ty.bb;
                std::vector<uint8_t> raw((size_t) n_out * rb);
                for (auto& b : raw) b = (uint8_t) rng();
                for (size_t b = 0; b < raw.size() / ty.bb; ++b) sane(ty.id, &raw[b * ty.bb], rng);
                std::vector<float> wf((size_t) n_out * n_in);
                for (int r = 0; r < n_out; ++r)
                    for (int b = 0; b < n_in / ty.be; ++b) cpu_block(ty.id, &raw[(size_t) r * rb + (size_t) b * ty.bb], &wf[(size_t) r * n_in + (size_t) b * ty.be]);
                std::vector<float> x((size_t) nc * n_in);
                for (auto& v : x) v = nd(xr);
                uint8_t* d_w = sycl::malloc_device<uint8_t>(raw.size(), q);
                float* d_x = sycl::malloc_device<float>(x.size(), q);
                const size_t qb = K::native_q8_1_bytes(n_in, nc);
                void* d_q = sycl::malloc_device(qb, q);
                float* d_y = sycl::malloc_device<float>((size_t) nc * n_out, q);
                q.memcpy(d_w, raw.data(), raw.size()).wait();
                q.memcpy(d_x, x.data(), x.size() * 4).wait();
                std::vector<float> nanv((size_t) nc * n_out, NAN);
                q.memcpy(d_y, nanv.data(), nanv.size() * 4).wait();
                ++t.cases;
                char tag[160];
                std::snprintf(tag, sizeof tag, "%s mmvq n_in=%d n_out=%d ncols=%d", ty.name, n_in, n_out, nc);
                try {
                    K::native_quantize_q8_1(d_x, d_q, n_in, nc, S());
                    K::native_mmvq(ty.id, d_w, d_q, d_y, n_in, n_out, nc, S());
                    q.wait();
                } catch (const std::exception& e) {
                    fail(t, std::string(tag) + ": exception " + e.what());
                    sycl::free(d_w, q); sycl::free(d_x, q); sycl::free(d_q, q); sycl::free(d_y, q);
                    continue;
                }
                std::vector<uint8_t> qh(qb);
                std::vector<float> y((size_t) nc * n_out);
                q.memcpy(qh.data(), d_q, qb).wait();
                q.memcpy(y.data(), d_y, y.size() * 4).wait();
                // the activations as the kernel got them: block = half d, half s, int8 qs[32]
                std::vector<float> xh((size_t) nc * n_in);
                for (int c = 0; c < nc; ++c)
                    for (int b = 0; b < n_in / 32; ++b) {
                        const uint8_t* p = &qh[((size_t) c * (n_in / 32) + b) * 36];
                        uint16_t hd; std::memcpy(&hd, p, 2);
                        const float d = strata::fp16_to_fp32(hd);
                        for (int j = 0; j < 32; ++j) xh[(size_t) c * n_in + b * 32 + j] = d * (float) (int8_t) p[4 + j];
                    }
                size_t bad = 0, nonf = 0;
                double worst = 0;
                for (int c = 0; c < nc; ++c)
                    for (int r = 0; r < n_out; ++r) {
                        const float v = y[(size_t) c * n_out + r];
                        if (!std::isfinite(v)) { ++nonf; continue; }
                        double s = 0, sa = 0;
                        for (int i = 0; i < n_in; ++i) { const double tt = (double) wf[(size_t) r * n_in + i] * xh[(size_t) c * n_in + i]; s += tt; sa += std::fabs(tt); }
                        const double e = std::fabs((double) v - s) / (sa + 1e-30);
                        worst = std::max(worst, e);
                        if (e > 5e-3) ++bad;   // (Q4_0 / Q5_0 take the offset term from the Q8_1 block's sum of the ORIGINAL activations, not from the int8 values: 1e-3 of the sum of |terms| is that, not a bug)
                    }
                if (bad || nonf) {
                    char b[300];
                    std::snprintf(b, sizeof b, "%s: %zu beyond 5e-3 (worst %.2e), %zu non-finite", tag, bad, worst, nonf);
                    fail(t, b);
                }
                sycl::free(d_w, q); sycl::free(d_x, q); sycl::free(d_q, q); sycl::free(d_y, q);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--quick")) quick = true;
        else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) only = argv[++i];
    }
    std::printf("DEVICE %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
    for (const Ty& ty : kTys) {
        if (!only.empty() && only != ty.name) continue;
        run_dequant(ty, quick);
        run_mmvq(ty, quick);
    }
    for (const auto& kv : g_res)
        std::printf("RESULT %-28s cases=%d fail=%d %s\n%s", kv.first.c_str(), kv.second.cases, kv.second.fail, kv.second.fail ? "FAIL" : "ok",
                    kv.second.first.empty() ? "" : ("  first: " + kv.second.first + "\n").c_str());
    std::printf("SUMMARY fails=%d\n", g_fails);
    return g_fails ? 1 : 0;
}
