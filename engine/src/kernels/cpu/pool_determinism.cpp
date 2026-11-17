// src/kernels/cpu/pool_determinism.cpp - the CPU expert pool's native rows are deterministic: the same inputs through native_quant_act / native_quant_h /
// native_gu_rows / native_down_rows (Default and WindowInvariant) give the same bits however many times, in whatever thread count, with whatever row split, and
// whatever the output and activation buffers held before.
//
//   pool_determinism [--reps N] [--seed S] [--type G,D]
//
// For every (gate/up, down) format pair the pool accepts (native_fmt) and nt = 1..8 tokens it checks
//   1. the activation quantizers write every byte they own: a destination pre-filled with 0x00 and one with 0xFF end up equal (an uninitialised byte, such as a
//      Q8_K block sum of an all-zero block, would show) - bytes beyond act_bytes / h_bytes are never read: a canary after the buffer must stay intact;
//   2. the rows of a pass are independent of how they are split: the full range against random chunks (what the pool's job split does);
//   3. T threads (1, 2, 4, 8, 16) each taking a share of the rows at the same time, `reps` times over, equal the single-thread reference bit for bit;
//   4. the output buffers' previous contents do not matter (pre-filled with a NaN pattern and with zeros).
// Exit status 0 = nothing differed.  For the buffer checks configure a build with -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" or run it under valgrind (the
// canary and the pre-fill catch the same class without them).
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool_quant.hpp"

namespace kc = strata::kernels::cpu;

namespace {
constexpr int kH = 2048, kFF = 512;   // whole blocks of every format (256-value super-blocks, 64-value Q2_0 / NVFP4 blocks)
constexpr int kMaxNt = 8;
int g_fail = 0;
long g_checks = 0;

std::map<std::string, int> g_kinds;   // failure kind (the message without its numbers) -> count
void fail(const std::string& what) {
    ++g_fail;
    std::string kind;
    for (char ch : what)
        if (!std::isdigit((unsigned char) ch)) kind += ch;
    if (g_kinds[kind]++ < 3) std::printf("FAIL %s\n", what.c_str());
}

// fp16 scale at the front of every block where the format has it there; everything else random bytes (the result may be inf / NaN: compared as bits)
void fill_blob(std::vector<uint8_t>& blob, const kc::NativeFmt& f, std::mt19937_64& rng) {
    for (size_t i = 0; i < blob.size(); i += 8) {
        const uint64_t v = rng();
        std::memcpy(&blob[i], &v, std::min<size_t>(8, blob.size() - i));
    }
    auto fix = [&](size_t off, size_t bytes, int type) {
        const size_t bs = kc::pq::type_size_of(type);
        if (bs == 0 || type == kc::kNvfp4Type) return;
        for (size_t o = 0; o + bs <= bytes; o += bs) {
            const uint16_t h = kc::pq::fp16_bits(0.004f + 0.001f * (float) (rng() % 16));
            std::memcpy(&blob[off + o], &h, 2);
        }
    };
    fix(0, f.down_off, f.gu_type);
    fix(f.down_off, f.bytes - f.down_off - (f.tail_off ? 16 : 0), f.d_type);
    if (f.tail_off) {   // NVFP4's per-expert scales
        const float tail[4] = {1.f, 1.f, 1.f, 0.f};
        std::memcpy(&blob[f.tail_off], tail, sizeof tail);
    }
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }

struct Case {
    kc::NativeFmt f;
    std::vector<uint8_t> blob;
    std::vector<std::vector<float>> x, h;
};

// quantize `nt` tokens into buffers pre-filled with `fill`, each followed by a canary
struct Acts {
    std::vector<std::vector<uint8_t>> a, hq;
    std::vector<const void*> ap, hp;
};
constexpr size_t kCanary = 64;
void make_acts(const Case& c, int nt, uint8_t fill, Acts& o, int& canary_bad) {
    o.a.assign(nt, std::vector<uint8_t>(c.f.act_bytes + kCanary, fill));
    o.hq.assign(nt, std::vector<uint8_t>(c.f.h_bytes + kCanary, fill));
    for (int t = 0; t < nt; ++t) {
        std::memset(o.a[t].data() + c.f.act_bytes, 0x5A, kCanary);
        std::memset(o.hq[t].data() + c.f.h_bytes, 0x5A, kCanary);
        kc::native_quant_act(c.f, c.x[t].data(), o.a[t].data());
        kc::native_quant_h(c.f, c.h[t].data(), o.hq[t].data());
        for (size_t i = 0; i < kCanary; ++i)
            if (o.a[t][c.f.act_bytes + i] != 0x5A || o.hq[t][c.f.h_bytes + i] != 0x5A) ++canary_bad;
    }
    o.ap.resize(nt);
    o.hp.resize(nt);
    for (int t = 0; t < nt; ++t) o.ap[t] = o.a[t].data(), o.hp[t] = o.hq[t].data();
}

// A Q8_K block of all zeros (d == 0) leaves its 16 block sums unwritten - in ggml's quantize_row_q8_K_ref as in the layer's, byte for byte.  The dots multiply them by
// y.d == 0, so they never reach a result (the row checks below prove it); for the byte comparison of two quantizations they are don't-care.
void mask_zero_block_sums(uint8_t* p, int act_type, size_t bytes) {
    if (act_type != 15) return;
    for (size_t o = 0; o + 292 <= bytes; o += 292) {
        float d;
        std::memcpy(&d, p + o, 4);
        if (d == 0.f) std::memset(p + o + 4 + 256, 0, 32);
    }
}

using GuFn = void (*)(const kc::NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int);
using DnFn = void (*)(const kc::NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int);

// one pass over rows [0, kFF) (gate/up) and [0, kH) (down) in `chunks` pieces taken by `threads` threads; the outputs are the flat [t][row] arrays
void run_pass(const Case& c, const Acts& ac, int nt, GuFn gu, DnFn dn, int threads, const std::vector<int>& cuts_gu, const std::vector<int>& cuts_dn, float init,
              std::vector<float>& ff, std::vector<float>& out) {
    ff.assign((size_t) nt * kFF, init);
    out.assign((size_t) nt * kH, init);
    std::vector<float*> fp(nt), op(nt);
    for (int t = 0; t < nt; ++t) fp[t] = ff.data() + (size_t) t * kFF, op[t] = out.data() + (size_t) t * kH;
    std::atomic<int> next_gu{0}, next_dn{0};
    auto work = [&] {
        for (;;) {
            const int i = next_gu.fetch_add(1);
            if (i + 1 >= (int) cuts_gu.size()) break;
            gu(c.f, c.blob.data(), ac.ap.data(), nt, fp.data(), cuts_gu[i], cuts_gu[i + 1]);
        }
        for (;;) {
            const int i = next_dn.fetch_add(1);
            if (i + 1 >= (int) cuts_dn.size()) break;
            dn(c.f, c.blob.data(), ac.hp.data(), nt, op.data(), cuts_dn[i], cuts_dn[i + 1]);
        }
    };
    if (threads <= 1) {
        work();
    } else {
        std::vector<std::thread> th;
        for (int i = 0; i < threads; ++i) th.emplace_back(work);
        for (auto& t : th) t.join();
    }
}

std::vector<int> split(int n, int chunks, std::mt19937_64& rng, bool random) {
    std::vector<int> cuts{0};
    if (random) {
        std::vector<int> mid;
        for (int i = 1; i < chunks; ++i) mid.push_back((int) (rng() % (uint64_t) (n - 1)) + 1);
        std::sort(mid.begin(), mid.end());
        for (int m : mid)
            if (m > cuts.back()) cuts.push_back(m);
    } else {
        for (int i = 1; i < chunks; ++i) cuts.push_back(n * i / chunks);
    }
    cuts.push_back(n);
    return cuts;
}

void test_pair(int gt, int dt, int reps, uint64_t seed) {
    Case c;
    std::string err;
    if (!kc::native_fmt(gt, dt, kH, kFF, c.f, err)) return;   // a pair the pool does not take
    std::mt19937_64 rng(seed + (uint64_t) gt * 131 + (uint64_t) dt);
    c.blob.assign(c.f.bytes, 0);
    fill_blob(c.blob, c.f, rng);
    c.x.assign(kMaxNt, std::vector<float>(kH));
    c.h.assign(kMaxNt, std::vector<float>(kFF));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int t = 0; t < kMaxNt; ++t) {
        for (float& v : c.x[t]) v = nd(rng) * (t % 3 == 0 ? 1.f : 0.05f);
        for (float& v : c.h[t]) v = nd(rng) * (t % 2 == 0 ? 1.f : 0.3f);
        if (t == 5) std::fill(c.x[t].begin(), c.x[t].end(), 0.f);   // an all-zero activation: Q8_K's zero blocks
        if (t == 6) std::fill(c.h[t].begin(), c.h[t].begin() + 300, 0.f);
    }
    char tag[64];
    std::snprintf(tag, sizeof tag, "gu %d down %d", gt, dt);
    struct Mode {
        const char* name;
        GuFn gu;
        DnFn dn;
    } modes[] = {{"default", kc::native_gu_rows, kc::native_down_rows}, {"invariant", kc::native_gu_rows_invariant, kc::native_down_rows_invariant}};
    for (const Mode& m : modes) {
        for (int nt = 1; nt <= kMaxNt; ++nt) {
            int canary_bad = 0;
            Acts a0, a1;
            make_acts(c, nt, 0x00, a0, canary_bad);
            make_acts(c, nt, 0xFF, a1, canary_bad);
            ++g_checks;
            if (canary_bad) fail(std::string(tag) + " " + m.name + ": a quantizer wrote past act_bytes / h_bytes");
            for (int t = 0; t < nt; ++t) {
                std::vector<uint8_t> x0 = a0.a[t], x1 = a1.a[t], y0 = a0.hq[t], y1 = a1.hq[t];   // copies: the originals keep their unwritten bytes for the row passes
                mask_zero_block_sums(x0.data(), c.f.gu_act, c.f.act_bytes);
                mask_zero_block_sums(x1.data(), c.f.gu_act, c.f.act_bytes);
                mask_zero_block_sums(y0.data(), c.f.d_act, c.f.h_bytes);
                mask_zero_block_sums(y1.data(), c.f.d_act, c.f.h_bytes);
                ++g_checks;
                if (std::memcmp(x0.data(), x1.data(), c.f.act_bytes) != 0) fail(std::string(tag) + ": gate/up activation bytes depend on the buffer's previous content (token " + std::to_string(t) + ")");
                if (std::memcmp(y0.data(), y1.data(), c.f.h_bytes) != 0) fail(std::string(tag) + ": down activation bytes depend on the buffer's previous content (token " + std::to_string(t) + ")");
            }
            std::vector<float> ref_ff, ref_out, ff, out;
            run_pass(c, a0, nt, m.gu, m.dn, 1, {0, kFF}, {0, kH}, 0.f, ref_ff, ref_out);
            std::vector<float> ff2, out2;
            run_pass(c, a1, nt, m.gu, m.dn, 1, {0, kFF}, {0, kH}, std::nanf(""), ff2, out2);
            ++g_checks;
            if (!same_bits(ref_ff, ff2) || !same_bits(ref_out, out2)) fail(std::string(tag) + " " + m.name + " nt=" + std::to_string(nt) + ": the result depends on the buffers' previous content");
            for (int rep = 0; rep < reps; ++rep) {
                const int threads = (int) (std::vector<int>{1, 2, 4, 8, 16})[(size_t) (rep % 5)];
                const bool rnd = rep % 2 == 0;
                run_pass(c, rep % 3 ? a0 : a1, nt, m.gu, m.dn, threads, split(kFF, 1 + (int) (rng() % 12), rng, rnd), split(kH, 1 + (int) (rng() % 24), rng, rnd),
                         rep % 4 == 0 ? std::nanf("") : 0.f, ff, out);
                ++g_checks;
                if (!same_bits(ref_ff, ff) || !same_bits(ref_out, out)) {
                    fail(std::string(tag) + " " + m.name + " nt=" + std::to_string(nt) + " rep " + std::to_string(rep) + " threads " + std::to_string(threads) +
                         (!same_bits(ref_ff, ff) ? ": gate/up rows differ" : ": down rows differ"));
                    break;
                }
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    int reps = 40;
    uint64_t seed = 99;
    int only_g = -1, only_d = -1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t) std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) std::sscanf(argv[++i], "%d,%d", &only_g, &only_d);
    }
    static const int kGu[] = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 40, 42};
    static const int kDn[] = {2, 6, 7, 8, 10, 11, 12, 13, 14, 20, 21, 23, 40, 42};
    int pairs = 0;
    for (int g : kGu)
        for (int d : kDn) {
            if (only_g >= 0 && (g != only_g || d != only_d)) continue;
            const long before = g_checks;
            test_pair(g, d, reps, seed);
            if (g_checks != before) ++pairs;
        }
    for (const auto& k : g_kinds) std::printf("  %6d x %s\n", k.second, k.first.c_str());
    std::printf("pool_determinism: %d format pairs, %ld checks, %d failures -> %s\n", pairs, g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
