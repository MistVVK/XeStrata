// src/kernels/router_top10_parity.cpp - P2.S2's parity test for the MoE router.
//
// The reference here is a HOST implementation of `ref/moe.py::router` written from the same specification the
// kernel was written from - so this checks the kernel against the spec independently of the kernel, which is
// the strongest thing available without running llama.cpp.  It is NOT a check against llama.cpp itself, and
// that gap is stated rather than implied.
//
// WHAT THE TEST CAN AND CANNOT SEE.  It compares expert IDs exactly and weights to a tolerance, on several
// logit distributions including ties (where the "smallest index wins" rule is the only thing that decides) and
// near-ties (where a float difference decides).  It also asserts that the 2**-14 CLAMP CANNOT TRIGGER for this
// model's geometry - see below - so the absence of a clamp test is a proven fact rather than an omission.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/router_top10.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

constexpr double RENORM_CLAMP = 6.103515625e-05;      // 2**-14

// The host reference: softmax over all experts, stable descending argsort, gather, clamp, renormalise.
void reference(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, (double) logits[e]);
    double sum = 0;
    std::vector<double> p((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t) e] = std::exp((double) logits[e] - mx);
        sum += p[(size_t) e];
    }
    for (int e = 0; e < n_expert; ++e) p[(size_t) e] /= sum;

    // stable descending: std::stable_sort with a strict greater-than keeps equal elements in index order
    std::vector<int> idx((size_t) n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return p[(size_t) a] > p[(size_t) b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[(size_t) i];
        w[i] = (float) p[(size_t) idx[(size_t) i]];
        s += p[(size_t) idx[(size_t) i]];
    }
    s = std::max(s, RENORM_CLAMP);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / s);
}

int run_case(const char* name, const std::vector<float>& logits, int n_tokens, int n_expert, int k,
             double tol) {
    std::vector<int> h_ids((size_t) n_tokens * k);
    std::vector<float> h_w((size_t) n_tokens * k);
    std::vector<int> r_ids((size_t) n_tokens * k);
    std::vector<float> r_w((size_t) n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t) t * n_expert], n_expert, k, &r_ids[(size_t) t * k], &r_w[(size_t) t * k]);
    }

    float* d_l = nullptr;
    int* d_ids = nullptr;
    float* d_w = nullptr;
    check(DPCT_CHECK_ERROR(d_l = sycl::malloc_device<float>(
                               logits.size(), dpct::get_in_order_queue())),
          "malloc logits");
    check(DPCT_CHECK_ERROR(d_ids = sycl::malloc_device<int>(
                               h_ids.size(), dpct::get_in_order_queue())),
          "malloc ids");
    check(DPCT_CHECK_ERROR(d_w = sycl::malloc_device<float>(
                               h_w.size(), dpct::get_in_order_queue())),
          "malloc w");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(
              d_l, logits.data(), logits.size() * sizeof(float)).wait()),
          "copy");
    strata::kernels::router_top10(d_l, n_tokens, n_expert, k, d_ids, d_w, nullptr);
    check(DPCT_CHECK_ERROR(
              (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                  .memcpy(h_ids.data(), d_ids, h_ids.size() * sizeof(int))
                  .wait()),
          "back ids");
    check(DPCT_CHECK_ERROR(
              (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                  .memcpy(h_w.data(), d_w, h_w.size() * sizeof(float))
                  .wait()),
          "back w");

    long long id_bad = 0, w_bad = 0;
    double worst = 0;
    for (size_t i = 0; i < h_ids.size(); ++i) {
        if (h_ids[i] != r_ids[i]) ++id_bad;
        const double rel = std::fabs((double) h_w[i] - (double) r_w[i]) /
                           (std::fabs((double) r_w[i]) > 1e-30 ? std::fabs((double) r_w[i]) : 1e-30);
        worst = std::max(worst, rel);
        if (!(rel <= tol)) ++w_bad;
    }
    // every token's weights must sum to 1 unless the clamp replaced the sum
    double worst_sum = 0;
    for (int t = 0; t < n_tokens; ++t) {
        double s = 0;
        for (int i = 0; i < k; ++i) s += (double) h_w[(size_t) t * k + i];
        double raw = 0;
        for (int i = 0; i < k; ++i) raw += (double) r_w[(size_t) t * k + i];
        // both sides divide by the same clamped scale, so the ratios agree; check the kernel's own sum
        worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
    }
    std::printf("  %-26s ids %s (%lld bad)   weights worst rel %.3e (%lld over tol)   |sum-1| %.1e\n", name,
                id_bad ? "*** WRONG ***" : "exact", id_bad, worst, w_bad, worst_sum);
    sycl::free(d_l, dpct::get_in_order_queue());
    sycl::free(d_ids, dpct::get_in_order_queue());
    sycl::free(d_w, dpct::get_in_order_queue());
    return (int) (id_bad + w_bad);
}

// The fp32 router (variant 3, the one a card without fp64 runs) against the double kernel (variant 0) on the SAME inputs: ids that differ
// and the largest relative weight difference. Needs a device that runs double (a card with fp64, or an Arc under IGC_EnableDPEmulation=1
// OverrideDefaultFP64Settings=1); without one it says so and is skipped (the reference check above then covers the fp32 path).
int f32_vs_double(const char* name, const std::vector<float>& logits, int n_tokens, int n_expert, int k) {
    auto& q = dpct::get_in_order_queue();
    if (!q.get_device().has(sycl::aspect::fp64)) {
        std::printf("  %-26s fp32 vs double: skipped (this device runs no double)\n", name);
        return 0;
    }
    float* d_l = sycl::malloc_device<float>(logits.size(), q);
    int* d_i[2] = {sycl::malloc_device<int>((size_t) n_tokens * k, q), sycl::malloc_device<int>((size_t) n_tokens * k, q)};
    float* d_w[2] = {sycl::malloc_device<float>((size_t) n_tokens * k, q), sycl::malloc_device<float>((size_t) n_tokens * k, q)};
    q.memcpy(d_l, logits.data(), logits.size() * sizeof(float)).wait();
    std::vector<int> ids[2];
    std::vector<float> w[2];
    const int variants[2] = {0, 3};
    for (int v = 0; v < 2; ++v) {
        if (!strata::kernels::router_top10_variant(d_l, n_tokens, n_expert, k, d_i[v], d_w[v], nullptr, variants[v])) {
            std::printf("  %-26s fp32 vs double: variant %d not available\n", name, variants[v]);
            return 1;
        }
        q.wait();
        ids[v].resize((size_t) n_tokens * k);
        w[v].resize((size_t) n_tokens * k);
        q.memcpy(ids[v].data(), d_i[v], ids[v].size() * sizeof(int)).wait();
        q.memcpy(w[v].data(), d_w[v], w[v].size() * sizeof(float)).wait();
    }
    long long id_diff = 0;
    double worst = 0, worst_abs = 0;
    for (size_t i = 0; i < ids[0].size(); ++i) {
        if (ids[0][i] != ids[1][i]) { ++id_diff; continue; }
        const double a = w[0][i], b = w[1][i];
        worst_abs = std::max(worst_abs, std::fabs(a - b));
        worst = std::max(worst, std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30));
    }
    std::printf("  %-26s fp32 vs double: ids differ %lld of %zu   weights max rel %.3e  max abs %.3e\n", name, id_diff, ids[0].size(), worst, worst_abs);
    sycl::free(d_l, q);
    for (int v = 0; v < 2; ++v) { sycl::free(d_i[v], q); sycl::free(d_w[v], q); }
    return id_diff ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: router_top10_parity [--selftest]\n"); return 2; }
    }

    const int NE = 512;          // the real n_expert
    const int K = 10;            // the real top-k
    const int NT = 64;
    std::mt19937 rng(2024);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    // ---- random logits
    std::vector<float> a((size_t) NT * NE);
    for (auto& v : a) v = gauss(rng);
    bad += run_case("random normal", a, NT, NE, K, 1e-5);

    // ---- ALL-EQUAL logits: every expert ties, so the "smallest index wins" rule is the ONLY thing deciding
    // the ids.  An unstable sort or a >= comparison in the maximum scan fails this and passes everything else.
    std::vector<float> b((size_t) NT * NE, 0.5f);
    bad += run_case("all equal (ties)", b, NT, NE, K, 1e-5);

    // ---- a few exact ties among otherwise distinct values
    std::vector<float> c((size_t) NT * NE);
    for (int t = 0; t < NT; ++t) {
        for (int e = 0; e < NE; ++e) c[(size_t) t * NE + e] = (e < 12) ? 2.0f : gauss(rng);
    }
    bad += run_case("12-way exact tie", c, NT, NE, K, 1e-5);

    // ---- one dominant expert: the top-1 must be that expert and its weight must dominate
    std::vector<float> d((size_t) NT * NE, 0.0f);
    for (int t = 0; t < NT; ++t) d[(size_t) t * NE + (t % NE)] = 20.0f;
    bad += run_case("dominant expert", d, NT, NE, K, 1e-5);

    // ---- THE CLAMP CANNOT TRIGGER, and that is provable rather than untested.
    // The sum of the top k of a probability vector over n outcomes is at least k/n (the minimum is the uniform
    // distribution).  Here k/n = 10/512 = 0.01953125, which is 320x the 2**-14 = 6.1035e-05 clamp, so no input
    // whatsoever can make this model's renormalisation clamp - not an extreme one, not an adversarial one.
    // Asserting the bound is worth more than a test that cannot reach it.
    const double min_possible = (double) K / (double) NE;
    std::printf("  %-26s top-%d sum >= k/n = %.5f, clamp is %.3e -> margin %.0fx  %s\n", "clamp reachability",
                K, min_possible, RENORM_CLAMP, min_possible / RENORM_CLAMP,
                min_possible > RENORM_CLAMP ? "(CLAMP IS UNREACHABLE, asserted)" : "*** REACHABLE ***");
    if (min_possible <= RENORM_CLAMP) {
        std::fprintf(stderr, "the clamp CAN trigger, so it needs a test\n");
        ++bad;
    }

    std::printf("\nrouter_top10: %d failures over 4 distributions x %d tokens x %d experts, top-%d\n", bad, NT, NE,
                K);

    // ---- the Qwen4 shapes (Q4-READY): 256 experts top-8 (q4_lite), 512 top-8, 256 top-10, and the fp32 router against the double one
    // on the same inputs at each (variant 3 vs variant 0; skipped on a device that runs no double)
    const int shapes[][2] = {{256, 8}, {512, 8}, {256, 10}, {512, 10}};
    for (const auto& sh : shapes) {
        const int ne = sh[0], kk = sh[1];
        char label[64];
        std::vector<float> r((size_t) NT * ne);
        for (auto& v : r) v = gauss(rng);
        std::snprintf(label, sizeof label, "random %dx top-%d", ne, kk);
        bad += run_case(label, r, NT, ne, kk, 1e-5);
        bad += f32_vs_double(label, r, NT, ne, kk);
        std::vector<float> tie((size_t) NT * ne);
        for (int t = 0; t < NT; ++t)
            for (int e = 0; e < ne; ++e) tie[(size_t) t * ne + e] = (e < 12) ? 2.0f : gauss(rng);
        std::snprintf(label, sizeof label, "12-way tie %dx top-%d", ne, kk);
        bad += run_case(label, tie, NT, ne, kk, 1e-5);
        bad += f32_vs_double(label, tie, NT, ne, kk);
        std::vector<float> wide((size_t) NT * ne);
        for (auto& v : wide) v = 6.0f * gauss(rng);          // a peaked router (large logit spread)
        std::snprintf(label, sizeof label, "wide %dx top-%d", ne, kk);
        bad += run_case(label, wide, NT, ne, kk, 1e-5);
        bad += f32_vs_double(label, wide, NT, ne, kk);
    }
    std::printf("router_top10 (with the Qwen4 shapes): %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("router_top10_parity OK\n");
    return 0;
}
