// route_prior_parity - STRATA_ROUTE_PRIOR's biased top-10 (route_prior.cu) on synthetic router rows (#5655).
//
// The kernel runs after the router: ids / weights hold the router's own top-10, and the prior picks again with a
// bonus for resident experts.  A row it can rank in full (ten candidates with a finite selection score, a finite
// positive picked mass) gets the biased top-10 with the router's weights renormalised over it; checked here against
// a serial reference (ids exact, weights within a few ulp).  A row it cannot rank in full (all NaN, -Inf, fewer than
// ten usable scores, +Inf) keeps the router's row untouched, ids and weights alike:
// never a sentinel id, never a mix of the two selections.  Run it under compute-sanitizer as well: a degenerate row
// once wrote s[1 << 25] into the thread's local array.
//
// Run: route_prior_parity   (ctest: route_prior_parity; GPU, synthetic, no model)
#include "strata/kernels/route_prior.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
constexpr int kE = 512, kK = 10;
int checks = 0, failures = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}

struct Row { std::string name; std::vector<float> logits; bool rankable; };

// the spec: strictly greater selection score wins, smallest expert id keeps a tie, a candidate needs a score above
// -FLT_MAX; weights = softmax over the 512 unbiased logits, renormalised over the picks
bool reference(const std::vector<float>& lg, const std::vector<int32_t>& res, float lambda, int32_t* id, float* w) {
    float mx = -INFINITY;
    for (float v : lg) mx = std::fmax(mx, v);
    std::vector<float> s(kE);
    std::vector<double> p(kE);
    for (int e = 0; e < kE; ++e) { s[e] = lg[e] + (res[e] >= 0 ? lambda : 0.0f); p[e] = std::exp((double) lg[e] - (double) mx); }
    std::vector<char> used(kE, 0);
    double sum = 0.0;
    for (int r = 0; r < kK; ++r) {
        int be = -1;
        float best = -FLT_MAX;
        for (int e = 0; e < kE; ++e)
            if (!used[e] && s[e] > best) { best = s[e]; be = e; }
        if (be < 0) return false;
        used[be] = 1; id[r] = be; sum += p[be];
    }
    if (!std::isfinite(sum) || !(sum > 0.0)) return false;
    for (int r = 0; r < kK; ++r) w[r] = (float) (p[id[r]] / sum);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    // --rankable-only: only the rows the prior can rank in full (a build without the degenerate-row containment
    // survives them, so its hash can be compared with this one's)
    const bool rankable_only = argc > 1 && std::strcmp(argv[1], "--rankable-only") == 0;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no card"); return 77; }
    const float lambda = 2.0f, inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(6075 + 5655);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<int32_t> res(kE);
    for (int e = 0; e < kE; ++e) res[e] = (e % 3 == 0) ? e : -1;   // a third resident

    std::vector<Row> rows;
    auto random_row = [&] { std::vector<float> v(kE); for (float& x : v) x = nd(rng); return v; };
    rows.push_back({"all NaN", std::vector<float>(kE, nan), false});
    rows.push_back({"all -Inf", std::vector<float>(kE, -inf), false});
    { auto v = random_row(); v[77] = inf; rows.push_back({"one +Inf", v, false}); }
    { auto v = random_row(); v[3] = nan; rows.push_back({"one NaN among finite", v, true}); }
    for (int k = 0; k <= 10; ++k) {   // k finite logits, the rest -Inf (k < 10: not rankable)
        std::vector<float> v(kE, -inf);
        for (int j = 0; j < k; ++j) v[(j * 37 + 5) % kE] = nd(rng);
        rows.push_back({"finite candidates " + std::to_string(k), v, k == 10});
    }
    for (int k : {1, 9}) {   // k finite logits, the rest NaN
        std::vector<float> v(kE, nan);
        for (int j = 0; j < k; ++j) v[(j * 41 + 11) % kE] = nd(rng);
        rows.push_back({"finite among NaN " + std::to_string(k), v, false});
    }
    rows.push_back({"all equal", std::vector<float>(kE, 1.5f), true});
    { std::vector<float> v(kE); for (int e = 0; e < kE; ++e) v[e] = (float) (e % 4); rows.push_back({"ties in groups", v, true}); }
    { auto v = random_row(); for (int j = 0; j < 12; ++j) v[j * 40 + 1] = 3.0e38f; rows.push_back({"huge finite", v, true}); }
    { std::vector<float> v(kE, -3.0e38f); for (int j = 0; j < 10; ++j) v[j * 50 + 2] = 3.0e38f - j * 1e31f;
      rows.push_back({"extreme spread", v, true}); }
    { std::vector<float> v(kE, -FLT_MAX); rows.push_back({"all -FLT_MAX", v, false}); }
    for (int i = 0; i < 40; ++i) rows.push_back({"random " + std::to_string(i), random_row(), true});
    if (rankable_only)
        rows.erase(std::remove_if(rows.begin(), rows.end(), [](const Row& r) { return !r.rankable; }), rows.end());

    const int n = (int) rows.size();
    std::vector<float> h_logits((size_t) n * kE);
    std::vector<int32_t> h_ids((size_t) n * kK), router_ids((size_t) n * kK);
    std::vector<float> h_w((size_t) n * kK), router_w((size_t) n * kK);
    for (int t = 0; t < n; ++t) {
        std::memcpy(&h_logits[(size_t) t * kE], rows[t].logits.data(), kE * sizeof(float));
        for (int r = 0; r < kK; ++r) {   // the router's row, as the kernel finds it
            router_ids[(size_t) t * kK + r] = 500 - r - (t % 7);
            router_w[(size_t) t * kK + r] = 0.0625f + 0.0125f * (float) r;
        }
    }
    float* d_logits = nullptr; int32_t* d_res = nullptr; int32_t* d_ids = nullptr; float* d_w = nullptr;
    ck(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)), "malloc logits");
    ck(cudaMalloc(&d_res, kE * sizeof(int32_t)), "malloc res");
    ck(cudaMalloc(&d_ids, router_ids.size() * sizeof(int32_t)), "malloc ids");
    ck(cudaMalloc(&d_w, router_w.size() * sizeof(float)), "malloc w");
    ck(cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice), "logits");
    ck(cudaMemcpy(d_res, res.data(), kE * sizeof(int32_t), cudaMemcpyHostToDevice), "res");
    ck(cudaMemcpy(d_ids, router_ids.data(), router_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "ids");
    ck(cudaMemcpy(d_w, router_w.data(), router_w.size() * sizeof(float), cudaMemcpyHostToDevice), "w");
    strata::kernels::route_prior_top10(d_logits, d_res, lambda, d_ids, d_w, n, nullptr);
    const cudaError_t launch = cudaGetLastError(), done = cudaDeviceSynchronize();
    check(launch == cudaSuccess && done == cudaSuccess,
          std::string("kernel ran (") + cudaGetErrorString(launch) + " / " + cudaGetErrorString(done) + ")");
    if (done != cudaSuccess) { std::printf("route_prior_parity: %d checks, %d failed\n", checks, failures); return 1; }
    ck(cudaMemcpy(h_ids.data(), d_ids, h_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "ids back");
    ck(cudaMemcpy(h_w.data(), d_w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost), "w back");

    uint64_t hash = 1469598103934665603ull;   // FNV over the rankable rows' outputs: base and head builds compare it
    for (int t = 0; t < n; ++t) {
        const int32_t* id = &h_ids[(size_t) t * kK];
        const float* w = &h_w[(size_t) t * kK];
        const std::string& nm = rows[t].name;
        bool in_range = true;
        for (int r = 0; r < kK; ++r) in_range = in_range && id[r] >= 0 && id[r] < kE;
        check(in_range, nm + ": every id is an expert");
        int32_t rid[kK]; float rw[kK];
        const bool rankable = reference(rows[t].logits, res, lambda, rid, rw);
        check(rankable == rows[t].rankable, nm + ": the reference agrees on whether the row is rankable");
        if (rankable) {
            bool ids_eq = true, w_ok = true;
            for (int r = 0; r < kK; ++r) {
                ids_eq = ids_eq && id[r] == rid[r];
                w_ok = w_ok && std::isfinite(w[r]) && std::fabs(w[r] - rw[r]) <= 1e-5f * std::max(1.0f, std::fabs(rw[r]));
            }
            check(ids_eq, nm + ": the biased top-10 ids");
            check(w_ok, nm + ": the router's weights renormalised over the picks");
            for (int r = 0; r < kK; ++r) {
                uint32_t b; std::memcpy(&b, &w[r], 4);
                for (uint64_t x : {(uint64_t) (uint32_t) id[r], (uint64_t) b}) { hash ^= x; hash *= 1099511628211ull; }
            }
        } else {
            const bool kept = std::memcmp(id, &router_ids[(size_t) t * kK], kK * sizeof(int32_t)) == 0 &&
                              std::memcmp(w, &router_w[(size_t) t * kK], kK * sizeof(float)) == 0;
            check(kept, nm + ": the router's row is kept whole (ids and weights)");
        }
    }
    cudaFree(d_logits); cudaFree(d_res); cudaFree(d_ids); cudaFree(d_w);
    std::printf("route_prior_parity: %d rows, rankable-rows hash %016llx\n", n, (unsigned long long) hash);
    std::printf("route_prior_parity: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
