// FNX_PFX_GDN_TC=1 against the kernels it replaces, on one card (not a ctest: needs a GPU, no model).
//   bin/split_gdn_tc_test [nk nv [C ...]]       default nk 8, nv 24 (one card of two), C = 33 1000 2048 8192 32768
// The prompt GDN's recurrence for one card's slice, random inputs of the model's shapes (q, k unit rows, gates from ab / dt_bias / A as
// k_gdn_ch_prep makes them): the existing sub-chunk path (k_gdn_ch_prep + k_gdn_ch_rec, fp32 on the CUDA cores) and Strata's tensor-core scan
// (gdn_tc_run, f16 hi / lo split products) from the same state, compared on the outputs and the end state, and timed (the two launches each).
// Compiles kernels.cu's part 4 as one translation unit with its own main, so the kernels in the anonymous namespace are reachable.
#define FNX_KPART 4
#include "../src/kernels.cu"

#include <cstdint>
#include <random>

namespace fnx { std::vector<Q8Twin>& q8_twins() { static std::vector<Q8Twin> v; return v; } }   // part 0 has it; nothing here registers a twin

namespace {
using namespace fnx;

struct Rng {
    std::mt19937_64 g{12345};
    float uni() { return (float) std::uniform_real_distribution<double>(0.0, 1.0)(g); }
    float nrm() { return (float) std::normal_distribution<double>(0.0, 1.0)(g); }
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

double rel_diff(const std::vector<float>& r, const std::vector<float>& x, double& maxval) {
    double md = 0, mv = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        md = std::max(md, (double) std::fabs(r[i] - x[i]));
        mv = std::max(mv, (double) std::fabs(r[i]));
    }
    maxval = mv;
    return md / std::max(mv, 1e-30);
}
}  // namespace

int main(int argc, char** argv) {
    int nk = 8, nv = 24;
    std::vector<int> cases = {33, 1000, 2048, 8192, 32768};
    if (argc >= 3) { nk = std::atoi(argv[1]); nv = std::atoi(argv[2]); }
    if (argc > 3) { cases.clear(); for (int i = 3; i < argc; ++i) cases.push_back(std::atoi(argv[i])); }
    const int vpk = nv / nk;
    const int ch = (2 * nk + nv) * HS;
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, dev));
    std::printf("card %s (sm_%d%d, %d SMs), a slice of %d key heads and %d value heads\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount, nk, nv);
    CK(cudaFuncSetAttribute(k_gdn_ch_rec, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) GCH_REC_SMEM));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::vector<int> vmap(nv);
    for (int lv = 0; lv < nv; ++lv) vmap[lv] = (lv / vpk) | (lv % vpk == 0 ? 1 << 8 : 0);
    int* d_vmap;
    CK(cudaMalloc(&d_vmap, nv * 4));
    CK(cudaMemcpy(d_vmap, vmap.data(), nv * 4, cudaMemcpyHostToDevice));
    int fails = 0;
    for (int variant = 0; variant < 2; ++variant) {   // 0: random state, 1: zero state (a fresh prompt)
        for (int C : cases) {
            Rng rng;
            std::vector<float> xc((size_t) C * ch), ab((size_t) C * 2 * nv), dt(nv), a(nv), S0((size_t) nv * HS * HS);
            for (int t = 0; t < C; ++t) {
                float* row = xc.data() + (size_t) t * ch;
                for (int h = 0; h < 2 * nk; ++h) {   // q, k: unit rows per head
                    double ss = 0;
                    for (int i = 0; i < HS; ++i) { row[h * HS + i] = rng.nrm(); ss += (double) row[h * HS + i] * row[h * HS + i]; }
                    const float inv = (float) (1.0 / std::sqrt(ss + 1e-6));
                    for (int i = 0; i < HS; ++i) row[h * HS + i] *= inv;
                }
                for (int i = 0; i < nv * HS; ++i) row[2 * nk * HS + i] = 0.5f * rng.nrm();
                for (int i = 0; i < 2 * nv; ++i) ab[(size_t) t * 2 * nv + i] = rng.nrm();
            }
            for (int i = 0; i < nv; ++i) { dt[i] = 0.5f * rng.nrm(); a[i] = -std::exp(-3.5f + 5.5f * rng.uni()); }
            for (auto& x : S0) x = variant ? 0.0f : 0.05f * rng.nrm();
            float *d_xc, *d_ab, *d_dt, *d_a, *d_S, *d_y0, *d_y1, *d_chw, *d_S0;
            const size_t ny = (size_t) C * nv * HS, ns = (size_t) nv * HS * HS;
            const size_t chw_n = (size_t) ((C + GCH - 1) / GCH) * nv * GCH_PREP_FLOATS;
            CK(cudaMalloc(&d_xc, xc.size() * 4)); CK(cudaMalloc(&d_ab, ab.size() * 4)); CK(cudaMalloc(&d_dt, nv * 4)); CK(cudaMalloc(&d_a, nv * 4));
            CK(cudaMalloc(&d_S, ns * 4)); CK(cudaMalloc(&d_S0, ns * 4)); CK(cudaMalloc(&d_y0, ny * 4)); CK(cudaMalloc(&d_y1, ny * 4)); CK(cudaMalloc(&d_chw, chw_n * 4));
            CK(cudaMemcpy(d_xc, xc.data(), xc.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_ab, ab.data(), ab.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_dt, dt.data(), nv * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_a, a.data(), nv * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_S0, S0.data(), ns * 4, cudaMemcpyHostToDevice));
            auto run_old = [&] {
                k_gdn_ch_prep<<<dim3((C + GCH - 1) / GCH, nv), GCH_THREADS, 0, s>>>(d_xc, d_ab, d_dt, d_a, d_vmap, nk, nv, C, d_chw);
                k_gdn_ch_rec<<<nv * GCH_REC_BLOCKS_PER_HEAD, GCR_THREADS, GCH_REC_SMEM, s>>>(d_xc, d_chw, d_S, d_y0, nk, nv, d_vmap, C);
            };
            auto run_new = [&]() -> bool { return gdn_tc_run(d_xc, d_ab, d_dt, d_a, d_S, d_y1, nk, nv, d_vmap, C, s); };
            CK(cudaMemcpy(d_S, d_S0, ns * 4, cudaMemcpyDeviceToDevice));
            run_old();
            CK(cudaStreamSynchronize(s));
            std::vector<float> y0(ny), s_old(ns), y1(ny), s_new(ns);
            CK(cudaMemcpy(y0.data(), d_y0, ny * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s_old.data(), d_S, ns * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d_S, d_S0, ns * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemset(d_y1, 0, ny * 4));
            if (!run_new()) { std::printf("C %6d: gdn_tc_run declined (no scratch or too small a card)\n", C); ++fails; continue; }
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            CK(cudaMemcpy(y1.data(), d_y1, ny * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s_new.data(), d_S, ns * 4, cudaMemcpyDeviceToHost));
            double my, ms_;
            const double dy = rel_diff(y0, y1, my), ds = rel_diff(s_old, s_new, ms_);
            // timing: the two launches each, back to back, 5 runs after the checked one (the state is overwritten each run: the cost does not depend on it)
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            float t_old = 0, t_new = 0;
            const int reps = C >= 8192 ? 3 : 10;
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) run_old();
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t_old, e0, e1));
            CK(cudaEventRecord(e0, s));
            for (int r = 0; r < reps; ++r) run_new();
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t_new, e0, e1));
            const bool ok = dy < 2e-3 && ds < 2e-3 && std::isfinite(dy) && std::isfinite(ds);
            fails += !ok;
            std::printf("%s C %6d: y diff %.2e (max |y| %.3g), end state diff %.2e (max |S| %.3g) | old %.3f ms, tensor-core %.3f ms (%.2fx) %s\n",
                        variant ? "zero state  " : "random state", C, dy, my, ds, ms_, t_old / reps, t_new / reps, t_old / std::max(t_new, 1e-9f), ok ? "ok" : "FAIL");
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            cudaFree(d_xc); cudaFree(d_ab); cudaFree(d_dt); cudaFree(d_a); cudaFree(d_S); cudaFree(d_S0); cudaFree(d_y0); cudaFree(d_y1); cudaFree(d_chw);
        }
    }
    std::printf("%d failures\n", fails);
    return fails ? 1 : 0;
}
