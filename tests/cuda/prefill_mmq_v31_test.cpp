// prefill_mmq_v31_test - the prompt path's MMQ products for the expert formats of the Unsloth v3.1 Qwen3.8-Flash-Next
// sets (UD-IQ1_M .. UD-IQ4_XS): gate/up IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ4_XS at 1280 x 2560 and down Q2_0 /
// IQ4_NL / Q8_0 / Q5_0 at 2560 x 640, on synthetic weights quantized by ggml, against a double-precision product of ggml's
// dequantized weights.  MMQ rounds the activations to q8_1 by design, so this is a screen for layout, stride,
// expert-bound and row-id errors, not a bit-exactness test (the bounds are those of prefill_mmq_kquant_test).
//
// Q5_0 has an MMQ instance only in a build with STRATA_ORCA_Q4KS_MMQ=ON (#296): without it the Q5_0 down product is
// reported as "not in this build" (the prompt path then dequantizes to FP16 and runs cuBLAS) and does not fail.
//
//     prefill_mmq_v31_test [--require-q5_0]        --require-q5_0: a build without the Q5_0 instance fails
//     prefill_mmq_v31_test --bench [--reps N]       Q5_0 down: MMQ against the dequantize-to-FP16 + cuBLAS route
//                                                    (what a build without the instance runs), per expert-group size
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/mmq_ids_margin.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

std::vector<uint8_t> make_matrix(ggml_type t, int64_t rows, int64_t cols, int expert, int trial) {
    std::vector<float> w((size_t) (rows * cols));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k) {
            const int code = (int) ((k * 37 + r * 19 + expert * 23 + trial * 11) % 101) - 50;
            w[(size_t) (r * cols + k)] = 0.025f * std::sin(0.013f * (float) (k + 1) + 0.17f * (float) r + 0.31f * (float) expert) +
                                         0.0015f * (float) code;
        }
    std::vector<uint8_t> out((size_t) rows * ggml_row_size(t, cols));
    const std::vector<float> imatrix((size_t) cols, 1.0f);
    ggml_quantize_chunk(t, w.data(), out.data(), 0, rows, cols, ggml_quantize_requires_imatrix(t) ? imatrix.data() : nullptr);
    return out;
}

std::vector<float> make_x(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * (size_t) cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * (size_t) cols + (size_t) k] =
                0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
    std::fill(x.end() - cols, x.end(), 0.0f);   // an all-zero row: exactly zero out
    return x;
}

std::vector<int32_t> perm(int n, int shift, bool reverse) {
    std::vector<int32_t> p((size_t) n);
    for (int i = 0; i < n; ++i) p[(size_t) i] = (int32_t) (((reverse ? n - 1 - i : i) + shift) % n);
    return p;
}

// one grouped product: experts side by side at one stride (a zeroed tail after them, as prefill.cpp stages them)
void product(mmq::Context& ctx, cudaStream_t s, const char* name, ggml_type t, int64_t out_rows, int64_t cols,
             const std::vector<int>& counts, int trial) {
    const int n = (int) counts.size(), rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[(size_t) e + 1] = bounds[(size_t) e] + counts[(size_t) e];
    const auto src = perm(rows, trial + 1, false), dst = perm(rows, trial + 2, true);
    const std::vector<float> x = make_x(rows, cols, trial);
    const size_t eb = mmq::matrix_bytes((int) t, out_rows, cols);
    if (eb != (size_t) out_rows * ggml_row_size(t, cols)) throw std::runtime_error(std::string(name) + ": matrix_bytes");
    std::vector<std::vector<uint8_t>> ws;
    std::vector<uint8_t> w((size_t) n * eb + 4096, 0);
    for (int e = 0; e < n; ++e) {
        ws.push_back(make_matrix(t, out_rows, cols, e, trial));
        std::copy(ws.back().begin(), ws.back().end(), w.begin() + (ptrdiff_t) ((size_t) e * eb));
    }
    // llama.cpp's mul_mat_q and its stream-k fixup read the row-id table up to a J-column tile (J <= 128) past the last row, up to roundup(max_rows, J)
    // from an expert's first row (mmq_ids_margin.hpp): the table has the margin, its extra entries hold row index 0 (valid; they are read, never used)
    const size_t ids_rows = (size_t) mmq::mmq_ids_table_rows((uint64_t) rows, (uint64_t) *std::max_element(counts.begin(), counts.end()));
    std::vector<int32_t> dst_table(dst);
    dst_table.resize(ids_rows, 0);
    Dev dx(x.size() * 4), dsrc((size_t) rows * 4), ddst(ids_rows * 4), db(bounds.size() * 4), dw(w.size()),
        dxq(mmq::q8_bytes(rows, cols)), dy((size_t) rows * (size_t) out_rows * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(ddst.p, dst_table.data(), dst_table.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemset(dy.p, 0xff, (size_t) rows * (size_t) out_rows * 4), "sentinel");
    ck(cudaDeviceSynchronize(), "inputs");
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, dxq.p, (int) t, cols, cols, rows, s);
    mmq::Product p;
    p.w = dw.p; p.type = (int) t; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dxq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) ddst.p; p.total_rows = rows;
    p.max_rows = *std::max_element(counts.begin(), counts.end()); p.dst = (float*) dy.p; p.ld_dst = out_rows;
    ctx.run(p, s);
    ck(cudaGetLastError(), "launch");
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> got((size_t) rows * (size_t) out_rows);
    ck(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost), "y");

    const auto* tr = ggml_get_type_traits(t);
    std::vector<float> ref(got.size(), 0.0f), wd((size_t) out_rows * (size_t) cols);
    for (int e = 0; e < n; ++e) {
        const size_t rb = ggml_row_size(t, cols);
        for (int64_t o = 0; o < out_rows; ++o)
            tr->to_float(ws[(size_t) e].data() + (size_t) o * rb, wd.data() + (size_t) o * (size_t) cols, cols);
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            const float* xr = x.data() + (size_t) src[(size_t) r] * (size_t) cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double acc = 0;
                const float* wr = wd.data() + (size_t) o * (size_t) cols;
                for (int64_t k = 0; k < cols; ++k) acc += (double) wr[k] * xr[k];
                ref[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = (float) acc;
            }
        }
    }
    double err2 = 0, ref2 = 0, max_abs = 0, zero_max = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error(std::string(name) + ": unwritten or non-finite output");
        const double d = (double) got[i] - ref[i];
        err2 += d * d; ref2 += (double) ref[i] * ref[i]; max_abs = std::max(max_abs, std::fabs(d));
    }
    for (int r = 0; r < rows; ++r)
        if (src[(size_t) r] == rows - 1)   // the zero row
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max = std::max(zero_max, (double) std::fabs(got[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o]));
    const double rms = std::sqrt(ref2 / (double) ref.size()), rel = std::sqrt(err2 / (double) ref.size()) / rms;
    std::printf("%-22s rows %2d experts %d  ref_rms %.4g  rel_l2 %.5f  max/rms %.4f  zero row %.3g\n", name, rows, n, rms,
                rel, max_abs / rms, zero_max);
    if (rel > 0.04 || max_abs / rms > 0.35 || zero_max > std::max(1e-5, 1e-4 * rms))
        throw std::runtime_error(std::string(name) + ": outside the screen");
}

// ---- --bench: Q5_0 down, MMQ against dequantize-to-FP16 + cuBLAS, per expert group -------------------------------
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <class F>
double time_ms(F fn, cudaStream_t s, int reps, int rounds) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<double> r;
    for (int k = 0; k < rounds; ++k) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        r.push_back(ms / reps);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return median(r);
}

void bench(ggml_type t, int reps) {
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
    cublasSetStream(h, s);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    const bool have_mmq = mmq::supported((int) t);
    constexpr int64_t N = 2560, FF = 640;
    constexpr int NE = 16;   // experts a group holds (kGatherGroupMax)
    std::printf("bench %s down [%lld x %lld], %d experts a group, median of 5 rounds x %d calls\n", ggml_type_name(t),
                (long long) N, (long long) FF, NE, reps);
    std::printf("%-8s %12s %12s %12s %8s\n", "rows/exp", "MMQ ms", "f16+cuBLAS ms", "(dequant ms)", "ratio");
    const size_t eb = (size_t) N * ggml_row_size(t, FF);
    std::vector<uint8_t> w((size_t) NE * eb + 4096, 0);
    for (int e = 0; e < NE; ++e) {
        auto m = make_matrix(t, N, FF, e, 1);
        std::copy(m.begin(), m.end(), w.begin() + (ptrdiff_t) ((size_t) e * eb));
    }
    Dev dw(w.size());
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    mmq::Context ctx;
    for (int per : {16, 64, 256, 1024}) {
        const int rows = per * NE;
        const std::vector<float> x = make_x(rows, FF, 3);
        const size_t ids_rows = (size_t) mmq::mmq_ids_table_rows((uint64_t) rows, (uint64_t) per);   // (the identity over the whole table: mmq_ids_margin.hpp)
        std::vector<int32_t> bounds((size_t) NE + 1), iota(ids_rows);
        for (int e = 0; e <= NE; ++e) bounds[(size_t) e] = e * per;
        std::iota(iota.begin(), iota.end(), 0);
        Dev dx(x.size() * 4), db(bounds.size() * 4), di(iota.size() * 4), dxq(mmq::q8_bytes(rows, FF)),
            dy((size_t) rows * N * 4), dwh((size_t) N * FF * 2), dxh((size_t) rows * FF * 2);
        ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "b");
        ck(cudaMemcpy(di.p, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice), "i");
        std::vector<__half> xh(x.size());
        for (size_t i = 0; i < x.size(); ++i) xh[i] = __float2half(x[i]);
        ck(cudaMemcpy(dxh.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice), "xh");
        ck(cudaDeviceSynchronize(), "setup");
        double t_mmq = -1;
        if (have_mmq) {
            mmq::Product p;
            p.w = dw.p; p.type = (int) t; p.w_rows = N; p.w_cols = FF; p.expert_bytes = eb; p.n = NE;
            p.xq = dxq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p; p.total_rows = rows;
            p.max_rows = per; p.dst = (float*) dy.p; p.ld_dst = N;
            t_mmq = time_ms([&] {
                mmq::quantize((const float*) dx.p, nullptr, dxq.p, (int) t, FF, FF, rows, s);
                ctx.run(p, s);
            }, s, reps, 5);
        }
        const float one = 1.f, zero = 0.f;
        // the non-MMQ route: per expert, dequantize its matrix to FP16 and one HGEMM over its rows (what prefill.cpp does
        // for a type without an instance); the dequantize alone is timed too
        const double t_deq = time_ms([&] {
            for (int e = 0; e < NE; ++e)
                strata::kernels::dequant_f16((int) t, (const uint8_t*) dw.p + (size_t) e * eb, 0, N, FF, (uint16_t*) dwh.p, s);
        }, s, reps, 5);
        const double t_f16 = time_ms([&] {
            for (int e = 0; e < NE; ++e) {
                strata::kernels::dequant_f16((int) t, (const uint8_t*) dw.p + (size_t) e * eb, 0, N, FF, (uint16_t*) dwh.p, s);
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, per, (int) FF, &one, dwh.p, CUDA_R_16F, (int) FF,
                             (const __half*) dxh.p + (size_t) e * per * FF, CUDA_R_16F, (int) FF, &zero,
                             (float*) dy.p + (size_t) e * per * N, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            }
        }, s, reps, 5);
        if (have_mmq) std::printf("%-8d %12.4f %12.4f %12.4f %7.2fx\n", per, t_mmq, t_f16, t_deq, t_f16 / t_mmq);
        else std::printf("%-8d %12s %12.4f %12.4f %8s\n", per, "(no instance)", t_f16, t_deq, "-");
    }
    cublasDestroy(h);
    ck(cudaStreamDestroy(s), "destroy");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        bool bench_mode = false, require_q5 = false;
        int reps = 20;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--bench")) bench_mode = true;
            else if (!std::strcmp(argv[i], "--require-q5_0")) require_q5 = true;
            else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        }
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        int nd = 0;
        if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) { std::printf("no CUDA device\n"); return 77; }
        if (bench_mode) {
            bench(GGML_TYPE_Q5_0, reps);
            return 0;
        }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        int skipped = 0;
        {
            mmq::Context ctx;
            const std::vector<std::vector<int>> batches{{1, 3, 3}, {4, 1, 2}, {2, 3}, {17}};
            struct Case { const char* name; ggml_type t; int64_t out_rows, cols; int salt; };
            const Case cases[] = {
                {"IQ2_XXS gate/up", GGML_TYPE_IQ2_XXS, 1280, 2560, 0}, {"IQ2_XS gate/up", GGML_TYPE_IQ2_XS, 1280, 2560, 3},
                {"IQ2_S gate/up", GGML_TYPE_IQ2_S, 1280, 2560, 5},     {"IQ3_XXS gate/up", GGML_TYPE_IQ3_XXS, 1280, 2560, 7},
                {"IQ4_XS gate/up", GGML_TYPE_IQ4_XS, 1280, 2560, 11},  {"Q2_0 down", GGML_TYPE_Q2_0, 2560, 640, 13},
                {"IQ4_NL down", GGML_TYPE_IQ4_NL, 2560, 640, 17},      {"Q8_0 down", GGML_TYPE_Q8_0, 2560, 640, 19},
                {"Q5_0 down", GGML_TYPE_Q5_0, 2560, 640, 23},
            };
            int trial = 0;
            for (const auto& counts : batches) {
                const std::string tag = "-" + std::to_string(trial);
                for (const Case& c : cases) {
                    if (!mmq::supported((int) c.t)) {
                        if (trial == 0) {
                            std::printf("%-22s not in this build (the prompt path dequantizes to FP16)\n", c.name);
                            if (c.t == GGML_TYPE_Q5_0) { ++skipped; if (require_q5) throw std::runtime_error("Q5_0 MMQ required"); }
                            else throw std::runtime_error(std::string(c.name) + ": MMQ must cover this format");
                        }
                        continue;
                    }
                    product(ctx, s, (std::string(c.name) + tag).c_str(), c.t, c.out_rows, c.cols, counts, trial + c.salt);
                }
                ++trial;
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill MMQ v3.1 screen passed%s\n", skipped ? " (Q5_0 not built)" : "");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill MMQ v3.1 screen failed: %s\n", e.what());
        return 1;
    }
}
