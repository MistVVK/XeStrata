// native_ple_postops / _batch against an FP64 reference of the arithmetic the CUDA source documents, and against each
// other: a batch of T tokens must equal T single-token calls with the history appended in between, bitwise.
// The llama.cpp model graph these kernels reproduce is not in the pinned checkout (it carries only ggml), so no
// ggml-cpu graph is built here.
#include "probe_common.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"

namespace {

constexpr int N = 2560, H = 4, D = N * H, HIST = 9;

struct HostWeights {
    std::vector<float> nk, nq, nc;
    std::vector<uint16_t> conv;
};

// one token in FP64: returns result (hidden + gated + silu(conv)) and the normalized row to append
void reference(const std::vector<float>& key, const std::vector<float>& hidden, const std::vector<float>& value,
               const std::vector<float>& history, const HostWeights& w, std::vector<float>& result,
               std::vector<float>& normalized) {
    auto rms = [&](const std::vector<double>& x, const std::vector<float>& g) {
        std::vector<double> y(D);
        for (int h = 0; h < H; ++h) {
            double ss = 0;
            for (int i = 0; i < N; ++i) ss += x[h * N + i] * x[h * N + i];
            const double sc = 1.0 / std::sqrt(ss / N + strata::kernels::NG_RMS_EPS);
            for (int i = 0; i < N; ++i) y[h * N + i] = x[h * N + i] * sc * g[h * N + i];
        }
        return y;
    };
    const auto kn = rms(std::vector<double>(key.begin(), key.end()), w.nk);
    const auto qn = rms(std::vector<double>(hidden.begin(), hidden.end()), w.nq);
    std::vector<double> gated(D);
    for (int h = 0; h < H; ++h) {
        double s = 0;
        for (int i = 0; i < N; ++i) s += kn[h * N + i] * qn[h * N + i];
        s /= std::sqrt((double) N);
        const double mag = std::sqrt(std::fmax(std::fabs(s), 1e-6));
        const double g = 1.0 / (1.0 + std::exp(-((s > 0) - (s < 0)) * mag));
        for (int i = 0; i < N; ++i) gated[h * N + i] = value[i] * g;
    }
    const auto nm = rms(gated, w.nc);
    result.assign(D, 0.0f);
    normalized.assign(D, 0.0f);
    for (int c = 0; c < D; ++c) {
        double sum = 0;
        for (int k = 0; k < 4; ++k) {
            const double x = k == 3 ? nm[c] : history[c * HIST + 3 * k];
            sum += x * strata::kernels::f32_from_f16(w.conv[c * 4 + k]);
        }
        const double act = sum / (1.0 + std::exp(-sum));
        result[c] = (float) (hidden[c] + gated[c] + act);
        normalized[c] = (float) nm[c];
    }
}

}  // namespace

int main() {
    namespace k = strata::kernels;
    int bad = 0;
    void* st = strata::parity::stream();
    HostWeights hw;
    hw.nk = probe::normal(D, 1, 0.3f);
    hw.nq = probe::normal(D, 2, 0.3f);
    hw.nc = probe::normal(D, 3, 0.3f);
    const auto cf = probe::normal((size_t) D * 4, 4, 0.5f);
    hw.conv.resize(cf.size());
    for (size_t i = 0; i < cf.size(); ++i) hw.conv[i] = k::f16_from_f32(cf[i]);
    k::PleWeights w;
    w.norm_key = probe::upload(hw.nk);
    w.norm_query = probe::upload(hw.nq);
    w.norm_conv = probe::upload(hw.nc);
    w.conv1d_f16 = probe::upload(hw.conv);

    const int T = 12;   // more than the nine-token history, so the batch reads both history and earlier rows
    std::vector<std::vector<float>> keys, hiddens, values;
    for (int t = 0; t < T; ++t) {
        keys.push_back(probe::normal(D, 100 + t));
        hiddens.push_back(probe::normal(D, 200 + t));
        values.push_back(probe::normal(N, 300 + t));
    }
    const auto history0 = probe::normal((size_t) D * HIST, 5, 0.5f);

    // ---- single-token calls, the history appended between them
    std::vector<float> history = history0, single_results, single_history;
    float* d_key = static_cast<float*>(strata::parity::alloc_bytes(D * 4));
    float* d_hidden = static_cast<float*>(strata::parity::alloc_bytes(D * 4));
    float* d_value = static_cast<float*>(strata::parity::alloc_bytes(N * 4));
    float* d_hist = static_cast<float*>(strata::parity::alloc_bytes(D * HIST * 4));
    k::NativePlePostopsBuffers b;
    for (float** p : {&b.key, &b.query, &b.gated, &b.normalized, &b.conv, &b.result})
        *p = static_cast<float*>(strata::parity::alloc_bytes(D * 4));
    b.gate = static_cast<float*>(strata::parity::alloc_bytes(H * 4));
    double worst_ref = 0, ref_scale = 0, worst_norm = 0;
    for (int t = 0; t < T; ++t) {
        strata::parity::copy_bytes_in(d_key, keys[t].data(), D * 4);
        strata::parity::copy_bytes_in(d_hidden, hiddens[t].data(), D * 4);
        strata::parity::copy_bytes_in(d_value, values[t].data(), N * 4);
        strata::parity::copy_bytes_in(d_hist, history.data(), D * HIST * 4);
        k::native_ple_postops(d_key, d_hidden, d_value, d_hist, w, b, st);
        strata::parity::sync();
        const auto res = probe::download(b.result, D);
        const auto nm = probe::download(b.normalized, D);
        std::vector<float> ref_res, ref_nm;
        reference(keys[t], hiddens[t], values[t], history, hw, ref_res, ref_nm);
        const auto d1 = probe::compare(res, ref_res);
        const auto d2 = probe::compare(nm, ref_nm);
        worst_ref = std::fmax(worst_ref, d1.max_rel);
        ref_scale = std::fmax(ref_scale, d1.ref_scale);
        worst_norm = std::fmax(worst_norm, d2.max_rel);
        single_results.insert(single_results.end(), res.begin(), res.end());
        for (int c = 0; c < D; ++c) {   // append: drop the oldest row, the newest last
            for (int r = 0; r < HIST - 1; ++r) history[c * HIST + r] = history[c * HIST + r + 1];
            history[c * HIST + HIST - 1] = nm[c];
        }
    }
    single_history = history;
    std::printf("  %-44s result %.3e, normalized %.3e (rel to scale; scale %.3e)\n", "single vs FP64, worst of 12 tokens",
                worst_ref, worst_norm, ref_scale);
    if (!(worst_ref <= 1e-5 && worst_norm <= 1e-5)) ++bad;

    // ---- the batch over the same tokens
    std::vector<float> all_keys, all_hidden, all_values;
    for (int t = 0; t < T; ++t) {
        all_keys.insert(all_keys.end(), keys[t].begin(), keys[t].end());
        all_hidden.insert(all_hidden.end(), hiddens[t].begin(), hiddens[t].end());
        all_values.insert(all_values.end(), values[t].begin(), values[t].end());
    }
    for (int batch : {T, 1}) {
        std::vector<float> hist = history0, results;
        float* dk = static_cast<float*>(strata::parity::alloc_bytes((size_t) batch * D * 4));
        float* dh = static_cast<float*>(strata::parity::alloc_bytes((size_t) batch * D * 4));
        float* dv = static_cast<float*>(strata::parity::alloc_bytes((size_t) batch * N * 4));
        float* dq = static_cast<float*>(strata::parity::alloc_bytes((size_t) batch * D * 4));
        float* dg = static_cast<float*>(strata::parity::alloc_bytes((size_t) batch * D * 4));
        float* dgate = static_cast<float*>(strata::parity::alloc_bytes((size_t) batch * H * 4));
        strata::parity::copy_bytes_in(d_hist, hist.data(), D * HIST * 4);
        for (int t0 = 0; t0 < T; t0 += batch) {
            strata::parity::copy_bytes_in(dk, all_keys.data() + (size_t) t0 * D, (size_t) batch * D * 4);
            strata::parity::copy_bytes_in(dh, all_hidden.data() + (size_t) t0 * D, (size_t) batch * D * 4);
            strata::parity::copy_bytes_in(dv, all_values.data() + (size_t) t0 * N, (size_t) batch * N * 4);
            k::native_ple_postops_batch(dk, dh, dv, d_hist, w, dq, dg, dgate, batch, st);
            strata::parity::sync();
            const auto r = probe::download(dh, (size_t) batch * D);
            results.insert(results.end(), r.begin(), r.end());
        }
        const auto hist_after = probe::download(d_hist, (size_t) D * HIST);
        char name[96];
        std::snprintf(name, sizeof name, "batch of %d vs single calls: result", batch);
        const auto d1 = probe::compare(results, single_results);
        probe::print(name, d1);
        std::snprintf(name, sizeof name, "batch of %d vs single calls: history", batch);
        const auto d2 = probe::compare(hist_after, single_history);
        probe::print(name, d2);
        if (d1.bit_diff || d2.bit_diff) ++bad;
    }
    std::printf("native_ple_postops_check: %d failures (criteria: single-token result and normalized row within 1e-5 of "
                "FP64 relative to scale; batches of 12 and 1 bitwise equal to 12 single-token calls, history included)\n",
                bad);
    return bad ? 1 : 0;
}
