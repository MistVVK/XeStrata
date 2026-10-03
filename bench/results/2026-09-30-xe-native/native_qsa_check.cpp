// The native QSA kernels:
//   native_qsa_rms_norm_weighted, native_qsa_gate_apply  against ggml-cpu rms_norm * gamma and attn * sigmoid(gate half)
//   native_qsa_score  against ggml-cpu mul_mat -> relu -> the four heads added in order -> + bias -> + mask, and FP64
//   native_qsa_indexer_append / _append_batch  against an FP64 reference of the pooled key, and against each other:
//     a batch (split into chunks of several sizes) must equal the single-cell appends bitwise
#include "probe_common.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/qsa.hpp"

namespace k = strata::kernels;

namespace {

int bad = 0;
void check(const char* name, const std::vector<float>& got, const std::vector<float>& ref, double tol, bool bitwise = false) {
    const auto d = probe::compare(got, ref);
    probe::print(name, d);
    if (bitwise ? d.bit_diff != 0 : !(d.max_rel <= tol)) ++bad;
}

void norm_and_gate() {
    void* st = strata::parity::stream();
    for (int cols : {256, 128, 2560}) {
        const int rows = 24;
        const auto x = probe::normal((size_t) rows * cols, cols), gamma = probe::normal((size_t) cols, cols + 1, 0.3f);
        probe::Ggml g(64u << 20);
        ggml_tensor* y = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, g.f32({cols, rows}, x), 1e-6f), g.f32({cols}, gamma));
        g.compute(y);
        float* d_x = probe::upload(x);
        float* d_g = probe::upload(gamma);
        float* d_y = static_cast<float*>(strata::parity::alloc_bytes((size_t) rows * cols * 4));
        k::native_qsa_rms_norm_weighted(d_x, d_g, d_y, cols, rows, 1e-6f, st);
        k::native_qsa_rms_norm_weighted(d_x, d_g, d_x, cols, rows, 1e-6f, st);   // in place
        strata::parity::sync();
        char name[96];
        std::snprintf(name, sizeof name, "rms_norm_weighted cols=%d vs ggml-cpu", cols);
        const auto out = probe::download(d_y, (size_t) rows * cols);
        check(name, out, probe::Ggml::read(y, 0, (size_t) rows * cols), 1e-6);
        std::snprintf(name, sizeof name, "rms_norm_weighted cols=%d in place vs out of place", cols);
        check(name, probe::download(d_x, (size_t) rows * cols), out, 0, true);
    }
    const int n_head = 24, hd = 256;
    const auto attn = probe::normal((size_t) n_head * hd, 7), qf = probe::normal((size_t) n_head * hd * 2, 8, 2.0f);
    probe::Ggml g(64u << 20);
    ggml_tensor* q = g.f32({hd * 2, n_head}, qf);
    ggml_tensor* gate = ggml_view_2d(g.ctx, q, hd, n_head, q->nb[1], hd * sizeof(float));
    ggml_tensor* y = ggml_mul(g.ctx, g.f32({hd, n_head}, attn), ggml_sigmoid(g.ctx, ggml_cont(g.ctx, gate)));
    g.compute(y);
    float* d_a = probe::upload(attn);
    float* d_q = probe::upload(qf);
    float* d_y = static_cast<float*>(strata::parity::alloc_bytes((size_t) n_head * hd * 4));
    k::native_qsa_gate_apply(d_a, d_q, d_y, n_head, hd, strata::parity::stream());
    strata::parity::sync();
    check("gate_apply vs ggml-cpu", probe::download(d_y, (size_t) n_head * hd), probe::Ggml::read(y, 0, (size_t) n_head * hd), 1e-6);
}

void score() {
    const auto s = k::qsa_real_shapes();
    constexpr int D = 128, HEADS = 4, R = 4;
    for (int n : {1, 7, 400, 2051, 9001}) {
        for (bool with_bias : {false, true}) {
            const int64_t max_cells = 9004, max_blocks = max_cells / R + 1, full = n / R;
            const auto pooled = probe::normal((size_t) max_blocks * D, 20 + n, 0.3f);
            const auto query = probe::normal((size_t) HEADS * D, 21 + n, 0.3f);
            const auto bias = probe::normal((size_t) max_blocks, 22 + n);
            std::vector<int32_t> step(k::kStepCount);
            k::qsa_step_fill(step.data(), n - 1, s);
            // references over rows 0..full
            std::vector<float> ref64((size_t) n);
            for (int64_t row = 0; row <= full; ++row) {
                double sum = 0;
                for (int j = 0; j < HEADS; ++j) {
                    double a = 0;
                    for (int d = 0; d < D; ++d) a += (double) pooled[row * D + d] * query[j * D + d];
                    sum += std::fmax(a, 0.0);
                }
                if (with_bias) sum += bias[row];
                if (row == full && n % R) sum += 1e9;
                for (int64_t i = row * R; i < n && i < (row + 1) * R; ++i) ref64[i] = (float) sum;
            }
            probe::Ggml g(64u << 20);
            std::vector<float> rows_pooled(pooled.begin(), pooled.begin() + (full + 1) * D);
            ggml_tensor* sc = ggml_relu(g.ctx, ggml_mul_mat(g.ctx, g.f32({D, full + 1}, rows_pooled), g.f32({D, HEADS}, query)));
            // sc is [full + 1, HEADS]: add the heads in order
            ggml_tensor* sum = nullptr;
            for (int j = 0; j < HEADS; ++j) {
                ggml_tensor* col = ggml_view_1d(g.ctx, sc, full + 1, (size_t) j * sc->nb[1]);
                sum = sum ? ggml_add(g.ctx, sum, col) : ggml_cont(g.ctx, col);
            }
            if (with_bias) sum = ggml_add(g.ctx, sum, g.f32({full + 1}, std::vector<float>(bias.begin(), bias.begin() + full + 1)));
            std::vector<float> mask((size_t) full + 1, 0.0f);
            if (n % R) mask[full] = 1e9f;
            sum = ggml_add(g.ctx, sum, g.f32({full + 1}, mask));
            g.compute(sum);
            const auto per_row = probe::Ggml::read(sum, 0, (size_t) full + 1);
            std::vector<float> ref_g((size_t) n);
            for (int64_t i = 0; i < n; ++i) ref_g[i] = per_row[i / R];

            float* d_p = probe::upload(pooled);
            float* d_q = probe::upload(query);
            float* d_b = with_bias ? probe::upload(bias) : nullptr;
            int32_t* d_s = probe::upload(step);
            float* d_c = static_cast<float*>(strata::parity::alloc_bytes((size_t) max_cells * 4));
            k::native_qsa_score(d_p, d_q, d_b, s, d_s, max_blocks, max_cells, d_c, strata::parity::stream());
            strata::parity::sync();
            const auto got = probe::download(d_c, (size_t) n);
            // scores carry the 1e9 mask; compare the unmasked cells relative to their own scale
            auto strip = [&](std::vector<float> v) {
                if (n % R) for (int64_t i = full * R; i < n; ++i) v[i] = 0.0f;
                return v;
            };
            char name[96];
            std::snprintf(name, sizeof name, "score n=%d bias=%d vs ggml-cpu (unmasked)", n, (int) with_bias);
            check(name, strip(got), strip(ref_g), 1e-5);
            std::snprintf(name, sizeof name, "score n=%d bias=%d vs FP64 (unmasked)", n, (int) with_bias);
            check(name, strip(got), strip(ref64), 1e-5);
            if (n % R) {
                bool masked = true;
                for (int64_t i = full * R; i < n; ++i) masked &= got[i] > 1e8f;
                std::printf("  %-44s %s\n", "  partial block carries the +1e9 mask", masked ? "yes" : "*** NO ***");
                if (!masked) ++bad;
            }
        }
    }
}

void indexer() {
    const auto s = k::qsa_real_shapes();
    constexpr int D = 128, R = 4, ROT = 64;
    const int64_t max_cells = 64, cells = 37;
    const float eps = 1e-6f, freq = 1e7f;
    const int32_t pos_base = 8;
    const auto raw = probe::normal((size_t) cells * D, 30);
    const auto gamma = probe::normal((size_t) D, 31, 0.3f);
    float* d_raw = probe::upload(raw);
    float* d_gamma = probe::upload(gamma);
    auto alloc = [&](k::QsaIndexerBuffers& b) {
        b.tail = static_cast<float*>(strata::parity::alloc_bytes((R - 1) * D * 4));
        b.dead = static_cast<float*>(strata::parity::alloc_bytes(D * 4));
        b.pooled = static_cast<float*>(strata::parity::alloc_bytes((size_t) (max_cells / R + 1) * D * 4));
        b.block_pos = static_cast<int32_t*>(strata::parity::alloc_bytes(4));
        std::vector<float> z((size_t) (max_cells / R + 1) * D, 0.0f);
        strata::parity::copy_bytes_in(b.pooled, z.data(), z.size() * 4);
    };
    auto read = [&](const k::QsaIndexerBuffers& b) {
        auto v = probe::download(b.pooled, (size_t) (max_cells / R + 1) * D);
        const auto t = probe::download(b.tail, (size_t) (R - 1) * D);
        v.insert(v.end(), t.begin(), t.end());
        v.push_back((float) probe::download(b.block_pos, 1)[0]);
        return v;
    };
    // single-cell appends
    k::QsaIndexerBuffers bs;
    alloc(bs);
    int32_t* d_pos = static_cast<int32_t*>(strata::parity::alloc_bytes(4));
    for (int32_t p = 0; p < cells; ++p) {
        strata::parity::copy_bytes_in(d_pos, &p, 4);
        k::native_qsa_indexer_append(d_raw + (size_t) p * D, d_pos, pos_base, d_gamma, eps, bs, s, max_cells, freq,
                                     strata::parity::stream());
    }
    strata::parity::sync();
    const auto single = read(bs);

    // FP64 reference of the pooled keys: F16-rounded keys, block mean, RMS norm, gamma, NEOX rotation at the block's
    // first cell (pos_base + 4b); pooled[0] is overwritten by block 0, the spare follows the last completed block
    std::vector<float> ref(single.begin(), single.begin() + (size_t) (max_cells / R + 1) * D);
    const double theta_scale = std::pow((double) freq, -2.0 / ROT);
    auto pool = [&](const std::vector<double>& mean, int rope_pos, float* out) {
        double ss = 0;
        for (double m : mean) ss += m * m;
        const double sc = 1.0 / std::sqrt(ss / D + eps);
        std::vector<double> v(D);
        for (int d = 0; d < D; ++d) v[d] = mean[d] * sc * gamma[d];
        for (int d = 0; d < D; ++d) {
            double y = v[d];
            if (d < ROT) {
                const int pair = d % (ROT / 2);
                const double th = rope_pos * std::pow(theta_scale, pair);
                const double a = v[pair], z = v[pair + ROT / 2];
                y = d < ROT / 2 ? a * std::cos(th) - z * std::sin(th) : a * std::sin(th) + z * std::cos(th);
            }
            out[d] = (float) y;
        }
    };
    auto f16 = [](float x) { return (double) k::f32_from_f16(k::f16_from_f32(x)); };
    std::vector<float> dead(D);
    {
        std::vector<double> m(D);
        for (int d = 0; d < D; ++d) m[d] = f16(raw[d]);
        pool(m, 0, dead.data());
    }
    const int64_t nb = cells / R;
    for (int64_t b = 0; b < nb; ++b) {
        std::vector<double> m(D);
        for (int d = 0; d < D; ++d) {
            double sum = 0;
            for (int j = 0; j < R; ++j) sum += f16(raw[(b * R + j) * D + d]);
            m[d] = sum / R;
        }
        pool(m, pos_base + R * (int) b, ref.data() + b * D);
    }
    for (int d = 0; d < D; ++d) ref[nb * D + d] = dead[d];
    const std::vector<float> got_pooled(single.begin(), single.begin() + (size_t) (nb + 1) * D);
    const std::vector<float> ref_pooled(ref.begin(), ref.begin() + (size_t) (nb + 1) * D);
    check("indexer single appends: pooled vs FP64", got_pooled, ref_pooled, 1e-5);

    // batches in chunks of several sizes against the single appends
    for (int chunk : {37, 1, 3, 4, 5, 16}) {
        k::QsaIndexerBuffers bb;
        alloc(bb);
        for (int64_t p0 = 0; p0 < cells; p0 += chunk) {
            const int64_t n = std::min<int64_t>(chunk, cells - p0);
            k::native_qsa_indexer_append_batch(d_raw + (size_t) p0 * D, n, p0, pos_base, d_gamma, eps, bb, s, max_cells,
                                               freq, strata::parity::stream());
        }
        strata::parity::sync();
        char name[96];
        std::snprintf(name, sizeof name, "indexer batch in chunks of %d vs single", chunk);
        check(name, read(bb), single, 0, true);
    }
}

}  // namespace

int main() {
    norm_and_gate();
    score();
    indexer();
    std::printf("native_qsa_check: %d failures (criteria: norm and gate within 1e-6 of ggml-cpu, in-place norm bitwise; "
                "unmasked scores within 1e-5 of ggml-cpu and FP64 with the partial-block mask present; pooled keys "
                "within 1e-5 of FP64; every batch chunking bitwise equal to single appends; all relative to scale)\n",
                bad);
    return bad ? 1 : 0;
}
