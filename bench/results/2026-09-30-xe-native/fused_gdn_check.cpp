// fused_gdn_{conv_l2, ab, step_norm} against FP64 references of the semantics in fused_gdn.hpp, and the first and
// last against ggml-cpu graphs of the same operations (ssm_conv + silu + l2_norm; gated_delta_net + rms_norm *
// gamma * sigmoid(z)).  fused_gdn_ab has no ggml-cpu comparison: ggml-cpu rounds the activation to BF16 for a BF16
// weight, the CUDA path (and this port) keeps it FP32.
//
// ggml's l2_norm is x / max(sqrt(sum), eps); the header's is x / sqrt(sum + eps), so the ggml comparison of the
// conv output carries that difference (about eps / (2 sum) relative).
#include "probe_common.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"

namespace {

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

int main() {
    namespace k = strata::kernels;
    constexpr int S = 128;
    const float eps = 1e-6f;
    int bad = 0;

    // ---- conv + SiLU + L2 over 2 q/k heads and 2 value heads (4 heads of 128 channels)
    for (int qk_heads : {2, 4}) {
        const int heads = 4, C = heads * S;
        const auto hist = probe::normal((size_t) C * 3, 71);
        const auto x = probe::normal((size_t) C, 72);
        const auto w = probe::normal((size_t) C * 4, 73, 0.5f);
        std::vector<float> ref((size_t) C), hist_ref((size_t) C * 3);
        std::vector<double> y64((size_t) C);
        for (int c = 0; c < C; ++c) {
            const double s = (double) hist[c * 3] * w[c * 4] + (double) hist[c * 3 + 1] * w[c * 4 + 1] +
                             (double) hist[c * 3 + 2] * w[c * 4 + 2] + (double) x[c] * w[c * 4 + 3];
            y64[c] = s * sigmoid(s);
            hist_ref[c * 3] = hist[c * 3 + 1];
            hist_ref[c * 3 + 1] = hist[c * 3 + 2];
            hist_ref[c * 3 + 2] = x[c];
        }
        for (int h = 0; h < heads; ++h) {
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += y64[h * S + i] * y64[h * S + i];
            const double sc = h < qk_heads ? 1.0 / std::sqrt(ss + eps) : 1.0;
            for (int i = 0; i < S; ++i) ref[h * S + i] = (float) (y64[h * S + i] * sc);
        }
        // ggml: sx [4, C] = (h0, h1, h2, x) per channel, c [4, C]
        std::vector<float> sx((size_t) C * 4);
        for (int c = 0; c < C; ++c) {
            sx[c * 4] = hist[c * 3]; sx[c * 4 + 1] = hist[c * 3 + 1]; sx[c * 4 + 2] = hist[c * 3 + 2]; sx[c * 4 + 3] = x[c];
        }
        probe::Ggml g(64u << 20);
        ggml_tensor* conv = ggml_silu(g.ctx, ggml_ssm_conv(g.ctx, g.f32({4, C, 1}, sx), g.f32({4, C}, w)));
        ggml_tensor* normed = ggml_l2_norm(g.ctx, ggml_reshape_2d(g.ctx, conv, S, heads), eps);
        g.compute(normed);
        g.compute(conv);
        auto gg = probe::Ggml::read(normed, 0, (size_t) C);
        const auto raw = probe::Ggml::read(conv, 0, (size_t) C);
        for (int i = qk_heads * S; i < C; ++i) gg[i] = raw[i];

        float* d_hist = probe::upload(hist);
        float* d_x = probe::upload(x);
        float* d_w = probe::upload(w);
        float* d_h = static_cast<float*>(strata::parity::alloc_bytes((size_t) C * 4));
        k::fused_gdn_conv_l2(d_hist, d_x, d_w, d_h, C, qk_heads, eps, strata::parity::stream());
        strata::parity::sync();
        const auto got = probe::download(d_h, (size_t) C);
        const auto got_hist = probe::download(d_hist, (size_t) C * 3);
        char name[96];
        std::snprintf(name, sizeof name, "conv_l2 qk_heads=%d vs FP64", qk_heads);
        const auto d1 = probe::compare(got, ref);
        probe::print(name, d1);
        std::snprintf(name, sizeof name, "conv_l2 qk_heads=%d vs ggml-cpu", qk_heads);
        const auto d2 = probe::compare(got, gg);
        probe::print(name, d2);
        const auto d3 = probe::compare(got_hist, hist_ref);
        probe::print("conv_l2 history slide", d3);
        if (!(d1.max_rel <= 1e-6 && d2.max_rel <= 1e-5 && d3.bit_diff == 0)) ++bad;
    }

    // ---- alpha/beta projections: n_embd 2560, h_v 32
    {
        const int n = 2560, h_v = 32;
        const auto x = probe::normal((size_t) n, 81);
        std::vector<uint16_t> wa((size_t) h_v * n), wb((size_t) h_v * n);
        const auto fa = probe::normal(wa.size(), 82, 0.02f), fb = probe::normal(wb.size(), 83, 0.02f);
        for (size_t i = 0; i < wa.size(); ++i) { wa[i] = k::bf16_from_f32(fa[i]); wb[i] = k::bf16_from_f32(fb[i]); }
        const auto dt = probe::normal((size_t) h_v, 84);
        auto ssm_a = probe::normal((size_t) h_v, 85);
        for (auto& a : ssm_a) a = -std::fabs(a);
        std::vector<float> ref_g((size_t) h_v), ref_b((size_t) h_v);
        for (int r = 0; r < h_v; ++r) {
            double a = 0, b = 0;
            for (int i = 0; i < n; ++i) {
                a += (double) k::f32_from_bf16(wa[(size_t) r * n + i]) * x[i];
                b += (double) k::f32_from_bf16(wb[(size_t) r * n + i]) * x[i];
            }
            const double v = a + dt[r];
            ref_g[r] = (float) ((v > 20.0 ? v : std::log1p(std::exp(v))) * ssm_a[r]);
            ref_b[r] = (float) sigmoid(b);
        }
        float* d_x = probe::upload(x);
        uint16_t* d_wa = probe::upload(wa);
        uint16_t* d_wb = probe::upload(wb);
        float* d_dt = probe::upload(dt);
        float* d_a = probe::upload(ssm_a);
        float* d_g = static_cast<float*>(strata::parity::alloc_bytes((size_t) h_v * 4));
        float* d_b = static_cast<float*>(strata::parity::alloc_bytes((size_t) h_v * 4));
        k::fused_gdn_ab(d_x, d_wa, d_wb, d_dt, d_a, d_g, d_b, n, h_v, strata::parity::stream());
        strata::parity::sync();
        const auto dg = probe::compare(probe::download(d_g, (size_t) h_v), ref_g);
        const auto db = probe::compare(probe::download(d_b, (size_t) h_v), ref_b);
        probe::print("ab gate vs FP64", dg);
        probe::print("ab beta vs FP64", db);
        if (!(dg.max_rel <= 1e-5 && db.max_rel <= 1e-5)) ++bad;
    }

    // ---- the step with its output norm
    for (auto [h_k, h_v] : {std::pair{16, 32}, std::pair{2, 8}}) {
        const auto q = probe::normal((size_t) S * h_k, 91, 0.09f);
        const auto kv = probe::normal((size_t) S * h_k, 92, 0.09f);
        const auto v = probe::normal((size_t) S * h_v, 93);
        auto gate = probe::normal((size_t) h_v, 94, 0.5f);
        for (auto& x : gate) x = -std::fabs(x);
        auto beta = probe::normal((size_t) h_v, 95);
        for (auto& x : beta) x = (float) sigmoid(x);
        const auto z = probe::normal((size_t) S * h_v, 96);
        const auto gamma = probe::normal((size_t) S, 97, 0.3f);
        const auto st = probe::normal((size_t) S * S * h_v, 98, 0.2f);

        std::vector<float> ref_y((size_t) S * h_v), ref_s((size_t) S * S * h_v);
        for (int h = 0; h < h_v; ++h) {
            const int qh = h % h_k;
            const double dec = std::exp((double) gate[h]);
            std::vector<double> o(S);
            for (int j = 0; j < S; ++j) {
                double kvd = 0;
                for (int i = 0; i < S; ++i) kvd += dec * st[((size_t) i * h_v + h) * S + j] * kv[qh * S + i];
                const double delta = ((double) v[h * S + j] - kvd) * beta[h];
                double a = 0;
                for (int i = 0; i < S; ++i) {
                    const double s = dec * st[((size_t) i * h_v + h) * S + j] + kv[qh * S + i] * delta;
                    ref_s[((size_t) i * h_v + h) * S + j] = (float) s;
                    a += s * q[qh * S + i];
                }
                o[j] = a / std::sqrt((double) S);
            }
            double ss = 0;
            for (double x : o) ss += x * x;
            const double sc = 1.0 / std::sqrt(ss / S + eps);
            for (int j = 0; j < S; ++j) ref_y[h * S + j] = (float) (o[j] * sc * gamma[j] * sigmoid(z[h * S + j]));
        }
        // ggml: gated_delta_net on the transposed state, then rms_norm * gamma * sigmoid(z)
        std::vector<float> st_g((size_t) S * S * h_v);
        for (int i = 0; i < S; ++i)
            for (int h = 0; h < h_v; ++h)
                for (int j = 0; j < S; ++j) st_g[((size_t) h * S + j) * S + i] = st[((size_t) i * h_v + h) * S + j];
        probe::Ggml g(256u << 20);
        ggml_tensor* dn = ggml_gated_delta_net(g.ctx, g.f32({S, h_k, 1, 1}, q), g.f32({S, h_k, 1, 1}, kv),
                                               g.f32({S, h_v, 1, 1}, v), g.f32({1, h_v, 1, 1}, gate),
                                               g.f32({1, h_v, 1, 1}, beta), g.f32({S, S, h_v, 1}, st_g), 1);
        ggml_tensor* o = ggml_view_2d(g.ctx, dn, S, h_v, S * sizeof(float), 0);
        ggml_tensor* yv = ggml_mul(g.ctx, ggml_mul(g.ctx, ggml_rms_norm(g.ctx, ggml_cont(g.ctx, o), eps), g.f32({S}, gamma)),
                                   ggml_sigmoid(g.ctx, g.f32({S, h_v}, z)));
        g.compute(yv);
        const auto gg = probe::Ggml::read(yv, 0, (size_t) S * h_v);

        float* d_st = probe::upload(st);
        float* d_q = probe::upload(q);
        float* d_k = probe::upload(kv);
        float* d_v = probe::upload(v);
        float* d_g = probe::upload(gate);
        float* d_b = probe::upload(beta);
        float* d_z = probe::upload(z);
        float* d_gm = probe::upload(gamma);
        float* d_y = static_cast<float*>(strata::parity::alloc_bytes((size_t) S * h_v * 4));
        k::fused_gdn_step_norm(d_st, d_q, d_k, d_v, d_g, d_b, d_z, d_gm, eps, d_y, h_k, h_v, strata::parity::stream());
        strata::parity::sync();
        const auto got_y = probe::download(d_y, (size_t) S * h_v);
        const auto got_s = probe::download(d_st, (size_t) S * S * h_v);
        char name[96];
        std::snprintf(name, sizeof name, "step_norm h_k=%d h_v=%d y vs FP64", h_k, h_v);
        const auto d1 = probe::compare(got_y, ref_y);
        probe::print(name, d1);
        std::snprintf(name, sizeof name, "step_norm h_k=%d h_v=%d y vs ggml-cpu", h_k, h_v);
        const auto d2 = probe::compare(got_y, gg);
        probe::print(name, d2);
        std::snprintf(name, sizeof name, "step_norm h_k=%d h_v=%d state vs FP64", h_k, h_v);
        const auto d3 = probe::compare(got_s, ref_s);
        probe::print(name, d3);
        if (!(d1.max_rel <= 1e-5 && d2.max_rel <= 1e-5 && d3.max_rel <= 1e-6)) ++bad;
    }
    std::printf("fused_gdn_check: %d failures (criteria: conv_l2 within 1e-6 of FP64 and 1e-5 of ggml-cpu, history "
                "exact; ab within 1e-5 of FP64; step_norm y within 1e-5 of FP64 and ggml-cpu, state within 1e-6 of "
                "FP64; all relative to the output scale)\n", bad);
    return bad ? 1 : 0;
}
