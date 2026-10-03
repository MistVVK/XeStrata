// native_gdn_{conv_silu, l2_norm, beta_gate, gate, out_norm} against ggml-cpu graphs of the operators the CUDA source
// names: ssm_conv then silu; rms_norm(eps / 128) then scale(1 / sqrt(128)); sigmoid; softplus(alpha + dt) * ssm_a;
// rms_norm(eps) * gamma * sigmoid(z).  ggml-cpu sums the norms' squares in double and may contract the convolution,
// so the norms and the convolution are compared to a tolerance; the elementwise epilogues are compared bitwise and
// reported, but accepted within a tolerance because exp and log are not correctly rounded on either side.
#include "probe_common.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"

int main() {
    namespace k = strata::kernels;
    constexpr int S = 128;
    const float eps = 1e-6f;
    int bad = 0;
    void* st = strata::parity::stream();
    auto check = [&](const char* name, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
        const auto d = probe::compare(got, ref);
        probe::print(name, d);
        if (!(d.max_rel <= tol)) ++bad;
    };

    {   // conv + SiLU over 8 heads of channels
        const int C = 8 * S;
        const auto hist = probe::normal((size_t) C * 3, 1), x = probe::normal((size_t) C, 2);
        const auto w = probe::normal((size_t) C * 4, 3, 0.5f);
        std::vector<float> sx((size_t) C * 4), hist_ref((size_t) C * 3);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < 3; ++t) sx[c * 4 + t] = hist[c * 3 + t];
            sx[c * 4 + 3] = x[c];
            for (int t = 0; t < 3; ++t) hist_ref[c * 3 + t] = sx[c * 4 + t + 1];
        }
        probe::Ggml g(64u << 20);
        ggml_tensor* raw = ggml_ssm_conv(g.ctx, g.f32({4, C, 1}, sx), g.f32({4, C}, w));
        ggml_tensor* silu = ggml_silu(g.ctx, raw);
        g.compute(silu);
        float* d_hist = probe::upload(hist);
        float* d_x = probe::upload(x);
        float* d_w = probe::upload(w);
        float* d_raw = static_cast<float*>(strata::parity::alloc_bytes((size_t) C * 4));
        float* d_silu = static_cast<float*>(strata::parity::alloc_bytes((size_t) C * 4));
        k::native_gdn_conv_silu(d_hist, d_x, d_w, d_raw, d_silu, C, 4, st);
        strata::parity::sync();
        check("conv raw vs ggml-cpu", probe::download(d_raw, (size_t) C), probe::Ggml::read(raw, 0, (size_t) C), 1e-6);
        check("conv silu vs ggml-cpu", probe::download(d_silu, (size_t) C), probe::Ggml::read(silu, 0, (size_t) C), 1e-6);
        check("conv history slide (exact)", probe::download(d_hist, (size_t) C * 3), hist_ref, 0.0);
    }
    {   // L2 norm as RMS norm + scale, 32 rows
        const int rows = 32;
        const auto x = probe::normal((size_t) rows * S, 4, 0.7f);
        probe::Ggml g(64u << 20);
        ggml_tensor* y = ggml_scale(g.ctx, ggml_rms_norm(g.ctx, g.f32({S, rows}, x), eps / S), 1.0f / std::sqrt((float) S));
        g.compute(y);
        float* d = probe::upload(x);
        k::native_gdn_l2_norm(d, rows, S, eps, st);
        strata::parity::sync();
        check("l2_norm vs ggml-cpu rms_norm + scale", probe::download(d, (size_t) rows * S),
              probe::Ggml::read(y, 0, (size_t) rows * S), 1e-6);
    }
    {   // beta and gate epilogues, 32 heads
        const int H = 32;
        const auto b = probe::normal((size_t) H, 5, 3.0f), a = probe::normal((size_t) H, 6, 8.0f);
        const auto dt = probe::normal((size_t) H, 7, 8.0f);
        auto ssm_a = probe::normal((size_t) H, 8);
        for (auto& v : ssm_a) v = -std::fabs(v);
        probe::Ggml g(16u << 20);
        ggml_tensor* sb = ggml_sigmoid(g.ctx, g.f32({H}, b));
        ggml_tensor* gt = ggml_mul(g.ctx, ggml_softplus(g.ctx, ggml_add(g.ctx, g.f32({H}, a), g.f32({H}, dt))),
                                   g.f32({H}, ssm_a));
        g.compute(sb);
        g.compute(gt);
        float* d_b = probe::upload(b);
        float* d_a = probe::upload(a);
        float* d_dt = probe::upload(dt);
        float* d_sa = probe::upload(ssm_a);
        float* d_g = static_cast<float*>(strata::parity::alloc_bytes((size_t) H * 4));
        k::native_gdn_beta_gate(d_b, H, st);
        k::native_gdn_gate(d_a, d_dt, d_sa, d_g, H, st);
        strata::parity::sync();
        check("beta sigmoid vs ggml-cpu", probe::download(d_b, (size_t) H), probe::Ggml::read(sb, 0, (size_t) H), 1e-6);
        check("gate softplus * ssm_a vs ggml-cpu", probe::download(d_g, (size_t) H), probe::Ggml::read(gt, 0, (size_t) H), 1e-6);
    }
    {   // output norm, 32 heads
        const int H = 32;
        const auto o = probe::normal((size_t) H * S, 9, 0.05f), z = probe::normal((size_t) H * S, 10);
        const auto gamma = probe::normal((size_t) S, 11, 0.3f);
        probe::Ggml g(64u << 20);
        ggml_tensor* y = ggml_mul(g.ctx, ggml_mul(g.ctx, ggml_rms_norm(g.ctx, g.f32({S, H}, o), eps), g.f32({S}, gamma)),
                                  ggml_sigmoid(g.ctx, g.f32({S, H}, z)));
        g.compute(y);
        float* d_o = probe::upload(o);
        float* d_z = probe::upload(z);
        float* d_gm = probe::upload(gamma);
        float* d_y = static_cast<float*>(strata::parity::alloc_bytes((size_t) H * S * 4));
        k::native_gdn_out_norm(d_o, d_z, d_gm, d_y, H, S, eps, st);
        strata::parity::sync();
        check("out_norm vs ggml-cpu", probe::download(d_y, (size_t) H * S), probe::Ggml::read(y, 0, (size_t) H * S), 1e-6);
    }
    std::printf("native_gdn_preprocess_check: %d failures (criterion: within 1e-6 of ggml-cpu relative to the output "
                "scale; history exact)\n", bad);
    return bad ? 1 : 0;
}
