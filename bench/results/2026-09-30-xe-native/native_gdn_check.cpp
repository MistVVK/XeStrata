// native_gdn_step against ggml-cpu's ggml_gated_delta_net (one token, scalar gate) and an FP64 reference.
// Strata keeps the state as [i][head][j] (j fastest) for S[i][j]; ggml keeps it per head transposed, [head][j][i].
// ggml-cpu takes the dot product after the decay, the CUDA kernel before it (then scales), so the two agree to
// rounding, not bitwise.
#include "probe_common.hpp"
#include "strata/kernels/native_gdn.hpp"

int main() {
    namespace k = strata::kernels;
    constexpr int S = 128;
    int bad = 0;
    for (auto [h_k, h_v] : {std::pair{16, 32}, std::pair{4, 4}, std::pair{2, 8}}) {
        for (int step = 0; step < 3; ++step) {
            const uint32_t seed = 100 * h_v + step;
            const auto q = probe::normal((size_t) S * h_k, seed + 1, 0.09f);
            const auto kv = probe::normal((size_t) S * h_k, seed + 2, 0.09f);
            const auto v = probe::normal((size_t) S * h_v, seed + 3);
            auto gate = probe::normal((size_t) h_v, seed + 4, 0.5f);
            for (auto& x : gate) x = -std::fabs(x);                        // log decay <= 0
            auto beta = probe::normal((size_t) h_v, seed + 5);
            for (auto& x : beta) x = 1.0f / (1.0f + std::exp(-x));          // in (0, 1)
            const auto st = probe::normal((size_t) S * S * h_v, seed + 6, 0.2f);   // Strata layout

            // ggml layout: state[head][j][i] = S[i][j]
            std::vector<float> st_g((size_t) S * S * h_v);
            for (int i = 0; i < S; ++i)
                for (int h = 0; h < h_v; ++h)
                    for (int j = 0; j < S; ++j)
                        st_g[((size_t) h * S + j) * S + i] = st[((size_t) i * h_v + h) * S + j];
            probe::Ggml g(256u << 20);
            ggml_tensor* out = ggml_gated_delta_net(g.ctx, g.f32({S, h_k, 1, 1}, q), g.f32({S, h_k, 1, 1}, kv),
                                                    g.f32({S, h_v, 1, 1}, v), g.f32({1, h_v, 1, 1}, gate),
                                                    g.f32({1, h_v, 1, 1}, beta), g.f32({S, S, h_v, 1}, st_g), 1);
            g.compute(out);
            const auto ref_o = probe::Ggml::read(out, 0, (size_t) S * h_v);
            const auto ref_sg = probe::Ggml::read(out, (size_t) S * h_v, (size_t) S * S * h_v);
            std::vector<float> ref_s((size_t) S * S * h_v);
            for (int i = 0; i < S; ++i)
                for (int h = 0; h < h_v; ++h)
                    for (int j = 0; j < S; ++j)
                        ref_s[((size_t) i * h_v + h) * S + j] = ref_sg[((size_t) h * S + j) * S + i];

            // FP64 reference in Strata's layout
            std::vector<float> f64_o((size_t) S * h_v);
            for (int h = 0; h < h_v; ++h) {
                const int qh = h % h_k;
                const double dec = std::exp((double) gate[h]);
                for (int j = 0; j < S; ++j) {
                    double kvd = 0;
                    for (int i = 0; i < S; ++i) kvd += dec * st[((size_t) i * h_v + h) * S + j] * kv[qh * S + i];
                    const double delta = ((double) v[h * S + j] - kvd) * beta[h];
                    double a = 0;
                    for (int i = 0; i < S; ++i)
                        a += (dec * st[((size_t) i * h_v + h) * S + j] + kv[qh * S + i] * delta) * q[qh * S + i];
                    f64_o[h * S + j] = (float) (a / std::sqrt((double) S));
                }
            }

            float* d_st = probe::upload(st);
            float* d_q = probe::upload(q);
            float* d_k = probe::upload(kv);
            float* d_v = probe::upload(v);
            float* d_g = probe::upload(gate);
            float* d_b = probe::upload(beta);
            float* d_o = static_cast<float*>(strata::parity::alloc_bytes((size_t) S * h_v * 4));
            k::GdnShapes sh;
            sh.S = S; sh.h_k = h_k; sh.h_v = h_v;
            k::native_gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, strata::parity::stream());
            strata::parity::sync();
            const auto got_o = probe::download(d_o, (size_t) S * h_v);
            const auto got_s = probe::download(d_st, (size_t) S * S * h_v);

            char name[96];
            std::snprintf(name, sizeof name, "h_k=%d h_v=%d #%d output vs ggml-cpu", h_k, h_v, step);
            const auto d1 = probe::compare(got_o, ref_o);
            probe::print(name, d1);
            std::snprintf(name, sizeof name, "h_k=%d h_v=%d #%d state vs ggml-cpu", h_k, h_v, step);
            const auto d2 = probe::compare(got_s, ref_s);
            probe::print(name, d2);
            const auto d3 = probe::compare(got_o, f64_o);
            const auto d4 = probe::compare(ref_o, f64_o);
            std::printf("  %-44s Xe %.3e, ggml-cpu %.3e (rel to scale)\n", "  output vs FP64", d3.max_rel, d4.max_rel);
            // both FP32 paths must sit at the same distance from FP64 (factor 4) and close to each other
            if (!(d1.max_rel <= 1e-5 && d2.max_rel <= 1e-5 && d3.max_rel <= std::fmax(4 * d4.max_rel, 1e-6))) ++bad;
        }
    }
    std::printf("native_gdn_check: %d failures (criterion: output and state within 1e-5 of ggml-cpu relative to "
                "scale; Xe no further from FP64 than 4x ggml-cpu)\n", bad);
    return bad ? 1 : 0;
}
