// native_moe_combine / _multi against ggml-cpu computing the graph ggml-cuda fuses (moe-weighted-reduction):
// mul(experts, weights) broadcast over the rows, the expert rows added in order, then the shared expert output.
#include "probe_common.hpp"
#include "strata/kernels/native_moe.hpp"

int main() {
    namespace k = strata::kernels;
    const int64_t n_embd = 2560;
    int bad = 0;
    for (int kk : {1, 2, 10, 15}) {
        for (int n_tok : {1, 4}) {
            for (bool with_shared : {false, true}) {
                const auto parts = probe::normal((size_t) n_embd * kk * n_tok, 11 + kk);
                const auto weights = probe::normal((size_t) kk * n_tok, 23 + kk, 0.3f);
                const auto shared = probe::normal((size_t) n_embd * n_tok, 37);

                probe::Ggml g(64u << 20);
                ggml_tensor* ex = g.f32({n_embd, kk, n_tok}, parts);
                ggml_tensor* w = g.f32({1, kk, n_tok}, weights);
                ggml_tensor* weighted = ggml_mul(g.ctx, ex, w);
                ggml_tensor* sum = nullptr;
                for (int e = 0; e < kk; ++e) {
                    ggml_tensor* row = ggml_view_2d(g.ctx, weighted, n_embd, n_tok, weighted->nb[2],
                                                    (size_t) e * weighted->nb[1]);
                    sum = sum ? ggml_add(g.ctx, sum, row) : row;
                }
                if (with_shared) sum = ggml_add(g.ctx, sum, g.f32({n_embd, n_tok}, shared));
                ggml_tensor* out = ggml_cont(g.ctx, sum);
                g.compute(out);
                const auto ref = probe::Ggml::read(out, 0, (size_t) n_embd * n_tok);

                float* d_parts = probe::upload(parts);
                float* d_w = probe::upload(weights);
                float* d_sh = with_shared ? probe::upload(shared) : nullptr;
                float* d_out = static_cast<float*>(strata::parity::alloc_bytes((size_t) n_embd * n_tok * 4));
                if (n_tok == 1) k::native_moe_combine(d_parts, d_w, d_sh, d_out, n_embd, kk, strata::parity::stream());
                else k::native_moe_combine_multi(d_parts, d_w, d_sh, d_out, n_embd, kk, n_tok, strata::parity::stream());
                strata::parity::sync();
                const auto got = probe::download(d_out, (size_t) n_embd * n_tok);
                char name[96];
                std::snprintf(name, sizeof name, "k=%d tokens=%d shared=%d vs ggml-cpu", kk, n_tok, (int) with_shared);
                const auto d = probe::compare(got, ref);
                probe::print(name, d);
                if (d.bit_diff) ++bad;
            }
        }
    }
    std::printf("native_moe_check: %d failures (criterion: bitwise equal to ggml-cpu)\n", bad);
    return bad ? 1 : 0;
}
