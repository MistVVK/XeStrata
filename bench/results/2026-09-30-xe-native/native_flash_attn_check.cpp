// native_flash_attn_short_step against an FP64 softmax attention and ggml-cpu's ggml_flash_attn_ext.
// ggml-cpu rounds Q to F16 (K's vec_dot_type) before the dot products, the CUDA kernel and this port keep it FP32,
// so the ggml-cpu comparison checks the semantics (scale 1/16, GQA head / 12, mask, cell layout) to a loose bound
// and the FP64 comparison checks the arithmetic.  An inconsistent step must set the status and write NaN.
#include "probe_common.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/qsa.hpp"

namespace k = strata::kernels;

int main() {
    const auto s = k::qsa_real_shapes();
    constexpr int NH = 24, NKV = 2, HD = 256, CAP = 256;
    int bad = 0;
    for (int width : {1, 5, 64, 200, 256}) {
        for (bool with_mask : {false, true}) {
            const auto qf = probe::normal((size_t) NH * HD, 40 + width, 1.0f);
            const auto kf = probe::normal((size_t) CAP * NKV * HD, 41 + width, 1.0f);
            const auto vf = probe::normal((size_t) CAP * NKV * HD, 42 + width, 1.0f);
            std::vector<uint16_t> kh(kf.size()), vh(vf.size()), mh(CAP, 0);
            for (size_t i = 0; i < kf.size(); ++i) { kh[i] = k::f16_from_f32(kf[i]); vh[i] = k::f16_from_f32(vf[i]); }
            if (with_mask) {
                const auto mf = probe::normal(CAP, 43 + width, 2.0f);
                for (int c = 0; c < CAP; ++c) mh[c] = k::f16_from_f32(c % 7 == 3 && c != 0 ? -INFINITY : mf[c]);
            }
            // FP64 reference
            std::vector<float> ref((size_t) NH * HD);
            for (int h = 0; h < NH; ++h) {
                const int kv = h / 12;
                std::vector<double> sc(width);
                double mx = -1e300;
                for (int c = 0; c < width; ++c) {
                    double a = 0;
                    for (int d = 0; d < HD; ++d)
                        a += (double) qf[h * HD + d] * k::f32_from_f16(kh[((size_t) c * NKV + kv) * HD + d]);
                    sc[c] = a / 16.0 + (with_mask ? (double) k::f32_from_f16(mh[c]) : 0.0);
                    mx = std::fmax(mx, sc[c]);
                }
                double l = 0;
                for (auto& x : sc) { x = std::exp(x - mx); l += x; }
                for (int d = 0; d < HD; ++d) {
                    double a = 0;
                    for (int c = 0; c < width; ++c) a += sc[c] * k::f32_from_f16(vh[((size_t) c * NKV + kv) * HD + d]);
                    ref[h * HD + d] = (float) (a / l);
                }
            }
            // ggml-cpu
            probe::Ggml g(256u << 20);
            ggml_tensor* q = g.f32({HD, 1, NH}, qf);
            ggml_tensor* kt = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, HD, NKV, CAP);
            ggml_tensor* vt = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, HD, NKV, CAP);
            std::memcpy(kt->data, kh.data(), kh.size() * 2);
            std::memcpy(vt->data, vh.data(), vh.size() * 2);
            ggml_tensor* kv_k = ggml_view_3d(g.ctx, kt, HD, width, NKV, kt->nb[2], kt->nb[1], 0);
            ggml_tensor* kv_v = ggml_view_3d(g.ctx, vt, HD, width, NKV, vt->nb[2], vt->nb[1], 0);
            ggml_tensor* m = nullptr;
            if (with_mask) {
                m = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, width, 1);
                std::memcpy(m->data, mh.data(), (size_t) width * 2);
            }
            ggml_tensor* fa = ggml_flash_attn_ext(g.ctx, q, kv_k, kv_v, m, 1.0f / 16.0f, 0.0f, 0.0f);
            g.compute(fa);
            const auto gg = probe::Ggml::read(fa, 0, (size_t) NH * HD);

            std::vector<int32_t> step(k::kStepCount);
            k::qsa_step_fill(step.data(), width - 1, s);
            float* d_q = probe::upload(qf);
            uint16_t* d_k = probe::upload(kh);
            uint16_t* d_v = probe::upload(vh);
            uint16_t* d_m = with_mask ? probe::upload(mh) : nullptr;
            int32_t* d_s = probe::upload(step);
            float* d_o = static_cast<float*>(strata::parity::alloc_bytes((size_t) NH * HD * 4));
            int32_t* d_st = static_cast<int32_t*>(strata::parity::alloc_bytes(4));
            k::native_flash_attn_short_step(d_q, d_k, d_v, d_s, CAP, 256, s, d_o, d_st, d_m, strata::parity::stream());
            strata::parity::sync();
            const auto got = probe::download(d_o, (size_t) NH * HD);
            const int status = probe::download(d_st, 1)[0];
            char name[96];
            std::snprintf(name, sizeof name, "width=%d mask=%d vs FP64", width, (int) with_mask);
            const auto d1 = probe::compare(got, ref);
            probe::print(name, d1);
            std::snprintf(name, sizeof name, "width=%d mask=%d vs ggml-cpu (Q rounded to F16)", width, (int) with_mask);
            const auto d2 = probe::compare(got, gg);
            probe::print(name, d2);
            if (!(status == k::kNativeFlashAttnSuccess && d1.max_rel <= 1e-5 && d2.max_rel <= 1e-2)) ++bad;
        }
    }
    {   // an inconsistent step: status set, output NaN
        std::vector<int32_t> step(k::kStepCount);
        k::qsa_step_fill(step.data(), 9, s);
        step[k::kStepNKv] = 3;
        const auto qf = probe::normal((size_t) NH * HD, 1);
        std::vector<uint16_t> zero((size_t) CAP * NKV * HD, 0);
        float* d_q = probe::upload(qf);
        uint16_t* d_k = probe::upload(zero);
        uint16_t* d_v = probe::upload(zero);
        int32_t* d_s = probe::upload(step);
        float* d_o = static_cast<float*>(strata::parity::alloc_bytes((size_t) NH * HD * 4));
        int32_t* d_st = static_cast<int32_t*>(strata::parity::alloc_bytes(4));
        k::native_flash_attn_short_step(d_q, d_k, d_v, d_s, CAP, 256, s, d_o, d_st, nullptr, strata::parity::stream());
        strata::parity::sync();
        const auto got = probe::download(d_o, (size_t) NH * HD);
        bool all_nan = true;
        for (float x : got) all_nan &= std::isnan(x);
        const int status = probe::download(d_st, 1)[0];
        const bool ok = all_nan && status == k::kNativeFlashAttnUnsupportedStep;
        std::printf("  %-44s %s (status %d)\n", "inconsistent step -> status and NaN output", ok ? "yes" : "*** NO ***", status);
        if (!ok) ++bad;
    }
    std::printf("native_flash_attn_check: %d failures (criteria: within 1e-5 of FP64 and 1e-2 of ggml-cpu relative to "
                "scale, status success; an inconsistent step reports it and writes NaN)\n", bad);
    return bad ? 1 : 0;
}
