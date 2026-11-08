// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_block_probe.cpp - strata-image-probe: one transformer block of Qwen-Image-2.1's DiT on the GPU,
// against the NumPy reference's input and output of that block (tools/image/qi21_ref.py --dump-block), and timed.
//
//   strata-image-probe DIT.gguf BLOCK_DIR [BLOCK] [REPS]
//
// Prints how far the block's output and its attention are from the reference, then the block's time (the median of
// REPS runs after a warm-up), the products' share and the attention's.

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/flash_attn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace gpu = strata::gpu;

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T> std::vector<T> load(const std::string& dir, const std::string& name) {
    const auto raw = read_file(dir + "/" + name + ".bin");
    std::vector<T> v(raw.size() / sizeof(T));
    std::copy(raw.begin(), raw.end(), reinterpret_cast<char*>(v.data()));
    return v;
}

template <class T> T* upload(const T* host, size_t n) {
    T* d = nullptr;
    if (!gpu::alloc_device(&d, n * sizeof(T)) || !gpu::copy(d, host, n * sizeof(T)))
        throw std::runtime_error(std::string("upload: ") + gpu::last_error());
    return d;
}

template <class T> T* device(size_t n) {
    T* d = nullptr;
    if (!gpu::alloc_device(&d, n * sizeof(T))) throw std::runtime_error(std::string("alloc: ") + gpu::last_error());
    return d;
}

struct Weight {
    int type = 0;
    int64_t n = 0, k = 0;   // n rows of k
    const void* d = nullptr;
};

void compare(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0, mx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        dot += (double) got[i] * want[i];
        ng += (double) got[i] * got[i];
        nw += (double) want[i] * want[i];
        const double e = std::fabs((double) got[i] - want[i]);
        err += e;
        mag += std::fabs((double) want[i]);
        mx = std::max(mx, e);
    }
    std::printf("%-6s cos %.6f  rel %.3g  max %.3g\n", what, dot / std::sqrt(ng * nw), err / mag, mx);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s DIT.gguf BLOCK_DIR [BLOCK] [REPS]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    const int block = argc > 3 ? (int) std::strtol(argv[3], nullptr, 10) : 0;
    const int reps = argc > 4 ? (int) std::strtol(argv[4], nullptr, 10) : 5;
    try {
        strata::GgufFile g(argv[1]);
        const std::string p = "model.diffusion_model.transformer_blocks." + std::to_string(block) + ".";
        auto weight = [&](const std::string& name) {
            const strata::TensorInfo* t = g.find(p + name);
            if (!t) throw std::runtime_error("missing " + p + name);
            Weight w;
            w.type = (int) t->type;
            w.k = (int64_t) t->shape[0];
            w.n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
            w.d = upload(g.tensor_data(*t), strata::tensor_payload_bytes(*t));
            return w;
        };
        const Weight wq = weight("attn.to_q.weight"), wk = weight("attn.to_k.weight"), wv = weight("attn.to_v.weight"),
                     wo = weight("attn.to_out.0.weight"), wgu = weight("img_mlp.gate_up.weight"),
                     wd = weight("img_mlp.out.weight"), nq = weight("attn.norm_q.weight"),
                     nk = weight("attn.norm_k.weight");
        const auto x_in = load<float>(dir, "x_in"), mods = load<float>(dir, "mods"), cosv = load<float>(dir, "rope_cos"),
                   sinv = load<float>(dir, "rope_sin"), want = load<float>(dir, "x_out"),
                   want_attn = load<float>(dir, "attn");
        const int64_t n_text = load<int32_t>(dir, "n_text")[0];
        const int64_t C = wq.k, T = (int64_t) x_in.size() / C, H = C / 128, inter = wgu.n / 2;
        std::printf("block %d: T %lld (text %lld), C %lld, heads %lld, inter %lld, weights type %d\n", block,
                    (long long) T, (long long) n_text, (long long) C, (long long) H, (long long) inter, wq.type);

        void* s = strata::core::Runtime::get().create_stream();
        strata::prefill::Gemm gemm;
        std::string err;
        if (!gemm.init(s, wgu.n * wgu.k, err)) throw std::runtime_error("gemm: " + err);
        std::printf("products: %s\n", strata::prefill::Gemm::path());

        float* x0 = upload(x_in.data(), x_in.size());
        float* x = device<float>(T * C);
        float* md = upload(mods.data(), mods.size());
        float* cs = upload(cosv.data(), cosv.size());
        float* sn = upload(sinv.data(), sinv.size());
        uint16_t* h16 = device<uint16_t>(T * inter);
        float* q = device<float>(T * C);
        float* k = device<float>(T * C);
        float* v = device<float>(T * C);
        float* a = device<float>(T * C);
        float* gu = device<float>(T * 2 * inter);
        const int64_t TP = strata::image::qi_attn_rows(T);
        uint16_t *q16 = device<uint16_t>(TP * C), *k16 = device<uint16_t>(TP * C), *v16 = device<uint16_t>(TP * C);
        for (uint16_t* z : {q16, k16, v16}) gpu::memset(z, 0, TP * C * sizeof(uint16_t));
        auto mod = [&](int i) { return md + (int64_t) i * 2 * C; };

        auto products_qkv = [&]() {
            strata::image::qi_ln_mod16(x, mod(0), n_text, T, C, h16, s);
            gemm.native(h16, wq.type, wq.d, q, T, wq.n, wq.k);
            gemm.native(h16, wk.type, wk.d, k, T, wk.n, wk.k);
            gemm.native(h16, wv.type, wv.d, v, T, wv.n, wv.k);
            strata::image::qi_qk_norm_rope16(q, k, v, (const float*) nq.d, (const float*) nk.d, cs, sn, T, H, 128,
                                             q16, k16, v16, s);
        };
        auto products_rest = [&]() {
            strata::kernels::f32_to_f16_bulk(a, h16, T * C, s);
            gemm.native(h16, wo.type, wo.d, q, T, wo.n, wo.k);
            strata::image::qi_gate_add(x, q, mod(1), n_text, T, C, s);
            strata::image::qi_ln_mod16(x, mod(2), n_text, T, C, h16, s);
            gemm.native(h16, wgu.type, wgu.d, gu, T, wgu.n, wgu.k);
            strata::prefill::swiglu_pair(gu, gu + inter, h16, T, s, inter, 2 * inter);
            gemm.native(h16, wd.type, wd.d, q, T, wd.n, wd.k);
            strata::image::qi_gate_add(x, q, mod(3), n_text, T, C, s);
        };
        auto attention = [&]() {   // the FMA form of the shared attention (flash_attn.hpp), FP32 out
            strata::kernels::FlashAttnArgs fa;
            fa.q = q16;
            fa.k = k16;
            fa.v = v16;
            fa.q_ld = fa.kv_ld = fa.out_ld = H * 128;
            fa.n_head = fa.n_kv = H;
            fa.head_dim = 128;
            fa.rows = fa.keys = fa.row_end = T;
            fa.n_causal = n_text;
            fa.out = a;
            fa.out_f32 = true;
            fa.scale = 1.0f / std::sqrt(128.0f);
            fa.fma = true;
            strata::kernels::flash_attn(fa, s);
        };
        uint16_t* o16 = device<uint16_t>(T * C);
        auto attention_mx = [&]() { strata::image::qi_attention(q16, k16, v16, T, H, 128, n_text, o16, s); };
        auto sync = [&]() {
            if (!gpu::stream_sync(s)) throw std::runtime_error(std::string("sync: ") + gpu::last_error());
        };
        auto reset = [&]() {
            if (!gpu::copy_async(x, x0, T * C * sizeof(float), s)) throw std::runtime_error("reset");
        };

        reset();
        products_qkv();
        attention();
        sync();
        std::vector<float> got_attn(T * C);
        gpu::copy(got_attn.data(), a, got_attn.size() * sizeof(float));
        products_rest();
        sync();
        std::vector<float> got(T * C);
        gpu::copy(got.data(), x, got.size() * sizeof(float));
        compare("attn", got_attn, want_attn);
        compare("block", got, want);
        reset();
        products_qkv();
        attention_mx();
        sync();
        {
            std::vector<uint16_t> h(T * C);
            gpu::copy(h.data(), o16, h.size() * sizeof(uint16_t));
            std::vector<float> f(h.size());
            for (size_t i = 0; i < h.size(); ++i) f[i] = strata::kernels::f32_from_f16(h[i]);
            std::printf("attention path: %s\n", strata::image::qi_attention_path());
            compare("attn16", f, want_attn);
        }

        auto timed = [&](auto&& fn) {
            std::vector<double> ms;
            for (int r = 0; r < reps + 1; ++r) {
                reset();
                sync();
                const auto t0 = std::chrono::steady_clock::now();
                fn();
                sync();
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            ms.erase(ms.begin());
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];
        };
        const double t_prod = timed([&]() { products_qkv(); products_rest(); });
        const double t_attn = timed(attention);
        const double t_mx = timed(attention_mx);
        const double flops = 2.0 * (double) T * (4.0 * (double) C * (double) C + 3.0 * (double) C * (double) inter);
        const double aflops = 4.0 * (double) T * (double) T * (double) C;
        std::printf("time: products and element kernels %.2f ms (%.1f TFLOPS), plain attention %.2f ms, "
                    "attention %.2f ms (%.1f TFLOPS)\n", t_prod, flops / t_prod / 1e9, t_attn, t_mx,
                    aflops / t_mx / 1e9);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-probe: %s\n", e.what());
        return 1;
    }
    return 0;
}
