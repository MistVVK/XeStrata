// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_text_parity.cpp - MOSS-Image's text encoder on a synthetic Qwen3.5 (qwen35: three GDN blocks and an
// attention block, 256 wide, the 9B's heads; BF16 projections, the FFN's gate and up in Q8_0 and its down in Q6_K, as
// the engine lays out their rows) against the chat's prompt (GPU, no model):
//
//   1. the chat engine's encode (the server's ENCODE): its last row is what the prompt's logits are made from - the
//      BF16 head over it on the host (FP64) gives the prompt's logits to 1e-4 of their largest (the head's FP32 sums);
//   2. TextEncoder with every block resident: bit for bit the chat engine's encode (the same kernels in the same order);
//   3. TextEncoder with only the two slots, and with some blocks resident: bit for bit 2 (the blocks are copies).
//
// STRATA_PLAIN_DENSE_MMQ=0: every dense product through the FP16 path, so that two loads do not choose between the
// int8 and the FP16 products by their timings.
#include "strata/core/plain_model.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/text_encoder.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int64_t kEmbd = 256, kFf = 512, kVocab = 512, kHead = 16, kKv = 4, kHd = 256;
constexpr int64_t kState = 128, kGroups = 16, kRank = 32, kInner = kRank * kState, kConv = 4, kLayers = 4;
constexpr uint32_t kF32 = 0, kQ8_0 = 8, kQ6_K = 14, kBF16 = 30;

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

uint16_t bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;   // ne0 (the input) first
    uint32_t type = kF32;
    std::vector<uint8_t> data;
};

// A tensor of n_out rows of n_in values ~ N(mean, sd) in `type`
Tensor make(const std::string& name, std::vector<uint64_t> shape, uint32_t type, std::mt19937& rng, float sd,
            float mean = 0.0f) {
    Tensor t{name, std::move(shape), type, {}};
    uint64_t n = 1;
    for (uint64_t d : t.shape) n *= d;
    std::normal_distribution<float> nd(mean, sd);
    if (type == kF32) {
        for (uint64_t i = 0; i < n; ++i) put<float>(t.data, nd(rng));
    } else if (type == kBF16) {
        for (uint64_t i = 0; i < n; ++i) put<uint16_t>(t.data, bf16(nd(rng)));
    } else if (type == kQ8_0) {   // blocks of 32: an FP16 scale, 32 int8
        std::uniform_int_distribution<int> q(-127, 127);
        for (uint64_t b = 0; b < n / 32; ++b) {
            put<uint16_t>(t.data, strata::kernels::f16_from_f32(sd / 64.0f));
            for (int i = 0; i < 32; ++i) put<int8_t>(t.data, (int8_t) q(rng));
        }
    } else {   // Q6_K, blocks of 256: ql[128], qh[64], scales[16] (int8), an FP16 scale
        std::uniform_int_distribution<int> byte(0, 255), sc(-32, 32);
        for (uint64_t b = 0; b < n / 256; ++b) {
            for (int i = 0; i < 192; ++i) put<uint8_t>(t.data, (uint8_t) byte(rng));
            for (int i = 0; i < 16; ++i) put<int8_t>(t.data, (int8_t) sc(rng));
            put<uint16_t>(t.data, strata::kernels::f16_from_f32(sd / 1024.0f));
        }
    }
    return t;
}

// The synthetic Qwen3.5 as a GGUF v3 file; its BF16 head (for the host's logits) into `head`
std::filesystem::path write_model(std::vector<float>& head) {
    std::mt19937 rng(20261027);  // NOLINT(bugprone-random-generator-seed): the same model each run.
    std::vector<Tensor> ts;
    const float w_sd = 1.0f / std::sqrt((float) kEmbd);
    ts.push_back(make("token_embd.weight", {kEmbd, kVocab}, kBF16, rng, 1.0f));   // a host row (ggml has no F32 to_float)
    ts.push_back(make("output_norm.weight", {kEmbd}, kF32, rng, 0.1f, 1.0f));
    ts.push_back(make("output.weight", {kEmbd, kVocab}, kBF16, rng, w_sd));
    for (int64_t l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ts.push_back(make(p + "attn_norm.weight", {kEmbd}, kF32, rng, 0.1f, 1.0f));
        ts.push_back(make(p + "post_attention_norm.weight", {kEmbd}, kF32, rng, 0.1f, 1.0f));
        if (l % 4 != 3) {   // full_attention_interval 4: blocks 0-2 recurrent
            const uint64_t C = 2 * kGroups * kState + kInner;
            ts.push_back(make(p + "attn_qkv.weight", {kEmbd, C}, kBF16, rng, w_sd));
            ts.push_back(make(p + "attn_gate.weight", {kEmbd, kInner}, kBF16, rng, w_sd));
            ts.push_back(make(p + "ssm_alpha.weight", {kEmbd, kRank}, kBF16, rng, w_sd));
            ts.push_back(make(p + "ssm_beta.weight", {kEmbd, kRank}, kBF16, rng, w_sd));
            ts.push_back(make(p + "ssm_out.weight", {kInner, kEmbd}, kBF16, rng, 1.0f / std::sqrt((float) kInner)));
            ts.push_back(make(p + "ssm_conv1d.weight", {kConv, C}, kF32, rng, 0.5f));
            ts.push_back(make(p + "ssm_dt.bias", {kRank}, kF32, rng, 0.5f));
            ts.push_back(make(p + "ssm_a", {kRank}, kF32, rng, 0.2f, -1.0f));
            ts.push_back(make(p + "ssm_norm.weight", {kState}, kF32, rng, 0.1f, 1.0f));
        } else {
            ts.push_back(make(p + "attn_q.weight", {kEmbd, 2 * kHead * kHd}, kBF16, rng, w_sd));
            ts.push_back(make(p + "attn_k.weight", {kEmbd, kKv * kHd}, kBF16, rng, w_sd));
            ts.push_back(make(p + "attn_v.weight", {kEmbd, kKv * kHd}, kBF16, rng, w_sd));
            ts.push_back(make(p + "attn_output.weight", {kHead * kHd, kEmbd}, kBF16, rng, 1.0f / 64.0f));
            ts.push_back(make(p + "attn_q_norm.weight", {kHd}, kF32, rng, 0.1f, 1.0f));
            ts.push_back(make(p + "attn_k_norm.weight", {kHd}, kF32, rng, 0.1f, 1.0f));
        }
        ts.push_back(make(p + "ffn_gate.weight", {kEmbd, kFf}, kQ8_0, rng, w_sd));
        ts.push_back(make(p + "ffn_up.weight", {kEmbd, kFf}, kQ8_0, rng, w_sd));
        ts.push_back(make(p + "ffn_down.weight", {kFf, kEmbd}, kQ6_K, rng, 1.0f / std::sqrt((float) kFf)));
    }
    head.clear();
    for (size_t i = 0; i < ts[2].data.size(); i += 2) head.push_back(from_bf16((uint16_t) (ts[2].data[i] | ts[2].data[i + 1] << 8)));

    const std::string a = "qwen35.";
    std::vector<uint8_t> kv;
    uint64_t n_kv = 0;
    auto u32 = [&](const std::string& k, uint32_t v) {
        put_str(kv, k); put<uint32_t>(kv, 4); put<uint32_t>(kv, v); ++n_kv;
    };
    auto f32 = [&](const std::string& k, float v) {
        put_str(kv, k); put<uint32_t>(kv, 6); put<float>(kv, v); ++n_kv;
    };
    put_str(kv, "general.architecture"); put<uint32_t>(kv, 8); put_str(kv, "qwen35"); ++n_kv;
    u32(a + "block_count", kLayers);
    u32(a + "context_length", 4096);
    u32(a + "embedding_length", kEmbd);
    u32(a + "feed_forward_length", kFf);
    u32(a + "attention.head_count", kHead);
    u32(a + "attention.head_count_kv", kKv);
    u32(a + "attention.key_length", kHd);
    u32(a + "attention.value_length", kHd);
    f32(a + "attention.layer_norm_rms_epsilon", 1e-6f);
    u32(a + "rope.dimension_count", 64);
    put_str(kv, a + "rope.dimension_sections"); put<uint32_t>(kv, 9); put<uint32_t>(kv, 4); put<uint64_t>(kv, 4);
    for (uint32_t x : {11u, 11u, 10u, 0u}) put<uint32_t>(kv, x);
    ++n_kv;
    f32(a + "rope.freq_base", 1e7f);
    u32(a + "ssm.conv_kernel", kConv);
    u32(a + "ssm.inner_size", kInner);
    u32(a + "ssm.state_size", kState);
    u32(a + "ssm.time_step_rank", kRank);
    u32(a + "ssm.group_count", kGroups);
    u32(a + "full_attention_interval", 4);

    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, n_kv);
    b.insert(b.end(), kv.begin(), kv.end());
    uint64_t off = 0;
    for (const Tensor& t : ts) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.shape.size());
        for (uint64_t d : t.shape) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        off += (t.data.size() + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32, 0);
    for (const Tensor& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        b.resize((b.size() + 31) / 32 * 32, 0);
    }
    const auto path = std::filesystem::temp_directory_path() / "xestrata_moss_text_parity.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

}  // namespace

int main() {
#if defined(_WIN32)
    _putenv_s("XESTRATA_PLAIN_DENSE_MMQ", "0");
#else
    setenv("XESTRATA_PLAIN_DENSE_MMQ", "0", 1);
#endif
    std::vector<float> head;
    const std::filesystem::path path = write_model(head);
    const int64_t n = 40, chunk = 64;
    std::vector<int32_t> ids((size_t) n);
    for (int64_t i = 0; i < n; ++i) ids[(size_t) i] = (int32_t) ((i * 37 + 11) % kVocab);
    std::string err;
    try {
        strata::core::Runtime::get();
        std::vector<float> chat((size_t) (n * kEmbd));
        {   // 1. the chat engine: the prompt's logits, then its encode
            strata::core::PlainModel m;
            strata::core::PlainOptions o;
            o.gguf = path.string();
            o.max_cells = 256;
            o.prompt_chunk = chunk;
            o.prompt_chunk_given = true;
            o.spec = 0;
            if (!m.load(o, err) || !m.prompt_warm(err)) throw std::runtime_error(err);
            std::vector<float> logits((size_t) kVocab);
            int32_t next = -1;
            if (!m.prompt(ids.data(), n, 0, &next, logits.data(), err) || !m.encode(ids.data(), n, chat.data(), err))
                throw std::runtime_error(err);
            const float* last = chat.data() + (n - 1) * kEmbd;
            double worst = 0, top = 0;
            int64_t best = 0;
            for (int64_t v = 0; v < kVocab; ++v) {
                double y = 0;
                for (int64_t i = 0; i < kEmbd; ++i) y += (double) head[(size_t) (v * kEmbd + i)] * last[i];
                worst = std::max(worst, std::fabs(y - logits[(size_t) v]));
                top = std::max(top, std::fabs((double) logits[(size_t) v]));
                if (logits[(size_t) v] > logits[(size_t) best]) best = v;
            }
            std::printf("  the head over encode's last row against the prompt's logits: %.2e of %.2f\n", worst, top);
            check(worst <= 1e-4 * top, "1. encode's last row gives the prompt's logits (FP64 head, 1e-4)");
            check(best == next, "1. and its greedy token");
        }
        auto same = [&](const std::vector<float>& a, const std::vector<float>& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        strata::image::TextEncoder te;
        if (!te.open(path.string(), chunk, err)) throw std::runtime_error(err);
        const std::vector<float> all = te.encode(ids, err);
        check(te.resident_layers() == kLayers && same(all, chat), "2. TextEncoder, every block resident: the chat's encode");
        te.set_layer_budget(te.layer_min());
        const std::vector<float> slots = te.encode(ids, err);
        check(te.resident_layers() == 0 && same(slots, all), "3. two slots, every block streamed: bit for bit");
        te.set_layer_budget(te.layer_min() + te.layer_bytes() / 3);   // a block or two of the four
        const std::vector<float> half = te.encode(ids, err);
        check(te.resident_layers() > 0 && te.resident_layers() < kLayers && same(half, all),
              "3. some blocks resident, the rest streamed: bit for bit");
    } catch (const std::exception& e) {
        std::printf("  %s\n", e.what());
        ++g_fail;
    }
    std::filesystem::remove(path);
    std::printf("moss_text_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
