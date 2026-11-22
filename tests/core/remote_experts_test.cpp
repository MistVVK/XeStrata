// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// A helper GPU's expert tier (RemoteExperts, --expert-cache-device1) on a mixed pack - R1Q2 experts, each with its own
// K and size - against the CPU's rows: every routed entry the helper owns, computed on the second GPU, is the row the
// CPU kernels give for that expert and token from the activations as a GPU quantizes them (one scale for each 256
// values, canon_math.hpp's: a verify window's experts in VRAM take the same; the CPU pool's own quantization has a
// scale for each 32 and differs from either by the int8 rounding, about 1e-2 here).  Needs a second GPU that is not
// the processor's own (exits 77 without one: under a selector that shows one GPU, as the per-GPU test runs do).
#include "strata/core/remote_experts.hpp"

#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int H = 256, FF = 128, LAYERS = 2, E = 4, K = 2, NT = 3;

struct Blobs final : strata::core::ExpertSource {
    std::vector<std::vector<uint8_t>> blob_of;   // layer * E + expert
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        return blob_of[(size_t) (layer * E + expert)].data();
    }
};

// A row's activations as quantize_q8_1_rows gives them to a GPU's experts (q8act::Canon<256>), in the Q8_0 blocks
// the CPU's R1Q2 rows read: an fp16 scale and 32 int8 values a block, the scale one for the 256 values.
void gpu_quant_act(const float* x, uint8_t* dst) {
    namespace canon = strata::kernels::canon;
    for (int g = 0; g < H / 256; ++g) {
        const float* v = x + (size_t) g * 256;
        float amax = 0.f;
        for (int i = 0; i < 256; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float d = canon::q8_scale(amax), id = canon::q8_inv(d);
        const uint16_t dh = strata::kernels::f16_from_f32(d);
        for (int b = 0; b < 8; ++b) {
            uint8_t* blk = dst + ((size_t) g * 8 + (size_t) b) * 34;
            std::memcpy(blk, &dh, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) canon::q8_value(v[(size_t) b * 32 + (size_t) i], id);
        }
    }
}
}  // namespace

int main() {
    int helper = -1;
    try {
        for (int d = 1; d < strata::core::Runtime::count() && helper < 0; ++d)
            if (!strata::core::device_info(d).integrated) helper = d;
    } catch (const std::exception& e) {
        std::puts(e.what());
        return 77;
    }
    if (helper < 0) {
        std::puts("remote_experts_test: one GPU here, a helper is another: skipped");
        return 77;
    }
    // the layout: two K pairs in turn, so that neighbours differ in type and size
    const float p2[3][4] = {{0.25f, 0.5f, 1.f, 2.f}, {0.5f, 1.f, 1.5f, 2.f}, {0.25f, 0.5f, 1.f, 2.f}};
    const int types[2][2] = {{182, 183}, {188, 189}};
    std::string err;
    cpu::ExpertLayout lay;
    lay.native = true;
    lay.version = 6;
    lay.n_layers = LAYERS;
    lay.n_expert = E;
    lay.expert_fmt.resize((size_t) LAYERS * E);
    lay.expert_off.resize((size_t) LAYERS * E);
    lay.layer_size.assign(LAYERS, 0);
    uint64_t at = 0;
    for (int l = 0; l < LAYERS; ++l) {
        uint64_t largest = 0;
        lay.offset.push_back(at);
        for (int e = 0; e < E; ++e) {
            cpu::NativeFmt& f = lay.expert_fmt[(size_t) l * E + (size_t) e];
            const int* t = types[(l + e) % 2];
            if (!cpu::native_r1q2_fmt(t[0], t[1], H, FF, p2, f, err)) {
                std::printf("remote_experts_test: %s\n", err.c_str());
                return 1;
            }
            lay.expert_off[(size_t) l * E + (size_t) e] = at;
            at += f.bytes;
            lay.layer_size[(size_t) l] += f.bytes;
            if (f.bytes > largest) largest = f.bytes;
        }
        lay.bytes.push_back(largest);
        lay.fmt.push_back(lay.expert_fmt[(size_t) l * E + (size_t) ((l + 1) % 2)]);   // the layer's: its larger K's
        lay.max_blob = std::max<uint64_t>(l == 0 ? 0 : lay.max_blob, largest);
    }
    lay.total = at;
    cpu::expert_layout_set(lay);

    std::mt19937 rng(770);   // NOLINT(cert-msc51-cpp,bugprone-random-generator-seed): a reproducible input
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    Blobs src;
    src.blob_of.resize((size_t) LAYERS * E);
    for (int i = 0; i < LAYERS * E; ++i) {
        const cpu::NativeFmt& f = lay.expert_fmt[(size_t) i];
        auto& blob = src.blob_of[(size_t) i];
        blob.resize(f.bytes);
        for (auto& v : blob) v = (uint8_t) rng();
        for (int m = 0; m < 3; ++m) {   // FP16 row bases of 0.05 and column scales of 0.25, the codes as they fell
            const int rows = m == 2 ? H : FF, blocks = (m == 2 ? FF : H) / 128;
            uint8_t* sc = blob.data() + f.sc_off + cpu::native_r1q2_scale_offset(f, m);
            for (int r = 0; r < rows; ++r) { sc[(size_t) 2 * r] = 0x66; sc[(size_t) 2 * r + 1] = 0x2a; }
            for (int b = 0; b < blocks; ++b) { sc[2 * rows + 2 * b] = 0x00; sc[2 * rows + 2 * b + 1] = 0x34; }
        }
    }

    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int l = 0; l < LAYERS; ++l)
        for (int e = 0; e < E; ++e) ranked.emplace_back(l, e);
    std::vector<uint8_t> claimed((size_t) LAYERS * E, 0);
    const strata::core::ExpertCache primary;   // never opened: it holds nothing, every pair is the helper's
    strata::core::RemoteExperts remote;
    if (!remote.open(helper, LAYERS * E, LAYERS, E, ranked, primary, src, claimed, err)) {
        std::printf("remote_experts_test: open: %s\n", err.c_str());
        return 1;
    }
    if (remote.resident() != (int64_t) LAYERS * E) {
        std::printf("remote_experts_test: the helper holds %lld of %d experts\n", (long long) remote.resident(),
                    LAYERS * E);
        return 1;
    }

    double worst = 0.0;
    for (int l = 0; l < LAYERS; ++l) {
        std::vector<float> x((size_t) NT * H), got((size_t) NT * K * H, 0.f), want(got.size(), 0.f);
        for (float& v : x) v = dist(rng);
        // token t routes experts t and t + 1 + l (mod E): every expert, some twice, the two K pairs in one window
        int32_t ids[NT * K], kind[NT * K];
        for (int t = 0; t < NT; ++t) {
            const size_t at_t = (size_t) t * K;
            ids[at_t] = t % E;
            ids[at_t + 1] = (t + 1 + l) % E;
        }
        for (int i = 0; i < NT * K; ++i) kind[i] = -1;
        kind[1] = 0;   // an entry the first GPU takes: not the helper's
        if (!remote.begin(l, x.data(), ids, NT, K, kind, nullptr, err) || !remote.finish(got.data(), err)) {
            std::printf("remote_experts_test: layer %d: %s\n", l, err.c_str());
            return 1;
        }
        for (int i = 0; i < NT * K; ++i) {
            if (remote.owns(i) != (i != 1)) {
                std::printf("remote_experts_test: layer %d entry %d: owned %d\n", l, i, (int) remote.owns(i));
                return 1;
            }
            if (i == 1) continue;
            const cpu::NativeFmt& f = lay.format(l, ids[i]);
            const uint8_t* blob = src.blob(l, ids[i]);
            cpu::NativeAct act, hq;
            gpu_quant_act(x.data() + (size_t) (i / K) * H, (uint8_t*) act.bytes);
            const void* a[] = {act.bytes};
            std::vector<float> hidden(FF);
            float* hp[] = {hidden.data()};
            cpu::native_gu_rows(f, blob, a, 1, hp, 0, FF);
            cpu::native_quant_h(f, hidden.data(), hq.bytes);
            const void* ha[] = {hq.bytes};
            float* op[] = {want.data() + (size_t) i * H};
            cpu::native_down_rows(f, blob, ha, 1, op, 0, H);
        }
        double n = 0, d = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            n += std::fabs((double) got[i] - want[i]);
            d += std::fabs((double) want[i]);
        }
        const double rel = n / (d + 1e-30);
        std::printf("remote experts, mixed R1Q2, layer %d on GPU %d: rel %.9g\n", l, helper, rel);
        if (!(rel == rel) || rel > worst) worst = rel == rel ? rel : 1.0;
    }
    remote.close();
    return worst <= 1e-6 ? 0 : 1;
}
