// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_dit_probe.cpp - strata-image-moss-dit-probe: MOSS-Image's DiT (MossDit) against the reference
// pipeline's first step (diffusers' MOSS-Image pipeline in bf16: its noise-short.npy [1][128][h][w], te-short.npy and
// te-negative.npy [rows][4096], timesteps.npy, and the two velocities of the first step, step0-cond.npy (the prompt)
// and step0-uncond-skip22.npy (the empty negative prompt, block 22 skipped)).
//
//   strata-image-moss-dit-probe TRANSFORMER_DIR REF_DIR [q8|bf16]
//
// First the pipeline's schedule (moss_sigmas: 50 steps, shift 5) and seed 7's noise (torch_cpu_randn) against the
// reference's sigmas.npy, timesteps.npy and noise-short.npy.  Then prints each velocity's cosine to the reference's, its relative error (L1: sum |got - want| / sum |want|; L2), and
// the time of a step (both contexts, then the prompt's alone).  MOSS_BLOCK_BUDGET_MIB: the blocks' VRAM (the rest
// streamed); without it, what the GPU has free beside the working buffers and 512 MiB.  MOSS_PROBE_RUNS: the timed
// steps (3).  MOSS_MLP_ROWS: the MLP's rows a band (MossDit::set_mlp_rows; without it, all).
// An edit's references: MOSS_PROBE_REFS, .npy latents [1][128][h][w] (commas between them), in the layout
// MOSS_PROBE_LAYOUT (moss, aitk, aitk-nokv or index: MossDit::set_refs); MOSS_PROBE_DUMP, a folder the two velocities
// go to (cond.npy and uncond.npy, [1][128][h][w]) beside the comparison.

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/moss_dit.hpp"
#include "strata/image/moss_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// a .npy file's float32 data (C order), and its shape
std::vector<float> npy(const std::string& path, std::vector<int64_t>* shape = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy file");
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": expected '<f4' in C order: " + header);
    const size_t l = header.find('('), r = header.find(')');
    std::vector<int64_t> dims;
    size_t n = 1;
    for (size_t i = l + 1; i < r;) {
        const size_t j = header.find_first_of(",)", i);
        const std::string tok = header.substr(i, j - i);
        if (tok.find_first_not_of(' ') != std::string::npos) {
            dims.push_back(std::stoll(tok));
            n *= (size_t) dims.back();
        }
        i = j + 1;
    }
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * 4));
    if (!f) throw std::runtime_error(path + ": short");
    if (shape) *shape = dims;
    return v;
}

void compare(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    double dot = 0, ng = 0, nw = 0, l1 = 0, mag = 0, l2 = 0, mx = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double g = got[i], w = want[i], d = g - w;
        dot += g * w;
        ng += g * g;
        nw += w * w;
        l1 += std::fabs(d);
        mag += std::fabs(w);
        l2 += d * d;
        mx = std::max(mx, std::fabs(d));
    }
    std::printf("%-8s cos %.6f  rel L1 %.3e  rel L2 %.3e  max |diff| %.3g (of %.3g)\n", what,
                dot / std::sqrt(ng * nw), l1 / mag, std::sqrt(l2 / nw), mx, std::sqrt(nw / (double) want.size()));
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<float> fetch(const float* d, size_t n) {
    std::vector<float> h(n);
    strata::gpu::copy(h.data(), d, n * 4);
    return h;
}

// v as a .npy file of float32, C order, of these dims
void save_npy(const std::string& path, const std::vector<float>& v, const std::vector<int64_t>& dims) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (const int64_t d : dims) header += std::to_string(d) + ", ";
    header += "), }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    const uint16_t hl = (uint16_t) header.size();
    f.write("\x93NUMPY\x01\x00", 8);
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(header.data(), (std::streamsize) header.size());
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize) (v.size() * 4));
    if (!f) throw std::runtime_error("cannot write " + path);
}

float* put(const std::vector<float>& h) {
    float* d = nullptr;
    if (!strata::gpu::alloc_device(&d, h.size() * 4)) throw std::runtime_error("GPU memory");
    strata::gpu::copy(d, h.data(), h.size() * 4);
    return d;
}

// the schedule and the noise the pipeline makes against the reference's
void schedule(const std::string& dir) {
    const auto sig = npy(dir + "/sigmas.npy"), ts = npy(dir + "/timesteps.npy");
    const auto mine = strata::image::moss_sigmas(50, 5.0f);
    size_t same = 0, tsame = 0;
    for (size_t i = 0; i < std::min(sig.size(), mine.size()); ++i) same += sig[i] == mine[i];
    for (size_t i = 0; i < std::min(ts.size(), mine.size()); ++i) tsame += ts[i] == mine[i] * 1000.f;
    std::printf("sigmas: %zu of %zu equal (%zu made); timesteps: %zu of %zu equal\n", same, sig.size(), mine.size(),
                tsame, ts.size());
    std::vector<int64_t> ns;
    const auto noise = npy(dir + "/noise-short.npy", &ns);
    const auto n = strata::image::torch_cpu_randn(7, noise.size());
    size_t eq = 0;
    double mx = 0;
    for (size_t i = 0; i < n.size(); ++i) {
        eq += n[i] == noise[i];
        mx = std::max(mx, (double) std::fabs(n[i] - noise[i]));
    }
    std::printf("noise (seed 7): %zu of %zu equal, max |diff| %.3g\n", eq, n.size(), mx);
}

int run(const std::string& path, const std::string& dir, bool q8) {
    schedule(dir);
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MossDit d;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!d.open(path, s, err, false, q8)) throw std::runtime_error(err);
    std::printf("open (%s): %.1f s, blocks %.2f GiB, the rest %.2f GiB\n",
                !q8 ? "FP16" : d.w8a8() ? "W8A8" : "Q8_0", seconds_since(t0), (double) d.block_bytes() / 1073741824.0,
                (double) d.weight_bytes() / 1073741824.0);
    std::vector<int64_t> ns, ps, gs;
    const auto noise = npy(dir + "/noise-short.npy", &ns);   // [1][128][h][w]
    const auto te = npy(dir + "/te-short.npy", &ps);
    const auto neg = npy(dir + "/te-negative.npy", &gs);
    const float t = npy(dir + "/timesteps.npy")[0];
    if (ns.size() != 4 || ps.size() != 2 || gs.size() != 2) throw std::runtime_error(dir + ": unexpected shapes");
    const int64_t Cl = ns[1], h = ns[2], w = ns[3], HW = h * w;
    t0 = std::chrono::steady_clock::now();
    d.set_context(0, te.data(), ps[0]);
    d.set_context(1, neg.data(), gs[0]);
    d.set_skip(1, {22});
    strata::gpu::stream_sync(s);
    std::printf("text embedder (%lld and %lld rows): %.1f ms\n", (long long) ps[0], (long long) gs[0],
                1e3 * seconds_since(t0));
    // the blocks' budget: what is free beside the working buffers and 512 MiB
    uint64_t budget = d.block_bytes();
    if (const char* e = std::getenv("MOSS_BLOCK_BUDGET_MIB")) {
        budget = (uint64_t) std::strtoull(e, nullptr, 10) << 20;
    } else {   // (the blocks are resident from open on: what they hold counts as free)
        size_t free_b = 0, total_b = 0;
        const uint64_t act = d.activation_bytes(HW, 2, std::max(ps[0], gs[0])) + (512ull << 20);
        if (strata::gpu::mem_info(&free_b, &total_b)) {
            const uint64_t avail = free_b + d.blocks_vram();
            if (avail < act + d.block_bytes()) budget = std::max<uint64_t>(d.block_min(), avail > act ? avail - act : 0);
        }
    }
    if (budget < d.block_bytes()) {
        d.set_block_budget(budget);
        std::printf("blocks: %d of %d resident (budget %.2f GiB)\n", d.resident_blocks(), d.blocks(),
                    (double) budget / 1073741824.0);
    }
    if (const char* e = std::getenv("MOSS_MLP_ROWS")) d.set_mlp_rows(std::strtoll(e, nullptr, 10));
    std::vector<float*> refs;
    if (const char* e = std::getenv("MOSS_PROBE_REFS")) {
        const std::string list = e, lay = std::getenv("MOSS_PROBE_LAYOUT") ? std::getenv("MOSS_PROBE_LAYOUT") : "";
        using L = strata::image::QiRefLayout;
        const L layout = lay == "aitk"        ? L::AitkKv
                         : lay == "aitk-nokv" ? L::AitkNoKv
                         : lay == "appended"  ? L::Appended
                                              : L::MossEdit;
        std::vector<strata::image::MossDit::RefLatent> lat;
        for (size_t i = 0; i < list.size();) {
            const size_t j = std::min(list.find(',', i), list.size());
            std::vector<int64_t> rs;
            const auto z = npy(list.substr(i, j - i), &rs);
            if (rs.size() != 4 || rs[1] != Cl) throw std::runtime_error(list.substr(i, j - i) + ": not [1][128][h][w]");
            const int64_t rhw = rs[2] * rs[3];
            std::vector<float> zt((size_t) (rhw * Cl));
            for (int64_t c = 0; c < Cl; ++c)
                for (int64_t p = 0; p < rhw; ++p) zt[(size_t) (p * Cl + c)] = z[(size_t) (c * rhw + p)];
            refs.push_back(put(zt));
            lat.push_back({refs.back(), rs[2], rs[3]});
            i = j + 1;
        }
        t0 = std::chrono::steady_clock::now();
        d.set_refs(lat, layout);
        strata::gpu::stream_sync(s);
        std::printf("references (%s, %zu): %.1f ms\n", lay.empty() ? "moss" : lay.c_str(), lat.size(),
                    1e3 * seconds_since(t0));
    }
    std::vector<float> xt((size_t) (HW * Cl));   // the noise as tokens [h w][128]
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t p = 0; p < HW; ++p) xt[(size_t) (p * Cl + c)] = noise[(size_t) (c * HW + p)];
    float* x = put(xt);
    float* out = nullptr;
    if (!strata::gpu::alloc_device(&out, (size_t) (2 * HW * Cl) * 4)) throw std::runtime_error("GPU memory");
    t0 = std::chrono::steady_clock::now();
    d.forward(x, t, h, w, 2, out);
    strata::gpu::stream_sync(s);
    std::printf("first step (both contexts, compiling): %.1f ms\n", 1e3 * seconds_since(t0));
    const auto o = fetch(out, (size_t) (2 * HW * Cl));
    for (int i = 0; i < 2; ++i) {   // back to [128][h w]
        std::vector<float> v((size_t) (HW * Cl));
        for (int64_t p = 0; p < HW; ++p)
            for (int64_t c = 0; c < Cl; ++c) v[(size_t) (c * HW + p)] = o[(size_t) ((i * HW + p) * Cl + c)];
        if (const char* e = std::getenv("MOSS_PROBE_DUMP"))
            save_npy(std::string(e) + (i == 0 ? "/cond.npy" : "/uncond.npy"), v, {1, Cl, h, w});
        compare(i == 0 ? "cond" : "uncond", v, npy(dir + (i == 0 ? "/step0-cond.npy" : "/step0-uncond-skip22.npy")));
    }
    const char* re = std::getenv("MOSS_PROBE_RUNS");
    const int runs = re != nullptr ? std::max(1, (int) std::strtol(re, nullptr, 10)) : 3;
    for (int n = 2; n >= 1; --n) {
        std::vector<double> ms;
        for (int r = 0; r < runs; ++r) {
            t0 = std::chrono::steady_clock::now();
            d.forward(x, t, h, w, n, out);
            strata::gpu::stream_sync(s);
            ms.push_back(1e3 * seconds_since(t0));
        }
        std::sort(ms.begin(), ms.end());
        std::printf("step %lldx%lld, %s: best %.1f ms, median %.1f ms (%d runs)\n", (long long) h * 16,
                    (long long) w * 16, n == 2 ? "cond + uncond (skip 22)" : "cond alone", ms.front(),
                    ms[ms.size() / 2], runs);
    }
    const auto [f16, w8] = d.time_forms(HW);
    std::printf("time_forms(%lld rows): FP16 %.2f ms, W8A8 %.2f ms\n", (long long) HW, 1e3 * f16, 1e3 * w8);
    strata::gpu::free(out);
    strata::gpu::free(x);
    for (float* r : refs) strata::gpu::free(r);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s TRANSFORMER_DIR REF_DIR [q8|bf16]\n", argv[0]);
        return 2;
    }
    try {
        return run(argv[1], argv[2], argc < 4 || std::string(argv[3]) != "bf16");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
