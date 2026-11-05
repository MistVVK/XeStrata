// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_entry_exit.cpp - see include/strata/image/hunyuanimage3_entry_exit.hpp.
//
// The two paths in the reference's order (records/2026-10-23-hunyuanimage3-plan, CONFORM-2026-10-24.md):
//
//   patch_embed: conv3x3(32 -> 1024) | ResBlock(1024 -> 4096, time_embed(t))           -> rows [n][4096]
//   final_layer: ResBlock(4096 -> 1024, time_embed_2(t)) | GroupNorm(1024) | swish | conv3x3(1024 -> 32)
//                                                                                     -> the velocity [32][n]
//
// a ResBlock being  x -> GroupNorm(in) -> swish -> conv3x3(in -> out) -> GroupNorm(out) -> (1 + scale) z + shift,
// swish -> conv3x3(out -> out) -> + skip(x),  scale and shift out of the swish'd timestep embedding's
// Linear(hidden -> 2 out).  The skip is a 1x1 convolution (one tap), and the patch layer's embedding is a ResBlock's
// like the exit's - the reference's GeneralResBlock around both.
//
// The numbers a step takes are the convolutions': at a 1024 x 1024 image (64 x 64 tokens) the entry and the exit
// are about a thousand GFLOP together, which the host FP32 version ran at 30 s a step.  Nothing here is tiled past
// the unfold's row chunks; tile and local-memory forms come if a measurement asks for them.
//
// Every product's operand is two FP16 buffers, the value and what its rounding to FP16 left ([2][P][C], the
// second half at + P * C), multiplied by the same weight and summed: the operand to about 22 bits.  With one FP16
// operand the paths stood at the FP16 floor (rel 2e-4 against the host FP32), which the 32 layers and the 8 steps
// grew into 5.7 dB of the picture's PSNR against the golden (B70: 30.07 dB with the floor, 35.78 with the host's
// FP32, the acceptance 30): records/2026-10-23-hunyuanimage3-plan, entry-leg6.  The weights are the file's BF16,
// which FP16 holds exactly but for the few under its normal range.
#include "strata/image/hunyuanimage3_entry_exit.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/hunyuanimage3_host.hpp"
#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace strata::image {

namespace {

namespace gpu = strata::gpu;

// The GroupNorm(32) of every ResBlock here has the host reference's eps 1e-5 (the VAE's is 1e-6).
constexpr float kGnEps = 1e-5f;
constexpr int64_t kColBytes = 32ll << 20;   // the unfolded bands a product keeps at most

std::string oom(const char* what) {
    return std::string("HunyuanImage 3.0 (GPU entry and exit): ") + what + ": " + gpu::last_error();
}

}  // namespace

HunyuanImage3EntryExit::HunyuanImage3EntryExit() = default;

HunyuanImage3EntryExit::~HunyuanImage3EntryExit() {
    release_grid();
    if (s_ != nullptr) gpu::stream_sync(s_);
    release(weights_);
}

uint64_t HunyuanImage3EntryExit::held() const { return weight_bytes_ + scratch_bytes(th_, tw_); }

uint64_t HunyuanImage3EntryExit::scratch_bytes(int64_t th, int64_t tw) const {
    if (th <= 0 || tw <= 0 || hidden_ <= 0) return 0;
    const int64_t n = th * tw;
    // the bands: nine neighbourhoods an input channel, in row chunks of at most kColBytes (the widest input is the
    // exit's first convolution, hidden channels)
    const uint64_t col = std::min<uint64_t>((uint64_t) n * 9 * (uint64_t) hidden_ * 2, (uint64_t) kColBytes);
    uint64_t b = 0;
    // the FP16 operands, each with the rounding it left (two halves)
    b += 2 * (uint64_t) n * (uint64_t) latent_ * 2;   // x16: the latent's FP16 [n][32]
    b += 2 * (uint64_t) n * (uint64_t) hidden_ * 2;   // a16s: a swish'd norm at the hidden width
    b += 2 * (uint64_t) n * 1024 * 2;                 // a16c: one at the block's inner width
    b += 2 * (uint64_t) n * 1024 * 2;                 // skip16: the entry's skip input
    b += 2 * (uint64_t) n * (uint64_t) hidden_ * 2;   // rows16: the exit's skip input
    b += col;                                         // col16
    b += (uint64_t) std::max<int64_t>(n, 2) * (uint64_t) hidden_ * 4;   // lo32: a product of the second halves
    b += 3 * (uint64_t) n * (uint64_t) hidden_ * 4;   // f32s: the entry's three hidden-wide products
    b += 3 * (uint64_t) n * 1024 * 4;                 // f32c: the exit's three
    b += (uint64_t) n * (uint64_t) latent_ * 4;       // lat32
    b += (uint64_t) n * (uint64_t) latent_ * 4;       // vel32
    b += 2 * (uint64_t) hidden_ * 2;                  // emb16
    b += 2 * (uint64_t) hidden_ * 4;                  // es32: the widest embedding projection (2 * 4096)
    b += (uint64_t) qi_group_norm_ws_floats(n) * 4;
    return b;
}

void HunyuanImage3EntryExit::release(std::vector<void*>& owned) {
    for (void* p : owned) gpu::free(p);
    owned.clear();
}

void HunyuanImage3EntryExit::release_grid() {
    release(grid_);
    n_ = th_ = tw_ = col_elems_ = 0;
    x16_ = a16s_ = a16c_ = skip16_ = rows16_ = col16_ = emb16_ = nullptr;
    for (float*& p : f32s_) p = nullptr;
    for (float*& p : f32c_) p = nullptr;
    lat32_ = vel32_ = es32_ = gn_ws_ = lo32_ = nullptr;
}

void HunyuanImage3EntryExit::drop_grid() { release_grid(); }

bool HunyuanImage3EntryExit::set_grid(int64_t th, int64_t tw, std::string& err) {
    if (dry_) {
        err = "HunyuanImage 3.0 (GPU entry and exit): a dry part cannot run";
        return false;
    }
    if (th <= 0 || tw <= 0) {
        err = "HunyuanImage 3.0 (GPU entry and exit): an empty grid";
        return false;
    }
    if (th == th_ && tw == tw_ && n_ == th * tw) return true;
    release_grid();
    const int64_t n = th * tw;
    auto dev = [&](int64_t bytes, const char* what) -> void* {
        void* p = nullptr;
        if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(bytes, 1))) throw std::runtime_error(oom(what));
        grid_.push_back(p);
        return p;
    };
    try {
        const uint64_t col = std::min<uint64_t>((uint64_t) n * 9 * (uint64_t) hidden_ * 2, (uint64_t) kColBytes);
        col_elems_ = (int64_t) (col / 2);
        x16_ = static_cast<uint16_t*>(dev(2 * n * latent_ * 2, "the latent's FP16"));
        a16s_ = static_cast<uint16_t*>(dev(2 * n * hidden_ * 2, "a hidden-wide FP16 operand"));
        a16c_ = static_cast<uint16_t*>(dev(2 * n * 1024 * 2, "an inner-wide FP16 operand"));
        skip16_ = static_cast<uint16_t*>(dev(2 * n * 1024 * 2, "the entry's skip input"));
        rows16_ = static_cast<uint16_t*>(dev(2 * n * hidden_ * 2, "the exit's skip input"));
        col16_ = static_cast<uint16_t*>(dev(col_elems_ * 2, "the unfold's bands"));
        lo32_ = static_cast<float*>(dev(std::max<int64_t>(n, 2) * hidden_ * 4, "a product of the second halves"));
        for (float*& p : f32s_) p = static_cast<float*>(dev(n * hidden_ * 4, "a hidden-wide product"));
        for (float*& p : f32c_) p = static_cast<float*>(dev(n * 1024 * 4, "an inner-wide product"));
        lat32_ = static_cast<float*>(dev(n * latent_ * 4, "the latent"));
        vel32_ = static_cast<float*>(dev(n * latent_ * 4, "the velocity"));
        emb16_ = static_cast<uint16_t*>(dev(2 * hidden_ * 2, "the timestep embedding"));
        es32_ = static_cast<float*>(dev(2 * hidden_ * 4, "the embedding projection"));
        gn_ws_ = static_cast<float*>(dev(qi_group_norm_ws_floats(n) * 4, "the norms' statistics"));
    } catch (const std::exception& e) {
        release_grid();
        err = e.what();
        return false;
    }
    n_ = n;
    th_ = th;
    tw_ = tw;
    return true;
}

// A tensor of the file's BF16 as FP16 rows on the device, `kind` saying how it lies.
void HunyuanImage3EntryExit::load_rows(const std::string& name, int64_t kind, Conv& c, std::string& err) {
    const SafeTensor* t = f_.tensors().find(name + ".weight");
    if (t == nullptr) {
        err = name + ": no such tensor";
        return;
    }
    if (t->dtype != "BF16") {
        err = name + ": " + t->dtype + ", not the BF16 these pieces are";
        return;
    }
    int64_t out = 0, in = 0;
    if (kind == 9) {
        if (t->shape.size() != 4 || t->shape[2] != 3 || t->shape[3] != 3) {
            err = name + ": not a 3x3 convolution (" + std::to_string(t->shape.size()) + " dimensions)";
            return;
        }
        out = t->shape[0], in = t->shape[1];
    } else if (kind == 1) {
        if (t->shape.size() != 4 || t->shape[2] != 1 || t->shape[3] != 1) {
            err = name + ": not a 1x1 convolution (" + std::to_string(t->shape.size()) + " dimensions)";
            return;
        }
        out = t->shape[0], in = t->shape[1];
    } else {
        if (t->shape.size() != 2) {
            err = name + ": not a 2-dimensional matrix (" + std::to_string(t->shape.size()) + " dimensions)";
            return;
        }
        out = t->shape[0], in = t->shape[1];
    }
    c.in = in;
    c.out = out;
    c.taps = kind == 9 ? 9 : 1;
    const int64_t K = c.taps * in;
    if (kind != 0) load_vec(name + ".bias", out, c.b, err);
    if (!err.empty()) return;
    if (dry_) {
        weight_bytes_ += (uint64_t) out * K * 2;
        return;
    }
    const uint8_t* src = nullptr;
    uint64_t bytes = 0;
    if (!f_.raw(name + ".weight", src, bytes, err)) return;
    const int64_t stride = c.taps * in;   // a 3x3 row is in * 9 values, a 1x1 or a matrix's in
    if (bytes < (uint64_t) out * (uint64_t) stride * 2) {
        err = name + ": " + std::to_string(bytes) + " bytes, short of its own shape";
        return;
    }
    const auto* bits = reinterpret_cast<const uint16_t*>(src);
    uint16_t* dst = nullptr;
    if (!gpu::alloc_device(&dst, (size_t) ((uint64_t) out * K * 2))) throw std::runtime_error(oom(name.c_str()));
    weights_.push_back(dst);
    weight_bytes_ += (uint64_t) out * K * 2;
    c.w = dst;
    // rows into slabs of about 16 MiB: the largest of these tensors is 302 MiB of FP16
    const int64_t slab_rows = std::max<int64_t>(1, (int64_t) ((16ll << 20) / (K * 2)));
    std::vector<uint16_t> stage((size_t) (std::min<int64_t>(slab_rows, out) * K));
    for (int64_t o0 = 0; o0 < out; o0 += slab_rows) {
        const int64_t rows = std::min(slab_rows, out - o0);
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t o = o0 + r;
            uint16_t* row = stage.data() + (size_t) (r * K);
            const uint16_t* s = bits + (uint64_t) o * (uint64_t) stride;
            if (kind == 9) {
                // the file's [c][dy][dx] into the unfold's [tap][c]: the reads run over the source
                for (int64_t ci = 0; ci < in; ++ci)
                    for (int64_t tap = 0; tap < 9; ++tap)
                        row[tap * in + ci] = strata::kernels::f16_from_f32(
                            strata::kernels::f32_from_bf16(s[ci * 9 + tap]));
            } else {
                for (int64_t ci = 0; ci < in; ++ci)
                    row[ci] = strata::kernels::f16_from_f32(strata::kernels::f32_from_bf16(s[ci]));
            }
        }
        if (!gpu::copy(dst + (uint64_t) o0 * (uint64_t) K, stage.data(), (size_t) (rows * K * 2))) {
            err = oom("a convolution's weight");
            return;
        }
    }
}

void HunyuanImage3EntryExit::load_vec(const std::string& name, int64_t n, const float*& d, std::string& err) {
    const SafeTensor* t = f_.tensors().find(name);
    if (t == nullptr) {
        err = name + ": no such tensor";
        return;
    }
    if (t->elements() != n) {
        err = name + ": " + std::to_string(t->elements()) + " values, wanted " + std::to_string(n);
        return;
    }
    if (dry_) {
        weight_bytes_ += (uint64_t) n * 4;
        return;
    }
    std::vector<float> h((size_t) n);
    if (!f_.bf16(name, h.data())) {
        err = name + ": not a BF16 vector";
        return;
    }
    float* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * 4)) throw std::runtime_error(oom(name.c_str()));
    weights_.push_back(p);
    if (!gpu::copy(p, h.data(), (size_t) n * 4)) {
        err = oom(name.c_str());
        return;
    }
    weight_bytes_ += (uint64_t) n * 4;
    d = p;
}

void HunyuanImage3EntryExit::dump(const std::string& name, const float* x, int64_t n) {
    if (dump_dir_.empty() || x == nullptr) return;
    std::vector<float> h((size_t) n);
    gpu::stream_sync(s_);   // the copy below runs on the compute queue, the kernels on the caller's stream
    if (!gpu::copy(h.data(), x, (size_t) n * 4)) return;
    std::error_code ec;
    std::filesystem::create_directories(dump_dir_, ec);
    std::FILE* fp = std::fopen((dump_dir_ + "/" + name + ".bin").c_str(), "wb");
    if (fp == nullptr) return;
    std::fwrite(h.data(), sizeof(float), h.size(), fp);
    std::fclose(fp);
}

void HunyuanImage3EntryExit::dump16(const std::string& name, const uint16_t* x, int64_t n) {
    if (dump_dir_.empty() || x == nullptr) return;
    std::vector<uint16_t> bits((size_t) n);
    gpu::stream_sync(s_);
    if (!gpu::copy(bits.data(), x, (size_t) n * 2)) return;
    std::vector<float> h((size_t) n);
    for (int64_t i = 0; i < n; ++i) h[(size_t) i] = strata::kernels::f32_from_f16(bits[(size_t) i]);
    std::error_code ec;
    std::filesystem::create_directories(dump_dir_, ec);
    std::FILE* fp = std::fopen((dump_dir_ + "/" + name + ".bin").c_str(), "wb");
    if (fp == nullptr) return;
    std::fwrite(h.data(), sizeof(float), h.size(), fp);
    std::fclose(fp);
}

bool HunyuanImage3EntryExit::open(const std::string& path, void* stream, std::string& err, bool dry) {
    s_ = stream;
    dry_ = dry;
    if (const char* dir = std::getenv("HUNYUANIMAGE3_ENTRY_DUMP"); dir != nullptr && dir[0] != '\0') dump_dir_ = dir;
    if (!f_.open(path, err)) return false;
    if (!f_.quantized()) {
        err = path + ": not a HunyuanImage 3.0 quantized file";
        return false;
    }
    try {
        if (!dry && !gemm_.init(s_, 0, err)) return false;
        load_rows("patch_embed.model.0", 9, patch_in_, err);
        if (err.empty()) load_rows("patch_embed.model.1.in_layers.2", 9, patch_res_.in, err);
        if (err.empty()) load_rows("patch_embed.model.1.out_layers.3", 9, patch_res_.out, err);
        if (err.empty()) load_rows("patch_embed.model.1.skip_connection", 1, patch_res_.skip, err);
        if (err.empty()) load_rows("final_layer.model.0.in_layers.2", 9, final_res_.in, err);
        if (err.empty()) load_rows("final_layer.model.0.out_layers.3", 9, final_res_.out, err);
        if (err.empty()) load_rows("final_layer.model.0.skip_connection", 1, final_res_.skip, err);
        if (err.empty()) load_rows("final_layer.model.1.2", 9, final_out_, err);
        if (!err.empty()) return false;
        // the embedding projections: their own rows, hidden -> 2 out
        Conv pe, fe;
        load_rows("patch_embed.model.1.emb_layers.1", 0, pe, err);
        if (err.empty()) load_rows("final_layer.model.0.emb_layers.1", 0, fe, err);
        if (!err.empty()) return false;
        patch_res_.emb = pe.w;
        patch_res_.emb_n = pe.out;
        patch_res_.emb_k = pe.in;
        final_res_.emb = fe.w;
        final_res_.emb_n = fe.out;
        final_res_.emb_k = fe.in;
        load_vec("patch_embed.model.1.emb_layers.1.bias", patch_res_.emb_n, patch_res_.emb_b, err);
        if (err.empty()) load_vec("final_layer.model.0.emb_layers.1.bias", final_res_.emb_n, final_res_.emb_b, err);
        if (!err.empty()) return false;
        // a ResBlock's two norms into the block itself (its widths are its input's and its output's)
        auto gn = [&](const std::string& p, int64_t c, const float*& w, const float*& b) {
            load_vec(p + ".weight", c, w, err);
            if (err.empty()) load_vec(p + ".bias", c, b, err);
        };
        gn("patch_embed.model.1.in_layers.0", patch_res_.in.in, patch_res_.g1, patch_res_.b1);
        if (err.empty()) gn("patch_embed.model.1.out_layers.0", patch_res_.out.out, patch_res_.g2, patch_res_.b2);
        if (err.empty()) gn("final_layer.model.0.in_layers.0", final_res_.in.in, final_res_.g1, final_res_.b1);
        if (err.empty()) gn("final_layer.model.0.out_layers.0", final_res_.out.out, final_res_.g2, final_res_.b2);
        if (err.empty()) {
            final_gn_.c = final_out_.in;
            gn("final_layer.model.1.0", final_gn_.c, final_gn_.w, final_gn_.b);
        }
        if (!err.empty()) return false;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    // the relationships the two paths assume: the latent's channels at both ends and one hidden width through
    latent_ = patch_in_.in;
    hidden_ = patch_res_.out.out;
    auto bad = [&](const std::string& why) {
        err = path + ": " + why;
        return false;
    };
    if (latent_ != 32)
        return bad("the entry's first convolution takes " + std::to_string(latent_) + " channels, not the latent's 32");
    // the buffers are the widths these two blocks have in the files: the entry's inner product and the exit's are
    // 1024 wide, everything else the hidden width
    if (patch_in_.out != 1024)
        return bad("the entry's first convolution writes " + std::to_string(patch_in_.out) + " channels, not 1024");
    if (patch_res_.in.in != patch_in_.out || patch_res_.in.out != hidden_ || patch_res_.out.in != hidden_ ||
        patch_res_.out.out != hidden_)
        return bad("the entry's ResBlock is not " + std::to_string(patch_res_.in.in) + " -> " +
                   std::to_string(patch_res_.in.out) + " -> " + std::to_string(patch_res_.out.out) + ", the hidden "
                   "width being " + std::to_string(hidden_));
    if (patch_res_.skip.in != patch_res_.in.in || patch_res_.skip.out != hidden_)
        return bad("the entry's skip is not 1x1 " + std::to_string(patch_res_.in.in) + " -> " +
                   std::to_string(hidden_));
    if (patch_res_.emb_k != hidden_ || patch_res_.emb_n != 2 * hidden_)
        return bad("the entry's embedding projection is not [2 * " + std::to_string(hidden_) + "][" +
                   std::to_string(hidden_) + "]");
    if (final_res_.in.in != hidden_ || final_res_.in.out != 1024 || final_res_.out.in != 1024 ||
        final_res_.out.out != 1024)
        return bad("the exit's ResBlock is not " + std::to_string(final_res_.in.in) + " -> " +
                   std::to_string(final_res_.in.out) + " -> " + std::to_string(final_res_.out.out));
    if (final_res_.skip.in != hidden_ || final_res_.skip.out != 1024)
        return bad("the exit's skip is not 1x1 " + std::to_string(hidden_) + " -> 1024");
    if (final_res_.emb_k != hidden_ || final_res_.emb_n != 2048)
        return bad("the exit's embedding projection is not [2048][" + std::to_string(hidden_) + "]");
    if (final_out_.in != 1024 || final_out_.out != latent_)
        return bad("the exit's last convolution is not 1024 -> " + std::to_string(latent_));
    if (final_gn_.c != final_out_.in) return bad("the exit's last GroupNorm's width is not the tensor it norms");
    return true;
}

void HunyuanImage3EntryExit::conv3x3(const Conv& c, const uint16_t* in16, int64_t th, int64_t tw, float* out,
                                     const float* bias) {
    const int64_t P = th * tw, K = 9 * c.in, rows = std::max<int64_t>(1, col_elems_ / K);
    const uint16_t* lo16 = in16 + P * c.in;
    for (int64_t p0 = 0; p0 < P; p0 += rows) {
        const int64_t r = std::min(rows, P - p0);
        hunyuanimage3_im2col16(in16, th, tw, c.in, p0, r, col16_, s_);
        gemm_.f16(col16_, c.w, out + p0 * c.out, r, c.out, K);
        hunyuanimage3_im2col16(lo16, th, tw, c.in, p0, r, col16_, s_);
        gemm_.f16(col16_, c.w, lo32_ + p0 * c.out, r, c.out, K);
    }
    hunyuanimage3_add(out, lo32_, P * c.out, s_);
    if (bias != nullptr) hunyuanimage3_bias_add32(out, bias, P, c.out, s_);
}

void HunyuanImage3EntryExit::conv1x1(const Conv& c, const uint16_t* in16, int64_t P, float* out, const float* bias) {
    gemm_.f16(in16, c.w, out, P, c.out, c.in);
    gemm_.f16(in16 + P * c.in, c.w, lo32_, P, c.out, c.in);
    hunyuanimage3_add(out, lo32_, P * c.out, s_);
    if (bias != nullptr) hunyuanimage3_bias_add32(out, bias, P, c.out, s_);
}

void HunyuanImage3EntryExit::norm(const float* x, const Gn& g, const float* scale, const float* shift, int64_t P,
                                  uint16_t* y16, const char* tag) {
    qi_group_norm16(x, P, g.c, g.w, g.b, kGnEps, true, y16, gn_ws_, s_, scale, shift, y16 + P * g.c);
    dump(stage_ + "_" + tag + "_mean", gn_ws_, 32);
    dump(stage_ + "_" + tag + "_inv", gn_ws_ + 32, 32);
    dump16(stage_ + "_" + tag + "_y16", y16, P * g.c);
}

void HunyuanImage3EntryExit::res_block(const Res& r, const float* x, const uint16_t* x16, uint16_t* a_in,
                                       uint16_t* a_out, int64_t P, int64_t th, int64_t tw, float* mid,
                                       float* out, float* skip32) {
    // the embedding's projection: hidden -> 2 out, its scale (the first half of `es32_`) and shift (the second)
    gemm_.f16(emb16_, r.emb, es32_, 1, r.emb_n, r.emb_k);
    gemm_.f16(emb16_ + r.emb_k, r.emb, lo32_, 1, r.emb_n, r.emb_k);
    hunyuanimage3_add(es32_, lo32_, r.emb_n, s_);
    hunyuanimage3_bias_add32(es32_, r.emb_b, 1, r.emb_n, s_);
    const float* scale = es32_;
    const float* shift = es32_ + r.emb_n / 2;
    dump(stage_ + "_es", es32_, r.emb_n);
    norm(x, {r.g1, r.b1, r.in.in}, nullptr, nullptr, P, a_in, "gn1");
    conv3x3(r.in, a_in, th, tw, mid, r.in.b);
    dump(stage_ + "_c2", mid, P * r.in.out);
    norm(mid, {r.g2, r.b2, r.in.out}, scale, shift, P, a_out, "gn2");
    conv3x3(r.out, a_out, th, tw, out, r.out.b);
    dump(stage_ + "_c3", out, P * r.out.out);
    conv1x1(r.skip, x16, P, skip32, r.skip.b);
    dump(stage_ + "_skip", skip32, P * r.out.out);
    hunyuanimage3_add(out, skip32, P * r.out.out, s_);
    dump(stage_ + "_out", out, P * r.out.out);
}

void HunyuanImage3EntryExit::embed16(const std::string& which, float t, std::string& err) {
    // the embedding the projection takes is the swish'd one (the host reference's `eb = emb; silu_all(eb)`)
    std::vector<float> h((size_t) hidden_);
    if (!hunyuanimage3_time_host(f_, which, t, h.data(), err)) return;
    std::vector<uint16_t> bits((size_t) (2 * hidden_));   // the value, then the rounding it left
    for (int64_t i = 0; i < hidden_; ++i) {
        const float v = h[(size_t) i], e = v / (1.0f + std::exp(-v));
        bits[(size_t) i] = strata::kernels::f16_from_f32(e);
        bits[(size_t) (hidden_ + i)] =
            strata::kernels::f16_from_f32(e - strata::kernels::f32_from_f16(bits[(size_t) i]));
    }
    if (!gpu::copy(emb16_, bits.data(), bits.size() * 2)) err = oom("the timestep embedding");
}

bool HunyuanImage3EntryExit::patch_embed(const float* latent, int64_t th, int64_t tw, float t, float* rows,
                                         std::string& err) {
    if (dry_ || n_ <= 0 || th != th_ || tw != tw_) {
        err = "HunyuanImage 3.0 (GPU entry and exit): the grid is not this part's";
        return false;
    }
    const int64_t P = n_;
    stage_ = "e";
    try {
        if (!gpu::copy(lat32_, latent, (size_t) (latent_ * P * 4))) throw std::runtime_error(oom("the latent"));
        embed16("time_embed", t, err);
        if (!err.empty()) return false;
        hunyuanimage3_grid_in16(lat32_, latent_, P, x16_, s_, x16_ + P * latent_);
        conv3x3(patch_in_, x16_, th, tw, f32c_[0], patch_in_.b);
        dump("e_c1", f32c_[0], P * 1024);
        hunyuanimage3_to16(f32c_[0], P * 1024, skip16_, s_, skip16_ + P * 1024);
        res_block(patch_res_, f32c_[0], skip16_, a16c_, a16s_, P, th, tw, f32s_[0], f32s_[1], f32s_[2]);
        // the block's rows out, as the host reference hands them to the walk
        if (!gpu::stream_sync(s_)) throw std::runtime_error(oom("the entry's drain"));
        if (!gpu::copy(rows, f32s_[1], (size_t) (P * hidden_ * 4))) throw std::runtime_error(oom("the rows"));
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool HunyuanImage3EntryExit::final_layer(const float* rows, int64_t th, int64_t tw, float t, float* vel,
                                         std::string& err) {
    if (dry_ || n_ <= 0 || th != th_ || tw != tw_) {
        err = "HunyuanImage 3.0 (GPU entry and exit): the grid is not this part's";
        return false;
    }
    const int64_t P = n_;
    stage_ = "f";
    try {
        if (!gpu::copy(f32s_[0], rows, (size_t) (P * hidden_ * 4))) throw std::runtime_error(oom("the rows"));
        embed16("time_embed_2", t, err);
        if (!err.empty()) return false;
        hunyuanimage3_to16(f32s_[0], P * hidden_, rows16_, s_, rows16_ + P * hidden_);   // the skip's operand
        res_block(final_res_, f32s_[0], rows16_, a16s_, a16c_, P, th, tw, f32c_[0], f32c_[1], f32c_[2]);
        norm(f32c_[1], final_gn_, nullptr, nullptr, P, a16c_, "gn_");
        conv3x3(final_out_, a16c_, th, tw, vel32_, final_out_.b);
        if (!gpu::stream_sync(s_)) throw std::runtime_error(oom("the exit's drain"));
        // [P][32] as the sampler's [32][P]
        std::vector<float> tmp((size_t) (P * latent_));
        if (!gpu::copy(tmp.data(), vel32_, (size_t) (P * latent_ * 4))) throw std::runtime_error(oom("the velocity"));
        for (int64_t p = 0; p < P; ++p)
            for (int64_t c = 0; c < latent_; ++c) vel[(size_t) (c * P + p)] = tmp[(size_t) (p * latent_ + c)];
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::image