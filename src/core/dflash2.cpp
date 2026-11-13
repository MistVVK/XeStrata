// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/dflash2.cpp - see include/strata/core/dflash2.hpp.  The drafter over XeStrata's GPU operations
// (strata/core/gpu.hpp): one in-order stream, one graph per context-row count, the host stages the positions and the
// rope tables (no device FP64).  The ring log rebuilds a resumed conversation's window from the logged activation
// rows with the same kernels, so the rebuilt K/V bytes are those of the original update.
#include "strata/core/dflash2.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// A number of the pack's index or of a setting, read whole: text that is not one fails, where atoi took the digits up
// to the first other character, and 0 for none.
bool read_int(const char* s, int64_t& v) {
    char* end = nullptr;
    errno = 0;
    v = std::strtoll(s, &end, 10);
    return end != s && *end == '\0' && errno == 0;
}

bool read_int(const char* s, int& v) {
    int64_t w = 0;
    if (!read_int(s, w) || w < INT_MIN || w > INT_MAX) return false;
    v = (int) w;
    return true;
}

bool read_float(const char* s, float& v) {
    char* end = nullptr;
    errno = 0;
    v = std::strtof(s, &end);
    return end != s && *end == '\0' && errno == 0;
}

// The index's comma-separated list of tap layers ("3,11,19,"; a trailing comma is allowed).
bool read_int_list(const char* s, std::vector<int>& out) {
    out.clear();
    for (const char* q = s; *q != '\0';) {
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(q, &end, 10);
        if (end == q || errno != 0 || v < INT_MIN || v > INT_MAX || (*end != ',' && *end != '\0')) return false;
        out.push_back((int) v);
        q = *end == ',' ? end + 1 : end;
    }
    return true;
}

// A tensor's line of the index: "t <name> <dtype> <rows> <cols> <offset> <bytes>".
bool read_tensor_line(const std::string& line, char (&name)[256], char (&dtype)[64], int64_t (&v)[4]) {
    char n[4][32];
    if (std::sscanf(line.c_str(), "t %255s %63s %31s %31s %31s %31s", name, dtype, n[0], n[1], n[2], n[3]) != 6)
        return false;
    for (int k = 0; k < 4; ++k)
        if (!read_int(n[k], v[k])) return false;
    return true;
}

}  // namespace

Dflash2Drafter::~Dflash2Drafter() { release(); }

void Dflash2Drafter::release() {
    if (device_ < 0) return;
    {
        const OnDevice on_device(device_);
        if (cs_) strata::gpu::stream_sync(cs_);
        for (auto& shape : exec_) for (auto& e : shape)
            if (e) { strata::gpu::graph_destroy(e); e = nullptr; }
        strata::gpu::free(blob_);
        strata::gpu::free(slab_);
        strata::gpu::free(logits_);
        if (!dhead_shared_) {
            // the subset's rows and ids are the DraftSubset's (it frees them); a raw bind's vocab is ours, a host
            // bind's the engine's
            if (dvocab_ && !subset_ && !vocab_borrowed_) strata::gpu::free(const_cast<int32_t*>(dvocab_));
        }
        subset_.reset();
        strata::gpu::free(h_logq_);
        strata::gpu::free(h_logd_);
        strata::gpu::free(h_meta_);
        strata::gpu::free(h_out_);
        if (cs_) strata::gpu::stream_destroy(cs_);
    }
    device_ = -1;
    cs_ = nullptr;
    blob_ = slab_ = nullptr;
    logits_ = nullptr;
    h_logq_ = nullptr;
    h_logd_ = nullptr;
    h_meta_ = nullptr;
    h_out_ = nullptr;
    emb_mask_ = nullptr;
    dvocab_ = nullptr;
    head_w_ = nullptr;
    vocab_borrowed_ = dhead_shared_ = bound_ = sel_q4_ = ctx_pending_ = false;
    shared_ = SharedHead{};
    embed_ = nullptr;
    host_embed_ = nullptr;
    head_fn_ = nullptr;
    tensors_.clear();
    tensors_.shrink_to_fit();
    taps_.clear();
    vram_ = 0;
    graph_taps_ = nullptr;
    have_ctx_ = 0;
    ndraft_ = kDrafts;
    block_rows_ = capacity_rows_ = kRows;
    history_ = DraftRingHistory{};
    pend_pos0_ = -1;
    pend_n_ = 0;
    log_on_ = true;
}

uint64_t Dflash2Drafter::ram_bytes() const {
    if (device_ < 0) return 0;
    uint64_t b = meta_bytes() + (uint64_t) N_ * 4 + 64 * sizeof(int32_t) + ring_log_bytes() + ring_stash_bytes();
    if (h_logq_ != nullptr) b += (uint64_t) capacity_rows_ * N_ + (uint64_t) capacity_rows_ * (N_ / 32) * 4;
    return b;
}

const Dflash2Drafter::Tensor& Dflash2Drafter::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("dflash2: tensor " + name + " is missing from the pack");
}

bool Dflash2Drafter::read_shape(const std::string& path, PackShape& shape, std::string& err) {
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (f == nullptr) { err = "dflash2: cannot open " + path; return false; }
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f.get()) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1 || hdr[2] > (64u << 20)) {
        err = "dflash2: " + path + " is not a DF2P v1 pack";
        return false;
    }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f.get()) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    shape = PackShape{};
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        int64_t v[4];
        bool ok = true;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            const std::string key = b;
            if (key == "hidden") ok = read_int(c, shape.hidden);
            else if (key == "vocab") ok = read_int(c, shape.vocab);
            else if (key == "taps") ok = read_int_list(c, shape.taps);
        } else if (read_tensor_line(line, a, b, v)) {
            shape.bytes = std::max<uint64_t>(shape.bytes, (uint64_t) v[2] + (uint64_t) v[3]);
        } else if (line.starts_with("t ")) {
            ok = false;
        }
        if (!ok) {
            err = "dflash2: " + path + " has a bad line in its index: ";
            err += line;
            return false;
        }
    }
    if (shape.taps.empty() || shape.hidden <= 0) { err = "dflash2: " + path + " has no taps in its index"; return false; }
    return true;
}

void Dflash2Drafter::set_capacity_rows(int rows) {
    if (device_ >= 0 || (rows != kRows && rows != kMaxRows))
        throw std::invalid_argument("dflash2: capacity must be 8 or 16, set before load");
    requested_capacity_rows_ = rows;
}

bool Dflash2Drafter::load(const std::string& path, int device, std::string& err) {
    device_ = device;
    capacity_rows_ = requested_capacity_rows_;
    const OnDevice on_device(device_);
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (f == nullptr) { err = "dflash2: cannot open " + path; return false; }
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f.get()) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) {
        err = "dflash2: " + path + " is not a DF2P v1 pack";
        return false;
    }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f.get()) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    const long data_off = 12 + (long) hdr[2];
    uint64_t total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        int64_t v[4];
        bool ok = true;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            const std::string key = b;
            if (key == "hidden") ok = read_int(c, N_);
            else if (key == "layers") ok = read_int(c, L_);
            else if (key == "heads") ok = read_int(c, NH_);
            else if (key == "kv_heads") ok = read_int(c, NKV_);
            else if (key == "head_dim") ok = read_int(c, HD_);
            else if (key == "inter") ok = read_int(c, I_);
            else if (key == "window") ok = read_int(c, window_);
            else if (key == "rope_theta") ok = read_float(c, theta_);
            else if (key == "eps") ok = read_float(c, eps_);
            else if (key == "conv_group") ok = read_int(c, group_);
            else if (key == "rank") ok = read_int(c, rank_);
            else if (key == "mask_token") ok = read_int(c, mask_token_);
            else if (key == "vocab") ok = read_int(c, vocab_);
            else if (key == "taps") ok = read_int_list(c, taps_);
        } else if (read_tensor_line(line, a, b, v)) {
            Tensor t;
            t.name = a; t.dtype = b; t.rows = v[0]; t.cols = v[1]; t.off = (uint64_t) v[2]; t.bytes = (uint64_t) v[3];
            tensors_.push_back(t);
            total = std::max(total, t.off + t.bytes);
        } else if (line.starts_with("t ")) {
            ok = false;
        }
        if (!ok) {
            err = "dflash2: " + path + " has a bad line in its index: ";
            err += line;
            return false;
        }
    }
    // what the kernels take (kernels/dflash2.hpp): 32 query and 8 K/V heads of 128, a rank-256 selector, products of
    // a multiple of 512 inputs, a hidden of at most 4096 (df2_conv_rms_quant), at most 8 layers (the rings), and the
    // whole vocabulary in one top-k (4096 x kDf2TopBlocks).  The three drafters (Qwen3.8-Flash-Next 2560, GLM-5.3-Flash
    // 4096, Qwen3.6-35B-A3B 2048) are all of this shape.
    const char* bad = tensors_.empty() ? "no tensors"
                    : NH_ != 32 || NKV_ != 8 || HD_ != 128 ? "not 32 query and 8 K/V heads of 128"
                    : rank_ != 256 ? "a selector rank other than 256"
                    : N_ < 512 || N_ > 4096 || N_ % 512 != 0 ? "a hidden size the kernels do not take"
                    : I_ <= 0 || I_ % 512 != 0 ? "an FFN width the kernels do not take"
                    : group_ != 16 ? "a conv group other than 16"
                    : L_ < 1 || L_ > 8 ? "more layers than the rings take"
                    : window_ + 16 > kRing ? "a window longer than the ring"
                    : taps_.empty() || (taps_.size() * (size_t) N_) % 512 != 0 ? "no taps"
                    : vocab_ > (int64_t) 4096 * strata::kernels::kDf2TopBlocks ? "a vocabulary larger than the top-k takes"
                    : nullptr;
    if (bad != nullptr) {
        err = std::string("dflash2: the pack's shape is not one this engine's kernels take: ") + bad;
        return false;
    }
    in_max_ = std::max({(int) taps_.size() * N_, I_, NH_ * HD_, N_});
    if (cs_ == nullptr) cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "dflash2: the drafter's stream"; return false; }
    if (!strata::gpu::alloc_device(&blob_, total)) {
        err = "dflash2: the pack does not fit in VRAM (" + std::to_string(total >> 20) + " MiB)";
        return false;
    }
    vram_ += total;
    uint64_t max_bytes = 0;
    for (const Tensor& t : tensors_) max_bytes = std::max(max_bytes, t.bytes);
    std::vector<uint8_t> stage(max_bytes);
    uint8_t* tmp = nullptr;
    if (!strata::gpu::alloc_device(&tmp, max_bytes)) { err = "dflash2: upload scratch"; return false; }
    struct TmpFree { uint8_t* p; ~TmpFree() { strata::gpu::free(p); } } tmp_free{tmp};
    for (const Tensor& t : tensors_) {
#if defined(_WIN32)
        _fseeki64(f.get(), data_off + (long long) t.off, SEEK_SET);
#else
        fseeko(f.get(), data_off + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(stage.data(), 1, t.bytes, f.get()) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        const bool codebook = t.name.find("_codebook") != std::string::npos;
        if (t.dtype == "q4_0" && codebook) {
            sel_q4_ = true;   // the converted pack: Q4_0 rows in block order, the select kernel dequantizes them
            if (!strata::gpu::copy(blob_ + t.off, stage.data(), t.bytes)) { err = "dflash2: codebook upload"; return false; }
        } else if (t.dtype == "q4_0") {   // ggml Q4_0 blocks -> the tiled nibble plane, then the d plane (same total bytes)
            if (!strata::gpu::copy(tmp, stage.data(), t.bytes)) { err = "dflash2: weight upload"; return false; }
            strata::kernels::df2_q4_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols / 2,
                                         (int) t.cols, t.rows * t.cols / 32, cs_);
            if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: weight tiling"; return false; }
        } else if (t.dtype == "q8_0") {
            if (!strata::gpu::copy(tmp, stage.data(), t.bytes)) { err = "dflash2: weight upload"; return false; }
            strata::kernels::df2_q8_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols,
                                         (int) t.cols, t.rows * t.cols / 32, cs_);
            if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: weight tiling"; return false; }
        } else {
            if (!strata::gpu::copy(blob_ + t.off, stage.data(), t.bytes)) { err = "dflash2: tensor upload"; return false; }
        }
    }
    std::fprintf(stderr, "xestrata dflash2: %s pack, %d layers, taps", sel_q4_ ? "Q4_0 + Q4_0 selector" : "Q4_0/Q8_0", L_);
    for (int l : taps_) std::fprintf(stderr, " %d", l);
    std::fprintf(stderr, ", window %d\n", window_);
    return alloc(err);
}

bool Dflash2Drafter::alloc(std::string& err) {
    // One device allocation for the work buffers, the rings and the rebuild scratch: the host cost of a launch grows
    // with the number of live device allocations (the Arc A-series measurement, upstream 903badd90).
    size_t slab_bytes = 0;
    uint8_t* slab = nullptr;
    auto dev = [&](auto** p, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
        const size_t off = slab_bytes;
        slab_bytes += (n * sizeof(T) + 255) & ~(size_t) 255;
        if (slab != nullptr) *p = reinterpret_cast<T*>(slab + off);
        return true;
    };
    const size_t B = (size_t) capacity_rows_;
    const size_t rb_cap = (size_t) kRing + capacity_rows_;
    auto carve = [&]() {
        bool ok = dev(&d_meta_, (size_t) meta_ints() + table_floats() + N_) && dev(&d_out_, 64) && dev(&emb_mask_, N_) &&
              dev(&xq_, strata::kernels::native_q8_1_rows_bytes(N_, capacity_rows_) + 4096) &&
              dev(&xqp_, (size_t) in_max_ * capacity_rows_ + 4096) && dev(&xdp_, (size_t) in_max_ / 32 * capacity_rows_ + 64) &&
              dev(&xqc_, (size_t) in_max_ * capacity_rows_ + 4096) && dev(&xdc_, (size_t) in_max_ / 32 * capacity_rows_ + 64) &&
              dev(&fcy_, B * N_) && dev(&kc_, B * NKV_ * HD_) && dev(&vc_, B * NKV_ * HD_) && dev(&h_, B * N_) &&
              dev(&xn_, B * N_) && dev(&dyn_, B * (size_t) (N_ / 4)) && dev(&q_, B * NH_ * HD_) && dev(&qn_, B * NH_ * HD_) &&
              dev(&k_, B * NKV_ * HD_) && dev(&v_, B * NKV_ * HD_) && dev(&kblk_, B * NKV_ * HD_) && dev(&o_, B * N_) &&
              dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
              dev(&part_, (size_t) strata::kernels::kDf2PartChunks * 8 * capacity_rows_ * 4 * 130) && dev(&fn_, B * N_) &&
              dev(&topval_, (size_t) (capacity_rows_ - 1) * 16) && dev(&toptok_, (size_t) (capacity_rows_ - 1) * 16) &&
              dev(&hsel_, (size_t) (capacity_rows_ - 1) * rank_) &&
              dev(&topscratch_, (size_t) (capacity_rows_ - 1) * strata::kernels::kDf2TopBlocks * strata::kernels::kDf2TopK * 2) &&
              dev(&tag_, (size_t) kRing);
        for (int l = 0; ok && l < L_; ++l)
            ok = dev(&kring_[l], (size_t) kRing * NKV_ * HD_) && dev(&vring_[l], (size_t) kRing * NKV_ * HD_);
        ok = ok && dev(&rb_q_, rb_cap * N_) && dev(&rb_d_, rb_cap * (N_ / 32)) && dev(&rb_pos_, rb_cap);
        return ok;
    };
    carve();
    const size_t slab_total = slab_bytes;
    if (!strata::gpu::alloc_device(&slab_, slab_total)) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    slab = slab_;
    slab_bytes = 0;
    carve();
    vram_ += slab_total;
    rt_ctx_ = reinterpret_cast<float*>(d_meta_ + meta_ints());
    rt_blk_ = rt_ctx_ + B * 128;
    cur_ = Lineage{(size_t) N_, {}, {}};
    if (!strata::gpu::alloc_host(&h_meta_, meta_bytes() + (size_t) N_ * 4) || !strata::gpu::alloc_host(&h_out_, 64 * 4)) {
        err = "dflash2: host staging";
        return false;
    }
    std::memset(h_meta_, 0, meta_bytes() + (size_t) N_ * 4);
    std::memset(h_out_, 0, 64 * sizeof(int32_t));
    if (!strata::gpu::memset_async(d_meta_, 0, meta_bytes() + (size_t) N_ * 4, cs_) ||
        !strata::gpu::memset_async(tag_, 0xff, (size_t) kRing * 8, cs_)) {   // -1: no slot holds a context row yet
        err = "dflash2: ring init";
        return false;
    }
    for (int64_t& t : slot_pos_) t = -1;
    if (const char* e = strata::env_get("DF2_RING_LOG"); e != nullptr && e[0] == '0') log_on_ = false;
    int log_mib = 0;
    if (const char* e = strata::env_get("DF2_LOG_MIB"); e != nullptr && read_int(e, log_mib))
        stash_budget_ = (uint64_t) std::max(0, log_mib) << 20;
    if (log_on_) {
        if (!strata::gpu::alloc_host(&h_logq_, (size_t) capacity_rows_ * N_) ||
            !strata::gpu::alloc_host(&h_logd_, (size_t) capacity_rows_ * (N_ / 32) * 4)) {
            strata::gpu::free(h_logq_); h_logq_ = nullptr;
            strata::gpu::free(h_logd_); h_logd_ = nullptr;
            log_on_ = false;
            std::fprintf(stderr, "xestrata dflash2: the ring log's staging does not fit in host memory: the log is off\n");
        }
    }
    if (!strata::gpu::memset_async(xqp_, 0, (size_t) in_max_ * capacity_rows_ + 4096, cs_) ||
        !strata::gpu::memset_async(xdp_, 0, ((size_t) in_max_ / 32 * capacity_rows_ + 64) * 4, cs_) ||
        !strata::gpu::memset_async(xqc_, 0, (size_t) in_max_ * capacity_rows_ + 4096, cs_) ||
        !strata::gpu::memset_async(xdc_, 0, ((size_t) in_max_ / 32 * capacity_rows_ + 64) * 4, cs_)) {
        err = "dflash2: activation init";
        return false;
    }
    for (int l = 0; l < L_; ++l) {
        if (!strata::gpu::memset_async(kring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4, cs_) ||
            !strata::gpu::memset_async(vring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4, cs_)) {
            err = "dflash2: ring init";
            return false;
        }
    }
    return strata::gpu::stream_sync(cs_);
}

uint64_t Dflash2Drafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, const std::string& vocab_file) const {
    int64_t dv = DraftSubset::file_tokens(vocab_file);
    if (dv <= 0 || dv > n_vocab) dv = n_vocab;
    return (uint64_t) dv * head_row_bytes + (uint64_t) (requested_capacity_rows_ - 1) * (uint64_t) dv * 4 + (uint64_t) dv * 4;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, HeadFn head, int head_type, int64_t head_rows,
                              const std::vector<int32_t>& vocab, std::string& err) {
    const OnDevice on_device(device_);
    embed_ = std::move(embed);
    head_fn_ = std::move(head);
    head_type_ = head_type;
    n_dvocab_ = head_rows;
    if (!vocab.empty()) {
        n_dvocab_ = (int64_t) vocab.size();
        int32_t* dv = nullptr;
        if (!strata::gpu::alloc_device(&dv, vocab.size() * 4)) { err = "dflash2: draft vocabulary"; return false; }
        if (!strata::gpu::copy(dv, vocab.data(), vocab.size() * 4)) { strata::gpu::free(dv); err = "dflash2: draft vocabulary"; return false; }
        dvocab_ = dv;
        vram_ += vocab.size() * 4;
    }
    if ((n_dvocab_ + strata::kernels::kDf2TopBlocks - 1) / strata::kernels::kDf2TopBlocks > 4096) {
        err = "dflash2: the head has too many rows for the top-k";
        return false;
    }
    if (logits_ == nullptr && !strata::gpu::alloc_device(&logits_, (size_t) (capacity_rows_ - 1) * (size_t) n_dvocab_ * 4)) {
        err = "dflash2: the draft logits do not fit";
        return false;
    }
    vram_ += (uint64_t) (capacity_rows_ - 1) * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

bool Dflash2Drafter::bind_host(HostEmbedFn embed, HeadFn head, int64_t head_rows, const int32_t* vocab_dev,
                               std::string& err) {
    const OnDevice on_device(device_);
    std::vector<float> mask((size_t) N_);
    if (!embed(mask_token_, mask.data())) { err = "dflash2: the mask token is outside the model's embedding"; return false; }
    if (!strata::gpu::copy(emb_mask_, mask.data(), mask.size() * 4)) { err = "dflash2: the mask row"; return false; }
    host_embed_ = std::move(embed);
    head_fn_ = std::move(head);
    n_dvocab_ = head_rows;
    dvocab_ = vocab_dev;
    vocab_borrowed_ = vocab_dev != nullptr;
    if ((n_dvocab_ + strata::kernels::kDf2TopBlocks - 1) / strata::kernels::kDf2TopBlocks > 4096) {
        err = "dflash2: the head has too many rows for the top-k";
        return false;
    }
    if (logits_ == nullptr && !strata::gpu::alloc_device(&logits_, (size_t) (capacity_rows_ - 1) * (size_t) n_dvocab_ * 4)) {
        err = "dflash2: the draft logits do not fit";
        return false;
    }
    vram_ += (uint64_t) (capacity_rows_ - 1) * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

bool Dflash2Drafter::bind(const WeightTable& wt, const ModelGeometry& g, const NativeHead* head,
                          const std::string& vocab_file, std::string& err) {
    const OnDevice on_device(device_);
    if (head == nullptr || !head->loaded()) { err = "dflash2: the drafter needs the native head (--native)"; return false; }
    const int64_t row_bytes = (int64_t) head->row_bytes();
    const int64_t head_rows = row_bytes > 0 ? (int64_t) (head->weight_bytes() / (uint64_t) row_bytes) : 0;
    const WeightTable* wtp = &wt;
    const ModelGeometry* gp = &g;
    EmbedFn embed = [wtp, gp](const int32_t* tok, int n, float* out, void* s, std::string& e) {
        return ::strata::core::embed_rows(*wtp, *gp, tok, 0, n, out, s, e);
    };
    HeadFn head_fn = [this](const uint8_t* rows, const uint8_t* xq, const float*, int n, float* y, void* s) {
        strata::kernels::native_mmvq(head_type_, rows, xq, y, N_, (int) n_dvocab_, n, s);
    };
    if (shared_.rows != nullptr && shared_.n > 0) {   // the MTP drafter's gathered head: the same bytes, no second copy
        head_w_ = shared_.rows;
        dvocab_ = shared_.vocab_dev;
        n_dvocab_ = shared_.n;
        dhead_shared_ = true;
        head_type_ = shared_.type;
        embed_ = std::move(embed);
        head_fn_ = std::move(head_fn);
        if (logits_ == nullptr && !strata::gpu::alloc_device(&logits_, (size_t) (capacity_rows_ - 1) * (size_t) n_dvocab_ * 4)) {
            err = "dflash2: the draft logits do not fit";
            return false;
        }
        vram_ += (uint64_t) (capacity_rows_ - 1) * (uint64_t) n_dvocab_ * 4;
        bound_ = true;
        std::fprintf(stderr, "xestrata dflash2: %lld-token draft head shared with the MTP, drafter %.0f MiB of VRAM\n",
                     (long long) n_dvocab_, (double) vram_ / 1048576.0);
        return true;
    }
    if (DraftSubset::file_tokens(vocab_file) > 0) {
        auto sub = std::make_shared<DraftSubset>();
        if (!sub->load(vocab_file, head_rows, head->weights(), head->row_bytes(), cs_, "xestrata dflash2", err))
            return false;
        if (sub->size() > 0) {
            subset_ = sub;
            head_w_ = sub->rows();
            vram_ += sub->bytes();
        }
    }
    if (head_w_ == nullptr) head_w_ = static_cast<const uint8_t*>(head->weights());
    if (!bind_raw(std::move(embed), std::move(head_fn), head->type(), subset_ ? subset_->size() : head_rows, {}, err))
        return false;
    if (subset_) dvocab_ = subset_->ids_device();   // the subset's own device ids (it owns them)
    std::fprintf(stderr, "xestrata dflash2: %lld-token draft head (%.1f MiB), drafter %.0f MiB of VRAM\n",
                 (long long) n_dvocab_, (double) ((int64_t) n_dvocab_ * row_bytes) / 1048576.0, (double) vram_ / 1048576.0);
    return true;
}

bool Dflash2Drafter::gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols,
                          const uint8_t* xq, const float* xd, std::string& err) {
    try {
        strata::kernels::Df2Mat mats[3];
        int n = 0, n_in = 0;
        auto o = outs.begin();
        for (const char* name : names) {
            const Tensor& t = find(name);
            mats[n] = strata::kernels::Df2Mat{blob_ + t.off, *o++, (int) t.rows, t.dtype == "q4_0"};
            if (n == 0) n_in = (int) t.cols;
            ++n;
        }
        strata::kernels::df2_gemm(mats, n, xq, xd, n_in, ncols, cs_);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash2: ") + e.what();
        return false;
    }
}

void Dflash2Drafter::stage(int n_ctx, int64_t pos0, int64_t block_s, int32_t anchor, bool block) {
    for (int i = 0; i < capacity_rows_; ++i) {
        h_meta_[i] = (int32_t) (pos0 + std::min(i, std::max(n_ctx, 1) - 1));
        h_meta_[block_pos_offset() + i] = (int32_t) (block_s + i);
        h_meta_[token_offset() + i] = i == 0 ? anchor : mask_token_;
    }
    h_meta_[16] = anchor;
    h_meta_[17] = (int32_t) block_s;
    h_meta_[18] = history_.epoch();
    float* tables = reinterpret_cast<float*>(h_meta_ + meta_ints());
    strata::kernels::df2_rope_table_cpu(h_meta_, capacity_rows_, theta_, tables);
    strata::kernels::df2_rope_table_cpu(h_meta_ + block_pos_offset(), capacity_rows_, theta_, tables + (ptrdiff_t) capacity_rows_ * 128);
    size_t bytes = meta_bytes();
    if (host_embed_ && block) {   // bind_host: the anchor's row behind the tables, in the same copy
        float* row = reinterpret_cast<float*>(h_meta_ + meta_ints()) + table_floats();
        if (!host_embed_(anchor, row)) std::memset(row, 0, (size_t) N_ * 4);   // an id outside the table: a poor draft, never a wrong token
        bytes += (size_t) N_ * 4;
    }
    strata::gpu::copy_async(d_meta_, h_meta_, bytes, cs_, "dflash2 meta");
}

bool Dflash2Drafter::record_context(const float* taps, int n, std::string& err) {
    using namespace strata::kernels;
    try {
        const int tap_n = (int) taps_.size() * N_;
        df2_quant_rows(taps, tap_n, n, xqc_, xdc_, cs_);
        if (!gemm({"fc.weight"}, {fcy_}, n, xqc_, xdc_, err)) return false;
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_, xqc_, xdc_, cs_);
        return record_kv(n, xqc_, xdc_, d_meta_ + 0, err);
    } catch (const std::exception& e) {
        err = std::string("dflash2: ") + e.what();
        return false;
    }
}

// The part of a context update after the activations are quantized: per layer the keys and values (projection,
// k-norm, rope, ring write).  Also what a ring rebuild runs over logged rows.
bool Dflash2Drafter::record_kv(int n, const uint8_t* xq, const float* xd, const int32_t* pos, std::string& err) {
    using namespace strata::kernels;
    try {
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            if (!gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, xq, xd, err)) return false;
            df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), pos, rt_ctx_, kring_[l], vring_[l], kRing, n, NKV_,
                        eps_, cs_, tag_, d_meta_ + 18);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash2: ") + e.what();
        return false;
    }
}

bool Dflash2Drafter::record_block(std::string& err) {
    using namespace strata::kernels;
    try {
    const int B = block_rows_;
    const int drafts = std::min(ndraft_, B - 1);
    const int dyn_w = N_ / 4;   // a row's two convolutions' kernels: [prepare, finish] x [2 taps] x N / 16 groups
    if (!embed_rows(err)) return false;
    df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_, xqp_, xdp_, cs_);
    for (int l = 0; l < L_; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        const float* base_a = f32(p + "attention_conv.base_kernel");
        const float* base_m = f32(p + "mlp_conv.base_kernel");
        const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
        const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
        const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
        const std::string op = p + "self_attn.o_proj.weight";
        const std::string dp = p + "mlp.down_proj.weight";
        if (!gemm({akp.c_str()}, {dyn_}, B, xqp_, xdp_, err)) return false;
        df2_conv_quant(xn_, dyn_, dyn_w, base_a, B, N_, group_, xqp_, xdp_, cs_);
        if (!gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, xqp_, xdp_, err)) return false;
        df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_,
                         kblk_, B, NH_, NKV_, eps_, cs_);
        df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs_, tag_,
                      d_meta_ + 18, B);
        if (!gemm({op.c_str()}, {o_}, B, xqp_, xdp_, err)) return false;
        df2_conv_rms_quant(o_, dyn_ + dyn_w / 2, dyn_w, base_a + (ptrdiff_t) 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_,
                           B, N_, group_, eps_, xqp_, xdp_, cs_);
        if (!gemm({mkp.c_str()}, {dyn_}, B, xqp_, xdp_, err)) return false;
        df2_conv_quant(xn_, dyn_, dyn_w, base_m, B, N_, group_, xqp_, xdp_, cs_);
        if (!gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, xqp_, xdp_, err)) return false;
        df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs_);
        if (!gemm({dp.c_str()}, {mlp_}, B, xqp_, xdp_, err)) return false;
        const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight"
                                                  : std::string("norm.weight");
        df2_conv_rms_quant(mlp_, dyn_ + dyn_w / 2, dyn_w, base_m + (ptrdiff_t) 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_,
                           group_, eps_, xqp_, xdp_, cs_);
    }
    native_quantize_q8_1_rows(fn_ + N_, xq_, N_, drafts, cs_);   // rows 1..ndraft, the head's format (fp32: fn_ + N_)
    df2_quant_rows(fn_ + N_, N_, drafts, xqp_, xdp_, cs_);   // and the selector projection's
    if (!gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, drafts, xqp_, xdp_, err)) return false;
    for (int r = 0; r < drafts; r += 8)
        head_fn_(head_w_, xq_ + strata::kernels::native_q8_1_rows_bytes(N_, r), fn_ + (size_t) (r + 1) * N_,
                 std::min(8, drafts - r), logits_ + (size_t) r * n_dvocab_, cs_);
    df2_topk(logits_, (int) n_dvocab_, dvocab_, drafts, topscratch_, topval_, toptok_, cs_);
    df2_select(hsel_, topval_, toptok_,
               reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
               reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off), rank_,
               drafts, d_meta_ + 16, d_out_, reinterpret_cast<float*>(d_out_) + 16, cs_, sel_q4_);
    return true;
    } catch (const std::exception& e) {
        err = std::string("dflash2: ") + e.what();
        return false;
    }
}

bool Dflash2Drafter::embed_rows(std::string& err) {
    if (embed_) return embed_(d_meta_ + token_offset(), block_rows_, h_, cs_, err);
    const float* anchor = reinterpret_cast<const float*>(d_meta_ + meta_ints()) + table_floats();
    strata::kernels::df2_embed_rows(anchor, emb_mask_, h_, N_, block_rows_, cs_);
    return true;
}

bool Dflash2Drafter::capture(int n, const float* taps, std::string& err) {
    auto& graph = exec_[block_rows_ == kRows ? 0 : 1][n];
    if (graph) return true;
    if (!strata::gpu::begin_capture(cs_)) { err = "dflash2: begin capture"; return false; }
    bool ok = record_context(taps, n, err);
    if (ok) ok = record_block(err);
    if (!ok) { strata::gpu::abandon_capture(cs_); return false; }
    if (!strata::gpu::end_capture(cs_, &graph)) { err = "dflash2: end capture"; return false; }
    return true;
}

void Dflash2Drafter::set_max_drafts(int n) {
    n = std::max(1, std::min(n, capacity_rows_ - 1));
    if (n == ndraft_) return;
    const OnDevice on_device(device_);
    if (cs_) strata::gpu::stream_sync(cs_);
    for (auto& shape : exec_) for (auto& e : shape)
        if (e) { strata::gpu::graph_destroy(e); e = nullptr; }
    graph_taps_ = nullptr;
    ndraft_ = n;
    if (ndraft_ < kMaxDrafts) block_rows_ = kRows;
}

void Dflash2Drafter::set_block_rows(int rows) {
    if (rows != kRows && rows != kMaxRows) throw std::invalid_argument("dflash2: block rows must be 8 or 16");
    if (rows == kMaxRows && (capacity_rows_ < kMaxRows || ndraft_ < kMaxDrafts)) throw std::invalid_argument("dflash2: 16 rows need 15 drafts enabled");
    block_rows_ = rows;
}

bool Dflash2Drafter::prepare(const float* taps, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    // Graph recording cannot time a candidate.  Warm the Q4 shapes with zero activations before capturing so the
    // measured choice also reaches every captured window.  The scratch is private to this drafter and is overwritten
    // by the first real window.  One tensor per shape is enough; the choice is per device and shape.
    const char* pair = strata::env_get("DF2_GEMM_PAIR16");
    if (!pair || !*pair || std::strcmp(pair, "auto") == 0) {
        if (!strata::gpu::memset_async(xqc_, 0, (size_t) 5 * N_ * kRows, cs_) ||
            !strata::gpu::memset_async(xdc_, 0, (size_t) 5 * N_ / 32 * kRows * sizeof(float), cs_)) {
            err = "dflash2: GEMM timing activation init";
            return false;
        }
        std::set<std::pair<int64_t, int64_t>> shapes;
        try {
            for (const Tensor& t : tensors_) {
                if (t.dtype != "q4_0" || !t.name.ends_with(".weight") || t.rows <= 0 || t.rows > I_ ||
                    t.cols <= 0 || t.cols > (int64_t) 5 * N_ || t.rows % 16 || t.cols % 512 ||
                    !shapes.emplace(t.rows, t.cols).second) continue;
                strata::kernels::Df2Mat mat{blob_ + t.off, gate_, (int) t.rows, true};
                for (int n = 1; n <= kRows; ++n)
                    strata::kernels::df2_gemm(&mat, 1, xqc_, xdc_, (int) t.cols, n, cs_);
            }
        } catch (const std::exception& e) {
            err = std::string("dflash2: GEMM timing: ") + e.what();
            return false;
        }
    }
    const int saved_rows = block_rows_;
    const int max_context_rows = ndraft_ > kDrafts ? kMaxRows : kRows;
    for (int rows : {kRows, kMaxRows}) {
        if (rows == kMaxRows && ndraft_ <= kDrafts) break;
        block_rows_ = rows;
        for (int n = 1; n <= max_context_rows; ++n) {
            if (!capture(n, taps, err)) { block_rows_ = saved_rows; return false; }
        }
    }
    block_rows_ = saved_rows;
    if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: prepare"; return false; }
    graph_taps_ = taps;
    ms_capture = ms_since(t0);
    return true;
}

// ---- ring log --------------------------------------------------------------------------------------------
uint64_t Dflash2Drafter::ring_log_bytes() const { return cur_.bytes(); }
uint64_t Dflash2Drafter::ring_stash_bytes() const {
    uint64_t b = 0;
    for (const Lineage& l : stash_) b += l.bytes();
    return b;
}

void Dflash2Drafter::reset() {
    have_ctx_ = 0;
    cur_ = Lineage{(size_t) N_, {}, {}};
    stash_.clear();
    invalidate_ring();
}

void Dflash2Drafter::invalidate_ring() {
    ++epoch_;
    ring_hi_ = 0;
    for (int64_t& t : slot_pos_) t = -1;
}

void Dflash2Drafter::stash_push(Lineage&& l) {
    if (!log_on_ || stash_budget_ == 0 || l.hist.size() < 128 || l.chunks.empty()) return;
    stash_.push_back(std::move(l));
    ++ring_stats.stash_stores;
    uint64_t total = ring_stash_bytes();
    while (!stash_.empty() && total > stash_budget_) {
        total -= std::min(total, stash_.front().bytes());
        stash_.pop_front();
        ++ring_stats.stash_evictions;
    }
}

void Dflash2Drafter::log_stage(int64_t pos0, int n) {
    if (!log_on_) return;
    pend_pos0_ = pos0;
    pend_n_ = n;
    for (int i = 0; i < n; ++i) slot_pos_[(size_t) ((pos0 + i) % kRing)] = ((int64_t) epoch_ << 32) | (pos0 + i);
}

// The update's quantized rows are still in xqc_ / xdc_ when its stream has run (nothing writes them afterwards):
// copy them to the host behind the update.
void Dflash2Drafter::log_fetch(int n) {
    if (!log_on_ || pend_pos0_ < 0) return;
    strata::gpu::copy_async(h_logq_, xqc_, (size_t) n * N_, cs_, "dflash2 ring log");
    strata::gpu::copy_async(h_logd_, xdc_, (size_t) n * (N_ / 32) * 4, cs_, "dflash2 ring log");
}

void Dflash2Drafter::log_flush() {
    if (!log_on_ || pend_pos0_ < 0) return;
    for (int i = 0; i < pend_n_; ++i) {
        const int64_t pos = pend_pos0_ + i;
        LogChunk& ch = cur_.writable((size_t) (pos / kLogChunk));
        const size_t r = (size_t) (pos % kLogChunk);
        const size_t W = (size_t) N_, D = (size_t) N_ / 32;
        std::memcpy(&ch.q[r * W], h_logq_ + (size_t) i * W, W);
        std::memcpy(&ch.d[r * D], h_logd_ + (size_t) i * D, D * sizeof(float));
        ch.ok[r] = 1;
    }
    pend_pos0_ = -1;
}

bool Dflash2Drafter::rebuild(int64_t lo, int64_t hi, int64_t* rebuilt, int64_t* missing, std::string& err) {
    *rebuilt = 0;
    *missing = 0;
    std::vector<int32_t> pos;
    for (int64_t p = lo; p < hi; ++p) {
        if (slot_pos_[(size_t) (p % kRing)] == (((int64_t) epoch_ << 32) | p)) continue;   // the ring already holds this row
        if (cur_.logged(p)) pos.push_back((int32_t) p);
        else ++*missing;
    }
    if (pos.empty()) return true;
    const OnDevice on_device(device_);
    std::vector<float> table((size_t) rows * 128);
    strata::kernels::df2_rope_table_cpu(positions, rows, theta_, table.data());
    h_meta_[18] = epoch;
    if (!strata::gpu::copy_async(d_meta_, h_meta_, (size_t) meta_ints() * 4, cs_, "dflash2 rebuild") ||
        !strata::gpu::copy_async(rb_q_, q, (size_t) rows * N_, cs_, "dflash2 rebuild") ||
        !strata::gpu::copy_async(rb_d_, scales, (size_t) rows * (N_ / 32) * 4, cs_, "dflash2 rebuild") ||
        !strata::gpu::copy_async(rb_pos_, positions, (size_t) rows * sizeof(int32_t), cs_, "dflash2 rebuild") ||
        !strata::gpu::copy_async(rt_ctx_, table.data(), table.size() * sizeof(float), cs_, "dflash2 rebuild")) {
        strata::gpu::stream_sync(cs_);   // earlier copies can still reference these host buffers
        err = "dflash2: ring rebuild upload";
        return false;
    }
    bool ok = true;
    for (size_t r0 = 0; ok && r0 < rows; r0 += kRows) {
        const int m = (int) std::min<size_t>(kRows, rows - r0);
        // this batch's table before its kernels (an in-order stream)
        ok = strata::gpu::copy_async(rt_ctx_, tab.data() + r0 * 128, (size_t) m * 128 * 4, cs_, "dflash2 rebuild") &&
             record_kv(m, rb_q_ + r0 * W, rb_d_ + r0 * D, rb_pos_ + r0, err);
    }
    if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: ring rebuild"; return false; }   // the host vectors die with this scope
    if (!ok) return false;
    for (int32_t p : pos) slot_pos_[(size_t) (p % kRing)] = ((int64_t) epoch_ << 32) | p;
    *rebuilt = (int64_t) rows;
    return true;
}

bool Dflash2Drafter::begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold) {
    resume = std::max<int64_t>(0, std::min(resume, n));
    if (cold) *cold = false;
    ++ring_stats.requests;
    last_rebuilt = last_missing = 0;
    last_ms = 0;
    last_swapped = false;
    std::string err;
    const Clock::time_point t0 = Clock::now();
    auto lcp_of = [&](const std::vector<int32_t>& h) {
        int64_t l = 0;
        const int64_t lim = std::min<int64_t>((int64_t) h.size(), resume);
        while (l < lim && h[(size_t) l] == ids[l]) ++l;
        return l;
    };
    if (!log_on_) {   // the pre-log rule: a history that does not cover the resumed prefix starts the ring over
        const int64_t lcp = lcp_of(cur_.hist);
        const bool stale = ring_hi_ > lcp && lcp < resume;
        if (stale) invalidate_ring();
        if (cold) *cold = stale;
        cur_.hist.assign(ids, ids + resume);
        return true;
    }
    bool covered = lcp_of(cur_.hist) >= resume;
    bool swapped = false;
    if (!covered) {   // a parked lineage may be the one this request resumes
        for (size_t i = stash_.size(); i-- > 0;) {
            if (lcp_of(stash_[i].hist) < resume) continue;
            Lineage got = std::move(stash_[i]);
            stash_.erase(stash_.begin() + (std::ptrdiff_t) i);
            stash_push(std::move(cur_));
            cur_ = std::move(got);
            invalidate_ring();
            covered = swapped = true;
            ++ring_stats.stash_hits;
            break;
        }
    }
    if (!covered) {   // another conversation: the live lineage is parked, this one starts empty
        stash_push(std::move(cur_));
        cur_ = Lineage{(size_t) N_, {}, {}};
        invalidate_ring();
        ++ring_stats.cold_starts;
    } else if ((int64_t) cur_.hist.size() >= resume + 128) {
        // the rows past `resume` are about to be rewritten (a rewound or branching request): keep the old branch
        Lineage copy = cur_;
        stash_push(std::move(copy));
    }
    if ((int64_t) cur_.hist.size() > resume) { cur_.hist.resize((size_t) resume); cur_.truncate(resume); }
    if (!covered) cur_.hist.assign(ids, ids + resume);
    // the window the first draft reads, below `resume` (the prompt's own taps cover [resume, n))
    int64_t rebuilt = 0, missing = 0;
    const int64_t lo = std::max<int64_t>(0, n - (window_ - 1)), hi = resume;
    if (hi > lo) {
        if (!rebuild(lo, hi, &rebuilt, &missing, err)) {
            std::fprintf(stderr, "xestrata dflash2: %s: the ring starts over for this request\n", err.c_str());
            invalidate_ring();
            missing = hi - lo;
            rebuilt = 0;
        }
    }
    if (strata::env_get("DF2_RING_STATS") != nullptr)
        std::fprintf(stderr, "xestrata dflash2 ring: request %lld resume %lld of %lld: live log %.1f MiB (%lld positions), "
                     "parked %zu lineages %.1f MiB, rebuilt %lld rows (missing %lld) in %.2f ms%s\n",
                     (long long) ring_stats.requests, (long long) resume, (long long) n, (double) cur_.bytes() / 1048576.0,
                     (long long) cur_.hist.size(), stash_.size(), (double) ring_stash_bytes() / 1048576.0,
                     (long long) rebuilt, (long long) missing, ms_since(t0), swapped ? " (swapped in)" : "");
    ring_stats.rows_rebuilt += rebuilt;
    ring_stats.rows_missing += missing;
    if (rebuilt > 0) ++ring_stats.rebuilds;
    const double ms = ms_since(t0);
    if (rebuilt > 0) ring_stats.ms_rebuild += ms;
    if (swapped) ring_stats.ms_swap += ms;
    if (cold) *cold = missing > 0;
    last_rebuilt = rebuilt;
    last_missing = missing;
    last_ms = ms;
    last_swapped = swapped;
    return true;
}

void Dflash2Drafter::note_rows(const int32_t* toks, int64_t n, int64_t pos0) {
    if (pos0 < 0) return;
    if ((int64_t) cur_.hist.size() > pos0) { cur_.hist.resize((size_t) pos0); if (log_on_) cur_.truncate(pos0); }
    if ((int64_t) cur_.hist.size() < pos0) return;   // a gap: the history stops being a prefix (the next begin_request starts over)
    cur_.hist.insert(cur_.hist.end(), toks, toks + n);
}

bool Dflash2Drafter::debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err) {
    const OnDevice on_device(device_);
    const size_t per = (size_t) kRing * NKV_ * HD_;
    kv.resize((size_t) L_ * 2 * per);
    tags.resize((size_t) kRing);
    if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: debug ring"; return false; }
    for (int l = 0; l < L_; ++l) {
        if (!strata::gpu::copy(kv.data() + ((size_t) l * 2) * per, kring_[l], per * 4) ||
            !strata::gpu::copy(kv.data() + ((size_t) l * 2 + 1) * per, vring_[l], per * 4)) {
            err = "dflash2: debug ring";
            return false;
        }
    }
    if (!strata::gpu::copy(tags.data(), tag_, (size_t) kRing * 8)) { err = "dflash2: debug ring"; return false; }
    return true;
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    if (ctx_pending_ && !context_wait(err)) return false;
    const int tap_n = (int) taps_.size() * N_;
    const Clock::time_point t0 = Clock::now();
    for (int64_t r0 = 0; r0 < n; r0 += kRows) {
        const int m = (int) std::min<int64_t>(kRows, n - r0);
        stage(m, pos0 + r0, pos0 + r0 + m, 0, false);
        log_stage(pos0 + r0, m);
        if (!record_context(taps + (size_t) r0 * (size_t) tap_n, m, err)) return false;
        log_fetch(m);
        if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: context"; return false; }
        log_flush();
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    ms_ctx += ms_since(t0);
    return true;
}

bool Dflash2Drafter::context_async(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    if (n < 1 || n > capacity_rows_) { err = "dflash2: context_async rows exceed capacity"; return false; }
    if (ctx_pending_ && !context_wait(err)) return false;
    stage((int) n, pos0, pos0 + n, 0, false);
    log_stage(pos0, (int) n);
    if (!record_context(taps, (int) n, err)) return false;
    log_fetch((int) n);
    ctx_pending_ = true;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::context_wait(std::string& err) {
    if (!ctx_pending_) return true;
    const OnDevice on_device(device_);
    if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: context wait"; return false; }
    log_flush();
    ctx_pending_ = false;
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs,
                           std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (n_new < 1 || n_new > capacity_rows_) { err = "dflash2: draft rows out of range"; return false; }
    const OnDevice on_device(device_);
    if (ctx_pending_ && !context_wait(err)) return false;
    const Clock::time_point t0 = Clock::now();
    if (graph_taps_ != nullptr && graph_taps_ != taps) { err = "dflash2: the taps buffer moved after capture"; return false; }
    auto& graph = exec_[block_rows_ == kRows ? 0 : 1][n_new];
    if (!graph) {
        if (!capture(n_new, taps, err)) return false;
        graph_taps_ = taps;
    }
    const int64_t s = pos0 + n_new;
    stage(n_new, pos0, s, anchor);
    log_stage(pos0, n_new);
    if (!strata::gpu::graph_launch(graph, cs_)) { err = "dflash2: graph launch"; return false; }
    if (!strata::gpu::copy_async(h_out_, d_out_, 64 * sizeof(int32_t), cs_, "dflash2 picks")) { err = "dflash2: picks"; return false; }
    log_fetch(n_new);
    if (!strata::gpu::stream_sync(cs_)) { err = "dflash2: draft"; return false; }
    log_flush();
    for (int i = 0; i < block_rows_ - 1; ++i) {
        const bool made = i < ndraft_;
        drafts[i] = made ? h_out_[i] : 0;
        if (probs) probs[i] = made ? reinterpret_cast<const float*>(h_out_)[16 + i] : 0.0f;
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, s);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n_new);
    ms_draft += ms_since(t0);
    ++rounds;
    if (block_rows_ == kRows) ++rounds_short; else ++rounds_wide;
    return true;
}

}  // namespace strata::core