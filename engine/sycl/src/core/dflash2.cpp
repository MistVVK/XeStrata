// sycl/src/core/dflash2.cpp - the DFlash2 drafter on an Intel GPU.  Port of src/core/dflash2.cpp; see sycl/include/strata/core/dflash2.hpp for the differences.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/dflash2.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <stdexcept>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kMetaInts = 32;
constexpr int kTableFloats = 16 * 128;   // 16 rows x [cos 64, sin 64]
constexpr size_t kMetaBytes = (size_t) (kMetaInts + kTableFloats) * 4;

bool finish_capture(sycl::queue* cs, bool ok, dpct::experimental::command_graph_exec_ptr& exec, std::string& err) {
    dpct::experimental::command_graph_ptr graph = nullptr;
    int ce = 0;
    try {
        ce = DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs, &graph));
        if (!ok) { delete graph; return false; }
        if (ce != 0) { delete graph; err = "dflash2: end capture"; return false; }
        exec = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
        delete graph;
    } catch (const std::exception& e) {
        delete graph;
        err = std::string("dflash2: capture: ") + e.what();
        return false;
    }
    return true;
}

}  // namespace

Dflash2Drafter::~Dflash2Drafter() try {
    if (device_ < 0) return;
    const OnDevice on_device(device_);
    if (cs_) cs_->wait();
    for (auto& e : exec_)
        if (e) delete e;
    auto fr = [&](void* p) { if (p) sycl::free(p, *cs_); };
    fr(blob_); fr(slab_);   // every work buffer, the rings and the rebuild scratch live in the slab
    fr(logits_); fr(dvocab_); fr(dhead_);
    if (h_logq_) sycl::free(h_logq_, *cs_);
    if (h_logd_) sycl::free(h_logd_, *cs_);
    if (h_meta_) strata::host_free_polled(h_meta_, *cs_);
    if (h_out_) strata::host_free_polled(h_out_, *cs_);
    if (cs_) dpct::get_current_device().destroy_queue(cs_);
} catch (...) {
}

const Dflash2Drafter::Tensor& Dflash2Drafter::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("dflash2: tensor " + name + " is missing from the pack");
}

bool Dflash2Drafter::load(const std::string& path, int device, std::string& err) try {
    device_ = device;
    const OnDevice on_device(device_);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    const long data_off = 12 + (long) hdr[2];
    uint64_t total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1, v2, v3, v4;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            const std::string key = b;
            if (key == "hidden") N_ = std::atoi(c);
            else if (key == "layers") L_ = std::atoi(c);
            else if (key == "heads") NH_ = std::atoi(c);
            else if (key == "kv_heads") NKV_ = std::atoi(c);
            else if (key == "head_dim") HD_ = std::atoi(c);
            else if (key == "inter") I_ = std::atoi(c);
            else if (key == "window") window_ = std::atoi(c);
            else if (key == "rope_theta") theta_ = (float) std::atof(c);
            else if (key == "eps") eps_ = (float) std::atof(c);
            else if (key == "conv_group") group_ = std::atoi(c);
            else if (key == "rank") rank_ = std::atoi(c);
            else if (key == "mask_token") mask_token_ = std::atoi(c);
            else if (key == "vocab") vocab_ = std::atoll(c);
            else if (key == "taps") {
                taps_.clear();
                for (const char* q = c; *q;) { taps_.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            Tensor t;
            t.name = a; t.dtype = b; t.rows = v1; t.cols = v2; t.off = (uint64_t) v3; t.bytes = (uint64_t) v4;
            tensors_.push_back(t);
            total = std::max(total, t.off + t.bytes);
        }
    }
    if (tensors_.empty() || NH_ != 32 || NKV_ != 8 || HD_ != 128 || rank_ != 256 || N_ != 2560 || L_ > 8 || window_ + 16 > kRing) {
        err = "dflash2: the pack's shape is not the Flash-Next drafter this engine was written for";
        return false;
    }
    if (cs_ == nullptr) cs_ = dpct::get_current_device().create_queue(true);
    blob_ = sycl::malloc_device<uint8_t>(total, *cs_);
    if (blob_ == nullptr) { err = "dflash2: the pack does not fit in VRAM (" + std::to_string(total >> 20) + " MiB)"; return false; }
    vram_ += total;
    uint64_t max_bytes = 0;
    for (const Tensor& t : tensors_) max_bytes = std::max(max_bytes, t.bytes);
    std::vector<uint8_t> stage(max_bytes);
    uint8_t* tmp = sycl::malloc_device<uint8_t>(max_bytes, *cs_);
    if (tmp == nullptr) { err = "dflash2: upload scratch"; return false; }
    struct TmpFree { uint8_t* p; sycl::queue* q; ~TmpFree() { sycl::free(p, *q); } } tmp_free{tmp, cs_};
    for (const Tensor& t : tensors_) {
#if defined(_WIN32)
        _fseeki64(f, data_off + (long long) t.off, SEEK_SET);
#else
        fseeko(f, data_off + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(stage.data(), 1, t.bytes, f) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        if (t.dtype == "q4_0" && t.name.find("_codebook") != std::string::npos) {
            // The pack with the selector as Q4_0 rows (tools/dflash2_pack.py --q4_0-selector; strata_control/drafting.py CONVERTED_NAME) is for engines that print
            // "selector codebooks: Q4_0 rows" (CUDA, HIP, Metal). This build's select kernel reads bf16 rows only, so it says so instead of reading garbage; the app
            // never hands it that pack (it looks for that text in the engine program and this file does not have it).
            err = "dflash2: this build reads bf16 selector codebooks only; " + path + " has them as Q4_0 (use the pack without --q4_0-selector)";
            return false;
        }
        if (t.dtype == "q4_0") {   // GGML Q4_0 blocks -> the tiled nibble plane at the tensor's offset, then the d plane (same total bytes)
            cs_->memcpy(tmp, stage.data(), t.bytes).wait();
            kernels::df2_q4_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols / 2, (int) t.cols, t.rows * t.cols / 32, cs_);
            cs_->wait();
        } else if (t.dtype == "q8_0") {
            err = "dflash2: this build runs the Q4_0 pack only (tools/dflash2_pack.py --q4_0); " + path + " holds Q8_0 matrices";
            return false;
        } else {
            cs_->memcpy(blob_ + t.off, stage.data(), t.bytes).wait();
        }
    }
    if (const char* e = std::getenv("STRATA_DF2_TIMING"); e && e[0] == '1') timing_ = true;
    return alloc(err);
} catch (const std::exception& e) {
    err = std::string("dflash2: ") + e.what();
    return false;
}

bool Dflash2Drafter::alloc(std::string& err) try {
    // One device allocation for all of the drafter's work buffers, rings and the rebuild scratch: on the Arc A-series (i915) the host cost of every
    // kernel launch grows with the number of live device allocations (12 us with none, 131 us with 300), and the eager prompt path launches a lot.
    // A first pass adds the sizes up, the second hands out 256-byte aligned pieces.
    size_t slab_bytes = 0;
    uint8_t* slab = nullptr;
    auto dev = [&](auto** p, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
        const size_t off = slab_bytes;
        slab_bytes += (n * sizeof(T) + 255) & ~(size_t) 255;
        if (slab != nullptr) *p = reinterpret_cast<T*>(slab + off);
        return true;
    };
    const size_t B = kRows;
    const size_t rb_cap = (size_t) kRing + kRows;
    auto carve = [&]() {
        bool ok = dev(&d_meta_, (size_t) kMetaInts + kTableFloats) && dev(&d_out_, 64) && dev(&xq_, kernels::native_q8_1_bytes(N_, kRows) + 4096) &&
              dev(&xqp_, (size_t) 5 * N_ * 8 + 4096) && dev(&xdp_, (size_t) 5 * N_ / 32 * 8 + 64) && dev(&xqc_, (size_t) 5 * N_ * 8 + 4096) && dev(&xdc_, (size_t) 5 * N_ / 32 * 8 + 64) &&
              dev(&fcy_, B * N_) && dev(&kc_, B * NKV_ * HD_) && dev(&vc_, B * NKV_ * HD_) && dev(&h_, B * N_) && dev(&xn_, B * N_) &&
              dev(&dyn_, B * 640) && dev(&q_, B * NH_ * HD_) && dev(&qn_, B * NH_ * HD_) && dev(&k_, B * NKV_ * HD_) && dev(&v_, B * NKV_ * HD_) &&
              dev(&kblk_, B * NKV_ * HD_) && dev(&o_, B * N_) && dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
              dev(&part_, (size_t) kernels::kDf2PartChunks * 8 * 32 * 130) && dev(&fn_, B * N_) && dev(&topval_, kDrafts * 16) && dev(&toptok_, kDrafts * 16) &&
              dev(&hsel_, kDrafts * rank_) && dev(&topscratch_, (size_t) kDrafts * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2) && dev(&tag_, (size_t) kRing);
    for (int l = 0; ok && l < L_; ++l) ok = dev(&kring_[l], (size_t) kRing * NKV_ * HD_) && dev(&vring_[l], (size_t) kRing * NKV_ * HD_);
        ok = ok && dev(&rb_q_, rb_cap * 2560) && dev(&rb_d_, rb_cap * 80) && dev(&rb_pos_, rb_cap);
        return ok;
    };
    carve();
    const size_t slab_total = slab_bytes;
    slab_ = sycl::malloc_device<uint8_t>(slab_total, *cs_);
    if (slab_ == nullptr) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    slab = slab_;
    slab_bytes = 0;
    const bool ok = carve();
    vram_ += slab_total;
    rb_bytes_ = rb_cap * (2560 + 80 * sizeof(float) + sizeof(int32_t));
    if (!ok) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    h_meta_ = (int32_t*) strata::host_malloc_polled(kMetaBytes, *cs_);
    h_out_ = (int32_t*) strata::host_malloc_polled(64 * 4, *cs_);
    if (h_meta_ == nullptr || h_out_ == nullptr) { err = "dflash2: host staging"; return false; }
    std::memset(h_meta_, 0, kMetaBytes);
    std::memset(h_out_, 0, 64 * 4);
    rt_ctx_ = reinterpret_cast<float*>(d_meta_ + kMetaInts);
    rt_blk_ = rt_ctx_ + B * 128;
    cs_->memset(d_meta_, 0, kMetaBytes);
    cs_->memset(tag_, 0xff, (size_t) kRing * 8);   // -1: no slot holds a context row yet
    for (int64_t& t : slot_pos_) t = -1;
    if (const char* e = std::getenv("STRATA_DF2_RING_LOG"); e && e[0] == '0') log_on_ = false;
    if (const char* e = std::getenv("STRATA_DF2_LOG_MIB"); e && e[0]) stash_budget_ = (uint64_t) std::max(0, std::atoi(e)) << 20;
    if (log_on_) {
        h_logq_ = (int8_t*) sycl::malloc_host((size_t) kRows * N_, *cs_);
        h_logd_ = (float*) sycl::malloc_host((size_t) kRows * (N_ / 32) * 4, *cs_);
        if (h_logq_ == nullptr || h_logd_ == nullptr) {
            if (h_logq_) sycl::free(h_logq_, *cs_);
            if (h_logd_) sycl::free(h_logd_, *cs_);
            h_logq_ = nullptr; h_logd_ = nullptr; log_on_ = false;
            std::fprintf(stderr, "strata dflash2: the ring log's staging does not fit in host memory: the log is off\n");
        }
    }
    cs_->memset(xqp_, 0, (size_t) 5 * N_ * 8 + 4096);
    cs_->memset(xdp_, 0, ((size_t) 5 * N_ / 32 * 8 + 64) * 4);
    cs_->memset(xqc_, 0, (size_t) 5 * N_ * 8 + 4096);
    cs_->memset(xdc_, 0, ((size_t) 5 * N_ / 32 * 8 + 64) * 4);
    for (int l = 0; l < L_; ++l) { cs_->memset(kring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4); cs_->memset(vring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4); }
    cs_->wait();
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: ") + e.what();
    return false;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err) try {
    const OnDevice on_device(device_);
    embed_ = std::move(embed);
    head_w_ = head_w;
    head_type_ = head_type;
    n_dvocab_ = head_rows;
    if (!vocab.empty()) {
        n_dvocab_ = (int64_t) vocab.size();
        dvocab_ = sycl::malloc_device<int32_t>(vocab.size(), *cs_);
        if (dvocab_ == nullptr) { err = "dflash2: draft vocabulary"; return false; }
        vram_ += vocab.size() * 4;
        cs_->memcpy(dvocab_, vocab.data(), vocab.size() * 4).wait();
    }
    if ((n_dvocab_ + kernels::kDf2TopBlocks - 1) / kernels::kDf2TopBlocks > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    if (!logits_) logits_ = sycl::malloc_device<float>((size_t) kDrafts * (size_t) n_dvocab_, *cs_);
    if (logits_ == nullptr) { err = "dflash2: the draft logits do not fit"; return false; }
    vram_ += (uint64_t) kDrafts * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: ") + e.what();
    return false;
}

bool Dflash2Drafter::bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err) try {
    const OnDevice on_device(device_);
    if (head == nullptr || !head->loaded()) { err = "dflash2: the drafter needs the native head (--native)"; return false; }
    const WeightRef* we = wt.find("token_embd.weight");
    if (we == nullptr) { err = "dflash2: token_embd.weight is missing"; return false; }
    EmbedFn embed = [we](const int32_t* tok, int n, float* out, sycl::queue* cs) {
        if (const NativeEmbed* ne = native_embed()) {
            ne->gather_dev(tok, n, out, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            kernels::embedding_gather_dev(codes, scales, offsets, tok, n, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                          (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), out, cs);
        }
    };
    std::vector<int32_t> vocab;
    if (!vocab_file.empty()) {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            std::fseek(vf, 0, SEEK_END);
            const long size = std::ftell(vf);
            std::fseek(vf, 0, SEEK_SET);
            if (size >= 4 && size % 4 == 0) {
                vocab.resize((size_t) size / 4);
                if (std::fread(vocab.data(), 4, vocab.size(), vf) != vocab.size()) vocab.clear();
            }
            std::fclose(vf);
        }
    }
    const void* hw = head->weights();
    const int64_t row_bytes = (int64_t) head->row_bytes();
    int64_t head_rows = row_bytes > 0 ? (int64_t) (head->weight_bytes() / (uint64_t) row_bytes) : 0;
    if (!vocab.empty()) {
        int32_t* dv = sycl::malloc_device<int32_t>(vocab.size(), *cs_);
        dhead_ = sycl::malloc_device<uint8_t>((size_t) ((int64_t) vocab.size() * row_bytes), *cs_);
        if (dv == nullptr || dhead_ == nullptr) {
            if (dv) sycl::free(dv, *cs_);
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        cs_->memcpy(dv, vocab.data(), vocab.size() * 4).wait();
        kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), dhead_, cs_);
        cs_->wait();
        sycl::free(dv, *cs_);
        vram_ += (uint64_t) ((int64_t) vocab.size() * row_bytes);
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    }
    if (!bind_raw(embed, hw, head->type(), head_rows, vocab, err)) return false;
    std::fprintf(stderr, "strata dflash2: %lld-token draft head (%.1f MiB), drafter %.0f MiB of VRAM (Q4_0 weights)\n", (long long) n_dvocab_,
                 (double) (n_dvocab_ * row_bytes) / 1048576.0, (double) vram_ / 1048576.0);
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: ") + e.what();
    return false;
}

void Dflash2Drafter::gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, const uint8_t* xq, const float* xd) {
    kernels::Df2Mat mats[3];
    int n = 0, n_in = 0;
    auto o = outs.begin();
    for (const char* name : names) {
        const Tensor& t = find(name);
        mats[n] = kernels::Df2Mat{blob_ + t.off, *o++, (int) t.rows, t.dtype == "q4_0"};
        if (n == 0) n_in = (int) t.cols;
        ++n;
    }
    kernels::df2_gemm(mats, n, xq, xd, n_in, ncols, cs_);
}

void Dflash2Drafter::stage(int n_ctx, int64_t pos0, int block_n, int64_t block_s, int32_t anchor) {
    for (int i = 0; i < 8; ++i) {
        h_meta_[i] = (int32_t) (pos0 + std::min(i, std::max(n_ctx, 1) - 1));
        h_meta_[8 + i] = (int32_t) (block_s + i);
    }
    h_meta_[16] = anchor;
    h_meta_[17] = (int32_t) block_s;
    h_meta_[18] = epoch_;
    h_meta_[24] = anchor;
    for (int i = 1; i < 8; ++i) h_meta_[24 + i] = mask_token_;
    (void) block_n;
    kernels::df2_rope_table_cpu(h_meta_, 16, theta_, reinterpret_cast<float*>(h_meta_ + kMetaInts));
    cs_->memcpy(d_meta_, h_meta_, kMetaBytes);
}

bool Dflash2Drafter::record_context(const float* taps, int n, std::string& err) {
    using namespace kernels;
    try {
        const int tap_n = (int) taps_.size() * N_;
        df2_quant_rows(taps, tap_n, n, xqc_, xdc_, cs_);
        gemm({"fc.weight"}, {fcy_}, n, xqc_, xdc_);
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_, xqc_, xdc_, cs_);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return record_kv(n, xqc_, xdc_, d_meta_ + 0, err);
}

// The part of a context update after the activations are quantized: per layer the keys and values (projection, k-norm, rope, ring write).  Also what a ring
// rebuild runs over logged rows: `xq` / `xd` are then the log's codes and scales, `pos` the rows' positions (device), rt_ctx_ their rope table.
bool Dflash2Drafter::record_kv(int n, const uint8_t* xq, const float* xd, const int32_t* pos, std::string& err) {
    using namespace kernels;
    try {
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, xq, xd);
            df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), pos, rt_ctx_, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs_, tag_, d_meta_ + 18);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block(std::string& err) {
    using namespace kernels;
    try {
        const int B = kRows;
        embed_(d_meta_ + 24, B, h_, cs_);
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
            gemm({akp.c_str()}, {dyn_}, B, xqp_, xdp_);
            df2_conv_quant(xn_, dyn_, 640, base_a, B, N_, group_, xqp_, xdp_, cs_);
            gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, xqp_, xdp_);
            df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_, B, NH_, NKV_, eps_, cs_);
            df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs_, tag_, d_meta_ + 18);
            gemm({op.c_str()}, {o_}, B, xqp_, xdp_);
            df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, B, N_, group_, eps_, xqp_, xdp_, cs_);
            gemm({mkp.c_str()}, {dyn_}, B, xqp_, xdp_);
            df2_conv_quant(xn_, dyn_, 640, base_m, B, N_, group_, xqp_, xdp_, cs_);
            gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, xqp_, xdp_);
            df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs_);
            gemm({dp.c_str()}, {mlp_}, B, xqp_, xdp_);
            // the next layer's input norm, or after the last layer the final norm (-> fn_, the rows the head and the selector read)
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_, group_, eps_, xqp_, xdp_, cs_);
        }
        native_quantize_q8_1(fn_ + N_, xq_, N_, ndraft_, cs_);   // rows 1..ndraft, the head's format
        df2_quant_rows(fn_ + N_, N_, ndraft_, xqp_, xdp_, cs_);   // and the selector projection's
        gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, ndraft_, xqp_, xdp_);
        native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, ndraft_, cs_);
        df2_topk(logits_, (int) n_dvocab_, dvocab_, ndraft_, topscratch_, topval_, toptok_, cs_);
        df2_select(hsel_, topval_, toptok_, reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                   reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off), rank_, ndraft_, d_meta_ + 16, d_out_,
                   reinterpret_cast<float*>(d_out_) + 16, cs_);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::capture(int n, const float* taps, std::string& err) {
    if (exec_[n]) return true;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) { err = "dflash2: begin capture"; return false; }
    bool ok = record_context(taps, n, err);
    if (ok) ok = record_block(err);
    return finish_capture(cs_, ok, exec_[n], err);
}

void Dflash2Drafter::set_max_drafts(int n) {
    n = std::max(1, std::min(n, (int) kDrafts));
    if (n == ndraft_) return;
    const OnDevice on_device(device_);
    if (cs_) cs_->wait();
    for (auto& e : exec_)
        if (e) { delete e; e = nullptr; }
    graph_taps_ = nullptr;
    ndraft_ = n;
}

bool Dflash2Drafter::prepare(const float* taps, std::string& err) try {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (int n = 1; n <= kRows; ++n)
        if (!capture(n, taps, err)) return false;
    cs_->wait();
    graph_taps_ = taps;
    ms_capture = ms_since(t0);
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: ") + e.what();
    return false;
}

// ---- ring log -----------------------------------------------------------------------------------------------------------------------------------------
// The ring holds the last 2064 positions of the lineage it was filled for.  Resuming a prompt below the highest position written (a repeated or rewound prefix,
// a parked conversation brought back) leaves the part of the first draft's window [n - 2047, resume) either overwritten by the later positions (the slots are
// position % 2064) or never written for this lineage.  The K/V rows are a function of the quantized hidden rows (codes and block scales after hidden_norm) alone,
// so those rows are logged as the updates make them (2.9 KB a position), per lineage, and a rebuild runs record_kv over them: the same kernels on the same bytes,
// so the rebuilt K/V rows are bitwise those of the original update.  (Port of the CUDA drafter's ring log, DF2-NV's f-0291-df2-ring.)
bool Dflash2Drafter::Lineage::logged(int64_t pos) const {
    const size_t c = (size_t) (pos / kLogChunk);
    return pos >= 0 && c < chunks.size() && chunks[c] && chunks[c]->ok[(size_t) (pos % kLogChunk)] != 0;
}

uint64_t Dflash2Drafter::Lineage::bytes() const {
    uint64_t b = (uint64_t) hist.capacity() * 4;
    for (const auto& c : chunks)
        if (c) b += c->q.size() + c->d.size() * 4 + c->ok.size();
    return b;
}

Dflash2Drafter::LogChunk& Dflash2Drafter::Lineage::writable(size_t chunk) {
    if (chunks.size() <= chunk) chunks.resize(chunk + 1);
    if (!chunks[chunk]) {
        auto c = std::make_shared<LogChunk>();
        c->q.assign((size_t) kLogChunk * 2560, 0);
        c->d.assign((size_t) kLogChunk * 80, 0.0f);
        c->ok.assign((size_t) kLogChunk, 0);
        chunks[chunk] = std::move(c);
    } else if (chunks[chunk].use_count() > 1) {
        chunks[chunk] = std::make_shared<LogChunk>(*chunks[chunk]);   // copy on write: a stashed copy keeps the old rows
    }
    return *chunks[chunk];
}

void Dflash2Drafter::Lineage::truncate(int64_t keep) {
    keep = std::max<int64_t>(0, keep);
    const size_t c0 = (size_t) (keep / kLogChunk);
    for (size_t c = c0 + 1; c < chunks.size(); ++c) chunks[c].reset();
    if (chunks.size() > c0 + 1) chunks.resize(c0 + 1);
    if (c0 < chunks.size() && chunks[c0]) {
        if (keep % kLogChunk == 0) {
            chunks[c0].reset();
        } else {
            LogChunk& ch = writable(c0);
            for (int i = (int) (keep % kLogChunk); i < kLogChunk; ++i) ch.ok[(size_t) i] = 0;
        }
    }
}

uint64_t Dflash2Drafter::ring_log_bytes() const { return cur_.bytes(); }
uint64_t Dflash2Drafter::ring_stash_bytes() const {
    uint64_t b = 0;
    for (const Lineage& l : stash_) b += l.bytes();
    return b;
}

void Dflash2Drafter::reset() {
    have_ctx_ = 0;
    cur_ = Lineage{};
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

// The update's quantized rows are still in xqc_ / xdc_ when its queue has run (nothing writes them afterwards): copy them to the host behind the update.
void Dflash2Drafter::log_fetch(int n) {
    if (!log_on_ || pend_pos0_ < 0) return;
    cs_->memcpy(h_logq_, xqc_, (size_t) n * N_);
    cs_->memcpy(h_logd_, xdc_, (size_t) n * (N_ / 32) * 4);
}

void Dflash2Drafter::log_flush() {
    if (!log_on_ || pend_pos0_ < 0) return;
    for (int i = 0; i < pend_n_; ++i) {
        const int64_t pos = pend_pos0_ + i;
        LogChunk& ch = cur_.writable((size_t) (pos / kLogChunk));
        const size_t r = (size_t) (pos % kLogChunk);
        std::memcpy(&ch.q[r * 2560], h_logq_ + (size_t) i * 2560, 2560);
        std::memcpy(&ch.d[r * 80], h_logd_ + (size_t) i * 80, 80 * sizeof(float));
        ch.ok[r] = 1;
    }
    pend_pos0_ = -1;
}

bool Dflash2Drafter::rebuild(int64_t lo, int64_t hi, int64_t* rebuilt, int64_t* missing, std::string& err) try {
    *rebuilt = 0;
    *missing = 0;
    std::vector<int32_t> pos;
    for (int64_t p = lo; p < hi; ++p) {
        if (slot_pos_[(size_t) (p % kRing)] == (((int64_t) epoch_ << 32) | p)) continue;   // the ring already holds this row
        if (cur_.logged(p)) pos.push_back((int32_t) p);
        else {
            if (++*missing <= 8 && std::getenv("STRATA_DF2_RING_STATS") != nullptr) std::fprintf(stderr, "strata dflash2 ring: position %lld was never logged (hist %zu)\n", (long long) p, cur_.hist.size());
        }
    }
    if (pos.empty()) return true;
    const OnDevice on_device(device_);
    const size_t rows = pos.size();
    const size_t cap = (size_t) kRing;   // a window is at most 2048 rows
    std::vector<int8_t> q(rows * 2560);
    std::vector<float> d(rows * 80);
    std::vector<float> tab(rows * 128);   // the rope tables of the rows, two floats [64] each, computed on the host (no device FP64)
    for (size_t i = 0; i < rows; ++i) {
        const int64_t p = pos[i];
        const LogChunk& ch = *cur_.chunks[(size_t) (p / kLogChunk)];
        const size_t r = (size_t) (p % kLogChunk);
        std::memcpy(&q[i * 2560], &ch.q[r * 2560], 2560);
        std::memcpy(&d[i * 80], &ch.d[r * 80], 80 * sizeof(float));
    }
    for (size_t r0 = 0; r0 < rows; r0 += kRows)
        kernels::df2_rope_table_cpu(pos.data() + r0, (int) std::min<size_t>(kRows, rows - r0), theta_, tab.data() + r0 * 128);
    h_meta_[18] = epoch_;
    cs_->memcpy(d_meta_, h_meta_, kMetaInts * 4);
    cs_->memcpy(rb_q_, q.data(), q.size());
    cs_->memcpy(rb_d_, d.data(), d.size() * sizeof(float));
    cs_->memcpy(rb_pos_, pos.data(), rows * sizeof(int32_t));
    bool ok = true;
    for (size_t r0 = 0; ok && r0 < rows; r0 += kRows) {
        const int m = (int) std::min<size_t>(kRows, rows - r0);
        cs_->memcpy(rt_ctx_, tab.data() + r0 * 128, (size_t) m * 128 * 4);   // this batch's table before its kernels (an in-order queue)
        ok = record_kv(m, rb_q_ + r0 * 2560, rb_d_ + r0 * 80, rb_pos_ + r0, err);
    }
    cs_->wait();   // (the host vectors die with this scope)
    if (!ok) return false;
    for (int32_t p : pos) slot_pos_[(size_t) (p % kRing)] = ((int64_t) epoch_ << 32) | p;
    *rebuilt = (int64_t) rows;
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: ring rebuild: ") + e.what();
    return false;
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
        if (stale) { ++epoch_; ring_hi_ = 0; }
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
        cur_ = Lineage{};
        invalidate_ring();
        ++ring_stats.cold_starts;
    } else if ((int64_t) cur_.hist.size() >= resume + 128) {
        // the rows past `resume` are about to be rewritten (a rewound or branching request): keep the old branch as it is, sharing the log's chunks
        Lineage copy;
        copy.hist = cur_.hist;
        copy.chunks = cur_.chunks;
        stash_push(std::move(copy));
    }
    if ((int64_t) cur_.hist.size() > resume) { cur_.hist.resize((size_t) resume); cur_.truncate(resume); }
    if (!covered) cur_.hist.assign(ids, ids + resume);
    // the window the first draft reads, below `resume` (the prompt's own taps cover [resume, n)): every slot that does not hold its row is rebuilt from the log
    int64_t rebuilt = 0, missing = 0;
    const int64_t lo = std::max<int64_t>(0, n - (window_ - 1)), hi = resume;
    if (hi > lo) {
        if (!rebuild(lo, hi, &rebuilt, &missing, err)) {
            std::fprintf(stderr, "strata dflash2: %s: the ring starts over for this request\n", err.c_str());
            invalidate_ring();
            missing = hi - lo;
            rebuilt = 0;
        }
    }
    static const bool verbose = std::getenv("STRATA_DF2_RING_STATS") != nullptr;
    if (verbose)
        std::fprintf(stderr, "strata dflash2 ring: request %lld resume %lld of %lld: live log %.1f MiB (%lld positions), parked %zu lineages %.1f MiB, rebuilt %lld rows (missing %lld) in %.2f ms%s\n",
                     (long long) ring_stats.requests, (long long) resume, (long long) n, (double) cur_.bytes() / 1048576.0, (long long) cur_.hist.size(), stash_.size(),
                     (double) ring_stash_bytes() / 1048576.0, (long long) rebuilt, (long long) missing, ms_since(t0), swapped ? " (swapped in)" : "");
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
    if ((int64_t) cur_.hist.size() < pos0) return;   // a gap: the history stops being a prefix, so it stays where it is (the next begin_request starts over)
    cur_.hist.insert(cur_.hist.end(), toks, toks + n);
}

bool Dflash2Drafter::debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err) try {
    const OnDevice on_device(device_);
    const size_t per = (size_t) kRing * NKV_ * HD_;
    kv.resize((size_t) L_ * 2 * per);
    tags.resize((size_t) kRing);
    cs_->wait();
    for (int l = 0; l < L_; ++l) {
        cs_->memcpy(kv.data() + ((size_t) l * 2) * per, kring_[l], per * 4);
        cs_->memcpy(kv.data() + ((size_t) l * 2 + 1) * per, vring_[l], per * 4);
    }
    cs_->memcpy(tags.data(), tag_, (size_t) kRing * 8);
    cs_->wait();
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: debug ring: ") + e.what();
    return false;
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) try {
    const OnDevice on_device(device_);
    const int tap_n = (int) taps_.size() * N_;
    for (int64_t r0 = 0; r0 < n; r0 += kRows) {
        const int m = (int) std::min<int64_t>(kRows, n - r0);
        stage(m, pos0 + r0, 0, pos0 + r0 + m, 0);
        log_stage(pos0 + r0, m);
        if (!record_context(taps + (size_t) r0 * (size_t) tap_n, m, err)) return false;
        log_fetch(m);
        cs_->wait();
        log_flush();
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2: ") + e.what();
    return false;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) try {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (n_new < 1 || n_new > kRows) { err = "dflash2: draft rows out of range"; return false; }
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    if (graph_taps_ != nullptr && graph_taps_ != taps) { err = "dflash2: the taps buffer moved after capture"; return false; }
    if (!exec_[n_new]) {
        if (!capture(n_new, taps, err)) return false;
        graph_taps_ = taps;
    }
    const int64_t s = pos0 + n_new;
    stage(n_new, pos0, kRows, s, anchor);
    log_stage(pos0, n_new);
    cs_->ext_oneapi_graph(*exec_[n_new]);
    cs_->memcpy(h_out_, d_out_, 64 * 4);
    log_fetch(n_new);
    cs_->wait();
    log_flush();
    for (int i = 0; i < kDrafts; ++i) {
        const bool made = i < ndraft_;
        drafts[i] = made ? h_out_[i] : 0;
        if (probs) probs[i] = made ? reinterpret_cast<const float*>(h_out_)[16 + i] : 0.0f;
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, s);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n_new);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2 draft: ") + e.what();
    return false;
}

}  // namespace strata::core
