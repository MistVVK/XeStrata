// src/core/dflash2.cpp - see include/strata/core/dflash2.hpp.
#include "strata/core/dflash2.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <atomic>
#include <initializer_list>
#include <stdexcept>

namespace strata::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

constexpr int GGML_Q8_0 = 8;

}  // namespace

Dflash2Drafter::~Dflash2Drafter() {
    if (device_ < 0) return;
    prof_report();
    prof_free();
    const OnDevice on_device(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    for (cudaGraphExec_t& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    auto fr = [](void* p) { if (p) cudaFree(p); };
    if (h_logq_) cudaFreeHost(h_logq_);
    if (h_logd_) cudaFreeHost(h_logd_);
    fr(rb_q_); fr(rb_d_); fr(rb_pos_);
    if (!blob_shared_) fr(blob_);
    fr(landing_); if (dpend_) cudaFreeHost(dpend_); fr(tag_); fr(d_meta_); fr(xq_); fr(xil_); fr(xqp_); fr(xdp_); fr(xqc_); fr(xdc_); fr(rt_ctx_); fr(rope_inv_);
    for (int l = 0; l < 8; ++l) { fr(kring_[l]); fr(vring_[l]); }
    fr(fcy_); fr(hctx_); fr(kc_); fr(vc_); fr(h_); fr(xn_); fr(xc_); fr(dyn_); fr(q_); fr(qn_); fr(k_); fr(v_); fr(kblk_); fr(ao_); fr(o_);
    fr(gate_); fr(up_); fr(mlp_); fr(part_); fr(fn_); fr(logits_); fr(topval_); fr(hsel_); fr(toptok_); fr(topscratch_);
    if (!dhead_shared_ && !blob_shared_) { fr(dvocab_); fr(dhead_); }
    if (h_meta_) cudaFreeHost(h_meta_);
    if (h_out_) cudaFreeHost(h_out_);
    if (cs_) cudaStreamDestroy(cs_);
    if (cs2_) cudaStreamDestroy(cs2_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    for (cudaEvent_t e : ev_ctx_) if (e) cudaEventDestroy(e);
    if (ev_a_) cudaEventDestroy(ev_a_);
    if (ev_b_) cudaEventDestroy(ev_b_);
}

// STRATA_DF2_PROFILE=1 (measurement only): the round runs without its graph and an event follows every stage; the per-stage averages print at exit
struct Dflash2Drafter::Prof {
    std::vector<cudaEvent_t> ev;
    std::vector<std::string> name;
    std::vector<std::pair<std::string, double>> total;
    int64_t rounds = 0;
};

void Dflash2Drafter::prof_free() {
    delete prof_;
    prof_ = nullptr;
}

// STRATA_DF2_SKIP=attn,head,ctx,mlp,qkvo (measurement only): leaves those parts out of the round, so the graph's GPU time without them can be read
static bool skip_part(const char* what) {
    static const std::string list = [] { const char* e = std::getenv("STRATA_DF2_SKIP"); return std::string(e ? e : ""); }();
    return !list.empty() && list.find(what) != std::string::npos;
}

void Dflash2Drafter::stamp(const char* name) {
    if (!prof_) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, cs_);
    prof_->ev.push_back(e);
    prof_->name.push_back(name);
}

void Dflash2Drafter::prof_collect() {
    if (!prof_ || prof_->ev.size() < 2) return;
    cudaStreamSynchronize(cs_);
    for (size_t i = 1; i < prof_->ev.size(); ++i) {
        float ms = 0;
        cudaEventElapsedTime(&ms, prof_->ev[i - 1], prof_->ev[i]);
        auto it = std::find_if(prof_->total.begin(), prof_->total.end(), [&](const auto& t) { return t.first == prof_->name[i]; });
        if (it == prof_->total.end()) prof_->total.emplace_back(prof_->name[i], ms); else it->second += ms;
    }
    for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);
    prof_->ev.clear();
    prof_->name.clear();
    ++prof_->rounds;
}

void Dflash2Drafter::prof_report() const {
    if (!prof_ || prof_->rounds == 0) return;
    double sum = 0;
    for (const auto& t : prof_->total) sum += t.second;
    std::fprintf(stderr, "strata dflash2 profile (%lld rounds, ms per round, total %.3f):", (long long) prof_->rounds, sum / (double) prof_->rounds);
    for (const auto& t : prof_->total) std::fprintf(stderr, " %s %.3f", t.first.c_str(), t.second / (double) prof_->rounds);
    std::fputc(10, stderr);
}

const Dflash2Drafter::Tensor& Dflash2Drafter::find(const std::string& name) const {
    for (const Tensor& t : tensors_)
        if (t.name == name) return t;
    throw std::runtime_error("dflash2: tensor " + name + " is missing from the pack");
}

bool Dflash2Drafter::load(const std::string& path, int device, std::string& err) {
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
    if (cudaMalloc((void**) &blob_, total) != cudaSuccess) { err = "dflash2: the pack does not fit in VRAM (" + std::to_string(total >> 20) + " MiB)"; return false; }
    vram_ += total;
    uint64_t max_bytes = 0;
    for (const Tensor& t : tensors_) max_bytes = std::max(max_bytes, t.bytes);
    std::vector<uint8_t> stage(max_bytes);
    uint8_t* tmp = nullptr;
    if (cudaMalloc((void**) &tmp, max_bytes) != cudaSuccess) { err = "dflash2: upload scratch"; return false; }
    struct TmpFree { uint8_t* p; ~TmpFree() { cudaFree(p); } } tmp_free{tmp};
    for (const Tensor& t : tensors_) {
#if defined(_WIN32)
        _fseeki64(f, data_off + (long long) t.off, SEEK_SET);
#else
        fseeko(f, data_off + (off_t) t.off, SEEK_SET);
#endif
        if (std::fread(stage.data(), 1, t.bytes, f) != t.bytes) { err = "dflash2: the pack is shorter than its index"; return false; }
        if (t.dtype == "q8_0") {   // GGUF blocks -> the tiled qs plane at the tensor's offset, then the d plane (same total bytes)
            if (cudaMemcpy(tmp, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
            kernels::df2_q8_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols, (int) t.cols, t.rows * t.cols / 32, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: weight conversion failed"; return false; }
        } else if (t.dtype == "q4_0" && t.name.find("_codebook") != std::string::npos) {   // the selector's Q4_0 rows stay in block order (read row by row)
            if (cudaMemcpy(blob_ + t.off, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
        } else if (t.dtype == "q4_0") {   // GGML Q4_0 blocks -> the tiled nibble plane at the tensor's offset, then the d plane (same total bytes)
            if (cudaMemcpy(tmp, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: weight upload failed"; return false; }
            kernels::df2_q4_tile(tmp, blob_ + t.off, blob_ + t.off + (uint64_t) t.rows * (uint64_t) t.cols / 2, (int) t.cols, t.rows * t.cols / 32, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: weight conversion failed"; return false; }
            q4_ = true;
        } else if (cudaMemcpy(blob_ + t.off, stage.data(), t.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dflash2: weight upload failed";
            return false;
        }
    }
    if (!make_streams(err)) return false;
    return alloc(err);
}

bool Dflash2Drafter::make_streams(std::string& err) {
    // the draft round runs while the verifier's async commit still has work on the card: a high-priority stream keeps it from queueing behind that
    // (CUDA by default; on HIP only with STRATA_DF2_PRIO=1, where the engine's other streams keep the default priority)
    int prio_lo = 0, prio_hi = 0;
#if defined(STRATA_USE_HIP)
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return e && e[0] == '1'; }();
    if (hi_prio) hipDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (hipStreamCreateWithPriority(&cs_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        hipStreamCreateWithPriority(&cs2_, hipStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#else
    static const bool hi_prio = [] { const char* e = std::getenv("STRATA_DF2_PRIO"); return !(e && e[0] == '0'); }();
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess ||
        cudaStreamCreateWithPriority(&cs2_, cudaStreamNonBlocking, hi_prio ? prio_hi : 0) != cudaSuccess) {
#endif
        err = "dflash2: stream";
        return false;
    }
    for (auto& e : ev_hs_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    if (cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "dflash2: events";
        return false;
    }
    for (int l = 0; l < 8; ++l)
        if (cudaEventCreateWithFlags(&ev_ctx_[l], cudaEventDisableTiming) != cudaSuccess) { err = "dflash2: events"; return false; }
    if (const char* e = std::getenv("STRATA_DF2_PROFILE"); e && e[0] == '1') prof_ = new Prof();
    if (const char* e = std::getenv("STRATA_DF2_TIMING"); e && e[0] == '1') {
        timing_ = true;
        cudaEventCreate(&ev_a_);
        cudaEventCreate(&ev_b_);
    }
    return true;
}

bool Dflash2Drafter::alloc(std::string& err) {
    auto dev = [&](auto** p, size_t n) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
        if (cudaMalloc((void**) p, n * sizeof(T)) != cudaSuccess) return false;
        vram_ += n * sizeof(T);
        return true;
    };
    const size_t B = kRows;
    bool ok = dev(&d_meta_, 32) && dev(&xq_, kernels::native_q8_1_bytes(N_, kRows) + 4096) && dev(&xil_, kernels::native_q8_1_il_bytes(N_, kRows) + 4096) && dev(&xqp_, (size_t) 5 * N_ * 8 + 4096) && dev(&xdp_, (size_t) 5 * N_ / 32 * 8 + 64) && dev(&xqc_, (size_t) 5 * N_ * 8 + 4096) && dev(&xdc_, (size_t) 5 * N_ / 32 * 8 + 64) && dev(&rt_ctx_, 2 * B * 128) && dev(&rope_inv_, 64) &&
              dev(&fcy_, B * N_) && dev(&hctx_, B * N_) && dev(&kc_, B * NKV_ * HD_) && dev(&vc_, B * NKV_ * HD_) && dev(&h_, B * N_) && dev(&xn_, B * N_) &&
              dev(&xc_, B * N_) && dev(&dyn_, B * 640) && dev(&q_, B * NH_ * HD_) && dev(&qn_, B * NH_ * HD_) && dev(&k_, B * NKV_ * HD_) && dev(&v_, B * NKV_ * HD_) &&
              dev(&kblk_, B * NKV_ * HD_) && dev(&ao_, B * NH_ * HD_) && dev(&o_, B * N_) && dev(&gate_, B * I_) && dev(&up_, B * I_) && dev(&mlp_, B * N_) &&
              dev(&part_, (size_t) kernels::kDf2MaxChunks * 8 * 32 * 130) && dev(&fn_, B * N_) && dev(&topval_, kDrafts * 16) && dev(&toptok_, kDrafts * 16) &&
              dev(&hsel_, kDrafts * rank_) && dev(&topscratch_, (size_t) kDrafts * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2);
    ok = ok && dev(&landing_, (size_t) kRows * std::max<size_t>(taps_.size(), 1) * (size_t) N_);
    for (int l = 0; ok && l < L_; ++l) ok = dev(&kring_[l], (size_t) kRing * NKV_ * HD_) && dev(&vring_[l], (size_t) kRing * NKV_ * HD_);
    if (!ok) { err = "dflash2: the drafter's buffers do not fit in VRAM"; return false; }
    void* hd = nullptr;
    void* dd = nullptr;
    if (!mapped(32 * 4, (void**) &hd, &dd)) { err = "dflash2: mapped staging"; return false; }
    h_meta_ = (int32_t*) hd;
    m_meta_ = (int32_t*) dd;
    if (!mapped(64 * 4, &hd, &dd)) { err = "dflash2: mapped output"; return false; }
    h_out_ = (int32_t*) hd;
    out_tok_ = (int32_t*) dd;            // the select kernel writes the 7 picks here (host-visible) ...
    out_prob_ = (float*) dd + 16;        // ... and their probabilities at +16 ints
    rt_blk_ = rt_ctx_ + B * 128;
    cudaMemset(rt_ctx_, 0, 2 * B * 128 * 4);
    kernels::df2_rope_inv(theta_, rope_inv_, nullptr);
    if (cudaMalloc((void**) &tag_, (size_t) kRing * 8) != cudaSuccess) { err = "dflash2: ring tags"; return false; }
    vram_ += (uint64_t) kRing * 8;
    cudaMemset(tag_, 0xff, (size_t) kRing * 8);   // -1: no slot holds a context row yet
    for (int64_t& t : slot_pos_) t = -1;
    init_ring_log();
    cudaMemset(xqp_, 0, (size_t) 5 * N_ * 8 + 4096);
    cudaMemset(xdp_, 0, ((size_t) 5 * N_ / 32 * 8 + 64) * 4);
    cudaMemset(xqc_, 0, (size_t) 5 * N_ * 8 + 4096);
    cudaMemset(xdc_, 0, ((size_t) 5 * N_ / 32 * 8 + 64) * 4);
    for (int l = 0; l < L_; ++l) { cudaMemset(kring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4); cudaMemset(vring_[l], 0, (size_t) kRing * NKV_ * HD_ * 4); }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool Dflash2Drafter::bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err) {
    const OnDevice on_device(device_);
    embed_ = std::move(embed);
    head_w_ = head_w;
    head_type_ = head_type;
    n_dvocab_ = head_rows;
    if (!vocab.empty()) {
        n_dvocab_ = (int64_t) vocab.size();
        if (dhead_shared_) {   // the MTP drafter's copy of the same vocabulary (checked in bind)
            dvocab_ = const_cast<int32_t*>(shared_.vocab_dev);
        } else {
            if (cudaMalloc((void**) &dvocab_, vocab.size() * 4) != cudaSuccess) { err = "dflash2: draft vocabulary"; return false; }
            vram_ += vocab.size() * 4;
            cudaMemcpy(dvocab_, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
        }
    }
    if ((n_dvocab_ + kernels::kDf2TopBlocks - 1) / kernels::kDf2TopBlocks > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    if (!logits_ && cudaMalloc((void**) &logits_, (size_t) kDrafts * (size_t) n_dvocab_ * 4) != cudaSuccess) { err = "dflash2: the draft logits do not fit"; return false; }
    vram_ += (uint64_t) kDrafts * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

#ifndef STRATA_DF2_NO_BIND   // the split engine (src/split) binds through bind_raw only: bind() copies head rows between cards with the peer API
bool Dflash2Drafter::bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err, int head_device) {
    const OnDevice on_device(device_);
    if (head == nullptr || !head->loaded()) { err = "dflash2: the drafter needs the native head (--native)"; return false; }
    const WeightRef* we = wt.find("token_embd.weight");
    if (we == nullptr && native_embed() == nullptr) { err = "dflash2: token_embd.weight is missing"; return false; }   // (a later stage of a layer split may not carry it: the native table serves)
    EmbedFn embed = [we](const int32_t* tok, int n, float* out, cudaStream_t cs) {
        if (const NativeEmbed* ne = native_embed()) {
            ne->gather_dev(tok, n, out, cs);
        } else if (we != nullptr) {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            kernels::embedding_gather_dev(codes, scales, offsets, tok, n, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                          (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), out, cs);
        }
    };
    std::vector<int32_t> vocab;
    // STRATA_DF2_HEAD=native: the drafter runs its head on the target's own resident head (every vocabulary row, no 213 MiB pruned copy, no vocabulary file read)
    // (a head on another card, a layer split's last stage, is not read in place: the pruned copy is gathered there and copied over, as before)
    const bool head_native = Dflash2Drafter::head_native() && (head_device < 0 || head_device == device_);
    if (!vocab_file.empty() && !head_native) {
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
    if (!vocab.empty() && shared_.rows != nullptr && !std::getenv("STRATA_DF2_SHARE_HEAD_OFF") && shared_.type == head->type() &&
        shared_.n == (int64_t) vocab.size() && shared_.vocab_host != nullptr && *shared_.vocab_host == vocab && shared_.vocab_dev != nullptr) {
        // DF-05: the MTP drafter gathered these very rows (same vocabulary ids in the same order, the main head's own layout and type): use them
        dhead_ = const_cast<uint8_t*>(shared_.rows);
        dhead_shared_ = true;
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    } else if (!vocab.empty() && head_device >= 0 && head_device != device_) {
        // the head is on another card: gather the subset's rows there, copy them over once
        const size_t bytes = (size_t) ((int64_t) vocab.size() * row_bytes);
        uint8_t* tmp = nullptr;
        int32_t* dv = nullptr;
        {
            const OnDevice on_head(head_device);
            if (cudaMalloc((void**) &dv, vocab.size() * 4) != cudaSuccess || cudaMalloc((void**) &tmp, bytes) != cudaSuccess) {
                if (dv) cudaFree(dv);
                err = "dflash2: the draft vocabulary's head rows do not fit on the head's card";
                return false;
            }
            cudaMemcpy(dv, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
            kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), tmp, nullptr);
            cudaDeviceSynchronize();
            cudaFree(dv);
        }
        if (cudaMalloc((void**) &dhead_, bytes) != cudaSuccess) {
            { const OnDevice on_head(head_device); cudaFree(tmp); }
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        const cudaError_t pe = cudaMemcpyPeer(dhead_, device_, tmp, head_device, bytes);
        { const OnDevice on_head(head_device); cudaFree(tmp); }
        if (pe != cudaSuccess) { err = std::string("dflash2: the head rows' copy to the drafter's card: ") + cudaGetErrorString(pe); return false; }
        vram_ += bytes;
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    } else if (!vocab.empty()) {
        int32_t* dv = nullptr;
        if (cudaMalloc((void**) &dv, vocab.size() * 4) != cudaSuccess || cudaMalloc((void**) &dhead_, (size_t) ((int64_t) vocab.size() * row_bytes)) != cudaSuccess) {
            if (dv) cudaFree(dv);
            err = "dflash2: the draft vocabulary's head rows do not fit";
            return false;
        }
        cudaMemcpy(dv, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice);
        kernels::gather_rows((const uint8_t*) hw, row_bytes, dv, (int64_t) vocab.size(), dhead_, nullptr);
        cudaDeviceSynchronize();
        cudaFree(dv);
        vram_ += (uint64_t) ((int64_t) vocab.size() * row_bytes);
        hw = dhead_;
        head_rows = (int64_t) vocab.size();
    }
    if (!bind_raw(embed, hw, head->type(), head_rows, vocab, err)) return false;
    std::fprintf(stderr, "strata dflash2: %lld-token draft head (%.1f MiB%s), drafter %.0f MiB of VRAM (%s weights)\n", (long long) n_dvocab_,
                 (double) (n_dvocab_ * row_bytes) / 1048576.0, dhead_shared_ ? ", the MTP drafter's copy, no second one" : "", (double) vram_ / 1048576.0, q4_ ? "Q4_0" : "Q8_0");
    // the pack's selector codebooks as Q4_0 rows (tools/dflash2_pack.py --q4_0-selector; strata_control/drafting.py reads this text from the program to know it can use that pack)
    if (find("candidate_selector.predecessor_codebook").dtype == "q4_0")
        std::fprintf(stderr, "strata dflash2: selector codebooks: Q4_0 rows (about 174 MiB less than bf16)\n");
    return true;
}

#endif  // STRATA_DF2_NO_BIND
void Dflash2Drafter::gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd) {
    kernels::Df2Mat mats[3];
    int n = 0, n_in = 0;
    auto o = outs.begin();
    for (const char* name : names) {
        const Tensor& t = find(name);
        mats[n] = kernels::Df2Mat{blob_ + t.off, *o++, (int) t.rows, t.dtype == "q4_0"};
        if (n == 0) n_in = (int) t.cols;
        ++n;
    }
    kernels::df2_gemm(mats, n, xq, xd, n_in, ncols, cs);
}

// The part of a context update after the activations are quantized: per layer the keys and values (projection, k-norm, rope, ring write).  Also what a ring
// rebuild runs over logged rows: `xq` / `xd` are then the log's codes and scales, `pos` the rows' positions (device), rt_ctx_ their rope table.
bool Dflash2Drafter::record_kv(int n, cudaStream_t cs, const uint8_t* xq, const float* xd, const int32_t* pos, std::string& err) {
    using namespace kernels;
    try {
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            gemm({kp.c_str(), vp.c_str()}, {kc_, vc_}, n, cs, xq, xd);
            stamp("gemm:kp");
            df2_kv_ring(kc_, vc_, f32(p + "self_attn.k_norm.weight"), pos, rt_ctx_, kring_[l], vring_[l], kRing, n, NKV_, eps_, cs, tag_, d_meta_ + 18);
            stamp("df2_kv_ring");
            if (fork_) cudaEventRecord(ev_ctx_[l], cs);   // layer l's keys and values are in the ring
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_context(const float* taps, int n, cudaStream_t cs, std::string& err, bool rope) {
    using namespace kernels;
    if (skip_part("ctx")) return true;
    try {
        const int tap_n = (int) taps_.size() * N_;
        if (rope) df2_rope_table_inv(d_meta_ + 0, n, rope_inv_, rt_ctx_, cs);
        stamp("df2_rope_table");
        df2_quant_rows(taps, tap_n, n, xqc_, xdc_, cs);
        stamp("df2_quant_rows");
        gemm({"fc.weight"}, {fcy_}, n, cs, xqc_, xdc_);
        stamp("gemm:fc");
        df2_rms_quant(fcy_, f32("hidden_norm.weight"), nullptr, n, N_, eps_, xqc_, xdc_, cs);
        stamp("df2_rms_quant");
        if (log_on_) {   // the ring log: these codes and scales are all the K/V of the five layers are made from (see begin_request)
            cudaMemcpyAsync(h_logq_, xqc_, (size_t) n * N_, cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(h_logd_, xdc_, (size_t) n * (N_ / 32) * 4, cudaMemcpyDeviceToHost, cs);
        }
        if (!record_kv(n, cs, xqc_, xdc_, d_meta_ + 0, err)) return false;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::record_block(cudaStream_t cs, std::string& err, bool rope) {
    using namespace kernels;
    try {
        const int B = kRows;
        if (rope) df2_rope_table_inv(d_meta_ + 8, B, rope_inv_, rt_blk_, cs);
        stamp("df2_rope_table");
        embed_(d_meta_ + 24, B, h_, cs);
        stamp("embed_");
        df2_rms_quant(h_, f32("layers.0.input_layernorm.weight"), xn_, B, N_, eps_, xqp_, xdp_, cs);
        stamp("df2_rms_quant");
        for (int l = 0; l < L_; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            const float* base_a = f32(p + "attention_conv.base_kernel");
            const float* base_m = f32(p + "mlp_conv.base_kernel");
            const std::string akp = p + "attention_conv.kernel_projection.weight", mkp = p + "mlp_conv.kernel_projection.weight";
            const std::string qp = p + "self_attn.q_proj.weight", kp = p + "self_attn.k_proj.weight", vp = p + "self_attn.v_proj.weight";
            const std::string gp = p + "mlp.gate_proj.weight", up = p + "mlp.up_proj.weight";
            const std::string op = p + "self_attn.o_proj.weight";
            if (!skip_part("tiny")) gemm({akp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:akp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_a, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("qkvo")) gemm({qp.c_str(), kp.c_str(), vp.c_str()}, {q_, k_, v_}, B, cs, xqp_, xdp_);
            stamp("gemm:qp");
            if (!skip_part("small")) df2_norm_rope_qk(q_, k_, f32(p + "self_attn.q_norm.weight"), f32(p + "self_attn.k_norm.weight"), rt_blk_, qn_, kblk_, B, NH_, NKV_, eps_, cs);
            stamp("df2_norm_rope_qk");
            if (fork_) cudaStreamWaitEvent(cs, ev_ctx_[l], 0);   // the context update has written this layer's ring
            if (!skip_part("attn")) df2_attention(qn_, kring_[l], vring_[l], kblk_, v_, d_meta_ + 17, kRing, window_, part_, xqp_, xdp_, cs, tag_, d_meta_ + 18);
            stamp("df2_attention");
            if (!skip_part("qkvo")) gemm({op.c_str()}, {o_}, B, cs, xqp_, xdp_);
            stamp("gemm:o_proj");
            if (!skip_part("small")) df2_conv_rms_quant(o_, dyn_ + 320, 640, base_a + 2 * N_, h_, f32(p + "post_attention_layernorm.weight"), xn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
            if (!skip_part("tiny")) gemm({mkp.c_str()}, {dyn_}, B, cs, xqp_, xdp_);
            stamp("gemm:mkp");
            if (!skip_part("small")) df2_conv_quant(xn_, dyn_, 640, base_m, B, N_, group_, xqp_, xdp_, cs);
            stamp("df2_conv_quant");
            if (!skip_part("mlp")) gemm({gp.c_str(), up.c_str()}, {gate_, up_}, B, cs, xqp_, xdp_);
            stamp("gemm:gp");
            if (!skip_part("small")) df2_swiglu_quant(gate_, up_, B, I_, xqp_, xdp_, cs);
            stamp("df2_swiglu_quant");
            if (!skip_part("mlp")) gemm({(p + "mlp.down_proj.weight").c_str()}, {mlp_}, B, cs, xqp_, xdp_);
            stamp("gemm:mlp.down_proj");
            // the next layer's input norm, or after the last layer the final norm (-> fn_, the rows the head and the selector read)
            const std::string next_gamma = l + 1 < L_ ? "layers." + std::to_string(l + 1) + ".input_layernorm.weight" : std::string("norm.weight");
            if (!skip_part("small")) df2_conv_rms_quant(mlp_, dyn_ + 320, 640, base_m + 2 * N_, h_, f32(next_gamma), l + 1 < L_ ? xn_ : fn_, B, N_, group_, eps_, xqp_, xdp_, cs);
            stamp("df2_conv_rms_quant");
        }
        native_quantize_q8_1(fn_ + N_, xq_, N_, ndraft_, cs);   // rows 1..7, the head's format
        df2_quant_rows(fn_ + N_, N_, ndraft_, xqp_, xdp_, cs);   // and the selector projection's
        const bool hs_fork = fork_ && cs2_ != nullptr && ev_hs_[0] != nullptr;   // the selector's projection beside the head, the top-k and their launches
        if (hs_fork) {
            cudaEventRecord(ev_hs_[0], cs);
            cudaStreamWaitEvent(cs2_, ev_hs_[0], 0);
            gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, ndraft_, cs2_, xqp_, xdp_);
            cudaEventRecord(ev_hs_[1], cs2_);
        }
        if (!skip_part("head")) {
            // the verify window's head kernel (interleaved activations, rows a warp by the card's table): bitwise native_mmvq's output, so the drafts are the same
            static const bool il_on = [] { const char* e = std::getenv("STRATA_DF2_HEAD_IL"); return !(e && e[0] == '0'); }();
            if (il_on && native_mmvq_il_supported(head_type_, ndraft_, (int) n_dvocab_)) {
                native_q8_1_interleave(xq_, xil_, N_, ndraft_, cs);
                native_mmvq_il(head_type_, head_w_, xq_, xil_, logits_, N_, (int) n_dvocab_, ndraft_, cs);
            } else
                native_mmvq(head_type_, head_w_, xq_, logits_, N_, (int) n_dvocab_, ndraft_, cs);
        }
        stamp("native_mmvq");
        df2_topk(logits_, (int) n_dvocab_, dvocab_, ndraft_, topscratch_, topval_, toptok_, cs);
        stamp("df2_topk");
        if (hs_fork) cudaStreamWaitEvent(cs, ev_hs_[1], 0);
        else gemm({"candidate_selector.hidden_projection.weight"}, {hsel_}, ndraft_, cs, xqp_, xdp_);
        stamp("gemm:candidate_selector.hidden_projection");
        df2_select(hsel_, topval_, toptok_, reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.predecessor_codebook").off),
                   reinterpret_cast<const uint16_t*>(blob_ + find("candidate_selector.successor_codebook").off), rank_, ndraft_, d_meta_ + 16, out_tok_,
                   out_prob_, cs, find("candidate_selector.predecessor_codebook").dtype == "q4_0");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool Dflash2Drafter::capture(int n, const float* taps, std::string& err) {
    if (exec_[n]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "dflash2: begin capture"; return false; }
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    kernels::df2_rope_table_inv(d_meta_, 16, rope_inv_, rt_ctx_, cs_);   // the context rows' and the block rows' tables in one launch, before the branches split
    // the context update (fc, the keys and values) and the block's front run side by side: they meet before each layer's attention
    // measured: -0.3 ms a round on an RTX 3060, +0.0..0.7 on a Strix Halo (gfx1151), so the HIP builds keep one stream (STRATA_DF2_FORK=1 / 0 overrides)
#if defined(STRATA_USE_HIP)
    static const bool fork_default = false;
#else
    static const bool fork_default = true;
#endif
    static const bool fork_on = [] { const char* e = std::getenv("STRATA_DF2_FORK"); return e && e[0] ? e[0] == '1' : fork_default; }();
    fork_ = fork_on && prof_ == nullptr && cs2_ != nullptr;
    if (fork_) {
        cudaEventRecord(ev_fork_, cs_);
        cudaStreamWaitEvent(cs2_, ev_fork_, 0);
    }
    bool ok = record_context(taps, n, fork_ ? cs2_ : cs_, err, false);
    if (fork_) cudaEventRecord(ev_join_, cs2_);
    if (ok) ok = record_block(cs_, err, false);
    if (fork_) cudaStreamWaitEvent(cs_, ev_join_, 0);
    fork_ = false;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) { if (graph) cudaGraphDestroy(graph); return false; }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec_[n], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("dflash2: capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(exec_[n], cs_);
    cudaStreamSynchronize(cs_);
    return true;
}

bool Dflash2Drafter::clone_alloc(const Dflash2Drafter& parent, std::string& err) {
    if (parent.blob_ == nullptr) { err = "dflash2: the parent drafter has no pack loaded"; return false; }
    device_ = parent.device_;
    const OnDevice on_device(device_);
    N_ = parent.N_; L_ = parent.L_; NH_ = parent.NH_; NKV_ = parent.NKV_; HD_ = parent.HD_; I_ = parent.I_; window_ = parent.window_;
    group_ = parent.group_; rank_ = parent.rank_; mask_token_ = parent.mask_token_; theta_ = parent.theta_; eps_ = parent.eps_;
    taps_ = parent.taps_; vocab_ = parent.vocab_; q4_ = parent.q4_; tensors_ = parent.tensors_;
    blob_ = parent.blob_; blob_shared_ = true;
    return make_streams(err) && alloc(err);
}

bool Dflash2Drafter::clone_bind(const Dflash2Drafter& parent, std::string& err) {
    if (!parent.bound_ || !blob_shared_) { err = "dflash2: the parent drafter is not bound"; return false; }
    const OnDevice on_device(device_);
    embed_ = parent.embed_; head_w_ = parent.head_w_; head_type_ = parent.head_type_; n_dvocab_ = parent.n_dvocab_;
    dvocab_ = parent.dvocab_; dhead_ = parent.dhead_; dhead_shared_ = true;
    ndraft_ = parent.ndraft_;
    if (cudaMalloc((void**) &logits_, (size_t) kDrafts * (size_t) n_dvocab_ * 4) != cudaSuccess) { err = "dflash2: the slot drafter's draft logits do not fit"; return false; }
    vram_ += (uint64_t) kDrafts * (uint64_t) n_dvocab_ * 4;
    bound_ = true;
    return true;
}

bool Dflash2Drafter::copy_state_from(Dflash2Drafter& src, std::string& err) {
    if (src.device_ != device_ || src.L_ != L_ || src.NKV_ != NKV_) { err = "dflash2: the two drafters are not alike"; return false; }
    if (!src.context_wait(err) || !context_wait(err)) return false;
    const OnDevice on_device(device_);
    if (cudaStreamSynchronize(src.cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash2: sync before a state copy"; return false; }
    const size_t ring_bytes = (size_t) kRing * NKV_ * HD_ * 4;
    for (int l = 0; l < L_; ++l)
        if (cudaMemcpyAsync(kring_[l], src.kring_[l], ring_bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
            cudaMemcpyAsync(vring_[l], src.vring_[l], ring_bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) { err = "dflash2: ring copy"; return false; }
    if (cudaMemcpyAsync(tag_, src.tag_, (size_t) kRing * 8, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) { err = "dflash2: tag copy"; return false; }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: state copy: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    copy_log_state_from(src);   // history, the log (shared copy-on-write), the tag shadow, the epoch, ring_hi_ and have_ctx_
    return true;
}

float* Dflash2Drafter::defer_rows(int64_t pos0, int n, std::string& err) {
    if (dpend_ == nullptr) {
        const OnDevice on_device(device_);
        if (cudaHostAlloc((void**) &dpend_, (size_t) kRing2048 * taps_.size() * (size_t) N_ * 4, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            dpend_ = nullptr;
            err = "dflash2: the deferred-rows buffer does not fit in pinned memory";
            return nullptr;
        }
    }
    if (!has_deferred() || pos0 != dpend_hi_) { dpend_lo_ = pos0; dpend_hi_ = pos0; }
    dpend_hi_ = pos0 + n;
    if (dpend_hi_ - dpend_lo_ > kRing2048) dpend_lo_ = dpend_hi_ - kRing2048;
    n_deferred_rows += n;
    return dpend_ + (size_t) (pos0 % kRing2048) * (size_t) (taps_.size() * (size_t) N_);
}

bool Dflash2Drafter::catch_up(int64_t upto, std::string& err) {
    if (!has_deferred()) return true;
    const Clock::time_point t0 = Clock::now();
    const OnDevice on_device(device_);
    const size_t row = taps_.size() * (size_t) N_;
    int64_t from = std::max<int64_t>(dpend_lo_, upto - 2047);
    upto = std::min<int64_t>(upto, dpend_hi_);
    int64_t replayed = 0;
    while (from < upto) {
        const int m = (int) std::min<int64_t>(kRows, upto - from);
        if (!context_wait(err)) return false;
        for (int i = 0; i < m; ++i)   // (the held run wraps at 2048 positions: row by row)
            if (cudaMemcpyAsync(landing_ + (size_t) i * row, dpend_ + (size_t) ((from + i) % kRing2048) * row, row * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
                err = "dflash2: the deferred rows' upload";
                return false;
            }
        if (!context(landing_, m, from, err)) return false;   // (waits for the stream, so the next chunk may reuse the landing)
        from += m;
        replayed += m;
    }
    dpend_lo_ = std::max<int64_t>(upto, dpend_lo_);
    if (dpend_lo_ >= dpend_hi_) dpend_lo_ = dpend_hi_ = 0;
    ++n_catchups;
    n_catchup_rows += replayed;
    ms_catchup += ms_since(t0);
    return true;
}

void Dflash2Drafter::set_max_drafts(int n) {
    n = std::max(1, std::min(n, (int) kDrafts));
    if (n == ndraft_) return;
    const OnDevice on_device(device_);
    cudaDeviceSynchronize();
    for (auto& e : exec_)
        if (e) { cudaGraphExecDestroy(e); e = nullptr; }
    graph_taps_ = nullptr;
    ndraft_ = n;
}

bool Dflash2Drafter::prepare(const float* taps, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (prof_ != nullptr) return true;   // the profile runs without graphs
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (int i = 0; i < 8; ++i) h_meta_[i] = i;
    for (int n = 1; n <= kRows; ++n)
        if (!capture(n, taps, err)) return false;
    graph_taps_ = taps;
    ms_capture = ms_since(t0);
    return true;
}

// ---- ring log -----------------------------------------------------------------------------------------------------------------------------------------
// The ring holds the last 2064 positions of the lineage it was filled for.  Resuming a prompt below the highest position written (a repeated or rewound
// prefix, a parked conversation brought back) leaves the part of the first draft's window [n - 2047, resume) either overwritten by the later positions
// (the slots are position % 2064) or never written for this lineage.  The drafts are then poor until the window refills, and the engine said nothing for the
// first kind.  The K/V rows are a function of the quantized hidden rows (codes and block scales after hidden_norm) alone, so those rows are logged as the
// updates make them (2.9 KB a position; the K/V is 40 KB and the taps 51 KB), per lineage, and a rebuild runs record_kv over them: the same kernels on the
// same bytes, so the rebuilt K/V rows are bitwise those of the original update.
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

// The log is OFF until this has allocated its pinned staging: a drafter that never ran alloc() (a --parallel slot's clone) keeps the old behaviour instead of
// copying into a null staging buffer. Callable again by a clone's allocation, on the drafter's device.
void Dflash2Drafter::init_ring_log() {
    const OnDevice on_device(device_);
    for (int64_t& t : slot_pos_) t = -1;
    const char* off = std::getenv("STRATA_DF2_RING_LOG");
    if (off && off[0] == '0') { log_on_ = false; return; }
    if (const char* e = std::getenv("STRATA_DF2_LOG_MIB"); e && e[0]) stash_budget_ = (uint64_t) std::max(0, std::atoi(e)) << 20;
    if (h_logq_ == nullptr && cudaHostAlloc((void**) &h_logq_, (size_t) kRows * N_, cudaHostAllocDefault) != cudaSuccess) h_logq_ = nullptr;
    if (h_logd_ == nullptr && cudaHostAlloc((void**) &h_logd_, (size_t) kRows * (N_ / 32) * 4, cudaHostAllocDefault) != cudaSuccess) h_logd_ = nullptr;
    log_on_ = h_logq_ != nullptr && h_logd_ != nullptr;
    if (!log_on_) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata dflash2: the ring log's staging does not fit in pinned memory: the log is off\n");
    }
}

void Dflash2Drafter::reset() {
    have_ctx_ = 0;
    cur_ = Lineage{};
    stash_.clear();
    invalidate_ring();
}

void Dflash2Drafter::copy_log_state_from(const Dflash2Drafter& src) {
    cur_.hist = src.cur_.hist;
    cur_.chunks = src.cur_.chunks;
    std::memcpy(slot_pos_, src.slot_pos_, sizeof(slot_pos_));
    epoch_ = src.epoch_;
    ring_hi_ = src.ring_hi_;
    have_ctx_ = src.have_ctx_;
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

bool Dflash2Drafter::rebuild(int64_t lo, int64_t hi, int64_t* rebuilt, int64_t* missing, std::string& err) {
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
    if (rb_q_ == nullptr) {
        if (cudaMalloc((void**) &rb_q_, (cap + kRows) * 2560) != cudaSuccess || cudaMalloc((void**) &rb_d_, (cap + kRows) * 80 * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &rb_pos_, (cap + kRows) * sizeof(int32_t)) != cudaSuccess) {
            if (rb_q_) cudaFree(rb_q_);
            if (rb_d_) cudaFree(rb_d_);
            rb_q_ = nullptr; rb_d_ = nullptr; rb_pos_ = nullptr;
            (void) cudaGetLastError();
            err = "dflash2: the ring rebuild's scratch does not fit";
            return false;
        }
        cudaMemset(rb_q_, 0, (cap + kRows) * 2560);
        cudaMemset(rb_d_, 0, (cap + kRows) * 80 * sizeof(float));
        rb_bytes_ = (cap + kRows) * (2560 + 80 * sizeof(float) + sizeof(int32_t));
        vram_ += rb_bytes_;
    }
    std::vector<int8_t> q(rows * 2560);
    std::vector<float> d(rows * 80);
    for (size_t i = 0; i < rows; ++i) {
        const int64_t p = pos[i];
        const LogChunk& ch = *cur_.chunks[(size_t) (p / kLogChunk)];
        const size_t r = (size_t) (p % kLogChunk);
        std::memcpy(&q[i * 2560], &ch.q[r * 2560], 2560);
        std::memcpy(&d[i * 80], &ch.d[r * 80], 80 * sizeof(float));
    }
    h_meta_[18] = epoch_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    cudaMemcpyAsync(rb_q_, q.data(), q.size(), cudaMemcpyHostToDevice, cs_);
    cudaMemcpyAsync(rb_d_, d.data(), d.size() * sizeof(float), cudaMemcpyHostToDevice, cs_);
    cudaMemcpyAsync(rb_pos_, pos.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice, cs_);
    // (the host vectors die with this scope: wait for the copies)
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ring rebuild: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    const bool was_fork = fork_;
    fork_ = false;
    bool ok = true;
    for (size_t r0 = 0; ok && r0 < rows; r0 += kRows) {
        const int m = (int) std::min<size_t>(kRows, rows - r0);
        try { kernels::df2_rope_table_inv(rb_pos_ + r0, m, rope_inv_, rt_ctx_, cs_); } catch (const std::exception& e) { err = e.what(); ok = false; break; }
        ok = record_kv(m, cs_, reinterpret_cast<const uint8_t*>(rb_q_) + r0 * 2560, rb_d_ + r0 * 80, rb_pos_ + r0, err);
    }
    fork_ = was_fork;
    if (!ok) return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ring rebuild: ") + cudaGetErrorString(cudaGetLastError()); return false; }
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
    (void) context_wait(err);
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

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (!context_wait(err)) return false;
    const OnDevice on_device(device_);
    const int tap_n = (int) taps_.size() * N_;
    for (int64_t r0 = 0; r0 < n; r0 += kRows) {
        const int m = (int) std::min<int64_t>(kRows, n - r0);
        for (int i = 0; i < m; ++i) h_meta_[i] = (int32_t) (pos0 + r0 + i);
        h_meta_[18] = epoch_;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
        log_stage(pos0 + r0, m);
        if (!record_context(taps + (size_t) r0 * (size_t) tap_n, m, cs_, err)) return false;
        if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
        log_flush();
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::context_async(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (n < 1 || n > kRows) { err = "dflash2: context_async rows out of range"; return false; }
    if (!context_wait(err)) return false;   // h_meta_ is read by the queued copy: the previous call must be done
    const OnDevice on_device(device_);
    for (int i = 0; i < (int) n; ++i) h_meta_[i] = (int32_t) (pos0 + i);
    h_meta_[18] = epoch_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
    log_stage(pos0, (int) n);
    if (!record_context(taps, (int) n, cs_, err)) return false;
    ctx_pending_ = true;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::context_wait(std::string& err) {
    if (!ctx_pending_) return true;
    ctx_pending_ = false;
    const OnDevice on_device(device_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    log_flush();
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    return draft_launch(taps, n_new, pos0, anchor, err) && draft_finish(drafts, probs, err);
}

bool Dflash2Drafter::draft_launch(const float* taps, int n_new, int64_t pos0, int32_t anchor, std::string& err) {
    if (!bound_) { err = "dflash2: not bound"; return false; }
    if (!context_wait(err)) return false;
    if (n_new < 1 || n_new > kRows) { err = "dflash2: draft rows out of range"; return false; }
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    if (graph_taps_ != nullptr && graph_taps_ != taps) { err = "dflash2: the taps buffer moved after capture"; return false; }
    if (!prof_ && !exec_[n_new]) {
        // the staging must hold valid positions while the capture runs the (empty) kernels' parameter checks
        for (int i = 0; i < 8; ++i) h_meta_[i] = (int32_t) (pos0 + i);
        if (!capture(n_new, taps, err)) return false;
        graph_taps_ = taps;
    }
    const int64_t s = pos0 + n_new;
    for (int i = 0; i < 8; ++i) { h_meta_[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); h_meta_[8 + i] = (int32_t) (s + i); }
    h_meta_[16] = anchor;
    h_meta_[18] = epoch_;
    h_meta_[17] = (int32_t) s;
    h_meta_[24] = anchor;
    for (int i = 1; i < 8; ++i) h_meta_[24 + i] = mask_token_;
    for (int i = 0; i < kDrafts; ++i) { ((volatile int32_t*) h_out_)[i] = -1; }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    log_stage(pos0, n_new);
    if (timing_) cudaEventRecord(ev_a_, cs_);
    if (prof_) {   // the same round without its graph, an event after every stage
        cudaStreamSynchronize(cs_);
        for (cudaEvent_t e : prof_->ev) cudaEventDestroy(e);   // the stamps of context() calls are not a round
        prof_->ev.clear();
        prof_->name.clear();
        kernels::copy_i32_from_mapped(d_meta_, m_meta_, 32, cs_);
        stamp("start");
        if (!record_context(taps, n_new, cs_, err) || !record_block(cs_, err)) return false;
        prof_collect();
    } else
    if (cudaGraphLaunch(exec_[n_new], cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    if (timing_) cudaEventRecord(ev_b_, cs_);
    (void) cudaStreamQuery(cs_);
    launch_t0_ = t0;
    launch_n_ = n_new;
    launch_pos0_ = pos0;
    launched_ = true;
    return true;
}

bool Dflash2Drafter::draft_finish(int32_t* drafts, float* probs, std::string& err) {
    if (!launched_) { err = "dflash2: draft_finish without draft_launch"; return false; }
    launched_ = false;
    const OnDevice on_device(device_);
    const int n_new = launch_n_;
    const int64_t pos0 = launch_pos0_, s = pos0 + n_new;
    const Clock::time_point t0 = launch_t0_;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = std::string("dflash2 draft: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    log_flush();
    for (int i = 0; i < kDrafts; ++i) {
        const bool made = i < ndraft_;
        drafts[i] = made ? ((volatile int32_t*) h_out_)[i] : 0;
        if (probs) probs[i] = made ? ((volatile float*) h_out_)[16 + i] : 0.0f;
    }
    if (timing_) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, ev_a_, ev_b_) == cudaSuccess) ms_gpu += ms;
    }
    have_ctx_ = std::max<int64_t>(have_ctx_, s);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n_new);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool Dflash2Drafter::debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err) {
    const OnDevice on_device(device_);
    if (!context_wait(err)) return false;
    const size_t per = (size_t) kRing * NKV_ * HD_;
    kv.resize((size_t) L_ * 2 * per);
    tags.resize((size_t) kRing);
    if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash2: debug ring sync"; return false; }
    for (int l = 0; l < L_; ++l) {
        if (cudaMemcpy(kv.data() + ((size_t) l * 2) * per, kring_[l], per * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(kv.data() + ((size_t) l * 2 + 1) * per, vring_[l], per * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug ring copy"; return false; }
    }
    if (cudaMemcpy(tags.data(), tag_, (size_t) kRing * 8, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug tag copy"; return false; }
    return true;
}

bool Dflash2Drafter::debug_rows(std::vector<float>& rows, std::string& err) {
    rows.resize((size_t) kRows * (size_t) N_);
    if (cudaMemcpy(rows.data(), fn_, rows.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "dflash2: debug copy"; return false; }
    return true;
}

}  // namespace strata::core
