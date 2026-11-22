#include "strata/core/native_head.hpp"
#include "strata/core/spill_gate.hpp"
#include "strata/core/run_manifest.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/vhead_bitplane.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace strata::core {

namespace {
struct ScopedDev {
    int prev = -1;
    bool switched = false;
    explicit ScopedDev(int dev) {
        if (dev >= 0 && cudaGetDevice(&prev) == cudaSuccess && prev != dev) {
            cudaSetDevice(dev);
            switched = true;
        }
    }
    ~ScopedDev() { if (switched) (void) cudaSetDevice(prev); }
};

void ck_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}  // namespace

NativeHead::~NativeHead() {
    unload();
}

void NativeHead::unload() {
    if (bitplane_) { strata::kernels::vhead_bitplane_destroy(bitplane_); bitplane_ = nullptr; }
    if (peer_device_ >= 0) {
        ScopedDev on1(peer_device_);
        if (ev_done_) cudaEventDestroy((cudaEvent_t) ev_done_);
        if (peer_stream_) cudaStreamDestroy((cudaStream_t) peer_stream_);
        if (peer_weights_) cudaFree(peer_weights_);
        if (peer_xq_) cudaFree(peer_xq_);
        if (peer_xil_) cudaFree(peer_xil_);
        if (peer_logits_) cudaFree(peer_logits_);
        if (peer_arg_scratch_) cudaFree(peer_arg_scratch_);
        if (peer_max_val_) cudaFree(peer_max_val_);
        if (peer_max_idx_) cudaFree(peer_max_idx_);
        if (peer_nf_) cudaFree(peer_nf_);
        (void) cudaGetLastError();
    }
    if (device_ >= 0) {
        ScopedDev on0(device_);
        if (ev_in_) cudaEventDestroy((cudaEvent_t) ev_in_);
        if (max_val0_) cudaFree(max_val0_);
        if (max_idx0_) cudaFree(max_idx0_);
        if (max_val1_) cudaFree(max_val1_);
        if (max_idx1_) cudaFree(max_idx1_);
        if (nf1_) cudaFree(nf1_);
        if (weights_) strata::kernels::native_q6_k_unpack(weights_);
        if (weights_) strata::kernels::native_q8_0_sm60_release(weights_);
        if (scratch_) cudaFree(scratch_);
        if (host_) cudaFreeHost(host_);   // the mapped-host fallback (weights_ is its device alias)
        else if (weights_) cudaFree(weights_);
        (void) cudaGetLastError();
    } else {
        if (weights_) strata::kernels::native_q6_k_unpack(weights_);
        if (weights_) strata::kernels::native_q8_0_sm60_release(weights_);
        if (scratch_) cudaFree(scratch_);
        if (host_) cudaFreeHost(host_);
        else if (weights_) cudaFree(weights_);
    }
    scratch_ = nullptr;
    weights_ = nullptr;
    host_ = nullptr;
    peer_device_ = -1;
    split_rows0_ = split_rows1_ = max_split_t_ = 0;
    peer_weights_ = peer_stream_ = ev_in_ = ev_done_ = nullptr;
    peer_xq_ = peer_xil_ = peer_arg_scratch_ = nullptr;
    peer_logits_ = max_val0_ = max_val1_ = peer_max_val_ = nullptr;
    max_idx0_ = max_idx1_ = peer_max_idx_ = nullptr;
    nf1_ = peer_nf_ = nullptr;
    bytes_ = 0;
    n_in_ = n_out_ = 0;
    type_ = -1;
    device_ = -1;
}

bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err,
                      int peer_device) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const strata::GgufModel model(shards);
        err = strata::check_engine_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        void* host = nullptr;
        int dev0 = -1;
        if (cudaGetDevice(&dev0) != cudaSuccess) { (void) cudaGetLastError(); dev0 = -1; }
        const bool ask_host = std::getenv("STRATA_HEAD_HOST") != nullptr;
#if !defined(__HIPCC__) && !defined(STRATA_GPU_METAL)
        const char* sp = std::getenv("STRATA_PEER_HEAD_SPLIT");
        const char* np = std::getenv("STRATA_PEER_NO_P2P");
        if (sp != nullptr && std::atoi(sp) != 0 && peer_device >= 0 && peer_device != dev0 && dev0 >= 0 && !ask_host &&
            !strata::kernels::native_q6_k_packed_enabled() && !(tensor->type == 8 && std::getenv("STRATA_Q8_SM60") != nullptr) &&
            n_out >= 2 && !(np != nullptr && std::atoi(np) != 0)) {
            int p01 = 0, p10 = 0;
            cudaDeviceCanAccessPeer(&p01, dev0, peer_device);
            cudaDeviceCanAccessPeer(&p10, peer_device, dev0);
            if (p01 && p10) {
                cudaError_t e0 = cudaDeviceEnablePeerAccess(peer_device, 0);
                if (e0 == cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); e0 = cudaSuccess; }
                cudaError_t e1 = cudaSuccess;
                {
                    ScopedDev on1(peer_device);
                    e1 = cudaDeviceEnablePeerAccess(dev0, 0);
                    if (e1 == cudaErrorPeerAccessAlreadyEnabled) { (void) cudaGetLastError(); e1 = cudaSuccess; }
                }
                if (e0 == cudaSuccess && e1 == cudaSuccess) {
                    const int rows0 = (int) (n_out / 2), rows1 = (int) (n_out - rows0);
                    const uint64_t row_b = bytes / (uint64_t) n_out;
                    const uint64_t bytes0 = (uint64_t) rows0 * row_b, bytes1 = (uint64_t) rows1 * row_b;
                    const int qcols = (int) strata::kernels::kVerifyMaxT;
                    const int max_st = std::max((int) strata::kernels::kVerifyLongT, qcols);
                    bool ok0 = false, ok1 = false;
                    cudaEvent_t ein = nullptr, edone = nullptr;
                    cudaStream_t ps = nullptr;
                    device_ = dev0;
                    peer_device_ = peer_device;
                    ok0 = cudaMalloc(&weights_, bytes0) == cudaSuccess &&
                          cudaMalloc(&scratch_, strata::kernels::native_q8_1_bytes((int) n_in, 1)) == cudaSuccess &&
                          cudaEventCreateWithFlags(&ein, cudaEventDisableTiming) == cudaSuccess &&
                          cudaMalloc((void**) &max_val0_, (size_t) max_st * sizeof(float)) == cudaSuccess &&
                          cudaMalloc((void**) &max_idx0_, (size_t) max_st * sizeof(int32_t)) == cudaSuccess &&
                          cudaMalloc((void**) &max_val1_, (size_t) max_st * sizeof(float)) == cudaSuccess &&
                          cudaMalloc((void**) &max_idx1_, (size_t) max_st * sizeof(int32_t)) == cudaSuccess &&
                          cudaMalloc((void**) &nf1_, (size_t) max_st * strata::kernels::kNonfiniteBlocks * sizeof(uint32_t)) == cudaSuccess;
                    ev_in_ = ein;
                    if (ok0) {
                        ScopedDev on1(peer_device);
                        ok1 = cudaMalloc(&peer_weights_, bytes1) == cudaSuccess &&
                              cudaStreamCreateWithFlags(&ps, cudaStreamNonBlocking) == cudaSuccess &&
                              cudaEventCreateWithFlags(&edone, cudaEventDisableTiming) == cudaSuccess &&
                              cudaMalloc((void**) &peer_xq_, strata::kernels::native_q8_1_bytes((int) n_in, qcols)) == cudaSuccess &&
                              cudaMalloc((void**) &peer_xil_, strata::kernels::native_q8_1_il_bytes((int) n_in, std::max(4, qcols))) == cudaSuccess &&
                              cudaMalloc((void**) &peer_logits_, (size_t) max_st * (size_t) rows1 * sizeof(float)) == cudaSuccess &&
                              cudaMalloc((void**) &peer_arg_scratch_, strata::kernels::argmax_rows_scratch_bytes(max_st)) == cudaSuccess &&
                              cudaMalloc((void**) &peer_max_val_, (size_t) max_st * sizeof(float)) == cudaSuccess &&
                              cudaMalloc((void**) &peer_max_idx_, (size_t) max_st * sizeof(int32_t)) == cudaSuccess &&
                              cudaMalloc((void**) &peer_nf_, (size_t) max_st * strata::kernels::kNonfiniteBlocks * sizeof(uint32_t)) == cudaSuccess;
                        peer_stream_ = ps;
                        ev_done_ = edone;
                        if (ok1) ok1 = cudaMemset(peer_arg_scratch_, 0, strata::kernels::argmax_rows_scratch_bytes(max_st)) == cudaSuccess;
                    }
                    if (ok0 && ok1) {
                        const auto* raw = static_cast<const uint8_t*>(gguf.tensor_data(*tensor));
                        strata::platform::advise_willneed(raw, bytes);
                        ok0 = cudaMemcpy(weights_, raw, bytes0, cudaMemcpyHostToDevice) == cudaSuccess;
                        if (ok0) {
                            ScopedDev on1(peer_device);
                            ok1 = cudaMemcpy(peer_weights_, raw + bytes0, bytes1, cudaMemcpyHostToDevice) == cudaSuccess;
                        }
                    }
                    if (ok0 && ok1) {
                        split_rows0_ = rows0;
                        split_rows1_ = rows1;
                        max_split_t_ = max_st;
                        bytes_ = bytes;
                        n_in_ = (int) n_in;
                        n_out_ = (int) n_out;
                        type_ = (int) tensor->type;
                        std::fprintf(stderr,
                                     "strata generate: STRATA_PEER_HEAD_SPLIT=1: split output head (%s) across GPU %d (%d rows, %.1f MiB) "
                                     "and peer GPU %d (%d rows, %.1f MiB) over P2P\n",
                                     strata::ggml_type_name(tensor->type), dev0, rows0, (double) bytes0 / 1048576.0,
                                     peer_device, rows1, (double) bytes1 / 1048576.0);
                        return true;
                    }
                    (void) cudaGetLastError();
                    unload();
                    std::fprintf(stderr, "strata generate: STRATA_PEER_HEAD_SPLIT=1 fell back to single-GPU output head (P2P allocation failed)\n");
                }
            }
        }
#else
        (void) peer_device;
#endif
        // SHORT OF VRAM: the head in mapped pinned host memory.  The head (about 0.4 GiB) is the one buffer a card that holds its sessions and weights but not this can
        // spill whole; every logits row then reads it over PCIe (about 18 ms per row at 28 GB/s), so decode is slower, but the start works.  Taken only when the
        // allocation in VRAM fails (or STRATA_HEAD_HOST=1 asks for it); the GEMV kernels read the same pointer either way.  Q5_K and the plain MMVQ types: the
        // repacked Q6_K / SM60 layouts rewrite the weights in place and stay in VRAM.
        cudaError_t status = ask_host ? cudaErrorMemoryAllocation : cudaMalloc(&weights, bytes);
        bool discrete = false;
        { int d = 0; cudaDeviceProp pr{}; if (cudaGetDevice(&d) == cudaSuccess && cudaGetDeviceProperties(&pr, d) == cudaSuccess) discrete = strata::spill_allowed(strata::kSpillMetalBuild, pr.integrated != 0); else (void) cudaGetLastError(); }
        if (status != cudaSuccess && discrete && std::getenv("STRATA_HEAD_HOST_OFF") == nullptr && !strata::kernels::native_q6_k_packed_enabled()) {
            (void) cudaGetLastError();
            void* hp = nullptr;
            void* dp = nullptr;
            if (cudaHostAlloc(&hp, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess && cudaHostGetDevicePointer(&dp, hp, 0) == cudaSuccess) {
                std::memcpy(hp, gguf.tensor_data(*tensor), bytes);
                host = hp;
                weights = dp;
                status = cudaSuccess;
                std::fprintf(stderr, "strata generate: short of GPU memory: the output head (%llu MiB) stays in pinned host memory and is read over PCIe for every logits row: "
                                     "decode is slower%s", (unsigned long long) (bytes >> 20), "\n");
            } else {
                if (hp) cudaFreeHost(hp);
                (void) cudaGetLastError();
            }
        }
        if (status == cudaSuccess)
            status = cudaMalloc(&scratch, strata::kernels::native_q8_1_bytes((int) n_in, 1));
        strata::platform::advise_willneed(gguf.tensor_data(*tensor), bytes);
        if (status == cudaSuccess && host == nullptr)
            status = cudaMemcpy(weights, gguf.tensor_data(*tensor), bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            if (host) cudaFreeHost(host);
            weights = host ? nullptr : weights;
            if (scratch) cudaFree(scratch);
            if (weights) cudaFree(weights);
            err = std::string("native head upload: ") + cudaGetErrorString(status);
            return false;
        }
        weights_ = weights;
        host_ = host;
        { int d = -1; if (cudaGetDevice(&d) == cudaSuccess) device_ = d; }
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        if (type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
            strata::kernels::native_q6_k_pack(weights_, n_in_, n_out_, "output head");
        if (type_ == 8)   // STRATA_Q8_SM60=1 (a no-op otherwise); the MTP draft head reads the same matrix
            strata::kernels::native_q8_0_sm60_pack(weights_, n_in_, n_out_, "output head");
        if (host_ == nullptr && (type_ == 8 || type_ == 14) && strata::kernels::vhead_bitplane_enabled() &&
            !(type_ == 14 && strata::kernels::native_q6_k_packed_enabled())) {
            size_t free_b = 0, total_b = 0;
            // Require >= 1.5 GiB VRAM headroom so small-VRAM cards keep their expert slots.
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b > (1536ull << 20)) {
                bitplane_ = strata::kernels::vhead_bitplane_create(type_, weights_, n_in_, n_out_);
                if (bitplane_ != nullptr) {
                    std::fprintf(stderr, "strata: exact bit-plane LM head enabled (%s, +%.1f MiB VRAM; STRATA_VHEAD_BITPLANE=0 disables)\n",
                                 strata::ggml_type_name((uint32_t) type_),
                                 (double) strata::kernels::vhead_bitplane_bytes(bitplane_) / (1024.0 * 1024.0));
                }
            } else {
                (void) cudaGetLastError();
            }
        }
        if (strata::core::manifest::enabled()) {   // STRATA_MANIFEST: what the head holds and where
            namespace mf = strata::core::manifest;
            mf::instance().tensors("head", strata::ggml_type_name((uint32_t) type_), 1, bytes);
            mf::instance().path("head", "output.weight", true,
                                host_ != nullptr ? "held in mapped pinned host memory (short of VRAM or STRATA_HEAD_HOST)" : "held in VRAM",
                                std::string("type ") + strata::ggml_type_name((uint32_t) type_) +
                                    (type_ == 14 && strata::kernels::native_q6_k_packed_enabled() ? ", repacked (STRATA_Q6_PACKED)" : "") +
                                    (type_ == 8 && std::getenv("STRATA_Q8_SM60") != nullptr ? ", repacked (STRATA_Q8_SM60)" : ""));
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::to_host(std::string& err) {
    if (is_split() || weights_ == nullptr || host_ != nullptr || (type_ == 14 && strata::kernels::native_q6_k_packed_enabled()) || (type_ == 8 && std::getenv("STRATA_Q8_SM60") != nullptr)) return false;   // (repacked/split stays in VRAM)
    if (bitplane_) { strata::kernels::vhead_bitplane_destroy(bitplane_); bitplane_ = nullptr; }
    { int d = 0; cudaDeviceProp pr{}; if (cudaGetDevice(&d) != cudaSuccess || cudaGetDeviceProperties(&pr, d) != cudaSuccess || !strata::spill_allowed(strata::kSpillMetalBuild, pr.integrated != 0)) { (void) cudaGetLastError(); return false; } }   // (Q6_K / Q8_0 may be repacked in place: those stay)
    void* hp = nullptr;
    void* dp = nullptr;
    if (cudaHostAlloc(&hp, bytes_, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "native head: no pinned host memory for the head";
        return false;
    }
    if (cudaHostGetDevicePointer(&dp, hp, 0) != cudaSuccess || cudaMemcpy(hp, weights_, bytes_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFreeHost(hp);
        err = "native head: the head could not be moved to host memory";
        return false;
    }
    cudaFree(weights_);
    weights_ = dp;
    host_ = hp;
    return true;
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (is_split()) {
            auto cs0 = static_cast<cudaStream_t>(stream);
            auto ps = static_cast<cudaStream_t>(peer_stream_);
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, cs0);
            ck_cuda(cudaEventRecord((cudaEvent_t) ev_in_, cs0), "native head ev_in");
            {
                ScopedDev on1(peer_device_);
                ck_cuda(cudaStreamWaitEvent(ps, (cudaEvent_t) ev_in_, 0), "native head wait ev_in");
                ck_cuda(cudaMemcpyPeerAsync(peer_xq_, peer_device_, scratch_, device_,
                                            (size_t) strata::kernels::native_q8_1_bytes(n_in_, 1), ps),
                        "native head push q8");
                strata::kernels::native_mmvq(type_, peer_weights_, peer_xq_, peer_logits_, n_in_, split_rows1_, 1, ps);
                ck_cuda(cudaMemcpyPeerAsync(logits + split_rows0_, device_, peer_logits_, peer_device_,
                                            (size_t) split_rows1_ * sizeof(float), ps),
                        "native head pull logits");
                ck_cuda(cudaEventRecord((cudaEvent_t) ev_done_, ps), "native head ev_done");
            }
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, split_rows0_, 1, cs0);
            ck_cuda(cudaStreamWaitEvent(cs0, (cudaEvent_t) ev_done_, 0), "native head wait ev_done");
            return true;
        }
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        err = std::string("native head launch: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool NativeHead::gather_rows(const int32_t* vocab_host, int64_t n_rows, int dst_dev, uint8_t* dst, std::string& err) const {
    if (!is_split() || vocab_host == nullptr || n_rows <= 0 || dst == nullptr) {
        err = "native head gather_rows: split head not active or invalid arguments";
        return false;
    }
    const size_t rb = row_bytes();
    std::vector<int32_t> id0, id1;
    std::vector<int64_t> pos0, pos1;
    id0.reserve((size_t) n_rows);
    id1.reserve((size_t) n_rows);
    pos0.reserve((size_t) n_rows);
    pos1.reserve((size_t) n_rows);
    for (int64_t i = 0; i < n_rows; ++i) {
        const int32_t id = vocab_host[i];
        if (id < 0 || id >= n_out_) {
            err = "native head gather_rows: vocabulary id out of range";
            return false;
        }
        if (id < split_rows0_) {
            id0.push_back(id);
            pos0.push_back(i);
        } else {
            id1.push_back(id - split_rows0_);
            pos1.push_back(i);
        }
    }
    auto gather_half = [&](int dev, const void* wsrc, const std::vector<int32_t>& ids, const std::vector<int64_t>& pos,
                           std::vector<uint8_t>& h_dst) -> bool {
        if (ids.empty()) return true;
        ScopedDev on_dev(dev);
        int32_t* dv = nullptr;
        uint8_t* tmp = nullptr;
        const size_t bytes = ids.size() * rb;
        if (cudaMalloc((void**) &dv, ids.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &tmp, bytes) != cudaSuccess) {
            if (dv) cudaFree(dv);
            if (tmp) cudaFree(tmp);
            (void) cudaGetLastError();
            err = "native head gather_rows: device temporary allocation failed";
            return false;
        }
        std::vector<uint8_t> h_tmp(bytes);
        const bool ok = cudaMemcpy(dv, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) {
            strata::kernels::gather_rows((const uint8_t*) wsrc, (int64_t) rb, dv, (int64_t) ids.size(), tmp, nullptr);
        }
        const bool ok2 = ok && cudaDeviceSynchronize() == cudaSuccess &&
                           cudaMemcpy(h_tmp.data(), tmp, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(dv);
        cudaFree(tmp);
        if (!ok2) {
            (void) cudaGetLastError();
            err = "native head gather_rows: device gather failed";
            return false;
        }
        for (size_t k = 0; k < ids.size(); ++k)
            std::memcpy(h_dst.data() + (size_t) pos[k] * rb, h_tmp.data() + k * rb, rb);
        return true;
    };
    std::vector<uint8_t> h_dst((size_t) n_rows * rb);
    if (!gather_half(device_, weights_, id0, pos0, h_dst)) return false;
    if (!gather_half(peer_device_, peer_weights_, id1, pos1, h_dst)) return false;
    ScopedDev on_dst(dst_dev);
    if (cudaMemcpy(dst, h_dst.data(), (size_t) n_rows * rb, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "native head gather_rows: destination upload failed";
        return false;
    }
    return true;
}

void NativeHead::run_verify_split(const void* xq, void* xil, float* head_logits_dev0, int h0, int HT, bool use_il,
                                  void* arg_scratch0, int32_t* m_out_row, uint32_t* m_nf_row, void* stream0) const {
    if (!is_split() || HT <= 0) throw std::runtime_error("run_verify_split: split head not active");
    auto cs0 = static_cast<cudaStream_t>(stream0);
    auto ps = static_cast<cudaStream_t>(peer_stream_);
    const size_t q8_bytes = (size_t) strata::kernels::native_q8_1_bytes(n_in_, HT);

    ck_cuda(cudaEventRecord((cudaEvent_t) ev_in_, cs0), "split head ev_in");
    {
        ScopedDev on1(peer_device_);
        ck_cuda(cudaStreamWaitEvent(ps, (cudaEvent_t) ev_in_, 0), "split head peer wait");
        ck_cuda(cudaMemcpyAsync(peer_xq_, xq, q8_bytes, cudaMemcpyDefault, ps), "split head peer xq");
        float* const pl = peer_logits_ + (size_t) h0 * (size_t) split_rows1_;
        const bool il1 = use_il && strata::kernels::native_mmvq_il_supported(type_, HT, split_rows1_);
        if (il1) {
            if (HT >= 2) strata::kernels::native_q8_1_interleave(peer_xq_, peer_xil_, n_in_, HT, ps);
            strata::kernels::native_mmvq_il(type_, peer_weights_, peer_xq_, peer_xil_, pl, n_in_, split_rows1_, HT, ps);
        } else {
            strata::kernels::native_mmvq(type_, peer_weights_, peer_xq_, pl, n_in_, split_rows1_, HT, ps);
        }
        strata::kernels::argmax_rows_val_idx(pl, HT, split_rows1_, split_rows0_, peer_arg_scratch_, peer_max_val_,
                                             peer_max_idx_, ps);
        if (m_nf_row != nullptr)
            strata::kernels::logits_nonfinite_rows(pl, HT, split_rows1_, peer_nf_, ps);
        ck_cuda(cudaMemcpyAsync(max_val1_, peer_max_val_, (size_t) HT * sizeof(float), cudaMemcpyDefault, ps),
                "split head max_val1");
        ck_cuda(cudaMemcpyAsync(max_idx1_, peer_max_idx_, (size_t) HT * sizeof(int32_t), cudaMemcpyDefault, ps),
                "split head max_idx1");
        if (m_nf_row != nullptr)
            ck_cuda(cudaMemcpyAsync(nf1_, peer_nf_, (size_t) HT * strata::kernels::kNonfiniteBlocks * sizeof(uint32_t),
                                    cudaMemcpyDefault, ps),
                    "split head nf1");
        ck_cuda(cudaEventRecord((cudaEvent_t) ev_done_, ps), "split head ev_done");
    }

    float* const hl0 = head_logits_dev0 + (size_t) h0 * (size_t) split_rows0_;
    const bool il0 = use_il && strata::kernels::native_mmvq_il_supported(type_, HT, split_rows0_);
    if (il0) {
        if (HT >= 2) strata::kernels::native_q8_1_interleave(xq, xil, n_in_, HT, cs0);
        strata::kernels::native_mmvq_il(type_, weights_, xq, xil, hl0, n_in_, split_rows0_, HT, cs0);
    } else {
        strata::kernels::native_mmvq(type_, weights_, xq, hl0, n_in_, split_rows0_, HT, cs0);
    }
    strata::kernels::argmax_rows_val_idx(hl0, HT, split_rows0_, 0, arg_scratch0, max_val0_, max_idx0_, cs0);
    if (m_nf_row != nullptr)
        strata::kernels::logits_nonfinite_rows(hl0, HT, split_rows0_, m_nf_row, cs0);
    ck_cuda(cudaStreamWaitEvent(cs0, (cudaEvent_t) ev_done_, 0), "split head join ev_done");
    strata::kernels::argmax_merge_halves(max_val0_, max_idx0_, max_val1_, max_idx1_, HT, m_out_row, cs0);
    if (m_nf_row != nullptr)
        strata::kernels::or_u32_rows(m_nf_row, nf1_, HT * strata::kernels::kNonfiniteBlocks, cs0);
}

void NativeHead::materialize_logits(float* head_logits_dev0, int T, void* stream0) const {
    if (!is_split() || head_logits_dev0 == nullptr || T <= 0 || T > max_split_t_) return;
    auto cs0 = static_cast<cudaStream_t>(stream0);
    for (int t = T - 1; t >= 1; --t) {
        cudaMemcpyAsync(head_logits_dev0 + (size_t) t * (size_t) n_out_,
                        head_logits_dev0 + (size_t) t * (size_t) split_rows0_,
                        (size_t) split_rows0_ * sizeof(float), cudaMemcpyDeviceToDevice, cs0);
    }
    for (int t = 0; t < T; ++t) {
        cudaMemcpyAsync(head_logits_dev0 + (size_t) t * (size_t) n_out_ + (size_t) split_rows0_,
                        peer_logits_ + (size_t) t * (size_t) split_rows1_,
                        (size_t) split_rows1_ * sizeof(float), cudaMemcpyDefault, cs0);
    }
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) cudaFreeHost(host_);
    else if (dev_) cudaFree(const_cast<void*>(dev_));   // the VRAM fallback below
}

bool NativeEmbed::load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err) {
    try {
        const strata::GgufModel model(shards);
        // --embd-gguf's one-tensor file (tools/embd_bf16_pack.py) says "strata-embd": only its tensor is checked
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_engine_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        strata::platform::advise_willneed(gguf.tensor_data(*t), bytes_);   // copied out of the mapping below
        // STRATA_EMBED_VRAM=1: the table in the current device's VRAM instead of mapped host memory. The same
        // bytes, read by the same gather kernels through dev_ (bitwise the same rows); what changes is where a
        // prompt's scattered row reads go: over a PCIe Gen2 x4 link they cost more than the table's size in VRAM
        // is worth. Taken only with the table plus 512 MiB free; otherwise, or when the upload fails, the mapped
        // placement below is used as before (and the fallback is said).
        if (const char* ev = std::getenv("STRATA_EMBED_VRAM"); ev != nullptr && ev[0] == '1') {
            size_t free_b = 0, total_b = 0;
            const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
            void* d = nullptr;
            if (have_free && (uint64_t) free_b > bytes_ + ((uint64_t) 512 << 20) &&
                cudaMalloc(&d, bytes_) == cudaSuccess) {
                if (cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) == cudaSuccess) {
                    std::fprintf(stderr, "strata: native embedding: STRATA_EMBED_VRAM=1: %llu MiB placed in VRAM "
                                         "(%llu MiB were free)\n",
                                 (unsigned long long) (bytes_ >> 20), (unsigned long long) (free_b >> 20));
                    dev_ = d;   // host_ stays null: the destructor frees dev_
                    type_ = (int) t->type;
                    n_embd_ = n_embd;
                    n_vocab_ = n_vocab;
                    return true;
                }
                (void) cudaFree(d);
            }
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata: native embedding: STRATA_EMBED_VRAM=1 not taken (%llu MiB free, the table "
                                 "needs %llu MiB plus 512 MiB): kept in mapped host memory\n",
                         (unsigned long long) (have_free ? free_b >> 20 : 0), (unsigned long long) (bytes_ >> 20));
        }
        cudaError_t host_err = cudaHostAlloc(&host_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable);
        if (host_err != cudaSuccess) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            cudaGetLastError();
            host_ = nullptr;
            void* d = nullptr;
            if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                if (d) cudaFree(d);
                cudaGetLastError();
                err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB: " +
                      cudaGetErrorString(host_err) + ", nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            void* d = nullptr;
#if defined(STRATA_USE_HIP)
            // #325: the Windows HIP stack can refuse the device alias of a mapped allocation (and, when it gives
            // one, it is the host address itself - unified addressing; kernels read it correctly there, a
            // device-to-device copy into it does not land: tests/hip/mapped_alias). The table is only gathered from,
            // so without an alias it goes into VRAM like the unpinnable case above, instead of failing the start.
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                cudaGetLastError();
                d = nullptr;
                if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                    cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (d) cudaFree(d);
                    cudaGetLastError();
                    err = "native embedding: no device alias for the mapped table, and no VRAM to copy it into";
                    return false;
                }
                cudaFreeHost(host_);   // the destructor frees dev_ when host_ is null
                host_ = nullptr;
                std::fprintf(stderr, "strata: native embedding: no device alias for the mapped table, kept in VRAM\n");
            }
#else
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
#endif
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
