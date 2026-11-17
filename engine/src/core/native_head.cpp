#include "strata/core/native_head.hpp"
#include "strata/core/spill_gate.hpp"
#include "strata/core/run_manifest.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::core {

NativeHead::~NativeHead() {
    unload();
}

void NativeHead::unload() {
    if (weights_) strata::kernels::native_q6_k_unpack(weights_);
    if (weights_) strata::kernels::native_q8_0_sm60_release(weights_);
    if (scratch_) cudaFree(scratch_);
    if (host_) cudaFreeHost(host_);   // the mapped-host fallback (weights_ is its device alias)
    else if (weights_) cudaFree(weights_);
    scratch_ = nullptr;
    weights_ = nullptr;
    host_ = nullptr;
}

bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err) {
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
        // SHORT OF VRAM: the head in mapped pinned host memory.  The head (about 0.4 GiB) is the one buffer a card that holds its sessions and weights but not this can
        // spill whole; every logits row then reads it over PCIe (about 18 ms per row at 28 GB/s), so decode is slower, but the start works.  Taken only when the
        // allocation in VRAM fails (or STRATA_HEAD_HOST=1 asks for it); the GEMV kernels read the same pointer either way.  Q5_K and the plain MMVQ types: the
        // repacked Q6_K / SM60 layouts rewrite the weights in place and stay in VRAM.
        const bool ask_host = std::getenv("STRATA_HEAD_HOST") != nullptr;
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
    if (weights_ == nullptr || host_ != nullptr || (type_ == 14 && strata::kernels::native_q6_k_packed_enabled()) || (type_ == 8 && std::getenv("STRATA_Q8_SM60") != nullptr)) return false;   // (repacked in place: those stay in VRAM)
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
