// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 hiruhiru2005
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/native_head.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/run_manifest.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace strata::core {

NativeHead::~NativeHead() {
    if (scratch_) strata::gpu::free(scratch_);
    if (weights_) strata::gpu::free(weights_);
}

bool NativeHead::load(const std::string& path, int64_t n_in, int64_t n_out, std::string& err) {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard of `path`'s model
        // holds it (shard 2 of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor; upstream 02cfe36).  GgufModel
        // refuses a missing shard and a duplicate across shards.
        const strata::GgufModel model = strata::GgufModel::open(path);
        err = strata::check_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const DenseNqEntry* nq = dense_nq_find("output.weight");
        std::vector<uint8_t> nq_data;
        if (nq && !dense_nq_read("output.weight", n_in, n_out, nq_data, err)) return false;
        const int source_type = nq ? nq->type : (int) tensor->type;
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes(source_type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (!nq && (tensor->offset > payload || bytes > payload - tensor->offset)) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        bool status = strata::gpu::alloc_device(&weights, bytes);
        if (status)
            status = strata::gpu::alloc_device(&scratch, strata::kernels::native_q8_1_bytes((int) n_in, 1));
        if (!nq) strata::platform::advise_willneed(gguf.tensor_data(*tensor), bytes);
        if (status)
            status = strata::gpu::copy(weights, nq ? nq_data.data() : gguf.tensor_data(*tensor), bytes);
        if (!status) {
            if (scratch) strata::gpu::free(scratch);
            if (weights) strata::gpu::free(weights);
            err = std::string("native head upload: ") + strata::gpu::last_error();
            return false;
        }
        int type = source_type;
        if (type == 14 && strata::kernels::native_q6_k_rows_ok((int) n_in)) {   // aligned loads (native_mmvq.hpp)
            sycl::queue& q = strata::core::Runtime::get().compute();
            strata::kernels::native_q6_k_to_rows(weights, (int) n_in, (int) n_out, &q);
            q.wait();
            type = strata::kernels::kNativeQ6KRows;
        } else if (type == 8 && strata::kernels::native_q8_0_rows_ok((int) n_in)) {
            sycl::queue& q = strata::core::Runtime::get().compute();
            strata::kernels::native_q8_0_to_rows(weights, (int) n_in, (int) n_out, &q);
            q.wait();
            type = strata::kernels::kNativeQ8Rows;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = type;
        if (strata::core::manifest::enabled()) {   // upstream 89f4fa18 (XESTRATA_MANIFEST): what the head holds and where
            namespace mf = strata::core::manifest;
            mf::instance().tensors("head", strata::ggml_type_name((uint32_t) source_type), 1, bytes);
            mf::instance().path("head", "output.weight", true, "held in VRAM",
                                std::string("type ") + strata::ggml_type_name((uint32_t) source_type) +
                                    (type == strata::kernels::kNativeQ6KRows ? ", repacked (aligned rows)" : "") +
                                    (type == strata::kernels::kNativeQ8Rows ? ", repacked (aligned rows)" : ""));
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
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
    return true;
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    for (void* c : copies_)
        if (c) strata::gpu::free(c);
    if (host_) strata::gpu::free(host_);
    else if (dev_) strata::gpu::free(dev_);   // a table placed in its own VRAM block
}

const void* NativeEmbed::table_for(void* stream) const {
    const Runtime* table_rt = Runtime::owner(host_ ? host_ : dev_);
    const Runtime* run_rt = stream ? Runtime::owner_of_stream(stream) : &Runtime::get();
    if (!table_rt || !run_rt || table_rt == run_rt) return dev_;
    const int dev = run_rt->ordinal();
    std::lock_guard<std::mutex> lock(copies_mutex_);
    if (copies_.size() <= (size_t) dev) copies_.resize((size_t) dev + 1, nullptr);
    if (!copies_[(size_t) dev]) {
        void* c = nullptr;
        if (host_) {
            const OnDevice on(dev);
            if (!strata::gpu::alloc_host(&c, bytes_))
                throw std::runtime_error("native embedding: cannot pin a copy of " + std::to_string(bytes_ >> 20) +
                                         " MiB for GPU " + std::to_string(dev));
            std::memcpy(c, host_, bytes_);
        } else {
            // a table in one GPU's VRAM, which another GPU's context cannot read: that GPU gets a copy of its own,
            // through the host
            std::vector<uint8_t> bounce(bytes_);
            bool ok = false;
            {
                const OnDevice from(table_rt->ordinal());
                ok = strata::gpu::copy(bounce.data(), dev_, bytes_);
            }
            const OnDevice on(dev);
            if (ok) ok = strata::gpu::alloc_device(&c, bytes_);
            if (ok) ok = strata::gpu::copy(c, bounce.data(), bytes_);
            if (!ok) {
                if (c) strata::gpu::free(c);
                throw std::runtime_error("native embedding: cannot copy the " + std::to_string(bytes_ >> 20) +
                                         " MiB table into GPU " + std::to_string(dev) + "'s VRAM");
            }
        }
        copies_[(size_t) dev] = c;
    }
    return copies_[(size_t) dev];
}

bool NativeEmbed::load(const std::string& path, int64_t n_embd, int64_t n_vocab, std::string& err) {
    try {
        // token_embd.weight from whichever shard of `path`'s model holds it (upstream 02cfe36)
        const strata::GgufModel model = strata::GgufModel::open(path);
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
        if (!strata::gpu::alloc_host(&host_, bytes_)) {
            host_ = nullptr;
            err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB";
            return false;
        }
        std::memcpy(host_, gguf.tensor_data(*t), bytes_);
        void* d = nullptr;
        if (!strata::gpu::device_pointer(&d, host_)) {
            err = "native embedding: no device alias for the mapped table";
            return false;
        }
        dev_ = d;
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

strata::kernels::EmbedTable NativeEmbed::table(void* stream) const {
    strata::kernels::EmbedTable t;
    t.type = type_;
    t.rows = table_for(stream);
    t.row_bytes = row_;
    t.n_rows = n_vocab_;
    t.n_embd = n_embd_;
    return t;
}

}  // namespace strata::core
