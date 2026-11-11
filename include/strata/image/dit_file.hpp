// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/dit_file.hpp - a DiT's weight file read the one way its engine and its trainer both take it: a
// name's tensor, its rows as FP16 bits (what a training step's products read) and its vectors as FP32 (the norms,
// the modulations' biases).  One reader a file form: QiFile the GGUF Qwen-Image-2.1's DiT and trainer read, MossFile
// the safetensors shards MOSS-Image's, AnimaFile the safetensors file Anima's (its prefix among the names it may
// carry).  The DiTs read the weights themselves through these (qi_dit.cpp, moss_dit.cpp, anima_dit.cpp), and the
// trainers load their frozen base through the same reader (qi_trainer.cpp, moss_trainer.cpp, anima_trainer.cpp).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/platform/mapped_file.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// Qwen-Image-2.1's DiT file: a GGUF whose weights are "model.diffusion_model." + the name.
class QiFile {
public:
    explicit QiFile(const std::string& path);
    const GgufFile& gguf() const { return g_; }
    /// The tensor "model.diffusion_model." + name, null when there is none
    const TensorInfo* find(const std::string& name) const;
    /// The same, thrown on when there is none
    const TensorInfo& get(const std::string& name) const;
    const uint8_t* data(const TensorInfo& t) const { return g_.tensor_data(t); }
    /// A vector's values as FP32 (F32, F16 or BF16)
    std::vector<float> f32(const std::string& name) const;

private:
    GgufFile g_;
};

/// MOSS-Image's DiT file: its transformer's shards, a folder's *.safetensors or one file, a name looked up in each.
class MossFile {
public:
    struct Shard {
        std::unique_ptr<SafeTensors> st;
        std::unique_ptr<platform::MappedFile> map;
    };
    explicit MossFile(const std::string& path);
    /// The tensor and its shard, or null
    const SafeTensor* find(const std::string& name, const Shard** at = nullptr) const;
    bool has(const std::string& name) const { return find(name) != nullptr; }
    /// A bf16 tensor: its bytes in the mapping
    const SafeTensor& bf16(const std::string& name, const uint8_t** data) const;
    std::vector<float> f32(const std::string& name) const;
    /// A bf16 matrix's rows as FP16 bits, N x K (the mapping's bytes, not an intermediate)
    void rows16(const std::string& name, std::vector<uint16_t>& out, int64_t& N, int64_t& K) const;
    void drop_cache(const std::string& name) const;

private:
    std::vector<Shard> s_;
};

/// Anima's DiT file: a safetensors whose tensors carry one of the prefixes it may have.
class AnimaFile {
public:
    explicit AnimaFile(const std::string& path);
    /// A file's path is Anima's DiT (its llm adapter's embedder and its patch embedder are in it)
    static bool is_dit(const std::string& path);
    const std::string& prefix() const { return pre_; }
    const SafeTensor* find(const std::string& name) const;
    const SafeTensor& get(const std::string& name) const;
    const uint8_t* raw(const SafeTensor& t) const { return map_.data() + st_.data_base() + t.begin; }
    /// A tensor's values as bf16: the mapping itself for a BF16 one, else converted into tmp
    const uint16_t* bf16(const std::string& name, std::vector<uint16_t>& tmp);
    std::vector<float> f32(const std::string& name);
    /// A matrix's rows as FP16 bits (BF16, F16 or F32 in the file), its first `cols` columns (0: every one), `name`
    /// without its ".weight"; N and K the file's shape
    void rows16(const std::string& name, int64_t cols, std::vector<uint16_t>& h, int64_t& N, int64_t& K);
    void drop_cache(const std::string& name) const;

private:
    SafeTensors st_;
    platform::MappedFile map_;
    std::string pre_;
};

}  // namespace strata::image
