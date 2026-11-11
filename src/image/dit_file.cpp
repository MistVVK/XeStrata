// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/dit_file.cpp - see include/strata/image/dit_file.hpp.  What each reader adds to the file form it wraps: the
// name a model's tensors carry, the dtype conversions the trainers and the DiTs read (BF16 widened to FP16's bits, a
// matrix's rows), and the cache drops of a mapped file read once.
#include "strata/image/dit_file.hpp"

#include "strata/image/anima_dit.hpp"   // e4m3, an fp8 save's value

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace strata::image {

namespace {

constexpr uint32_t kF32 = 0, kF16 = 1, kBF16 = 30;   // GGUF's type numbers (the 16-bit file a trainer reads)

}  // namespace

QiFile::QiFile(const std::string& path) : g_(path) {}

const TensorInfo* QiFile::find(const std::string& name) const {
    return g_.find("model.diffusion_model." + name);
}

const TensorInfo& QiFile::get(const std::string& name) const {
    const TensorInfo* t = find(name);
    if (t == nullptr) throw std::runtime_error("image engine: the DiT has no " + name);
    return *t;
}

std::vector<float> QiFile::f32(const std::string& name) const {
    const TensorInfo& t = get(name);
    const auto n = (int64_t) t.elements();
    std::vector<float> v((size_t) n);
    const uint8_t* p = data(t);
    for (int64_t i = 0; i < n; ++i) {
        if (t.type == kF32) std::memcpy(&v[(size_t) i], p + 4 * i, 4);
        else if (t.type == kF16 || t.type == kBF16) {
            uint16_t b;
            std::memcpy(&b, p + 2 * i, 2);
            v[(size_t) i] = t.type == kF16 ? kernels::f32_from_f16(b) : kernels::f32_from_bf16(b);
        } else {
            throw std::runtime_error("image engine: " + name + ": a vector of type " + std::to_string(t.type));
        }
    }
    return v;
}

MossFile::MossFile(const std::string& path) {
    std::vector<std::string> files;
    if (std::filesystem::is_directory(path)) {
        for (const auto& e : std::filesystem::directory_iterator(path))
            if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(path);
    }
    if (files.empty()) throw std::runtime_error("image engine (MOSS-Image DiT): no .safetensors in " + path);
    for (const std::string& f : files)
        s_.push_back({std::make_unique<SafeTensors>(f), std::make_unique<platform::MappedFile>(f)});
}

const SafeTensor* MossFile::find(const std::string& name, const Shard** at) const {
    for (const Shard& x : s_)
        if (const SafeTensor* t = x.st->find(name); t != nullptr) {
            if (at != nullptr) *at = &x;
            return t;
        }
    return nullptr;
}

const SafeTensor& MossFile::bf16(const std::string& name, const uint8_t** data) const {
    const Shard* at = nullptr;
    const SafeTensor* t = find(name, &at);
    if (t == nullptr) throw std::runtime_error("image engine (MOSS-Image DiT): no " + name);
    if (t->dtype != "BF16") throw std::runtime_error("image engine (MOSS-Image DiT): " + name + " is not bf16");
    if (data != nullptr) *data = at->map->data() + at->st->data_base() + t->begin;
    return *t;
}

std::vector<float> MossFile::f32(const std::string& name) const {
    const Shard* at = nullptr;
    if (find(name, &at) == nullptr) throw std::runtime_error("image engine (MOSS-Image DiT): no " + name);
    return at->st->f32(name);
}

void MossFile::rows16(const std::string& name, std::vector<uint16_t>& out, int64_t& N, int64_t& K) const {
    const uint8_t* data = nullptr;
    const SafeTensor& t = bf16(name, &data);
    if (t.shape.size() != 2)
        throw std::runtime_error("image engine (MOSS-Image training): " + name + " is not 2D");
    N = t.shape[0];
    K = t.shape[1];
    const auto* src = reinterpret_cast<const uint16_t*>(data);
    const size_t o = out.size();
    out.resize(o + (size_t) (N * K));
    for (int64_t k = 0; k < N * K; ++k)
        out[o + (size_t) k] = kernels::f16_from_f32(kernels::f32_from_bf16(src[k]));
}

void MossFile::drop_cache(const std::string& name) const {
    const Shard* at = nullptr;
    if (const SafeTensor* t = find(name, &at); t != nullptr)
        at->map->drop_cache(at->st->data_base() + t->begin, t->end - t->begin);
}

AnimaFile::AnimaFile(const std::string& path) : st_(path), map_(path) {
    for (const char* p : {"", "net.", "model.diffusion_model.", "diffusion_model."})
        if (st_.find(std::string(p) + "llm_adapter.embed.weight") != nullptr) {
            pre_ = p;
            return;
        }
    throw std::runtime_error("image engine (Anima DiT): " + path + " has no llm_adapter.embed.weight");
}

bool AnimaFile::is_dit(const std::string& path) {
    if (path.size() < 12 || path.compare(path.size() - 12, 12, ".safetensors") != 0) return false;
    try {
        SafeTensors st(path);
        for (const char* p : {"", "net.", "model.diffusion_model.", "diffusion_model."})
            if (st.find(std::string(p) + "llm_adapter.embed.weight") != nullptr &&
                st.find(std::string(p) + "x_embedder.proj.1.weight") != nullptr)
                return true;
    } catch (const std::exception&) {
        return false;   // not a safetensors file this engine reads: not Anima's
    }
    return false;
}

const SafeTensor* AnimaFile::find(const std::string& name) const { return st_.find(pre_ + name); }

const SafeTensor& AnimaFile::get(const std::string& name) const {
    const SafeTensor* t = find(name);
    if (t == nullptr) throw std::runtime_error("image engine (Anima DiT): no " + name);
    return *t;
}

const uint16_t* AnimaFile::bf16(const std::string& name, std::vector<uint16_t>& tmp) {
    const SafeTensor& t = get(name);
    if (t.dtype == "BF16") return reinterpret_cast<const uint16_t*>(raw(t));
    const std::vector<float> v = f32(name);
    tmp.resize(v.size());
    for (size_t i = 0; i < v.size(); ++i) tmp[i] = kernels::bf16_from_f32(v[i]);
    return tmp.data();
}

std::vector<float> AnimaFile::f32(const std::string& name) {
    const SafeTensor& t = get(name);
    if (t.dtype != "F8_E4M3") return st_.f32(pre_ + name);
    // scaled fp8: the weight times its scale_weight (one value), where the file has one
    float scale = 1.f;
    const std::string base = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0
                                 ? name.substr(0, name.size() - 7) : name;
    if (find(base + ".scale_weight") != nullptr) scale = st_.f32(pre_ + base + ".scale_weight").at(0);
    const uint8_t* p = raw(t);
    std::vector<float> out((size_t) t.elements());
    for (size_t i = 0; i < out.size(); ++i) out[i] = anima_dit_detail::e4m3(p[i]) * scale;
    return out;
}

void AnimaFile::rows16(const std::string& name, int64_t cols, std::vector<uint16_t>& h, int64_t& N, int64_t& K) {
    const SafeTensor& t = get(name + ".weight");
    if (t.shape.size() != 2) throw std::runtime_error("image engine (Anima training): " + name + " is not 2D");
    N = t.shape[0];
    K = t.shape[1];
    const int64_t Kc = cols > 0 ? cols : K;
    const uint8_t* src = raw(t);
    h.resize((size_t) (N * Kc));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < Kc; ++k) {
            const size_t i = (size_t) (n * K + k);
            float v;
            if (t.dtype == "BF16") v = kernels::f32_from_bf16(reinterpret_cast<const uint16_t*>(src)[i]);
            else if (t.dtype == "F16") v = kernels::f32_from_f16(reinterpret_cast<const uint16_t*>(src)[i]);
            else if (t.dtype == "F32") v = reinterpret_cast<const float*>(src)[i];
            else throw std::runtime_error("image engine (Anima training): " + name + " is " + t.dtype);
            h[(size_t) (n * Kc + k)] = kernels::f16_from_f32(v);
        }
}

void AnimaFile::drop_cache(const std::string& name) const {
    if (const SafeTensor* t = find(name); t != nullptr) map_.drop_cache(st_.data_base() + t->begin, t->end - t->begin);
}

}  // namespace strata::image
