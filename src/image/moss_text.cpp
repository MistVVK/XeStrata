// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_text.cpp - see include/strata/image/moss_text.hpp.

#include "strata/image/moss_text.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_spec.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

// The engine's blocks from the layer stream: block l is layer l
struct MossText::Source : core::PlainModel::BlockSource {
    LayerStream* layers = nullptr;
    const uint8_t* begin(int l) override { return layers->begin(l); }
    void end(int l) override { layers->end(l); }
};

// The adapters after the blocks' products (PlainModel::ProductHook): a layer's product's deltas, by the layer and the
// product
struct MossText::Hook : core::PlainModel::ProductHook {
    std::map<std::pair<int64_t, int>, std::unique_ptr<LoraLayer>> layers;
    std::vector<std::shared_ptr<const LoraFile>> files;
    std::string key;
    prefill::Gemm gemm;
    bool gemm_ready = false;
    int64_t cols = 0, cap_t = 0;
    float* f32 = nullptr;
    uint16_t* f16 = nullptr;
    void* stream = nullptr;
    ~Hook() override { free_scratch(); }
    void free_scratch() {
        if (f32 != nullptr || f16 != nullptr) gpu::stream_sync(stream);
        gpu::free(f32);
        gpu::free(f16);
        f32 = nullptr;
        f16 = nullptr;
        cap_t = 0;
    }
    void after(int64_t layer, Which which, const uint16_t* x16, int64_t T, float* y, int64_t ldy) override {
        const auto it = layers.find({layer, (int) which});
        if (it == layers.end()) return;
        if (T > cap_t) {
            free_scratch();
            if (!gpu::alloc_device(&f32, (size_t) (T * cols) * 4) || !gpu::alloc_device(&f16, (size_t) (T * cols) * 2))
                throw std::runtime_error(std::string("MOSS-Image's text encoder's adapters: GPU memory: ") +
                                         gpu::last_error());
            cap_t = T;
        }
        int64_t n_rows = 0;
        for (const LoraPart& p : it->second->parts) n_rows = std::max(n_rows, p.row0 + p.n);
        it->second->add(x16, T, y, ldy, 0, n_rows, nullptr, gemm, f32, f16, stream);
    }
};

MossText::MossText() = default;
MossText::~MossText() { close(); }

const std::string& MossText::lora_key() const {
    static const std::string none;
    return hook_ ? hook_->key : none;
}

bool MossText::set_loras(const std::vector<LoraUse>& loras, std::string& err) {
    const std::string key = lora_key_of(loras);
    if (key == lora_key()) return true;
    if (model_) model_->set_product_hook(nullptr);
    hook_.reset();
    if (key.empty()) return true;
    if (!model_) {
        err = "MOSS-Image's text encoder is not open";
        return false;
    }
    using H = core::PlainModel::ProductHook;
    static const std::pair<const char*, H::Which> kNames[] = {
        {"linear.attn.in.proj.qkv", H::kQkv}, {"linear.attn.in.proj.z", H::kZ}, {"linear.attn.in.proj.a", H::kAlpha},
        {"linear.attn.in.proj.b", H::kBeta},  {"linear.attn.out.proj", H::kSsmOut}, {"self.attn.q.proj", H::kQ},
        {"self.attn.k.proj", H::kK},          {"self.attn.v.proj", H::kV},          {"self.attn.o.proj", H::kO},
        {"mlp.gate.proj", H::kGate},          {"mlp.up.proj", H::kUp},              {"mlp.down.proj", H::kDown}};
    const ModelSpec& s = model_->spec();
    // llama.cpp's converter puts the GDN's value heads from grouped (by key head) to tiled order where the key heads
    // are fewer: head (g, r) of HK x (HV / HK) goes to r HK + g
    const int64_t HK = s.ssm_n_group, HV = s.ssm_dt_rank, dk = s.ssm_d_state, dv = HV > 0 ? s.ssm_d_inner / HV : 0;
    const bool reorder = HK > 0 && HV > 0 && HK != HV;
    auto tiled = [&](int64_t i, int64_t hd) {   // the GGUF's index of the file's index i (head size hd)
        const int64_t r_per = HV / HK, head = i / hd, d = i % hd, g = head / r_per, r = head % r_per;
        return (r * HK + g) * hd + d;
    };
    auto h = std::make_unique<Hook>();
    h->stream = model_->stream();
    if (!h->gemm.init(h->stream, 0, err)) return false;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        for (const LoraEntry& e0 : u.file->entries()) {
            const std::string name = lora_text_encoder_name(e0.key);
            int64_t layer = -1;
            const char* rest = nullptr;
            if (name.rfind("layers.", 0) == 0) {
                const size_t dot = name.find('.', 7);
                if (dot != std::string::npos) {
                    layer = std::atoll(name.substr(7, dot - 7).c_str());
                    rest = name.c_str() + dot + 1;
                }
            }
            const auto* hit = rest == nullptr ? nullptr
                                              : std::find_if(std::begin(kNames), std::end(kNames),
                                                             [&](const auto& n) { return std::strcmp(n.first, rest) == 0; });
            int64_t rows = 0, cols = 0;
            if (hit == nullptr || hit == std::end(kNames) || !model_->product_shape(layer, hit->second, rows, cols)) {
                err = "image engine: " + u.file->path() + ": " + e0.key + ": MOSS-Image's text encoder has no such weight";
                return false;
            }
            if ((e0.kind != LoraKind::lora && e0.kind != LoraKind::diff) || !e0.dora.empty() || !e0.bias.empty() ||
                e0.replaces || e0.rows != rows || e0.cols != cols) {
                err = "image engine: " + u.file->path() + ": " + e0.key + ": MOSS-Image's text encoder takes a plain "
                      "LoRA or a whole delta of its weight's " + std::to_string(cols) + " x " + std::to_string(rows);
                return false;
            }
            LoraEntry e = e0;   // the GDN's value heads in the GGUF's order
            const H::Which w = hit->second;
            if (reorder && (w == H::kQkv || w == H::kZ || w == H::kAlpha || w == H::kBeta || w == H::kSsmOut)) {
                const int64_t first = w == H::kQkv ? 2 * HK * dk : 0, hd = (w == H::kAlpha || w == H::kBeta) ? 1 : dv;
                const bool by_cols = w == H::kSsmOut;
                const int64_t n = by_cols ? cols : rows, inner = e.kind == LoraKind::lora ? e.rank : (by_cols ? rows : cols);
                if (by_cols) {   // the columns: lora's a [rank][cols], a delta's [rows][cols]
                    std::vector<uint16_t>& m = e.kind == LoraKind::lora ? e.a : e.delta;
                    const std::vector<uint16_t> src = m;
                    for (int64_t r = 0; r < inner; ++r)
                        for (int64_t c = 0; c < n; ++c) m[(size_t) (r * n + tiled(c, hd))] = src[(size_t) (r * n + c)];
                } else {          // the rows: lora's b [rows][rank], a delta's [rows][cols]
                    std::vector<uint16_t>& m = e.kind == LoraKind::lora ? e.b : e.delta;
                    const std::vector<uint16_t> src = m;
                    for (int64_t r = first; r < n; ++r)
                        std::copy_n(src.begin() + r * inner, inner, m.begin() + (first + tiled(r - first, hd)) * inner);
                }
            }
            const float scale = u.scale * e.scale();
            if (scale == 0.0f) continue;
            auto dev = std::make_shared<LoraDeviceEntry>();
            if (!dev->upload(e, h->stream, err)) return false;
            auto& layer_ptr = h->layers[{layer, (int) w}];
            if (!layer_ptr) layer_ptr = std::make_unique<LoraLayer>();
            LoraPart part;
            part.entry = dev;
            part.n = rows;
            part.scale = scale;
            layer_ptr->parts.push_back(std::move(part));
            h->cols = std::max(h->cols, lora_scratch_cols(e));
        }
        h->files.push_back(u.file);
    }
    h->key = key;
    hook_ = std::move(h);
    model_->set_product_hook(hook_.get());
    return true;
}

bool MossText::open(const std::string& gguf, int64_t max_tokens, std::string& err, uint64_t layer_budget) {
    close();
    core::PlainOptions o;
    o.gguf = gguf;
    o.encoder = true;
    o.stream_blocks = true;
    o.vision = true;   // pictures among the prompt's tokens (an ai-toolkit edit adapter's references)
    // the prompt in one chunk of its length (the encoder's states do not depend on the chunking, its K/V cells are the
    // prompt's), the windows of one token it never runs
    o.max_cells = max_tokens;
    o.prompt_chunk = max_tokens;
    o.prompt_chunk_given = true;
    o.max_t = 1;
    auto m = std::make_unique<core::PlainModel>();
    auto layers = std::make_unique<LayerStream>();
    auto src = std::make_unique<Source>();
    try {
        if (!m->load(o, err)) return false;
        std::vector<std::vector<uint64_t>> offsets;
        for (int64_t l = 0; l < m->spec().n_layer; ++l) {
            std::vector<LayerStream::Span> spans;
            for (const core::PlainModel::HostSpan& h : m->block_spans(l)) spans.push_back({h.host, h.bytes, 0});
            const int i = layers->add(spans);
            offsets.emplace_back();
            for (size_t k = 0; k < spans.size(); ++k) offsets.back().push_back(layers->offset(i, (int) k));
        }
        core::PlainModel* mp = m.get();
        layers->set_prepare([mp](int l, uint8_t* d) { mp->prepare_block(l, d); });
        layers->configure(std::min(std::max(layer_budget, layers->min_budget()), layers->total()), m->stream());
        src->layers = layers.get();
        m->set_block_source(src.get(), std::move(offsets));
        // the products' kernels loaded and the dense products' forms timed, as the chat's engine does at its start
        if (!m->prompt_warm(err)) return false;
    } catch (const std::exception& e) {
        err = std::string("MOSS-Image's text encoder: ") + e.what();
        return false;
    }
    try {
        image_pad_ = (int32_t) GgufModel::open(gguf).meta().string_index("tokenizer.ggml.tokens", "<|image_pad|>");
    } catch (const std::exception&) {
        image_pad_ = -1;   // no such token: no pictures
    }
    layers_ = std::move(layers);
    source_ = std::move(src);
    model_ = std::move(m);
    return true;
}

// the layer stream first: it waits for the model's stream (its compute stream), which the model destroys with itself
void MossText::close() {
    if (model_) model_->set_product_hook(nullptr);
    hook_.reset();
    layers_.reset();
    source_.reset();
    model_.reset();
    pos_set_ = 0;
}

void MossText::set_layer_budget(uint64_t bytes) {
    if (layers_ != nullptr)
        layers_->configure(std::min(std::max(bytes, layers_->min_budget()), layers_->total()), model_->stream());
}

std::vector<float> MossText::encode(const std::vector<int32_t>& ids, std::string& err,
                                    const std::vector<Picture>& pictures) {
    const int32_t pad = image_pad_;
    if (model_ == nullptr) {
        err = "MOSS-Image's text encoder is not open";
        return {};
    }
    const int64_t n = (int64_t) ids.size(), N = hidden();
    // the cells' positions: the pictures' grids, the text's one after another (the identity where there are none,
    // also over the cells an earlier prompt's pictures set)
    std::vector<int32_t> pos3;
    std::vector<const float*> rows;
    if (!pictures.empty() || pos_set_ > 0) {
        const int64_t cells = std::max(n, pos_set_);
        pos3.resize((size_t) cells * 3);
        rows.assign((size_t) n, nullptr);
        int64_t p = 0, i = 0;
        size_t k = 0;
        while (i < n) {
            if (ids[(size_t) i] != pad || k >= pictures.size()) {
                pos3[(size_t) i * 3] = pos3[(size_t) i * 3 + 1] = pos3[(size_t) i * 3 + 2] = (int32_t) p++;
                ++i;
                continue;
            }
            const Picture& pic = pictures[k++];
            for (int64_t j = 0; j < pic.nx * pic.ny; ++j) {
                if (i + j >= n || ids[(size_t) (i + j)] != pad) {
                    err = "MOSS-Image's text encoder: picture " + std::to_string(k) + " has fewer <|image_pad|> tokens "
                          "than its " + std::to_string(pic.nx * pic.ny) + " rows";
                    return {};
                }
                pos3[(size_t) (i + j) * 3] = (int32_t) p;
                pos3[(size_t) (i + j) * 3 + 1] = (int32_t) (p + j / pic.nx);
                pos3[(size_t) (i + j) * 3 + 2] = (int32_t) (p + j % pic.nx);
                rows[(size_t) (i + j)] = pic.rows + j * N;
            }
            i += pic.nx * pic.ny;
            p += std::max(pic.nx, pic.ny);
        }
        if (k != pictures.size()) {
            err = "MOSS-Image's text encoder: more pictures than runs of <|image_pad|> tokens";
            return {};
        }
        for (int64_t c = n; c < cells; ++c, ++p) pos3[(size_t) c * 3] = pos3[(size_t) c * 3 + 1] = pos3[(size_t) c * 3 + 2] = (int32_t) p;
        if (pictures.empty())   // back to the identity
            for (int64_t c = 0; c < cells; ++c) pos3[(size_t) c * 3] = pos3[(size_t) c * 3 + 1] = pos3[(size_t) c * 3 + 2] = (int32_t) c;
        if (!model_->set_positions(pos3.data(), cells, err)) return {};
        pos_set_ = pictures.empty() ? 0 : n;
    }
    std::vector<float> out(ids.size() * (size_t) N);
    try {
        if (!pictures.empty()) model_->set_embeddings(&rows);
        const bool ok = model_->encode(ids.data(), n, out.data(), err);
        model_->set_embeddings(nullptr);
        if (!ok) return {};
    } catch (const std::exception& e) {   // a streamed block's copy failed
        model_->set_embeddings(nullptr);
        err = std::string("MOSS-Image's text encoder: ") + e.what();
        return {};
    }
    return out;
}

int64_t MossText::hidden() const { return model_ != nullptr ? model_->spec().n_embd : 0; }

bool MossText::read_states(const std::string& path, std::vector<float>& rows, int64_t& n, int64_t& width,
                           std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    int32_t hdr[3] = {};
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    bool ok = f && hdr[0] == 0x31455453 && hdr[1] > 0 && hdr[2] > 0;
    if (ok) {
        n = hdr[1];
        width = hdr[2];
        rows.resize((size_t) (n * width));
        f.read(reinterpret_cast<char*>(rows.data()), (std::streamsize) (rows.size() * sizeof(float)));
        ok = f && f.peek() == std::char_traits<char>::eof();
    }
    if (!ok) err = path + " is not a states file (STE1)";
    return ok;
}

}  // namespace strata::image
