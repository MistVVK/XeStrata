// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/dit_loras.cpp - the DiTs' one adapter binder (dit_loras.hpp), which each DiT had its own copy of.
#include "strata/image/dit_loras.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/gpu_buf.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace strata::image {

namespace {

// A file's key that changes a text encoder (kohya's lora_te_, diffusers' text_encoder. prefix, the te. other tools
// write): the image engine applies adapters to the DiT only, and says so rather than leaving it out.
bool is_text_encoder_key(const std::string& k) {
    return k.rfind("lora_te", 0) == 0 || k.rfind("text_encoder", 0) == 0 || k.rfind("te.", 0) == 0;
}

// Six of the names, then "and N more": the errors' list of a file's keys the DiT does not take.
std::string name_list(const std::vector<std::string>& names) {
    std::string out;
    const size_t n = std::min<size_t>(names.size(), 6);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) out += ", ";
        out += names[i];
    }
    if (names.size() > n) out += " (and " + std::to_string(names.size() - n) + " more)";
    return out;
}

// a weight's blocks' copy the bake rewrites: its form as LoraLayer::bake takes it (0: none it takes)
int bake_type(const DitW& w) {
    if (w.w8 && !w.w8t) return kDitW8;
    return w.type == kDitF16 || w.type == kDitQ8_0 || w.type == kDitW8 ? w.type : 0;
}

}  // namespace

void DitLoras::add(std::string name, DitW& w, int64_t row0, int64_t rows, int64_t src_row, bool biased) {
    if (w.n <= 0 || w.k <= 0 || rows <= 0 || row0 + rows > w.n) return;
    slots_.push_back({std::move(name), &w, row0, rows, src_row, biased});
}

std::vector<DitLoraTarget> DitLoras::targets() const {
    std::vector<DitLoraTarget> out;
    out.reserve(slots_.size());
    for (const DitLoraSlot& s : slots_) out.push_back({s.name, s.row0, s.rows, s.w->k, s.src_row});
    return out;
}

int DitLoras::block_of(const std::string& name) const {
    if (block_path.empty() || name.compare(0, block_path.size(), block_path) != 0) return -1;
    size_t i = block_path.size();
    int b = 0;
    if (i >= name.size() || name[i] < '0' || name[i] > '9') return -1;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) b = b * 10 + (name[i] - '0');
    return i < name.size() && name[i] == '.' ? b : -1;
}

int DitLoras::blocks() const {
    int n = 0;
    for (const DitLoraSlot& s : slots_) n = std::max(n, block_of(s.name) + 1);
    return n;
}

bool DitLoras::set(const std::vector<LoraUse>& loras, std::string& err, bool& changed) {
    changed = false;
    // the set as the same again (its files and strengths): nothing to bind
    const std::string key = lora_key_of(loras);
    if (key == key_) return true;
    changed = true;
    hooks.drop();
    if (key.empty()) return true;
    const auto fail = [this]() {
        hooks.drop();
        return false;
    };
    try {
    void* const s = *s_;
    std::vector<std::string> unknown, text_encoder;
    const int nblocks = blocks();
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        if (!u.blocks.empty() && (int) u.blocks.size() != nblocks) {
            err = "image engine: " + u.file->path() + ": " + std::to_string(u.blocks.size()) +
                  " block weights, and the DiT has " + std::to_string(nblocks) + " blocks";
            return fail();
        }
        if (hooks.file) hooks.file(*u.file);
        for (const LoraEntry& raw : u.file->entries()) {
            LoraEntry moved;
            const LoraEntry* ep = &raw;
            std::string renamed;
            if (hooks.rename && hooks.rename(raw, renamed)) {
                moved = raw;   // the key changes, the entry's tensors do not
                moved.key = std::move(renamed);
                ep = &moved;
            }
            const LoraEntry& entry = *ep;
            if (is_text_encoder_key(entry.key)) {
                text_encoder.push_back(entry.key);
                continue;
            }
            // the slots of the entry's name: one weight, or the weights a fused one is split into
            std::vector<const DitLoraSlot*> slots;
            for (const DitLoraSlot& sl : slots_)
                if (lora_key_names(entry.key, sl.name)) slots.push_back(&sl);
            if (slots.empty()) {
                unknown.push_back(entry.key);
                continue;
            }
            int64_t rows = 0;   // the entry's rows the weight takes
            for (const DitLoraSlot* sl : slots) rows = std::max(rows, sl->src_row + sl->rows);
            // a GLoRA's four tensors may fit both of its readings: the weight's rows and columns say which
            LoraEntry alt;
            const LoraEntry& e = *lora_glora_for_weight(entry, rows, slots[0]->w->k, alt);
            const bool oft = e.kind == LoraKind::oft;
            const bool glora = e.kind == LoraKind::glora;
            const std::string at = "image engine: " + u.file->path() + ": " + e.key;
            if (oft && (slots.size() != 1 || slots[0]->src_row != 0)) {
                // a rotation is one weight's rows, block by block: a name the DiT splits across weights' rows takes none
                err = at + ": an OFT, and the DiT's rows of that name are several weights or a part of one";
                return fail();
            }
            std::string why;
            if (hooks.refuse && hooks.refuse(*slots[0], e, why)) {
                err.assign(at).append(": ").append(why);
                return fail();
            }
            if (e.kind == LoraKind::bias) {   // a bias's delta alone: its rows are the bias's, its columns none
                if (e.rows != rows) {
                    err = at + ": a bias's delta (diff_b/b_norm) of " + std::to_string(e.rows) +
                          " values, not the weight's " + std::to_string(rows) + " rows";
                    return fail();
                }
            } else if (e.rows != rows || (oft ? e.cols != 0 : e.cols != slots[0]->w->k)) {
                err = at + " is " + std::to_string(e.cols) + " x " + std::to_string(e.rows) + ", not the DiT's " +
                      std::to_string(slots[0]->w->k) + " x " + std::to_string(rows);
                return fail();
            }
            if (!e.bias.empty()) {   // a bias's delta is added where the weight adds its bias: the weight must have one
                for (const DitLoraSlot* sl : slots)
                    if (!sl->biased) {
                        err = at + ": a bias's delta (diff_b/b_norm), and the DiT's weight adds no bias (its block's "
                                   "modulation does the shift)";
                        return fail();
                    }
            }
            // the request's strength, times the factor of the weight's block where the request gives one a block
            // (the slots of an entry share its name, so its block)
            const int blk = u.blocks.empty() ? -1 : block_of(slots[0]->name);
            const float us = blk >= 0 ? u.scale * u.blocks[(size_t) blk] : u.scale;
            const float scale = us * e.scale();
            // a strength of 0 changes nothing: the weight keeps its product's path (an alpha of 0 leaves the bias's
            // delta and a DoRA's base factors, as a DoRA's decomposition scales the weight with no delta)
            if (us == 0.0f || (scale == 0.0f && e.bias.empty() && e.dora.empty())) continue;
            // a DoRA's: the entry's rows' delta factors (its B's, by the entry's rows), and each part's base factors
            const bool dora = !e.dora.empty();
            std::vector<float> coef((size_t) e.rows, 1.0f);
            std::vector<std::vector<float>> bases(slots.size());
            for (size_t i = 0; i < slots.size(); ++i) {
                const DitLoraSlot* slot = slots[i];
                std::vector<float> norms;
                if (dora && !hooks.dora_norms(*slot, e, norms, err)) return fail();
                if (dora) {
                    std::vector<float> c;
                    if (!lora_dora_rows(e, slot->src_row, slot->rows, norms, us, c, bases[i], why)) {
                        err.assign(at).append(": ").append(why);
                        return fail();
                    }
                    for (int64_t j = 0; j < slot->rows; ++j) coef[(size_t) (slot->src_row + j)] = c[(size_t) j];
                }
                if (e.replaces)   // a set_weight: the weight interpolates to the file's, W + s (S - W)
                    bases[i].assign((size_t) slot->rows, 1.0f - us);
                if (const LoraPart* clash = lora_dora_overlap(slot->w->lora, slot->row0, slot->rows, dora, oft, glora,
                                                              e.replaces)) {
                    err = at + ": the weight's rows " + std::to_string(slot->row0) + " .. " +
                          std::to_string(slot->row0 + slot->rows) + " are another adapter's, and " +
                          lora_patch_clash_why(*clash, oft, glora, e.replaces);
                    return fail();
                }
            }
            auto dev = std::make_shared<LoraDeviceEntry>();
            if (!dev->upload(e, s, err, dora ? coef.data() : nullptr)) return fail();
            for (size_t i = 0; i < slots.size(); ++i) {
                const DitLoraSlot* slot = slots[i];
                if (slot->w->lora == nullptr) {
                    layers_.push_back(std::make_unique<LoraLayer>());
                    slot->w->lora = layers_.back().get();
                }
                LoraPart part;
                part.entry = dev;
                part.row0 = slot->row0;
                part.n = slot->rows;
                part.src_row = slot->src_row;
                part.scale = scale;
                part.bias_scale = us;   // a diff patch has no alpha: the request's strength only
                part.dora = dora;
                part.oft = oft;
                part.glora = glora;
                part.replaces = e.replaces;
                part.base = std::move(bases[i]);
                slot->w->lora->parts.push_back(std::move(part));
            }
            scratch_->cols = std::max(scratch_->cols, lora_scratch_cols(e));
            if (glora) {
                scratch_->glora_cols = std::max(scratch_->glora_cols, e.cols);
                scratch_->glora_rows = std::max(scratch_->glora_rows, e.rows);
            }
        }
    }
    if (!text_encoder.empty()) {
        err = "image engine: a text encoder's adapter: the image engine applies adapters to the DiT only (" +
              name_list(text_encoder) + ")";
        return fail();
    }
    if (!unknown.empty()) {
        err = "image engine: the DiT takes no adapter for " + name_list(unknown);
        return fail();
    }
    if (layers_.empty()) {   // every strength 0: as no adapter at all (the products are untouched)
        hooks.drop();
        return true;
    }
    for (const DitLoraSlot& slot : slots_)   // the base products' row factors, once every part is bound
        if (slot.w->lora != nullptr && !slot.w->lora->build_base(slot.w->n, s, err)) return fail();
    // a GLoRA's own path's buffers, once every part is bound
    if (scratch_->glora_cols > 0) scratch_->glora_alloc(*owned_, what_);
    if (bake) {   // the blocks' plain LoRAs folded into their weights as the blocks come back to the GPU
        std::vector<DitW*> ws;
        hooks.block_weights(ws);
        int64_t nk = 0, kr = 0;
        for (DitW* w : ws)
            if (w->lora != nullptr && w->lora->bakeable() && bake_type(*w) != 0) {
                w->baked = true;
                nk = std::max(nk, w->n * w->k);
                for (const LoraPart& p : w->lora->parts) kr = std::max(kr, w->k * p.entry->rank());
            }
        if (nk > 0) {
            baked_any_ = true;
            bake_tmp_ = alloc<float>(*owned_, nk, what_);
            bake_at16_ = alloc<uint16_t>(*owned_, kr, what_);
            bake_bytes_ = (uint64_t) nk * 4 + (uint64_t) kr * 2;
            hooks.reprepare();   // the queued work drained before the prepare changes
        }
    }
    for (const LoraUse& u : loras)
        if (u.file) files_.push_back(u.file);
    key_ = key;
    return true;
    } catch (...) {
        hooks.drop();
        throw;
    }
}

void DitLoras::drop() {
    for (DitLoraSlot& sl : slots_) {
        sl.w->lora = nullptr;
        sl.w->baked = false;
    }
    if (baked_any_) {   // the weights as the file has them again: the blocks copied anew, without the bake
        baked_any_ = false;
        hooks.reprepare();
        for (void** p : {reinterpret_cast<void**>(&bake_tmp_), reinterpret_cast<void**>(&bake_at16_)})
            if (*p != nullptr) {
                release(*owned_, *p);
                *p = nullptr;
            }
        bake_bytes_ = 0;
    }
    layers_.clear();
    files_.clear();
    key_.clear();
    scratch_->drop(*owned_, *s_);
}

uint64_t DitLoras::bytes() const {
    uint64_t n = 0;
    for (const std::unique_ptr<LoraLayer>& l : layers_) n += l->bytes();
    // a GLoRA's own path's buffers, and the bake's (taken before the plan that places the blocks beside them)
    return n + scratch_->glora_bytes() + bake_bytes_;
}

DitTensorRows dit_bf16_rows(const uint16_t* h, int64_t k) {
    return [h, k](int64_t r0, int64_t n, float* out, std::string&) {
        for (int64_t r = 0; r < n; ++r)
            for (int64_t c = 0; c < k; ++c) out[r * k + c] = kernels::f32_from_bf16(h[(r0 + r) * k + c]);
        return true;
    };
}

bool dit_dora_norms(const LoraEntry& e, int64_t src_row, int64_t trow0, int64_t rows, int64_t k,
                    const DitTensorRows& read, std::vector<float>& norms, std::string& err) {
    const int64_t rank = e.rank;
    const float s = e.scale();
    if (rank <= 0 || k <= 0 || rows <= 0 || (int64_t) e.a.size() < rank * k ||
        (int64_t) e.b.size() < (src_row + rows) * rank) {
        err = "a DoRA's factors are not the weight's shapes";
        return false;
    }
    // the factors as FP32 once, and A A^T [rank][rank] (the delta's own gram, the s^2 term below reads)
    std::vector<float> af((size_t) (rank * k));
    for (size_t i = 0; i < af.size(); ++i) af[i] = kernels::f32_from_f16(e.a[i]);
    std::vector<float> aat((size_t) (rank * rank));
    for (int64_t j = 0; j < rank; ++j)
        for (int64_t l = 0; l <= j; ++l) {
            const float* aj = af.data() + j * k;
            const float* al = af.data() + l * k;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            int64_t c = 0;
            for (; c + 4 <= k; c += 4)
                for (int u = 0; u < 4; ++u) acc[u] += aj[c + u] * al[c + u];
            float t = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            for (; c < k; ++c) t += aj[c] * al[c];
            aat[(size_t) (j * rank + l)] = aat[(size_t) (l * rank + j)] = t;
        }
    norms.assign((size_t) rows, 0.f);
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    constexpr int64_t kBand = 128;   // the weight's rows read at a time (the reader's own bands may be smaller)
    std::vector<float> w((size_t) (std::min(kBand, rows) * k));
    for (int64_t b = 0; b < rows; b += kBand) {
        const int64_t nb = std::min(kBand, rows - b);
        if (!read(trow0 + b, nb, w.data(), err)) return false;
        // one row: |W_r|^2 (double, as the base norms were), (W A^T)_rj (FP32, PEFT's own precision), then the
        // composed square; a square gone negative by rounding is a zero row's
        const auto work = [&](int64_t r0, int64_t r1) {
            std::vector<float> wat((size_t) rank), bf((size_t) rank);
            for (int64_t r = r0; r < r1; ++r) {
                const float* wr = w.data() + (size_t) r * k;
                double ss = 0;
                for (int64_t c = 0; c < k; ++c) ss += (double) wr[c] * (double) wr[c];
                for (int64_t j = 0; j < rank; ++j) {
                    const float* aj = af.data() + j * k;
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
                    int64_t c = 0;
                    for (; c + 4 <= k; c += 4)
                        for (int u = 0; u < 4; ++u) acc[u] += wr[c + u] * aj[c + u];
                    float t = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                    for (; c < k; ++c) t += wr[c] * aj[c];
                    wat[(size_t) j] = t;
                }
                for (int64_t j = 0; j < rank; ++j)
                    bf[(size_t) j] = kernels::f32_from_f16(e.b[(size_t) ((src_row + b + r) * rank + j)]);
                double cross = 0, quad = 0;
                for (int64_t j = 0; j < rank; ++j) {
                    cross += (double) bf[(size_t) j] * wat[(size_t) j];
                    double t = 0;
                    for (int64_t l = 0; l < rank; ++l) t += (double) aat[(size_t) (j * rank + l)] * bf[(size_t) l];
                    quad += (double) bf[(size_t) j] * t;
                }
                const double n2 = ss + 2.0 * s * cross + (double) s * s * quad;
                norms[(size_t) (b + r)] = (float) std::sqrt(n2 > 0 ? n2 : 0);
            }
        };
        const int64_t per = (nb + nt - 1) / nt;
        std::vector<std::thread> ts;
        for (int t = 1; t < nt && t * per < nb; ++t)
            ts.emplace_back(work, t * per, std::min(nb, (t + 1) * per));
        work(0, std::min(nb, per));
        for (std::thread& t : ts) t.join();
    }
    return true;
}

}  // namespace strata::image
