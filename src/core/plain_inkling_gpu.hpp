// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/device.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/platform/memory.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::core {

// First GPU path: stream selected experts, keeping the model's FP32 reference state on the host.
class InklingProducts {
    struct Free { void operator()(void* p) const { if (p) gpu::free(p); } };
    template<class T> using Buffer = std::unique_ptr<T, Free>;
    template<class T> static Buffer<T> alloc(size_t n) {
        Buffer<T> p(static_cast<T*>(gpu::alloc_device(n * sizeof(T))));
        if (!p) throw std::runtime_error("inkling2: GPU product buffer allocation failed");
        return p;
    }
    struct Stream {
        gpu::Stream s = gpu::stream_create();
        ~Stream() { if (s) { gpu::stream_sync(s); gpu::stream_destroy(s); } }
    } stream_;
    Buffer<uint8_t> raw_;
    Buffer<uint16_t> weights_, weights_lo_, input_;
    Buffer<float> result_, correction_, decoded_;
    Buffer<int32_t> bounds_, positions_;
    Buffer<float> aq_, ak_, av_, ar_, at_, asink_, ascores_, ao_, taps_;
    std::vector<uint16_t> input_host_, input_lo_;
    std::vector<int32_t> bounds_host_;
    // The resident experts' VRAM is the storage of every engine's tier (ExpertCache): a slot an expert, its gate/up
    // from the slot's start and its down from the next 256-byte boundary, so each blob starts where a slot of the
    // cache does.  The cache's table says which experts are resident; `hot_` is the lookup grouped_rows takes, from a
    // blob's bytes in the file to its address in the slot.
    ExpertCache tier_;
    std::unordered_map<const uint8_t*, const uint8_t*> hot_;
    uint64_t cold_hint_ = 0, gpu_hits_ = 0, gpu_lookups_ = 0;

    struct ResidentKv {
        Buffer<float> k, v;
        Buffer<int32_t> positions;
        bool valid = false;
    };
    std::vector<ResidentKv> resident_kv_;

    Buffer<float> head_in_, head_out_;
    std::vector<Buffer<float>> head_;
    int64_t head_width_ = 0, head_rows_ = 0, head_tile_ = 0;

public:
    static constexpr int kWindowRows = 8;

    InklingProducts(int64_t groups, int64_t width, int64_t ff, size_t raw_bytes, int64_t heads, int64_t kv, int64_t dim, int64_t rel, int64_t extent)
        : raw_(alloc<uint8_t>(raw_bytes)), weights_(alloc<uint16_t>((size_t) (groups * 2 * ff * width))),
          weights_lo_(alloc<uint16_t>((size_t) (groups * 2 * ff * width))),
          input_(alloc<uint16_t>((size_t) (kWindowRows * groups * std::max(width, ff)))),
          result_(alloc<float>((size_t) (kWindowRows * groups * std::max(2 * ff, width)))),
          correction_(alloc<float>((size_t) (kWindowRows * groups * std::max(2 * ff, width)))),
          decoded_(alloc<float>((size_t) (2 * ff * width))),
          bounds_(alloc<int32_t>((size_t) groups + 1)), positions_(alloc<int32_t>((size_t) extent)),
          aq_(alloc<float>((size_t) (heads * dim))), ak_(alloc<float>((size_t) (extent * kv * dim))),
          av_(alloc<float>((size_t) (extent * kv * dim))), ar_(alloc<float>((size_t) (heads * rel))),
          at_(alloc<float>((size_t) (extent * rel))), asink_(alloc<float>((size_t) heads)),
          ascores_(alloc<float>((size_t) (extent * heads))), ao_(alloc<float>((size_t) (heads * dim))),
          taps_(alloc<float>((size_t) width * 4)),
          input_host_((size_t) (kWindowRows * groups * std::max(width, ff))), input_lo_(input_host_.size()), bounds_host_((size_t) groups + 1) {
        if (!stream_.s) throw std::runtime_error("inkling2: GPU stream creation failed");
    }

    bool load_head(int type, const uint8_t* data, size_t stride, int64_t width, int64_t rows) {
        if (!kernels::embed_type_supported(type) || width % 256 != 0)
            throw std::runtime_error("inkling2: unsupported resident head format");
        size_t free = 0, total = 0;
        if (!gpu::mem_info(&free, &total)) throw std::runtime_error("inkling2: cannot query head memory budget");
        const size_t bytes = (size_t) (width * rows) * sizeof(float);
        const uint64_t vr_reserve = vram_room(total);
        const uint64_t vr_budget = free - std::min<uint64_t>(free, vr_reserve);
        const auto host = sample_host_memory();
        const uint64_t host_reserve = detail::ram_headroom(host ? host->limit / 32 : 0);
        const uint64_t host_budget = host ? host->available - std::min(host->available, host_reserve) : 0;
        std::fprintf(stderr, "xestrata: Inkling placement: VRAM total/free/reserve/budget %zu/%zu/%llu/%llu bytes; "
                             "DRAM limit/available/reserve/budget %llu/%llu/%llu/%llu bytes\n", total, free,
                     (unsigned long long) vr_reserve, (unsigned long long) vr_budget,
                     (unsigned long long) (host ? host->limit : 0), (unsigned long long) (host ? host->available : 0),
                     (unsigned long long) host_reserve, (unsigned long long) host_budget);
        if (bytes > vr_budget) return false;
        head_width_ = width;
        head_rows_ = rows;
        const auto max_rows = max_alloc_bytes() / ((size_t) width * sizeof(float));
        head_tile_ = (int64_t) std::min<uint64_t>({max_rows, max_alloc_bytes() / stride, (uint64_t) rows});
        if (head_tile_ < 1) throw std::runtime_error("inkling2: device cannot hold one head row");
        // Preserve the driver's margin even while staging the native blocks for the resident tiles.
        const uint64_t vectors = (uint64_t) (width + rows) * sizeof(float);
        if (vectors >= vr_budget - bytes) return false;
        head_tile_ = std::min(head_tile_, (int64_t) ((vr_budget - bytes - vectors) / stride));
        if (head_tile_ < 1) return false;
        auto raw = alloc<uint8_t>((size_t) head_tile_ * stride);
        head_in_ = alloc<float>((size_t) width);
        head_out_ = alloc<float>((size_t) rows);
        for (int64_t at = 0; at < rows; at += head_tile_) {
            const int64_t n = std::min(head_tile_, rows - at);
            head_.push_back(alloc<float>((size_t) (n * width)));
            upload(raw.get(), data + (size_t) at * stride, (size_t) n * stride);
            kernels::iq_dequant_f32(type, raw.get(), n * width, head_.back().get(), stream_.s);
            if (!gpu::stream_sync(stream_.s)) throw std::runtime_error("inkling2: resident head upload failed");
        }
        size_t after = 0;
        if (!gpu::mem_info(&after, &total)) throw std::runtime_error("inkling2: cannot query remaining device memory");
        std::fprintf(stderr, "xestrata: Inkling head resident: %zu bytes, device free %zu -> %zu bytes\n",
                     bytes, free, after);
        return true;
    }

    bool has_head() const { return !head_.empty(); }

    bool project_head(const float* x, float* y, int64_t rows) {
        if (rows < 1 || rows > head_rows_) return false;
        upload(head_in_.get(), x, (size_t) head_width_ * sizeof(float));
        int64_t at = 0;
        for (const auto& tile : head_) {
            const int64_t n = std::min(head_tile_, rows - at);
            if (!kernels::gemv_f32_ordered(head_in_.get(), tile.get(), head_out_.get() + at, n, head_width_, stream_.s))
                throw std::runtime_error("inkling2: ordered GPU head needs device FP64; set XESTRATA_INKLING_PLACEMENT=host");
            at += n;
            if (at == rows) break;
        }
        transfer(y, head_out_.get(), (size_t) rows * sizeof(float));
        return true;
    }

    // Where an expert's down starts in its slot, and the slot's bytes (whole 256s, as ExpertCache lays its slots).
    static size_t down_at(size_t gate_up_bytes) { return (gate_up_bytes + 255) / 256 * 256; }
    static size_t slot_bytes(size_t gate_up_bytes, size_t down_bytes) {
        return (down_at(gate_up_bytes) + down_bytes + 255) / 256 * 256;
    }
    // The tier: one slot for each resident expert, of the sizes given, in the order reserve_hot is then called.
    // In pieces where it passes the device's largest allocation; tier_slots() says how many the device gave.
    void open_tier(const std::vector<int64_t>& sizes, int64_t layers, int64_t experts) {
        std::string err;
        tier_.set_max_piece(max_alloc_bytes());
        if (!tier_.open_sized(sizes, layers, experts, err)) throw std::runtime_error("inkling2: expert tier: " + err);
    }
    int64_t tier_slots() const { return tier_.slots(); }
    int64_t resident() const { return tier_.resident(); }
    void reserve_hot(int64_t layer, int64_t expert, const uint8_t* gate_up, size_t gate_up_bytes, const uint8_t* down,
                     size_t down_bytes) {
        const int32_t slot = tier_.admit(layer, expert);
        if (slot < 0 || tier_.slot_capacity(slot) < down_at(gate_up_bytes) + down_bytes)
            throw std::runtime_error("inkling2: the expert tier has no slot for a planned expert");
        uint8_t* at = tier_.device_slot(slot);
        transfer(at, gate_up, gate_up_bytes);
        transfer(at + down_at(gate_up_bytes), down, down_bytes);
        hot_.emplace(gate_up, at);
        hot_.emplace(down, at + down_at(gate_up_bytes));
    }
    void set_cold_hint(uint64_t bytes) { cold_hint_ = bytes; }
    uint64_t gpu_hits() const { return gpu_hits_; }
    uint64_t gpu_lookups() const { return gpu_lookups_; }

    // Inputs and results are packed by expert. One tile decodes each selected expert once,
    // then takes all its rows. The caller can send additional tiles when fan-out exceeds the buffer's group capacity.
    void grouped_rows(int type, const uint8_t* source, size_t bytes_per_expert, const int64_t* ids, int groups,
                      const float* x, int64_t in, int64_t out, float* y, const int32_t* counts) {
        if (groups < 1 || (size_t) groups >= bounds_host_.size())
            throw std::runtime_error("inkling2: expert tile exceeds group capacity");
        const auto stream = stream_.s;
        int max_rows = 1;
        for (int g = 0; g < groups; ++g) {
            const int n = counts[g];
            if (n < 1 || n > kWindowRows) throw std::runtime_error("inkling2: invalid expert tile row count");
            bounds_host_[(size_t) g + 1] = bounds_host_[(size_t) g] + n;
            max_rows = std::max(max_rows, n);
        }
        upload(bounds_.get(), bounds_host_.data(), ((size_t) groups + 1) * sizeof(int32_t));
        uint64_t hint_left = cold_hint_;
        for (int g = 0; g < groups && hint_left >= bytes_per_expert; ++g) {
            const auto* src = source + (size_t) ids[g] * bytes_per_expert;
            if (hot_.contains(src)) continue;
            platform::advise_willneed(src, bytes_per_expert);
            hint_left -= bytes_per_expert;
        }
        for (int g = 0; g < groups; ++g) {
            // The native dequantizer consumes raw GGUF blocks; the per-expert scalar is applied at the caller.
            const auto* src = source + (size_t) ids[g] * bytes_per_expert;
            const auto hit = hot_.find(src);
            ++gpu_lookups_;
            const uint8_t* weight;
            if (hit != hot_.end()) {
                ++gpu_hits_;
                weight = hit->second;
            } else {
                auto* raw = raw_.get() + (size_t) g * bytes_per_expert;
                upload(raw, src, bytes_per_expert);
                weight = raw;
            }
            kernels::iq_dequant_f32(type, weight, in * out, decoded_.get(), stream);
            kernels::f32_split_f16(decoded_.get(), weights_.get() + g * in * out,
                                   weights_lo_.get() + g * in * out, in * out, stream);
        }
        const int64_t rows = bounds_host_[(size_t) groups];
        for (int64_t t = 0; t < rows; ++t) {
            for (int64_t i = 0; i < in; ++i) {
                const float v = x[t * in + i];
                const size_t j = (size_t) (t * in + i);
                input_host_[j] = kernels::f16_from_f32(v);
                input_lo_[j] = kernels::f16_from_f32((v - kernels::f32_from_f16(input_host_[j])) * 1024.0f);
            }
        }
        const auto gemm = [&](const uint16_t* w, float* result) {
            if (!prefill::Gemm::f16_groups(input_.get(), w, in * out, result, bounds_host_.data(),
                                           groups, out, in, stream))
                kernels::xmx_gemm_grouped(input_.get(), w, in * out, result, bounds_.get(), groups, max_rows,
                                          out, in, stream);
        };
        const size_t bytes = (size_t) (rows * in) * sizeof(uint16_t);
        upload(input_.get(), input_host_.data(), bytes);
        gemm(weights_.get(), result_.get());
        gemm(weights_lo_.get(), correction_.get());
        kernels::add_inplace(result_.get(), correction_.get(), rows * out, stream, 1.0f / 1024.0f);
        upload(input_.get(), input_lo_.data(), bytes);
        gemm(weights_.get(), correction_.get());
        kernels::add_inplace(result_.get(), correction_.get(), rows * out, stream, 1.0f / 1024.0f);
        transfer(y, result_.get(), (size_t) (rows * out) * sizeof(float));
    }
    struct RoutedRow {
        int64_t expert;
        int32_t row;
        int32_t rank;
    };

    // Make deterministic groups of all (query, rank) pairs. A recurring expert uses one decoded weight tile.
    static void group_routes(std::vector<RoutedRow>& routes) {
        std::stable_sort(routes.begin(), routes.end(), [](const RoutedRow& a, const RoutedRow& b) {
            return a.expert < b.expert;
        });
    }

    void configure_kv(size_t slots, size_t extent, size_t width, uint64_t budget) {
        const uint64_t plane = extent * width * sizeof(float);
        const uint64_t tags = extent * sizeof(int32_t);
        const uint64_t slot_bytes = 2 * plane + tags;
        const uint64_t limit = max_alloc_bytes();
        const size_t count = plane <= limit && tags <= limit ? std::min<uint64_t>(slots, budget / slot_bytes) : 0;
        resident_kv_.reserve(count);
        for (size_t i = 0; i < count; ++i)
            resident_kv_.push_back({alloc<float>(extent * width), alloc<float>(extent * width), alloc<int32_t>(extent), false});
        std::fprintf(stderr, "xestrata: Inkling target KV: %zu/%zu resident slots, %llu bytes on GPU, "
                             "%llu bytes host shadow, %llu bytes budget\n", count, slots,
                     (unsigned long long) count * slot_bytes, (unsigned long long) slots * slot_bytes,
                     (unsigned long long) budget);
    }
    void invalidate_kv(size_t slot) noexcept {
        if (slot < resident_kv_.size()) resident_kv_[slot].valid = false;
    }
    // After a rejection the host shadow holds the restored cells again. Copy only those cells (ring order, split at
    // the wrap) so the resident copy stays valid instead of reloading every cell on the next attention.
    void refresh_kv(size_t slot, const float* k, const float* v, const int32_t* positions, int64_t first, int64_t n,
                    int64_t extent, size_t row) {
        if (slot >= resident_kv_.size() || !resident_kv_[slot].valid) return;
        auto& c = resident_kv_[slot];
        for (int64_t done = 0; done < n;) {
            const int64_t cell = (first + done) % extent, run = std::min(n - done, extent - cell);
            upload(c.k.get() + cell * row, k + cell * row, (size_t) run * row * sizeof(float));
            upload(c.v.get() + cell * row, v + cell * row, (size_t) run * row * sizeof(float));
            upload(c.positions.get() + cell, positions + cell, (size_t) run * sizeof(int32_t));
            done += run;
        }
    }

    void attention(const float* q, const float* k, const float* v, const float* rel, const float* table,
                   const float* sink, const int32_t* positions, int64_t pos, int64_t cells, int64_t heads,
                   int64_t kv, int64_t dim, int64_t rdim, int64_t extent, int64_t window, float* out, size_t slot) {
        upload(aq_.get(), q, (size_t) (heads * dim) * sizeof(float));
        auto* cache = slot < resident_kv_.size() ? &resident_kv_[slot] : nullptr;
        auto* dk = cache ? cache->k.get() : ak_.get();
        auto* dv = cache ? cache->v.get() : av_.get();
        auto* dp = cache ? cache->positions.get() : positions_.get();
        if (cells > 0) {
            const size_t row = (size_t) (kv * dim);
            const size_t first = cache && cache->valid ? (size_t) (pos % extent) : 0;
            const size_t count = cache && cache->valid ? 1 : (size_t) cells;
            upload(dk + first * row, k + first * row, count * row * sizeof(float));
            upload(dv + first * row, v + first * row, count * row * sizeof(float));
            upload(dp + first, positions + first, count * sizeof(int32_t));
        }
        upload(ar_.get(), rel, (size_t) (heads * rdim) * sizeof(float));
        upload(at_.get(), table, (size_t) (extent * rdim) * sizeof(float));
        upload(asink_.get(), sink, (size_t) heads * sizeof(float));
        kernels::relative_prompt_attn(aq_.get(), dk, dv, ar_.get(), at_.get(), asink_.get(), dp,
                                     pos, cells, heads, kv, dim, rdim, extent, window, ascores_.get(), ao_.get(), stream_.s);
        transfer(out, ao_.get(), (size_t) (heads * dim) * sizeof(float));
        if (cache) cache->valid = true;
    }
    void capture(const float* p, size_t elements) { transfer(taps_.get(), p, elements * sizeof(float)); }
    const float* taps() const { return taps_.get(); }
    void read_taps(float* p, size_t elements) { transfer(p, taps(), elements * sizeof(float)); }

private:
    void upload(void* dst, const void* src, size_t bytes) {
        if (!gpu::copy_async(dst, src, bytes, stream_.s))
            throw std::runtime_error("inkling2: GPU product copy failed");
    }
    void transfer(void* dst, const void* src, size_t bytes) {
        upload(dst, src, bytes);
        if (!gpu::stream_sync(stream_.s)) throw std::runtime_error("inkling2: GPU product stream failed");
    }
};

}  // namespace strata::core
