// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/remote_experts.cpp - the helper-GPU expert tier (--expert-cache-device1..3), on the Xe runtime.
//
// A second GPU holds a static tier of the experts GPU 0's cache does not, and computes the decode's CPU-candidate rows
// that fall on them; the CPU pool computes the rest.  Each layer's activations go to it through its own host USM and
// its compact result rows come back the same way: no peer path is needed (a B70 and a 4070 have none).  After the
// CUDA implementation (upstream src/core/remote_experts.cpp, v0.1.44.1), without its cache reads for the prompt path
// and its reduction on the helper.
#include "strata/core/remote_experts.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/window_limits.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = (int64_t) strata::kernels::kWindowMaxT * 10;   // a window's rows x top-10 (upstream 9d3c8c366)

// Small enough to copy as one buffer per layer; the kernels read the arrays through pointers into one allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t ids[CAP];   ///< a mixed pack: the expert of each compact row (the R1Q2 product reads its type by it)
    int32_t count;
};

std::string failed(int device, const char* what) {
    return "GPU" + std::to_string(device) + " experts: " + what + ": " + gpu::last_error();
}
}  // namespace

RemoteExperts::~RemoteExperts() { close(); }

// Opens the device's runtime (its context and queues) and reads its free memory.
std::vector<std::pair<int32_t, int32_t>> remote_ranking(const std::vector<std::pair<int32_t, int32_t>>& profile,
                                                        int64_t layers, int64_t experts) {
    std::vector<std::pair<int32_t, int32_t>> ranked = profile;
    std::vector<uint8_t> seen((size_t) layers * (size_t) experts, 0);
    for (const auto& pair : ranked)
        if (pair.first >= 0 && pair.first < layers && pair.second >= 0 && pair.second < experts)
            seen[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    for (int64_t e = 0; e < experts; ++e)
        for (int64_t l = 0; l < layers; ++l)
            if (!seen[(size_t) l * (size_t) experts + (size_t) e]) ranked.emplace_back((int32_t) l, (int32_t) e);
    return ranked;
}

std::array<std::vector<std::pair<int32_t, int32_t>>, 3>
remote_placement(const std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t layers, int64_t experts,
                 const std::array<int, 3>& caps, bool by_layer, const std::function<bool(int32_t, int32_t)>& held) {
    std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
    std::vector<uint8_t> assigned((size_t) layers * (size_t) experts, 0);
    // the helpers actually asked for (caps[r] > 0), in order.  A lone --expert-cache-device2 or -device3 (no -device1)
    // is one of them: the count is not "1 + the later two", which placed nothing when caps[0] was 0 (the helper then
    // opened with an empty list, "need ... ranked experts").
    std::array<int, 3> active{};
    int devices = 0;
    for (int r = 0; r < 3; ++r) if (caps[(size_t) r] > 0) active[(size_t) devices++] = r;
    if (devices == 0) return by_device;
    int next = 0;
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts ||
            held(pair.first, pair.second))
            continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (assigned[index]) continue;
        int target = -1;   // an index into active
        if (by_layer) {
            target = pair.first % devices;
            // other layers' owners may still have room: keep scanning the ranks
            const int r = active[(size_t) target];
            if (by_device[(size_t) r].size() >= (size_t) caps[(size_t) r]) continue;
        } else {
            for (int i = 0; i < devices; ++i) {
                const int c = (next + i) % devices;
                const int r = active[(size_t) c];
                if (by_device[(size_t) r].size() < (size_t) caps[(size_t) r]) {
                    target = c;
                    break;
                }
            }
            if (target < 0) break;
        }
        assigned[index] = 1;
        by_device[(size_t) active[(size_t) target]].push_back(pair);
        next = (target + 1) % devices;
    }
    return by_device;
}

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    free_gib = 0;
    try {
        const OnDevice on(device);
        (void) Runtime::get();
        size_t free_bytes = 0, total = 0;
        if (!gpu::mem_info(&free_bytes, &total)) {
            err = "GPU " + std::to_string(device) + ": " + gpu::last_error();
            return false;
        }
        free_gib = (double) free_bytes / 1073741824.0;
        return true;
    } catch (const std::exception& e) {
        err = "GPU " + std::to_string(device) + ": " + e.what();
        return false;
    }
}

bool RemoteExperts::computes_layout(std::string& why) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l)
        if (lay.fmt[l].gu_type == 30 || lay.fmt[l].d_type == 30) {
            why = "BF16 experts are not computed on a helper GPU";
            return false;
        }
    return true;
}

void RemoteExperts::close() {
    if (device_ < 0) return;
    try {
        const OnDevice on(device_);
        if (stream_) gpu::stream_sync(stream_);
        if (read_stream_) gpu::stream_sync(read_stream_);
        cache_.close();
        for (const void* p : {(const void*) d_x_, (const void*) d_out_, (const void*) d_q8_, (const void*) d_scales_,
                              (const void*) d_scratch_, (const void*) d_meta_, (const void*) d_types_,
                              (const void*) h_x_, (const void*) h_out_, (const void*) h_meta_})
            gpu::free(p);
        if (stream_) gpu::stream_destroy(stream_);
        if (read_stream_) gpu::stream_destroy(read_stream_);
    } catch (const std::exception& e) {   // teardown, from the destructor too: said, not thrown
        std::fprintf(stderr, "GPU%d experts: close: %s\n", device_, e.what());
    }
    device_ = -1;
    stream_ = nullptr;
    read_stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_scratch_ = nullptr;
    d_types_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    zero_copy_ = false;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool fit) {
    close();
    if (device < 1 || device >= Runtime::count() || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts) {
        err = "GPU" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    if (std::string why; !computes_layout(why)) {
        err = "GPU" + std::to_string(device) + " experts: " + why + " (--expert-cache-device1..3)";
        return false;
    }
    std::vector<std::pair<int32_t, int32_t>> selected;
    try {
        const OnDevice on(device);
        device_ = device;
        n_expert_ = experts;
        const auto& lay = strata::kernels::cpu::expert_layout();
        size_t free_bytes = 0, total_bytes = 0;
        if (!gpu::mem_info(&free_bytes, &total_bytes)) { err = failed(device, "free memory"); close(); return false; }
        // the arena is registered for copies with GPU 0's context only: fill from it as fast on this one
        std::string why;
        if (!source.register_on(device, why))
            std::fprintf(stderr, "xestrata generate: GPU%d experts: %s; its fill from the arena is slower\n", device,
                         why.c_str());
        uint64_t needed = 0;
        selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
        std::vector<uint8_t> picked(claimed.size(), 0);
        for (const auto& pair : ranked) {
            if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
            const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
            if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
                const uint64_t blob =
                    lay.native ? (lay.blob_bytes(pair.first, pair.second) + 255) / 256 * 256 : lay.max_blob;
                if (fit && needed + blob + (512ull << 20) > free_bytes) break;
                selected.push_back(pair);
                needed += blob;
                picked[index] = 1;
                if ((int) selected.size() >= slots) break;
            }
        }
        if (selected.empty()) { err = "GPU" + std::to_string(device) + " experts: no unclaimed experts remain"; close(); return false; }
        // room for the runtime, the staging and the driver's later allocations
        if (needed + (512ull << 20) > free_bytes) {
            err = "GPU" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce "
                  "--expert-cache-device" + std::to_string(device);
            close(); return false;
        }
        std::vector<int64_t> sizes;
        if (lay.native) {
            sizes.reserve(selected.size());
            for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first, pair.second));
        }
        const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                         : cache_.open((int64_t) selected.size(), layers, experts,
                                                       (int64_t) lay.max_blob, err);
        if (!cache_ok) { err = "GPU" + std::to_string(device) + " experts: " + err; close(); return false; }
        for (const auto& pair : selected) {
            const int32_t slot = cache_.admit(pair.first, pair.second);
            const uint8_t* blob = source.blob(pair.first, pair.second);
            if (slot < 0 || !blob ||
                !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first, pair.second))) {
                err = "GPU" + std::to_string(device) + " experts: " + (err.empty() ? "cache fill failed" : err);
                close(); return false;
            }
        }
        const auto& first = selected.front();
        if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                                err, (int64_t) lay.blob_bytes(first.first, first.second))) {
            err = "GPU" + std::to_string(device) + " experts: " + err;
            close(); return false;
        }

        // the rows' width and the experts' inner size: the model's in the native layouts (either engine's model),
        // else Qwen3.8-Flash-Next's packs'
        h_ = H;
        ff_ = FF;
        if (lay.native && !lay.fmt.empty()) {
            h_ = ff_ = 0;
            for (const auto& f : lay.fmt) {
                h_ = std::max<int64_t>(h_, f.n_embd);
                ff_ = std::max<int64_t>(ff_, f.n_ff);
            }
        }
        size_t scratch = std::max<size_t>(
            (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, h_, ff_),
            strata::kernels::native_expert_scratch_bytes(CAP, ff_));
        // a mixed pack (R1Q2, a type and a size each expert): the grouped R1Q2 product's scratch, and each expert's
        // (gate/up K, down K) on this device, as the verify window's (moe_window.cpp)
        std::vector<uint8_t> types;
        if (lay.mixed()) {
            scratch = std::max(scratch, strata::kernels::bih3_r1q2_grouped_scratch_bytes(CAP, h_, ff_));
            types.resize((size_t) layers * (size_t) experts * 2);
            for (int64_t l = 0; l < layers; ++l)
                for (int64_t e = 0; e < experts; ++e) {
                    const auto& f = lay.format(l, e);
                    if (!f.r1q2 || f.n_embd != h_ || f.n_ff != ff_ || f.gu_type < 182 || f.gu_type > 194 ||
                        f.d_type < 182 || f.d_type > 194) {
                        err = "GPU" + std::to_string(device) + " experts: invalid mixed R1Q2 geometry or expert type";
                        close(); return false;
                    }
                    const size_t i = lay.expert_index(l, e) * 2;
                    types[i] = (uint8_t) (f.gu_type - 180);
                    types[i + 1] = (uint8_t) (f.d_type - 180);
                }
        }
        const size_t rows = (size_t) CAP * h_ * sizeof(float);
        stream_ = gpu::stream_create();
        read_stream_ = gpu::stream_create_copy();
        const char* step = stream_ == nullptr                                   ? "stream"
                         : read_stream_ == nullptr                              ? "cache read stream"
                         : !gpu::alloc_host(&h_x_, rows)                        ? "input staging"
                         : !gpu::alloc_host(&h_out_, rows)                      ? "result staging"
                         : !gpu::alloc_host(&h_meta_, sizeof(RemoteMeta))       ? "metadata staging"
                         : !gpu::alloc_device(&d_x_, rows)                      ? "input"
                         : !gpu::alloc_device(&d_out_, rows)                    ? "result"
                         : !gpu::alloc_device(&d_q8_, (size_t) CAP * (h_ / 32) * 36) ? "activation"
                         : !gpu::alloc_device(&d_scales_, (size_t) CAP * (h_ / 32) * sizeof(float)) ? "activation scales"
                         : !gpu::alloc_device(&d_scratch_, scratch)             ? "scratch"
                         : !gpu::alloc_device(&d_meta_, sizeof(RemoteMeta))     ? "group metadata"
                         : !types.empty() && (!gpu::alloc_device(&d_types_, types.size()) ||
                                              !gpu::copy(d_types_, types.data(), types.size())) ? "expert types"
                                                                                : nullptr;
        if (step != nullptr) { err = failed(device, step); close(); return false; }
        // Zero copy: the helper's kernels read their input from, and write their compact rows into, its own host USM
        // - two copies fewer a layer, each a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
        const char* zc = strata::env_get("REMOTE_ZEROCOPY");
        zero_copy_ = !(zc && zc[0] == '0') && gpu::kernels_read_host(h_x_) && gpu::kernels_read_host(h_out_);
        auto* meta = (RemoteMeta*) d_meta_;
        d_ptr_ = meta->ptr;
        d_start_ = meta->start;
        d_dst_ = meta->dst;
        d_tok_ = meta->tok;
        d_count_ = &meta->count;
    } catch (const std::exception& e) {
        err = "GPU" + std::to_string(device) + " experts: " + e.what();
        close();
        return false;
    }
    owned_.assign(CAP, 0);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::kWindowMaxT || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "GPU" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // the kernels read the activation from `tok`, so the output rows are packed densely for the return
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Buffers of this helper's own live until its GPU has read them: the CPU pool writes other rows meanwhile.
    std::memcpy(h_x_, x, (size_t) n_tok * h_ * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    for (size_t j = 0; j < original_row_.size(); ++j) meta->ids[j] = ids[original_row_[j]];
    meta->count = (int32_t) group_id_.size();
    groups_ = (int32_t) group_id_.size();
    const uint64_t compact_bytes = (uint64_t) dst_.size() * h_ * sizeof(float);
    try {
        const OnDevice on(device_);
        void* s = stream_;
        if ((!zero_copy_ && !gpu::copy_async(d_x_, h_x_, (size_t) n_tok * h_ * sizeof(float), s)) ||
            !gpu::copy_async(d_meta_, h_meta_, sizeof(RemoteMeta), s)) {
            err = failed(device_, "copy input");
            return false;
        }
        const float* xin = zero_copy_ ? h_x_ : d_x_;
        float* out = zero_copy_ ? h_out_ : d_out_;
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (lay.native) {
            const auto& fmt = lay.fmt[(size_t) layer];
            if (fmt.had_gu) {   // APR: Hadamard-folded gate/up take H (s * x), rotated into the device rows
                strata::kernels::hadamard_rows(xin, d_x_, n_tok, h_, fmt.had_block,
                                               strata::kernels::hadamard_device_signs(fmt.gu_signs, (int) h_), s);
                xin = d_x_;
            }
            strata::kernels::quantize_q8_1_rows(xin, n_tok, h_, d_q8_, s);
            // APR: and the down weights the rotated h
            const float* d_signs = fmt.had_d ? strata::kernels::hadamard_device_signs(fmt.d_signs, (int) fmt.n_ff)
                                             : nullptr;
            if (lay.mixed()) {   // R1Q2, a type each expert: the verify window's grouped product (moe_window.cpp),
                                 // on the path this GPU's report gives it
                strata::kernels::Bih3R1q2Layer B;
                B.n_embd = (int) fmt.n_embd;
                B.n_ff = (int) fmt.n_ff;
                B.n_expert = (int) lay.n_expert;
                B.k_pairs = d_types_ + (size_t) layer * (size_t) lay.n_expert * 2;
                B.route_ids = static_cast<const RemoteMeta*>(d_meta_)->ids;
                B.had_block = fmt.had_d ? fmt.had_block : 0;
                B.d_signs = d_signs;
                std::memcpy(B.p2, fmt.p2, sizeof B.p2);
                strata::kernels::bih3_r1q2_grouped(B, d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups_,
                                                   (int64_t) dst_.size(), d_q8_, d_scratch_, out, s);
            } else {
                auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff,
                                                               fmt.up_type);
                if (fmt.had_d) {
                    L.had_block = fmt.had_block;
                    L.d_signs = d_signs;
                }
                strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups_,
                                                       (int64_t) dst_.size(), d_q8_, d_scratch_, out, s);
            }
        } else {
            strata::kernels::quantize_q8_0_scaled(xin, d_q8_, d_scales_, n_tok * H, s);
            strata::kernels::moe_grouped_s2_window(d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups_,
                                                   (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, out, s,
                                                   (int) n_tok);
        }
        if (!zero_copy_ && !gpu::copy_async(h_out_, d_out_, (size_t) compact_bytes, s)) {
            err = failed(device_, "copy results");
            return false;
        }
    } catch (const std::exception& e) {
        err = "GPU" + std::to_string(device_) + " experts: " + e.what();
        return false;
    }
    ++launched_layers_;
    launched_at_ = std::chrono::steady_clock::now();
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * h_ * sizeof(float);
    return true;
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (group_id_.empty()) return true;
    // This thread polls the helper's queue while it runs, as upstream's CUDA helper spins (cudaDeviceScheduleSpin):
    // a wait in the driver is a barrier submitted and a sleep woken by an interrupt, every layer of every window, on
    // the decode's critical path once the CPU pool is done.  The polling stops 4x the helper's layer time after the
    // launch (begin to done, a moving average; none yet: the first wait blocks), then blocks as before, so a stuck
    // GPU still reaches the runtime's watchdog.  STRATA_REMOTE_SPIN=0 always blocks.
    static const bool spin = [] {
        const char* v = strata::env_get("REMOTE_SPIN");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const auto w0 = std::chrono::steady_clock::now();
    try {
        const OnDevice on(device_);
        bool done = false;
        if (spin && run_avg_ms_ > 0) {
            const auto bound = launched_at_ + std::chrono::duration<double, std::milli>(4.0 * run_avg_ms_);
            for (;;) {
                const gpu::Status st = gpu::stream_status(stream_);
                if (st == gpu::Status::Failed) { err = failed(device_, "finish"); return false; }
                if (st == gpu::Status::Done) { done = true; break; }
                if (std::chrono::steady_clock::now() > bound) break;
                for (int i = 0; i < 16; ++i) _mm_pause();
            }
        }
        if (!done && !gpu::stream_sync(stream_)) { err = failed(device_, "finish"); return false; }
    } catch (const std::exception& e) {
        err = "GPU" + std::to_string(device_) + " experts: " + e.what();
        return false;
    }
    const auto w1 = std::chrono::steady_clock::now();
    ms_wait_ += std::chrono::duration<double, std::milli>(w1 - w0).count();
    const double run = std::chrono::duration<double, std::milli>(w1 - launched_at_).count();
    run_avg_ms_ = run_avg_ms_ > 0 ? 0.9 * run_avg_ms_ + 0.1 * run : run;
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * h_, h_out_ + i * h_, (size_t) h_ * sizeof(float));
    return true;
}

bool RemoteExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (device_ < 0 || read_stream_ == nullptr || n == 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::lock_guard<std::mutex> lock(read_mu_);
    try {
        const OnDevice on(device_);
        bool queued = false;
        for (size_t i = 0; i < n; ++i) {
            if (dst[i] == nullptr) continue;
            const int32_t slot = cache_.slot_of(layers[i], experts[i]);
            if (slot < 0) continue;
            if (!gpu::copy_async(dst[i], cache_.device_slot(slot), (size_t) lay.blob_bytes(layers[i], experts[i]),
                                 read_stream_)) {
                err = failed(device_, "cache read");
                // nothing may still be writing a caller's buffer when this returns: if the drain fails, forget
                // every hit and let the caller refill all of them from its fallback
                if (!gpu::stream_sync(read_stream_)) std::fill(hit, hit + n, 0);
                return false;
            }
            hit[i] = 1;
            queued = true;
        }
        if (queued && !gpu::stream_sync(read_stream_)) {
            err = failed(device_, "cache read wait");
            std::fill(hit, hit + n, 0);   // a failed wait means no copied buffer may be trusted
            return false;
        }
    } catch (const std::exception& e) {
        err = "GPU" + std::to_string(device_) + " experts: " + e.what();
        std::fill(hit, hit + n, 0);
        return false;
    }
    return true;
}

bool RemotePrefillSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst != nullptr && from_helper(layer, expert)) {
        const int32_t l = (int32_t) layer, e = (int32_t) expert;
        uint8_t* d = dst;
        std::string err;
        for (RemoteExperts* h : helpers_) {
            uint8_t got = 0;
            const bool ok = h->copy_cached_blobs(&l, &e, &d, 1, &got, err);
            if (got) {
                helper_blobs_.fetch_add(1, std::memory_order_relaxed);
                helper_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer, expert),
                                        std::memory_order_relaxed);
                return true;
            }
            if (!ok) break;
        }
        warn(err);
    }
    return fallback_.copy_blob(layer, expert, dst);
}

void RemotePrefillSource::warn(const std::string& err) {
    if (err.empty() || warned_.exchange(true)) return;
    std::fprintf(stderr, "prefill: a helper cache read failed (%s); the file source fills that blob instead\n",
                 err.c_str());
}

}  // namespace strata::core
