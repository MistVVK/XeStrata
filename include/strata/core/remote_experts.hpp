// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The pairs several helpers fill from (--expert-cache-device1..3): `profile` first, then every other (layer, expert)
/// expert by expert, each layer in turn, so that the tail spreads over the layers instead of piling on layer zero (the
/// shipped profile names a third of Qwen3.8-Flash-Next's experts).
std::vector<std::pair<int32_t, int32_t>> remote_ranking(const std::vector<std::pair<int32_t, int32_t>>& profile,
                                                        int64_t layers, int64_t experts);

/// --expert-cache-remote-placement: the pairs each helper r fills (caps[r] its slots; helpers 1 and 2 count while
/// their caps are > 0), from `ranked` without those `held` (a primary cache's).  "stripe" deals them out in turn for
/// parallel work on the helpers; "layer" (`by_layer`) gives each layer one helper (layer % helpers), as far as it has
/// room, to cut switches and transfers over a shared link (USB4).  Both engines' (generate.cpp, PlainModel).
std::array<std::vector<std::pair<int32_t, int32_t>>, 3>
remote_placement(const std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t layers, int64_t experts,
                 const std::array<int, 3>& caps, bool by_layer, const std::function<bool(int32_t, int32_t)>& held);

/// A static, profile-filled expert tier on another GPU. GPU 0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
class RemoteExperts {
public:
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Whether a helper computes the experts of the process's layout: every native format but BF16 experts, whose rows
    /// take FP32 activations (native_expert_grouped_bf16) that begin() does not run.  `why` says which it does not.
    /// open() refuses those, and a role choice (--layer-split auto) leaves the helper out for them.
    static bool computes_layout(std::string& why);
    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    /// `fit`: up to `slots` as far as the device's free memory takes them (HELPER on, after an image job that kept
    /// part of the card), instead of refusing when they do not all fit.
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err, bool fit = false);
    void close();

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    /// This helper's cache holds (layer, expert): begin() will take its rows unless the plan gave them away.
    bool holds(int64_t layer, int32_t expert) const { return cache_.slot_of(layer, expert) >= 0; }
    /// The engine device this helper runs on (-1: not open).
    int device() const { return device_; }
    /// (layer, expert)'s blob in this helper's VRAM, or null where it does not hold it.
    const uint8_t* blob_on_device(int64_t layer, int32_t expert) const {
        const int32_t slot = cache_.slot_of(layer, expert);
        return slot >= 0 ? cache_.device_slot(slot) : nullptr;
    }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// STRATA_PREFILL_HELPER_SOURCE (opt-in): copy up to `n` cached blobs into the host buffers `dst[i]`
    /// on this helper's own stream and wait.  `hit[i]` is set where this helper held (layer, expert); the
    /// bytes are the ones open() filled from the same source, so the prompt path gets identical bytes.
    /// Misses are left to the caller.  False with `err` on a failure; the hits copied so far are complete
    /// (the stream is drained), so the caller falls back for the rest.  Safe from several threads (one
    /// stream, one lock): the prompt path's stager is threaded.
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err);
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }

private:
    int device_ = -1;
    int64_t n_expert_ = 0;
    int32_t groups_ = 0;
    int64_t computed_ = 0;
    int64_t h_ = 0, ff_ = 0;   ///< the rows' width and the experts' inner size (open)
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    std::chrono::steady_clock::time_point launched_at_;   ///< begin()'s last launch
    double run_avg_ms_ = 0;    ///< launch to done, a moving average: how long finish() polls before it blocks
    ExpertCache cache_;
    void* stream_ = nullptr;
    void* read_stream_ = nullptr;   ///< STRATA_PREFILL_HELPER_SOURCE: cache -> host copies
    std::mutex read_mu_;            ///< one cache reader at a time (the stager is threaded)
    float* h_x_ = nullptr;     ///< host USM of this GPU's runtime: its kernels read it in place (zero copy)
    float* h_out_ = nullptr;   ///< the same, written in place
    void* h_meta_ = nullptr;
    bool zero_copy_ = false;
    float* d_x_ = nullptr;
    float* d_out_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    uint8_t* d_types_ = nullptr;   ///< a mixed pack: each expert's (gate/up K, down K), layer by layer
    void* d_meta_ = nullptr;  ///< one upload of the grouped indices, instead of five small copies
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, original_row_;
    std::vector<unsigned long long> ptr_;
};

/// STRATA_PREFILL_HELPER_SOURCE (opt-in): the prompt path's view of the expert tiers, where a blob a helper
/// GPU already holds in its VRAM cache is copied from that cache instead of being read from the files again.
/// Everything else is forwarded to `fallback` unchanged, so its reads, its pinned RAM copy and the prompt
/// path's own decisions keep their meaning.  Nothing is cached here: every call asks the helpers.
///
/// The serve loop builds this only for a serial request path (no batch slots, no layer split), while the
/// helpers are idle: during a prompt read the helper GPUs compute nothing, so their caches can be read
/// without ordering the copies against their kernel streams.
class RemotePrefillSource final : public ExpertSource {
public:
    RemotePrefillSource(ExpertSource& fallback, const std::vector<RemoteExperts*>& helpers)
        : fallback_(fallback), helpers_(helpers) {}

    /// Whether any helper's VRAM cache holds (layer, expert) right now.
    bool from_helper(int64_t layer, int64_t expert) const {
        for (const RemoteExperts* h : helpers_) if (h->holds(layer, (int32_t) expert)) return true;
        return false;
    }
    /// Blobs / bytes this source has taken from the helpers so far (the per-request evidence line).
    uint64_t helper_blobs() const { return helper_blobs_.load(std::memory_order_relaxed); }
    uint64_t helper_bytes() const { return helper_bytes_.load(std::memory_order_relaxed); }

    const uint8_t* blob(int64_t layer, int64_t expert) override { return fallback_.blob(layer, expert); }
    int64_t reads() const override { return fallback_.reads(); }
    bool pinned(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? false : fallback_.pinned(layer, expert);
    }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override {
        fallback_.begin_layer(layer, ids, k);
    }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? nullptr : fallback_.device_alias(layer, expert);
    }
    bool make_kernel_readable(std::string& err) override { return fallback_.make_kernel_readable(err); }
    void prefetch(int64_t layer, int64_t expert) override { fallback_.prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override {
        return from_helper(layer, expert) ? 0 : fallback_.release(layer, expert);
    }
    bool pcie_layer(int64_t layer) const override { return fallback_.pcie_layer(layer); }
    /// True where a helper holds the blob: the prompt path's stager must then take the copy_blob route,
    /// which is where the cache-to-host copy happens.  Everywhere else the fallback answers as it always did.
    bool transient(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? true : fallback_.transient(layer, expert);
    }
    double cached_share(int64_t samples) const override { return fallback_.cached_share(samples); }
    bool register_on(int device, std::string& err) override { return fallback_.register_on(device, err); }
    bool registered_on(int device) const override { return fallback_.registered_on(device); }
    void hold_registration(int device) override { fallback_.hold_registration(device); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override {
        return fallback_.advise_pairs(pairs, n);
    }
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.prefetch(layer, experts, n);
    }

private:
    void warn(const std::string& err);
    ExpertSource& fallback_;
    std::vector<RemoteExperts*> helpers_;
    std::atomic<uint64_t> helper_blobs_{0}, helper_bytes_{0};
    std::atomic<bool> warned_{false};
};

} // namespace strata::core
