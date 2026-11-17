// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/draft_head.cpp - see include/strata/core/draft_head.hpp.
#include "strata/core/draft_head.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/draft_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

namespace strata::core {
namespace k = strata::kernels;

namespace {

// The candidates taken again: 16 kept the rerank at 4-5 us of a draft step on the B70 and the RTX 4070, and the
// top-k at 22-58 us (32: 39-126, 64: 74-222; records/2026-10-26-plan/d1).  STRATA_DRAFT_INT2_K sets it (1..64).
int candidates() {
    static const int k = [] {
        const char* v = strata::env_get("DRAFT_INT2_K");
        const int n = v != nullptr ? (int) std::strtol(v, nullptr, 10) : 16;
        return std::clamp(n, 1, k::kTopKMax);
    }();
    return k;
}

}  // namespace

namespace {

// #474 (upstream f2aaba1): what to change when the draft head's token subset does not fit the VRAM left - a smaller
// subset that setup ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  Text only, after
// the failure: nothing changes for a start that fits.
void draft_head_hint(const char* who, int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = gpu::mem_info(&free_b, &total_b);
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "%s: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n", who, (long long) n_tokens,
                 mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "%s: hint: a smaller draft vocabulary needs less VRAM: %s. Run ./setup.sh once "
                             "with it (./setup.sh --draft-vocab en) and the model keeps it (\"draft_vocab\" in its "
                             "xestrata-*.json config); or a smaller --context in setup.\n",
                     who, opts.c_str());
    else
        std::fprintf(stderr, "%s: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n", who);
}

}  // namespace

DraftSubset::~DraftSubset() {
    if (mapping_ != nullptr) gpu::vmem_unmap(rows_, mapping_bytes_, mapping_);
    if (mapping_bytes_ != 0 && rows_ != nullptr) gpu::vmem_free(rows_, mapping_bytes_);
    else if (rows_ != nullptr) gpu::free(rows_);
    if (saved_rows_ != nullptr) gpu::free(saved_rows_);
    if (d_ids_ != nullptr) gpu::free(d_ids_);
}

int64_t DraftSubset::file_tokens(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return 0;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fclose(f);
    return size >= 4 && size % 4 == 0 ? size / 4 : 0;
}

bool DraftSubset::load(const std::string& path, int64_t n_vocab, const void* head_rows, size_t row_bytes, void* stream,
                       const char* who, std::string& err, bool virtual_rows) {
    ids_.clear();
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        int32_t v = 0;
        while (std::fread(&v, sizeof v, 1, f) == 1)
            if (v >= 0 && v < n_vocab) ids_.push_back(v);
        std::fclose(f);
    }
    if (ids_.empty() || row_bytes == 0) {
        std::fprintf(stderr, "%s: %s: no draft tokens read; the drafts use the whole head\n", who, path.c_str());
        ids_.clear();
        return true;
    }
    const int64_t n = (int64_t) ids_.size();
    if (!gpu::alloc_device(&d_ids_, ids_.size() * 4)) {
        err = std::string(who) + ": the draft vocabulary does not fit";
        return false;
    }
    const size_t row_size = (size_t) n * row_bytes;
    if (virtual_rows) {
        const size_t gran = gpu::vmem_granularity();
        if (gran == 0) { err = std::string(who) + ": virtual memory granularity"; return false; }
        const size_t page = std::lcm(gran, size_t(2) << 20);
        mapping_bytes_ = ((row_size + page - 1) / page) * page;
        rows_ = static_cast<uint8_t*>(gpu::vmem_reserve(mapping_bytes_));
        if (rows_ != nullptr) mapping_ = gpu::vmem_map(rows_, mapping_bytes_);
    } else {
        rows_ = static_cast<uint8_t*>(gpu::alloc_device(row_size));
    }
    if (rows_ == nullptr || (virtual_rows && mapping_ == nullptr)) {
        err = std::string(who) + ": the draft head does not fit: " + gpu::last_error();
        draft_head_hint(who, n, (int64_t) row_bytes);
        return false;
    }
    if (!gpu::copy(d_ids_, ids_.data(), ids_.size() * 4)) {
        err = std::string(who) + ": the draft head's tokens";
        return false;
    }
    k::gather_rows(static_cast<const uint8_t*>(head_rows), (int64_t) row_bytes, d_ids_, n, rows_, stream);
    if (!(stream != nullptr ? gpu::stream_sync(stream) : gpu::device_sync())) {
        err = std::string(who) + ": the draft head's rows";
        return false;
    }
    bytes_ = (uint64_t) n * row_bytes + ids_.size() * 4;
    std::fprintf(stderr, "%s: draft head over %lld tokens (%.1f MiB)\n", who, (long long) n,
                 (double) ((uint64_t) n * row_bytes) / 1048576.0);
    return true;
}

bool DraftSubset::park(std::string& err, bool* returned) {
    if (mapping_ == nullptr) { err = "draft head: not mapped"; return false; }
    if (saved_rows_ == nullptr && !gpu::alloc_host(&saved_rows_, mapping_bytes_)) {
        err = "draft head: pinned memory allocation failed"; return false;
    }
    if (!gpu::copy(saved_rows_, rows_, mapping_bytes_) || !gpu::device_sync()) {
        err = "draft head: copy to host failed"; return false;
    }
    if (!(returned != nullptr ? gpu::vmem_unmap_returns(rows_, mapping_bytes_, mapping_, returned)
                              : gpu::vmem_unmap(rows_, mapping_bytes_, mapping_))) {
        err = std::string("draft head: unmap failed: ") + gpu::last_error(); return false;
    }
    mapping_ = nullptr;
    return true;
}

bool DraftSubset::restore(std::string& err) {
    if (mapping_ != nullptr || saved_rows_ == nullptr) { err = "draft head: not parked"; return false; }
    mapping_ = gpu::vmem_map(rows_, mapping_bytes_);
    if (mapping_ == nullptr || !gpu::copy(rows_, saved_rows_, mapping_bytes_) || !gpu::device_sync()) {
        err = "draft head: restore failed"; return false;
    }
    return true;
}

DraftHeadInt2::~DraftHeadInt2() {
    for (const void* p : {(const void*) codes_, (const void*) cand_, (const void*) cand_rows_, (const void*) cand_y_,
                          (const void*) scratch_})
        if (p != nullptr) gpu::free(p);
}

bool DraftHeadInt2::build(int type, int64_t n_rows, int n_in, const std::function<const uint8_t*(int64_t)>& row,
                          size_t exact_row_bytes, std::string& err) {
    if (n_rows <= 0 || n_in <= 0 || n_in % k::kInt2Group != 0) {
        err = "the draft head's rows are not whole groups of 128";
        return false;
    }
    if (k::cpu::native_row_bytes(type, n_in) == 0) {
        err = "the draft head's type has no host dequantizer";
        return false;
    }
    const size_t rb = k::int2_row_bytes(n_in);
    std::vector<uint8_t> host((size_t) n_rows * rb);
    // the rows to floats (ggml's dequantizer) and to the int2 form, on a few threads
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    const int threads = (int) std::clamp<unsigned>(std::thread::hardware_concurrency() / 2, 1u, 16u);
    std::vector<std::thread> pool;
    pool.reserve((size_t) threads);
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            std::vector<float> f((size_t) n_in);
            constexpr int64_t kStep = 256;
            for (int64_t r0 = next.fetch_add(kStep); r0 < n_rows && !bad; r0 = next.fetch_add(kStep))
                for (int64_t r = r0; r < std::min(n_rows, r0 + kStep); ++r) {
                    if (!k::cpu::native_row_to_float(type, row(r), f.data(), n_in)) { bad = true; break; }
                    k::int2_pack_rows(f.data(), 1, n_in, host.data() + (size_t) r * rb);
                }
        });
    for (auto& th : pool) th.join();
    if (bad) {
        err = "the draft head's rows could not be dequantized";
        return false;
    }
    k_ = candidates();
    const size_t scratch = k::topk_scratch_bytes(n_rows, k_);
    scratch_ = gpu::alloc_device(scratch);
    if (scratch_ == nullptr || !gpu::alloc_device(&codes_, host.size()) || !gpu::alloc_device(&cand_, (size_t) k_ * 4) ||
        !gpu::alloc_device(&cand_rows_, (size_t) k_ * exact_row_bytes) || !gpu::alloc_device(&cand_y_, (size_t) k_ * 4) ||
        !gpu::copy(codes_, host.data(), host.size())) {
        err = "no VRAM for the draft head's int2 copy";
        return false;
    }
    bytes_ = host.size() + (size_t) k_ * (8 + exact_row_bytes) + scratch;
    n_rows_ = n_rows;
    n_in_ = n_in;
    exact_row_bytes_ = exact_row_bytes;
    return true;
}

void DraftHeadInt2::logits(const float* x, const uint8_t* head_rows, const Exact& exact, float* out, void* stream,
                           const int32_t* pos) const {
    k::int2_gemv(codes_, x, out, n_in_, n_rows_, stream);
    k::topk(out, n_rows_, k_, cand_, scratch_, stream);
    k::gather_rows(head_rows, (int64_t) exact_row_bytes_, cand_, k_, cand_rows_, stream);
    exact(cand_rows_, k_, cand_y_, stream);
    // From kPadFrom tokens of context the coarse products only pick the candidates: left in the softmax, their error
    // raised its sum and lowered every draft probability, and at long contexts (flatter distributions, more of the
    // sum in the rest of the vocabulary) the drafts stopped early: RTX 4070, 64K serve, 0.45 drafts a token against
    // 0.76 with the head as it is, 56.5 -> 71.8 tok/s with the rest at a probability of 0; RX 9060 XT 64K t07
    // 51.5 -> 56.7.  Shorter, the rest left out made the draft probabilities too high: RX 9060 XT 32K greedy 149 -> 136
    // of ~214 drafts accepted (-5%), t07 1K and 32K 5-16% slower; the RTX 4070 the same either way.  Measured at 1K,
    // 32K and 64K only: the bound sits between 32K and 64K.
    k::int2_pad_from(pos, kPadFrom, out, n_rows_, stream);
    k::scatter_values(cand_y_, cand_, k_, out, stream);
}

bool DraftHeadInt2::choose(const float* x, const uint8_t* head_rows, const Exact& exact,
                           const std::function<void(float*, void*)>& full, float* out, void* stream, const char* who) {
    if (const char* v = strata::env_get("DRAFT_INT2"); v != nullptr && v[0] != 0)
        return std::strtol(v, nullptr, 10) != 0;
    auto time_us = [&](const std::function<void()>& f) {
        constexpr int kWarm = 50, kReps = 50;   // the GPU's clock ramps from idle (LOCAL.md: B70 400 -> 2800 MHz)
        for (int i = 0; i < kWarm; ++i) f();
        gpu::stream_sync(stream);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kReps; ++i) f();
        gpu::stream_sync(stream);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kReps;
    };
    const double a = time_us([&] { full(out, stream); });
    const double b = time_us([&] { logits(x, head_rows, exact, out, stream); });
    const bool on = b < a;
    std::fprintf(stderr, "%s: the drafts' head over %lld tokens: %.0f us a step as it is, %.0f us at 2 bits with %d "
                         "taken again: %s\n", who, (long long) n_rows_, a, b, k_, on ? "2 bits" : "as it is");
    return on;
}

}  // namespace strata::core
