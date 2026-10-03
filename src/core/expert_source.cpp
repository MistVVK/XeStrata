// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/core/pinned.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include "strata/core/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::core {

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    n_expert_ = n_expert;
    n_layers_ = n_layers;
    blobs_ = n_layers * n_expert;
    // A native (IQ) pack's experts.bin (tools/iq_pack.py --experts-bin) has per-layer blob sizes: its size and
    // every blob's offset are the loaded layout's (upstream de159b5); a canonical pack's are n_layers x n_expert
    // x BLOB as before.
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native && (lay.n_layers != n_layers || lay.n_expert != n_expert)) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    const uint64_t want = lay.native ? lay.total : (uint64_t) blobs_ * (uint64_t) strata::kernels::cpu::BLOB;
    const std::string path = pack_dir + "/experts.bin";
    // a native pack without experts.bin: the experts from the GGUF files in place (set_gguf)
    if (lay.native && !gguf_.empty() && !std::ifstream(path, std::ios::binary)) {
        native_ = true;
        if (!open_gguf(err)) { close(); return false; }
        return true;
    }

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if ((uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the expert layout of %lld layers x %lld experts is %llu B - "
                      "this is not the pack this geometry came from",
                      path.c_str(), (unsigned long long) st.st_size, (long long) n_layers, (long long) n_expert,
                      (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
    bytes_ = want;
    native_ = lay.native;
    return true;
}

void FileExpertSource::close() {
    if (base_ != nullptr && maps_.empty()) munmap((void*) base_, (size_t) bytes_);
    for (const Map& m : maps_) {
        if (m.base != nullptr) munmap((void*) m.base, (size_t) m.bytes);
        if (m.fd >= 0) ::close(m.fd);
    }
    if (res_base_ != nullptr) munmap(res_base_, (size_t) res_bytes_);   // the unmap releases the lock
    res_base_ = nullptr;
    res_locked_ = false;
    res_bytes_ = 0;
    res_count_ = 0;
    res_off_.clear();
    ram_reads_.store(0);
    maps_.clear();
    role_ptr_.clear();
    role_bytes_.clear();
    file_reads_.store(0);
    file_read_bytes_.store(0);
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_buf_.clear();
        stage_key_.clear();
        stage_epoch_.clear();
        stage_busy_.clear();
        stage_of_.clear();
        stage_blob_ = 0;
        epoch_ = 0;
        last_layer_ = -1;
    }
    bytes_ = 0;
    native_ = false;
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    base_ = nullptr;
    blobs_ = 0;
    reads_ = 0;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0) return nullptr;
    // **`expert >= n_expert_` IS CHECKED SEPARATELY, AND THE FLAT INDEX ALONE DOES NOT CATCH IT.**  The blob
    // index is `layer * n_expert + expert`, so `blob(0, 512)` has flat index 512 - which is in range, and is
    // `blob(1, 0)`.  A router id one past the end of a layer would then read the NEXT LAYER's first expert:
    // finite, correctly sized, and wrong.  Layer and expert are separate axes and are validated as such.
    if (expert >= n_expert_) return nullptr;
    const int64_t i = layer * n_expert_ + expert;
    if (i >= blobs_ || layer >= n_layers_) return nullptr;
    ++reads_;
    if (const uint8_t* r = resident(layer, expert)) {
        ram_reads_.fetch_add(1, std::memory_order_relaxed);
        return r;
    }
    if (!role_ptr_.empty()) return staged_blob(layer, expert);
    if (native_) return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
    return base_ + (size_t) i * strata::kernels::cpu::BLOB;
}

bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

// A native pack without experts.bin (upstream 3889344): every file native_experts.txt names is mapped, and an
// expert's blob [gate rows | up rows | down rows] is three slices of three tensors, possibly in two files (UD-Q4_K_XL's
// layer 11).  Nothing is read at open; a blob is assembled when it is asked for (`blob`, into a pool of buffers) or
// copied where it is needed (`copy_blob`: the prompt path's stager buffers).
bool FileExpertSource::open_gguf(std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.gguf_off.size() != (size_t) (3 * n_layers_) || lay.fmt.size() != (size_t) n_layers_) {
        err = "FileExpertSource: the expert layout has no GGUF offsets for --mmap-experts without experts.bin";
        return false;
    }
    const size_t cut = gguf_.find_last_of('/');
    const std::string dir = cut == std::string::npos ? std::string() : gguf_.substr(0, cut + 1);
    std::vector<std::pair<std::string, size_t>> index;
    role_ptr_.assign((size_t) (3 * n_layers_), nullptr);
    role_bytes_.assign((size_t) (3 * n_layers_), 0);
    for (int64_t l = 0; l < n_layers_; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * l + r);
            const std::string path = lay.gguf_file.size() > i && !lay.gguf_file[i].empty() ? dir + lay.gguf_file[i]
                                                                                            : gguf_;
            auto it = std::find_if(index.begin(), index.end(), [&](const auto& x) { return x.first == path; });
            if (it == index.end()) {
                Map m;
                const int fd = ::open(path.c_str(), O_RDONLY);
                struct stat st{};
                if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
                    if (fd >= 0) ::close(fd);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                void* view = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: cannot map " + path; return false; }
                m.fd = fd;
                m.bytes = (uint64_t) st.st_size;
                m.base = (const uint8_t*) view;
                maps_.push_back(m);
                index.emplace_back(path, maps_.size() - 1);
                it = index.end() - 1;
            }
            const Map& m = maps_[it->second];
            const uint64_t at = lay.gguf_off[i], bytes = per[r] * (uint64_t) n_expert_;
            if (at == 0 || at > m.bytes || bytes > m.bytes - at) {
                err = "FileExpertSource: layer " + std::to_string(l) + "'s expert span runs past the end of " + path;
                return false;
            }
            role_ptr_[i] = m.base + (size_t) at;
            role_bytes_[i] = per[r];
        }
    }
    base_ = maps_.front().base;   // "opened"
    for (int64_t l = 0; l < n_layers_; ++l) stage_blob_ = std::max<uint64_t>(stage_blob_, lay.bytes[(size_t) l]);
    return true;
}

bool FileExpertSource::copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const {
    if (dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    uint64_t at = 0;
    for (int r = 0; r < 3; ++r) {
        const size_t i = (size_t) (3 * layer + r);
        const uint64_t per = role_bytes_[i];
        std::memcpy(dst + at, role_ptr_[i] + (size_t) ((uint64_t) expert * per), (size_t) per);
        at += per;
    }
    return true;
}

// A blob assembled from the three role slices.  The buffer of a (layer, expert) is reused for another only once
// its blob has not been asked for during kStageAge layers (begin_layer), and never while it is being filled: the
// pool computes a layer's misses before it starts the next, and every other consumer copies the blob at once
// (`transient`).
bool FileExpertSource::claim_stage(int64_t key, size_t& v, bool& fill) {
    fill = false;
    const auto it = stage_of_.find(key);
    if (it != stage_of_.end()) {
        v = it->second;
        stage_epoch_[v] = epoch_;
        return true;
    }
    v = stage_buf_.size();
    for (size_t i = 0; i < stage_buf_.size(); ++i)
        if (!stage_busy_[i] && stage_epoch_[i] + kStageAge <= epoch_) { v = i; break; }
    if (v == stage_buf_.size()) {
        stage_buf_.emplace_back(new (std::nothrow) uint8_t[(size_t) stage_blob_]);
        if (!stage_buf_.back()) { stage_buf_.pop_back(); return false; }
        stage_key_.push_back(-1);
        stage_epoch_.push_back(0);
        stage_busy_.push_back(0);
        if (stage_buf_.size() == 1024 && !stage_grew_) {
            stage_grew_ = true;
            std::fprintf(stderr, "FileExpertSource: %zu blobs assembled from the GGUF are in use at once (%.2f GiB)\n",
                         stage_buf_.size(), (double) stage_buf_.size() * (double) stage_blob_ / 1073741824.0);
        }
    } else {
        stage_of_.erase(stage_key_[v]);
    }
    stage_key_[v] = key;
    stage_epoch_[v] = epoch_;
    stage_busy_[v] = 1;
    stage_of_[key] = v;
    fill = true;
    return false;
}

// Fills buffer `v` (reserved by claim_stage) outside the lock, then publishes it.
bool FileExpertSource::fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst) {
    const bool ok = copy_from_files(layer, expert, dst);
    if (ok) {
        file_read_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer), std::memory_order_relaxed);
        file_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_busy_[v] = 0;
        if (!ok) {
            stage_of_.erase(stage_key_[v]);
            stage_key_[v] = -1;
        }
    }
    stage_cv_.notify_all();
    return ok;
}

// A blob assembled from the three role slices.  The buffer of a (layer, expert) is reused for another only once
// its blob has not been asked for during kStageAge layers (begin_layer), and never while it is being filled: the
// pool computes a layer's misses before it starts the next, and every other consumer copies the blob at once
// (`transient`).
const uint8_t* FileExpertSource::staged_blob(int64_t layer, int64_t expert) {
    const int64_t key = layer * n_expert_ + expert;
    size_t v = 0;
    bool fill = false;
    uint8_t* dst = nullptr;
    {
        std::unique_lock<std::mutex> lk(stage_mu_);
        const bool have = claim_stage(key, v, fill);
        if (!have && !fill) return nullptr;
        dst = stage_buf_[v].get();
        if (have) {   // another thread (a prefetch) may be filling it: wait for that
            stage_cv_.wait(lk, [&] { return !stage_busy_[v] || stage_key_[v] != key; });
            return stage_key_[v] == key ? dst : nullptr;
        }
    }
    return fill_stage(v, layer, expert, dst) ? dst : nullptr;
}

bool FileExpertSource::set_resident(uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>& rank,
                                    size_t first, int threads, std::string& err) {
    if (base_ == nullptr) { err = "FileExpertSource::set_resident: nothing is open"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto bytes_of = [&](int64_t l) {
        return lay.native ? (uint64_t) lay.blob_bytes(l) : (uint64_t) strata::kernels::cpu::BLOB;
    };
    std::vector<uint64_t> off((size_t) blobs_, kNotResident);
    std::vector<std::pair<int32_t, int32_t>> take;
    uint64_t total = 0;
    std::vector<char> ranked((size_t) blobs_, 0);
    auto consider = [&](int64_t l, int64_t e) {
        if (l < 0 || e < 0 || l >= n_layers_ || e >= n_expert_) return true;
        const size_t i = (size_t) (l * n_expert_ + e);
        if (off[i] != kNotResident) return true;
        const uint64_t b = (bytes_of(l) + 63) & ~(uint64_t) 63;
        if (total + b > budget_bytes) return false;
        off[i] = total;
        total += b;
        take.emplace_back((int32_t) l, (int32_t) e);
        return true;
    };
    for (size_t j = 0; j < rank.size(); ++j) {
        const auto& pr = rank[j];
        if (pr.first >= 0 && pr.first < n_layers_ && pr.second >= 0 && pr.second < n_expert_)
            ranked[(size_t) (pr.first * n_expert_ + pr.second)] = 1;   // the GPU cache's part (j < first) too
        if (j >= first && !consider(pr.first, pr.second)) break;
    }
    for (int64_t i = 0; i < blobs_ && total < budget_bytes; ++i)   // then whatever the profile does not rank
        if (!ranked[(size_t) i] && !consider(i / n_expert_, i % n_expert_)) break;
    if (take.empty() || total == 0) return true;
    void* mem = mmap(nullptr, (size_t) total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        err = "FileExpertSource: cannot reserve " + std::to_string(total >> 20) + " MiB of RAM for the resident experts";
        return false;
    }
    auto* base = static_cast<uint8_t*>(mem);
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    auto work = [&] {
        for (size_t j; (j = next.fetch_add(1)) < take.size() && !bad;) {
            const auto [l, e] = take[j];
            uint8_t* dst = base + off[(size_t) (l * n_expert_ + e)];
            const bool ok = !role_ptr_.empty() ? copy_from_files(l, e, dst)
                                               : [&] {
                                                     const uint8_t* b = native_ ? base_ + lay.blob_offset(l, e)
                                                                                : base_ + (size_t) (l * n_expert_ + e) *
                                                                                              strata::kernels::cpu::BLOB;
                                                     std::memcpy(dst, b, (size_t) bytes_of(l));
                                                     return true;
                                                 }();
            if (!ok) bad = true;
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < std::max(1, threads); ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
    if (bad) {
        munmap(mem, (size_t) total);
        err = "FileExpertSource: reading the resident experts from the files failed";
        return false;
    }
    // locked in RAM where the limit allows it: swapped out, the resident experts were page faults again (3-4.6 GiB of
    // a 40 GiB copy went to swap on the 92 GB development PC, and the decode ran at half speed)
    res_locked_ = mlock(mem, (size_t) total) == 0;
    res_off_ = std::move(off);
    res_base_ = base;
    res_bytes_ = total;
    res_count_ = (int64_t) take.size();
    return true;
}

void FileExpertSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    if (role_ptr_.empty() || n <= 0 || layer < 0 || layer >= n_layers_) return;
    struct Fill { size_t v; int64_t e; uint8_t* dst; };
    std::vector<Fill> todo;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_ || resident(layer, e) != nullptr) continue;
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(layer * n_expert_ + e, v, fill) && fill) todo.push_back({v, e, stage_buf_[v].get()});
        }
    }
    if (todo.empty()) return;
    // the page faults of a mapped read are one outstanding request each: several threads keep the SSD's queue full
    static const int threads = [] {
        const char* v = std::getenv("STRATA_FETCH_THREADS");
        return v != nullptr ? std::clamp((int) std::strtol(v, nullptr, 10), 1, 64) : 8;
    }();
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < todo.size();) (void) fill_stage(todo[i].v, layer, todo[i].e, todo[i].dst);
    };
    const size_t nt = std::min<size_t>(todo.size(), (size_t) threads);
    std::vector<std::thread> th;
    for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
}

bool FileExpertSource::transient(int64_t layer, int64_t expert) const {
    return !role_ptr_.empty() && resident(layer, expert) == nullptr;
}

bool FileExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (role_ptr_.empty() || resident(layer, expert) != nullptr) return ExpertSource::copy_blob(layer, expert, dst);
    if (!copy_from_files(layer, expert, dst)) return false;
    file_read_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer), std::memory_order_relaxed);
    return true;
}

void FileExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    (void) ids;
    (void) k;
    std::lock_guard<std::mutex> lk(stage_mu_);
    if (layer != last_layer_) {
        last_layer_ = layer;
        ++epoch_;
    }
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`) are fixed
// arrays of this many entries: MAXT tokens of the model's 10 routed experts must fit, and a larger k is refused at
// run time rather than written past them (upstream 64cfe3c)
constexpr int64_t kMaxWindowEntries = 128;
static_assert((int64_t) strata::kernels::cpu::MAXT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (k < 1 || n_tok * k > kMaxWindowEntries) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->device_alias(d.layers, 0) != nullptr;
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        // pinned() first: blob() is a file read with the GGUF in place (upstream cacffa9)
                        const uint8_t* src = d.src->pinned(d.layers, e) ? d.src->blob(d.layers, e) : nullptr;
                        if (src != nullptr) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (native && lay.fmt[(size_t) d.layers].gu_type == 42)   // a native Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (native)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    else
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    const auto c2 = std::chrono::steady_clock::now();
    {   // the experts the CPU computes, fetched together (the GGUF in place reads them on several threads)
        static thread_local std::vector<int64_t> miss;
        miss.clear();
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (kind[i] < 0 && ids[i] >= 0 && ids[i] < d.n_expert &&
                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())
                miss.push_back(ids[i]);
        d.src->prefetch(d.layers, miss.data(), (int64_t) miss.size());
    }
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    strata::gpu::Stream cs = stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (!strata::gpu::memset_async(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) ||
            !strata::gpu::copy_async(d.d_slot, d.h_slot.data(), list_bytes, cs) ||
            !strata::gpu::copy_async(d.d_dst, d.h_dst.data(), list_bytes, cs)) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) strata::gpu::event_record((strata::gpu::Event*) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) strata::gpu::event_query((strata::gpu::Event*) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (strata::gpu::event_query((strata::gpu::Event*) d.hit_done)) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
LoadStats load_experts_gguf(const std::string& gguf, const std::vector<uint8_t*>& layer_dst,
                            const strata::kernels::cpu::ExpertLayout& lay, int threads) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    // a layer's tensors may sit in other shards of the model (native_experts.txt v3/v4): names beside `gguf`
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l, int r) -> std::string {
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) (3 * l + r)].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) (3 * l + r)];
    };
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string name = file_of(l, r);
                if (name != open_name) {
                    f.close();
                    f.clear();
                    f.open(name, std::ios::binary);
                    if (!f) { bad = true; return; }
                    open_name = name;
                }
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        std::memcpy(layer_dst[(size_t) l] + (lay.blob_offset(l, (int64_t) e) - lay.layer_offset(l)) + at[r],
                                    buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want) {
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    // STRATA_ARENA_HOST_USM=1: every layer in its own host USM block, which GPU kernels can read (the PCIe share)
    static const bool host_usm = [] { const char* e = std::getenv("STRATA_ARENA_HOST_USM"); return e && *e == '1'; }();
    PinnedArena* a = nullptr;
    try {
        a = new PinnedArena(want + (uint64_t) blob, bounds, (uint64_t) blob, host_usm);
    } catch (const std::exception& e) {
        err = std::string("ArenaExpertSource: ") + e.what();
        return false;
    }
    if (!a->valid()) {
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(want) + " B)";
        return false;
    }
    std::vector<uint8_t*> ldst;
    for (int64_t l = 0; l < n_layers; ++l) ldst.push_back(a->at(lay.layer_offset(l)));
    const LoadStats st = from_gguf ? load_experts_gguf(gguf_, ldst, lay, threads)
                                   : load_experts_ranges(path, ldst, loff, lbytes, threads, /*chunk=*/8u << 20);
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (!strata::gpu::device_pointer(&d, (void*) a->at(off))) {
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    return true;
}

void ArenaExpertSource::close() {
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return ((const PinnedArena*) arena_)->at(strata::kernels::cpu::expert_layout().blob_offset(layer, expert));
}

}  // namespace strata::core
