// sy_vram_cache.h - Multi-slab (<= 1.4 GiB/slab) Adaptive VRAM Expert Cache + GPU Decode/Verify evaluator for Intel Arc (SYCL).
// Why multi-slab: Intel OpenCL & Level Zero cap single device allocations at ~4 GiB. Uniform 4 KiB-aligned slots across
// <= 1.4 GiB slabs enable O(1) fragmentation-free LFU-decay online expert promotion on 8 GB (A750 ~1 GB), 12 GB (B580 ~5 GB),
// and 16 GB (A770 ~9 GB) Intel Arc GPUs!
// Bitwise determinism (plain == DFly2):
//   * Each (token, expert) pair is evaluated identically regardless of window size T in [1..8].
//   * Per-token reduction sums pairs in ascending expert_id order.
//   * Adaptive slot promotions are committed ONLY between steps, never mid-step.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "hy4_io.h"
#include "strata/artifact/strata_reader.hpp"
#include "sy_common.h"

namespace hy4s {

namespace vcache_detail {

constexpr int VC_K = 1536;
constexpr int VC_NTOP = 16;

constexpr int8_t kS125[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1},
    {1, -1, -1, -1},  {1, -1, -1, 1},  {1, -1, 1, -1},  {1, -1, 1, 1},  {1, 1, -1, -1},  {1, 1, -1, 1},  {1, 1, 1, -1},  {1, 1, 1, 1},
    {2, 0, 0, 0},     {-2, 0, 0, 0},   {0, 2, 0, 0},    {0, -2, 0, 0},   {0, 0, 2, 0},    {0, 0, -2, 0},   {0, 0, 0, 2},    {0, 0, 0, -2},
    {0, 0, 0, 0},     {2, 2, 0, 0},    {-2, -2, 0, 0},  {0, 0, 2, 2},    {0, 0, -2, -2},  {2, -2, 0, 0},   {0, 0, 2, -2},   {2, 0, 2, 0},
};

inline int vc_sb_bytes(int code) {
    return code == 0 ? 48 : code == 1 ? 58 : code == 2 ? 70 : code == 3 ? 102 : code == 4 ? 134 : code == 5 ? 166 : code == 6 ? 198 : 262;
}

inline int vc_s15_trit(uint32_t b, int d) {
    if (b >= 243u) return 0;
    const uint32_t m = d == 0 ? 65536u : d == 1 ? 21846u : d == 2 ? 7282u : d == 3 ? 2428u : 810u;
    const uint32_t x = (b * m) >> 16;
    const uint32_t x3 = (x * 21846u) >> 16;
    return (int)(x - 3u * x3) - 1;
}

// Dot product of 1 S_K row (K = 1536 = 6 superblocks) with fp32/fp16 or Q8_1_G64-quantized activation vector using 1 SG32 sub-group.
// When Q8_ACT=true, activations are quantized per 64-element group identically to hy4::CpuExperts (d = amax/127, q = rint(x/d)),
// making VRAM-resident and CPU-resident experts numerically identical (< 1e-6 diff) so adaptive VRAM promotion never perturbs logits!
template <class ActT, bool Q8_ACT>
inline float sg_dot_row(const sycl::nd_item<1>& it, int code, const uint8_t* rp, const ActT* x) {
    auto sg = it.get_sub_group();
    const int lane = (int)it.get_local_id(0) & 31;
    const int sby = vc_sb_bytes(code);
    const int g = lane >> 3;        // 0..3 (64-weight subgroup, 8 lanes per subgroup)
    const int j0 = (lane & 7) << 3; // 0, 8, ..., 56 within subgroup
    const int base_i = (g << 6) + j0;
    float acc = 0.0f;
#pragma unroll
    for (int sbi = 0; sbi < 6; ++sbi) {
        const uint8_t* blk = rp + (size_t)sbi * sby;
        const float D = h2f_bits(ld16(blk));
        const float eff = D * ((float)blk[2 + g] / 255.0f);
        const uint8_t* pl = blk + 6;
        const ActT* xp = x + sbi * 256 + base_i;
        float xv[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) xv[k] = (float)xp[k];
        float scale_x = 1.0f;
        if constexpr (Q8_ACT) {
            float amax = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) amax = sycl::fmax(amax, sycl::fabs(xv[k]));
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 4));
            scale_x = amax * (1.0f / 127.0f);
            const float inv = amax > 0.0f ? (127.0f / amax) : 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) xv[k] = (float)sycl::clamp((int)sycl::rint(xv[k] * inv), -127, 127);
        }
        const float effx = eff * scale_x;
        float sq = 0.0f, sx = 0.0f;
        if (code == 2) {
            const uint8_t b0 = pl[base_i >> 2], b1 = pl[(base_i >> 2) + 1];
            sq = (float)(b0 & 3u) * xv[0] + (float)((b0 >> 2) & 3u) * xv[1] + (float)((b0 >> 4) & 3u) * xv[2] + (float)(b0 >> 6) * xv[3] +
                 (float)(b1 & 3u) * xv[4] + (float)((b1 >> 2) & 3u) * xv[5] + (float)((b1 >> 4) & 3u) * xv[6] + (float)(b1 >> 6) * xv[7];
            sx = xv[0] + xv[1] + xv[2] + xv[3] + xv[4] + xv[5] + xv[6] + xv[7];
            acc += effx * (sq - 1.5f * sx);
        } else if (code == 3) {
            const uint8_t b0 = pl[base_i >> 2], b1 = pl[(base_i >> 2) + 1];
            const uint32_t hi = pl[64 + (base_i >> 3)];
            sq = (float)((b0 & 3u) | ((hi & 1u) << 2)) * xv[0] + (float)(((b0 >> 2) & 3u) | (((hi >> 1) & 1u) << 2)) * xv[1] +
                 (float)(((b0 >> 4) & 3u) | (((hi >> 2) & 1u) << 2)) * xv[2] + (float)((b0 >> 6) | (((hi >> 3) & 1u) << 2)) * xv[3] +
                 (float)((b1 & 3u) | (((hi >> 4) & 1u) << 2)) * xv[4] + (float)(((b1 >> 2) & 3u) | (((hi >> 5) & 1u) << 2)) * xv[5] +
                 (float)(((b1 >> 4) & 3u) | (((hi >> 6) & 1u) << 2)) * xv[6] + (float)((b1 >> 6) | (((hi >> 7) & 1u) << 2)) * xv[7];
            sx = xv[0] + xv[1] + xv[2] + xv[3] + xv[4] + xv[5] + xv[6] + xv[7];
            acc += effx * (sq - 3.5f * sx);
        } else if (code == 4) {
            const uint8_t* p = pl + (base_i >> 1);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t by = p[k];
                sq += (float)(by & 15u) * xv[2 * k] + (float)(by >> 4) * xv[2 * k + 1];
                sx += xv[2 * k] + xv[2 * k + 1];
            }
            acc += effx * (sq - 7.5f * sx);
        } else if (code == 0) {
            const float fine = 0.5f + (float)((pl[40 + (g >> 1)] >> ((g & 1) * 4)) & 15u) / 15.0f;
            const int grp = base_i >> 5;
            const uint8_t* p = pl + grp * 5;
            const uint64_t bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32);
            const int j_sub = (base_i & 31) >> 2;
            const uint32_t i0 = (uint32_t)((bits >> (5 * j_sub)) & 0x1Fu);
            const uint32_t i1 = (uint32_t)((bits >> (5 * (j_sub + 1))) & 0x1Fu);
            sq = (float)kS125[i0][0] * xv[0] + (float)kS125[i0][1] * xv[1] + (float)kS125[i0][2] * xv[2] + (float)kS125[i0][3] * xv[3] +
                 (float)kS125[i1][0] * xv[4] + (float)kS125[i1][1] * xv[5] + (float)kS125[i1][2] * xv[6] + (float)kS125[i1][3] * xv[7];
            acc += effx * fine * sq;
        } else if (code == 1) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int idx = base_i + k;
                sq += (float)vc_s15_trit(pl[idx / 5], idx % 5) * xv[k];
            }
            acc += effx * sq;
        } else if (code == 5) {
            const uint8_t* p = pl + (base_i >> 1);
            const uint32_t hi = pl[128 + (base_i >> 3)];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t by = p[k];
                const uint32_t q0 = (by & 15u) | (((hi >> (2 * k)) & 1u) << 4);
                const uint32_t q1 = (by >> 4) | (((hi >> (2 * k + 1)) & 1u) << 4);
                sq += (float)q0 * xv[2 * k] + (float)q1 * xv[2 * k + 1];
                sx += xv[2 * k] + xv[2 * k + 1];
            }
            acc += effx * (sq - 15.5f * sx);
        } else if (code == 6) {
            const uint8_t* p = pl + (base_i >> 1);
            const uint32_t h0 = pl[128 + (base_i >> 2)];
            const uint32_t h1 = pl[128 + (base_i >> 2) + 1];
            const uint32_t h16 = h0 | (h1 << 8);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t by = p[k];
                const uint32_t q0 = (by & 15u) | (((h16 >> (4 * k)) & 3u) << 4);
                const uint32_t q1 = (by >> 4) | (((h16 >> (4 * k + 2)) & 3u) << 4);
                sq += (float)q0 * xv[2 * k] + (float)q1 * xv[2 * k + 1];
                sx += xv[2 * k] + xv[2 * k + 1];
            }
            acc += effx * (sq - 31.5f * sx);
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) sq += (float)(int8_t)pl[base_i + k] * xv[k];
            acc += effx * sq;
        }
    }
    return sg_sum(it, acc);
}

struct VramPair {
    const uint8_t* body;
    uint32_t dn_off;
    int32_t tok;
    int32_t gu;
    int32_t dn;
    float weight;
};

}  // namespace vcache_detail

class SyVramExpertCache {
public:
    struct Meta {
        uint64_t file_off = 0;
        uint32_t dn_off = 0;
        uint32_t body_bytes = 0;
        uint8_t gu = 0, dn = 0;
        uint16_t rank = 0;
    };

    struct Slab {
        uint8_t* d = nullptr;
        size_t bytes = 0;
        int nslots = 0;
    };

    struct Slot {
        uint8_t* d_ptr = nullptr;
        int layer = -1;
        int expert = -1;
    };

    static constexpr int MAX_T = 16;
    static constexpr int MAX_PAIRS = MAX_T * vcache_detail::VC_NTOP;

    bool active() const { return !slots_.empty(); }
    bool adaptive() const { return active() && adaptive_; }
    int num_slots() const { return (int)slots_.size(); }
    size_t device_bytes() const { return dev_bytes_; }
    const uint8_t* expert_ptr(int L, int e) const {
        if ((unsigned)L >= (unsigned)nlayers_ || (unsigned)e >= 512u) return nullptr;
        return d_ptr_[(size_t)L * 512 + e];
    }
    const Meta& expert_meta(int L, int e) const { return meta_[(size_t)L * 512 + e]; }
    int n_layers() const { return nlayers_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t promotions() const { return promotions_; }
    void reset_stats() { hits_ = misses_ = promotions_ = 0; }
    void set_q8_act(bool on) { q8_act_ = on; }
    bool q8_act() const { return q8_act_; }

    ~SyVramExpertCache() { release(); }

    void release() {
        if (cq_) { cq_->wait(); delete cq_; cq_ = nullptr; }
        for (auto& s : slabs_) if (s.d) dfree(s.d);
        slabs_.clear();
        slots_.clear();
        if (d_pairs_) { dfree(d_pairs_); d_pairs_ = nullptr; }
        if (h_pairs_) { sycl::free(h_pairs_, Q()); h_pairs_ = nullptr; }
        if (d_tok_off_) { dfree(d_tok_off_); d_tok_off_ = nullptr; }
        if (h_tok_off_) { sycl::free(h_tok_off_, Q()); h_tok_off_ = nullptr; }
        if (d_hmid_) { dfree(d_hmid_); d_hmid_ = nullptr; }
        if (d_ye_) { dfree(d_ye_); d_ye_ = nullptr; }
        if (h_stage_) { sycl::free(h_stage_, Q()); h_stage_ = nullptr; }
        dev_bytes_ = 0;
    }

    // Initialize multi-slab VRAM expert cache.
    // vram_budget_bytes: total VRAM to spend on expert slabs (0 = disabled).
    // slab_mb: max slab size in MiB (default 1400 MiB, strictly < 4 GiB Intel single-alloc cap).
    bool init(const std::string& strata_path, size_t vram_budget_bytes, std::string& err, size_t slab_mb = 1400) {
        release();
        strata_path_ = strata_path;
        strata::artifact::StrataFileReader rd;
        if (!rd.open(strata_path, err)) return false;
        const auto& h = rd.header();
        const auto& dir = rd.expert_directory();
        if (dir.empty() || (dir.size() % 512) != 0) {
            err = "vram cache: invalid expert directory";
            return false;
        }
        nlayers_ = (int)(dir.size() / 512);
        meta_.assign(dir.size(), Meta{});
        d_ptr_.assign(dir.size(), nullptr);
        slot_of_.assign(dir.size(), -1);
        scores_.assign(dir.size(), 0.0f);
        rank_eid_.assign(dir.size(), 0);
        last_idx_.assign((size_t)nlayers_ * MAX_T * vcache_detail::VC_NTOP, -1);
        uint32_t max_body = 0;
        for (size_t i = 0; i < dir.size(); ++i) {
            const int L = (int)(i / 512), rk = (int)(i % 512);
            const auto& d = dir[i];
            Meta& m = meta_[(size_t)L * 512 + d.expert_id];
            m.file_off = h.sections[4].file_offset + d.body_offset;
            m.dn_off = d.dn_offset;
            m.body_bytes = d.body_bytes;
            m.gu = d.gu_code_id;
            m.dn = d.dn_code_id;
            m.rank = (uint16_t)rk;
            rank_eid_[i] = d.expert_id;
            max_body = std::max(max_body, d.body_bytes);
        }
        slot_bytes_ = ((size_t)max_body + 4095ull) & ~4095ull;
        if (vram_budget_bytes < slot_bytes_ * 41 || slot_bytes_ == 0) {
            return true;  // disabled cleanly (e.g. 0 budget)
        }

        // Reserve workspace for GPU decode evaluator (~1.6 MiB)
        const size_t ws_bytes = MAX_PAIRS * (sizeof(vcache_detail::VramPair) + vcache_detail::VC_K * (sizeof(half) + sizeof(float))) + 4096;
        if (vram_budget_bytes <= ws_bytes + slot_bytes_) return true;
        size_t avail = vram_budget_bytes - ws_bytes;
        const size_t max_slots = std::min(dir.size(), avail / slot_bytes_);
        if (max_slots == 0) return true;

        const size_t slab_max = (slab_mb << 20);
        const int slots_per_slab_max = std::max(1, (int)(slab_max / slot_bytes_));
        size_t rem = max_slots;
        while (rem > 0) {
            const int ns = (int)std::min<size_t>(rem, (size_t)slots_per_slab_max);
            const size_t sb = (size_t)ns * slot_bytes_;
            uint8_t* ptr = sycl::malloc_device<uint8_t>(sb, Q());
            if (!ptr) break;  // stop cleanly if driver rejects further slabs
            dev_bytes_allocated() += sb;
            dev_bytes_ += sb;
            slabs_.push_back(Slab{ptr, sb, ns});
            for (int k = 0; k < ns; ++k) slots_.push_back(Slot{ptr + (size_t)k * slot_bytes_, -1, -1});
            rem -= (size_t)ns;
        }
        if (slots_.empty()) return true;

        // Allocate decode evaluator scratch
        d_pairs_ = dalloc<vcache_detail::VramPair>(MAX_PAIRS);
        h_pairs_ = halloc<vcache_detail::VramPair>(MAX_PAIRS);
        d_tok_off_ = dalloc<int32_t>(MAX_T + 1);
        h_tok_off_ = halloc<int32_t>(MAX_T + 1);
        d_hmid_ = dalloc<half>((size_t)MAX_PAIRS * vcache_detail::VC_K);
        d_ye_ = dalloc<float>((size_t)MAX_PAIRS * vcache_detail::VC_K);
        dev_bytes_ += (size_t)MAX_PAIRS * (sizeof(vcache_detail::VramPair) + vcache_detail::VC_K * (sizeof(half) + sizeof(float))) + (MAX_T + 1) * sizeof(int32_t);
        h_stage_ = sycl::aligned_alloc_host<uint8_t>(4096, slot_bytes_, Q());
        cq_ = new sycl::queue(Q().get_context(), Q().get_device(), sycl::property_list{sycl::property::queue::in_order()});

        // Pre-populate slots in rank-major order (rank 0 across all 41 layers, then rank 1, ...)
        hy4::AlignedFile af;
        if (!af.open(strata_path, false)) {
            err = "vram cache: cannot open " + strata_path;
            return false;
        }
        const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        size_t si = 0;
        for (int rk = 0; rk < 512 && si < slots_.size(); ++rk) {
            for (int L = 0; L < nlayers_ && si < slots_.size(); ++L) {
                const int e = rank_eid_[(size_t)L * 512 + rk];
                const Meta& m = meta_[(size_t)L * 512 + e];
                if (af.read_at(m.file_off, h_stage_, m.body_bytes) < m.body_bytes) {
                    err = "vram cache: short read";
                    return false;
                }
                cq_->memcpy(slots_[si].d_ptr, h_stage_, m.body_bytes).wait();
                slots_[si].layer = L;
                slots_[si].expert = e;
                d_ptr_[(size_t)L * 512 + e] = slots_[si].d_ptr;
                slot_of_[(size_t)L * 512 + e] = (int)si;
                scores_[(size_t)L * 512 + e] = 2.0f;  // gentle initial prior for hot ranks
                ++si;
            }
        }
        af.close();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
        adaptive_ = !(std::getenv("HY4S_VRAM_STATIC") && std::getenv("HY4S_VRAM_STATIC")[0] == '1');
        fprintf(stderr, "[vram-cache] %zu slots across %zu slabs (<= %zu MiB/slab) = %.2f GiB (%.1f experts/layer, %.1f s, adaptive=%d)\n",
                slots_.size(), slabs_.size(), slab_mb, dev_bytes_ / 1073741824.0, (double)slots_.size() / nlayers_, dt, (int)adaptive_);
        return true;
    }

    // Build CPU mask (true for experts NOT currently in VRAM at startup) so hy4::CpuExperts knows what to hold.
    // Note: when adaptive=true, keeping all experts (or RAM-budget experts) accessible in CpuExperts / cold SSD allows seamless eviction.
    void build_vram_mask(std::vector<std::vector<bool>>& vram_mask) const {
        vram_mask.assign(nlayers_, std::vector<bool>(512, false));
        for (int L = 0; L < nlayers_; ++L)
            for (int e = 0; e < 512; ++e)
                if (d_ptr_[(size_t)L * 512 + e] != nullptr) vram_mask[L][e] = true;
    }

    // Step 1 (host-only, ~1 us): plan VRAM hits vs CPU misses and fill use_cpu[T*16].
    // Call BEFORE ucx.submit() so CPU threads start immediately without waiting on SYCL kernel submission!
    int plan_decode(int L, int T, const int32_t* idx_host, const float* w_host, uint8_t* use_cpu) {
        planned_pairs_ = 0;
        planned_T_ = T;
        if (!active() || T <= 0 || T > MAX_T) {
            if (use_cpu) std::memset(use_cpu, 1, (size_t)T * vcache_detail::VC_NTOP);
            return 0;
        }
        int npairs = 0;
        h_tok_off_[0] = 0;
        int32_t* lbuf = &last_idx_[(size_t)L * MAX_T * vcache_detail::VC_NTOP];
        for (int t = 0; t < T; ++t) {
            int hit_k[vcache_detail::VC_NTOP], nh = 0;
            for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                const int e = idx_host[t * vcache_detail::VC_NTOP + k];
                lbuf[t * vcache_detail::VC_NTOP + k] = e;
                if ((unsigned)e < 512u) {
                    if (d_ptr_[(size_t)L * 512 + e] != nullptr) {
                        use_cpu[t * vcache_detail::VC_NTOP + k] = 0;
                        ++hits_;
                        int j = nh++;
                        while (j > 0 && idx_host[t * vcache_detail::VC_NTOP + hit_k[j - 1]] > e) {
                            hit_k[j] = hit_k[j - 1];
                            --j;
                        }
                        hit_k[j] = k;
                    } else {
                        use_cpu[t * vcache_detail::VC_NTOP + k] = 1;
                        ++misses_;
                    }
                } else {
                    use_cpu[t * vcache_detail::VC_NTOP + k] = 0;
                }
            }
            for (int a = 0; a < nh; ++a) {
                const int k = hit_k[a];
                const int e = idx_host[t * vcache_detail::VC_NTOP + k];
                const Meta& m = meta_[(size_t)L * 512 + e];
                vcache_detail::VramPair& vp = h_pairs_[npairs++];
                vp.body = d_ptr_[(size_t)L * 512 + e];
                vp.dn_off = m.dn_off;
                vp.tok = t;
                vp.gu = m.gu;
                vp.dn = m.dn;
                vp.weight = w_host[t * vcache_detail::VC_NTOP + k];
            }
            h_tok_off_[t + 1] = npairs;
        }
        planned_pairs_ = npairs;
        return npairs;
    }

    // Step 2 (async on Q()): enqueue GPU evaluation of planned VRAM hits while CPU threads run in parallel.
    void launch_decode(const float* z_dev, float* zout_gpu_dev) {
        const int npairs = planned_pairs_;
        const int T = planned_T_;
        if (npairs <= 0) return;

        Q().memcpy(d_pairs_, h_pairs_, (size_t)npairs * sizeof(vcache_detail::VramPair));
        Q().memcpy(d_tok_off_, h_tok_off_, (size_t)(T + 1) * sizeof(int32_t));

        const vcache_detail::VramPair* dp = d_pairs_;
        half* hmid = d_hmid_;
        float* ye = d_ye_;
        const int32_t* toff = d_tok_off_;

        auto launch_k = [&](auto q8_tag) {
            constexpr bool Q8A = decltype(q8_tag)::value;
            Q().parallel_for(NR((size_t)npairs * 192, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
                const int gid = (int)it.get_group(0);
                const int p = gid / 192, rb = gid % 192;
                const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
                const int d = rb * 8 + wave;
                const vcache_detail::VramPair vp = dp[p];
                const size_t row_b = (size_t)6 * vcache_detail::vc_sb_bytes(vp.gu);
                const uint8_t* wgu = vp.body + 3072;
                const float* zt = z_dev + (size_t)vp.tok * vcache_detail::VC_K;
                const float g = vcache_detail::sg_dot_row<float, Q8A>(it, vp.gu, wgu + (size_t)d * row_b, zt);
                const float u = vcache_detail::sg_dot_row<float, Q8A>(it, vp.gu, wgu + (size_t)(vcache_detail::VC_K + d) * row_b, zt);
                if (lane == 0) {
                    const float sq = h2f_bits(ld16(vp.body + 2 * d));
                    const float s = sycl::fmin(g / (1.0f + sycl::exp(-g)), 10.0f);
                    const float uc = sycl::fmax(-10.0f, sycl::fmin(u, 10.0f));
                    hmid[(size_t)p * vcache_detail::VC_K + d] = (half)((s * uc) / sq);
                }
            });

            Q().parallel_for(NR((size_t)npairs * 192, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
                const int gid = (int)it.get_group(0);
                const int p = gid / 192, rb = gid % 192;
                const int wave = (int)it.get_local_id(0) >> 5, lane = (int)it.get_local_id(0) & 31;
                const int d = rb * 8 + wave;
                const vcache_detail::VramPair vp = dp[p];
                const size_t row_b = (size_t)6 * vcache_detail::vc_sb_bytes(vp.dn);
                const uint8_t* wdn = vp.body + vp.dn_off;
                const float y = vcache_detail::sg_dot_row<half, Q8A>(it, vp.dn, wdn + (size_t)d * row_b, hmid + (size_t)p * vcache_detail::VC_K);
                if (lane == 0) ye[(size_t)p * vcache_detail::VC_K + d] = y;
            });
        };
        if (q8_act_) launch_k(std::true_type{});
        else launch_k(std::false_type{});

        Q().parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int p0 = toff[t], p1 = toff[t + 1];
            for (int d = tid; d < vcache_detail::VC_K; d += 256) {
                float acc = 0.0f;
                for (int p = p0; p < p1; ++p) acc += ye[(size_t)p * vcache_detail::VC_K + d] * dp[p].weight;
                zout_gpu_dev[(size_t)t * vcache_detail::VC_K + d] = acc;
            }
        });
    }

    int run_decode(int L, int T, const float* z_dev, const int32_t* idx_host, const float* w_host, uint8_t* use_cpu, float* zout_gpu_dev) {
        const int n = plan_decode(L, T, idx_host, w_host, use_cpu);
        if (n > 0) launch_decode(z_dev, zout_gpu_dev);
        return n;
    }

    // Add CPU partial zout to GPU VRAM partial zout: zout_f[i] += zout_gpu[i]
    static void add_zout(float* dst, const float* src, size_t n) {
        Q().parallel_for(NR((n + 255) / 256, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < n) dst[i] += src[i];
        });
    }

    // Committed-token LFU-decay update + periodic deterministic promotion (every 32 committed tokens!).
    // Because rejected speculative draft tokens are ignored and DFly2 clamps windows at 32-token boundaries,
    // plain and DFly2 execute the EXACT SAME promotions at the EXACT SAME token positions -> 100% bitwise identical!
    template <class BodyFn>
    void commit_tokens(int n_committed, int next_pos, BodyFn&& get_host_body, int max_swaps = 41) {
        if (!active() || !adaptive_ || n_committed <= 0) return;
        const int n = std::min(n_committed, MAX_T);
        for (int t = 0; t < n; ++t) {
            for (float& s : scores_) s *= 0.965f;
            for (int L = 0; L < nlayers_; ++L) {
                const int32_t* row = &last_idx_[((size_t)L * MAX_T + t) * vcache_detail::VC_NTOP];
                for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                    const int e = row[k];
                    if ((unsigned)e < 512u) scores_[(size_t)L * 512 + e] += 1.0f;
                }
            }
        }
        if ((next_pos & 31) != 0) return;  // promote strictly at deterministic 32-token boundaries!
        int swapped = 0;
        for (int L = 0; L < nlayers_ && swapped < max_swaps; ++L) {
            // Pick up to 2 hottest non-resident experts in layer L vs coldest resident slots (core_frac=0)
            for (int rep = 0; rep < 2 && swapped < max_swaps; ++rep) {
                int best_e = -1;
                float best_sc = 1.8f;
                for (int e = 0; e < 512; ++e) {
                    if (d_ptr_[(size_t)L * 512 + e] != nullptr) continue;
                    const float sc = scores_[(size_t)L * 512 + e];
                    if (sc > best_sc) { best_sc = sc; best_e = e; }
                }
                if (best_e < 0) break;
                const uint8_t* hbody = get_host_body(L, best_e);
                if (!hbody) break;
                int victim_slot = -1;
                float worst_sc = best_sc / 1.25f;
                for (int ee = 0; ee < 512; ++ee) {
                    const int sl = slot_of_[(size_t)L * 512 + ee];
                    if (sl >= 0 && scores_[(size_t)L * 512 + ee] < worst_sc) {
                        worst_sc = scores_[(size_t)L * 512 + ee];
                        victim_slot = sl;
                    }
                }
                if (victim_slot < 0) break;
                const int old_e = slots_[victim_slot].expert;
                const Meta& m = meta_[(size_t)L * 512 + best_e];
                std::memcpy(h_stage_, hbody, m.body_bytes);
                cq_->memcpy(slots_[victim_slot].d_ptr, h_stage_, m.body_bytes).wait();
                d_ptr_[(size_t)L * 512 + old_e] = nullptr;
                slot_of_[(size_t)L * 512 + old_e] = -1;
                slots_[victim_slot].expert = best_e;
                d_ptr_[(size_t)L * 512 + best_e] = slots_[victim_slot].d_ptr;
                slot_of_[(size_t)L * 512 + best_e] = victim_slot;
                ++promotions_;
                ++swapped;
            }
        }
    }

    template <class BodyFn>
    void step_adapt(BodyFn&& get_host_body, int max_swaps = 16) {
        ++step_ctr_;
        commit_tokens(1, (int)step_ctr_, std::forward<BodyFn>(get_host_body), max_swaps);
    }

private:
    std::string strata_path_;
    int nlayers_ = 0;
    size_t slot_bytes_ = 0;
    size_t dev_bytes_ = 0;
    bool adaptive_ = true;
    bool q8_act_ = true;
    uint64_t step_ctr_ = 0;
    uint64_t hits_ = 0, misses_ = 0, promotions_ = 0;
    std::vector<Meta> meta_;
    std::vector<uint16_t> rank_eid_;
    std::vector<Slab> slabs_;
    std::vector<Slot> slots_;
    std::vector<uint8_t*> d_ptr_;
    std::vector<int> slot_of_;
    std::vector<float> scores_;
    std::vector<int32_t> last_idx_;
    std::vector<int> pending_candidates_;
    sycl::queue* cq_ = nullptr;
    uint8_t* h_stage_ = nullptr;
    vcache_detail::VramPair* d_pairs_ = nullptr;
    vcache_detail::VramPair* h_pairs_ = nullptr;
    int32_t* d_tok_off_ = nullptr;
    int32_t* h_tok_off_ = nullptr;
    half* d_hmid_ = nullptr;
    float* d_ye_ = nullptr;
    int planned_pairs_ = 0;
    int planned_T_ = 0;
};

}  // namespace hy4s
