// sy_peer.h - Multi-GPU Peer Expert Evaluator for Hy4-Flash Intel Arc SYCL runner (--gpus 0,1 / --gpus 2 / --devices / HY4_GPUS).
// MIT License - Copyright (c) 2026 Strata contributors
//
// Architecture:
//   * Primary GPU (ids[0]) owns trunk, MLA attention, KV cache, Engram, Vision, DFly2, and hot VRAM experts (SyVramExpertCache).
//   * Secondary GPU(s) (ids[1..]) own the next hottest routed experts in Section 3 rank order across multi-slab (<= 1.4 GiB/slab)
//     USM device allocations, each with its own in-order sycl::queue.
//   * Transfers use direct P2P USM copy when ext_oneapi_can_access_peer + 16-byte roundtrip hardware probe succeed, and cleanly
//     fall back to USM host-pinned bounce buffers otherwise.
//   * Decode/Verify (T <= 16): runs in parallel with GPU 0 SyVramExpertCache AND hy4::CpuExperts, with identical SG32 Q8/FP32
//     math and ascending expert_id reduction order (bitwise reproducible, compatible with command_graph & DFly2).
//   * Prefill (T > 16): runs in parallel with GPU 0 GpuExperts (which skips peer experts via `only` mask) and accumulates partial
//     sums into zout_f on GPU 0.
#pragma once

#include <algorithm>
#include <chrono>
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
#include "sy_vram_cache.h"

namespace hy4s {

class SyPeerExperts {
public:
    static constexpr int MAX_DEC_T = SyVramExpertCache::MAX_T;
    static constexpr int MAX_DEC_PAIRS = SyVramExpertCache::MAX_PAIRS;
    static constexpr int MAX_PF_TILE_T = 256;
    static constexpr int MAX_PF_PAIRS = MAX_PF_TILE_T * vcache_detail::VC_NTOP;

    struct Meta {
        uint64_t file_off = 0;
        uint32_t dn_off = 0;
        uint32_t body_bytes = 0;
        uint8_t gu = 0, dn = 0;
    };

    struct Slab {
        uint8_t* d = nullptr;
        size_t bytes = 0;
    };

    struct PeerDev {
        int dev_idx = -1;
        std::string name;
        size_t global_mem = 0;
        size_t dev_bytes = 0;
        int n_experts = 0;
        bool p2p_ok = false;

        sycl::queue* q = nullptr;
        std::vector<Slab> slabs;

        // Decode scratch (on peer device + pinned host bounce)
        float* d_z = nullptr;           // [MAX_PF_TILE_T * 1536] (shared between decode & prefill!)
        float* d_zout = nullptr;        // [MAX_PF_TILE_T * 1536]
        float* h_zout = nullptr;        // pinned host [MAX_PF_TILE_T * 1536]
        float* d_zpeer0 = nullptr;      // on primary GPU 0 [MAX_PF_TILE_T * 1536]
        vcache_detail::VramPair* d_pairs = nullptr;  // [MAX_PF_PAIRS]
        vcache_detail::VramPair* h_pairs = nullptr;  // [MAX_PF_PAIRS]
        int32_t* d_tok_off = nullptr;   // [MAX_PF_TILE_T + 1]
        int32_t* h_tok_off = nullptr;   // [MAX_PF_TILE_T + 1]
        half* d_hmid = nullptr;         // [MAX_PF_PAIRS * 1536]
        float* d_ye = nullptr;          // [MAX_PF_PAIRS * 1536]

        int planned_pairs = 0;
        int planned_T = 0;
        bool active_launch = false;
        sycl::event done_ev;
    };

    ~SyPeerExperts() { release(); }

    bool active() const { return !peers_.empty() && total_experts_ > 0; }
    int num_peers() const { return (int)peers_.size(); }
    int total_experts() const { return total_experts_; }
    size_t total_device_bytes() const { return total_dev_bytes_; }
    uint64_t hits() const { return hits_; }
    void reset_stats() { hits_ = 0; }
    void set_q8_act(bool on) { q8_act_ = on; }
    bool has_expert(int L, int e) const {
        if ((unsigned)L >= (unsigned)nlayers_ || (unsigned)e >= 512u) return false;
        return owner_[(size_t)L * 512 + e] >= 0;
    }

    std::string describe() const {
        if (peers_.empty()) return "";
        std::string s;
        for (size_t i = 0; i < peers_.size(); ++i) {
            if (!s.empty()) s += ", ";
            char buf[256];
            snprintf(buf, sizeof buf, "GPU%d(%s, %d exp, %.2f GiB, %s)",
                     peers_[i].dev_idx, peers_[i].name.c_str(), peers_[i].n_experts,
                     peers_[i].dev_bytes / 1073741824.0, peers_[i].p2p_ok ? "p2p" : "pinned");
            s += buf;
        }
        return s;
    }

    void release() {
        for (auto& p : peers_) {
            if (p.q) {
                try { p.q->wait(); } catch (...) {}
                for (auto& sl : p.slabs) if (sl.d) sycl::free(sl.d, *p.q);
                if (p.d_z) sycl::free(p.d_z, *p.q);
                if (p.d_zout) sycl::free(p.d_zout, *p.q);
                if (p.h_zout) sycl::free(p.h_zout, *p.q);
                if (p.d_pairs) sycl::free(p.d_pairs, *p.q);
                if (p.h_pairs) sycl::free(p.h_pairs, *p.q);
                if (p.d_tok_off) sycl::free(p.d_tok_off, *p.q);
                if (p.h_tok_off) sycl::free(p.h_tok_off, *p.q);
                if (p.d_hmid) sycl::free(p.d_hmid, *p.q);
                if (p.d_ye) sycl::free(p.d_ye, *p.q);
                delete p.q;
                p.q = nullptr;
            }
            if (p.d_zpeer0) { dfree(p.d_zpeer0); p.d_zpeer0 = nullptr; }
            p.slabs.clear();
        }
        peers_.clear();
        if (h_pf_z_) { sycl::free(h_pf_z_, Q()); h_pf_z_ = nullptr; }
        owner_.clear();
        d_ptr_.clear();
        meta_.clear();
        total_experts_ = 0;
        total_dev_bytes_ = 0;
        nlayers_ = 0;
    }

    // Probe optional P2P support between primary device and peer device with a 16-byte hardware roundtrip check.
    static bool probe_p2p(sycl::queue& q0, sycl::queue& qp) {
        const char* force_pinned = std::getenv("HY4S_PEER_PINNED");
        if (force_pinned && force_pinned[0] == '1') return false;
#if defined(SYCL_EXT_ONEAPI_PEER_ACCESS)
        try {
            auto d0 = q0.get_device();
            auto dp = qp.get_device();
            if (!d0.ext_oneapi_can_access_peer(dp, sycl::ext::oneapi::peer_access::access_supported) ||
                !dp.ext_oneapi_can_access_peer(d0, sycl::ext::oneapi::peer_access::access_supported)) {
                return false;
            }
            d0.ext_oneapi_enable_peer_access(dp);
            dp.ext_oneapi_enable_peer_access(d0);
            uint32_t h_tx[4] = {0x12345678u, 0xCAFEBABEu, 0xA5A55A5Au, 0xDEADBEEFu};
            uint32_t h_rx01[4] = {0, 0, 0, 0};
            uint32_t h_rx10[4] = {0, 0, 0, 0};
            uint32_t* p0 = sycl::malloc_device<uint32_t>(4, q0);
            uint32_t* p1 = sycl::malloc_device<uint32_t>(4, qp);
            bool ok = false;
            if (p0 && p1) {
                q0.memcpy(p0, h_tx, sizeof(h_tx)).wait();
                qp.memset(p1, 0, sizeof(h_tx)).wait();
                qp.memcpy(p1, p0, sizeof(h_tx)).wait();
                qp.memcpy(h_rx01, p1, sizeof(h_tx)).wait();
                q0.memset(p0, 0, sizeof(h_tx)).wait();
                q0.memcpy(p0, p1, sizeof(h_tx)).wait();
                q0.memcpy(h_rx10, p0, sizeof(h_tx)).wait();
                ok = (std::memcmp(h_tx, h_rx01, sizeof(h_tx)) == 0 &&
                      std::memcmp(h_tx, h_rx10, sizeof(h_tx)) == 0);
            }
            if (p0) sycl::free(p0, q0);
            if (p1) sycl::free(p1, qp);
            return ok;
        } catch (...) {
            return false;
        }
#else
        (void)q0; (void)qp;
        return false;
#endif
    }

    // Initialize peer GPU expert storage for `peer_devs` (indices into enum_gpu_devices()), skipping experts already in `vram0_mask`.
    // `per_peer_budget_bytes`: 0 => auto (90% of peer global_mem minus workspace).
    bool init(const std::string& strata_path, const std::vector<int>& peer_devs,
              const std::vector<std::vector<bool>>& vram0_mask,
              size_t per_peer_budget_bytes, std::string& err, size_t slab_mb = 1400) {
        release();
        if (peer_devs.empty()) return true;

        auto all_devs = enum_gpu_devices();
        if (all_devs.size() < 2) {
            fprintf(stderr, "[peer-sycl] only %zu GPU(s) visible; multi-GPU peer experts disabled\n", all_devs.size());
            return true;
        }

        strata::artifact::StrataFileReader rd;
        if (!rd.open(strata_path, err)) return false;
        const auto& h = rd.header();
        const auto& dir = rd.expert_directory();
        if (dir.empty() || (dir.size() % 512) != 0) {
            err = "peer-sycl: invalid expert directory";
            return false;
        }
        nlayers_ = (int)(dir.size() / 512);
        meta_.assign(dir.size(), Meta{});
        owner_.assign(dir.size(), -1);
        d_ptr_.assign(dir.size(), nullptr);
        std::vector<uint16_t> rank_eid(dir.size(), 0);
        uint32_t max_body = 0;
        for (size_t i = 0; i < dir.size(); ++i) {
            const int L = (int)(i / 512);
            const auto& d = dir[i];
            Meta& m = meta_[(size_t)L * 512 + d.expert_id];
            m.file_off = h.sections[4].file_offset + d.body_offset;
            m.dn_off = d.dn_offset;
            m.body_bytes = d.body_bytes;
            m.gu = d.gu_code_id;
            m.dn = d.dn_code_id;
            rank_eid[i] = d.expert_id;
            max_body = std::max(max_body, d.body_bytes);
        }
        const size_t slot_bytes = ((size_t)max_body + 4095ull) & ~4095ull;
        if (slot_bytes == 0) return true;

        // Scratch memory per peer (~42 MiB for up to MAX_PF_TILE_T=256 prefill tile + decode)
        const size_t ws_bytes =
            (size_t)MAX_PF_TILE_T * vcache_detail::VC_K * sizeof(float) * 2 +
            (size_t)MAX_PF_PAIRS * (sizeof(vcache_detail::VramPair) + vcache_detail::VC_K * (sizeof(half) + sizeof(float))) +
            (size_t)(MAX_PF_TILE_T + 1) * sizeof(int32_t) + (64ull << 20);

        // Create queues & query budgets for each valid peer device
        std::vector<size_t> peer_max_slots;
        for (int pidx : peer_devs) {
            if (pidx < 0 || pidx >= (int)all_devs.size() || pidx == primary_gpu_index()) continue;
            PeerDev pd;
            pd.dev_idx = pidx;
            auto dev = all_devs[pidx];
            pd.name = dev.get_info<sycl::info::device::name>();
            pd.global_mem = dev.get_info<sycl::info::device::global_mem_size>();
            try {
                pd.q = new sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order()});
            } catch (const sycl::exception& e) {
                fprintf(stderr, "[peer-sycl] skipping GPU%d (%s): queue creation failed (%s)\n", pidx, pd.name.c_str(), e.what());
                continue;
            }
            size_t budget = per_peer_budget_bytes;
            if (budget == 0) {
                const size_t safe_cap = (size_t)((double)pd.global_mem * 0.90);
                budget = (safe_cap > ws_bytes + (256ull << 20)) ? (safe_cap - ws_bytes - (256ull << 20)) : 0;
            }
            const size_t nslots = budget / slot_bytes;
            if (nslots == 0) {
                delete pd.q;
                continue;
            }
            pd.p2p_ok = probe_p2p(Q(), *pd.q);
            peers_.push_back(std::move(pd));
            peer_max_slots.push_back(nslots);
        }
        if (peers_.empty()) return true;

        // Partition non-GPU0 experts in Section 3 rank-major order across peers (round-robin per rank across available peers
        // so every layer gets balanced multi-GPU parallelism across all peers!)
        std::vector<std::vector<std::pair<int, int>>> assign(peers_.size());
        size_t rr = 0;
        for (int rk = 0; rk < 512; ++rk) {
            for (int L = 0; L < nlayers_; ++L) {
                const int e = rank_eid[(size_t)L * 512 + rk];
                if (!vram0_mask.empty() && L < (int)vram0_mask.size() && e < (int)vram0_mask[L].size() && vram0_mask[L][e]) {
                    continue;  // already resident on GPU 0
                }
                int chosen = -1;
                for (size_t k = 0; k < peers_.size(); ++k) {
                    const size_t pi = (rr + k) % peers_.size();
                    if (assign[pi].size() < peer_max_slots[pi]) {
                        chosen = (int)pi;
                        rr = (pi + 1) % peers_.size();
                        break;
                    }
                }
                if (chosen < 0) break;  // all peers full
                assign[chosen].push_back({L, e});
            }
        }

        hy4::AlignedFile af;
        if (!af.open(strata_path, false)) {
            err = "peer-sycl: cannot open " + strata_path;
            return false;
        }
        const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const size_t slab_max = (slab_mb << 20);
        const size_t slots_per_slab_max = std::max<size_t>(1, slab_max / slot_bytes);

        for (size_t pi = 0; pi < peers_.size(); ++pi) {
            PeerDev& pd = peers_[pi];
            const auto& list = assign[pi];
            if (list.empty()) continue;

            // Allocate peer scratch & primary GPU 0 receive buffer
            pd.d_z = sycl::malloc_device<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K, *pd.q);
            pd.d_zout = sycl::malloc_device<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K, *pd.q);
            pd.h_zout = sycl::malloc_host<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K, *pd.q);
            pd.d_pairs = sycl::malloc_device<vcache_detail::VramPair>(MAX_PF_PAIRS, *pd.q);
            pd.h_pairs = sycl::malloc_host<vcache_detail::VramPair>(MAX_PF_PAIRS, *pd.q);
            pd.d_tok_off = sycl::malloc_device<int32_t>(MAX_PF_TILE_T + 1, *pd.q);
            pd.h_tok_off = sycl::malloc_host<int32_t>(MAX_PF_TILE_T + 1, *pd.q);
            pd.d_hmid = sycl::malloc_device<half>((size_t)MAX_PF_PAIRS * vcache_detail::VC_K, *pd.q);
            pd.d_ye = sycl::malloc_device<float>((size_t)MAX_PF_PAIRS * vcache_detail::VC_K, *pd.q);
            pd.d_zpeer0 = dalloc<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K);
            if (!pd.d_z || !pd.d_zout || !pd.h_zout || !pd.d_pairs || !pd.h_pairs ||
                !pd.d_tok_off || !pd.h_tok_off || !pd.d_hmid || !pd.d_ye) {
                af.close();
                err = "peer-sycl: scratch allocation failed on GPU " + std::to_string(pd.dev_idx);
                return false;
            }
            uint8_t* h_stage = sycl::aligned_alloc_host<uint8_t>(4096, slot_bytes, *pd.q);
            if (!h_stage) {
                af.close();
                err = "peer-sycl: staging allocation failed";
                return false;
            }

            size_t idx = 0;
            while (idx < list.size()) {
                const size_t ns = std::min(list.size() - idx, slots_per_slab_max);
                const size_t sb = ns * slot_bytes;
                uint8_t* slab_ptr = sycl::malloc_device<uint8_t>(sb, *pd.q);
                if (!slab_ptr) break;
                pd.slabs.push_back(Slab{slab_ptr, sb});
                pd.dev_bytes += sb;
                total_dev_bytes_ += sb;

                for (size_t k = 0; k < ns; ++k) {
                    const int L = list[idx + k].first;
                    const int e = list[idx + k].second;
                    const Meta& m = meta_[(size_t)L * 512 + e];
                    if (af.read_at(m.file_off, h_stage, m.body_bytes) < m.body_bytes) {
                        sycl::free(h_stage, *pd.q);
                        af.close();
                        err = "peer-sycl: short read";
                        return false;
                    }
                    uint8_t* dst = slab_ptr + k * slot_bytes;
                    pd.q->memcpy(dst, h_stage, m.body_bytes).wait();
                    owner_[(size_t)L * 512 + e] = (int)pi;
                    d_ptr_[(size_t)L * 512 + e] = dst;
                    ++pd.n_experts;
                    ++total_experts_;
                }
                idx += ns;
            }
            sycl::free(h_stage, *pd.q);
        }
        af.close();

        if (total_experts_ > 0) {
            h_pf_z_ = halloc<float>((size_t)MAX_PF_TILE_T * vcache_detail::VC_K);
        }

        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
        fprintf(stderr, "[peer-sycl] loaded %d peer experts across %zu peer GPU(s) (%.2f GiB total) in %.1f s: %s\n",
                total_experts_, peers_.size(), total_dev_bytes_ / 1073741824.0, dt, describe().c_str());
        return true;
    }

    void mark_peer_mask(std::vector<std::vector<bool>>& mask) const {
        if (!active()) return;
        if ((int)mask.size() < nlayers_) mask.assign(nlayers_, std::vector<bool>(512, false));
        for (int L = 0; L < nlayers_; ++L)
            for (int e = 0; e < 512; ++e)
                if (owner_[(size_t)L * 512 + e] >= 0) mask[L][e] = true;
    }

    void build_non_peer_mask(int L, uint8_t only[512]) const {
        for (int e = 0; e < 512; ++e) {
            only[e] = (owner_[(size_t)L * 512 + e] >= 0) ? 0 : 1;
        }
    }

    // ---- Decode/Verify path (T <= 16): claim peer hits from `use_cpu`, launch peers concurrently, gather partial sums ----
    int plan_decode(int L, int T, const int32_t* idx_host, const float* w_host, uint8_t* use_cpu) {
        for (auto& pd : peers_) {
            pd.planned_pairs = 0;
            pd.planned_T = T;
            pd.active_launch = false;
        }
        if (!active() || T <= 0 || T > MAX_DEC_T) return 0;

        int total_hits = 0;
        for (size_t pi = 0; pi < peers_.size(); ++pi) {
            PeerDev& pd = peers_[pi];
            int npairs = 0;
            pd.h_tok_off[0] = 0;
            for (int t = 0; t < T; ++t) {
                int hit_k[vcache_detail::VC_NTOP], nh = 0;
                for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                    const int slot = t * vcache_detail::VC_NTOP + k;
                    if (use_cpu && !use_cpu[slot]) continue;  // already claimed by GPU 0
                    const int e = idx_host[slot];
                    if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)pi) {
                        if (use_cpu) use_cpu[slot] = 0;
                        ++hits_;
                        ++total_hits;
                        int j = nh++;
                        while (j > 0 && idx_host[t * vcache_detail::VC_NTOP + hit_k[j - 1]] > e) {
                            hit_k[j] = hit_k[j - 1];
                            --j;
                        }
                        hit_k[j] = k;
                    }
                }
                for (int a = 0; a < nh; ++a) {
                    const int k = hit_k[a];
                    const int e = idx_host[t * vcache_detail::VC_NTOP + k];
                    const Meta& m = meta_[(size_t)L * 512 + e];
                    vcache_detail::VramPair& vp = pd.h_pairs[npairs++];
                    vp.body = d_ptr_[(size_t)L * 512 + e];
                    vp.dn_off = m.dn_off;
                    vp.tok = t;
                    vp.gu = m.gu;
                    vp.dn = m.dn;
                    vp.weight = w_host[t * vcache_detail::VC_NTOP + k];
                }
                pd.h_tok_off[t + 1] = npairs;
            }
            pd.planned_pairs = npairs;
        }
        return total_hits;
    }

    void launch_decode(const float* z_dev_g0, const float* z_host_pinned) {
        if (!active()) return;
        for (auto& pd : peers_) {
            if (pd.planned_pairs <= 0) continue;
            const int T = pd.planned_T;
            const size_t nz = (size_t)T * vcache_detail::VC_K;
            sycl::queue& pq = *pd.q;
            if (pd.p2p_ok && z_dev_g0) {
                pq.memcpy(pd.d_z, z_dev_g0, nz * sizeof(float));
            } else {
                pq.memcpy(pd.d_z, z_host_pinned, nz * sizeof(float));
            }
            enqueue_peer_eval(pd, T, pd.planned_pairs);
            if (pd.p2p_ok) {
                pd.done_ev = pq.memcpy(pd.d_zpeer0, pd.d_zout, nz * sizeof(float));
            } else {
                pd.done_ev = pq.memcpy(pd.h_zout, pd.d_zout, nz * sizeof(float));
            }
            pd.active_launch = true;
        }
    }

    // Wait for all launched peer GPUs and accumulate partial sums into zout_f_g0 on primary GPU 0.
    void finish_decode(int T, float* zout_f_g0, bool& zout_initialized) {
        if (!active()) return;
        const size_t nz = (size_t)T * vcache_detail::VC_K;
        for (auto& pd : peers_) {
            if (!pd.active_launch) continue;
            pd.done_ev.wait();
            pd.active_launch = false;
            if (pd.p2p_ok) {
                if (!zout_initialized) {
                    d2d(zout_f_g0, pd.d_zpeer0, nz * sizeof(float));
                    zout_initialized = true;
                } else {
                    SyVramExpertCache::add_zout(zout_f_g0, pd.d_zpeer0, nz);
                }
            } else {
                if (!zout_initialized) {
                    h2d(zout_f_g0, pd.h_zout, nz * sizeof(float));
                    zout_initialized = true;
                } else {
                    h2d(pd.d_zpeer0, pd.h_zout, nz * sizeof(float));
                    SyVramExpertCache::add_zout(zout_f_g0, pd.d_zpeer0, nz);
                }
            }
        }
    }

    // ---- Prefill path (T > 16): evaluate all peer-resident experts across T tokens (tiled by MAX_PF_TILE_T) ----
    // Note: called alongside GpuExperts::run(..., only_non_peer); accumulates partial sums into zout_f_g0 after gx->run!
    void run_prefill_tile_and_add(int L, int T, const float* z_dev_g0, const int32_t* idx_host, const float* w_host, float* zout_f_g0) {
        if (!active() || T <= 0) return;
        bool any_p2p = true;
        for (const auto& pd : peers_) if (!pd.p2p_ok) any_p2p = false;

        for (int t0 = 0; t0 < T; t0 += MAX_PF_TILE_T) {
            const int tn = std::min(MAX_PF_TILE_T, T - t0);
            const size_t nz = (size_t)tn * vcache_detail::VC_K;
            const float* z_tile_g0 = z_dev_g0 + (size_t)t0 * vcache_detail::VC_K;
            float* zout_tile_g0 = zout_f_g0 + (size_t)t0 * vcache_detail::VC_K;

            int tile_hits = 0;
            for (size_t pi = 0; pi < peers_.size(); ++pi) {
                PeerDev& pd = peers_[pi];
                int npairs = 0;
                pd.h_tok_off[0] = 0;
                for (int t = 0; t < tn; ++t) {
                    const int gt = t0 + t;
                    int hit_k[vcache_detail::VC_NTOP], nh = 0;
                    for (int k = 0; k < vcache_detail::VC_NTOP; ++k) {
                        const int e = idx_host[gt * vcache_detail::VC_NTOP + k];
                        if ((unsigned)e < 512u && owner_[(size_t)L * 512 + e] == (int)pi) {
                            int j = nh++;
                            while (j > 0 && idx_host[gt * vcache_detail::VC_NTOP + hit_k[j - 1]] > e) {
                                hit_k[j] = hit_k[j - 1];
                                --j;
                            }
                            hit_k[j] = k;
                        }
                    }
                    for (int a = 0; a < nh; ++a) {
                        const int k = hit_k[a];
                        const int e = idx_host[gt * vcache_detail::VC_NTOP + k];
                        const Meta& m = meta_[(size_t)L * 512 + e];
                        vcache_detail::VramPair& vp = pd.h_pairs[npairs++];
                        vp.body = d_ptr_[(size_t)L * 512 + e];
                        vp.dn_off = m.dn_off;
                        vp.tok = t;
                        vp.gu = m.gu;
                        vp.dn = m.dn;
                        vp.weight = w_host[gt * vcache_detail::VC_NTOP + k];
                    }
                    pd.h_tok_off[t + 1] = npairs;
                }
                pd.planned_pairs = npairs;
                pd.planned_T = tn;
                pd.active_launch = false;
                tile_hits += npairs;
            }
            if (tile_hits == 0) continue;

            if (!any_p2p) {
                Q().memcpy(h_pf_z_, z_tile_g0, nz * sizeof(float)).wait();
            } else {
                Q().wait();
            }

            // Launch all peers in parallel for this tile
            for (auto& pd : peers_) {
                if (pd.planned_pairs <= 0) continue;
                sycl::queue& pq = *pd.q;
                if (pd.p2p_ok) {
                    pq.memcpy(pd.d_z, z_tile_g0, nz * sizeof(float));
                } else {
                    pq.memcpy(pd.d_z, h_pf_z_, nz * sizeof(float));
                }
                enqueue_peer_eval(pd, tn, pd.planned_pairs);
                if (pd.p2p_ok) {
                    pd.done_ev = pq.memcpy(pd.d_zpeer0, pd.d_zout, nz * sizeof(float));
                } else {
                    pd.done_ev = pq.memcpy(pd.h_zout, pd.d_zout, nz * sizeof(float));
                }
                pd.active_launch = true;
            }
            // Wait & accumulate into zout_tile_g0
            bool inited = true;
            finish_decode(tn, zout_tile_g0, inited);
        }
    }

private:
    void enqueue_peer_eval(PeerDev& pd, int T, int npairs) {
        sycl::queue& pq = *pd.q;
        pq.memcpy(pd.d_pairs, pd.h_pairs, (size_t)npairs * sizeof(vcache_detail::VramPair));
        pq.memcpy(pd.d_tok_off, pd.h_tok_off, (size_t)(T + 1) * sizeof(int32_t));

        const vcache_detail::VramPair* dp = pd.d_pairs;
        const float* z_dev = pd.d_z;
        half* hmid = pd.d_hmid;
        float* ye = pd.d_ye;
        float* zout_dev = pd.d_zout;
        const int32_t* toff = pd.d_tok_off;

        auto launch_k = [&](auto q8_tag) {
            constexpr bool Q8A = decltype(q8_tag)::value;
            pq.parallel_for(NR((size_t)npairs * 192, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
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

            pq.parallel_for(NR((size_t)npairs * 192, 256), [=](sycl::nd_item<1> it) REQD_SG32 {
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

        pq.parallel_for(NR(T, 256), [=](sycl::nd_item<1> it) {
            const int t = (int)it.get_group(0), tid = (int)it.get_local_id(0);
            const int p0 = toff[t], p1 = toff[t + 1];
            for (int d = tid; d < vcache_detail::VC_K; d += 256) {
                float acc = 0.0f;
                for (int p = p0; p < p1; ++p) acc += ye[(size_t)p * vcache_detail::VC_K + d] * dp[p].weight;
                zout_dev[(size_t)t * vcache_detail::VC_K + d] = acc;
            }
        });
    }

    int nlayers_ = 0;
    bool q8_act_ = true;
    int total_experts_ = 0;
    size_t total_dev_bytes_ = 0;
    uint64_t hits_ = 0;
    std::vector<Meta> meta_;
    std::vector<int> owner_;      // [nlayers * 512]: -1 or peer index in peers_
    std::vector<uint8_t*> d_ptr_; // [nlayers * 512]: device ptr on owning peer
    std::vector<PeerDev> peers_;
    float* h_pf_z_ = nullptr;     // pinned host bounce for non-P2P prefill tiles
};

}  // namespace hy4s
