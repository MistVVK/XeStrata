// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
#include "strata/core/peer_experts.hpp"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <cctype>
#include <climits>
#include <unistd.h>
#endif

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::kWindowMaxT * 10;   // entries per layer: a window's rows x top-10

struct Meta {                           // one block, uploaded with one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// #2666: `cudaDeviceCanAccessPeer` said yes and `cudaDeviceEnablePeerAccess` succeeded, and on a pair of GPUs behind a host bridge
/// (2x RTX A4000, "PHB" in nvidia-smi topo) `cudaMemcpyPeerAsync` then returned success having moved nothing: the peer's input rows
/// were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN.  Without
/// peer access enabled the same copies are right (the driver stages them).  So the pair is asked once, with a small copy each way,
/// whether the direct path really carries data.  true = both ways exact (or nothing could be tested: the old trust is kept).
/// Returns false when two CUDA GPUs sit behind separate PCIe root ports (PHB / SYS topology) without
/// NVLink / native P2P atomics. On such boards (e.g. Intel Xeon PHB + ACS, 2x RTX A4000) merely calling
/// cudaDeviceEnablePeerAccess + cudaMemcpyPeer can fault GPU 1's CUDA context (unspecified launch failure)
/// or stall at ~98 ms per copy. Skipping P2P upfront avoids poisoning the device and uses fast portable host staging.
bool p2p_topology_safe(int dev0, int dev1) {
    if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && np[0] != '\0' && std::atoi(np) == 0) return true;
#if !defined(STRATA_USE_HIP)
    int atomic_ok = 0;
    if (cudaDeviceGetP2PAttribute(&atomic_ok, cudaDevP2PAttrNativeAtomicSupported, dev0, dev1) == cudaSuccess && atomic_ok) {
        return true;   // NVLink or true hardware P2P switch with native atomics
    }
    (void) cudaGetLastError();
#if defined(__linux__)
    auto ancestors = [](int d) -> std::vector<std::string> {
        char bdf[64] = {};
        if (cudaDeviceGetPCIBusId(bdf, sizeof(bdf), d) != cudaSuccess) { (void) cudaGetLastError(); return {}; }
        for (char* p = bdf; *p; ++p) *p = (char) std::tolower((unsigned char) *p);
        std::string sys = std::string("/sys/bus/pci/devices/") + bdf;
        char resolved[PATH_MAX] = {};
        if (!realpath(sys.c_str(), resolved)) return {};
        std::vector<std::string> parts;
        std::string s(resolved);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t nxt = s.find('/', pos);
            if (nxt == std::string::npos) nxt = s.size();
            if (nxt > pos) parts.push_back(s.substr(pos, nxt - pos));
            pos = nxt + 1;
        }
        if (!parts.empty()) parts.pop_back();   // drop the GPU's own BDF
        return parts;
    };
    const auto a0 = ancestors(dev0), a1 = ancestors(dev1);
    if (!a0.empty() && !a1.empty()) {
        size_t common_bridges = 0;
        const size_t n = std::min(a0.size(), a1.size());
        for (size_t i = 0; i < n && a0[i] == a1[i]; ++i) {
            // PCI bridge components look like "0000:00:02.0" (contains ':' and '.'), whereas root complex is "pci0000:00"
            if (a0[i].find(':') != std::string::npos && a0[i].find('.') != std::string::npos) ++common_bridges;
        }
        if (common_bridges == 0) return false;   // PHB / NODE / SYS (no shared PCIe switch)
    }
#endif
#endif
    return true;
}

bool p2p_copies_data(int device) {
    constexpr size_t kN = 1 << 14;   // floats
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    cudaStream_t st[2] = {nullptr, nullptr};
    const int dev[2] = {primary_device(), device};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        On on(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st[k], cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            tested = false;
        }
    }
    double copy_ms = 0.0;
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {   // 0: primary -> peer, 1: peer -> primary
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        {
            On on(dev[from]);
            if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        }
        {
            On on(dev[to]);
            if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
            const auto t0 = std::chrono::steady_clock::now();
            if (cudaMemcpyPeerAsync(d[to], dev[to], d[from], dev[from], kN * 4, st[to]) != cudaSuccess ||
                cudaStreamSynchronize(st[to]) != cudaSuccess) { tested = false; break; }
            copy_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
        if (copy_ms > 5.0) break;
    }
    for (int k = 0; k < 2; ++k) {
        On on(dev[k]);
        if (st[k] != nullptr) cudaStreamDestroy(st[k]);
        if (d[k] != nullptr) cudaFree(d[k]);
    }
    if (!tested) { cudaGetLastError(); return false; }
    if (!ok) return false;
    return copy_ms <= 2.0;   // reject stalling PCIe host-bridge P2P (~98 ms/copy) in favor of portable host staging
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    if (device_ < 0) return;
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (d_gate_vals_) cudaFree(d_gate_vals_);
        if (h_gate_) cudaFreeHost((void*) h_gate_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_q8_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    d_gate_vals_ = nullptr;
    h_gate_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 0 || device >= count || device == primary_device()) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not a valid secondary GPU (" + std::to_string(count) +
              " devices, primary=" + std::to_string(primary_device()) + "; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, primary_device(), device);
        cudaDeviceCanAccessPeer(&b, device, primary_device());
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_ && !p2p_topology_safe(primary_device(), device)) {
            std::fprintf(stderr, "strata: peer GPU %d: GPUs sit behind a PCIe host bridge (PHB/SYS, no shared PCIe switch or NVLink); "
                                 "using portable host-pinned staging instead of P2P (STRATA_PEER_NO_P2P=0 overrides)\n", device);
            p2p_ = false;
        }
        if (p2p_) {
            On on0(primary_device());
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(primary_device(), 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
            // recommend, never force: the direct path is used only if a test copy each way arrives exact (#2666)
            if (p2p_ && !p2p_copies_data(device)) {
                std::fprintf(stderr, "strata: peer GPU %d: peer access is enabled but a test copy between the two GPUs arrived wrong "
                                     "(a host bridge between them, no real P2P): the prompt path takes the host route instead "
                                     "(STRATA_PEER_NO_P2P=1 does that by hand)\n", device);
                { On q0(primary_device()); cudaDeviceDisablePeerAccess(device); cudaGetLastError(); }
                { On q1(device); cudaDeviceDisablePeerAccess(primary_device()); cudaGetLastError(); }
                p2p_ = false;
            }
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, sizeof(Meta)), "plan", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) ff), (int) ff);
    }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    return true;
}

bool PeerExperts::direct_mode() {
    static const bool v = [] { const char* e = std::getenv("STRATA_PEER_DIRECT"); return e == nullptr || std::atoi(e) != 0; }();
    return v;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out) {
    const bool direct = direct_mode() && out != nullptr;
    struct Timer {
        double& acc;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } timer{ms_launch};
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    Meta& m = *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
        m.ptr[groups] = (unsigned long long) cache_.device_slot(slot);
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    // direct: x is the verifier's pinned (portable) doorbell row block - copied from where it is
    if (!direct) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
    On on(device_);
    const cudaStream_t s = stream_;
    if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "input", err) ||
        !ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err))
        return false;
    Meta* dm = (Meta*) d_meta_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& f = lay.fmt[(size_t) layer];
        // APR: Hadamard-folded gate/up take H (s * x); d_x_ is this device's private copy, rotated in place
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (f.had_d) { L.had_block = f.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
    } else {
        err = "peer experts: only native packs are supported";
        return false;
    }
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            strata::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        return true;
    }
    return ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
              "results", err);
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) return true;
    ++launches;
    const auto t0 = std::chrono::steady_clock::now();
    {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    const bool gated = gate_landed_ != nullptr;
    if (gated) {   // the copies go in layer order and each layer's landing raises the gate's counter
        std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.layer < b.layer; });
        gate_landed_->store(-1, std::memory_order_release);
        if (h_gate_ != nullptr) *h_gate_ = -1;
        gate_has_->assign((size_t) n_layers_, 0);
        *gate_last_ = -1;
    }
    On on(device_);
    int32_t issued_layer = -1;   // the layer whose copies are on the stream and not yet ticked
    auto tick = [&](int32_t layer) -> bool {
        if ((h_gate_ != nullptr
                 ? cudaMemcpyAsync((void*) h_gate_, d_gate_vals_ + layer, 4, cudaMemcpyDeviceToHost, refill_)
                 : cudaLaunchHostFunc(refill_, [](void* p) {
                auto* t = (Tick*) p;
                int32_t cur = t->counter->load(std::memory_order_relaxed);
                while (cur < t->layer && !t->counter->compare_exchange_weak(cur, t->layer, std::memory_order_release)) {}
            }, &ticks_[(size_t) layer])) == cudaSuccess)
            return true;
        (void) cudaGetLastError();
        cudaStreamSynchronize(refill_);   // no callback: wait here, the gate opens at once
        gate_landed_->store((int32_t) n_layers_, std::memory_order_release);
        return false;
    };
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        if (gated && issued_layer >= 0 && issued_layer != s.layer) tick(issued_layer);
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                refill_), "refill", err))
            return false;
        if (gated) {
            (*gate_has_)[(size_t) s.layer] = 1;
            *gate_last_ = std::max(*gate_last_, s.layer);
            issued_layer = s.layer;
        }
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    if (gated && issued_layer >= 0) tick(issued_layer);
    if (!ck(cudaEventRecord(refill_ev_, refill_), "refill event", err)) return false;
    // #2188: hand the copies and their landing signals to the GPU now (WDDM batches a stream's work until a driver
    // call flushes it, and the verify window only spins on the pinned int; Linux submits at once: a no-op)
    if (gated && h_gate_ != nullptr && issued_layer >= 0) (void) cudaStreamQuery(refill_);
    return true;
}

volatile int32_t* PeerExperts::set_overlap_gate(std::atomic<int32_t>* landed, std::vector<uint8_t>* has, int32_t* last,
                                                bool copy_signal) {
    gate_landed_ = landed;
    gate_has_ = has;
    gate_last_ = last;
    ticks_.resize((size_t) n_layers_);
    for (int64_t l = 0; l < n_layers_; ++l) ticks_[(size_t) l] = {landed, (int32_t) l};
    if (copy_signal && h_gate_ == nullptr) {
        On on(device_);
        std::vector<int32_t> iota((size_t) n_layers_);
        for (size_t i = 0; i < iota.size(); ++i) iota[i] = (int32_t) i;
        int32_t* vals = nullptr;
        void* hp = nullptr;
        if (cudaMalloc((void**) &vals, iota.size() * 4) == cudaSuccess &&
            cudaMemcpy(vals, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaHostAlloc(&hp, 64, cudaHostAllocPortable) == cudaSuccess) {
            *(volatile int32_t*) hp = -1;
            d_gate_vals_ = vals;
            h_gate_ = (volatile int32_t*) hp;
        } else {   // host functions then
            (void) cudaGetLastError();
            if (vals) cudaFree(vals);
        }
    }
    return h_gate_;
}

void PeerExperts::flip_pending() {
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
}

void PeerExperts::apply_pending(bool wait) {
    if (pending_.empty()) return;
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
    pending_.clear();
}

}  // namespace strata::core
