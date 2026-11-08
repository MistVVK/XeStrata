// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace strata::core {
namespace conversation_detail {
// Whether `want` bytes of VRAM can be taken for a copy of the conversation cache's (STRATA_CKPT_VRAM_SLOTS, the park's
// stage and slots) and still leave the device a sixteenth of its memory: upstream keeps 1.5 GB free, measured on a
// 24 GB RTX 3090, which is that share.  A device that does not report its free memory takes none.
bool conversation_vram_room(size_t want) {
    size_t free_bytes = 0, total = 0;
    if (!strata::gpu::mem_info(&free_bytes, &total)) return false;
    return free_bytes > want && free_bytes - want > total / 16;
}
// First thing a background copy thread does (STRATA_CKPT_ASYNC, STRATA_PARK_ASYNC; upstream ca378670b): a thread the
// host creates inherits the host's one CPU (strata/platform/aux_cpus.hpp), where its memcpys would take turns with the
// decode.  Wide first: the mask the aux placement keeps as this thread's own is the one restore() puts back, and the
// host's one CPU must not be it.  No output depends on where a thread runs.
void conversation_background_thread() {
#if defined(__linux__)
    cpu_set_t all;
    CPU_ZERO(&all);
    for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
    (void) sched_setaffinity(0, sizeof all, &all);   // the kernel keeps the CPUs the process may use
#endif
    strata::aux_cpus::pin_current_thread();          // (a no-op while the feature is off)
}
}  // namespace conversation_detail
namespace {
using conversation_detail::add;
using conversation_detail::product;

// Upstream #2049 #2065: a host-to-host copy that tells the watchdog it is moving.  Parked state (the running
// checkpoints) sits in ordinary pageable memory for the whole session; when the OS has written it out, copying it back
// runs at the drive's page-in speed and one copy of it stayed silent past the watchdog's limit, which ended an engine
// that was only slow.  One beat per 1 MiB slice keeps a copy that moves alive and still lets a real stall show.
template <class T>
std::vector<T> copy_beating(const std::vector<T>& src) {
    std::vector<T> out;
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(T));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        progress_beat();
    }
    return out;
}

ConversationCheckpoint copy_checkpoint_beating(const ConversationCheckpoint& c) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = c.imgs;
    o.used = c.used;
    o.pinned = c.pinned;
    o.gdn = copy_beating(c.gdn);
    o.dev_tag = c.dev_tag;   // the same bytes
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs) {
    for (const auto& c : cs) conversation_checkpoint_join(c.gdn.data());
    std::vector<ConversationCheckpoint> out;
    out.reserve(cs.size());
    for (const auto& c : cs) out.push_back(copy_checkpoint_beating(c));
    return out;
}

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    if (strata::gpu::device_sync()) return true;
    error = std::string("conversation snapshot synchronize: ") + strata::gpu::last_error();
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    if (strata::gpu::copy(dst, src, bytes)) return true;
    error = std::string("conversation snapshot running-state copy: ") + strata::gpu::last_error();
    return false;
}

// The session's carve (upstream #216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer,
// and so does one session_init did not carve (layer_hi 0: a session put together by hand, as the tests do).
bool carved(const SessionState& ss) { return ss.layer_hi > 0; }
size_t owned_qsa(const SessionState& ss, const ModelGeometry& g) {
    return (size_t) std::max<int64_t>(carved(ss) ? ss.qsa_alloc : g.n_qsa_layers(), 0);
}
const QsaState& owned(const SessionState& ss, size_t j) {
    return ss.qsa_states[(size_t) (carved(ss) ? ss.qsa_ord0 : 0) + j];
}

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > static_cast<uint64_t>(ss.max_cells) || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss, g) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss, g); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > static_cast<uint64_t>(st.max_cells) ||
            (tokens && tokens / static_cast<uint64_t>(block) >=
                           static_cast<uint64_t>(std::max<int64_t>(0, st.idx_pooled_rows))) ||
            !product(pooled_bytes, {tokens / static_cast<uint64_t>(block) + 1, static_cast<uint64_t>(g.idx_key_dim),
                                    sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

bool live_valid(const ConversationView& view, std::string& error) {
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    return true;
}

// a checkpoint's tokens and pictures start the live ones
bool checkpoint_prefix(const ConversationCheckpoint& c, const ConversationView& view, std::string& error) {
    if (c.ids.empty() || c.ids.size() > view.ids.size() || !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
        return fail(error, "checkpoint is not a live token prefix");
    size_t image = 0;
    for (const auto& key : view.images) {
        if ((uint64_t) key.start >= c.ids.size()) break;
        if (image >= c.imgs.size() || !(c.imgs[image] == key)) return fail(error, "checkpoint image identity differs");
        ++image;
    }
    if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    return true;
}

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (!live_valid(view, error)) return false;
    for (const auto& c : view.checkpoints) {
        if (!c.stage_parts.empty()) return fail(error, "checkpoint is not a live token prefix");
        if (!checkpoint_prefix(c, view, error) || !conversation_checkpoint_validate(c, ss, g, error)) return false;
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    return true;
}

// STRATA_CKPT_ASYNC=1 (opt-in, upstream 0f7224d16): a checkpoint's GDN state (~118 MB on Qwen3.8-Flash-Next) is
// copied device to device into a VRAM stage, and a background thread moves it over the link into the checkpoint's
// host vector (through a pinned buffer) or into the turn checkpoint's pinned staging (ckpt_staging), so the copy
// overlaps what follows (the rest of the prompt, the reply).  Every reader of a checkpoint's GDN bytes joins first
// (conversation_checkpoint_join).  The stage, the pinned buffer and the copy stream are allocated at the first such
// save; if one cannot be had, the save copies synchronously as without the switch.
struct CkptAsync {
    std::mutex mu;
    std::future<std::string> pending;   // the copy in flight: empty when it landed, else why it failed
    const void* dst = nullptr;          // the bytes it writes
    void* stage = nullptr;              // device
    void* host = nullptr;               // pinned: the DMA lands here, then a plain memcpy into the vector
    strata::gpu::Stream stream = nullptr;   // a copy stream: the copy engine where the device has one
    size_t cap = 0, host_cap = 0;
    int dev = -1;
};
CkptAsync& ckpt_async() { static CkptAsync a; return a; }
bool ckpt_async_on() {
    static const bool on = [] { const char* v = strata::env_get("CKPT_ASYNC"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool park_trace_env() {   // STRATA_PARK_TRACE: the timing of the save and park steps
    static const bool on = strata::env_get("PARK_TRACE") != nullptr;
    return on;
}
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
// The background copy runs only for a whole-model session: a layer split's stages save their parts from their own
// threads, and one stage serves one thread.  At least 1 MiB: a smaller state is not worth a thread.
bool ckpt_async_for(const SessionState& ss, const ModelGeometry& g, size_t gdn) {
    return ckpt_async_on() && gdn >= ((size_t) 1 << 20) &&
           (!carved(ss) || (ss.layer_lo == 0 && ss.layer_hi == g.n_layers));
}
// The stage for `n` bytes on `dev`, with the pinned bounce buffer when `pinned` (a copy into a host vector; the turn
// checkpoint's staging is pinned itself) and the copy stream.  The caller has joined.  false: the synchronous copies.
bool ckpt_async_ready(CkptAsync& a, size_t n, int dev, bool pinned) {
    if (a.dev != dev || a.cap < n || (pinned && a.host != nullptr && a.host_cap < n)) {
        if (a.stage) strata::gpu::free(a.stage);
        if (a.host) strata::gpu::free(a.host);
        if (a.stream) strata::gpu::stream_destroy(a.stream);
        a.stage = a.host = nullptr; a.stream = nullptr; a.cap = a.host_cap = 0; a.dev = -1;
    }
    a.dev = dev;
    if (a.stream == nullptr) a.stream = strata::gpu::stream_create_copy();
    if (a.stage == nullptr) {
        a.stage = strata::gpu::alloc_device(n);
        if (a.stage != nullptr) a.cap = n;
    }
    if (pinned && a.host == nullptr) {
        a.host = strata::gpu::alloc_host(n);
        if (a.host != nullptr) a.host_cap = n;
    }
    return a.stream != nullptr && a.stage != nullptr && (!pinned || a.host != nullptr);
}
// Queues the move of `n` bytes from the device buffer `src` into `dst` on the background thread: through `bounce`
// (pinned) when it is not null, else straight into `dst` (pinned itself).  In 4 MiB pieces, one in flight: a
// device-to-host copy queues behind the one before it on the copy engine, and the decode's own small reads would
// otherwise wait for the whole 100+ MB (upstream's measure on a PCIe Gen2 x4 link; the piece size is a latency
// bound, not a device's figure).
void ckpt_async_launch(CkptAsync& a, uint8_t* dst, void* bounce, const void* src, size_t n) {
    const strata::gpu::Stream st = a.stream;
    const int dev = a.dev;
    std::lock_guard<std::mutex> lk(a.mu);
    a.dst = dst;
    a.pending = std::async(std::launch::async, [dst, bounce, src, n, st, dev]() -> std::string {
        conversation_detail::conversation_background_thread();
        const OnDevice on(dev);
        constexpr size_t PIECE = (size_t) 4 << 20;
        auto* land = bounce != nullptr ? static_cast<uint8_t*>(bounce) : dst;
        for (size_t o = 0; o < n; o += PIECE) {
            const size_t k = std::min(PIECE, n - o);
            if (!strata::gpu::copy_async(land + o, static_cast<const uint8_t*>(src) + o, k, st) ||
                !strata::gpu::stream_sync(st))
                return std::string(strata::gpu::last_error());
            if (bounce != nullptr) std::memcpy(dst + o, land + o, k);
        }
        return std::string();
    });
}

// STRATA_CKPT_VRAM_SLOTS=N (with STRATA_CKPT_ASYNC, upstream 0f7224d16): the last N saved GDN states stay in VRAM, each
// tagged; a restore of a checkpoint whose tag a slot still holds copies device to device instead of over the link.  A
// slot is allocated only while the VRAM keeps its headroom beside it (conversation_vram_room).
struct VramSlot { void* p = nullptr; size_t n = 0; uint64_t tag = 0, used = 0; int dev = -1; };
struct VramSlots { std::vector<VramSlot> s; uint64_t clock = 0, next_tag = 0; };
VramSlots& vram_slots() { static VramSlots v; return v; }
int vram_slots_n() {
    static const int n = [] {
        const char* v = strata::env_get("CKPT_VRAM_SLOTS");
        return v ? std::max(0, std::min(16, (int) std::strtol(v, nullptr, 10))) : 0;
    }();
    return n;
}
// a slot for `n` bytes on `dev`: a new one while fewer than N are held and the VRAM allows, else the least recently
// used one of that size (nullptr: none)
VramSlot* vram_slot_for(size_t n, int dev) {
    VramSlots& v = vram_slots();
    VramSlot* best = nullptr;
    for (VramSlot& x : v.s)
        if (x.dev == dev && x.n == n && (best == nullptr || x.used < best->used)) best = &x;
    if ((int) v.s.size() < vram_slots_n() && conversation_detail::conversation_vram_room(n)) {
        VramSlot x;
        x.p = strata::gpu::alloc_device(n);
        if (x.p != nullptr) {
            x.n = n; x.dev = dev;
            v.s.push_back(x);
            return &v.s.back();
        }
    }
    return best;
}
// The GDN state into a VRAM slot (tagged into `tag`) or the stage, device to device; the background copy is then
// launched from there into `dst`.  1: queued; 0: no buffers (the caller copies synchronously); -1: the device copy
// failed (`error` set).
int ckpt_async_save_gdn(uint8_t* dst, const SessionState& ss, size_t n, bool pinned_dst, uint64_t& tag,
                        std::string& error) {
    tag = 0;
    CkptAsync& a = ckpt_async();
    const int dev = current_device();
    if (!ckpt_async_ready(a, n, dev, !pinned_dst)) return 0;
    const auto q0 = std::chrono::steady_clock::now();
    VramSlot* slot = vram_slots_n() > 0 ? vram_slot_for(n, dev) : nullptr;
    if (slot != nullptr) slot->tag = 0;   // (being rewritten)
    void* dsrc = slot != nullptr ? slot->p : a.stage;
    if (!copy(dsrc, ss.gdn_state, n, error)) return -1;   // device to device: fast
    if (slot != nullptr) {
        VramSlots& v = vram_slots();
        slot->tag = tag = ++v.next_tag;
        slot->used = ++v.clock;
    }
    if (park_trace_env()) std::fprintf(stderr, "xestrata parktrace: async ckpt d2d %.2f ms\n", ms_since(q0));
    ckpt_async_launch(a, dst, pinned_dst ? nullptr : a.host, dsrc, n);
    return 1;
}

// STRATA_CKPT_GATHER=1 (opt-in, upstream ca378670b): the small running-state vectors (the PLE history, then each
// layer's indexer tail, dead row and block position: 1 + 3 x layers pieces) are gathered device to device into one
// buffer, allocated once, and fetched with ONE copy into a pinned buffer, then copied into the vectors.  A small
// device-to-host copy issued while the checkpoint copy (STRATA_CKPT_ASYNC) is in flight waits behind its 4 MiB piece
// on the copy engine: one such wait instead of 37.  The same bytes in the same places.
struct CkptGather { std::mutex mu; void* stage = nullptr; void* host = nullptr; size_t cap = 0; int dev = -1; };
CkptGather& ckpt_gather() { static CkptGather a; return a; }
bool ckpt_gather_on() {
    static const bool on = [] { const char* v = strata::env_get("CKPT_GATHER"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// 1: the vectors are filled; 0: no buffers (the caller copies them one by one); -1: a copy failed (`error` set)
int ckpt_gather_small(ConversationCheckpoint& c, const SessionState& ss, const ConversationStateSizes& z,
                      size_t layers, std::string& error) {
    const size_t ple = c.ple.size(), tails = c.tails.size(), dead = c.dead.size(), block_pos = c.block_pos.size();
    const size_t total = ple + tails + dead + block_pos;
    if (total == 0) return 1;
    CkptGather& a = ckpt_gather();
    std::lock_guard<std::mutex> lk(a.mu);   // (a layer split's stages save from their own threads)
    const int dev = current_device();
    if (a.stage != nullptr && (a.cap < total || a.dev != dev)) {
        strata::gpu::free(a.stage);
        strata::gpu::free(a.host);
        a.stage = a.host = nullptr; a.cap = 0;
    }
    if (a.stage == nullptr) {
        a.stage = strata::gpu::alloc_device(total);
        a.host = a.stage != nullptr ? strata::gpu::alloc_host(total) : nullptr;
        if (a.host == nullptr) {
            if (a.stage) strata::gpu::free(a.stage);
            a.stage = nullptr;
            return 0;
        }
        a.cap = total; a.dev = dev;
    }
    // the pieces in the vectors' order, queued on the compute queue (in order, after the work already on it), then
    // the one copy to the host behind them
    auto* s = static_cast<uint8_t*>(a.stage);
    bool ok = true;
    auto put = [&](uint8_t* d, const void* src, size_t n) {
        if (ok && n) ok = src != nullptr && strata::gpu::copy_async(d, src, n, nullptr);
    };
    put(s, ss.ple_hist, ple);
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        put(s + ple + j * z.tail, st.idx_tail, z.tail);
        put(s + ple + tails + j * z.dead, st.idx_dead, z.dead);
        put(s + ple + tails + dead + j * z.block_pos, st.idx_block_pos, z.block_pos);
    }
    if (!ok) {
        error = std::string("conversation snapshot gather: ") + strata::gpu::last_error();
        return -1;
    }
    if (!copy(a.host, a.stage, total, error)) return -1;
    const auto* h = static_cast<const uint8_t*>(a.host);
    if (ple) std::memcpy(c.ple.data(), h, ple);
    if (tails) std::memcpy(c.tails.data(), h + ple, tails);
    if (dead) std::memcpy(c.dead.data(), h + ple + tails, dead);
    if (block_pos) std::memcpy(c.block_pos.data(), h + ple + tails + dead, block_pos);
    return 1;
}
} // namespace

void conversation_checkpoint_join(const void* only) {
    CkptAsync& a = ckpt_async();
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.pending.valid() && (only == nullptr || only == a.dst)) {
        const std::string e = a.pending.get();
        if (!e.empty()) std::fprintf(stderr, "xestrata: XESTRATA_CKPT_ASYNC: a checkpoint copy failed (%s)\n", e.c_str());
    }
}

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (!carved(ss)) return true;   // every layer
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss, g);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
    const auto j0 = std::chrono::steady_clock::now();
    conversation_checkpoint_join(nullptr);   // (the stage is reused)
    if (park_trace_env()) std::fprintf(stderr, "xestrata parktrace: save join %.2f ms\n", ms_since(j0));
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss, g);
    const bool ready = c.gdn.size() == z.gdn;   // (a vector of the right size has its pages in; a fresh one faults)
    const auto r0 = std::chrono::steady_clock::now();
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (park_trace_env()) std::fprintf(stderr, "xestrata parktrace: save vectors %s, resize %.2f ms\n",
                                       ready ? "ready" : "fresh", ms_since(r0));
    c.dev_tag = 0;
    int staged = 0;   // STRATA_CKPT_ASYNC: the GDN bytes land in c.gdn on the background thread
    if (ckpt_async_for(ss, g, c.gdn.size())) {
        staged = ckpt_async_save_gdn(c.gdn.data(), ss, c.gdn.size(), false, c.dev_tag, error);
        if (staged < 0) return false;
    }
    if (staged == 0 && !copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error)) return false;
    const auto s0 = std::chrono::steady_clock::now();
    const bool ok = [&] {
        // STRATA_CKPT_GATHER=1: the small vectors through one device gather and one copy (else one copy each)
        if (ckpt_gather_on()) {
            const int r = ckpt_gather_small(c, ss, z, layers, error);
            if (r != 0) return r > 0;
        }
        if (!copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
        for (size_t j = 0; j < layers; ++j) {
            const auto& st = owned(ss, j);
            if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
                !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
                !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
        }
        return true;
    }();
    if (park_trace_env()) std::fprintf(stderr, "xestrata parktrace: save small copies %.2f ms\n", ms_since(s0));
    if (!ok && staged > 0) conversation_checkpoint_join(c.gdn.data());   // a failed save's caller drops `c` at once
    return ok;
}

size_t conversation_checkpoint_staged_bytes(const SessionState& ss, const ModelGeometry& g) {
    ConversationStateSizes z;
    std::string error;
    if (!conversation_session_sizes(g, ss, z, error)) return 0;
    return z.gdn + (ss.ple_hist ? z.ple : 0) + owned_qsa(ss, g) * (z.tail + z.dead + z.block_pos);
}

// the order and sizes of conversation_checkpoint_save, packed one after another
bool conversation_checkpoint_stage(void* staging, void* stream, const SessionState& ss, const ModelGeometry& g,
                                   size_t tokens, std::string& error, uint64_t* dev_tag) {
    conversation_checkpoint_join(nullptr);   // (the stage is reused)
    if (dev_tag != nullptr) *dev_tag = 0;
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, tokens, z, error)) return false;
    auto* p = static_cast<uint8_t*>(staging);
    bool ok = true;
    auto put = [&](uint8_t* dst, const void* src, size_t bytes) {
        if (bytes == 0 || !ok) return;
        ok = src != nullptr && strata::gpu::copy_async(dst, src, bytes, stream);
    };
    const size_t ple = ss.ple_hist ? z.ple : 0;
    // STRATA_CKPT_ASYNC: the GDN state goes device to device now and into the staging on the background thread, beside
    // the reply; the small parts below are copied at once as before (unstage joins)
    int staged = 0;
    uint64_t tag = 0;
    if (ckpt_async_for(ss, g, z.gdn)) {
        staged = ckpt_async_save_gdn(p, ss, z.gdn, true, tag, error);
        if (staged < 0) return false;
    }
    if (dev_tag != nullptr) *dev_tag = tag;
    if (staged == 0) put(p, ss.gdn_state, z.gdn);
    put(p + z.gdn, ss.ple_hist, ple);
    p += z.gdn + ple;
    for (size_t j = 0; j < owned_qsa(ss, g); ++j) {
        const auto& st = owned(ss, j);
        put(p, st.idx_tail, z.tail);
        put(p + z.tail, st.idx_dead, z.dead);
        put(p + z.tail + z.dead, st.idx_block_pos, z.block_pos);
        p += z.tail + z.dead + z.block_pos;
    }
    if (ok && strata::gpu::stream_sync(stream)) return true;
    error = std::string("conversation checkpoint stage: ") + strata::gpu::last_error();
    return false;
}

bool conversation_checkpoint_unstage(ConversationCheckpoint& c, const void* staging, const SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss, g);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    const auto* p = static_cast<const uint8_t*>(staging);
    if (p == nullptr) return fail(error, "missing checkpoint staging");
    conversation_checkpoint_join(staging);   // STRATA_CKPT_ASYNC: the GDN state has landed in the staging
    auto take = [&](void* dst, size_t bytes) {
        if (bytes) std::memcpy(dst, p, bytes);
        p += bytes;
    };
    take(c.gdn.data(), c.gdn.size());
    take(c.ple.data(), c.ple.size());
    for (size_t j = 0; j < layers; ++j) {
        take(c.tails.data() + j * z.tail, z.tail);
        take(c.dead.data() + j * z.dead, z.dead);
        take(c.block_pos.data() + j * z.block_pos, z.block_pos);
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    // STRATA_CKPT_VRAM_SLOTS: a slot still tagged with this checkpoint's GDN bytes holds them on this device
    const VramSlot* hit = nullptr;
    if (c.dev_tag != 0) {
        const int dev = current_device();
        for (VramSlot& x : vram_slots().s)
            if (x.tag == c.dev_tag && x.dev == dev && x.n == c.gdn.size()) { x.used = ++vram_slots().clock; hit = &x; }
    }
    if (hit == nullptr) conversation_checkpoint_join(c.gdn.data());
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!copy(ss.gdn_state, hit != nullptr ? hit->p : (const void*) c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss, g); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!view_validate(view, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss, g);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + 1, sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i <= qsa; ++i) {
        const auto& st = i == qsa ? draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss, g) + 1;
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i + 1 != layers;
        const auto& st = index ? owned(ss, i) : draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

// (conversation_snapshot.cpp) STRATA_PARK_TRACE: conversation_kv_save's time since the park's begin, by part (ms)
void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms);

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes,
                                std::vector<ConversationCheckpoint>* moved) {
    using Steady = std::chrono::steady_clock;
    auto ms = [](Steady::time_point a, Steady::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto pj = Steady::now();
    conversation_checkpoint_join(nullptr);   // the last turn checkpoint's copy (STRATA_CKPT_ASYNC)
    conversation_park_join();                // the previous park's K/V copy (STRATA_PARK_ASYNC)
    const auto p0 = Steady::now();
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    const auto p1 = Steady::now();
    captured.cvec = view.cvec; captured.checkpoints = copy_checkpoints_beating(view.checkpoints);
    if (moved != nullptr)
        for (auto& c : *moved) {
            conversation_checkpoint_join(c.gdn.data());
            captured.checkpoints.push_back(std::move(c));
        }
    const auto p2 = Steady::now();
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss, g);
    captured.kv.resize(layers + 1);
    // STRATA_PARK_VRAM_MIB: the retained layers' tokens go now (joined above, so no wait): held past the launch below,
    // their replacement would join the copy just launched
    for (auto& k : captured.kv) k.pending.reset();
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    // STRATA_CKPT_ASYNC: the running state may land in live.gdn after the save returns; the image waits for that copy
    // before the vector is freed (an eviction within the copy's time; upstream ca378670b).  The join is selective: a
    // no-op once the copy has landed or another one is in flight.
    if (ckpt_async_on())
        captured.live_pending = std::shared_ptr<void>(new char, [dst = (const void*) captured.live.gdn.data()](void* q) {
            conversation_checkpoint_join(dst);
            delete static_cast<char*>(q);
        });
    const auto p3 = Steady::now();
    const int64_t upto = (int64_t) view.ids.size();
    // STRATA_PARK_ASYNC: the K/V bytes these saves copy (the stage they need; an over-estimate by the kept prefix)
    size_t kv_need = 0;
    for (size_t j = 0; j < layers; ++j) kv_need += conversation_kv_bytes(owned(ss, j), g, upto, true);
    kv_need += conversation_kv_bytes(draft, g, upto, false);
    const bool staged = conversation_park_async_begin(kv_need);
    const auto p4 = Steady::now();
    struct Abandon { bool on; ~Abandon() { if (on) conversation_park_async_abandon(); } } abandon{staged};
    double kv_max = 0, kv_sum = 0;   // (STRATA_PARK_TRACE: the per-layer save time)
    for (size_t j = 0; j < layers; ++j) {
        const auto k0 = Steady::now();
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
        const double kt = ms(k0, Steady::now());
        kv_sum += kt; kv_max = std::max(kv_max, kt);
    }
    if (park_trace_env()) {
        size_t kvb = 0;
        for (const auto& k : captured.kv) kvb += k.bytes();
        double attrs_ms = 0, resize_ms = 0, copy_ms = 0;
        conversation_kv_save_trace(attrs_ms, resize_ms, copy_ms);   // (the main layers' saves above)
        std::fprintf(stderr, "xestrata parktrace: joins %.1f ms, sync %.1f ms, %zu checkpoints copied %.1f ms, live state "
                             "%.1f ms, begin %.1f ms, K/V %zu MB %.1f ms (layer max %.2f, sum %.2f; attrs %.2f, resize "
                             "%.2f, copy %.2f)\n", ms(pj, p0), ms(p0, p1), captured.checkpoints.size(), ms(p1, p2),
                     ms(p2, p3), ms(p3, p4), kvb >> 20, ms(p4, Steady::now()), kv_max, kv_sum, attrs_ms, resize_ms,
                     copy_ms);
    }
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (!conversation_kv_save(captured.kv.back(), draft, g, upto, false, error,
                              std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    abandon.on = false;
    if (staged && !conversation_park_async_launch(error)) return false;
    if (staged) {
        // STRATA_PARK_VRAM_MIB: the image's slot goes with the image, after the copy that reads it; each K/V layer
        // carries a token of its own too (the layers outlive the image when they are retained for the next park, and
        // the copy fills their buffers).  Without a slot: the image's join, no layer tokens.
        const uint64_t slot = conversation_park_slot_tag();
        auto token = [slot] {
            return std::shared_ptr<void>(new char, [slot](void* q) {
                conversation_park_join();
                conversation_park_slot_release(slot);
                delete static_cast<char*>(q);
            });
        };
        captured.pending = token();
        if (slot != 0)
            for (auto& k : captured.kv) k.pending = token();
    }
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                   const QsaState& draft, std::string& error) {
    if (!view_validate(view, ss, g, error) || !sync(error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = copy_checkpoints_beating(view.checkpoints);
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    // STRATA_CKPT_ASYNC: the caller writes `meta` to the session file at once, so the running state's copy lands first
    conversation_checkpoint_join(captured.live.gdn.data());
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss, g);
    std::vector<SessionKvSource> out(layers + 1);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_source(out.back(), draft, g, upto, false, error)) return false;
    meta = std::move(captured);
    sources = std::move(out);
    return true;
}

bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& ss, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0) return fail(error, "invalid session cells");
    const uint64_t tokens = std::min<uint64_t>(max_tokens, (uint64_t) ss.max_cells);
    const size_t layers = owned_qsa(ss, g);
    if ((owned_qsa(ss, g) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    size_t tails = 0, dead = 0, block_pos = 0;
    if (!product(tails, {layers, z.tail}) || !product(dead, {layers, z.dead}) ||
        !product(block_pos, {layers, z.block_pos})) return fail(error, "indexer byte count overflow");
    SessionReadLimits l = limits;   // keeps the caller's admit / progress / max_file_bytes
    l.max_tokens = tokens;
    l.max_checkpoints = max_checkpoints;
    l.max_kv_layers = layers + 1;
    l.geometry = geometry_key(g);
    l.layer_range = std::make_pair(ss.layer_lo, ss.layer_hi);
    l.max_state_bytes = {z.gdn, ss.ple_hist ? z.ple : 0, tails, dead, block_pos};
    l.max_kv_bytes.assign(layers + 1, {});
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        const int64_t upto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(st.max_cells, 0));
        if (!conversation_kv_part_sizes(st, g, upto, true, l.max_kv_bytes[j], error)) return false;
    }
    const int64_t dupto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(draft.max_cells, 0));
    if (!conversation_kv_part_sizes(draft, g, dupto, false, l.max_kv_bytes.back(), error)) return false;
    limits = std::move(l);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!image.live.stage_parts.empty()) return fail(error, "layer-split parking is not supported");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss, g);
    if (image.kv.size() != layers + 1) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return conversation_kv_validate(image.kv.back(), draft, g, upto, false, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    conversation_checkpoint_join(image.live.gdn.data());
    for (const auto& c : image.checkpoints) conversation_checkpoint_join(c.gdn.data());
    conversation_park_join_for(image);   // (STRATA_PARK_VRAM_MIB: no wait when the K/V comes from its slot)
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss, g); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (!conversation_kv_restore(image.kv.back(), draft, g, upto, false, error) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
// ---- a layer split: every GPU's session, each on its own device (conversation_snapshot.hpp)
namespace {
bool sync_on(int device, std::string& error) {
    const OnDevice on(device);
    return sync(error);
}

// the stages' QSA layers in layer order (the one-GPU image's), each with its stage
std::vector<std::pair<const QsaState*, size_t>> split_layers(const std::vector<ConversationStage>& stages,
                                                             const ModelGeometry& g) {
    std::vector<std::pair<const QsaState*, size_t>> out;
    for (size_t i = 0; i < stages.size(); ++i)
        for (size_t j = 0; j < owned_qsa(*stages[i].session, g); ++j) out.emplace_back(&owned(*stages[i].session, j), i);
    return out;
}

bool split_stages(const std::vector<ConversationStage>& stages, std::string& error) {
    if (stages.size() < 2) return fail(error, "a layer split needs two or more stages");
    for (const auto& st : stages)
        if (st.session == nullptr) return fail(error, "a layer split stage has no session");
    return true;
}

// the live state and every checkpoint of a split view: a part for each later stage, each valid for its session
bool split_view_validate(const ConversationView& view, const ConversationCheckpoint* live,
                         const std::vector<ConversationStage>& stages, const ModelGeometry& g, std::string& error) {
    if (!split_stages(stages, error) || !live_valid(view, error)) return false;
    ConversationStateSizes z;
    for (const auto& st : stages)
        if (!checkpoint_targets(*st.session, g, view.ids.size(), z, error)) return false;
    auto parts_valid = [&](const ConversationCheckpoint& c) {
        if (c.stage_parts.size() + 1 != stages.size()) return fail(error, "checkpoint parts differ from the split");
        if (!conversation_checkpoint_validate(c, *stages[0].session, g, error)) return false;
        for (size_t i = 0; i < c.stage_parts.size(); ++i)
            if (c.stage_parts[i].ids.size() != c.ids.size() ||
                !conversation_checkpoint_validate(c.stage_parts[i], *stages[i + 1].session, g, error))
                return false;
        return true;
    };
    for (const auto& c : view.checkpoints)
        if (!checkpoint_prefix(c, view, error) || !parts_valid(c)) return false;
    return live == nullptr || parts_valid(*live);
}
}  // namespace

bool conversation_split_bytes(const ConversationView& view, const std::vector<ConversationStage>& stages,
                              const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!split_view_validate(view, nullptr, stages, g, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0;
    const auto qsa = split_layers(stages, g);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {(uint64_t) qsa.size() + 1, sizeof(ConversationKv)}))
        return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& st : stages) {   // each stage's running state, and its tokens again (its part's)
        ConversationStateSizes z;
        if (!conversation_session_sizes(g, *st.session, z, error)) return false;
        const uint64_t n = owned_qsa(*st.session, g);
        size_t tails = 0, dead = 0, positions = 0;
        if (!product(tails, {n, z.tail}) || !product(dead, {n, z.dead}) || !product(positions, {n, z.block_pos}))
            return fail(error, "snapshot byte count overflow");
        for (size_t k : {tails, dead, positions, z.gdn, st.session->ple_hist ? z.ple : 0, ids})
            if (!add(bytes, k)) return fail(error, "snapshot byte count overflow");
    }
    for (const auto& c : view.checkpoints) {
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
        for (const auto& part : c.stage_parts)
            if (!metadata_bytes(part, bytes)) return fail(error, "checkpoint byte count overflow");
    }
    const int64_t upto = (int64_t) view.ids.size();
    for (const auto& [st, stage] : qsa) {
        const size_t n = conversation_kv_bytes(*st, g, upto, true);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    const size_t n = conversation_kv_bytes(draft, g, upto, false);
    if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    return true;
}

bool conversation_split_save(SavedConversation& image, const ConversationView& view,
                             const std::vector<ConversationStage>& stages, const ModelGeometry& g,
                             const QsaState& draft, std::string& error) {
    size_t estimate = 0;
    if (!conversation_split_bytes(view, stages, g, draft, estimate, error)) return false;
    for (const auto& st : stages)
        if (!sync_on(st.device, error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.live.ids = view.ids;
    captured.live.imgs = view.images;
    captured.cvec = view.cvec;
    captured.checkpoints = copy_checkpoints_beating(view.checkpoints);
    for (size_t i = 0; i < stages.size(); ++i) {
        const OnDevice on(stages[i].device);
        ConversationCheckpoint part;
        part.ids = view.ids;
        part.imgs = view.images;
        ConversationCheckpoint& target = i == 0 ? captured.live : part;
        if (!conversation_checkpoint_save(target, *stages[i].session, g, error)) return false;
        if (i > 0) captured.live.stage_parts.push_back(std::move(part));
    }
    const int64_t upto = (int64_t) view.ids.size();
    const auto qsa = split_layers(stages, g);
    captured.kv.resize(qsa.size() + 1);
    for (size_t j = 0; j < qsa.size(); ++j) {
        const OnDevice on(stages[qsa[j].second].device);
        if (!conversation_kv_save(captured.kv[j], *qsa[j].first, g, upto, true, error)) return false;
    }
    {
        const OnDevice on(stages.back().device);
        if (!conversation_kv_save(captured.kv.back(), draft, g, upto, false, error)) return false;
    }
    image = std::move(captured);
    return true;
}

bool conversation_split_validate(const SavedConversation& image, const std::vector<ConversationStage>& stages,
                                 const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!split_view_validate(view, &image.live, stages, g, error)) return false;
    const auto qsa = split_layers(stages, g);
    if (image.kv.size() != qsa.size() + 1) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < qsa.size(); ++j)
        if (!conversation_kv_validate(image.kv[j], *qsa[j].first, g, upto, true, error)) return false;
    return conversation_kv_validate(image.kv.back(), draft, g, upto, false, error);
}

ConversationRestore conversation_split_restore(const SavedConversation& image,
                                               const std::vector<ConversationStage>& stages, const ModelGeometry& g,
                                               const QsaState& draft, std::string& error) {
    if (!conversation_split_validate(image, stages, g, draft, error)) return ConversationRestore::invalid;
    for (const auto& st : stages)
        if (!sync_on(st.device, error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    const auto qsa = split_layers(stages, g);
    for (size_t j = 0; j < qsa.size(); ++j) {
        const OnDevice on(stages[qsa[j].second].device);
        if (!conversation_kv_restore(image.kv[j], *qsa[j].first, g, upto, true, error))
            return ConversationRestore::transfer_failed;
    }
    {
        const OnDevice on(stages.back().device);
        if (!conversation_kv_restore(image.kv.back(), draft, g, upto, false, error))
            return ConversationRestore::transfer_failed;
    }
    for (size_t i = 0; i < stages.size(); ++i) {   // conversation_checkpoint_restore synchronizes its device
        const OnDevice on(stages[i].device);
        if (!conversation_checkpoint_restore(i == 0 ? image.live : image.live.stage_parts[i - 1], *stages[i].session,
                                             g, error))
            return ConversationRestore::transfer_failed;
    }
    return ConversationRestore::restored;
}
// upstream c9e571e0 (#1164): a borrowed checkpoint's K/V; the parked image stays as it is
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error) {
    conversation_checkpoint_join(image.live.gdn.data());
    for (const auto& c : image.checkpoints) conversation_checkpoint_join(c.gdn.data());
    conversation_park_join_for(image);   // (STRATA_PARK_VRAM_MIB: no wait when the K/V comes from its slot)
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    const int64_t captured = (int64_t) image.live.ids.size();
    if (upto < 1 || upto > captured) {
        fail(error, "the borrowed prefix is not inside the parked conversation");
        return ConversationRestore::invalid;
    }
    if (!sync(error)) return ConversationRestore::transfer_failed;
    for (size_t j = 0; j < owned_qsa(ss, g); ++j)
        if (!conversation_kv_restore_prefix(image.kv[j], owned(ss, j), g, captured, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (!conversation_kv_restore_prefix(image.kv.back(), draft, g, captured, upto, false, error))
        return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
