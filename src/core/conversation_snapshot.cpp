// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "conversation_checked.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/platform/env.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace strata::core {
namespace conversation_detail {   // (conversation_state.cpp)
bool conversation_vram_room(size_t want);
void conversation_background_thread();
}  // namespace conversation_detail
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode == 2 || st.kv_q4 || st.kv_int8)) {   // a streamed one reads its host copy
        error = "conversation snapshot: hybrid K8V4 cannot be a ring and needs distinct format flags";
        return false;
    }
    const auto s = strata::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        error = "conversation snapshot: invalid K/V geometry";
        return false;
    }
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t per = st.kv_q4 ? (size_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint
    // restore reconstructs that row when rewinding to an earlier prefix.
    const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V (#293,
    // STRATA_KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    l = {(st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated, cells, pooled, s.page_size, 0, 0, 0, 0, 0};
    using conversation_detail::product;
    if (!product(l.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(l.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                           int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(l.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        error = "conversation snapshot: K/V byte count overflow";
        return false;
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if (st.kv_hybrid) {
        if (!product(l.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                   (uint64_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            error = "conversation snapshot: hybrid V byte count overflow";
            return false;
        }
        l.value_scales = 0;
    }
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q4 : st.v_q4, host ? st.host.k_scale : st.k_scale,
                nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                    : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    return true;
}

// STRATA_PARK_VRAM_MIB=N (with STRATA_PARK_ASYNC; opt-in, upstream 85e3dde52): a park also keeps the whole K/V of its
// image in VRAM, in a slot of its own, while the slots' total stays within N MiB; a restore of that image copies from
// the slot device to device instead of from the host buffers over the link, and needs no wait for the background
// copy.  A slot is released with the image or its K/V layers (their tokens), or all at once under VRAM pressure
// (conversation_park_vram_release_all).  Host thread only; the background copy only reads a slot's bytes, and every
// reuse or free of one comes after a join (conversation_park_async_begin, release_all).
struct VramKvSlot {
    void* p = nullptr;
    size_t cap = 0, used = 0;
    uint64_t tag = 0;   // 0: free (kept for reuse by size)
    int dev = -1;
    struct Part { uint64_t tag; size_t part, offset, bytes; };   // one layer's part (k, v, k_scale, v_scale, pooled)
    std::vector<Part> parts;
};
struct VramKv {
    std::vector<std::unique_ptr<VramKvSlot>> slots;
    uint64_t next_tag = 1;   // slot and layer tags, never reused
};
VramKv& vram_kv() { static VramKv v; return v; }
size_t park_vram_budget() {
    static const size_t b = [] {
        const char* v = strata::env_get("PARK_VRAM_MIB");
        const long long mib = v != nullptr ? std::strtoll(v, nullptr, 10) : 0;
        return mib > 0 ? (size_t) mib << 20 : (size_t) 0;
    }();
    return b;
}
void vram_kv_free(VramKvSlot& s) {
    strata::gpu::free(s.p);
    s.p = nullptr; s.cap = 0;
}
void vram_kv_release(VramKvSlot& s) { s.tag = 0; s.used = 0; s.parts.clear(); }   // free, kept for reuse
// A slot of `need` bytes on `dev`: the best-fitting free one, else (the free ones freed first) a new one, rounded up
// to 64 MiB so a growing conversation's next parks reuse it, while the slots' total stays within the budget and the
// VRAM keeps its headroom beside it.  nullptr, with `why`, otherwise.
VramKvSlot* vram_kv_take(size_t need, int dev, const char*& why) {
    VramKv& v = vram_kv();
    const size_t budget = park_vram_budget();
    if (budget == 0) { why = "off"; return nullptr; }
    auto stamp = [&v](VramKvSlot* s) {   // a fresh tag, nothing recorded yet
        s->tag = v.next_tag++;
        s->used = 0;
        s->parts.clear();
        return s;
    };
    VramKvSlot* best = nullptr;
    for (auto& s : v.slots)
        if (s->tag == 0 && s->dev == dev && s->cap >= need && (best == nullptr || s->cap < best->cap)) best = s.get();
    if (best != nullptr) return stamp(best);
    size_t held = 0;
    for (size_t i = 0; i < v.slots.size();) {
        if (v.slots[i]->tag != 0) { held += v.slots[i]->cap; ++i; continue; }
        vram_kv_free(*v.slots[i]);
        v.slots.erase(v.slots.begin() + (std::ptrdiff_t) i);
    }
    const size_t want = (need + ((size_t) 64 << 20) - 1) & ~(((size_t) 64 << 20) - 1);
    if (held > budget || want > budget - held) { why = "budget"; return nullptr; }
    if (!conversation_detail::conversation_vram_room(want)) { why = "vram"; return nullptr; }
    void* p = strata::gpu::alloc_device(want);
    if (p == nullptr) { why = "alloc"; return nullptr; }
    auto fresh = std::make_unique<VramKvSlot>();
    fresh->p = p; fresh->cap = want; fresh->dev = dev;
    v.slots.push_back(std::move(fresh));
    return stamp(v.slots.back().get());
}
// The live slot on the current device holding parts under `tag` (nullptr: none, or tag 0)
const VramKvSlot* vram_kv_find(uint64_t tag) {
    if (tag == 0) return nullptr;
    const int dev = current_device();
    for (const auto& s : vram_kv().slots) {
        if (s->tag == 0 || s->dev != dev) continue;
        for (const auto& p : s->parts)
            if (p.tag == tag) return s.get();
    }
    return nullptr;
}
// Whether the slot holds, under `tag`, at least sizes[i] bytes of every part i that has any; `at` gets the offsets
bool vram_kv_locate(const VramKvSlot& s, uint64_t tag, const std::array<size_t,5>& sizes, std::array<size_t,5>& at) {
    std::array<bool,5> have{};
    for (const auto& p : s.parts)
        if (p.tag == tag && p.part < at.size() && p.bytes >= sizes[p.part]) { have[p.part] = true; at[p.part] = p.offset; }
    for (size_t i = 0; i < sizes.size(); ++i)
        if (sizes[i] > 0 && !have[i]) return false;
    return true;
}

// STRATA_PARK_ASYNC=1 (opt-in, upstream f73b3c86a): a park's device K/V goes device to device into a VRAM stage, and a
// background thread moves it into the image's host buffers through two pinned 4 MiB buffers in turn, so the next
// prompt does not wait for the copy over the link.  The stage is allocated while the VRAM keeps its headroom beside it
// (conversation_vram_room); without it the park copies as before.
struct ParkAsync {
    std::mutex mu;
    std::future<std::string> pending;   // the copy in flight: empty when it landed, else why it failed
    struct Item { uint8_t* dst; size_t n, at; };
    std::vector<Item> items;
    uint8_t* stage = nullptr;
    size_t cap = 0, used = 0;
    void* host[2] = {nullptr, nullptr};   // pinned, kParkPiece each
    strata::gpu::Stream stream = nullptr;   // a copy stream: the device copies into the stage, then the pieces
    int dev = -1;
    bool active = false;
    VramKvSlot* slot = nullptr;   // STRATA_PARK_VRAM_MIB: the begun park's slot (the stage is not used then)
    uint64_t slot_tag = 0;        // its tag, kept past launch for the image's and its layers' tokens
};
ParkAsync& park_async() { static ParkAsync a; return a; }
constexpr size_t kParkPiece = (size_t) 4 << 20;
bool park_async_on() {
    static const bool on = [] { const char* v = strata::env_get("PARK_ASYNC"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// Whether part `i` of a layer is device memory, which the stage takes: the K/V pools of a resident layer (kv_mode 0)
// and every layer's indexer rows; a streamed layer's and the draft ring's pools are their host copies.
bool on_device(const QsaState& st, const void* p) {
    return st.kv_mode == 0 || !strata::gpu::is_host_usm(p);
}
// STRATA_PARK_TRACE (upstream ca378670b): where conversation_kv_save's time goes since the park's begin (ms): the
// device-memory checks, the host buffer resizes and the copies (into the stage, or the synchronous ones)
struct KvSaveTrace { double attrs = 0, resize = 0, copy = 0; };
KvSaveTrace& kv_save_trace() { static KvSaveTrace t; return t; }
bool park_trace_on() {
    static const bool on = strata::env_get("PARK_TRACE") != nullptr;
    return on;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // device USM, host USM (the streamed layers' authoritative host pools) or plain host memory alike
    if (strata::gpu::copy(dst, src, n)) return true;
    error = std::string("conversation snapshot copy: ") + strata::gpu::last_error();
    return false;
}

// One endpoint is a ConversationBuffer (ordinary host memory), the other is a state pool.  Do not infer that an
// unknown pointer is host memory: virtual device allocations need not be reported as USM.  A host USM pool can be
// copied by the CPU after the compute queue is drained, preserving transfer's ordering without using a copy engine.
bool transfer_host(void* dst, const void* src, const void* pool, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    if (!strata::gpu::is_host_usm(pool)) return transfer(dst, src, n, error);
    if (!strata::gpu::stream_sync(nullptr)) {
        error = std::string("conversation snapshot copy: ") + strata::gpu::last_error();
        return false;
    }
    std::memcpy(dst, src, n);
    return true;
}
} // namespace

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    // STRATA_PARK_VRAM_MIB: this layer's parts go into the park's slot under a fresh tag (0: no slot this park)
    image.dev_tag = park_async().slot != nullptr ? vram_kv().next_tag++ : 0;
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / strata::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        {   // STRATA_PARK_VRAM_MIB: with a slot, the whole part goes device to device into it (a restore of this image
            // reads it from there) and the host buffer gets only the bytes past the kept prefix, read from the slot on
            // the background thread; a part the slot cannot take is copied synchronously - never through the stage,
            // whose offsets launch would read from the slot
            ParkAsync& a = park_async();
            if (a.slot != nullptr) {
                VramKvSlot& sl = *a.slot;
                const bool fits = src[i] != nullptr && sizes[i] > 0 && sl.used + sizes[i] <= sl.cap && on_device(st, src[i]);
                if (fits) {
                    const size_t at0 = sl.used;
                    if (!strata::gpu::copy_async(static_cast<uint8_t*>(sl.p) + at0, src[i], sizes[i], a.stream)) {
                        error = std::string("conversation snapshot slot copy: ") + strata::gpu::last_error();
                        return false;
                    }
                    sl.parts.push_back({image.dev_tag, i, at0, sizes[i]});
                    sl.used += sizes[i];
                    dst[i]->resize_for_overwrite(sizes[i]);
                    if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                            a.items.push_back({p, n, at0 + at});
                            return true;
                        })) return false;
                } else {
                    dst[i]->resize(sizes[i]);
                    if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                            return transfer_host(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, src[i], n, error);
                        })) return false;
                }
                if (reused_bytes) *reused_bytes += keep;
                continue;
            }
        }
        // STRATA_PARK_ASYNC: device bytes go to the VRAM stage now and into the buffer on the background thread
        ParkAsync& a = park_async();
        const bool trace = park_trace_on();
        using Steady = std::chrono::steady_clock;
        auto tick = [trace] { return trace ? Steady::now() : Steady::time_point{}; };
        const auto t0 = tick();
        const bool staged = a.active && src[i] != nullptr && sizes[i] > keep && a.used + (sizes[i] - keep) <= a.cap &&
                            on_device(st, src[i]);
        const auto t1 = tick();
        if (staged) dst[i]->resize_for_overwrite(sizes[i]);
        else dst[i]->resize(sizes[i]);
        const auto t2 = tick();
        const bool copied = dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                const uint8_t* s = src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr;
                if (!staged) return transfer_host(p, s, src[i], n, error);
                if (!strata::gpu::copy_async(a.stage + a.used, s, n, a.stream)) {
                    error = std::string("conversation snapshot stage copy: ") + strata::gpu::last_error();
                    return false;
                }
                a.items.push_back({p, n, a.used});
                a.used += n;
                return true;
            });
        if (trace) {
            auto ms = [](Steady::time_point x, Steady::time_point y) {
                return std::chrono::duration<double, std::milli>(y - x).count();
            };
            KvSaveTrace& t = kv_save_trace();
            t.attrs += ms(t0, t1); t.resize += ms(t1, t2); t.copy += ms(t2, tick());
        }
        if (!copied) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    return conversation_kv_restore_prefix(image, st, g, upto, upto, index, error);
}

bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool index, std::string& error) {
    // STRATA_PARK_VRAM_MIB: from the image's VRAM slot when it holds every part (no host buffer is read: no join)
    const VramKvSlot* slot = vram_kv_find(image.dev_tag);
    if (slot == nullptr) conversation_park_join();
    if (upto < 0 || upto > captured) {
        error = "conversation snapshot: prefix beyond the captured K/V";
        return false;
    }
    if (!conversation_kv_validate(image, st, g, captured, index, error)) return false;
    Layout l{};   // the prefix's extent; validation above checked the image against the captured one
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] > src[i]->size()) { error = "conversation snapshot: prefix beyond the K/V payload"; return false; }
    const auto dst = pools(st);
    // STRATA_PARK_VRAM_MIB: the prefix is the leading bytes of each part in the slot too
    std::array<size_t,5> off{};
    if (slot != nullptr && !vram_kv_locate(*slot, image.dev_tag, sizes, off)) { slot = nullptr; conversation_park_join(); }
    if (slot != nullptr) {
        for (size_t i = 0; i < src.size(); ++i)
            if (!transfer(dst[i], static_cast<const uint8_t*>(slot->p) + off[i], sizes[i], error)) return false;
    } else {
        for (size_t i = 0; i < src.size(); ++i)
            if (!src[i]->visit(0, sizes[i], [&](const uint8_t* p, size_t n, size_t at) {
                    return transfer_host(static_cast<uint8_t*>(dst[i]) + at, p, dst[i], n, error);
                })) return false;
    }
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    if (st.kv_mode == 2 && upto > 0) {
        auto shapes = strata::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        strata::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                        std::max<int64_t>(0, end - st.n_slots), end, st.n_slots, shapes, nullptr);
    }
    if (strata::gpu::device_sync()) return true;
    error = std::string("conversation snapshot residency restore: ") + strata::gpu::last_error();
    return false;
}

bool conversation_kv_part_sizes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index,
                                std::array<uint64_t, 5>& sizes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    return true;
}

bool conversation_kv_source(SessionKvSource& out, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    SessionKvSource s;
    s.format = l.format; s.cells = l.cells; s.heads = g.n_head_kv; s.head_dim = g.head_dim;
    s.page_size = l.page_size; s.pooled_rows = l.pooled_rows; s.idx_dim = g.idx_key_dim;
    s.sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const auto src = pools(st);
    const auto sizes = s.sizes;
    auto why = std::make_shared<std::string>();
    s.read = [src, sizes, why](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > sizes[part] || n > sizes[part] - offset) { *why = "K/V read out of range"; return false; }
        return transfer_host(dst, src[part] ? static_cast<const uint8_t*>(src[part]) + offset : nullptr, src[part], n, *why);
    };
    s.error = [why] { return *why; };
    out = std::move(s);
    return true;
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error) {
    conversation_park_join();
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer_host(buffer.data(), static_cast<const uint8_t*>(device) + offset, device, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto authoritative = pools(st);
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return compare(static_cast<const uint8_t*>(authoritative[i]) + at, p, n, true);
            })) return false;
    if (st.kv_mode == 2 && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

void conversation_park_join() {
    ParkAsync& a = park_async();
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.pending.valid()) {
        const std::string e = a.pending.get();
        if (!e.empty()) std::fprintf(stderr, "xestrata: XESTRATA_PARK_ASYNC: a parked K/V copy failed (%s)\n", e.c_str());
    }
}

void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms) {
    const KvSaveTrace& t = kv_save_trace();
    attrs_ms = t.attrs; resize_ms = t.resize; copy_ms = t.copy;
}

bool conversation_park_async_begin(size_t need) {
    if (park_trace_on()) kv_save_trace() = {};   // (the sums cover this park's saves)
    conversation_park_join();   // (the stage is reused)
    ParkAsync& a = park_async();
    a.active = false;
    a.items.clear();
    a.used = 0;
    a.slot = nullptr;
    a.slot_tag = 0;
    if (!park_async_on() || need == 0) return false;
    const int dev = current_device();
    // STRATA_PARK_VRAM_MIB: a slot of the whole K/V instead of the stage, while the budget allows - taken before the
    // stage would be allocated again, so a growing conversation's park pays no free (a drain) when it gets one
    const char* why = "";
    a.slot = vram_kv_take(need, dev, why);
    a.slot_tag = a.slot != nullptr ? a.slot->tag : 0;
    if (park_trace_on() && park_vram_budget() != 0) {
        if (a.slot != nullptr)
            std::fprintf(stderr, "xestrata parktrace: kv slot %zu MB tag %llu\n", need >> 20, (unsigned long long) a.slot_tag);
        else
            std::fprintf(stderr, "xestrata parktrace: kv slot: none (%s)\n", why);
    }
    if (a.dev != dev || (a.stage != nullptr && a.slot == nullptr && a.cap < need)) {
        if (a.stage) strata::gpu::free(a.stage);
        a.stage = nullptr; a.cap = 0;
    }
    if (a.dev != dev) {
        if (a.host[0]) strata::gpu::free(a.host[0]);
        if (a.host[1]) strata::gpu::free(a.host[1]);
        if (a.stream) strata::gpu::stream_destroy(a.stream);
        a.stream = nullptr; a.host[0] = a.host[1] = nullptr;
    }
    a.dev = dev;
    if (a.stream == nullptr) {
        a.stream = strata::gpu::stream_create_copy();
        if (a.stream != nullptr) {
            a.host[0] = strata::gpu::alloc_host(kParkPiece);
            a.host[1] = strata::gpu::alloc_host(kParkPiece);
        }
        if (a.stream == nullptr || a.host[0] == nullptr || a.host[1] == nullptr) {
            if (a.host[0]) strata::gpu::free(a.host[0]);
            if (a.host[1]) strata::gpu::free(a.host[1]);
            if (a.stream) strata::gpu::stream_destroy(a.stream);
            a.stream = nullptr; a.host[0] = a.host[1] = nullptr;
            if (a.slot != nullptr) { vram_kv_release(*a.slot); a.slot = nullptr; a.slot_tag = 0; }
            return false;
        }
    }
    if (a.slot != nullptr) {   // STRATA_PARK_VRAM_MIB: the slot stands in for the stage this park
        a.active = true;
        return true;
    }
    if (a.stage == nullptr) {
        // rounded up to 64 MiB: the next parks of a growing conversation reuse it instead of allocating it again (a
        // free drains the device's queues, and the checkpoint copy may be in flight at this point)
        const size_t want = (need + ((size_t) 64 << 20) - 1) & ~(((size_t) 64 << 20) - 1);
        if (!conversation_detail::conversation_vram_room(want)) return false;
        a.stage = static_cast<uint8_t*>(strata::gpu::alloc_device(want));
        if (a.stage == nullptr) return false;
        a.cap = want;
    }
    a.active = true;
    return true;
}

void conversation_park_async_abandon() {
    ParkAsync& a = park_async();
    if (!a.active) return;
    a.active = false;
    a.items.clear();
    (void) strata::gpu::stream_sync(a.stream);   // nothing reads the stage after
    if (a.slot != nullptr) { vram_kv_release(*a.slot); a.slot = nullptr; a.slot_tag = 0; }   // (STRATA_PARK_VRAM_MIB)
}

bool conversation_park_async_launch(std::string& error) {
    ParkAsync& a = park_async();
    if (!a.active) return true;
    a.active = false;
    if (!strata::gpu::stream_sync(a.stream)) {   // the device-to-device copies into the stage
        a.items.clear();
        if (a.slot != nullptr) { vram_kv_release(*a.slot); a.slot = nullptr; a.slot_tag = 0; }   // (STRATA_PARK_VRAM_MIB)
        error = std::string("conversation snapshot stage: ") + strata::gpu::last_error();
        return false;
    }
    // STRATA_PARK_VRAM_MIB: the host copies read the slot; it stays until the image and its K/V layers go (their tokens)
    const uint8_t* stage = a.slot != nullptr ? static_cast<const uint8_t*>(a.slot->p) : a.stage;
    a.slot = nullptr;
    if (a.items.empty()) return true;
    std::vector<ParkAsync::Item> items;
    items.swap(a.items);
    void* host[2] = {a.host[0], a.host[1]};
    const strata::gpu::Stream st = a.stream;
    const int dev = a.dev;
    std::lock_guard<std::mutex> lk(a.mu);
    // two pinned pieces in turn: the next piece's copy over the link runs while this one is copied into its segment
    a.pending = std::async(std::launch::async, [items = std::move(items), stage, host, st, dev]() -> std::string {
        conversation_detail::conversation_background_thread();
        const OnDevice on(dev);
        strata::gpu::Event* done[2] = {nullptr, nullptr};
        bool ok = strata::gpu::event_create(&done[0]) && strata::gpu::event_create(&done[1]);
        struct Piece { uint8_t* dst; size_t n; };
        Piece held[2] = {{nullptr, 0}, {nullptr, 0}};
        int b = 0;
        auto land = [&](int k) {
            if (held[k].dst == nullptr || !ok) return;
            ok = strata::gpu::event_sync(done[k]);
            if (ok) std::memcpy(held[k].dst, host[k], held[k].n);
            held[k] = {nullptr, 0};
        };
        for (const auto& it : items)
            for (size_t o = 0; ok && o < it.n; o += kParkPiece) {
                const size_t n = std::min(kParkPiece, it.n - o);
                land(b);
                ok = ok && strata::gpu::copy_async(host[b], stage + it.at + o, n, st) &&
                     strata::gpu::event_record(done[b], st);
                if (ok) held[b] = {it.dst + o, n};
                b ^= 1;
            }
        land(b);
        land(b ^ 1);
        std::string why = ok ? std::string() : std::string(strata::gpu::last_error());
        for (strata::gpu::Event* d : done) if (d) strata::gpu::event_destroy(d);
        return why;
    });
    return true;
}

uint64_t conversation_park_slot_tag() { return park_async().slot_tag; }

void conversation_park_slot_release(uint64_t tag) {
    if (tag == 0) return;
    for (auto& s : vram_kv().slots)
        if (s->tag == tag) { vram_kv_release(*s); return; }
}

size_t conversation_park_vram_release_all() {
    if (park_vram_budget() == 0) return 0;   // off: nothing was taken, and the stage stays as it was
    conversation_park_join();   // the background copy may be reading a slot or the stage
    ParkAsync& a = park_async();
    VramKv& v = vram_kv();
    size_t freed = 0;
    for (size_t i = 0; i < v.slots.size();) {
        if (a.active && v.slots[i].get() == a.slot) { ++i; continue; }   // a park in progress keeps its own
        freed += v.slots[i]->cap;
        vram_kv_free(*v.slots[i]);
        v.slots.erase(v.slots.begin() + (std::ptrdiff_t) i);
    }
    if (a.stage != nullptr && !a.active) {
        strata::gpu::free(a.stage);
        freed += a.cap;
        a.stage = nullptr; a.cap = 0;
    }
    return freed;
}

void conversation_park_join_for(const SavedConversation& image) {
    size_t hits = 0;
    for (const auto& k : image.kv) hits += vram_kv_find(k.dev_tag) != nullptr ? 1 : 0;
    if (park_trace_on() && park_vram_budget() != 0)
        std::fprintf(stderr, "xestrata parktrace: kv restore: %zu of %zu layers from the slot\n", hits, image.kv.size());
    if (hits < image.kv.size()) conversation_park_join();
}

} // namespace strata::core
