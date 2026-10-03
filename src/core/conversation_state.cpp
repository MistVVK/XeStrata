// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "strata/core/gpu.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

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

// XeStrata's sessions hold every layer (upstream's per-stage layer carve, #216, is not ported): every QSA state.
size_t owned_qsa(const ModelGeometry& g) { return (size_t) std::max<int64_t>(g.n_qsa_layers(), 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[j]; }

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
    if (!conversation_state_sizes(g, z, error)) return false;
    if (ss.max_cells < 0 || tokens > static_cast<uint64_t>(ss.max_cells) || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(g) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(g); ++j) {
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

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image] == key))
                return fail(error, "checkpoint image identity differs");
            ++image;
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
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
} // namespace

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

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(g);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(g);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error) ||
        !copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
            !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
            !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_state_sizes(g, z, error)) return false;
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(g); ++j) {
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
    if (!conversation_state_sizes(g, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(g);
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
    const size_t layers = owned_qsa(g) + 1;
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

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = view.checkpoints;
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(g);
    captured.kv.resize(layers + 1);
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (!conversation_kv_save(captured.kv.back(), draft, g, upto, false, error,
                              std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!image.live.stage_parts.empty()) return fail(error, "layer-split parking is not supported");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(g);
    if (image.kv.size() != layers + 1) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return conversation_kv_validate(image.kv.back(), draft, g, upto, false, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(g); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (!conversation_kv_restore(image.kv.back(), draft, g, upto, false, error) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
