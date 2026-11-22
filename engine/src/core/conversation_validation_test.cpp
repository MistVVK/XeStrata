// Host-only validation fixtures. CUDA is linked but must never be initialized:
// invalid restores must return before the first CUDA call or destination write.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <cstring>

#if defined(CONVERSATION_TEST_TRANSFERS)
// GNU/ELF link wrapping exercises the actual restore control flow without
// initializing CUDA or deliberately poisoning a real device context. This is
// host-backend fault injection, not a claim of hardware-failure recovery.
namespace {
int copy_calls = 0, sync_calls = 0, fail_copy = 0, fail_sync = 0;
size_t copied_bytes = 0;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
    copied_bytes += n;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
// the snapshot/checkpoint copies go through a private non-blocking stream (conversation_copy.hpp)
extern "C" cudaError_t __wrap_cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    *s = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1)); return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    return __wrap_cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

void fixture(int format, int experts, bool zero_qsa, bool ple) {
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "checked canonical/pruned/zero-QSA sizing");
    Pools first(g, format), last(g, format), draft(g, format);
    std::array<QsaState, 2> layers{first.st, last.st};
    std::vector<uint8_t> gdn(z.gdn, 0xa5), history(ple ? z.ple : 0, 0xa5);
    SessionState ss;
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    // the whole-model carve, as session_init leaves a one-GPU session
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_hi = g.n_layers;
    auto checkpoint = [&](size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    };
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));
    check(conversation_snapshot_validate(image, ss, g, draft.st, error), "complete image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},ss,g,draft.st,estimate,error),
          "capture estimate works without CUDA");
    check(estimate == image.bytes(), "estimate covers checkpoint/indexer/spare payloads");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn) || !pristine(history) || ss.ple_prev[0] != -1 || ss.ple_prev[1] != -1) return false;
        for (const auto* p : {&first,&last,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image; mutate(bad);
        check(conversation_snapshot_restore(bad,ss,g,draft.st,error) == ConversationRestore::invalid,label);
        check(unchanged(), "invalid restore did not touch any layer or running-state buffer");
    };
    reject([](auto& s){s.kv.back().k.pop_back();}, "late draft corruption rejected before first layer write");
    reject([](auto& s){s.geometry[16] = 128;}, "different expert geometry rejected");
    reject([](auto& s){s.live.stage_parts.emplace_back();}, "layer-split live state rejected before writes");
    reject([](auto& s){s.checkpoints[0].stage_parts.emplace_back();}, "layer-split checkpoint rejected before writes");
    reject([](auto& s){s.kv.pop_back();}, "missing KV layer rejected");
    reject([](auto& s){s.live.gdn.pop_back();}, "bad live recurrence rejected");
    reject([](auto& s){s.checkpoints.back().gdn.pop_back();}, "bad retained checkpoint rejected");
    reject([](auto& s){s.checkpoints.back().ids[0] = 99;}, "foreign checkpoint prefix rejected");
    reject([](auto& s){s.checkpoints.back().imgs[0].hash++;}, "foreign checkpoint image rejected");
    reject([](auto& s){s.live.imgs[0].start = -1;}, "negative image position rejected");
    reject([](auto& s){s.live.ids[0] = -1;}, "negative live token rejected");
    reject([](auto& s){s.live.ple.push_back(0);}, "PLE size/enabled mismatch rejected");
    reject([](auto& s){s.live.tails.push_back(0);}, "tail size/zero-QSA mismatch rejected");
    if (!zero_qsa) {
        reject([](auto& s){s.kv[1].pooled.pop_back();}, "late indexer spare row corruption rejected");
        reject([](auto& s){s.live.dead.pop_back();}, "missing indexer spare key rejected");
        reject([](auto& s){s.live.block_pos.pop_back();}, "missing indexer metadata rejected");
        ss.qsa_states = nullptr;
        check(!conversation_snapshot_validate(image,ss,g,draft.st,error), "missing QSA targets rejected");
        ss.qsa_states = layers.data();
        reject([](auto& s){s.layer_lo = 4;}, "snapshot from another layer range rejected");
        // #216's carve: a split stage owning layers [4, 8) holds GDN rows 3..5 and QSA ordinal 1 only
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g,stage,zs,error) && zs.gdn == z.gdn / 6 * 3 && zs.tail == z.tail,
              "carved session sizes cover the owned rows only");
        check(!conversation_checkpoint_validate(image.live,stage,g,error), "whole-model checkpoint rejected by a carve");
        ConversationCheckpoint part = image.live;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        check(conversation_checkpoint_validate(part,stage,g,error), "carve-sized checkpoint validates");
        check(!conversation_snapshot_validate(image,stage,g,draft.st,error), "whole-model snapshot rejected by a carve");
        stage.qsa_alloc = 2;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the last QSA ordinal rejected");
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the GDN rows rejected");
    }
    auto bad_geometry = g; bad_geometry.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!conversation_state_sizes(bad_geometry,z,error), "running-state arithmetic overflow rejected");
    bad_geometry = g; bad_geometry.qsa_interval = 0;
    check(!conversation_state_sizes(bad_geometry,z,error), "zero layer interval rejected before division");
    bad_geometry = g; bad_geometry.n_head_kv = std::numeric_limits<int64_t>::max();
    check(conversation_kv_bytes(draft.st,bad_geometry,9,false)==0, "KV byte overflow rejected");
#if defined(CONVERSATION_TEST_TRANSFERS)
    auto reset = [&] {
        for (auto* p : {&first,&last,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0xa5);
        std::fill(gdn.begin(),gdn.end(),0xa5); std::fill(history.begin(),history.end(),0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        copy_calls = sync_calls = fail_copy = fail_sync = 0;
    };
    reset(); fail_sync = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "pre-transfer CUDA synchronization failure is fatal");
    check(copy_calls==0 && unchanged(), "pre-sync failure did not mutate state");
    reset(); fail_copy = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "first CUDA transfer failure is fatal");
    check(unchanged(), "first-copy failure did not mutate state");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "partial CUDA transfer failure is fatal, not an invalid-image fallback");
    check(!unchanged(), "fault fixture genuinely produced partial state");
    reset(); fail_sync = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "post-transfer CUDA synchronization failure is fatal");
    check(copy_calls>0 && !unchanged(), "post-sync failure occurred after state writes");
    reset();
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored,
          "injected host transfer backend completes a valid restore");
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
    fail_copy = copy_calls + 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back transfer failure is not a successful fingerprint");
    reset(); copied_bytes = 0;
    const ConversationView view{image.live.ids,image.live.imgs,image.checkpoints,true};
    SavedConversation full,incremental;
    check(conversation_snapshot_save(full,view,ss,g,draft.st,error), "full capture through host transfer backend");
    const size_t full_copies = copied_bytes;
    ConversationKvReuse reuse{full.kv,9,9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "incremental capture admission without transfers");
    copied_bytes = 0;
    const bool saved_incrementally = conversation_snapshot_save(incremental,view,ss,g,draft.st,error,std::move(reuse),&reused);
    if (!saved_incrementally) std::fprintf(stderr,"capture error: %s\n",error.c_str());
    check(saved_incrementally, "incremental capture through host backend");
    check(reused > 0 && copied_bytes + reused == full_copies, "reused byte count measures transfers actually omitted");
    check(incremental.bytes() <= peak && incremental.live.gdn == full.live.gdn, "running state refreshed within admitted allocation");
    for (size_t i=0;i<full.kv.size();++i)
        check(incremental.kv[i].k==full.kv[i].k && incremental.kv[i].v==full.kv[i].v &&
              incremental.kv[i].k_scale==full.kv[i].k_scale && incremental.kv[i].v_scale==full.kv[i].v_scale &&
              incremental.kv[i].pooled==full.kv[i].pooled, "incremental and full capture agree across every payload");
    {
        QsaState unloaded;
        SavedConversation meta, without_draft;
        std::vector<SessionKvSource> sources;
        check(!conversation_snapshot_sources(meta,sources,view,ss,g,unloaded,error) &&
              error == "conversation snapshot: invalid K/V extent", "an explicitly supplied empty drafter remains invalid");
        // #4384 (the part kept): a K/V byte estimate that fails says which part and why, instead of a bare "invalid or
        // overflowing K/V byte estimate" - here the draft part, an empty drafter
        size_t empty_estimate = 0;
        error.clear();
        check(!conversation_snapshot_bytes(view,ss,g,&unloaded,empty_estimate,error) &&
              error.find("byte estimate (the draft part") != std::string::npos,
              "#4384: a byte estimate that fails names the part it failed on");
        check(conversation_snapshot_sources(meta,sources,view,ss,g,nullptr,error), "disk save without an MTP drafter");
        check(conversation_snapshot_save(without_draft,view,ss,g,nullptr,error), "capture reference without draft");
        check(meta.kv.empty() && sources.size()==without_draft.kv.size() &&
              sources.size()==(size_t)ss.qsa_alloc, "disk sources contain only initialized session layers");
        for (size_t j=0;j<sources.size();++j) {
            const auto& kv=without_draft.kv[j];
            const std::array<const ConversationBuffer*,5> parts={&kv.k,&kv.v,&kv.k_scale,&kv.v_scale,&kv.pooled};
            for (size_t i=0;i<parts.size();++i) {
                std::vector<uint8_t> actual(sources[j].sizes[i]), expected(parts[i]->size());
                check(sources[j].sizes[i]==parts[i]->size(), "no-draft source part size");
                check(sources[j].read(i,0,actual.data(),actual.size()) &&
                      parts[i]->read(expected.data(),0,expected.size()) && actual==expected,
                      "no-draft disk source matches captured bytes");
            }
        }
        SessionReadLimits limits, with_draft;
        check(conversation_session_read_limits(limits,ss,g,nullptr,96,1,error), "no-draft read limits");
        check(conversation_session_read_limits(with_draft,ss,g,draft.st,96,1,error), "draft read limits unchanged");
        check(limits.max_kv_layers==(uint64_t)ss.qsa_alloc && limits.max_kv_bytes.size()==limits.max_kv_layers &&
              with_draft.max_kv_layers==limits.max_kv_layers+1, "read limits omit exactly the absent draft layer");
        for (size_t j=0;j<limits.max_kv_bytes.size();++j)
            check(limits.max_kv_bytes[j]==with_draft.max_kv_bytes[j], "main layer byte bounds unchanged");
        check(conversation_snapshot_validate(without_draft,ss,g,nullptr,error), "no-draft snapshot validates");
        check(!conversation_snapshot_validate(without_draft,ss,g,draft.st,error) &&
              !conversation_snapshot_validate(full,ss,g,nullptr,error), "draft-presence mismatch rejected in both directions");
        check(conversation_snapshot_restore(without_draft,ss,g,nullptr,error)==ConversationRestore::restored,
              "no-draft restore through host transfer backend");
    }
    reuse = {full.kv,9,9};
    reuse.kv.back().k.pop_back();
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "malformed retained draft rejected before admission");
    check(!copy_calls && !sync_calls, "invalid retained storage performs no CUDA calls");
    reuse = {full.kv,9,9};
    fail_copy = copy_calls + 1;
    SavedConversation unpublished;
    check(!conversation_snapshot_save(unpublished,view,ss,g,draft.st,error,std::move(reuse)), "incremental capture copy failure is reported");
    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
#endif
}

#if defined(CONVERSATION_TEST_TRANSFERS) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
void swap_transfer_failure() {
    ModelGeometry g;
    g.n_layers=4; g.n_expert=256; g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    g.n_head_kv=1; g.head_dim=64; g.idx_key_dim=8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g,z,error),"swap mock state sizes");
    Pools pool(g,1);
    auto& st=pool.st;
    st.kv_mode=1; st.n_slots=4;
    std::vector<uint8_t> mapped(pool.data[0].size()+pool.data[1].size()+pool.data[2].size()+pool.data[3].size(),0x55);
    // This deliberately unmapped device alias is never safe to dereference on
    // the CPU. Only the recorded CPU base may be used for the exchange.
    const uintptr_t alias=static_cast<uintptr_t>(0x100000000ull);
    st.host_cpu_base=mapped.data(); st.host_device_base=reinterpret_cast<uint8_t*>(alias); st.host_allocation_bytes=mapped.size();
    size_t at=0;
    st.host.k_q=reinterpret_cast<int8_t*>(alias+at); at+=pool.data[0].size();
    st.host.v_q=reinterpret_cast<int8_t*>(alias+at); at+=pool.data[1].size();
    st.host.k_scale=reinterpret_cast<uint16_t*>(alias+at); at+=pool.data[2].size();
    st.host.v_scale=reinterpret_cast<uint16_t*>(alias+at);
    st.idx_pooled_allocation=st.idx_pooled; st.idx_pooled_allocation_bytes=pool.data[4].size();
    std::vector<int32_t> page(24), block(4), stamp(4), ref(4), miss_block(4), miss_slot(4), ctl(32);
    st.page_table=page.data(); st.map.page_table=page.data(); st.map.n_blocks=24; st.map.n_slots=4;
    st.map.slot_block=block.data(); st.map.slot_stamp=stamp.data(); st.map.slot_ref=ref.data();
    st.map.miss_block=miss_block.data(); st.map.miss_slot=miss_slot.data(); st.map.ctl=ctl.data();
    std::vector<uint8_t> gdn(z.gdn,0xa5), ple(z.ple,0xa5);
    SessionState ss;
    ss.max_cells=96; ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=1;
    ss.qsa_states=&st; ss.gdn_state=reinterpret_cast<float*>(gdn.data()); ss.ple_hist=reinterpret_cast<float*>(ple.data());
    SavedConversation incoming;
    incoming.geometry={g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                       g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                       g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    incoming.layer_hi=g.n_layers; incoming.live.ids={21,22,23,24,25,26,27,28,29};
    incoming.live.gdn.resize(z.gdn,13); incoming.live.ple.resize(z.ple,13);
    incoming.live.tails.resize(z.tail,13); incoming.live.dead.resize(z.dead,13); incoming.live.block_pos.resize(z.block_pos,13);
    incoming.kv.push_back(pool.image(g,true));
    const auto original=incoming;
    const std::vector<int32_t> active_ids{1,2,3,4,5,6,7,8,9};
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> checkpoints;
    const ConversationView active{active_ids,images,checkpoints,true};
    ConversationSwapPlan plan;
    constexpr size_t scratch=7;
    copy_calls=sync_calls=fail_copy=fail_sync=0;
    check(st.host_cpu_base!=st.host_device_base,"mock CPU mapping and device alias differ");
    const size_t disabled=conversation_cache_swap_block_bytes(conversation_cache_swap_default_block_mib);
    check(disabled==0 && !conversation_snapshot_swap_prepare(plan,incoming,active,ss,g,nullptr,disabled,error) &&
          copy_calls==0 && sync_calls==0 && incoming.kv[0].k==original.kv[0].k && mapped[0]==0x55,
          "default zero leaves both sessions untouched for the old path");
    {
        // K needs a new segment as well as spare capacity; V grows entirely
        // within spare capacity that may never have been resident.
        SavedConversation longer_incoming = incoming;
        std::array<ConversationBuffer*,4> parts{&longer_incoming.kv[0].k, &longer_incoming.kv[0].v,
                                                &longer_incoming.kv[0].k_scale, &longer_incoming.kv[0].v_scale};
        for (size_t i = 0; i < 2; ++i) {
            const size_t n = parts[i]->size();
            ConversationBuffer with_spare;
            with_spare.resize_for_overwrite(n + (i == 0 ? 100 : 1024));
            check(with_spare.visit(0,n,[](uint8_t* p, size_t count, size_t) {
                std::memset(p,13,count); return true;
            }), "initialize only B's used bytes, not its spare");
            with_spare.resize_for_overwrite(n);
            *parts[i] = std::move(with_spare);
        }
        const SavedConversation before_b = longer_incoming;
        const auto before_mapped = mapped, before_gdn = gdn, before_ple = ple;
        const auto before_pool = pool.data;
        const auto before_page = page, before_block = block, before_stamp = stamp, before_ref = ref;
        const auto before_miss_block = miss_block, before_miss_slot = miss_slot, before_ctl = ctl;
        const std::array<int32_t,2> before_prev{ss.ple_prev[0], ss.ple_prev[1]};
        std::array<size_t,4> old_bytes{}, old_spare{};
        for (size_t i = 0; i < parts.size(); ++i) {
            old_bytes[i] = parts[i]->bytes();
            size_t capacity = 0;
            check(parts[i]->visit_allocations([&](const uint8_t*, size_t n, size_t) { capacity += n; return true; }),
                  "measure parked capacity before growth");
            old_spare[i] = capacity - parts[i]->size();
        }
        const std::vector<int32_t> longer_ids{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
        const ConversationView longer_active{longer_ids,images,checkpoints,true};
        ConversationSwapPlan longer_plan;
        check(conversation_snapshot_swap_prepare(longer_plan,longer_incoming,longer_active,ss,g,nullptr,scratch,error),
              "prepare longer A for the final RAM recheck");
        constexpr uint64_t floor = 1024;
        const uint64_t initial_available = floor + longer_plan.additional_bytes + (1u << 20);
        check(conversation_memory_admit(initial_available,longer_plan.additional_bytes,floor),
              "initial admission succeeds with headroom");
        ConversationCache cache(1u << 22,2);
        SavedConversation unrelated;
        unrelated.live.ids = {99}; unrelated.live.gdn.resize(64,3);
        check(cache.put(std::move(unrelated)),"park an unrelated cache entry");
        check(cache.can_fit(longer_plan.image_bytes,longer_plan.held_bytes),"initial cache budget also fits");
        const size_t cache_bytes = cache.bytes();
        bool rechecked = false;
        auto recheck = [&](size_t image_bytes, size_t held_bytes) {
            rechecked = true;
            size_t shared = longer_incoming.kv.capacity() * sizeof(ConversationKv);
            size_t growth = 0;
            for (size_t i = 0; i < parts.size(); ++i) {
                shared += parts[i]->bytes();
                check(parts[i]->bytes() >= old_bytes[i],"reserved K/V capacity does not shrink");
                growth += parts[i]->bytes() - old_bytes[i];
                if (longer_plan.layers[0].active_bytes[i] > longer_plan.layers[0].incoming_bytes[i])
                    growth += old_spare[i];
            }
            check(parts[0]->bytes() > old_bytes[0] && parts[1]->bytes() == old_bytes[1] &&
                  old_spare[0] >= 100 && old_spare[1] >= 1024,
                  "recheck covers both new segments and previously reserved spare");
            check(shared <= image_bytes && shared + held_bytes == longer_incoming.bytes() &&
                  cache.can_fit(image_bytes,held_bytes),
                  "cache budget counts the shared B storage exactly once");
            const auto& pooled = longer_plan.layers[0].pooled;
            const size_t expected = scratch + growth + pooled.bytes() - pooled.size();
            check(longer_plan.final_additional_bytes == expected && growth > 0,
                  "final RAM charge includes scratch, growth and unused index capacity");
            const uint64_t available = floor + expected - 1;
            check(available < initial_available && conversation_memory_admit(available,0,floor),
                  "headroom shrank but an admit-zero recheck would incorrectly succeed");
            return conversation_memory_admit(available,longer_plan.final_additional_bytes,floor);
        };
        SavedConversation unpublished;
        check(conversation_snapshot_swap(longer_plan,longer_incoming,unpublished,longer_active,ss,g,scratch,recheck,error) ==
                  ConversationSwap::unavailable && rechecked,
              "reduced headroom falls back before the first exchange write");
        const auto& after = longer_incoming.kv[0];
        const auto& before = before_b.kv[0];
        check(after.k == before.k && after.v == before.v && after.k_scale == before.k_scale &&
              after.v_scale == before.v_scale && after.pooled == before.pooled &&
              after.format == before.format && after.cells == before.cells && after.heads == before.heads &&
              after.head_dim == before.head_dim && after.page_size == before.page_size &&
              after.pooled_rows == before.pooled_rows && after.idx_dim == before.idx_dim && after.dev_tag == before.dev_tag &&
              longer_incoming.geometry == before_b.geometry && longer_incoming.cvec == before_b.cvec &&
              longer_incoming.layer_lo == before_b.layer_lo && longer_incoming.layer_hi == before_b.layer_hi &&
              longer_incoming.checkpoints.empty() && longer_incoming.stage_images.empty() &&
              longer_incoming.live.ids == before_b.live.ids && longer_incoming.live.imgs == before_b.live.imgs &&
              longer_incoming.live.gdn == before_b.live.gdn && longer_incoming.live.ple == before_b.live.ple &&
              longer_incoming.live.tails == before_b.live.tails && longer_incoming.live.dead == before_b.live.dead &&
              longer_incoming.live.block_pos == before_b.live.block_pos,
              "RAM refusal leaves all parked B bytes and state unchanged");
        check(mapped == before_mapped && pool.data == before_pool && gdn == before_gdn && ple == before_ple &&
              page == before_page && block == before_block && stamp == before_stamp && ref == before_ref &&
              miss_block == before_miss_block && miss_slot == before_miss_slot && ctl == before_ctl &&
              ss.ple_prev[0] == before_prev[0] && ss.ple_prev[1] == before_prev[1],
              "RAM refusal leaves active A bytes, state and device indices unchanged");
        check(unpublished.kv.empty() && unpublished.live.ids.empty() && cache.bytes() == cache_bytes &&
              cache.size() == 1 && cache.evictions() == 0 && cache.can_fit(0,longer_incoming.bytes()),
              "rollback publishes nothing, keeps other entries, and accounts for B's actual retained capacity");
        SavedConversation other;
        other.live.ids = {99}; other.live.gdn.resize(64,3);
        ConversationCache exact(other.bytes() + longer_incoming.bytes() + 1,2);
        check(exact.put(std::move(other)) && exact.can_fit(1,longer_incoming.bytes()) &&
              !exact.can_fit(2,longer_incoming.bytes()),
              "fallback budget charges B's post-rollback capacity exactly once");
    }
    auto* cpu_base=st.host_cpu_base;
    st.host_cpu_base=nullptr;
    check(!conversation_snapshot_swap_prepare(plan,incoming,active,ss,g,nullptr,scratch,error),
          "unverifiable CPU mapping uses the old path");
    st.host_cpu_base=cpu_base;
    incoming.kv[0].dev_tag=1;
    check(!conversation_snapshot_swap_prepare(plan,incoming,active,ss,g,nullptr,scratch,error),
          "device-tagged image uses the old path");
    incoming.kv[0].dev_tag=0;
    incoming.stage_images.emplace_back();
    check(!conversation_snapshot_swap_prepare(plan,incoming,active,ss,g,nullptr,scratch,error),
          "split image uses the old path");
    incoming.stage_images.clear();
    check(conversation_snapshot_swap_prepare(plan,incoming,active,ss,g,nullptr,scratch,error),"prepare distinct-alias exchange");
    SavedConversation outgoing;
    check(conversation_snapshot_swap(plan,incoming,outgoing,active,ss,g,scratch,[](size_t,size_t){return false;},error)==
              ConversationSwap::unavailable,
          "pre-write admission failure permits the old path");
    check(incoming.kv[0].k==original.kv[0].k && mapped[0]==0x55,"pre-write failure preserves both K/V images");
    check(conversation_snapshot_swap_prepare(plan,incoming,active,ss,g,nullptr,scratch,error),"prepare injected transfer failure");
    auto fail_next=[&](size_t,size_t) { fail_copy=copy_calls+1; return true; };
    check(conversation_snapshot_swap(plan,incoming,outgoing,active,ss,g,scratch,fail_next,error)==ConversationSwap::transfer_failed,
          "device-index copy failure after first exchanged byte is fatal");
    uint8_t parked=0;
    check(incoming.kv[0].k.read(&parked,0,1) && mapped[0]==13 && parked==0x55,
          "exchange used the real CPU mapping before the injected post-write failure");
    fail_copy=0;
}
#endif
}

int main() {
#if defined(CONVERSATION_TEST_TRANSFERS) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    swap_transfer_failure();
#endif
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
