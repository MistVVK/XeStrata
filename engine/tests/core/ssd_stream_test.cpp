// SPDX-License-Identifier: MIT
// tests/core/ssd_stream_test.cpp - unit tests for 3-tier NVMe streaming primitives:
//   1. TwoPriorityExpertScheduler: q_demand preempts 512 KiB chunked q_prefetch, late slot binding,
//      promote_to_demand deduplication, drop_stale cancellation, and proj_ready[3] overlap + slot-reuse reset
//   2. MoeWavePlan: pe128 prompt-end recency scoring, Order A wave planning, resident pinning, <= 1.0 reads/expert
//   3. VramCopybackQueue + IdlePreloader (<100us active yield)
//   4. RouterLookahead v2: shared-expert residual input, 2-layer L+1/L+2 prediction, zero-drop latest mailbox

#include "strata/core/moe_wave_plan.hpp"
#include "strata/core/router_lookahead_v2.hpp"
#include "strata/core/ssd_stream_scheduler.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "ssd_stream_test FAIL at line %d: %s\n", __LINE__, #cond); \
        std::abort(); \
    } \
} while (0)

namespace {

using namespace strata::core;

void test_proj_ready_overlap_and_slot_reuse() {
    constexpr size_t kBlob = 1536 * 1024;  // 1.5 MiB: gate+up = 1.0 MiB, down = 512 KiB
    constexpr size_t kDnOff = 1024 * 1024;
    std::vector<uint8_t> disk(kBlob * 4);
    for (size_t i = 0; i < disk.size(); ++i) disk[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFFu);
    std::vector<uint8_t> slot_mem(kBlob * 2, 0);

    std::atomic<bool> gate_up_observed_before_down{false};
    std::atomic<bool> allow_down_finish{false};

    TwoPriorityExpertScheduler sched;
    sched.start(
        1, 2,
        [&](int /*file_idx*/, uint64_t off, uint8_t* dst, size_t len) -> bool {
            // Delay the down projection slightly so compute can prove gate+up became ready first!
            if (off % kBlob >= kDnOff) {
                while (!allow_down_finish.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            }
            std::memcpy(dst, disk.data() + off, len);
            return true;
        });

    // First load into slot 0
    auto req1 = StreamExpertReq::make_contiguous_split(
        10, 0, 1, 0, slot_mem.data(), 0, 0, kBlob, kDnOff);
    sched.enqueue_demand(req1);

    // Wait for gate+up while down is still blocked -> MUST succeed immediately!
    CHECK(sched.slot_state(0)->wait_gate_up());
    CHECK(sched.slot_state(0)->proj_ready[2].load(std::memory_order_acquire) == kProjPending);
    gate_up_observed_before_down.store(true);
    allow_down_finish.store(true, std::memory_order_release);
    CHECK(sched.slot_state(0)->wait_down());
    CHECK(std::memcmp(slot_mem.data(), disk.data(), kBlob) == 0);
    CHECK(gate_up_observed_before_down.load());

    // CRITICAL SLOT-REUSE INVARIANT: enqueueing a NEW expert into slot 0 MUST reset proj_ready[0..2]
    // to kProjPending BEFORE the queue push so compute never observes stale kProjReady!
    allow_down_finish.store(false, std::memory_order_release);
    auto req2 = StreamExpertReq::make_contiguous_split(
        11, 1, 1, 0, slot_mem.data(), 0, kBlob, kBlob, kDnOff);
    sched.enqueue_demand(req2);
    CHECK(sched.slot_state(0)->wait_gate_up());
    CHECK(sched.slot_state(0)->proj_ready[2].load(std::memory_order_acquire) == kProjPending);
    allow_down_finish.store(true, std::memory_order_release);
    CHECK(sched.slot_state(0)->wait_down());
    CHECK(std::memcmp(slot_mem.data(), disk.data() + kBlob, kBlob) == 0);

    sched.stop();
    std::printf("ssd_stream_test: proj_ready[3] overlap + slot-reuse reset PASS\n");
}

void test_demand_preempts_chunked_prefetch_and_drop_stale() {
    constexpr size_t kBlob = 4 * kPrefetchChunkBytes;  // 2 MiB = 4 chunks of 512 KiB
    std::vector<uint8_t> slot_mem(4 * kBlob, 0);
    std::atomic<int> next_pf_slot{1};
    std::atomic<bool> demand_finished_before_prefetch{false};
    std::atomic<int> pf_chunks_done{0};

    TwoPriorityExpertScheduler sched;
    sched.start(
        1, 4,
        [&](int /*file_idx*/, uint64_t off, uint8_t* dst, size_t len) -> bool {
            std::memset(dst, static_cast<int>((off >> 20) + 1), len);
            if (len == kPrefetchChunkBytes) {
                pf_chunks_done.fetch_add(1, std::memory_order_release);
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
            return true;
        },
        [&](const StreamExpertReq& /*req*/, int& out_slot, uint8_t*& out_dst) -> bool {
            int s = next_pf_slot.fetch_add(1);
            if (s >= 4) return false;
            out_slot = s;
            out_dst = slot_mem.data() + static_cast<size_t>(s) * kBlob;
            return true;
        });

    // Queue a 2 MiB late-bound prefetch for layer 5, plus a stale one for layer 1
    auto pf5 = StreamExpertReq::make_contiguous_split(500, 5, 1, -1, nullptr, 0, 0, kBlob, 0);
    auto pf1 = StreamExpertReq::make_contiguous_split(100, 1, 1, -1, nullptr, 0, kBlob, kBlob, 0);
    CHECK(sched.enqueue_prefetch(pf5));
    CHECK(sched.enqueue_prefetch(pf1));
    CHECK(!sched.enqueue_prefetch(pf5));  // deduplicated!

    // Wait for first 512 KiB chunk of pf5 to start, then inject a demand read on slot 0!
    while (pf_chunks_done.load(std::memory_order_acquire) < 1) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    auto dem = StreamExpertReq::make_contiguous_split(200, 2, 1, 0, slot_mem.data(), 0, 2 * kBlob, kBlob, 0);
    sched.enqueue_demand(dem);
    sched.wait_all_demand();
    // Demand MUST have finished BEFORE all 4 chunks of pf5 completed (preempted at chunk boundary!)
    demand_finished_before_prefetch.store(pf_chunks_done.load(std::memory_order_acquire) < 4);
    CHECK(demand_finished_before_prefetch.load());

    // drop_stale(layer=3, gen=1) must drop or cancel layer 1 prefetch
    sched.drop_stale(3, 1);
    sched.wait_all();

    const auto st = sched.stats();
    CHECK(st.demand_completed == 1);
    CHECK(st.prefetch_preempted >= 1);
    CHECK(st.prefetch_dropped >= 1);
    sched.stop();
    std::printf("ssd_stream_test: demand preempts 512 KiB prefetch + drop_stale PASS (preempted=%llu, dropped=%llu)\n",
                (unsigned long long) st.prefetch_preempted, (unsigned long long) st.prefetch_dropped);
}

void test_promote_to_demand() {
    constexpr size_t kBlob = 64 * 1024;
    std::vector<uint8_t> slot_mem(2 * kBlob, 0);
    std::atomic<int> read_calls{0};
    std::atomic<bool> hold{true};

    TwoPriorityExpertScheduler sched;
    sched.start(
        1, 2,
        [&](int, uint64_t, uint8_t* dst, size_t len) -> bool {
            while (hold.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            read_calls.fetch_add(1, std::memory_order_relaxed);
            std::memset(dst, 0x5A, len);
            return true;
        });

    // Block worker on job A, enqueue prefetch B, then promote B to demand!
    auto jobA = StreamExpertReq::make_contiguous_split(1, 0, 1, 0, slot_mem.data(), 0, 0, kBlob, 0);
    auto jobB = StreamExpertReq::make_contiguous_split(2, 1, 1, -1, nullptr, 0, kBlob, kBlob, 0);
    sched.enqueue_demand(jobA);
    CHECK(sched.enqueue_prefetch(jobB));
    CHECK(sched.promote_to_demand(2, 1, slot_mem.data() + kBlob));
    hold.store(false, std::memory_order_release);
    sched.wait_all();

    CHECK(read_calls.load() == 2);  // NO duplicate read for promoted key 2!
    CHECK(slot_mem[kBlob] == 0x5A);
    CHECK(sched.stats().prefetch_promoted == 1);
    sched.stop();
    std::printf("ssd_stream_test: promote_to_demand deduplication PASS\n");
}

void test_moe_wave_plan_pe128() {
    // Prompt of T=512 tokens, top_k=2, 32 experts; cache S=12 slots, 4 already resident.
    constexpr int T = 512, K = 2, E = 32;
    std::vector<int32_t> ids(T * K);
    for (int t = 0; t < T; ++t) {
        // Early tokens use low expert IDs; prompt-end tokens concentrate on experts 28..31!
        ids[t * K + 0] = (t < 384) ? (t % 20) : (28 + (t % 4));
        ids[t * K + 1] = (t < 384) ? ((t * 3) % 24) : (28 + ((t + 1) % 4));
    }
    const auto w_pe = compute_pe128_scores(ids.data(), T, K, E);
    CHECK(w_pe[28] > w_pe[0] * 2.0f);
    CHECK(w_pe[31] > w_pe[5] * 2.0f);

    std::vector<int32_t> active;
    for (int e = 0; e < E; ++e) if (w_pe[e] > 0.0f) active.push_back(e);
    std::vector<uint8_t> res(E, 0);
    res[28] = 1; res[29] = 1; res[0] = 1; res[1] = 1;

    const auto plan = plan_moe_waves(active, res, w_pe, 12);
    CHECK(plan.wave0_residents.size() == 4);
    CHECK(plan.wave0_residents[0] == 28 || plan.wave0_residents[0] == 29);
    CHECK(plan.reads_per_expert() < 1.0);
    // Hottest misses (30, 31) MUST appear in the FINAL cold wave so they stay resident for decode token 1!
    CHECK(!plan.cold_waves.empty());
    const auto& last_wave = plan.cold_waves.back();
    bool has30 = std::find(last_wave.begin(), last_wave.end(), 30) != last_wave.end();
    bool has31 = std::find(last_wave.begin(), last_wave.end(), 31) != last_wave.end();
    CHECK(has30 && has31);
    // Every miss read at most once across all waves
    std::vector<int32_t> seen;
    for (const auto& w : plan.cold_waves) seen.insert(seen.end(), w.begin(), w.end());
    CHECK(seen.size() == plan.total_misses);
    std::sort(seen.begin(), seen.end());
    CHECK(std::unique(seen.begin(), seen.end()) == seen.end());
    std::printf("ssd_stream_test: moe_wave_plan pe128 PASS (waves=%zu, pinned=%zu, reads/exp=%.3f)\n",
                plan.cold_waves.size(), plan.pinned_residents.size(), plan.reads_per_expert());
}

void test_copyback_and_idle_preloader() {
    std::vector<uint8_t> dev(1024, 0xAB), host(1024, 0);
    VramCopybackQueue cb(8);
    cb.start([](const VramCopybackQueue::Job& j) {
        std::memcpy(j.host_dst, j.dev_src, j.bytes);
        return true;
    });
    CHECK(cb.try_enqueue({42, 3, 7, dev.data(), host.data(), dev.size()}));
    cb.drain();
    CHECK(cb.completed() == 1);
    CHECK(host[0] == 0xAB && host[1023] == 0xAB);
    cb.stop();

    IdlePreloader idle;
    std::atomic<int> steps{0};
    idle.start([&](const std::atomic<bool>& should_yield) -> bool {
        if (should_yield.load(std::memory_order_acquire)) return false;
        return steps.fetch_add(1) < 5;
    }, 10);  // 10 ms threshold for fast test
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    idle.notify_active();
    CHECK(idle.steps_done() >= 5);
    idle.stop();
    std::printf("ssd_stream_test: VramCopybackQueue + IdlePreloader PASS\n");
}

void test_router_lookahead_v2() {
    constexpr int D = 16, E = 8;
    float x_attn[D], sh[D], x_pred[D];
    for (int i = 0; i < D; ++i) { x_attn[i] = 0.1f * i; sh[i] = (i == 3) ? 5.0f : 0.0f; }
    build_lookahead_residual_input(x_attn, sh, x_pred, D, 1.0f);
    CHECK(std::fabs(x_pred[3] - 5.3f) < 1e-5f);

    std::vector<float> rw(E * D, 0.0f);
    // Expert 6 aligns strongly with dim 3 (shared expert signal!)
    rw[6 * D + 3] = 2.0f;
    rw[2 * D + 3] = 1.0f;
    auto pred = predict_router_topk_f32(11, 2, x_pred, rw.data(), nullptr, D, E, 2, false);
    CHECK(pred.layer == 11 && pred.ahead == 2);
    CHECK(pred.experts.size() == 2);
    CHECK(pred.experts[0] == 6 && pred.experts[1] == 2);
    CHECK(pred.confidences[0] > pred.confidences[1]);

    LatestLookaheadMailbox<int> mb;
    mb.submit(1);
    mb.submit(2);
    mb.submit(3);
    int v = 0;
    CHECK(mb.take(v) && v == 3);
    CHECK(!mb.take(v));
    CHECK(mb.submitted() == 3 && mb.overwritten() == 2);
    std::printf("ssd_stream_test: router_lookahead_v2 PASS\n");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    test_proj_ready_overlap_and_slot_reuse();
    test_demand_preempts_chunked_prefetch_and_drop_stale();
    test_promote_to_demand();
    test_moe_wave_plan_pe128();
    test_copyback_and_idle_preloader();
    test_router_lookahead_v2();
    std::printf("ssd_stream_test: ALL 6 SUITES PASSED\n");
    return 0;
}
