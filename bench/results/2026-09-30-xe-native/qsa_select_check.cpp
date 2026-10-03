// qsa_block_scores against FP64, and qsa_block_topk / qsa_block_topk_ref against a host transcription of the
// selection rule applied to the SAME device scores: every cell of a block whose order key is above the threshold,
// then the lowest-index cells of blocks at the threshold up to the width, written ascending.  All but about 300
// pooled keys are zero, so the threshold falls among equal (zero) scores.  qsa_block_scores_tc must refuse.
#include "probe_common.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>

namespace k = strata::kernels;

namespace {

constexpr int D = 128, HEADS = 4, R = 4;

uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

std::vector<int32_t> host_topk(const float* sc, int64_t n_kv, int64_t n_bid, int64_t width) {
    std::vector<int32_t> out;
    if (n_kv <= width) {
        for (int64_t j = 0; j < n_kv; ++j) out.push_back((int32_t) j);
        return out;
    }
    auto weight = [&](int64_t b) -> int64_t { return b < n_bid ? R : n_kv - n_bid * R; };
    // the largest key thr with (cells whose key >= thr) >= width
    std::vector<std::pair<uint32_t, int64_t>> keys;
    for (int64_t b = 0; b <= n_bid; ++b)
        if (weight(b) > 0) keys.push_back({order_key(sc[b]), b});
    std::vector<std::pair<uint32_t, int64_t>> sorted = keys;
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) { return a.first > b.first; });
    int64_t cum = 0;
    uint32_t thr = 0;
    for (auto& [key, b] : sorted) {
        cum += weight(b);
        if (cum >= width) { thr = key; break; }
    }
    int64_t above = 0;
    for (auto& [key, b] : keys) if (key > thr) above += weight(b);
    int64_t eq_left = width - above;
    for (int64_t b = 0; b <= n_bid; ++b) {
        const int64_t w = weight(b);
        if (w == 0) continue;
        const uint32_t key = order_key(sc[b]);
        if (key > thr) {
            for (int64_t c = 0; c < w; ++c) out.push_back((int32_t) (b * R + c));
        } else if (key == thr) {
            for (int64_t c = 0; c < w && eq_left > 0; ++c, --eq_left) out.push_back((int32_t) (b * R + c));
        }
    }
    return out;
}

}  // namespace

int main() {
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    int bad = 0;
    // contexts: the identity, the 2,051 boundary, mid, long (register top-k), longer than the register path covers
    for (int64_t ctx : {1000L, 2052L, 9001L, 60001L, 140001L}) {
        const int64_t nq = 6, max_blocks = ctx / R + 2;
        auto pooled = probe::normal((size_t) max_blocks * D, 50, 0.3f);
        // about 300 random keys, every other block the zero key: the zero-score blocks tie, and the width of 2,051
        // cells reaches into them, so the threshold is a tie that the lowest-index rule has to break
        const int64_t keep = std::max<int64_t>(1, max_blocks / 300);
        for (int64_t b = 0; b < max_blocks; ++b)
            if (b % keep != 0) std::fill(pooled.begin() + b * D, pooled.begin() + (b + 1) * D, 0.0f);
        const auto dead = probe::normal(D, 51, 0.3f);
        const auto q = probe::normal((size_t) nq * HEADS * D, 52, 0.3f);
        std::vector<int32_t> steps((size_t) nq * k::kStepCount);
        for (int64_t i = 0; i < nq; ++i) k::qsa_step_fill(steps.data() + i * k::kStepCount, ctx - nq + i, s);

        float* d_p = probe::upload(pooled);
        float* d_dead = probe::upload(dead);
        float* d_q = probe::upload(q);
        int32_t* d_steps = probe::upload(steps);
        float* d_sc = static_cast<float*>(strata::parity::alloc_bytes((size_t) nq * max_blocks * 4));
        int32_t* d_ids = static_cast<int32_t*>(strata::parity::alloc_bytes((size_t) nq * cap * 4));
        int32_t* d_ids_ref = static_cast<int32_t*>(strata::parity::alloc_bytes((size_t) nq * cap * 4));
        const bool tc = k::qsa_block_scores_tc(d_p, d_dead, d_q, d_steps, nq, max_blocks, s, d_sc, strata::parity::stream(), -1);
        k::qsa_block_scores(d_p, d_dead, d_q, d_steps, nq, max_blocks, s, d_sc, strata::parity::stream());
        k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_ids, strata::parity::stream());
        k::qsa_block_topk_ref(d_sc, d_steps, nq, max_blocks, cap, s, d_ids_ref, strata::parity::stream());
        strata::parity::sync();
        const auto sc = probe::download(d_sc, (size_t) nq * max_blocks);
        const auto ids = probe::download(d_ids, (size_t) nq * cap);
        const auto ids_ref = probe::download(d_ids_ref, (size_t) nq * cap);

        double worst = 0, scale = 0;
        int64_t id_bad = 0, id_ref_bad = 0, tie_queries = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* st = steps.data() + i * k::kStepCount;
            const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid], width = st[k::kStepWidth];
            for (int64_t b = 0; b <= n_bid; ++b) {
                const float* key = b == n_bid ? dead.data() : pooled.data() + b * D;
                double score = 0;
                for (int h = 0; h < HEADS; ++h) {
                    double d = 0;
                    for (int j = 0; j < D; ++j) d += (double) key[j] * q[(i * HEADS + h) * D + j];
                    score += std::fmax(d, 0.0);
                }
                if (b == n_bid && n_kv % R) continue;   // carries the +1e9 mask
                scale = std::fmax(scale, std::fabs(score));
                worst = std::fmax(worst, std::fabs(score - sc[i * max_blocks + b]));
            }
            const auto want = host_topk(sc.data() + i * max_blocks, n_kv, n_bid, width);
            // a threshold tie: the last selected block's key also occurs in an unselected block
            for (size_t j = 0; j < want.size(); ++j)
                if (ids[i * cap + j] != want[j]) { ++id_bad; break; }
            for (size_t j = 0; j < want.size(); ++j)
                if (ids_ref[i * cap + j] != want[j]) { ++id_ref_bad; break; }
            if (n_kv > width) {
                std::vector<char> chosen((size_t) n_bid + 1, 0);
                for (int32_t c : want) chosen[c / R] = 1;
                uint32_t thr = UINT32_MAX;
                for (int64_t b = 0; b <= n_bid; ++b) if (chosen[b]) thr = std::min(thr, order_key(sc[i * max_blocks + b]));
                for (int64_t b = 0; b <= n_bid; ++b)
                    if (!chosen[b] && order_key(sc[i * max_blocks + b]) == thr) { ++tie_queries; break; }
            }
        }
        std::printf("  ctx %-7lld scores vs FP64 %.3e (rel to scale %.3e); topk %lld/%lld queries differ, topk_ref "
                    "%lld/%lld differ; queries with a threshold tie %lld; tc refused %s\n", (long long) ctx,
                    worst / scale, scale, (long long) id_bad, (long long) nq, (long long) id_ref_bad, (long long) nq,
                    (long long) tie_queries, tc ? "*** NO ***" : "yes");
        if (!(worst / scale <= 1e-5) || id_bad || id_ref_bad || tc) ++bad;
    }
    std::printf("qsa_select_check: %d failures (criteria: scores within 1e-5 of FP64 relative to scale; both top-k "
                "kernels equal to the host selection rule on the device scores; the TF32 scorer refuses)\n", bad);
    return bad ? 1 : 0;
}
