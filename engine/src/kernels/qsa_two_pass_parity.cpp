// STRATA_SELECT_TWO_PASS (qsa_two_pass.hpp) against today's prompt scorer, GPU, synthetic, no model: for prompt
// chunks of 256 queries at depths from 2K to 256K and several kinds of keys (normal, indexer-like, wide magnitudes,
// small integers with exact ties, near-duplicate and duplicate keys), qsa_block_topk over the two-pass row must return
// exactly the ids it returns over qsa_block_scores_tc's row, every scored-again block must hold today's score bit for
// bit, and every kept single-pass score must lie inside its bound (qsa_two_pass_check's counters).  Then, for the
// larger cases, the time of each scorer (one synthetic run, not a benchmark).  Needs CUDA sm_80+ (77 = skipped
// otherwise, and on HIP).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_two_pass.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace tp = strata::kernels::two_pass;

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}

enum Kind { kNormal, kIndexer, kWide, kSmall, kNear, kDup, kKinds };
const char* kKindName[kKinds] = {"normal", "indexer-like", "wide magnitudes", "small integers", "near-duplicate keys",
                                 "duplicate keys"};

struct Case { int64_t ctx, queries; Kind kind; bool time; };

// stride: the score row's stride in blocks; 0 = the case's own (ctx / 4 + 2), else the engine's (--max-context 262144:
// 65,538, so qsa_block_topk takes the engine's dispatch on this card)
bool run(const Case& c, int64_t stride) {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head, own = c.ctx / s.idx_block + 2;
    const int64_t max_blocks = stride > own ? stride : own;
    const int64_t nq = c.queries, nq_cap = 256, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(77001 + (unsigned) c.ctx + 13 * (unsigned) c.queries + 1009 * (unsigned) c.kind);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> small(-3, 3), expo(-16, 16), coin(0, 9);
    std::vector<float> pooled((size_t) (max_blocks * D)), dead((size_t) D), q((size_t) (nq * H * D));
    auto unit = [&](float* v, int n, float norm) {
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += (double) v[i] * v[i];
        const float f = ss > 0 ? (float) (norm / std::sqrt(ss)) : 0.f;
        for (int i = 0; i < n; ++i) v[i] *= f;
    };
    for (int64_t b = 0; b < max_blocks; ++b) {
        float* kb = pooled.data() + b * D;
        if (c.kind == kIndexer) {   // the mean of 4 RMS-normed cell keys, as the indexer pools them
            for (int i = 0; i < D; ++i) kb[i] = 0.f;
            for (int cell = 0; cell < 4; ++cell) {
                float ck_[128];
                for (int i = 0; i < D; ++i) ck_[i] = nd(rng);
                unit(ck_, (int) D, std::sqrt((float) D));
                for (int i = 0; i < D; ++i) kb[i] += 0.25f * ck_[i];
            }
        } else if (c.kind == kNear && b > 0 && coin(rng) < 5) {   // half the keys a hair from the one before
            for (int i = 0; i < D; ++i) kb[i] = kb[i - D] * (1.0f + 1e-6f * nd(rng));
        } else if (c.kind == kDup && b >= 8 && coin(rng) < 5) {   // half the keys copies of an earlier one: exact ties
            const int64_t src = (int64_t) (rng() % (uint32_t) b);
            std::memcpy(kb, pooled.data() + src * D, (size_t) D * 4);
        } else {
            for (int i = 0; i < D; ++i) {
                if (c.kind == kWide) kb[i] = coin(rng) == 0 ? 0.f : std::ldexp(nd(rng), expo(rng));
                else if (c.kind == kSmall) kb[i] = (float) small(rng);
                else kb[i] = nd(rng);
            }
        }
    }
    for (auto& v : dead) v = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int64_t h = 0; h < H; ++h) {
            float* qh = q.data() + (i * H + h) * D;
            if (c.kind == kIndexer) {   // RMS-normed, leaning towards a few keys (a peaked score row)
                for (int j = 0; j < D; ++j) qh[j] = nd(rng);
                for (int r = 0; r < 3; ++r) {
                    const float* kb = pooled.data() + (int64_t) (rng() % (uint32_t) (max_blocks - 2)) * D;
                    for (int j = 0; j < D; ++j) qh[j] += 0.5f * kb[j];
                }
                unit(qh, (int) D, std::sqrt((float) D));
            } else {
                for (int j = 0; j < D; ++j) {
                    if (c.kind == kWide) qh[j] = coin(rng) == 0 ? 0.f : std::ldexp(nd(rng), expo(rng));
                    else if (c.kind == kSmall) qh[j] = (float) small(rng);
                    else qh[j] = nd(rng);
                }
            }
        }
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = c.ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const int64_t last_nbid = steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)], active = last_nbid + 1;
    float *dp, *dd, *dq, *ref, *two;
    int32_t *dt, *ref_ids, *two_ids;
    void* scratch;
    const uint64_t sbytes = k::qsa_two_pass_scratch_bytes(nq_cap, max_blocks);
    const size_t n_sc = (size_t) (nq * max_blocks), n_id = (size_t) (nq * cap);
    ck(cudaMalloc(&dp, pooled.size() * 4)); ck(cudaMalloc(&dd, dead.size() * 4)); ck(cudaMalloc(&dq, q.size() * 4));
    ck(cudaMalloc(&dt, steps.size() * 4)); ck(cudaMalloc(&ref, n_sc * 4)); ck(cudaMalloc(&two, n_sc * 4));
    ck(cudaMalloc(&ref_ids, n_id * 4)); ck(cudaMalloc(&two_ids, n_id * 4)); ck(cudaMalloc(&scratch, sbytes));
    ck(cudaMemcpy(dp, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dd, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemset(ref, 0x7f, n_sc * 4)); ck(cudaMemset(two, 0x3e, n_sc * 4));
    ck(cudaMemset(ref_ids, 0xff, n_id * 4)); ck(cudaMemset(two_ids, 0xee, n_id * 4));
    if (!k::qsa_block_scores_tc(dp, dd, dq, dt, nq, max_blocks, s, ref, nullptr, active)) {
        std::puts("SKIP: no tensor-core scorer on this card");
        std::exit(77);
    }
    k::qsa_block_topk(ref, dt, nq, max_blocks, cap, s, ref_ids, nullptr, active);
    unsigned long long st0[tp::kStatCount];
    k::qsa_two_pass_stats(st0, true, nullptr);   // zero the counters
    if (!k::qsa_two_pass_keys(dp, last_nbid, nq_cap, max_blocks, scratch, sbytes, nullptr) ||
        !k::qsa_block_scores_two_pass(dp, dd, dq, dt, nq, nq_cap, max_blocks, s, two, scratch, sbytes, nullptr, active)) {
        std::fprintf(stderr, "FAIL: the two-pass scorer did not run (ctx %lld)\n", (long long) c.ctx);
        return false;
    }
    k::qsa_block_topk(two, dt, nq, max_blocks, cap, s, two_ids, nullptr, active);
    k::qsa_two_pass_check(two, two_ids, ref, ref_ids, dt, nq, nq_cap, max_blocks, cap, scratch, nullptr);
    ck(cudaDeviceSynchronize());
    unsigned long long st[tp::kStatCount];
    if (!k::qsa_two_pass_stats(st, true, nullptr)) { std::fprintf(stderr, "FAIL: counters\n"); return false; }
    // the ids on the host too, every entry qsa_block_topk wrote
    std::vector<int32_t> a(n_id), b(n_id);
    ck(cudaMemcpy(a.data(), ref_ids, n_id * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), two_ids, n_id * 4, cudaMemcpyDeviceToHost));
    int64_t bad_q = 0, compared = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t n_kv = steps[(size_t) (i * k::kStepCount + k::kStepNKv)];
        const int64_t w = std::min<int64_t>(n_kv, steps[(size_t) (i * k::kStepCount + k::kStepWidth)]);
        compared += w;
        if (!std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap)) ++bad_q;
    }
    const bool ok = bad_q == 0 && st[tp::kChkIdQueries] == 0 && st[tp::kChkRescoredDiff] == 0 && st[tp::kChkKeptOut] == 0 &&
                    st[tp::kChkQueries] == (unsigned long long) nq;
    float ratio;
    const uint32_t rb = (uint32_t) st[tp::kChkRatioMax];
    std::memcpy(&ratio, &rb, 4);
    const unsigned long long kept = st[tp::kStQueries] - st[tp::kStFull];
    std::printf("%s ctx=%7lld stride=%5lld queries=%3lld %-20s: ids %lld queries differ (%lld ids compared); scored "
                "again %llu (%.1f per query), %llu of %llu scored-again blocks not bitwise; flagged %llu (%llu tiles in "
                "full); kept %llu, outside the bound %llu, max |S1-S3|/eps %.4f\n",
                ok ? "ok  " : "FAIL", (long long) c.ctx, (long long) max_blocks, (long long) nq, kKindName[c.kind],
                (long long) bad_q, (long long) compared, st[tp::kStBand], kept ? (double) st[tp::kStBand] / kept : 0.0,
                st[tp::kChkRescoredDiff], st[tp::kChkRescored], st[tp::kStFull], st[tp::kStFullTiles], st[tp::kChkKept],
                st[tp::kChkKeptOut], (double) ratio);
    if (ok && c.time && stride > 0) {   // one synthetic observation of each scorer (the bench worker measures the engine)
        cudaEvent_t e0, e1, e2;
        ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1)); ck(cudaEventCreate(&e2));
        constexpr int N = 5;
        ck(cudaEventRecord(e0));
        for (int r = 0; r < N; ++r) k::qsa_block_scores_tc(dp, dd, dq, dt, nq, max_blocks, s, ref, nullptr, active);
        ck(cudaEventRecord(e1));
        for (int r = 0; r < N; ++r)
            k::qsa_block_scores_two_pass(dp, dd, dq, dt, nq, nq_cap, max_blocks, s, two, scratch, sbytes, nullptr, active);
        ck(cudaEventRecord(e2));
        ck(cudaEventSynchronize(e2));
        float t3 = 0, t2 = 0;
        ck(cudaEventElapsedTime(&t3, e0, e1)); ck(cudaEventElapsedTime(&t2, e1, e2));
        std::printf("     time per call (synthetic, one run): 3xTF32 scorer %.3f ms, two-pass %.3f ms (keys once per chunk "
                    "not included)\n", t3 / N, t2 / N);
        k::qsa_two_pass_stats(st, true, nullptr);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    }
    ck(cudaFree(dp)); ck(cudaFree(dd)); ck(cudaFree(dq)); ck(cudaFree(dt)); ck(cudaFree(ref)); ck(cudaFree(two));
    ck(cudaFree(ref_ids)); ck(cudaFree(two_ids)); ck(cudaFree(scratch));
    return ok;
}
}  // namespace

int main() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    // AMD: the two-pass scorer is CUDA-only (qsa_select.cu builds stubs under HIP), so the engine keeps today's scorer
    std::puts("SKIP: the two-pass scorer is CUDA-only");
    return 77;
#endif
#if defined(_WIN32)
    if (std::getenv("STRATA_SELECT_TWO_PASS") == nullptr) _putenv_s("STRATA_SELECT_TWO_PASS", "1");
#else
    setenv("STRATA_SELECT_TWO_PASS", "1", 0);   // read once, at the first call
#endif
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    if (!k::qsa_two_pass_on()) {
        std::puts("FAIL: STRATA_SELECT_TWO_PASS is set to something other than 1");
        return 1;
    }
    const Case cases[] = {
        {2100, 64, kNormal, false},     {4096, 256, kNormal, false},   {9000, 37, kNormal, false},
        {20000, 256, kNormal, false},   {20481, 256, kSmall, false},   {32768, 256, kIndexer, false},
        {40000, 256, kWide, false},     {50000, 200, kNear, false},    {70000, 256, kDup, false},
        {131072, 256, kIndexer, false}, {131075, 129, kNormal, false}, {262144, 256, kIndexer, true},
        {262144, 256, kNormal, true},   {200003, 256, kSmall, false},  {65536, 256, kIndexer, true},
    };
    bool all = true;
    for (const int64_t stride : {(int64_t) 0, (int64_t) 65538})
        for (const Case& c : cases) all = run(c, stride) && all;
    if (!all) {
        std::puts("FAIL: the two-pass selection differs from the 3xTF32 scorer's");
        return 1;
    }
    std::puts("PASS: two-pass ids identical to the 3xTF32 scorer's, scored-again blocks bitwise, kept blocks inside the bound");
    return 0;
}
