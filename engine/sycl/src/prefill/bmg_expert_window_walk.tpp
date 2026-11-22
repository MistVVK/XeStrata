// Private paired walk, instantiated with Prefill::Impl after its staging and timing types are defined.
namespace strata::prefill {
namespace {
template <typename State>
bool run_window_experts(State& m, const kernels::cpu::ExpertLayout& lay, int64_t l,
                        const core::GgufExpertSource* gsrc, PrefillStats& stats, PfTimer& pt, float*& all_down,
                        int*& slot_table, std::string& err) {
    const auto& fmt = lay.fmt[size_t(l)];
    const auto cs = strata::q_of(m.cs);
    const int i8m = i8_mode(m.cs);
    const bool can8 = fmt.gu_type == 22 && i8m && strata::prefill::i8::available(m.cs);
    if (can8 && i8m == 2 && m.pair_i8_cutoff < 0.0)
        m.pair_i8_cutoff = i8_calibrate(m, fmt).rows;
    const double cross = m.pair_i8_cutoff;
    const bool use_direct = strata::bmg::experts_enabled();
    auto work = bmg_window::plan_batches(m.pieces, int(m.T), fmt.gu_type, i8m, can8, cross, use_direct);
    const auto mark = [&](bmg_window::Phase phase) {
        const int step = phase == bmg_window::Phase::GateUp ? kPfGemmGU
                         : phase == bmg_window::Phase::Down ? kPfGemmD
                                                            : kPfDequant;
        pt.mark(step, cs);
    };
    std::vector<Stager::Job> jobs;
    auto& source = m.pair_source;
    source.resize(size_t(m.pieces[0].rows + m.pieces[1].rows) * K);
    size_t next_source = 0;
    auto& slots = m.pair_slots;
    for (int c = 0; c < 2; ++c)
        slots[c].resize(m.pieces[c].rows * K);
    std::array<std::array<int, 512>, 2> targets{};
    // Each input token index is below 2^14, and two 8K top-10 chunks have fewer than
    // 2^18 routed rows. Pack the owning interval into the uploaded row map.
    for (auto& group : work) {
        const int group_start = int(next_source);
        for (auto& prod : group.experts) {
            const int anchor = group.kind != bmg_window::BatchKind::Dense ? group_start : int(next_source);
            const uint32_t prefix = uint32_t(anchor) << 14;

            int e = prod.expert;
            for (auto& sl : prod.slices) {
                auto& pc = m.pieces[sl.piece];
                sl.target = int(next_source);
                targets[sl.piece][e] = sl.target;
                int piece_start = sl.piece * int(m.T);
                const int* input = pc.source.data() + sl.source;
                int* out = source.data() + next_source;
                for (int j = 0; j < sl.rows; ++j)
                    out[j] = int(prefix | uint32_t(piece_start + input[j]));
                next_source += sl.rows;
            }
            prod.slot = m.host_res ? m.host_res[size_t(l) * m.g->n_expert + e] : -1;
            if (prod.slot >= 0)
                continue;
            const uint8_t* raw = m.src->pinned(l, e) ? m.src->blob(l, e) : nullptr;
            if (dma_direct(m.src, l, e, raw)) {
                prod.pinned = raw;
                continue;
            }
            prod.job = int(jobs.size());
            const size_t bytes = lay.blob_bytes(l);
            if (gsrc)
                jobs.push_back({nullptr, bytes, nullptr, int(l), e, gsrc});
            else if (m.src->transient(l, e))
                jobs.push_back({nullptr, bytes, m.src, int(l), e});
            else {
                const uint8_t* blob = m.src->blob(l, e);
                if (!blob) {
                    err = "pair: missing weight";
                    return false;
                }
                jobs.push_back({blob, bytes});
            }
        }
    }
    for (int c = 0; c < 2; ++c) {
        auto next = targets[c];
        auto& ids = m.pieces[c].ids;
        for (size_t i = 0; i < ids.size(); ++i)
            slots[c][i] = next[ids[i]]++;
    }
    if (next_source != source.size()) {
        err = "pair: incomplete route";
        return false;
    }
    // Both chunks own disjoint down rows. The FP16 group input and temporary hidden
    // rows borrow images consumed by the two shared-expert projections.
    all_down = reinterpret_cast<float*>(m.region);

    auto* hidden = m.mixed_bf;
    auto* src = reinterpret_cast<int*>(hidden + size_t(2) * m.T * kFF);
    slot_table = src + size_t(2) * m.T * K;
    {
        if (size_t(slot_table + size_t(2) * m.T * K - reinterpret_cast<int*>(m.mixed_bf)) * 4 > size_t(m.T) * N * 2 ||
            size_t(2) * m.T * K * N * 4 > m.region_bytes) {
            err = "pair: tight workspace exceeds allocation";
            return false;
        }
    }
    m.cs->memcpy(src, source.data(), source.size() * sizeof(int));
    for (int c = 0; c < 2; ++c)
        m.cs->memcpy(slot_table + c * m.T * K, slots[c].data(), slots[c].size() * 4);
    m.stager->start(std::move(jobs));
    StagerDone release_host{m.stager.get()};
    pt.mark(kPfGather, cs);
    bmg_window::gather_rows(*m.cs, m.pieces, int(m.T), src, slot_table, all_down);
    // Whole-window gather leaves mixed and the shared-expert output available until
    // combine. Keep selected cold blobs there across batches within this layer.
    const size_t weight_bytes = lay.blob_bytes(l);
    const size_t stride = (weight_bytes + 255u) & ~size_t(255u);
    const int per_bank = stride ? int(size_t(m.T) * N * sizeof(float) / stride) : 0;
    std::array<int, 512> cached_slot{};
    std::array<bool, 512> copied{};
    std::optional<sycl::event> scratch_ready;
    if (per_bank) {
        cached_slot.fill(-1);
        std::array<int, 512> frequency{};
        std::vector<int> candidates;
        for (const auto& batch : work)
            for (const auto& product : batch.experts)
                if (product.slot < 0 && product.pinned && ++frequency[product.expert] == 1)
                    candidates.push_back(product.expert);
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](int a, int b) { return frequency[a] > frequency[b]; });
        int next = 0;
        for (int expert : candidates) {
            if (next == 2 * per_bank)
                break;
            cached_slot[expert] = next++;
        }
        if (next)
            scratch_ready = m.cs->ext_oneapi_submit_barrier();
    }
    auto cached = [&](const auto& product) {
        return scratch_ready && product.slot < 0 && product.pinned && cached_slot[product.expert] >= 0;
    };
    struct Held {
        std::array<const uint8_t*, 16> weights{};
        std::array<int, 16> slots{};
        std::optional<sycl::event> scratch_copy;
        int misses = 0, total = 0;
    };
    std::vector<Held> issued(work.size());
    int ring_next = 0, held = 0;
    size_t cursor = 0;
    for (size_t ix = 0; ix < work.size(); ++ix) {
        while (cursor < work.size() && cursor == ix) {
            auto& todo = work[cursor];
            int cold = 0;
            for (auto& p : todo.experts)
                cold += p.slot < 0 && !cached(p);
            if (cold + held > stage_slots())
                break;
            auto& entry = issued[cursor];
            int cc = 0;
            for (auto& prod : todo.experts) {
                for (auto& sl : prod.slices)
                    entry.total += sl.rows;
                if (prod.slot >= 0) {
                    entry.weights[cc++] = m.cache->device_slot(prod.slot);
                    ++stats.experts_resident;
                    continue;
                }
                if (cached(prod)) {
                    const int slot = cached_slot[prod.expert];
                    auto* bank = reinterpret_cast<uint8_t*>(slot < per_bank ? m.mixed : m.bo);
                    auto* dest = bank + size_t(slot % per_bank) * stride;
                    if (!copied[prod.expert]) {
                        entry.scratch_copy = m.copy->memcpy(dest, prod.pinned, weight_bytes, *scratch_ready);
                        copied[prod.expert] = true;
                        ++stats.experts_dma;
                    }
                    entry.weights[cc++] = dest;
                    ++stats.experts_streamed;
                    continue;
                }
                const uint8_t* base = prod.pinned ? prod.pinned : m.stager->wait(prod.job);
                int sl = ring_next;
                ring_next = (ring_next + 1) % stage_slots();
                entry.slots[entry.misses++] = sl;
                auto copy_done = m.stage_live[sl]
                                  ? m.copy->memcpy(m.stage_dev[sl], base, lay.blob_bytes(l), *m.used[m.used_of[sl]])
                                  : m.copy->memcpy(m.stage_dev[sl], base, lay.blob_bytes(l));
                *m.copied[sl] = copy_done;
                m.stage_live[sl] = true;
                if (!prod.pinned)
                    m.stager->issued_one(prod.job, m.copy);
                else
                    ++stats.experts_dma;
                entry.weights[cc++] = m.stage_dev[sl];
                ++stats.experts_streamed;
            }
            held += cold;
            ++cursor;
        }
        if (cursor <= ix) {
            err = "pair: insufficient DMA slots";
            return false;
        }
        auto& group = work[ix];
        auto& entry = issued[ix];
        auto* raw = entry.weights.data();
        auto* occupied = entry.slots.data();
        int total = entry.total, miss = entry.misses;
        held -= miss;
        int64_t offset = group.experts[0].slices[0].target;
        if (total > 2 * m.T) {
            err = "pair: group input exceeds scratch";
            return false;
        }
        // The in-order copy queue signals all newly populated slots in this
        // batch. Later batches consume their live blobs on the same compute queue.
        if (entry.scratch_copy)
            m.cs->ext_oneapi_submit_barrier({*entry.scratch_copy});
        uint16_t* group_input = reinterpret_cast<uint16_t*>(all_down + offset * N);
        if (miss) {
            pt.mark(kPfWaitCopy, cs);
            m.cs->ext_oneapi_submit_barrier({*m.copied[occupied[miss - 1]]});
        }
        if (!bmg_window::project_group(group, fmt, raw, group_input, hidden, all_down, m.dq_gu[0], m.dq_d[0], m.gemm,
                                       *m.cs, mark, err))
            return false;
        if (miss) {
            int last = occupied[miss - 1];
            *m.used[last] = m.cs->ext_oneapi_submit_barrier();
            for (int i = 0; i < miss; ++i)
                m.used_of[occupied[i]] = last;
        }
    }
    return true;
}
} // namespace
} // namespace strata::prefill
