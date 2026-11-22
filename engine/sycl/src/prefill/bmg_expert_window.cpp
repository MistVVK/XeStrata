#include "bmg_expert_window.hpp"
#include "bmg_direct_experts.hpp"
#include "strata/prefill/bmg_gate_up.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <algorithm>
#include <array>
#include <utility>

namespace strata::prefill::bmg_window {
std::vector<Batch> plan_batches(const PairPiece (&pieces)[2], int chunk, int gu_type, int i8m, bool can8,
                                double cross, bool use_direct) {
    std::vector<Batch> work;
    std::array<std::vector<Slice>, 512> dense;
    for (int c = 0; c < 2; ++c) {
        auto& pc = pieces[c];
        std::vector<int> active;
        for (int e = 0; e < 512; ++e)
            if (pc.count[e])
                active.push_back(e);
        for (size_t i = 0; i < active.size(); i += 16) {
            int size = int(std::min(size_t(16), active.size() - i)), small = 0, maximum = 0, total = 0;
            for (int j = 0; j < size; ++j) {
                int n = pc.count[active[i + j]];
                total += n;
                if (n <= 32) {
                    ++small;
                    maximum = std::max(maximum, n);
                }
            }
            bool use8 = can8 && (i8m != 2 || double(total) / size < cross);
            int variant = use_direct ? bmg_direct::layout(gu_type, small, maximum) : -1;
            Batch special;
            special.kind = use8 ? BatchKind::Int8 : BatchKind::Packed;
            special.variant = variant;
            for (int j = 0; j < size; ++j) {
                int e = active[i + j];
                Slice sl{c, pc.count[e], pc.off[e]};
                if (use8 || (variant >= 0 && sl.rows <= 32))
                    special.experts.push_back({e, {sl}});
                else
                    dense[e].push_back(sl);
            }
            if (!special.experts.empty())
                work.push_back(std::move(special));
        }
    }
    // The gate and down GEMMs have different M-dependent FP32 reduction orders.
    // A pair may share its weights even when one product must use each original row count.
    int batch_rows = 0;
    for (int e = 0; e < 512; ++e) {
        if (dense[e].empty())
            continue;
        int total = 0;
        for (auto& sl : dense[e])
            total += sl.rows;
        bool gate = false, down = false;
        if (dense[e].size() == 2) {
            const auto merge = merge_products(dense[e][0].rows, dense[e][1].rows);
            gate = merge.gate;
            down = merge.down;
        }
        if (work.empty() || work.back().kind != BatchKind::Dense || work.back().experts.size() == 16 ||
            batch_rows + total > 2 * chunk) {
            work.push_back({});
            batch_rows = 0;
        }
        work.back().experts.push_back({e, std::move(dense[e]), gate, down});
        batch_rows += total;
    }
    return work;
}

void gather_rows(sycl::queue& queue, const PairPiece (&pieces)[2], int chunk, const int* src, const int* slot_table,
                 float* all_down) {
    using Block = sycl::vec<uint32_t, 8>;
    const Block* left = reinterpret_cast<const Block*>(pieces[0].mixed);
    const Block* right = reinterpret_cast<const Block*>(pieces[1].mixed);
    Block* dest = reinterpret_cast<Block*>(all_down);
    const int cap = int(chunk);
    const size_t count = size_t(pieces[0].rows + pieces[1].rows);
    constexpr int lanes = 32;
    queue.parallel_for(
        sycl::nd_range<1>{count * lanes, lanes}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int token = int(it.get_group(0));
            int out[10];
#pragma unroll
            for (int j = 0; j < 10; ++j) {
                int row = slot_table[size_t(token) * 10 + j];
                out[j] = row + int(uint32_t(src[row]) >> 14);
            }
            const Block* x = token >= cap ? right + size_t(token - cap) * 160 : left + size_t(token) * 160;
#pragma unroll
            for (int k = it.get_local_id(0); k < 160; k += lanes) {
                const Block part = x[k];
#pragma unroll
                for (int j = 0; j < 10; ++j)
                    dest[size_t(out[j]) * 160 + k] = part;
            }
        });
}
bool project_group(const Batch& group, const kernels::cpu::NativeFmt& fmt, const uint8_t* const* raw,
                   const uint16_t* group_input, uint16_t* hidden, float* all_down, uint16_t* dq_gu, uint16_t* dq_down,
                   Gemm& gemm, sycl::queue& queue, const std::function<void(Phase)>& mark, std::string& err) {
    constexpr int N = 2560, kFF = 640, kFF2 = 2 * kFF;
    const int cc = int(group.experts.size());
    const int64_t offset = group.experts.front().slices.front().target;
    if (group.kind == BatchKind::Int8) {
        using namespace strata::prefill::i8;
        Layout lo;
        int32_t counts[16];
        for (int i = 0; i < cc; ++i)
            counts[i] = group.experts[i].slices[0].rows;
        make_layout(lo, counts, cc);
        Workspace ws;
        if (!workspace(lo, N, kFF, ws, &queue)) {
            err = "pair: original INT8 workspace unavailable";
            return false;
        }
        float* tmp = all_down + offset * N;
        mark(Phase::GateUp);
        quantize_rows(group_input, lo, N, ws.xq, ws.dx, &queue);
        if (!gate_up(fmt.gu_type, raw, fmt.up_off, kFF, lo, N, ws.xq, ws.dx, tmp, &queue)) {
            err = "pair: original INT8 gate/up failed";
            return false;
        }
        swiglu_quantize(tmp, lo, kFF, ws.hq, ws.dh, &queue);
        mark(Phase::Down);
        if (!down(fmt.d_type, raw, fmt.down_off, N, lo, kFF, ws.hq, ws.dh, tmp, &queue)) {
            err = "pair: original INT8 down failed";
            return false;
        }
    } else if (group.kind == BatchKind::Packed) {
        bmg_direct::Batch direct{};
        direct.count = cc;
        for (int i = 0; i < cc; ++i) {
            const Slice& sl = group.experts[i].slices[0];
            int local = sl.target - int(offset);
            direct.experts[i] = {group_input + int64_t(local) * N,
                                 raw[i],
                                 raw[i] + fmt.up_off,
                                 raw[i] + fmt.down_off,
                                 hidden + int64_t(local) * kFF,
                                 all_down + int64_t(sl.target) * N,
                                 sl.rows};
        }
        mark(Phase::GateUp);
        bmg_direct::gate(queue, direct, fmt.gu_type, group.variant);
        mark(Phase::Down);
        bmg_direct::down(queue, direct, group.variant);
    } else {
        mark(Phase::Dequant);
        strata::kernels::iq_dequant_gu_f16(fmt.gu_type, raw[0], raw[0] + fmt.up_off, kFF, N, dq_gu, &queue);
        for (int i = 0; i < cc; ++i) {
            auto& e = group.experts[i];
            const int first = e.slices[0].target;
            const int span = e.slices[0].rows + (e.slices.size() == 2 ? e.slices[1].rows : 0);
            auto* interval = all_down + size_t(first) * N;
            auto run = [&](bool down) {
                const bool merged = down ? e.merge_down : e.merge_gate;
                int count = merged ? 1 : int(e.slices.size());
                for (int z = 0; z < count; ++z) {
                    auto& sl = e.slices[z];
                    int nr = merged ? sl.rows + e.slices[1].rows : sl.rows;
                    int64_t at = sl.target, local = at - offset;
                    auto* h = hidden + local * kFF;
                    float* y = all_down + at * N;
                    const uint16_t* input = reinterpret_cast<uint16_t*>(interval) + (at - first) * N;
                    // All gate/up projections consume this expert's input before its
                    // first down write. Its second half holds the FP32 gate/up scratch.
                    float* gu = interval + size_t(span) * N / 2 + (at - first) * kFF2;

                    if (down)
                        gemm.f16(h, dq_down, y, nr, N, kFF);
                    else if (!bmg_gate_up::run(queue, input, dq_gu, h, nr)) {
                        gemm.f16(input, dq_gu, gu, nr, kFF2, N);
                        swiglu_interleaved(gu, h, nr, &queue);
                    }
                }
            };
            mark(Phase::GateUp);
            run(false);
            mark(Phase::Dequant);
            if (i + 1 < cc)
                strata::kernels::iq_dequant_expert_f16(fmt.gu_type, fmt.d_type, raw[i + 1], raw[i + 1] + fmt.up_off,
                                                       raw[i] + fmt.down_off, kFF, N, dq_gu, dq_down, &queue);
            else
                strata::kernels::iq_dequant_f16(fmt.d_type, raw[i] + fmt.down_off, N * kFF, dq_down, &queue);
            mark(Phase::Down);
            run(true);
        }
    }
    return true;
}
} // namespace strata::prefill::bmg_window
