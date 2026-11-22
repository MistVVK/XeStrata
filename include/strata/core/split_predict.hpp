// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
// Inputs in bytes and milliseconds. `mass` is empty for equally likely experts.
struct SplitPrediction {
    int64_t layers = 0;
    std::vector<std::pair<int32_t, int32_t>> profile;
    std::vector<double> mass, gpu_bw, link_scale, side_ms;
    std::vector<int> order;
    std::function<int64_t(int, int64_t, int64_t, bool)> capacity;
    std::function<int64_t(int64_t, int64_t)> cost;   // a (layer, expert) pair's bytes in a cache
    std::function<double(int64_t)> blob_bytes;       // the bytes a routed expert of a layer reads
    int64_t prompt_min = 0;
    double cpu_bw = 0, rows = 0, experts_per_row = 0, dense_layer = 0;
    double tail_keep = 1, head_bytes = 0;
    bool drafts = false, pipelined = false;
};
struct SplitPredictionResult {
    double ms = 0, held_mass = 0;
    int64_t held = 0;
    bool startable = true;
    std::vector<int64_t> capacity, used;
};
// A capacity-led policy for engines whose prompt path is not priced by the decode-window predictor.
std::vector<int64_t> choose_split_by_room(int64_t layers, uint64_t layer_bytes, bool mtp,
                                          const std::vector<double>& room, const std::vector<double>& link);
void enumerate_layer_splits(int64_t layers, const std::vector<double>& bandwidth,
                            const std::function<void(const std::vector<int64_t>&)>& visit);
SplitPredictionResult predict_layer_split(const SplitPrediction& p, const std::vector<int64_t>& cuts);
// Same measurements used for all engines, in bytes/ms.
double split_gpu_bandwidth(int device);
double split_cpu_bandwidth();
// What a role choice knows of a model's experts before anything is loaded, whatever engine runs it.
struct ExpertFacts {
    int64_t layers = 0;     ///< the layers that route experts (0: a dense model, with no miss to price)
    int64_t n_expert = 0;   ///< the experts of a layer
    int64_t layer = 0;      ///< the layer of the process's expert layout the pool is timed on
    /// Who computes a missed expert: the CPU pool, on the layout's native rows.  False: not the pool's native call
    /// (a canonical pack's rows, a latent pack's, a model that sends its misses to the GPU), which is not timed.
    bool pool_rows = false;
};
// The run's CPU pool, as the engine will build it.
struct PoolShape {
    int workers = 0;
    bool host_works = true;
    int tasks = 0;
};
// The rate (bytes a ms) a placement's CPU misses are priced at: `ram_rate` (split_cpu_bandwidth), or the pool's own
// rate on this model's experts where that is the lower - a pool of `pool`'s shape timed on stand-ins of the layout's
// experts (measure_cpu_expert_ms_unloaded), after the kernels' preload (beside it the CPU read up to 2x slow).
// `expert_ms`: the time one more expert took the pool, 0 when it was not timed, and `why` says why not (no experts,
// not the pool's rows, XESTRATA_SPLIT_CPU_GBPS given, or the timing's own reason).
struct MissRate {
    double rate = 0.0;
    double expert_ms = 0.0, expert_rate = 0.0;
    std::string why;
};
MissRate split_miss_rate(const ExpertFacts& facts, const PoolShape& pool, double ram_rate);
double probe_copy_latency_ms();
}
