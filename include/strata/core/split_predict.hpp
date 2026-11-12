// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>
#include <functional>
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
    std::function<int64_t(int64_t)> cost;
    std::function<double(int64_t)> blob_bytes;
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
double probe_copy_latency_ms();
}
