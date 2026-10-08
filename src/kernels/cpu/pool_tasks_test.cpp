// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU-only task partition parity. Optional real expert: pool_tasks_test GU_TYPE DOWN_TYPE BLOB_FILE
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) return 2;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(argc == 4 ? (int) std::strtol(argv[1], nullptr, 10) : GGML_TYPE_Q4_0,
                         argc == 4 ? (int) std::strtol(argv[2], nullptr, 10) : GGML_TYPE_Q4_0, cpu::H, cpu::FF, f,
                         err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    if (argc == 4) {
        std::FILE* file = std::fopen(argv[3], "rb");
        if (!file) return 2;
        const size_t got = std::fread(blob.data(), 1, blob.size(), file);
        std::fclose(file);
        if (got != blob.size()) return 2;
    } else {
        std::vector<float> weights((size_t) cpu::H * cpu::FF);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    for (int tasks : {-1, cpu::ExpertPool::kMaxTasks + 1}) {
        try { cpu::ExpertPool invalid(1, false, true, 0, tasks); return 1; }
        catch (const std::invalid_argument&) { continue; }   // refused, as it should be
    }
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact(cpu::MAXT);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[t].data());
        cpu::act_quant_any(x.data(), cpu::H, qact[t]);
    }
    int checks = 0;
    for (int n : {1, 7, cpu::ExpertPool::kMaxSplitMulti + 1}) {
        std::vector<cpu::ExpertJobMulti> jobs(n);
        std::vector<float> out((size_t) n * cpu::MAXT * cpu::H);
        for (int e = 0; e < n; ++e) {
            jobs[e].blob = blob.data();
            jobs[e].nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 4 : cpu::MAXT;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = act[t].data();
                jobs[e].act[t] = &qact[t];
                jobs[e].out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            }
        }
        { // The default partition is the arithmetic reference; inactive token slots remain zero.
            cpu::ExpertPool baseline(3, false);
            baseline.run_split_multi_native(f, jobs.data(), n);
        }
        const auto ref = out;
        for (float v : ref) if (!std::isfinite(v)) return 1;
        for (bool host : {false, true}) for (int workers : {1, 3})
            for (int tasks : {0, 1, 41, 128, 192, 256, cpu::ExpertPool::kMaxTasks}) {
                cpu::ExpertPool pool(workers, false, host, 0, tasks);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    std::fill(out.begin(), out.end(), 0.0f);
                    for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t)
                        std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
                    pool.run_split_multi_native(f, jobs.data(), n);
                    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison): bit-exact
                    if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "mismatch n=%d workers=%d host=%d tasks=%d repeat=%d\n",
                                     n, workers, host, tasks, repeat);
                        return 1;
                    }
                    ++checks;
                }
            }
    }
    std::printf("pool tasks: %d bitwise comparisons passed\n", checks);
    return 0;
}
