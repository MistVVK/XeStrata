// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
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

// XESTRATA_POOL_DEP is read by the pool's constructor: set it before each pool is built
static void set_pool_dep(const char* v) {
#if defined(_WIN32)
    _putenv_s("XESTRATA_POOL_DEP", v);
#else
    setenv("XESTRATA_POOL_DEP", v, 1);
#endif
}

static int check_types(int gu_type, int down_type, const char* blob_file, int& checks);

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) return 2;
    int checks = 0;
    if (argc == 4) {
        if (const int rc = check_types((int) std::strtol(argv[1], nullptr, 10), (int) std::strtol(argv[2], nullptr, 10),
                                       argv[3], checks))
            return rc;
    } else {
        // Q2_0 down rows take their own kernel (q2_rows_any) in both the two-phase and the one-batch call
        for (int down : {(int) GGML_TYPE_Q4_0, (int) GGML_TYPE_Q2_0})
            if (const int rc = check_types(GGML_TYPE_Q4_0, down, nullptr, checks)) return rc;
    }
    std::printf("pool tasks: %d bitwise comparisons passed\n", checks);
    return 0;
}

static int check_types(int gu_type, int down_type, const char* blob_file, int& checks) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, down_type, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    if (blob_file != nullptr) {
        std::FILE* file = std::fopen(blob_file, "rb");
        if (!file) return 2;
        const size_t got = std::fread(blob.data(), 1, blob.size(), file);
        std::fclose(file);
        if (got != blob.size()) return 2;
    } else {
        std::vector<float> weights((size_t) cpu::H * cpu::FF);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
        ggml_quantize_chunk((ggml_type) gu_type, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk((ggml_type) gu_type, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk((ggml_type) down_type, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    for (int tasks : {-1, cpu::ExpertPool::kMaxTasks + 1}) {
        try { cpu::ExpertPool invalid(1, false, true, 0, tasks); return 1; }
        catch (const std::invalid_argument&) { continue; }   // refused, as it should be
    }
    std::vector<cpu::NativeAct> act(cpu::MAXT);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[t].bytes);
    }
    for (int n : {1, 7, cpu::ExpertPool::kMaxSplitMulti + 1}) {
        std::vector<cpu::ExpertJobMulti> jobs(n);
        std::vector<float> out((size_t) n * cpu::MAXT * cpu::H);
        for (int e = 0; e < n; ++e) {
            jobs[e].blob = blob.data();
            jobs[e].nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 4 : cpu::MAXT;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = act[t].bytes;
                jobs[e].out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            }
        }
        { // The two-phase default partition is the arithmetic reference; inactive token slots remain zero.
            set_pool_dep("0");
            cpu::ExpertPool baseline(3, false);
            baseline.run_split_multi_native(f, jobs.data(), n);
        }
        const auto ref = out;
        for (float v : ref) if (!std::isfinite(v)) return 1;
        for (const char* dep : {"0", "1"}) for (bool host : {false, true}) for (int workers : {1, 3})
            for (int tasks : {0, 1, 41, 128, 192, 256, cpu::ExpertPool::kMaxTasks}) {
                set_pool_dep(dep);
                cpu::ExpertPool pool(workers, false, host, 0, tasks);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    std::fill(out.begin(), out.end(), 0.0f);
                    for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t)
                        std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
                    pool.run_split_multi_native(f, jobs.data(), n);
                    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison): bit-exact
                    if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "mismatch types=%d/%d dep=%s n=%d workers=%d host=%d tasks=%d repeat=%d\n",
                                     gu_type, down_type, dep, n, workers, host, tasks, repeat);
                        return 1;
                    }
                    ++checks;
                }
            }
    }
    return 0;
}
