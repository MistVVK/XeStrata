// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// pool_cores_test: --pool-cores (experimental, off by default) and the start-up line about where the CPU pool's workers
// run.  The pure helpers on made-up layouts (upstream's pool_plan_test), then real pools (no model, no GPU; upstream's
// pool_cores_test on XeStrata's pool): the default placement is physical_cores(true) (one logical processor per physical
// core, the host's core left out); a list given to set_pool_cores() is taken as it is.
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static int g_fail = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

static std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

int main() {
    // ---- --pool-cores and the start-up line (pure helpers) ----
    {
        std::vector<int> v;
        std::string err;
        expect(cpu::parse_pool_cores("0,2,4,6", v, err) && v == std::vector<int>({0, 2, 4, 6}), "list: 0,2,4,6");
        expect(cpu::parse_pool_cores("2-14:2", v, err) && v == std::vector<int>({2, 4, 6, 8, 10, 12, 14}), "list: 2-14:2");
        expect(cpu::parse_pool_cores("4-6", v, err) && v == std::vector<int>({4, 5, 6}), "list: 4-6");
        expect(cpu::parse_pool_cores(" 6 , 2 ,2, 0-1 ", v, err) && v == std::vector<int>({6, 2, 0, 1}),
               "list: spaces ok, the person's order kept, a repeated number once");
        expect(!cpu::parse_pool_cores("", v, err) && v.empty() && !err.empty(), "list: empty is an error with a message");
        expect(!cpu::parse_pool_cores("0,,2", v, err) && v.empty(), "list: an empty item");
        expect(!cpu::parse_pool_cores("0,a", v, err), "list: letters");
        expect(!cpu::parse_pool_cores("6-2", v, err), "list: a range that goes down");
        expect(!cpu::parse_pool_cores("0-8:0", v, err), "list: a step of 0");
        expect(!cpu::parse_pool_cores("-3", v, err), "list: a negative number");
        expect(!cpu::parse_pool_cores("0-99999", v, err) && v.empty(), "list: a silly range is refused with a message, not expanded");
        expect(cpu::parse_pool_cores("0-4095", v, err) && v.size() == 4096, "list: 4096 entries are fine");
        expect(cpu::format_cpu_list({2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}) == "2,4,...,30", "format: a stride");
        expect(cpu::format_cpu_list({0, 1, 2, 3, 4}) == "0-4", "format: a run");
        expect(cpu::format_cpu_list({2, 4, 7}) == "2,4,7", "format: short and irregular lists in full");
        expect(cpu::format_cpu_list({1, 2, 4, 5, 9}) == "1,2,4,5,9", "format: irregular");
        expect(cpu::format_cpu_list({}) == "none", "format: none");

        // Windows-style numbering of 16 cores / 32 threads: core k = logical 2k and 2k+1
        std::vector<cpu::LogicalCpuInfo> win;
        win.reserve(32);
        for (int i = 0; i < 32; ++i) win.push_back({i, i / 2});
        std::vector<int> w;
        for (int c = 1; c < 16; ++c) w.push_back(2 * c);
        const std::string line = cpu::describe_worker_placement(w, 0, win, false);
        std::printf("  %s\n", line.c_str());
        expect(line == "CPU workers on logical processors 2,4,...,30 (15 workers, one per physical core; host on 0)",
               "the default line on 16c/32t");
        // the mistake the person worried about: workers on logical 0..7 of an adjacent-numbering SMT PC
        const std::string bad = cpu::describe_worker_placement({1, 2, 3, 4, 5, 6, 7}, 0, win, false);
        std::printf("  %s\n", bad.c_str());
        expect(bad.find("one per physical core") == std::string::npos && bad.find("carry more than one worker") != std::string::npos,
               "workers on logical 1..7 of 16c/32t are reported as sharing cores");
        expect(bad.find("shares the host's core") != std::string::npos, "... and the worker on the host's core is named");
        const std::string unk = cpu::describe_worker_placement({1, 2, 3}, 0, {}, false);
        expect(unk.find("did not say") != std::string::npos && unk.find("one per physical core") == std::string::npos,
               "unknown layout: no claim");
        expect(cpu::describe_worker_placement({2, 4}, -1, win, true).find("your --pool-cores list") != std::string::npos &&
               cpu::describe_worker_placement({2, 4}, -1, win, true).find("host not pinned") != std::string::npos,
               "a given list and an unpinned host are said");

        // check_pool_cores: drop what cannot be used, warn about siblings and the host's processor, never refuse
        cpu::PoolCoresCheck c = cpu::check_pool_cores({2, 4, 6, 8}, win, 0);
        expect(c.cpus == std::vector<int>({2, 4, 6, 8}) && c.warnings.empty(), "check: one per core, no warning");
        c = cpu::check_pool_cores({2, 3, 6}, win, 0);
        expect(c.cpus == std::vector<int>({2, 3, 6}) && c.warnings.size() == 1 && c.warnings[0].find("two hardware threads") != std::string::npos,
               "check: SMT siblings are kept with one warning");
        c = cpu::check_pool_cores({0, 2}, win, 0);
        expect(c.cpus == std::vector<int>({0, 2}) && c.warnings.size() == 1 && c.warnings[0].find("host thread's") != std::string::npos,
               "check: the host's processor is kept with a warning");
        c = cpu::check_pool_cores({1, 2}, win, 0);
        expect(c.cpus == std::vector<int>({1, 2}) && c.warnings.size() == 1 && c.warnings[0].find("shares a core with the host") != std::string::npos,
               "check: the host's SMT sibling is kept with a warning");
        c = cpu::check_pool_cores({2, 40, 99}, win, 0);
        expect(c.cpus == std::vector<int>({2}) && c.warnings.size() == 1 && c.warnings[0].find("40,99") != std::string::npos,
               "check: processors this process may not use are left out and named");
        c = cpu::check_pool_cores({40}, win, 0);
        expect(c.cpus.empty() && c.warnings.size() == 1, "check: nothing usable");
        c = cpu::check_pool_cores({2, 3}, {{2, -1}, {3, -1}}, 0);
        expect(c.cpus.size() == 2 && c.warnings.empty(), "check: an unknown layout cannot tell siblings: no warning");
    }


    const std::vector<cpu::LogicalCpuInfo> layout = cpu::logical_cpu_layout();
    std::printf("pool_cores_test: %zu logical processors\n", layout.size());
    expect(!layout.empty(), "the system names its logical processors");
    const std::vector<int> cores = cpu::physical_cores(true);
    const std::vector<int> all = cpu::physical_cores(false);
    const int host = all.empty() ? -1 : all[0];

    // 1. the default: nothing set, one worker per physical core, the host's core left out
    {
        cpu::set_pool_cores({});
        const int n = (int) std::min<size_t>(2, cores.size());
        if (n >= 1) {
            cpu::ExpertPool pool(n, true, true);
            expect(!pool.explicit_cores(), "default: not an explicit list");
            expect(pool.workers() == n, "default: the worker count asked for");
            expect(std::equal(cores.begin(), cores.begin() + n, pool.worker_cpus().begin()),
                   "default: the workers sit on physical_cores(true), in its order");
            const std::string line = pool.placement_line();
            std::printf("  default: %s\n", line.c_str());
            expect(line.rfind("CPU workers on logical processors ", 0) == 0, "the start-up line starts as documented");
            std::vector<long> wc;
            bool known = true;
            for (int i = 0; i < n; ++i) {
                long c = -1;
                for (const auto& l : layout) if (l.cpu == pool.worker_cpus()[(size_t) i]) c = l.core;
                known = known && c >= 0;
                wc.push_back(c);
            }
            if (known) {
                std::sort(wc.begin(), wc.end());
                expect(std::unique(wc.begin(), wc.end()) == wc.end(), "default: no two workers on one physical core");
                expect(line.find("one per physical core") != std::string::npos, "the line says one per physical core");
            } else {
                expect(line.find("one per physical core") == std::string::npos, "unknown layout: the line does not claim it");
            }
            for (const auto& l : layout) if (l.cpu == host)
                for (int i = 0; i < n; ++i)
                    for (const auto& m : layout)
                        if (m.cpu == pool.worker_cpus()[(size_t) i] && l.core >= 0)
                            expect(m.core != l.core, "default: no worker on the host's core");
        }
    }

    // 2. --pool-cores: the list as given, in its order; the count is the list's length
    if (cores.size() >= 3) {
        const std::vector<int> want{cores[2], cores[0]};
        cpu::set_pool_cores(want);
        cpu::ExpertPool pool(0, true, true);
        expect(pool.explicit_cores(), "list: explicit");
        expect(pool.workers() == 2, "list: the count is the list's length");
        expect(std::equal(want.begin(), want.end(), pool.worker_cpus().begin()),
               "list: the workers sit on the listed processors, in order");
        const std::string line = pool.placement_line();
        std::printf("  list %s: %s\n", join(want).c_str(), line.c_str());
        expect(line.find("your --pool-cores list") != std::string::npos, "list: the line says it is the person's list");

        // --pool-workers under the list: the first N entries
        cpu::ExpertPool one(1, true, true);
        expect(one.workers() == 1 && one.worker_cpus()[0] == want[0], "list + --pool-workers 1: the first entry");
    }

    // 3. a processor that does not exist is left out (warning), the rest stays; nothing usable falls back to the default
    if (cores.size() >= 2) {
        cpu::set_pool_cores({99999, cores[1]});
        cpu::ExpertPool pool(0, true, true);
        expect(pool.explicit_cores() && pool.workers() == 1 && pool.worker_cpus()[0] == cores[1],
               "an unusable entry is left out, the usable one kept");
        cpu::set_pool_cores({99999});
        cpu::ExpertPool fallback(1, true, true);
        expect(!fallback.explicit_cores() && fallback.worker_cpus()[0] == cores[0], "nothing usable: the default placement");
    }

    // 4. two SMT siblings of one core are kept (and warned about by check_pool_cores, above)
    for (const auto& a : layout) {
        const cpu::LogicalCpuInfo* b = nullptr;
        for (const auto& c : layout) if (c.cpu != a.cpu && a.core >= 0 && c.core == a.core) { b = &c; break; }
        if (b == nullptr) continue;
        cpu::set_pool_cores({a.cpu, b->cpu});
        cpu::ExpertPool pool(0, true, true);
        expect(pool.explicit_cores() && pool.workers() == 2 && pool.worker_cpus()[0] == a.cpu &&
                   pool.worker_cpus()[1] == b->cpu,
               "SMT siblings in the list are kept as given");
        const std::string line = pool.placement_line();
        std::printf("  siblings %d,%d: %s\n", a.cpu, b->cpu, line.c_str());
        expect(line.find("more than one worker") != std::string::npos, "the line says two workers share a core");
        break;
    }
    cpu::set_pool_cores({});

    if (g_fail) {
        std::fprintf(stderr, "pool_cores_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("pool_cores_test: OK\n");
    return 0;
}
