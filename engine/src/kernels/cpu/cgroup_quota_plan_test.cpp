// cgroup_quota_plan_test: the PR#14 regression composition.  The container CPU quota read from a FAKE cgroup mount
// (cgroup_cpu.hpp, path-injectable, the layouts already pinned by cgroup_cpu_test.cpp) wired straight into the
// default worker-count plan (pool_plan.hpp) - the same two seams pool.cpp joins in fill_plan_fields /
// pool_plan_input.  No threads, no real cgroups, no root: it runs anywhere.
//
// What it guards:
//   * the quota-bounded worker count is the plan's, and the plan's only: 13.6 CPUs on a 64-core machine gives 12
//     workers, never the 32 of the cap, the 64 of the cores or the 128 of the logical processors;
//   * a quota that cannot be read ("max", garbage, no files, no cpu line) reads 0.0 and leaves the count EXACTLY at
//     the pre-cgroup value - the required failure path when cgroup values are unavailable: no silent collapse to 1;
//   * the parser-to-plan wiring cannot drift: every clamped row stays strictly below the unclamped baseline,
//     never below 1, and q == n does not clamp (the comparison is strict).
#include "strata/kernels/cpu/cgroup_cpu.hpp"
#include "strata/kernels/cpu/pool_plan.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static void put(const fs::path& file, const std::string& text) {
    fs::create_directories(file.parent_path());
    std::ofstream(file) << text;
}

static cpu::PoolPlanInput cpu_of(int logical, std::vector<int> per_package) {
    cpu::PoolPlanInput in;
    in.logical = logical;
    in.cores_per_package = std::move(per_package);
    return in;
}

// One composition row: read the quota exactly the way container_cpu_quota() will (cgroup_cpu_test.cpp pins these
// readings), then feed it to the plan the engine feeds it.  `want_quota` is the pin from the parser test - if the
// parser ever stops agreeing with it, this names the row that broke before the plan assertion does.
static void row(const std::string& name, const std::string& mount, const std::string& self_cgroup,
                double want_quota, const cpu::PoolPlanInput& base, int want_workers, const char* want_limit) {
    const double q = cpu::cgroup_cpu_quota_at(mount, self_cgroup);
    expect(near(q, want_quota), name + ": parser gives the pinned quota");
    cpu::PoolPlanInput in = base;
    in.cpu_quota = q;
    const cpu::PoolPlan p = cpu::plan_pool_workers(in);
    expect(p.workers == want_workers, name + ": plan -> " + std::to_string(want_workers) + " workers, got " +
                                     std::to_string(p.workers));
    if (want_limit != nullptr)
        expect(std::string(p.limit) == want_limit, name + ": limit is \"" + want_limit + "\", got \"" +
                                                 std::string(p.limit) + "\"");
    const int unclamped = cpu::plan_pool_workers(base).workers;
    expect(p.workers <= unclamped, name + ": a quota never RAISES the count");
    expect(p.workers >= 1, name + ": never 0 workers");
    std::printf("  %-52s quota %6.2f  unclamped %2d -> %2d workers [%s]\n", name.c_str(), q, unclamped, p.workers,
                p.limit);
}

// The "cgroup values unavailable" rows: every shape that yields no limit must leave the count EXACTLY at the
// pre-cgroup value (8c/16t -> 7).  A silent collapse to 1 worker, a crash, or a cap-sized read all fail here.
static void no_limit_row(const std::string& name, const std::string& mount, const std::string& self_cgroup) {
    const double q = cpu::cgroup_cpu_quota_at(mount, self_cgroup);
    expect(near(q, 0.0), name + ": unreadable quota reads 0.0, not a limit");
    cpu::PoolPlanInput in = cpu_of(16, {8});
    in.cpu_quota = q;
    const cpu::PoolPlan p = cpu::plan_pool_workers(in);
    expect(p.workers == 7, name + ": count stays 7 (no silent collapse), got " + std::to_string(p.workers));
    expect(std::string(p.limit) != "the container's CPU quota", name + ": an absent limit is not claimed");
    std::printf("  %-52s quota %6.2f            -> %2d workers [%s]\n", name.c_str(), q, p.workers, p.limit);
}

int main() {
    const fs::path root = fs::temp_directory_path() /
                          ("strata_cgroup_quota_plan_test_" + std::to_string((long long) std::time(nullptr)));
    fs::remove_all(root);
    const std::string r = root.string();
    std::printf("cgroup_quota_plan_test: quota -> plan composition (PR#14 regression)\n");

    // 1. RunPod / Docker with its own cgroup namespace: 13.6 CPUs on a 64-core box -> 12 workers, far below the
    //    unclamped 32, the cores 64 and the logical 128 (the container report PR#14 came out of)
    put(root / "ns" / "cpu.max", "1360000 100000\n");
    row("v2 own namespace 13.6 CPUs on 64c/128t", (root / "ns").string(), "0::/\n", 13.6, cpu_of(128, {64}), 12,
        "the container's CPU quota");

    // 2. the host's namespace: the limit sits on a PARENT of the process's group (systemd scope) - 4 CPUs on 8c -> 3
    put(root / "host" / "cpu.max", "max 100000\n");
    put(root / "host" / "system.slice" / "cpu.max", "400000 100000\n");
    put(root / "host" / "system.slice" / "docker-abc.scope" / "cpu.max", "max 100000\n");
    row("v2 parent scope 4.0 CPUs on 8c/16t", (root / "host").string(), "0::/system.slice/docker-abc.scope\n", 4.0,
        cpu_of(16, {8}), 3, "the container's CPU quota");

    // 3. the group's own limit tighter than the parent's: 1.5 CPUs -> 1 worker (the floor, never 0)
    put(root / "host" / "system.slice" / "docker-def.scope" / "cpu.max", "150000 100000\n");
    row("v2 own 1.5 CPUs on 8c/16t", (root / "host").string(), "0::/system.slice/docker-def.scope\n", 1.5,
        cpu_of(16, {8}), 1, "the container's CPU quota");

    // 4. a group path the mount does not have: the root's own file still counts - 2.0 CPUs -> 1
    put(root / "gone" / "cpu.max", "200000 100000\n");
    row("v2 missing path, root 2.0 CPUs on 8c/16t", (root / "gone").string(), "0::/not/here\n", 2.0,
        cpu_of(16, {8}), 1, "the container's CPU quota");

    // 5-6. cgroup v1: a parent's 3 CPUs -> 2; the group's own half CPU -> 1  (the tree cgroup_cpu_test.cpp pins)
    const fs::path v1 = root / "v1";
    put(v1 / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_quota_us", "-1\n");
    put(v1 / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_period_us", "100000\n");
    put(v1 / "cpu,cpuacct" / "docker" / "cpu.cfs_quota_us", "300000\n");
    put(v1 / "cpu,cpuacct" / "docker" / "cpu.cfs_period_us", "100000\n");
    put(v1 / "cpu,cpuacct" / "cpu.cfs_quota_us", "-1\n");
    put(v1 / "cpu,cpuacct" / "cpu.cfs_period_us", "100000\n");
    const std::string v1_self = "12:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n1:name=systemd:/docker/abc\n";
    row("v1 parent 3.0 CPUs on 8c/16t", v1.string(), v1_self, 3.0, cpu_of(16, {8}), 2, "the container's CPU quota");
    put(v1 / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_quota_us", "50000\n");
    row("v1 own 0.5 CPUs on 8c/16t", v1.string(), v1_self, 0.5, cpu_of(16, {8}), 1, "the container's CPU quota");

    // 7. both kinds of line (a hybrid system): the smaller limit wins - 1.0 CPU -> 1 worker
    put(root / "hyb" / "cpu.max", "100000 100000\n");
    put(root / "hyb" / "cpu" / "x" / "cpu.cfs_quota_us", "600000\n");
    put(root / "hyb" / "cpu" / "x" / "cpu.cfs_period_us", "100000\n");
    row("v2+v1 hybrid, smaller 1.0 CPU wins on 8c/16t", (root / "hyb").string(), "4:cpu:/x\n0::/\n", 1.0,
        cpu_of(16, {8}), 1, "the container's CPU quota");

    // 8. the failure path: cgroup values unavailable in every shape - "max", garbage, no files at all, no /proc/self
    //    line, a v1 group with no cpu line.  All must read 0.0 and leave the plan at exactly its pre-cgroup count.
    put(root / "nolimit" / "cpu.max", "max 100000\n");
    no_limit_row("unavailable: cpu.max 'max'", (root / "nolimit").string(), "0::/\n");
    put(root / "junk" / "cpu.max", "banana\n");
    no_limit_row("unavailable: cpu.max garbage", (root / "junk").string(), "0::/\n");
    no_limit_row("unavailable: no cgroup files", (root / "empty").string(), "0::/\n");
    no_limit_row("unavailable: no /proc/self/cgroup, root unlimited", (root / "nolimit").string(), "");
    no_limit_row("unavailable: no files and no /proc/self/cgroup", (root / "empty").string(), "");
    no_limit_row("unavailable: v1 without a cpu line", v1.string(), "12:memory:/docker/abc\n");

    // 9. the q == n boundary: quota 8.0 on the 8c/16t box computes q = 8 - 1 = 7 == n; the comparison is STRICT, so
    //    the count stays 7.  A flipped <= would silently drop a worker - and a flipped formula for q would too.
    {
        cpu::PoolPlanInput in = cpu_of(16, {8});
        in.cpu_quota = 8.0;
        expect(cpu::plan_pool_workers(in).workers == 7, "q == n: quota 8.0 on 8c does not clamp (strict <)");
        in.cpu_quota = 7.999;
        expect(cpu::plan_pool_workers(in).workers == 6, "just under: quota 7.999 clamps to 6");
    }

    // 10. drift guards on the clamp arithmetic itself, no OS at all: 3.999 truncates to 3 (no rounding), 4.5
    //     truncates to 3, a huge quota changes nothing, quota never lifts the count above the affinity bound.
    {
        cpu::PoolPlanInput in = cpu_of(16, {8});
        in.cpu_quota = 3.999;
        expect(cpu::plan_pool_workers(in).workers == 2, "quota 3.999 truncates to q=2 -> 2 workers");
        in.cpu_quota = 4.5;
        expect(cpu::plan_pool_workers(in).workers == 3, "quota 4.5 truncates to q=3 -> 3 workers");
        in.cpu_quota = 200.0;
        expect(cpu::plan_pool_workers(in).workers == 7, "a quota above the cores changes nothing");
        cpu::PoolPlanInput few = cpu_of(4, {4});
        few.cpu_quota = 8.0;
        expect(cpu::plan_pool_workers(few).workers == 3, "4 CPUs allowed under an 8-CPU quota: the mask bounds, not the quota");
    }

    fs::remove_all(root);
    if (g_fail) {
        std::fprintf(stderr, "cgroup_quota_plan_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("cgroup_quota_plan_test: OK\n");
    return 0;
}
