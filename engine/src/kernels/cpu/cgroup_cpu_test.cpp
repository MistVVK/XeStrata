// cgroup_cpu_test: the CPU quota a container or a systemd unit puts on the process (include/strata/kernels/cpu/cgroup_cpu.hpp), read from a
// fake cgroup mount in a temp directory and a made-up /proc/self/cgroup.  No threads, no real cgroups: it runs anywhere.
#include "strata/kernels/cpu/cgroup_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

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

int main() {
    const fs::path root = fs::temp_directory_path() / ("strata_cgroup_cpu_test_" + std::to_string((long long) std::time(nullptr)));
    fs::remove_all(root);
    const std::string r = root.string();

    // 1. a container with its own cgroup namespace (RunPod, Docker on a cgroup v2 host): the group is the mount's root
    put(root / "ns" / "cpu.max", "1360000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "ns").string(), "0::/\n"), 13.6), "own namespace, v2: 13.6 CPUs");
    // 2. no limit: "max"
    put(root / "nolimit" / "cpu.max", "max 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "nolimit").string(), "0::/\n"), 0.0), "max: no limit");
    // 3. nothing there, garbage, a zero period
    expect(near(cpu::cgroup_cpu_quota_at((root / "empty").string(), "0::/\n"), 0.0), "no files: no limit");
    put(root / "junk" / "cpu.max", "banana\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "junk").string(), "0::/\n"), 0.0), "garbage: no limit");
    put(root / "zero" / "cpu.max", "100000 0\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "zero").string(), "0::/\n"), 0.0), "period 0: no limit");
    // 4. the host's namespace: the limit is on a parent of the process's group, the group itself and the root say nothing
    const fs::path host = root / "host";
    put(host / "cpu.max", "max 100000\n");   // (the real root cgroup has no such file; "max" must not matter either way)
    put(host / "system.slice" / "cpu.max", "400000 100000\n");
    put(host / "system.slice" / "docker-abc.scope" / "cpu.max", "max 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at(host.string(), "0::/system.slice/docker-abc.scope\n"), 4.0), "limit on a parent: 4 CPUs");
    // 5. the group's own limit tighter than the parent's, and looser
    put(host / "system.slice" / "docker-def.scope" / "cpu.max", "150000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at(host.string(), "0::/system.slice/docker-def.scope\n"), 1.5), "own limit tighter: 1.5 CPUs");
    put(host / "system.slice" / "docker-ghi.scope" / "cpu.max", "800000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at(host.string(), "0::/system.slice/docker-ghi.scope\n"), 4.0), "own limit looser: the parent's 4 CPUs");
    // 6. a path the mount does not have: the root's own file still counts (what the engine read before)
    put(root / "gone" / "cpu.max", "200000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "gone").string(), "0::/not/here\n"), 2.0), "missing path: the root's limit");
    // 7. CRLF, a ".." path, no /proc/self/cgroup at all
    expect(near(cpu::cgroup_cpu_quota_at((root / "gone").string(), "0::/not/here\r\n"), 2.0), "CRLF line");
    expect(near(cpu::cgroup_cpu_quota_at(host.string(), "0::/../../etc\n"), 0.0), "a path with .. is not followed");
    expect(near(cpu::cgroup_cpu_quota_at((root / "ns").string(), ""), 13.6), "no /proc/self/cgroup: the root's file");
    // 8. cgroup v1: Docker's group under the cpu controller, the quota on the group or on a parent; -1 is no limit
    const fs::path v1 = root / "v1";
    put(v1 / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_quota_us", "-1\n");
    put(v1 / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_period_us", "100000\n");
    put(v1 / "cpu,cpuacct" / "docker" / "cpu.cfs_quota_us", "300000\n");
    put(v1 / "cpu,cpuacct" / "docker" / "cpu.cfs_period_us", "100000\n");
    put(v1 / "cpu,cpuacct" / "cpu.cfs_quota_us", "-1\n");
    put(v1 / "cpu,cpuacct" / "cpu.cfs_period_us", "100000\n");
    const std::string v1_self = "12:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n1:name=systemd:/docker/abc\n";
    expect(near(cpu::cgroup_cpu_quota_at(v1.string(), v1_self), 3.0), "v1: a parent's 3 CPUs, the group's -1 ignored");
    put(v1 / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_quota_us", "50000\n");
    expect(near(cpu::cgroup_cpu_quota_at(v1.string(), v1_self), 0.5), "v1: the group's own half CPU");
    // 9. v1 with a line that does not list the cpu controller: not read
    expect(near(cpu::cgroup_cpu_quota_at(v1.string(), "12:memory:/docker/abc\n"), 0.0), "v1: no cpu controller line");
    // 10. both kinds of line (a hybrid system): the smaller limit wins
    put(root / "hyb" / "cpu.max", "100000 100000\n");
    put(root / "hyb" / "cpu" / "x" / "cpu.cfs_quota_us", "600000\n");
    put(root / "hyb" / "cpu" / "x" / "cpu.cfs_period_us", "100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "hyb").string(), "4:cpu:/x\n0::/\n"), 1.0), "hybrid: the smaller of 1 and 6 CPUs");
    // 11. the real thing runs and gives a sane number (a machine without cgroups gives 0)
    expect(cpu::cgroup_cpu_quota_at("/sys/fs/cgroup", "") >= 0.0, "the real mount reads without trouble");

    fs::remove_all(root);
    if (g_fail) {
        std::fprintf(stderr, "cgroup_cpu_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("cgroup_cpu_test: OK\n");
    return 0;
}
