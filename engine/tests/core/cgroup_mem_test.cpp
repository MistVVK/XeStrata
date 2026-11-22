// SPDX-License-Identifier: MIT
// tests/core/cgroup_mem_test.cpp - synthetic fixture tests for Linux cgroup v1/v2 memory room (ancestor slices,
// reclaimable file-cache credit, binary GiB conversion) and Windows Job Object commit ceilings.

#include "strata/platform/cgroup_mem.hpp"
#include "strata/platform/commit.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using strata::platform::cgroup_mem::gib_to_bytes;
using strata::platform::cgroup_mem::kGiB;
using strata::platform::cgroup_mem::kUnlimited;
using strata::platform::cgroup_mem::probe_cgroup_room;

void write_text(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p);
    assert(static_cast<bool>(f));
    f << s;
}

fs::path make_temp_dir() {
    const auto pid =
#if defined(_WIN32)
        12345;
#else
        static_cast<long>(::getpid());
#endif
    fs::path dir = fs::temp_directory_path() / ("strata_cgroup_mem_test_" + std::to_string(pid));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

void test_gib_to_bytes_exact_binary() {
    // 50 GiB must be 50 * 2^30 (53,687,091,200 B), never 50 * 1e9 (50,000,000,000 B: -3.43 GiB bug).
    assert(gib_to_bytes(50.0) == 50ULL * kGiB);
    assert(gib_to_bytes(50.0) - 50000000000ULL == 3687091200ULL);
    assert(gib_to_bytes(0.0) == 0ULL);
    assert(gib_to_bytes(-2.5) == 0ULL);
}

void test_cgroup_v2_nested_parent_limit_and_reclaimable_file() {
    const fs::path tmp = make_temp_dir();
    const fs::path cg_root = tmp / "cgroup";
    const fs::path self_cg = tmp / "proc_self_cgroup";

    // Leaf slice has "max", parent kubepods.slice caps at 32 GiB with 20 GiB current, of which 12 GiB is clean
    // unmapped file cache (10 GiB inactive_file + 3 GiB active_file - 1 GiB file_mapped = 12 GiB reclaimable).
    // Effective used = 20 - 12 = 8 GiB -> available = 32 - 8 = 24 GiB.
    write_text(self_cg, "0::/kubepods.slice/pod-abc/ctr-1\n");
    write_text(cg_root / "kubepods.slice/pod-abc/ctr-1/memory.max", "max\n");
    write_text(cg_root / "kubepods.slice/memory.max", std::to_string(32ULL * kGiB) + "\n");
    write_text(cg_root / "kubepods.slice/memory.current", std::to_string(20ULL * kGiB) + "\n");
    write_text(cg_root / "kubepods.slice/memory.stat",
               "anon " + std::to_string(7ULL * kGiB) + "\n"
               "inactive_file " + std::to_string(10ULL * kGiB) + "\n"
               "active_file " + std::to_string(3ULL * kGiB) + "\n"
               "file_mapped " + std::to_string(1ULL * kGiB) + "\n"
               "file_dirty 0\n"
               "file_writeback 0\n");

    const auto r = probe_cgroup_room(/*host_total_bytes=*/128ULL * kGiB, self_cg.string(), cg_root.string());
    assert(r.limited);
    assert(r.limit_bytes == 32ULL * kGiB);
    assert(r.available_bytes == 24ULL * kGiB);
    fs::remove_all(tmp);
}

void test_cgroup_v1_hierarchy_and_unlimited_sentinel() {
    const fs::path tmp = make_temp_dir();
    const fs::path cg_root = tmp / "cgroup";
    const fs::path self_cg = tmp / "proc_self_cgroup";

    write_text(self_cg, "11:perf_event:/\n6:memory:/docker/abcdef\n");
    // Root has v1 PAGE_ALIGN(LONG_MAX) sentinel (~9223372036854771712 -> unlimited).
    write_text(cg_root / "memory/memory.limit_in_bytes", "9223372036854771712\n");
    // Leaf container caps at 16 GiB, usage 6 GiB, total_inactive_file 2 GiB -> eff used 4 GiB -> avail 12 GiB.
    write_text(cg_root / "memory/docker/abcdef/memory.limit_in_bytes", std::to_string(16ULL * kGiB) + "\n");
    write_text(cg_root / "memory/docker/abcdef/memory.usage_in_bytes", std::to_string(6ULL * kGiB) + "\n");
    write_text(cg_root / "memory/docker/abcdef/memory.stat",
               "cache " + std::to_string(2ULL * kGiB) + "\n"
               "total_inactive_file " + std::to_string(2ULL * kGiB) + "\n"
               "total_mapped_file 0\n");

    const auto r = probe_cgroup_room(/*host_total_bytes=*/64ULL * kGiB, self_cg.string(), cg_root.string());
    assert(r.limited);
    assert(r.limit_bytes == 16ULL * kGiB);
    assert(r.available_bytes == 12ULL * kGiB);
    fs::remove_all(tmp);
}

void test_unconstrained_host_not_marked_limited() {
    const fs::path tmp = make_temp_dir();
    const fs::path cg_root = tmp / "cgroup";
    const fs::path self_cg = tmp / "proc_self_cgroup";

    write_text(self_cg, "0::/user.slice\n");
    write_text(cg_root / "user.slice/memory.max", "max\n");
    write_text(cg_root / "memory.max", "max\n");

    const auto r = probe_cgroup_room(/*host_total_bytes=*/64ULL * kGiB, self_cg.string(), cg_root.string());
    assert(!r.limited);
    assert(r.limit_bytes == kUnlimited);
    assert(r.available_bytes == kUnlimited);
    fs::remove_all(tmp);
}

void test_windows_job_commit_room_from_limits() {
    using strata::platform::job_commit_room_from_limits;
    // Unconstrained -> UINT64_MAX
    assert(job_commit_room_from_limits(0, 4ULL * kGiB, 0, 8ULL * kGiB, 12ULL * kGiB) == UINT64_MAX);
    // Process limit 48 GiB, 8 GiB used, 12 GiB discrete VRAM reserve -> 28 GiB remaining host commit room
    assert(job_commit_room_from_limits(48ULL * kGiB, 8ULL * kGiB, 0, 0, 12ULL * kGiB) == 28ULL * kGiB);
    // Tighter job limit wins over looser process limit
    assert(job_commit_room_from_limits(64ULL * kGiB, 4ULL * kGiB, 32ULL * kGiB, 10ULL * kGiB, 2ULL * kGiB) == 20ULL * kGiB);
    // Saturates at 0 instead of underflowing
    assert(job_commit_room_from_limits(16ULL * kGiB, 14ULL * kGiB, 0, 0, 8ULL * kGiB) == 0ULL);
}

}  // namespace

int main() {
    test_gib_to_bytes_exact_binary();
    test_cgroup_v2_nested_parent_limit_and_reclaimable_file();
    test_cgroup_v1_hierarchy_and_unlimited_sentinel();
    test_unconstrained_host_not_marked_limited();
    test_windows_job_commit_room_from_limits();
    std::printf("cgroup_mem_test: ALL 5 SUITES PASSED\n");
    return 0;
}
