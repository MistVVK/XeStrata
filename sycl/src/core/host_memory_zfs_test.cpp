// #5519 (SYCL): the Intel engine's arena memory reader (sycl/src/core/expert_source.cpp detail::host_available_memory) counts the
// ZFS ARC's share above its floor as available RAM, as ENG-1116's reader does (src/core/expert_source.cpp, the shared
// strata::core::zfs_arc_reclaimable_bytes): MemAvailable leaves the ARC out though ZFS gives it back. A container's memory limit
// still bounds the result. Fixture files only (meminfo, cgroup, arcstats); no GPU, never the real /proc.
#include "strata/core/expert_source.hpp"
#include "strata/core/conversation_memory.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using strata::core::detail::HostMemory;
using strata::core::detail::host_available_memory;

static int failures = 0;
static void require(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
#if !defined(__linux__)
    std::printf("skipped: the reader reads /proc only on Linux\n");
    return 77;
#else
    const uint64_t GiB = uint64_t(1) << 30;
    const fs::path dir = fs::temp_directory_path() / ("host_memory_zfs_test." + std::to_string((unsigned long) ::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir / "fs");
    auto put = [&](const std::string& rel, const std::string& text) {
        fs::create_directories((dir / rel).parent_path());
        std::ofstream(dir / rel) << text;
    };
    put("meminfo", "MemTotal:       134217728 kB\nMemAvailable:   " + std::to_string(100 * GiB / 1024) + " kB\n");
    const std::string mi = (dir / "meminfo").string(), cg = (dir / "cgroup").string(), root = (dir / "fs").string(),
                      arc = (dir / "arcstats").string();
    auto arcstats = [&](uint64_t c_min, uint64_t size) {
        put("arcstats", "13 1 0x01 123 33456 1 2\nname                            type data\nhits                            4    1234\n"
                        "c_min                           4    " + std::to_string(c_min) + "\nc_max                           4    " +
                        std::to_string(64 * GiB) + "\nsize                            4    " + std::to_string(size) + "\n");
    };
    HostMemory m;
    put("cgroup", "");
    require(host_available_memory(m, mi, cg, root, arc) && m.available == 100 * GiB, "no ZFS: MemAvailable alone");

    arcstats(4 * GiB, 24 * GiB);
    require(host_available_memory(m, mi, cg, root, arc) && m.available == 120 * GiB, "ZFS: MemAvailable + the ARC above its floor (20 GiB)");
    require(strata::core::zfs_arc_reclaimable_bytes(arc) == 20 * GiB, "the shared reader says the same 20 GiB");

    arcstats(8 * GiB, 6 * GiB);
    require(host_available_memory(m, mi, cg, root, arc) && m.available == 100 * GiB, "an ARC at or below its floor gives nothing");

    put("arcstats", "name type data\nc_min 4 4096\nsize 4 12x\n");
    require(host_available_memory(m, mi, cg, root, arc) && m.available == 100 * GiB, "a malformed number gives nothing");

    put("arcstats", "name type data\nsize 4 " + std::to_string(24 * GiB) + "\n");
    require(host_available_memory(m, mi, cg, root, arc) && m.available == 100 * GiB, "no c_min line gives nothing");

    // a container (cgroup v1: 8 GiB limit, 2 GiB used) on the ZFS host: the ARC credit never lifts it past the container's room
    arcstats(4 * GiB, 24 * GiB);
    put("cgroup", "4:memory:/box\n");
    put("fs/memory/box/memory.limit_in_bytes", std::to_string(8 * GiB) + "\n");
    put("fs/memory/box/memory.usage_in_bytes", std::to_string(2 * GiB) + "\n");
    require(host_available_memory(m, mi, cg, root, arc) && m.available == 6 * GiB && m.cgroup_limit == 8 * GiB,
            "a container's limit still bounds it (6 GiB of room)");

    fs::remove_all(dir);
    std::printf("%s\n", failures ? "host_memory_zfs_test: FAILED" : "host_memory_zfs_test: all ok");
    return failures ? 1 : 0;
#endif
}
