// available_host_bytes(): MemAvailable, the cgroup limits above this process, and the operator's cap, on fixture
// directories standing in for /proc/meminfo, /proc/self/cgroup and /sys/fs/cgroup.  CPU only, no GPU.
#include "strata/core/conversation_memory.hpp"

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t GiB = 1ull << 30;

struct Fixture {
    fs::path dir;
    Fixture() {
        static int n = 0;
        dir = fs::temp_directory_path() / ("strata_memory_guard_" + std::to_string((long) ::getpid()) + "_" + std::to_string(n++));
        fs::remove_all(dir);
        fs::create_directories(dir / "cgroup");
        // this process's own mappings: none unless a test says so (the real /proc/self/status is never read)
        std::ofstream(dir / "status", std::ios::binary) << "Name:\tstrata\nRssAnon:\t    1024 kB\nRssFile:\t       0 kB\nRssShmem:\t       0 kB\n";
    }
    ~Fixture() { std::error_code ec; fs::remove_all(dir, ec); }
    void put(const std::string& rel, const std::string& text) {
        const fs::path p = dir / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p, std::ios::binary) << text;
    }
    void put(const std::string& rel, uint64_t n) { put(rel, std::to_string(n) + "\n"); }
    void meminfo(uint64_t total, uint64_t available) {
        put("meminfo", "MemTotal:       " + std::to_string(total / 1024) + " kB\nMemFree: 1 kB\nMemAvailable:   " +
                           std::to_string(available / 1024) + " kB\n");
    }
    void self(const std::string& text) { put("self_cgroup", text + "\n"); }
    MemoryProbePaths paths() const {
        return {(dir / "meminfo").string(), (dir / "self_cgroup").string(), (dir / "cgroup").string(), (dir / "status").string()};
    }
    void rss_file(uint64_t bytes) { put("status", "Name:\tstrata\nRssFile:\t" + std::to_string(bytes / 1024) + " kB\n"); }
    std::optional<HostMemoryReading> sample(uint64_t explicit_limit = 0) const { return sample_host_memory(paths(), explicit_limit); }
};

}  // namespace

int main() {
    // no cgroup limit anywhere ("max" at every level, and the root without memory files): plain MemAvailable
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/a/b");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.current", 90 * GiB);
        f.put("cgroup/a/memory.max", "max\n"); f.put("cgroup/a/memory.current", 60 * GiB);
        f.put("cgroup/a/b/memory.max", "max\n"); f.put("cgroup/a/b/memory.current", 50 * GiB);
        const auto r = f.sample();
        check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo, "max everywhere is plain MemAvailable");
        check(r->limit == 120 * GiB && r->current == 95 * GiB && r->mem_available == 25 * GiB, "meminfo reading: MemTotal and used");
    }
    // nested v2 limits: the tightest room of the group and its ancestors wins, "max" levels are skipped
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("0::/a/b");
        f.put("cgroup/memory.max", 100 * GiB); f.put("cgroup/memory.current", 90 * GiB);   // 10 GiB of room
        f.put("cgroup/a/memory.max", 60 * GiB); f.put("cgroup/a/memory.current", 58 * GiB);  // 2 GiB of room
        f.put("cgroup/a/b/memory.max", "max\n"); f.put("cgroup/a/b/memory.current", 50 * GiB);
        auto r = f.sample();
        check(r && r->available == 2 * GiB && r->source == MemorySource::cgroup, "tightest ancestor wins");
        check(r->limit == 60 * GiB && r->current == 58 * GiB, "reading names that group's limit and usage");
        f.put("cgroup/a/memory.current", 59 * GiB + 512 * MiB);
        r = f.sample();
        check(r && r->available == 512 * MiB, "the room follows memory.current");
        f.put("cgroup/a/memory.current", 61 * GiB);
        r = f.sample();
        check(r && r->available == 0 && r->source == MemorySource::cgroup, "usage over the limit is a known zero");
        f.put("cgroup/a/memory.current", 10 * GiB);
        r = f.sample();
        check(r && r->available == 10 * GiB && r->limit == 100 * GiB, "the root's room wins once the group has more");
        check(r->source == MemorySource::cgroup, "source cgroup");
        // the room under a limit larger than MemAvailable is bounded by MemAvailable
        f.meminfo(120 * GiB, 4 * GiB);
        r = f.sample();
        check(r && r->available == 4 * GiB && r->source == MemorySource::meminfo, "MemAvailable bounds a roomy cgroup");
    }
    // memory.high below memory.max counts, and a high of "max" does not
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("0::/g");
        f.put("cgroup/g/memory.max", 100 * GiB); f.put("cgroup/g/memory.high", 99 * GiB + 900 * MiB);
        f.put("cgroup/g/memory.current", 99 * GiB);
        auto r = f.sample();
        check(r && r->available == 900 * MiB && r->limit == 99 * GiB + 900 * MiB, "memory.high lower than memory.max");
        f.put("cgroup/g/memory.high", "max\n");
        r = f.sample();
        check(r && r->available == 1 * GiB && r->limit == 100 * GiB, "memory.high max leaves memory.max");
        f.put("cgroup/g/memory.max", "max\n"); f.put("cgroup/g/memory.high", 90 * GiB);
        r = f.sample();
        check(r && r->available == 0, "memory.high alone, already exceeded");
    }
    // the operator's cap: the container cannot see its real limit (memory.max is "max" at the namespace root)
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);   // the stale view: MemAvailable says 25 GiB
        f.self("0::/system.slice/strata.service");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.high", "max\n");
        f.put("cgroup/memory.current", 99 * GiB + 724 * MiB);
        f.put("cgroup/system.slice/strata.service/memory.max", "max\n");
        f.put("cgroup/system.slice/strata.service/memory.current", 97 * GiB);
        check(f.sample() && f.sample()->available == 25 * GiB && f.sample()->source == MemorySource::meminfo,
              "without the cap the stale MemAvailable is believed");
        auto r = f.sample(100 * GiB);
        check(r && r->available == 300 * MiB && r->source == MemorySource::flag, "the cap minus the root's memory.current");
        check(r->limit == 100 * GiB && r->current == 99 * GiB + 724 * MiB, "reading names the cap and the usage");
        r = f.sample(99 * GiB);
        check(r && r->available == 0 && r->source == MemorySource::flag, "usage over the cap is a known zero");
        r = f.sample(200 * GiB);
        check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo, "a cap above everything else changes nothing");
        // a real limit visible and tighter than the cap: the tighter one
        f.put("cgroup/system.slice/memory.max", 98 * GiB); f.put("cgroup/system.slice/memory.current", 97 * GiB + 512 * MiB);
        r = f.sample(100 * GiB);
        check(r && r->available == 300 * MiB && r->source == MemorySource::flag, "cap tighter than the visible cgroup limit");
        r = f.sample(150 * GiB);
        check(r && r->available == 512 * MiB && r->source == MemorySource::cgroup, "visible cgroup limit tighter than the cap");
        // equal rooms: the operator's number is the one reported
        r = f.sample(99 * GiB + 724 * MiB + 512 * MiB);
        check(r && r->available == 512 * MiB && r->source == MemorySource::flag, "a tie goes to the operator's cap");
        // no memory.current at the root: used = MemTotal - MemAvailable
        fs::remove(f.dir / "cgroup/system.slice/memory.max");
        fs::remove(f.dir / "cgroup/memory.current");
        r = f.sample(100 * GiB);
        check(r && r->available == 5 * GiB && r->source == MemorySource::flag && r->current == 95 * GiB,
              "without the root's memory.current the meminfo-derived usage is used");
        // ... and with no MemTotal either, the usage is unknown: unknown, not a guess
        f.put("meminfo", "MemAvailable: 1024 kB\n");
        check(!f.sample(100 * GiB), "cap with unknown usage is unknown");
        check(f.sample().has_value(), "no cap, no usage needed");
    }
    // malformed or unusable files: without a cap the guard falls back to MemAvailable alone (meminfo source, once-warned);
    // with a cap the sample is unknown (admission refuses), never a guess
    {
        const char* bad_values[] = {"garbage\n", "", "\n", "-5\n", "+5\n", "12x\n", "1 2\n", "18446744073709551616\n",
                                    "0x10\n", "MAX\n"};
        for (const char* file : {"memory.max", "memory.high", "memory.current"}) {
            for (const char* text : bad_values) {
                Fixture f;
                f.meminfo(120 * GiB, 25 * GiB);
                f.self("0::/g");
                f.put("cgroup/g/memory.max", 100 * GiB); f.put("cgroup/g/memory.high", "max\n");
                f.put("cgroup/g/memory.current", 50 * GiB);
                f.put(std::string("cgroup/g/") + file, text);
                {
                    const auto r = f.sample();
                    check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo,
                          "malformed cgroup file without a cap: MemAvailable alone");
                }
                check(!f.sample(100 * GiB), "malformed cgroup file is unknown with the cap too");
            }
        }
        Fixture f;
        f.self("0::/g");
        f.put("cgroup/g/memory.max", 100 * GiB); f.put("cgroup/g/memory.current", 50 * GiB);
        f.meminfo(120 * GiB, 25 * GiB);
        check(f.sample() && f.sample()->available == 25 * GiB, "the fixture itself is sound");
        fs::remove(f.dir / "cgroup/g/memory.current");
        check(f.sample() && f.sample()->available == 25 * GiB && f.sample()->source == MemorySource::meminfo,
              "a finite limit with no usage beside it, no cap: MemAvailable alone");
        check(!f.sample(100 * GiB), "... and with the cap it is unknown");
        f.put("cgroup/g/memory.max", "max\n");
        check(f.sample() && f.sample()->available == 25 * GiB, "no finite limit and no usage: the level is skipped");
        f.put("cgroup/g/memory.current", "max\n");
        check(f.sample() && f.sample()->source == MemorySource::meminfo, "memory.current is never \"max\": MemAvailable alone");
        check(!f.sample(100 * GiB), "... unknown with the cap");
        f.put("cgroup/g/memory.current", 1ull);
        for (const char* text : {"", "MemAvailable: 12 MB\n", "MemAvailable: 1x kB\n", "MemTotal: 5 kB\n",
                                 "MemAvailable: 5 kB\nMemAvailable: 6 kB\n"}) {
            f.put("meminfo", text);
            check(!f.sample(), "malformed MemAvailable is unknown");
        }
        fs::remove(f.dir / "meminfo");
        check(!f.sample(), "no meminfo is unknown");
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/a/../g");
        check(f.sample() && f.sample()->source == MemorySource::meminfo, "a .. component in the cgroup path: MemAvailable alone");
        check(!f.sample(100 * GiB), "a .. component with the cap is refused");
        f.self("garbage without colons");
        check(f.sample() && f.sample()->available == 25 * GiB, "unparsable /proc/self/cgroup lines are ignored");
    }
    // a path from /proc/self/cgroup that this mount does not show: the deepest group that is there (at worst the root)
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/lxc/105/ns/deep");
        f.put("cgroup/memory.max", 100 * GiB); f.put("cgroup/memory.current", 99 * GiB);
        auto r = f.sample();
        check(r && r->available == 1 * GiB && r->source == MemorySource::cgroup, "an invisible group falls back to the mount's root");
        f.put("cgroup/lxc/memory.max", 100 * GiB); f.put("cgroup/lxc/memory.current", 99 * GiB + 256 * MiB);
        r = f.sample();
        check(r && r->available == 1 * GiB - 256 * MiB, "the deepest visible prefix is read too");
    }
    // no /proc/self/cgroup file: the mount's root group; no mount: MemAvailable alone
    {
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.put("cgroup/memory.max", 100 * GiB); f.put("cgroup/memory.current", 98 * GiB);
        auto r = f.sample();
        check(r && r->available == 2 * GiB, "no cgroup file: the root group");
        fs::remove_all(f.dir / "cgroup");
        r = f.sample();
        check(r && r->available == 25 * GiB && r->source == MemorySource::meminfo, "no cgroup mount: MemAvailable");
        check(f.sample(100 * GiB) && f.sample(100 * GiB)->source == MemorySource::flag &&
                  f.sample(100 * GiB)->available == 5 * GiB, "no cgroup mount, a cap: meminfo-derived usage");
    }
    // cgroup v1
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("12:cpu,cpuacct:/x\n4:memory:/m/n\n1:name=systemd:/x");
        f.put("cgroup/memory/memory.limit_in_bytes", 9223372036854771712ull);
        f.put("cgroup/memory/memory.usage_in_bytes", 90 * GiB);
        f.put("cgroup/memory/m/memory.limit_in_bytes", 64 * GiB); f.put("cgroup/memory/m/memory.usage_in_bytes", 60 * GiB);
        f.put("cgroup/memory/m/n/memory.limit_in_bytes", 9223372036854771712ull);
        f.put("cgroup/memory/m/n/memory.usage_in_bytes", 1 * GiB);
        auto r = f.sample();
        check(r && r->available == 4 * GiB && r->source == MemorySource::cgroup && r->limit == 64 * GiB, "v1: the tightest limit, unlimited skipped");
        r = f.sample(100 * GiB);
        check(r && r->available == 4 * GiB && r->source == MemorySource::cgroup, "v1 with a looser cap");
        r = f.sample(92 * GiB);
        check(r && r->available == 2 * GiB && r->source == MemorySource::flag && r->current == 90 * GiB, "v1 with a tighter cap: the v1 root's usage");
        f.put("cgroup/memory/m/memory.limit_in_bytes", "junk\n");
        check(f.sample() && f.sample()->source == MemorySource::meminfo, "v1 malformed limit, no cap: MemAvailable alone");
        check(!f.sample(100 * GiB), "v1 malformed limit with the cap is unknown");
    }
    // hybrid: a v2 line, but the memory controller lives in the v1 tree
    {
        Fixture f;
        f.meminfo(120 * GiB, 80 * GiB);
        f.self("5:memory:/g\n0::/g");
        f.put("cgroup/cgroup.controllers", "cpu\n");
        f.put("cgroup/g/cpu.max", "max 100000\n");
        f.put("cgroup/memory/g/memory.limit_in_bytes", 10 * GiB); f.put("cgroup/memory/g/memory.usage_in_bytes", 9 * GiB);
        auto r = f.sample();
        check(r && r->available == 1 * GiB && r->source == MemorySource::cgroup, "hybrid falls through to v1");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.current", 1 * GiB);
        r = f.sample();
        check(r && r->available == 80 * GiB, "v2 memory files win over v1 when both exist");
    }
    // the one credited kind of reclaimable memory: clean inactive file cache from the same group's memory.stat
    {
        const auto stat = [](uint64_t inactive, uint64_t active, uint64_t shmem, uint64_t dirty = 0, uint64_t wb = 0) {
            return "anon 1\nfile 2\nshmem " + std::to_string(shmem) + "\ninactive_file " + std::to_string(inactive) +
                   "\nactive_file " + std::to_string(active) + "\nfile_dirty " + std::to_string(dirty) +
                   "\nfile_writeback " + std::to_string(wb) + "\nunevictable 5\n";
        };
        Fixture f;
        f.meminfo(120 * GiB, 25 * GiB);
        f.self("0::/system.slice/strata.service");
        f.put("cgroup/memory.max", "max\n"); f.put("cgroup/memory.high", "max\n");
        f.put("cgroup/memory.current", 98 * GiB + 324 * MiB);
        f.put("cgroup/memory.stat", stat(3 * GiB + 800 * MiB, 27 * GiB, 65 * GiB));
        f.put("cgroup/system.slice/strata.service/memory.max", "max\n");
        f.put("cgroup/system.slice/strata.service/memory.current", 90 * GiB);
        // the operator's cap: the root's memory.current less the root's inactive_file; active_file and shmem are not credited
        auto r = f.sample(99 * GiB);
        check(r && r->source == MemorySource::flag && r->available == 99 * GiB - (98 * GiB + 324 * MiB - 3 * GiB - 800 * MiB),
              "the cap path credits the root's inactive_file");
        check(r->credit == 3 * GiB + 800 * MiB && r->current == 98 * GiB + 324 * MiB, "reading shows the credit and the raw usage");
        f.put("cgroup/memory.stat", stat(0, 27 * GiB, 65 * GiB));
        r = f.sample(99 * GiB);
        check(r && r->available == 700 * MiB && r->credit == 0, "only inactive_file counts: active_file and shmem give nothing");
        // dirty and writeback pages are not clean: taken off the credit
        f.put("cgroup/memory.stat", stat(3 * GiB, 0, 0, 1 * GiB, 512 * MiB));
        r = f.sample(99 * GiB);
        check(r && r->credit == 1 * GiB + 512 * MiB && r->available == 700 * MiB + 1 * GiB + 512 * MiB, "dirty and writeback come off");
        f.put("cgroup/memory.stat", stat(3 * GiB, 0, 0, 2 * GiB, 2 * GiB));
        r = f.sample(99 * GiB);
        check(r && r->credit == 0 && r->available == 700 * MiB, "dirty plus writeback over inactive_file leaves no credit");
        // inactive_file larger than the usage: the credit stops at the usage, headroom never exceeds the cap
        f.put("cgroup/memory.stat", stat(500 * GiB, 0, 0));
        f.meminfo(300 * GiB, 200 * GiB);   // so the cap's room (not MemAvailable) is the smallest term
        r = f.sample(99 * GiB);
        check(r && r->credit == 98 * GiB + 324 * MiB && r->available == 99 * GiB && r->source == MemorySource::flag,
              "inactive_file above the usage cannot lift headroom over the cap");
        // no memory.stat, or one that cannot be read: no credit, still a sample
        fs::remove(f.dir / "cgroup/memory.stat");
        r = f.sample(99 * GiB);
        check(r && r->available == 700 * MiB && r->credit == 0, "no memory.stat: no credit");
        for (const char* text : {"", "garbage\n", "inactive_file\n", "inactive_file -1\n", "inactive_file 12x\n",
                                 "inactive_file 1\ninactive_file 2\n", "inactive_file 99999999999999999999\n",
                                 "inactive_file 4096\nfile_dirty zz\n"}) {
            f.put("cgroup/memory.stat", text);
            r = f.sample(99 * GiB);
            check(r && r->available == 700 * MiB && r->credit == 0, "unparsable memory.stat: no credit, not unknown");
        }
        // a cgroup limit's own group is credited from its own memory.stat
        f.put("cgroup/memory.stat", stat(0, 0, 0));
        f.put("cgroup/system.slice/strata.service/memory.max", 92 * GiB);
        f.put("cgroup/system.slice/strata.service/memory.stat", stat(1 * GiB, 0, 0));
        r = f.sample();
        check(r && r->source == MemorySource::cgroup && r->available == 3 * GiB && r->credit == 1 * GiB &&
                  r->limit == 92 * GiB && r->current == 90 * GiB, "a cgroup limit credits its own group's inactive_file");
        f.put("cgroup/system.slice/strata.service/memory.stat", stat(500 * GiB, 0, 0));
        f.meminfo(300 * GiB, 200 * GiB);
        r = f.sample();
        check(r && r->available == 92 * GiB && r->credit == 90 * GiB, "cgroup term: the credit stops at the usage too");
        // cgroup v1: total_inactive_file (and total_dirty / total_writeback)
        Fixture g;
        g.meminfo(120 * GiB, 80 * GiB);
        g.self("4:memory:/m");
        g.put("cgroup/memory/m/memory.limit_in_bytes", 64 * GiB); g.put("cgroup/memory/m/memory.usage_in_bytes", 62 * GiB);
        g.put("cgroup/memory/memory.limit_in_bytes", 9223372036854771712ull); g.put("cgroup/memory/memory.usage_in_bytes", 62 * GiB);
        g.put("cgroup/memory/m/memory.stat", "cache 1\ninactive_file 7\ntotal_inactive_file " + std::to_string(3 * GiB) +
                                                  "\ntotal_dirty " + std::to_string(1 * GiB) + "\ntotal_writeback 0\n");
        r = g.sample();
        check(r && r->available == 2 * GiB + 2 * GiB && r->credit == 2 * GiB && r->source == MemorySource::cgroup,
              "v1 credits total_inactive_file less total_dirty");
        g.put("cgroup/memory/m/memory.stat", "inactive_file " + std::to_string(3 * GiB) + "\n");
        r = g.sample();
        check(r && r->available == 2 * GiB && r->credit == 0, "v1 ignores the non-total inactive_file");
    }
    // the credit's two forms from one memory.stat text
    {
        const std::string v2 = "anon 9\nfile 999\nshmem 500\nfile_mapped 77\nactive_file 30\ninactive_file 20\n"
                               "file_dirty 3\nfile_writeback 2\nunevictable 400\nactive_anon 1\n";
        CacheCredit c = cache_credit_from_stat(v2, 1000, false);
        check(c.inactive == 15 && c.lists == 45, "inactive 20 and lists 50, each less dirty 3 and writeback 2");
        check(cache_credit_from_stat(v2, 40, false).lists == 35 && cache_credit_from_stat(v2, 40, false).inactive == 15,
              "the lists figure stops at memory.current too");
        c = cache_credit_from_stat("inactive_file 20\nfile 999\nshmem 900\nunevictable 900\n", 1000, false);
        check(c.inactive == 20 && c.lists == 20, "no active_file line: lists = inactive; file, shmem and unevictable never count");
        c = cache_credit_from_stat("inactive_file 20\nactive_file x\n", 1000, false);
        check(c.inactive == 20 && c.lists == 20, "a malformed active_file adds nothing and does not cost the inactive credit");
        c = cache_credit_from_stat("inactive_file 20\nactive_file 5\nactive_file 6\n", 1000, false);
        check(c.inactive == 20 && c.lists == 20, "a repeated active_file adds nothing");
        c = cache_credit_from_stat("active_file 30\n", 1000, false);
        check(c.inactive == 0 && c.lists == 0, "no inactive_file: no credit at all, as before");
        c = cache_credit_from_stat("inactive_file 18446744073709551615\nactive_file 18446744073709551615\n", 1000, false);
        check(c.inactive == 1000 && c.lists == 1000, "no overflow in inactive + active");
        c = cache_credit_from_stat("active_file 30\ninactive_file 20\ntotal_active_file 300\ntotal_inactive_file 200\n"
                                   "total_dirty 10\ntotal_writeback 0\n", 10000, true);
        check(c.inactive == 190 && c.lists == 490, "v1 reads the total_ names");
        const std::optional<uint64_t> total11(11), none0(0), own3(3), own20(20);
        check(cache_credit_for_limit({5, 9}, 10, total11, false, none0) == 9, "a limit below MemTotal: lists");
        check(cache_credit_for_limit({5, 9}, 10, total11, false, own3) == 6, "... less the process's own mappings");
        check(cache_credit_for_limit({5, 9}, 10, total11, false, own20) == 0, "... floor 0");
        check(cache_credit_for_limit({5, 9}, 10, total11, false, std::nullopt) == 5, "own mappings unknown: inactive");
        check(cache_credit_for_limit({5, 9}, 11, total11, false, none0) == 5, "a limit at MemTotal: inactive");
        check(cache_credit_for_limit({5, 9}, 12, total11, false, own3) == 5, "a limit above MemTotal: inactive, own mappings ignored");
        check(cache_credit_for_limit({5, 9}, 10, std::nullopt, false, none0) == 5, "MemTotal unknown: inactive");
        check(cache_credit_for_limit({5, 9}, 10, total11, true, none0) == 5, "the switch: inactive");
        // RssFile from /proc/self/status, less the locked file bytes
        const std::string status = "Name:\tstrata\nVmLck:\t  999 kB\nRssAnon:\t 100 kB\nRssFile:\t 2048 kB\nRssShmem:\t 64 kB\n";
        check(own_evictable_mapped_bytes(status, 0) == std::optional<uint64_t>(2 * MiB), "RssFile in bytes");
        check(own_evictable_mapped_bytes(status, 1 * MiB) == std::optional<uint64_t>(1 * MiB), "locked file bytes come off");
        check(own_evictable_mapped_bytes(status, 3 * MiB) == std::optional<uint64_t>(0), "floor 0");
        check(!own_evictable_mapped_bytes("Name:\tstrata\nRssAnon:\t 1 kB\n", 0), "no RssFile: unknown");
        check(!own_evictable_mapped_bytes("RssFile:\t 12 MB\n", 0) && !own_evictable_mapped_bytes("RssFile:\t x kB\n", 0) &&
                  !own_evictable_mapped_bytes("RssFile:\t 1 kB\nRssFile:\t 2 kB\n", 0), "malformed RssFile: unknown");
        check(locked_own_file_bytes() == 0, "nothing locked by default");
        set_locked_own_file_bytes(5 * MiB);
        check(locked_own_file_bytes() == 5 * MiB, "locked bytes round trip");
        set_locked_own_file_bytes(0);
        check(mem_credit_inactive_only("1") && !mem_credit_inactive_only("0") && !mem_credit_inactive_only(nullptr) &&
                  !mem_credit_inactive_only(""), "STRATA_MEM_CREDIT_INACTIVE=1 only");
    }
    // a container (limit below MemTotal) restarted with its model file's pages still cached: the same figure cold and warm
    {
        const auto stat = [](uint64_t active, uint64_t inactive) {
            return "anon " + std::to_string(5 * GiB) + "\nfile " + std::to_string(active + inactive) + "\nactive_file " +
                   std::to_string(active) + "\ninactive_file " + std::to_string(inactive) + "\nfile_dirty 0\nfile_writeback 0\n";
        };
        const uint64_t limit = 57 * GiB + 760 * MiB, table = 26 * GiB + 820 * MiB;
        Fixture f;
        f.meminfo(503 * GiB, 420 * GiB);   // the host's view
        f.self("0::/");
        f.put("cgroup/memory.max", limit);
        // cold: 7 GiB used, 1 GiB of it inactive cache
        f.put("cgroup/memory.current", 7 * GiB); f.put("cgroup/memory.stat", stat(0, 1 * GiB));
        const auto cold = f.sample();
        check(cold && cold->source == MemorySource::cgroup && cold->available == limit - 6 * GiB, "cold start: limit - (current - cache)");
        // warm, the table's pages on the active list (the pod: credited 1.0 GiB, available 26.7 GiB under the old rule)
        f.put("cgroup/memory.current", 7 * GiB + table); f.put("cgroup/memory.stat", stat(table, 1 * GiB));
        const auto warm = f.sample();
        check(warm && warm->available == cold->available && warm->credit == table + 1 * GiB, "warm, active: the same figure");
        f.put("cgroup/memory.stat", stat(1 * GiB, table));
        check(f.sample() && f.sample()->available == cold->available, "warm, inactive: the same figure");
        // the way back
#ifdef _WIN32
        _putenv_s("STRATA_MEM_CREDIT_INACTIVE", "1");
#else
        setenv("STRATA_MEM_CREDIT_INACTIVE", "1", 1);
#endif
        f.put("cgroup/memory.stat", stat(table, 1 * GiB));
        const auto old = f.sample();
        check(old && old->available == limit - (7 * GiB + table - 1 * GiB) && old->credit == 1 * GiB,
              "STRATA_MEM_CREDIT_INACTIVE=1: active pages are not credited, as before");
#ifdef _WIN32
        _putenv_s("STRATA_MEM_CREDIT_INACTIVE", "");
#else
        unsetenv("STRATA_MEM_CREDIT_INACTIVE");
#endif
        check(f.sample() && f.sample()->available == cold->available, "switch cleared: the lists rule again");
        // the engine maps the table (ram mode, lock refused): its resident pages are its own, not free - cold and warm alike
        f.rss_file(table + 300 * MiB);   // the table and 300 MiB of libraries
        const auto mapped = f.sample();
        check(mapped && mapped->credit == 1 * GiB - 300 * MiB && mapped->available == limit - (7 * GiB + table - (1 * GiB - 300 * MiB)),
              "own mapped table and libraries: not credited");
        // the lock worked: the table is on the unevictable list, not on the file lists, and is added back
        f.put("cgroup/memory.stat", stat(0, 1 * GiB));   // the locked table is in neither list
        set_locked_own_file_bytes(table);
        const auto locked = f.sample();
        check(locked && locked->credit == 1 * GiB - 300 * MiB, "locked table: only the libraries come off the file credit");
        set_locked_own_file_bytes(0);
        // no status file: the own mappings cannot be told, the inactive credit is taken
        f.put("cgroup/memory.stat", stat(table, 1 * GiB));
        fs::remove(f.dir / "status");
        check(f.sample() && f.sample()->credit == 1 * GiB, "no /proc/self/status: inactive credit");
        f.rss_file(0);
        // a limit at or above MemTotal (not a container's view): the inactive rule, bit for bit
        f.meminfo(57 * GiB, 50 * GiB);
        const auto at = f.sample();
        check(at && at->credit == 1 * GiB && at->available == limit - (7 * GiB + table - 1 * GiB) && at->source == MemorySource::cgroup,
              "limit above MemTotal: active pages are not credited, as before");
        f.meminfo(limit, 50 * GiB);
        check(f.sample() && f.sample()->credit == 1 * GiB, "limit equal to MemTotal: inactive only");
        // no limit at all: MemAvailable, untouched by memory.stat
        f.meminfo(503 * GiB, 420 * GiB);
        f.put("cgroup/memory.max", "max\n");
        check(f.sample() && f.sample()->available == 420 * GiB && f.sample()->source == MemorySource::meminfo && f.sample()->credit == 0,
              "no limit: MemAvailable");
        // the operator's cap keeps the inactive rule (its usage is the namespace root's)
        check(f.sample(40 * GiB) && f.sample(40 * GiB)->credit == 1 * GiB && f.sample(40 * GiB)->source == MemorySource::flag,
              "--memory-limit-mib: inactive only");
    }
    // the expert arena's figure under a container's limit: the same file-list credit (never shmem, never locked pages), its own margin
    {
        // the 24 GiB scope with a 3 GiB memfd held (l620, 11-14): the earlier figure credited the shmem in "file" and an
        // allocation up to it was OOM-killed; this one leaves it charged
        const std::string memfd = "anon 1048576\nfile " + std::to_string(3 * GiB) + "\nfile_mapped " + std::to_string(3 * GiB) +
                                  "\nshmem " + std::to_string(3 * GiB) + "\nactive_file 0\ninactive_file 0\nunevictable 0\n"
                                  "file_dirty 0\nfile_writeback 0\n";
        const std::optional<uint64_t> host(251 * GiB);
        ArenaCgroupRoom a = arena_cgroup_room(24 * GiB, 3 * GiB + 20 * MiB, memfd, host, false);
        check(a.applies && a.credit == 0 && a.room == 21 * GiB - 20 * MiB, "memfd: the shmem stays charged");
        // active and inactive file pages count, less min(half, limit / 64); locked pages (unevictable) do not
        const std::string cache = "file " + std::to_string(14 * GiB) + "\nactive_file " + std::to_string(8 * GiB) + "\ninactive_file " +
                                  std::to_string(4 * GiB) + "\nunevictable " + std::to_string(2 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n";
        a = arena_cgroup_room(48 * GiB, 16 * GiB, cache, host, false);
        check(a.applies && a.credit == 12 * GiB - 48 * GiB / 64 && a.room == 44 * GiB - 48 * GiB / 64, "file lists less the margin");
        a = arena_cgroup_room(48 * GiB, 16 * GiB, cache, std::optional<uint64_t>(48 * GiB), false);
        check(!a.applies, "a limit at MemTotal: the earlier rule (caller)");
        check(!arena_cgroup_room(48 * GiB, 16 * GiB, cache, std::nullopt, false).applies, "MemTotal unknown: the earlier rule");
        check(!arena_cgroup_room(48 * GiB, 16 * GiB, cache, host, true).applies, "STRATA_MEM_CREDIT_INACTIVE=1: the earlier rule");
        a = arena_cgroup_room(8 * GiB, 9 * GiB, "inactive_file 0\n", host, false);
        check(a.applies && a.room == 0, "usage over the limit: 0");
        a = arena_cgroup_room(48 * GiB, 16 * GiB, "", host, false);
        check(a.applies && a.credit == 0 && a.room == 32 * GiB, "no memory.stat text: no credit");
    }
    // the limit on a parent level, credited from that level's own memory.stat; memory.high below memory.max
    {
        Fixture f;
        f.meminfo(256 * GiB, 200 * GiB);
        f.self("0::/a/b");
        f.put("cgroup/a/memory.max", 24 * GiB); f.put("cgroup/a/memory.current", 20 * GiB);
        f.put("cgroup/a/memory.stat", "active_file " + std::to_string(10 * GiB) + "\ninactive_file " + std::to_string(2 * GiB) + "\n");
        f.put("cgroup/a/b/memory.max", "max\n"); f.put("cgroup/a/b/memory.current", 19 * GiB);
        f.put("cgroup/a/b/memory.stat", "active_file 0\ninactive_file 0\n");
        auto r = f.sample();
        check(r && r->source == MemorySource::cgroup && r->limit == 24 * GiB && r->credit == 12 * GiB && r->available == 16 * GiB,
              "parent limit: 24 - (20 - 12)");
        f.put("cgroup/a/memory.high", 18 * GiB);
        r = f.sample();
        check(r && r->limit == 18 * GiB && r->available == 10 * GiB && r->credit == 12 * GiB, "memory.high below max: 18 - (20 - 12)");
    }
    // cgroup v1 under a limit below MemTotal: total_active_file counts too
    {
        Fixture g;
        g.meminfo(120 * GiB, 80 * GiB);
        g.self("4:memory:/m");
        g.put("cgroup/memory/m/memory.limit_in_bytes", 64 * GiB); g.put("cgroup/memory/m/memory.usage_in_bytes", 62 * GiB);
        g.put("cgroup/memory/m/memory.stat", "total_active_file " + std::to_string(4 * GiB) + "\ntotal_inactive_file " +
                                                  std::to_string(3 * GiB) + "\ntotal_dirty " + std::to_string(1 * GiB) + "\n");
        const auto r = g.sample();
        check(r && r->credit == 6 * GiB && r->available == 8 * GiB && r->source == MemorySource::cgroup, "v1: 64 - (62 - (7 - 1))");
    }
    // the process-wide setting and the real files
    {
        check(memory_limit_bytes() == 0, "no cap by default");
        set_memory_limit_bytes(123 * MiB);
        check(memory_limit_bytes() == 123 * MiB, "cap round trip");
#if defined(__linux__)
        set_memory_limit_bytes(1);   // one byte: whatever this machine uses, nothing is left
        const auto r = sample_host_memory();
        check(!r || (r->available == 0 && r->source == MemorySource::flag && available_host_bytes() == 0u),
              "the real files honour the process-wide cap");
#endif
        set_memory_limit_bytes(0);
        const auto none = available_host_bytes();
        check(!none || conversation_memory_admit(none, 0, 0), "the real files without a cap: bytes or unknown");
        check(std::string(memory_source_name(MemorySource::flag)) == "flag" &&
                  std::string(memory_source_name(MemorySource::cgroup)) == "cgroup" &&
                  std::string(memory_source_name(MemorySource::meminfo)) == "meminfo", "source names");
    }
    std::printf("memory_guard_test: %d checks passed\n", checks);
    return 0;
}
