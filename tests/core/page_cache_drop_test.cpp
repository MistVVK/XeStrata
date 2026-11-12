// the file-cache give-back during and after the expert load (include/strata/core/page_cache_drop.hpp).  No GPU.  Cases: the flag and the
// switch, one range and one whole file of a temporary file, every error path (empty path, missing file, bad descriptor, empty range) ignored,
// the final call over a mixed list (a missing file and a repeated path among good ones), the meminfo reader, the draft line.  (The device rule
// is checked in src/core/arch_defaults_test.cpp.)
#include "strata/core/page_cache_drop.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace strata::core;
namespace pcd = strata::core::page_cache_drop;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

#if defined(__linux__)
// a temporary file of `bytes` bytes of a pattern, written through, flushed and read back once (so its pages are clean and cached); "" on failure
std::string make_file(size_t bytes) {
    char path[] = "/tmp/page_cache_drop_test_XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) return std::string();
    std::vector<unsigned char> buf(bytes);
    for (size_t i = 0; i < bytes; ++i) buf[i] = (unsigned char) (i * 131 + 7);
    bool ok = ::write(fd, buf.data(), bytes) == (ssize_t) bytes && ::fdatasync(fd) == 0;
    std::vector<unsigned char> back(bytes);
    ok = ok && ::pread(fd, back.data(), bytes, 0) == (ssize_t) bytes;
    ::close(fd);
    if (!ok) { ::unlink(path); return std::string(); }
    return std::string(path);
}
// how many pages of the file are resident (mincore); -1 when it cannot be told
long resident_pages(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size == 0) { ::close(fd); return -1; }
    void* m = ::mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { ::close(fd); return -1; }
    const size_t page = (size_t) ::sysconf(_SC_PAGESIZE);
    const size_t n = ((size_t) st.st_size + page - 1) / page;
    std::vector<unsigned char> vec(n);
    long r = ::mincore(m, (size_t) st.st_size, vec.data()) == 0 ? 0 : -1;
    if (r == 0) for (unsigned char c : vec) r += c & 1;
    ::munmap(m, (size_t) st.st_size);
    ::close(fd);
    return r;
}
#endif
}  // namespace

int main() {
    // 1. the flag and the switch
    {
        arena_cache_drop_flag().store(false);
        CHECK(!pcd::enabled());
#if defined(__linux__)
        ::unsetenv("STRATA_NO_ARENA_CACHE_DROP");
        arena_cache_drop_flag().store(true);
        CHECK(pcd::enabled());
        ::setenv("STRATA_NO_ARENA_CACHE_DROP", "1", 1);
        CHECK(!pcd::enabled() && pcd::off_by_env());
        ::setenv("STRATA_NO_ARENA_CACHE_DROP", "0", 1);
        CHECK(pcd::enabled() && !pcd::off_by_env());
        ::setenv("STRATA_NO_ARENA_CACHE_DROP", "", 1);
        CHECK(pcd::enabled() && !pcd::off_by_env());
        ::unsetenv("STRATA_NO_ARENA_CACHE_DROP");
#else
        arena_cache_drop_flag().store(true);
        CHECK(!pcd::enabled());   // never off Linux
#endif
        arena_cache_drop_flag().store(false);
    }
#if defined(__linux__)
    // 2. one range and one whole file of a real file: the calls work, the advised bytes add up, the file is unchanged
    {
        const size_t sz = 4u << 20;
        const std::string p = make_file(sz);
        CHECK(!p.empty());
        if (!p.empty()) {
            const int fd = ::open(p.c_str(), O_RDONLY);
            CHECK(fd >= 0);
            const uint64_t a0 = pcd::advised_bytes().load();
            CHECK(pcd::drop_range(fd, 0, 1u << 20));
            CHECK(pcd::advised_bytes().load() == a0 + (1u << 20));
            CHECK(pcd::drop_range(fd, 1u << 20, sz - (1u << 20)));
            CHECK(pcd::advised_bytes().load() == a0 + sz);
            CHECK(pcd::drop_range(fd, 100u << 20, 4096));   // beyond the end: advice, not an error
            const long before = resident_pages(p);
            CHECK(pcd::drop_file(p));
            const long after = resident_pages(p);
            CHECK(before < 0 || after < 0 || after <= before);   // (tmpfs keeps its pages: only "not more" is required)
            std::vector<unsigned char> back(sz);
            CHECK(::pread(fd, back.data(), sz, 0) == (ssize_t) sz);
            bool same = true;
            for (size_t i = 0; i < sz && same; ++i) same = back[i] == (unsigned char) (i * 131 + 7);
            CHECK(same);   // what a later reader needs is read again, unchanged
            ::close(fd);
            ::unlink(p.c_str());
        }
    }
    // 3. every error path is ignored and returns false; nothing is added to the advised bytes
    {
        const uint64_t a0 = pcd::advised_bytes().load();
        CHECK(!pcd::drop_range(-1, 0, 4096));
        CHECK(!pcd::drop_range(0x7fffffff, 0, 4096));   // not an open descriptor
        const std::string p = make_file(8192);
        const int fd = p.empty() ? -1 : ::open(p.c_str(), O_RDONLY);
        CHECK(fd >= 0);
        CHECK(!pcd::drop_range(fd, 0, 0));              // an empty range
        ::close(fd);
        CHECK(!pcd::drop_range(fd, 0, 4096));           // closed
        if (!p.empty()) ::unlink(p.c_str());
        CHECK(pcd::advised_bytes().load() == a0);
        CHECK(!pcd::drop_file(""));
        CHECK(!pcd::drop_file("/nonexistent/page_cache_drop_test/file.bin"));
        CHECK(!pcd::drop_file("/proc/self/mem/x"));      // a path that cannot be opened
    }
    // 4. the final call over a mixed list: a missing file, an empty path and a repeated path among good ones
    {
        const std::string a = make_file(1u << 20), b = make_file(1u << 20);
        CHECK(!a.empty() && !b.empty());
        const pcd::Outcome o = pcd::give_back({a, "/nonexistent/page_cache_drop_test/x.bin", "", b, a});
        CHECK(o.files == 3);    // a, the missing one, b (the empty path and the repeat are left out)
        CHECK(o.failed == 1);   // the missing one, ignored
        CHECK(o.ms >= 0.0);
        CHECK(o.cached_before > 0 && o.cached_after > 0);   // /proc/meminfo is readable
        const pcd::Outcome none = pcd::give_back({});
        CHECK(none.files == 0 && none.failed == 0);
        if (!a.empty()) ::unlink(a.c_str());
        if (!b.empty()) ::unlink(b.c_str());
    }
    // 5. the meminfo reader
    {
        CHECK(pcd::meminfo_bytes("Cached") > 0);
        CHECK(pcd::meminfo_bytes("MemFree") > 0);
        CHECK(pcd::meminfo_bytes("NoSuchKey") == 0);
        CHECK(pcd::meminfo_bytes("Cach") == 0);   // a prefix is not a key
    }
#endif
    // 6. the draft line: nothing when nothing was given back, a [DRAFT] otherwise, with the numbers and the switch
    {
        pcd::Outcome none;
        CHECK(pcd::draft_line(none).empty());
        pcd::Outcome o;
        o.advised_behind = 9ull << 30;
        o.cached_before = 7ull << 30;
        o.cached_after = 1ull << 30;
        o.ms = 520.0;
        const std::string s = pcd::draft_line(o);
        CHECK(s.rfind("[DRAFT] ", 0) == 0);
        CHECK(s.find("9.0 GiB") != std::string::npos);
        CHECK(s.find("6.0 GiB") != std::string::npos);
        CHECK(s.find("520 ms") != std::string::npos);
        CHECK(s.find("STRATA_NO_ARENA_CACHE_DROP=1") != std::string::npos);
        pcd::Outcome grew;   // the cache grew meanwhile and nothing was advised behind: nothing to say
        grew.cached_before = 1ull << 30;
        grew.cached_after = 2ull << 30;
        CHECK(pcd::draft_line(grew).empty());
    }
    if (failed != 0) {
        std::fprintf(stderr, "page_cache_drop_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("page_cache_drop_test: ok\n");
    return 0;
}
