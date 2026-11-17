// ple_issue_bench: host allocations and CPU time of PleReader::issue() (PR #4910), before/after, on a synthetic table.
// CPU and a temp file only, no model, no GPU.  Not a ctest target; build it against the reader under test:
//
//   g++ -O2 -std=c++20 -pthread -Iinclude tests/ngram/ple_issue_bench.cpp src/ngram/ple_reader.cpp \
//       src/platform/direct_file.cpp src/platform/memory.cpp -o /tmp/ple_issue_bench
//   /tmp/ple_issue_bench [--dir D] [--calls N] [--threaded 0|1]
//
// For each batch size it issues `calls` tickets of random rows (no row cache, so every row misses), collecting each before
// the next, and prints: heap allocations made by the calling thread inside issue() per call, thread CPU time inside issue()
// per call, and dedup_rows per call (the same on both builds: the rows that share a page).  Allocation counting replaces the
// global operator new, so the numbers are exact for this thread.
#include "strata/ngram/ple_reader.hpp"

#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <string>
#include <vector>

namespace ng = strata::ngram;

static thread_local uint64_t t_allocs = 0;
void* operator new(size_t n) {
    ++t_allocs;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

static double thread_cpu_us() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1e3;
}

int main(int argc, char** argv) {
    std::string dir = std::filesystem::temp_directory_path().string();
    int calls = 2000;
    bool threaded = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--calls" && i + 1 < argc) calls = std::atoi(argv[++i]);
        else if (a == "--threaded" && i + 1 < argc) threaded = std::atoi(argv[++i]) != 0;
    }
    const uint32_t N = 300000, rb = 90;
    const uint64_t header = 192;
    const std::string path = dir + "/ple_issue_bench.bin";
    {
        std::ofstream f(path, std::ios::binary);
        std::vector<uint8_t> h(header, 0xAB), r(rb);
        f.write((const char*) h.data(), (std::streamsize) h.size());
        for (uint32_t i = 0; i < N; ++i) {
            for (uint32_t b = 0; b < rb; ++b) r[b] = (uint8_t) ((i * 2654435761u + b * 97u) >> 7);
            std::memcpy(r.data(), &i, 4);
            f.write((const char*) r.data(), rb);
        }
    }
    ng::PleReader rd;
    std::string err;
    if (!rd.open(path, header, N, 64, 0, err, threaded, rb)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 2; }
    std::mt19937 rng(4910);
    std::printf("threaded=%d calls=%d\n", threaded, calls);
    // three row patterns: a 4000-row window (many rows share a page: decode neighbours), the whole table (a chunk of
    // distinct n-grams: nearly every row its own page), and a skewed draw (a few hot rows repeat, the rest spread)
    for (const char* pattern : {"window", "wide", "skewed"}) {
        std::printf("pattern %s\n%8s %14s %14s %12s\n", pattern, "rows", "allocs/issue", "cpu us/issue", "dedup/issue");
        for (size_t n : {(size_t) 1, (size_t) 4, (size_t) 16, (size_t) 64, (size_t) 65, (size_t) 256, (size_t) 3072, (size_t) 49152}) {
            std::vector<uint32_t> rows(n);
            std::vector<uint8_t> out(n * rb);
            double cpu = 0;
            uint64_t allocs = 0;
            const uint64_t dedup0 = rd.snapshot().dedup_rows;
            const int reps = n >= 3072 ? std::max(20, calls / 30) : calls;
            for (int c = 0; c < reps; ++c) {
                const uint32_t base = (uint32_t) (rng() % (N - 4000));
                for (auto& r : rows) {
                    if (pattern[0] == 'w' && pattern[1] == 'i') r = base + (uint32_t) (rng() % 4000);
                    else if (pattern[0] == 'w') r = (uint32_t) (rng() % N);
                    else { const uint32_t u = (uint32_t) (rng() % 1000); r = u < 400 ? (uint32_t) (rng() % 64) * 4099u % N : (uint32_t) (rng() % N); }
                }
                const uint64_t a0 = t_allocs;
                const double t0 = thread_cpu_us();
                const auto t = rd.issue(rows.data(), n, out.data());
                cpu += thread_cpu_us() - t0;
                allocs += t_allocs - a0;
                if (!rd.collect(t, err)) { std::fprintf(stderr, "collect: %s\n", err.c_str()); return 1; }
            }
            std::printf("%8zu %14.2f %14.3f %12.2f\n", n, (double) allocs / reps, cpu / reps,
                        (double) (rd.snapshot().dedup_rows - dedup0) / reps);
        }
    }
    rd.close();
    std::filesystem::remove(path);
    return 0;
}
