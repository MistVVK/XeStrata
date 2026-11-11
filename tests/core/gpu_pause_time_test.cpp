// The GPU pause counter (strata/core/gpu_pause_time.hpp): the reader against a fake KFD tree (present, missing,
// unreadable, garbage, several GPUs) and the two lines.  No GPU, no driver.  Off Linux the reader is always "not
// available", which is checked instead.
#include "strata/core/gpu_pause_time.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

#if defined(__linux__)
void put(const std::string& path, const char* text) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); ++failed; return; }
    std::fputs(text, f);
    std::fclose(f);
}
void gpu(const std::string& proc, const char* id, const char* text) {
    const std::string d = proc + "/stats_" + id;
    mkdir(d.c_str(), 0755);
    if (text != nullptr) put(d + "/evicted_ms", text);
}
#endif
}  // namespace

int main() {
#if defined(__linux__)
    char tmpl[] = "/tmp/gpu_pause_test_XXXXXX";
    const char* root = mkdtemp(tmpl);
    CHECK(root != nullptr);
    if (root == nullptr) return 1;
    const std::string r = root;
    auto proc = [&](long pid) {
        const std::string p = r + "/" + std::to_string(pid);
        mkdir(p.c_str(), 0755);
        return p;
    };

    CHECK(gpu_pause_ms(r.c_str(), 1) == -1);                        // no folder for the process: not available
    proc(2);
    CHECK(gpu_pause_ms(r.c_str(), 2) == -1);                        // a folder without GPUs
    put(proc(2) + "/pasid", "32770\n");                              // other entries are ignored
    CHECK(gpu_pause_ms(r.c_str(), 2) == -1);

    gpu(proc(3), "41945", "954\n");                                  // one GPU, as the driver writes it
    CHECK(gpu_pause_ms(r.c_str(), 3) == 954);
    gpu(proc(3), "17288", "0\n");                                    // a second GPU: summed
    CHECK(gpu_pause_ms(r.c_str(), 3) == 954);
    put(proc(3) + "/stats_17288/evicted_ms", "15\n");
    CHECK(gpu_pause_ms(r.c_str(), 3) == 969);
    put(proc(3) + "/stats_17288/cu_occupancy", "12\n");              // other counters beside it are not read
    CHECK(gpu_pause_ms(r.c_str(), 3) == 969);

    gpu(proc(4), "1", nullptr);                                      // a GPU folder without the counter: not available
    CHECK(gpu_pause_ms(r.c_str(), 4) == -1);
    gpu(proc(5), "1", "12");                                         // no newline is fine
    CHECK(gpu_pause_ms(r.c_str(), 5) == 12);
    const char* garbage[] = {"", "\n", "abc\n", "-5\n", "12 ms\n", "1.5\n", "99999999999999999999999\n"};
    long pid = 10;
    for (const char* g : garbage) {
        gpu(proc(pid), "1", g);
        CHECK(gpu_pause_ms(r.c_str(), pid) == -1);
        ++pid;
    }
    gpu(proc(30), "1", "100\n");                                     // one good GPU and one bad: not available
    gpu(proc(30), "2", "x\n");
    CHECK(gpu_pause_ms(r.c_str(), 30) == -1);

    gpu(proc(40), "1", "77\n");                                      // unreadable (skipped when run as root)
    const std::string locked = r + "/40/stats_1/evicted_ms";
    chmod(locked.c_str(), 0);
    if (geteuid() != 0) CHECK(gpu_pause_ms(r.c_str(), 40) == -1);
    chmod(locked.c_str(), 0644);
    CHECK(gpu_pause_ms(r.c_str(), 40) == 77);

    // the real tree for this process: whatever it is, it is a number or "not available", and it does not throw
    const int64_t self = gpu_pause_ms_self();
    CHECK(self >= -1);
    std::printf("this process on this machine: %lld\n", (long long) self);

    // no KFD tree (a CUDA box, a container without sysfs): not available, no line, and it costs one failed opendir.
    // 10,000 reads: the mean is printed; the bound (50 us each) only catches something gone badly wrong
    {
        const std::string none = r + "/no-kfd-here";
        const auto t0 = std::chrono::steady_clock::now();
        int64_t v = 0;
        for (int i = 0; i < 10000; ++i) v += gpu_pause_ms(none.c_str(), 1234);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 10000;
        CHECK(v == -10000);
        CHECK(gpu_pause_request_line(gpu_pause_ms(none.c_str(), 1234), gpu_pause_ms(none.c_str(), 1234)).empty());
        CHECK(gpu_pause_stall_line(gpu_pause_ms(none.c_str(), 1234), -1, -1).empty());
        CHECK(us < 50.0);
        std::printf("no KFD tree: %.2f us per read\n", us);
        // the present tree with two GPUs, for comparison (the real sysfs read costs more: measured on p620)
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i) v += gpu_pause_ms(r.c_str(), 3);
        std::printf("fake tree, 2 GPUs: %.2f us per read\n",
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / 10000);
    }

    const std::string rm = "rm -rf '" + r + "'";
    if (std::system(rm.c_str()) != 0) std::fprintf(stderr, "could not remove %s\n", r.c_str());
#else
    CHECK(gpu_pause_ms("C:/nowhere", 1) == -1);
    CHECK(gpu_pause_ms_self() == -1);
#endif

    // the stall report's line
    CHECK(gpu_pause_stall_line(-1, 5, -1).empty());                  // not available: no line
    CHECK(gpu_pause_stall_line(-1, -1, 7).empty());
    CHECK(gpu_pause_stall_line(6123, 5169, -1) ==
          "  GPU work paused by the graphics driver: 6123 ms since the engine started, 954 ms in this request (the AMD "
          "driver pauses all of the engine's GPU work while Linux moves memory the GPU uses)");
    CHECK(gpu_pause_stall_line(8073, 5169, 6123) ==
          "  GPU work paused by the graphics driver: 8073 ms since the engine started, 2904 ms in this request, 1950 ms "
          "in the last 2 s");
    CHECK(gpu_pause_stall_line(40, -1, -1) ==                        // between requests: no request part
          "  GPU work paused by the graphics driver: 40 ms since the engine started (the AMD driver pauses all of the "
          "engine's GPU work while Linux moves memory the GPU uses)");
    CHECK(gpu_pause_stall_line(40, -1, 40) ==
          "  GPU work paused by the graphics driver: 40 ms since the engine started, 0 ms in the last 2 s");

    // the request line: only at or above the threshold, and never from an unread or a reset counter
    CHECK(kGpuPauseReportMs == 100);
    CHECK(gpu_pause_request_line(-1, 900).empty());
    CHECK(gpu_pause_request_line(100, -1).empty());
    CHECK(gpu_pause_request_line(100, 50).empty());
    CHECK(gpu_pause_request_line(1000, 1015).empty());               // the usual 15 ms: no line
    CHECK(gpu_pause_request_line(1000, 1099).empty());
    CHECK(gpu_pause_request_line(1000, 1100) ==
          "strata serve: the graphics driver paused the GPU for 100 ms during this request while Linux moved memory "
          "the GPU uses; the request was slower by up to that much");
    CHECK(gpu_pause_request_line(15, 969) ==
          "strata serve: the graphics driver paused the GPU for 954 ms during this request while Linux moved memory "
          "the GPU uses; the request was slower by up to that much");

    if (failed) { std::fprintf(stderr, "gpu_pause_time_test: %d failed\n", failed); return 1; }
    std::printf("gpu_pause_time_test: ok\n");
    return 0;
}
