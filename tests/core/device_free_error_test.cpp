// #5708: device_free_bytes() ignored cudaMemGetInfo's result, so a failed query read as "0.00 GiB free" with nothing
// said.  Here the query fails for a known reason - no device is visible (CUDA_VISIBLE_DEVICES=-1, set before the first
// CUDA call) - and the engine must say that error (its text) on stderr, still answering 0 free.  No GPU is needed; it
// exits 77 (skipped) when a device is visible after all.
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
#else
#include <unistd.h>
#endif

int main() {
#if defined(_WIN32)
    _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
    _putenv_s("HIP_VISIBLE_DEVICES", "-1");
#else
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
    setenv("HIP_VISIBLE_DEVICES", "-1", 1);
#endif
    const char* path = "device_free_error_test.stderr.txt";
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    if (saved < 0 || !std::freopen(path, "w", stderr)) { std::printf("FAIL: could not redirect stderr\n"); return 1; }
    const size_t free_b = strata::core::device_free_bytes();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    std::string said;
    if (std::FILE* f = std::fopen(path, "rb")) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) said.append(buf, n);
        std::fclose(f);
    }
    std::remove(path);
    size_t f2 = 0, t2 = 0;
    const cudaError_t e = cudaMemGetInfo(&f2, &t2);
    (void) cudaGetLastError();
    if (e == cudaSuccess) { std::printf("SKIPPED: a device is visible, so the query does not fail here\n"); return 77; }
    const std::string name = cudaGetErrorString(e);
    std::printf("device_free_error_test: device_free_bytes() = %zu, the query's error %s, stderr: %s\n",
                free_b, name.c_str(), said.empty() ? "(nothing)" : said.c_str());
    if (free_b != 0) { std::printf("FAIL: a failed query must still count as nothing free\n"); return 1; }
    if (said.find(name) == std::string::npos) {
        std::printf("FAIL: the failed query (%s) was not said; a \"0.00 GiB free\" would read as a measurement\n", name.c_str());
        return 1;
    }
    std::printf("device_free_error_test: ok\n");
    return 0;
}
