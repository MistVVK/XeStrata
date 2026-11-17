// Standalone run of the engine's kernel self-test (src/selftest.cu) on one card: no model, no engine. Built by the optional CMake target strata-split-selftest
// (src/split/CMakeLists.txt; HIP only, off by default). Prints the card and the result; exit 0 when every test passed.
//   selftest_main [device]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/selftest.hpp"

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) {
        std::printf("no device %d\n", dev);
        return 2;
    }
    std::printf("device %d: %s (%s), %d CUs, %.1f GiB\n", dev, p.name, p.gcnArchName, p.multiProcessorCount, (double) p.totalGlobalMem / (1 << 30));
    std::string covered;
    const auto t0 = std::chrono::steady_clock::now();
    const std::string bad = fnx::selftest_card(dev, covered);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (bad.empty()) std::printf("self-test PASSED in %.0f ms: %s\n", ms, covered.c_str());
    else std::printf("self-test FAILED in %.0f ms: %s\n", ms, bad.c_str());
    return bad.empty() ? 0 : 1;
}
