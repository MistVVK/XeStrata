// sycl/src/core/runtime_files_test.cpp - strata::loaded_runtime_files (strata/runtime_files.hpp, #5914), host only, no GPU.
// The C++ runtime this test links is a loaded module like the Level Zero loader is in the engine: it must be found by its
// file-name prefix, with a full path, once; an unknown prefix finds nothing.
#include "strata/runtime_files.hpp"

#include <cstdio>
#include <string>

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main() {
#if defined(_WIN32)
    const std::vector<std::string> known = {"kernel32", "kernel32"};
#else
    const std::vector<std::string> known = {"libc.so", "libc.so"};
#endif
    const std::vector<std::string> f = strata::loaded_runtime_files(known);
    check(f.size() == 1, "a loaded C runtime is found once (a prefix given twice does not list it twice)");
    check(!f.empty() && f[0].find_first_of("/\\") != std::string::npos, "it is named by a full path");
    for (const std::string& s : f) std::printf("    %s\n", s.c_str());
    check(strata::loaded_runtime_files({"libno_such_runtime_strata"}).empty(), "an unknown prefix finds nothing");
    const std::string line = strata::intel_runtime_files_line();
    check(line.find("libc.so") == std::string::npos, "the Intel line lists only Intel runtime files (none in this test)");
    std::printf(g_fail ? "runtime_files_test: %d FAILED\n" : "runtime_files_test: all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
