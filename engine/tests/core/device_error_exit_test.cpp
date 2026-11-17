// #6054 / #6073: how a kernel library's error check ends the engine (strata/core/device_error_exit.hpp).  The test runs
// itself as a child: under --serve (set_serve_error_line(true)) the child prints the protocol line the server reads as an
// engine death ("ERR fatal GPU error: <what>: <error>") on stdout and ends at once with exit code 3; without it, the old
// stderr line and exit code 1, and nothing on stdout.  No GPU, no model.
#include "strata/core/device_error_exit.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
std::string slurp(const std::string& p) {
    std::ifstream f(p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
struct Destructor {   // a static whose destructor must NOT run under --serve (std::_Exit), and runs under std::exit
    ~Destructor() { std::fputs("static destructor ran\n", stderr); }
} g_destructor;
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "child-serve") == 0 || std::strcmp(argv[1], "child-cli") == 0)) {
        if (std::strcmp(argv[1], "child-serve") == 0) strata::core::set_serve_error_line(true);
        std::printf("before\n");
        strata::core::exit_device_error("prefill copy_i32", "out of memory");
    }
#if defined(_WIN32)
    std::puts("device_error_exit_test: the child runs are POSIX-only here; skipped");
    return 77;
#else
    const std::string me = argv[0];
    for (const char* mode : {"child-serve", "child-cli"}) {
        const std::string out = std::string("device_error_exit_test.") + mode + ".out", err = std::string("device_error_exit_test.") + mode + ".err";
        const int st = std::system(("'" + me + "' " + mode + " > " + out + " 2> " + err).c_str());
        const int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
        const std::string o = slurp(out), e = slurp(err);
        const bool serve = std::strcmp(mode, "child-serve") == 0;
        if (serve) {
            check(code == 3, "--serve: exit code 3");
            check(o == "before\nERR fatal GPU error: prefill copy_i32: out of memory\n", "--serve: stdout carries the line the server reads");
            check(e.find("prefill copy_i32: out of memory\n") == 0, "--serve: stderr keeps the check's own line first");
            check(e.find("static destructor ran") == std::string::npos, "--serve: no teardown (std::_Exit)");
        } else {
            check(code == 1, "command line: exit code 1, as before");
            check(o == "before\n", "command line: nothing more on stdout");
            check(e.find("prefill copy_i32: out of memory\n") == 0 && e.find("ERR") == std::string::npos, "command line: the old stderr line only");
        }
        std::remove(out.c_str());
        std::remove(err.c_str());
    }
    std::printf("device_error_exit_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
#endif
}
