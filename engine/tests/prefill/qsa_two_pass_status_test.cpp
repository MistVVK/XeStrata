// STRATA_SELECT_TWO_PASS's request and its line where the two-pass scorer cannot run (qsa_two_pass_status.hpp).
// GPU-free: the parse, quiet when unset or supported, one line per process where unsupported.
#include "strata/kernels/qsa_two_pass_status.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace tp = strata::kernels::two_pass;

static void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

static std::string output(FILE* log) {
    check(std::fflush(log) == 0 && std::fseek(log, 0, SEEK_SET) == 0, "rewind the log");
    std::string text;
    char buf[512];
    while (const size_t n = std::fread(buf, 1, sizeof(buf), log)) text.append(buf, n);
    check(!std::ferror(log), "read the log");
    return text;
}

static std::string line(const char* where) {
    return std::string("strata: STRATA_SELECT_TWO_PASS=1 is not used on ") + where +
           ": the two-pass scorer runs only in the CUDA prompt path's tensor-core selection (sm_80+); the selection is "
           "unchanged\n";
}

int main(int argc, char** argv) {
    check(argc == 2, "pass a scratch log path (no GPU or model needed)");
    FILE* log = std::fopen(argv[1], "w+");
    check(log != nullptr, "open the log");

    // exactly "1" is on, as the scorer parsed it before; unset or anything else is off and prints nothing anywhere
    check(tp::requested_from("1"), "1: on");
    const char* off[] = {nullptr, "", "0", "01", "1x", "true", "1 ", "2"};
    for (const char* flag : off) {
        check(!tp::requested_from(flag), "not exactly 1: off");
        tp::RequestStatus status(flag);
        check(!status.requested(), "not requested");
        check(!status.on("HIP", log), "off stays off where unsupported");
        check(!status.on(nullptr, log), "off stays off where supported");
    }
    check(output(log).empty(), "no line without the switch");

    // a supported path returns the request and prints nothing (the engine's per-segment line reports what ran)
    tp::RequestStatus cuda("1");
    check(cuda.on(nullptr, log) && cuda.on(nullptr, log), "supported: on");
    check(output(log).empty(), "supported: no line");

    // each unsupported path the engine names: off, one line, never repeated
    const char* where[] = {"HIP", "Metal", "SYCL", "this CUDA card or with STRATA_QSA_WARP",
                           "the layer-major prompt path"};
    std::string expected;
    for (const char* w : where) {
        tp::RequestStatus status("1");
        check(!status.on(w, log), "unsupported: off");
        expected += line(w);
        check(output(log) == expected, "unsupported: one line naming the path");
        check(!status.on(w, log) && !status.on(w, log), "unsupported again: off");
        check(output(log) == expected, "unsupported: the line is printed once");
    }

    // a layer split can mix cards: a supported query neither uses up the line nor is turned off by an unsupported one
    tp::RequestStatus mixed("1");
    check(mixed.on(nullptr, log), "supported card first");
    check(!mixed.on("this CUDA card or with STRATA_QSA_WARP", log), "then an unsupported card");
    check(mixed.on(nullptr, log), "the supported card stays on");
    expected += line("this CUDA card or with STRATA_QSA_WARP");
    check(output(log) == expected, "the unsupported card's line once");

    check(std::fclose(log) == 0, "close the log");
    check(std::remove(argv[1]) == 0, "remove the log");
    std::puts("qsa_two_pass_status_test: OK");
    return 0;
}
