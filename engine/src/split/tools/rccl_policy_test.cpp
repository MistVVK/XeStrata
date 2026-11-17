// The RCCL decision and loader (src/hip/rccl_dl.hpp): a missing librccl leaves the host exchange for the prompt sums with one plain line (forced here with
// FNX_RCCL_LIB pointing nowhere, and at a library that exists but is not RCCL), FNX_PF_RCCL=0 or unset is silent, RCCL is not tried without the peer push, and a
// library that opens is used with a line saying which. CPU only (the loader holds no HIP header).
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/hip/rccl_dl.hpp"

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main() {
    using namespace fnx;
    int calls = 0;
    auto pass = [&](std::string& why) { ++calls; why = "librccl.so.1"; return true; };
    // the real loader, pointed at a library that is not there: the engine's open() would return exactly this
    auto missing = [&](std::string& why) { ++calls; RcclApi api; return rccl_load(api, "", why); };

    // FNX_PF_RCCL=0 or unset with the default off: nothing, no open, no line
    unsetenv("FNX_RCCL_LIB");
    RcclDecision d = rccl_decide("0", true, true, pass);
    CHECK(!d.use && d.line.empty() && calls == 0);
    d = rccl_decide(nullptr, true, false, pass);
    CHECK(!d.use && d.line.empty() && calls == 0);
    // =1 without the peer push: the host exchange stays, one line, no open
    d = rccl_decide("1", false, false, pass);
    CHECK(!d.use && calls == 0 && d.line == "RCCL is not used for the prompt sums: the peer push between the cards is off, so the host exchange stays.");
    // the default alone (unset) without the peer push: the host exchange, quiet
    calls = 0;
    d = rccl_decide(nullptr, false, true, pass);
    CHECK(!d.use && calls == 0 && d.line.empty());
    calls = 0;
    // =1 with the peer push and a library that opens: used, the line says which
    d = rccl_decide("1", true, false, pass);
    CHECK(d.use && calls == 1 && d.line == "The prompt sums use RCCL (librccl.so.1).");
    // unset and the default on behaves like =1
    calls = 0;
    d = rccl_decide(nullptr, true, true, pass);
    CHECK(d.use && calls == 1);

    // the library is missing (FNX_RCCL_LIB points nowhere): the host exchange, one plain line naming the library and the loader's reason
    setenv("FNX_RCCL_LIB", "/nonexistent/librccl.so.1", 1);
    calls = 0;
    d = rccl_decide("1", true, false, missing);
    CHECK(!d.use && calls == 1);
    CHECK(d.line.rfind("RCCL is not available here (cannot load /nonexistent/librccl.so.1", 0) == 0);
    CHECK(d.line.find("): the prompt sums use the host exchange.") != std::string::npos);
    CHECK(d.line.find('\n') == std::string::npos);
    // the same through the default
    d = rccl_decide(nullptr, true, true, missing);
    CHECK(!d.use);
    // FNX_PF_RCCL=0 stays silent whatever FNX_RCCL_LIB says
    d = rccl_decide("0", true, true, missing);
    CHECK(!d.use && d.line.empty());

    // a library that opens but is not RCCL (the C library): no ncclCommInitAll, the same fallback line
    setenv("FNX_RCCL_LIB", "libm.so.6", 1);
    d = rccl_decide("1", true, false, missing);
    CHECK(!d.use && d.line == "RCCL is not available here (libm.so.6 has no ncclCommInitAll): the prompt sums use the host exchange.");
    unsetenv("FNX_RCCL_LIB");
    std::printf(bad ? "rccl_policy_test: %d FAILED\n" : "rccl_policy_test: ok\n", bad);
    return bad ? 1 : 0;
}
