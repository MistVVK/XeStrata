// RCCL for the prompt path's chunk sums (AMD, FNX_PF_RCCL=1 behind a passed peer probe): loaded at run time with dlopen, never linked, so a missing library can only mean
// "the host exchange for the prompt sums, one plain line" and the engine starts exactly as before. CPU only (no HIP header: a stream is an opaque pointer), so
// tools/rccl_policy_test.cpp holds the loader and the decision to their rules without a card. The library comes with the standard AMD Linux install (setup.py installs the
// rocm[libraries] wheel: librccl.so.1, BSD-3-clause text from NVIDIA with AMD modifications, in share/doc/rccl/LICENSE.txt of the SDK); we do not copy or ship it.
// The six functions and the two enum values are RCCL's (rccl.h: ncclSum = 0, ncclFloat16 = 6, ncclFloat32 = 7, ncclSuccess = 0).
#pragma once

#include <dlfcn.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>

namespace fnx {

constexpr int RCCL_SUCCESS = 0, RCCL_SUM = 0, RCCL_FLOAT16 = 6, RCCL_FLOAT32 = 7;
constexpr bool RCCL_DEFAULT_ON = true;   // the prompt sums through RCCL when FNX_PF_RCCL is unset: 103K prompt 33.3 -> 28.8 s, 61-task suite 56/61, first-token KL vs the layer split 0.0309; any failure is one line and the host sums (tests/split pins it)

struct RcclApi {
    void* handle = nullptr;
    int (*CommInitAll)(void** comms, int ndev, const int* devlist) = nullptr;
    int (*AllReduce)(const void* send, void* recv, size_t count, int dtype, int op, void* comm, void* stream) = nullptr;
    int (*GroupStart)() = nullptr;
    int (*GroupEnd)() = nullptr;
    int (*CommDestroy)(void* comm) = nullptr;
    const char* (*ErrorString)(int result) = nullptr;
};

// path: the library to load ("" = librccl.so.1 from the library path; FNX_RCCL_LIB overrides it, for tests). false with the reason in why.
inline bool rccl_load(RcclApi& api, const std::string& path, std::string& why) {
    const char* over = std::getenv("FNX_RCCL_LIB");
    const std::string lib = over && *over ? over : (path.empty() ? "librccl.so.1" : path);
    api.handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!api.handle) {
        const char* e = dlerror();
        why = "cannot load " + lib + (e ? std::string(": ") + e : "");
        return false;
    }
    auto sym = [&](const char* name, auto& fn) {
        void* p = dlsym(api.handle, name);
        if (!p) {
            why = lib + " has no " + name;
            return false;
        }
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
        return true;
    };
    if (!sym("ncclCommInitAll", api.CommInitAll) || !sym("ncclAllReduce", api.AllReduce) || !sym("ncclGroupStart", api.GroupStart) ||
        !sym("ncclGroupEnd", api.GroupEnd) || !sym("ncclCommDestroy", api.CommDestroy) || !sym("ncclGetErrorString", api.ErrorString)) {
        dlclose(api.handle);
        api.handle = nullptr;
        return false;
    }
    why = lib;
    return true;
}

// The decision: FNX_PF_RCCL=0 or unset (the default, until the screens say the prompt sums win): the host exchange, no line. =1: only with the peer push on (RCCL moves its
// data over the same peer links); open() loads the library and runs the self-test; any failure is the host exchange for the prompt sums and one plain line.
struct RcclDecision {
    bool use = false;
    std::string line;
};
inline RcclDecision rccl_decide(const char* env, bool peer_on, bool default_on, const std::function<bool(std::string&)>& open) {
    RcclDecision d;
    const bool wanted = env ? std::strcmp(env, "0") != 0 : default_on;
    if (!wanted) return d;
    if (!peer_on) {
        if (env) d.line = "RCCL is not used for the prompt sums: the peer push between the cards is off, so the host exchange stays.";   // (asked for: say so; the default stays quiet)
        return d;
    }
    std::string why;
    if (!open(why)) {
        d.line = "RCCL is not available here (" + why + "): the prompt sums use the host exchange.";
        return d;
    }
    d.use = true;
    d.line = "The prompt sums use RCCL (" + why + ").";
    return d;
}

}  // namespace fnx
