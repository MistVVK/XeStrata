// SPDX-License-Identifier: MIT
// tests/core/volta_forever_test.cpp - unit tests for volta-forever shim detection, LAZY vs EAGER module loading,
// and sm_70 512-thread CUDA Graph block bounds.

#include "strata/core/volta_forever.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using strata::core::allow_1024_thread_cuda_graphs;
using strata::core::apply_default_cuda_module_loading;
using strata::core::default_cuda_module_loading;
using strata::core::safe_cuda_graph_max_threads;
using strata::core::volta_forever_active_from;

void unset_var(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}

void set_var(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

void test_detection_precedence() {
    assert(!volta_forever_active_from(nullptr, nullptr, nullptr, nullptr));
    assert(volta_forever_active_from(nullptr, "/opt/vf/libvolta_forever.so:/usr/lib/foo.so", nullptr, nullptr));
    assert(volta_forever_active_from(nullptr, nullptr, "0.4.2", nullptr));
    assert(volta_forever_active_from(nullptr, nullptr, nullptr, "0.1.0"));
    // Explicit override wins in both directions
    assert(!volta_forever_active_from("0", "/opt/vf/libvolta_forever.so", "0.4.2", "0.1.0"));
    assert(volta_forever_active_from("1", nullptr, nullptr, nullptr));
    assert(strata::core::prefer_dp4a_mmq(true));
    assert(!strata::core::prefer_dp4a_mmq(false));
}

void test_module_loading_defaults() {
    assert(std::strcmp(default_cuda_module_loading(false), "EAGER") == 0);
    assert(std::strcmp(default_cuda_module_loading(true), "LAZY") == 0);

    unset_var("CUDA_MODULE_LOADING");
    unset_var("STRATA_VOLTA_FOREVER");
    unset_var("LD_PRELOAD");
    assert(!apply_default_cuda_module_loading());
    assert(std::strcmp(std::getenv("CUDA_MODULE_LOADING"), "EAGER") == 0);

    unset_var("CUDA_MODULE_LOADING");
    set_var("STRATA_VOLTA_FOREVER", "1");
    assert(apply_default_cuda_module_loading());
    assert(std::strcmp(std::getenv("CUDA_MODULE_LOADING"), "LAZY") == 0);

    // Explicit user CUDA_MODULE_LOADING is never overwritten
    set_var("CUDA_MODULE_LOADING", "EAGER");
    assert(!apply_default_cuda_module_loading());
    assert(std::strcmp(std::getenv("CUDA_MODULE_LOADING"), "EAGER") == 0);

    unset_var("STRATA_VOLTA_FOREVER");
    unset_var("CUDA_MODULE_LOADING");
}

void test_graph_thread_cap() {
    assert(safe_cuda_graph_max_threads(7, 0, true) == 512);
    assert(!allow_1024_thread_cuda_graphs(7, 0, true));
    // Native sm_70 (CUDA 12.x without volta-forever) and sm_75+/sm_80+/sm_120 allow 1024 threads in graphs
    assert(safe_cuda_graph_max_threads(7, 0, false) == 1024);
    assert(allow_1024_thread_cuda_graphs(7, 0, false));
    assert(safe_cuda_graph_max_threads(7, 5, true) == 1024);
    assert(safe_cuda_graph_max_threads(8, 0, true) == 1024);
    assert(safe_cuda_graph_max_threads(12, 0, false) == 1024);
}

}  // namespace

int main() {
    test_detection_precedence();
    test_module_loading_defaults();
    test_graph_thread_cap();
    std::printf("volta_forever_test: ALL 3 SUITES PASSED\n");
    return 0;
}
