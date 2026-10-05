// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/device.hpp"
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <list>
#include <mutex>
#include <thread>

namespace strata::core {

// A single context owns both queues and every USM allocation. Cross-queue
// consumers must depend on the producer event; in-order applies within a queue.
class Runtime {
public:
    static Runtime& get();
    sycl::queue& compute() { return compute_; }
    sycl::queue& transfer() { return transfer_; }
    const sycl::device& device() const { return device_; }
    const sycl::context& context() const { return context_; }

    // Engine streams: further in-order queues on this context, for the work CUDA put on its own streams.  A void*
    // stream handed to a kernel is one of these, compute() or null (compute()); anything else is refused.
    sycl::queue* create_stream();
    void destroy_stream(sycl::queue* stream);
    sycl::queue& stream(void* stream);

    sycl::event copy_async(void* dst, const void* src, uint64_t bytes,
                           const std::vector<sycl::event>& dependencies = {});
    sycl::event compute_after(const std::vector<sycl::event>& dependencies);
    // `what` names the wait in the watchdog's report; it must outlive the call.
    void wait(sycl::event event, const char* what = "event");
    void finish(sycl::queue& queue);
    // Drains compute, transfer and every engine stream.
    void finish();
    // Exceptional teardown cannot unwind past memory still in use by the GPU.
    void free(void* pointer) noexcept;
    // Builds every kernel the binary carries for the device, on a thread of its own (the program: strata generate),
    // so that their modules are loaded while the model is: SYCL builds a kernel's on its first launch, and the first
    // verify window took 22 module loads, 30 ms of its first token, on the RTX 4070.  A kernel the device cannot run
    // (another GPU's matrix shape) is skipped.  Once.
    void preload_kernels();

private:
    struct Waiter {
        std::chrono::steady_clock::time_point since;
        const char* what;
    };
    Runtime();
    ~Runtime();
    void check_async();
    sycl::async_handler handler();
    bool owns(const sycl::queue* queue);
    void watch();
    std::mutex error_mutex_;
    std::exception_ptr error_;
    sycl::device device_;
    sycl::context context_;
    sycl::queue compute_, transfer_;
    std::mutex streams_mutex_;
    std::list<sycl::queue> streams_;
    std::mutex watch_mutex_;
    std::condition_variable watch_cv_;
    std::list<Waiter> waiters_;
    bool stop_ = false;
    std::thread watchdog_;
    std::atomic<bool> preload_stop_{false};
    std::thread preload_;
};

}  // namespace strata::core
