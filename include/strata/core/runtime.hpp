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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <typeindex>
#include <unordered_map>

namespace strata::core {

/// Whether a GPU is the processor's own graphics (src/core/device.cpp).
bool integrated_gpu(const sycl::device& d);
/// The device's free memory as its runtime reports it, or for a GPU of the xe driver whose runtime cannot say, as the
/// driver does.  False when neither knows.
bool device_free_memory(const sycl::device& d, uint64_t& free);
/// The version of the Intel Graphics Compiler whose GEP simplification the runtime turned off for the engine's
/// Level Zero GPU ("2.28.4"; "unknown" when the library carries none), empty when it did not (src/core/device.cpp).
std::string compiler_workaround();
/// Whether a Linux kernel's release (uname's) is one whose drm/pagemap frees a page the xe driver's SVM moved back
/// to RAM with the moved flag still set (Ubuntu's 7.0.0 from ABI 40 on until the build with the fix; src/core/device.cpp).
/// The engine works around nothing: it tells the user to leave the kernel (svm_workaround's notice).
bool svm_broken_kernel(const char* release);
/// The kernel's release the engine tells the user to leave ("7.0.0-43-generic"), empty when there is nothing to
/// tell: Linux, svm_broken_kernel's release, and an xe GPU (src/core/device.cpp).  The start log, the web app's
/// notice and About, and xestrata --setup say it.
std::string svm_workaround();
/// Ends the process on SIGTERM and SIGINT after the GPUs' queues are drained (at most 5 s), on a thread of its own; called
/// first in main, before any thread exists (src/core/device.cpp).
void route_stop_signals();

// One runtime per engine device: its context owns its queues and every USM allocation made on it. Cross-queue
// consumers must depend on the producer event; in-order applies within a queue.  Device 0 is the GPU find_devices
// puts first; 1.. the other usable GPUs (device.cpp).  A runtime is created when its device is first used, so a run on
// one GPU never opens the others.
class Runtime {
public:
    /// The runtime of this thread's current device (on_device.hpp).
    static Runtime& get();
    static Runtime& at(int ordinal);
    /// The number of GPUs the engine can drive.
    static int count();
    /// The runtime whose context holds `pointer` (device or host USM), or null.
    static Runtime* owner(const void* pointer);
    /// The runtime one of whose queues `stream` is, or null.
    static Runtime* owner_of_stream(const void* stream);
    int ordinal() const { return ordinal_; }
    ~Runtime();
    sycl::queue& compute() { return compute_; }
    sycl::queue& transfer() { return transfer_; }
    const sycl::device& device() const { return device_; }
    const sycl::context& context() const { return context_; }
    /// Free device memory read immediately before this runtime created its context, where the device reports it.
    std::optional<uint64_t> free_before_context() const { return free_before_context_; }

    // Engine streams: further in-order queues on this context, for the work CUDA put on its own streams.  A void*
    // stream handed to a kernel is one of these, compute() or null (compute()), or a queue of another device's
    // runtime (the kernel then runs there); anything else is refused.  `high_priority`: sycl_ext_oneapi_queue_priority's
    // priority_high, a hint the backend maps to its own (CUDA and HIP stream priorities, Level Zero's queue priority);
    // a runtime without the extension gives an ordinary stream.
    sycl::queue* create_stream(bool high_priority = false);
    // An engine stream for copies and events alone, on the GPU's copy engine where the device reports a copy-only
    // queue group (Level Zero); elsewhere an ordinary engine stream.  No kernel may go on it.
    sycl::queue* create_copy_stream();
    // Whether the GPU still answers (issue #10): Level Zero's zeDeviceGetStatus, "answers" or "lost" (or its result
    // code); "unknown" for a backend or a build without that call.  Safe after a failure: it submits no work.
    std::string device_status();
    void destroy_stream(sycl::queue* stream);
    sycl::queue& stream(void* stream);
    /// The runtime that owns `q` (its compute, transfer or an engine stream); throws where none does.
    static Runtime& of_queue(sycl::queue& q);
    // A belonging tied to a queue's life (a kernel's scratch, a library's stream on it): made on the queue's first
    // use, held by this runtime, and destroyed once the queue is drained when its stream is destroyed
    // (destroy_stream) or the runtime ends.  A queue's address is reused after its destruction; an item keyed by the
    // bare address would outlive it and meet the next queue there, so the items go with the queue itself.  T is built
    // with the queue (T(sycl::queue&), keeping its own copy where its destructor frees GPU memory) and the queue is
    // idle when T's destructor runs.  The lookup is locked; two threads using the T itself synchronize as they would
    // any shared state (a mutex in T).  The queue must be this runtime's (of_queue).
    template <class T> T& queue_item(sycl::queue& q) {
        std::lock_guard<std::mutex> lock(items_mutex_);
        auto& slot = items_[&q][std::type_index(typeid(T))];
        if (!slot) slot.reset(new Item<T>(q));
        return static_cast<Item<T>*>(slot.get())->value;
    }

    sycl::event copy_async(void* dst, const void* src, uint64_t bytes,
                           const std::vector<sycl::event>& dependencies = {});
    sycl::event compute_after(const std::vector<sycl::event>& dependencies);
    // Watches the host submission itself: some backends can block inside an asynchronous copy call.
    // `what` must outlive the watch.
    class CopyWatch {
    public:
        CopyWatch(Runtime& runtime, const char* what, uint64_t bytes, int layer = -1);
        ~CopyWatch();
        CopyWatch(const CopyWatch&) = delete;
        CopyWatch& operator=(const CopyWatch&) = delete;
    private:
        Runtime& runtime_;
        void* entry_;
    };
    // `what` names the wait in the watchdog's report; it must outlive the call.
    void wait(sycl::event event, const char* what = "event");
    void finish(sycl::queue& queue);
    // Drains compute, transfer and every engine stream.
    void finish();
    // finish() on every runtime made so far (it makes none), errors ignored: before the process ends on a signal.
    static void finish_all() noexcept;
    // Exceptional teardown cannot unwind past memory still in use by the GPU.
    void free(void* pointer) noexcept;
    // Builds every kernel the binary carries for the device, on a thread of its own (the program: strata generate),
    // so that their modules are loaded while the model is: SYCL builds a kernel's on its first launch, and the first
    // verify window took 22 module loads, 30 ms of its first token, on the RTX 4070.  A kernel the device cannot run
    // (another GPU's matrix shape) is skipped.  Once.
    void preload_kernels();
    // Waits for preload_kernels' thread to end (timings taken beside it read its CPU work as theirs).
    void wait_preload() { if (preload_.joinable()) preload_.join(); }
    // Stops it after the kernel it is building, and waits.  Before main returns: exit() tears the GPU compiler (IGC)
    // down before the runtimes, and a build still running in it then hung or crashed the engine at QUIT.
    void stop_preload() {
        preload_stop_.store(true, std::memory_order_relaxed);
        wait_preload();
    }
    // Whether an event only ordered or waited on (gpu::event_record without timing) is recorded as an in-order
    // barrier rather than a timestamp (submit_profiling_tag): measured when the runtime is made (measure_marks).
    bool marks_by_barrier() const { return marks_by_barrier_; }

private:
    struct Waiter {
        std::chrono::steady_clock::time_point since;
        const char* what;
        uint64_t bytes = 0;
        std::chrono::steady_clock::time_point reported{};
        bool copy = false;
        int layer = -1;
    };
    Runtime(int ordinal, const sycl::device& device);
    void check_async();
    sycl::async_handler handler();
    bool owns(const sycl::queue* queue);
    void watch();
    bool measure_marks();
    int ordinal_;
    std::string pci_;
    std::mutex error_mutex_;
    std::exception_ptr error_;
    sycl::device device_;
    std::optional<uint64_t> free_before_context_;
    sycl::context context_;
    sycl::queue compute_, transfer_;
    std::mutex streams_mutex_;
    std::list<sycl::queue> streams_;
    // the queues' belongings (queue_item), destroyed before the queues and the context they free memory on
    struct ItemBase { virtual ~ItemBase() = default; };
    template <class T> struct Item : ItemBase {
        explicit Item(sycl::queue& q) : value(q) {}
        T value;
    };
    std::mutex items_mutex_;
    std::unordered_map<const sycl::queue*, std::map<std::type_index, std::unique_ptr<ItemBase>>> items_;
    std::mutex watch_mutex_;
    std::condition_variable watch_cv_;
    std::list<Waiter> waiters_;
    bool stop_ = false;
    std::thread watchdog_;
    std::atomic<bool> preload_stop_{false};
    std::thread preload_;
    bool marks_by_barrier_ = false;
};

}  // namespace strata::core
