// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/runtime.hpp"


#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {
namespace {
// The backends the engine runs on: Level Zero for Intel GPUs, CUDA and HIP where the build has their targets.  The
// same Intel GPU is listed by OpenCL as well, which the engine does not use.
bool engine_backend(const sycl::device& d) {
    const auto b = d.get_backend();
    return b == sycl::backend::ext_oneapi_level_zero || b == sycl::backend::ext_oneapi_cuda ||
           b == sycl::backend::ext_oneapi_hip;
}

// A kernel with no work, built for the device to learn whether the binary carries code for it: a CUDA or HIP GPU
// is listed whenever the runtime has the backend, also by a build without that target (STRATA_CUDA_ARCHS).
struct ProbeKernel;

// Why a GPU cannot run the engine, from what it reports (empty: it can).  The kernels take 32-wide sub-groups, FP16
// and device and host USM; nothing is chosen by device ID or name (AGENTS.md).
std::string unusable(const sycl::device& d) {
    if (!d.has(sycl::aspect::usm_device_allocations) || !d.has(sycl::aspect::usm_host_allocations))
        return "no device and host USM";
    if (!d.has(sycl::aspect::fp16)) return "no FP16";
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), size_t(32)) == sg.end()) return "no 32-wide sub-groups";
    try {
        const sycl::context ctx(d);
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {d},
                                                                       {sycl::get_kernel_id<ProbeKernel>()});
    } catch (const sycl::exception& e) {
        return std::string("this build has no code for it (") + e.what() + ")";
    }
    return {};
}

// Whether the GPU is the processor's own graphics (its memory is the system RAM), as the SYCL runtime reports it.
// Not asked of Level Zero directly: the engine links no maker's driver library, so a contrib build starts on a PC
// with only Intel's or only NVIDIA's (the runtime opens each backend's library when it is there).
bool integrated(const sycl::device& d) { return d.has(sycl::aspect::ext_oneapi_is_integrated_gpu); }

std::string pci_of(const sycl::device& d) {
    if (!d.has(sycl::aspect::ext_intel_pci_address)) return "?";
    std::string s = d.get_info<sycl::ext::intel::info::device::pci_address>();
    for (auto& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

sycl::device select_device() {
    std::vector<sycl::device> candidates;
    std::string refused;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (!engine_backend(d)) continue;
        if (const std::string why = unusable(d); !why.empty()) {
            refused += "; " + d.get_info<sycl::info::device::name>() + " (" + pci_of(d) + "): " + why;
            continue;
        }
        candidates.push_back(d);
    }
    if (candidates.empty()) throw DeviceError("no usable GPU" + refused);
    // setup names the card by PCI address: Level Zero's own numbering can include GPUs that setup does not list
    if (const char* want = std::getenv("STRATA_GPU_PCI"); want != nullptr && *want != '\0') {
        std::string w = want;
        for (auto& c : w) c = (char) std::tolower((unsigned char) c);
        std::vector<sycl::device> match;
        for (const auto& d : candidates)
            if (pci_of(d) == w) match.push_back(d);
        if (match.size() != 1)
            throw DeviceError(std::string("STRATA_GPU_PCI=") + want + ": no usable GPU at that PCI address" +
                              refused);
        return match.front();
    }
    // Without a choice, a discrete card before the processor's own graphics, then the most memory, then the most
    // compute units (whose size differs between GPU makers, so they only break a tie).  The engine drives one GPU.
    return *std::max_element(candidates.begin(), candidates.end(), [](const sycl::device& x, const sycl::device& y) {
        const bool ix = integrated(x), iy = integrated(y);
        if (ix != iy) return ix;
        const auto mx = x.get_info<sycl::info::device::global_mem_size>();
        const auto my = y.get_info<sycl::info::device::global_mem_size>();
        if (mx != my) return mx < my;
        return x.get_info<sycl::info::device::max_compute_units>() < y.get_info<sycl::info::device::max_compute_units>();
    });
}

// Instantiates ProbeKernel for the device compilers; never run.
[[maybe_unused]] void probe_kernel_instance(sycl::queue& q) {
    q.single_task<ProbeKernel>([] {});
}

[[noreturn]] void fatal(const char* message) {
    std::fprintf(stderr, "Xe runtime: %s; terminating without releasing in-flight allocations\n", message);
    std::fflush(stderr);
    std::_Exit(1);
}

// The xe driver resets a job after its job_timeout_ms (5 s here, at most 10 s), which completes the event with an
// error, so a healthy wait - even a drain of many queued jobs - returns well inside this.  A wait that outlives it
// is stuck in the driver or the host, and only a thread other than the waiter can report that.
constexpr auto kHangLimit = std::chrono::seconds(120);
}  // namespace

Runtime& Runtime::get() {
    static Runtime runtime;
    return runtime;
}

sycl::async_handler Runtime::handler() {
    return [this](sycl::exception_list errors) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!error_ && errors.size()) error_ = *errors.begin();
    };
}

Runtime::Runtime() : device_(select_device()), context_(device_),
    compute_(context_, device_, handler(), sycl::property::queue::in_order{}),
    transfer_(context_, device_, handler(), sycl::property::queue::in_order{}), watchdog_([this] { watch(); }) {}

sycl::queue* Runtime::create_stream() {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    return &streams_.emplace_back(context_, device_, handler(), sycl::property::queue::in_order{});
}

void Runtime::destroy_stream(sycl::queue* stream) {
    if (!stream) return;
    if (!owns(stream) || stream == &compute_ || stream == &transfer_)
        throw DeviceError("destroy_stream: not an engine stream of the Xe runtime");
    finish(*stream);
    std::lock_guard<std::mutex> lock(streams_mutex_);
    streams_.remove_if([stream](const sycl::queue& q) { return &q == stream; });
}

bool Runtime::owns(const sycl::queue* queue) {
    if (queue == &compute_ || queue == &transfer_) return true;
    std::lock_guard<std::mutex> lock(streams_mutex_);
    for (const auto& q : streams_)
        if (&q == queue) return true;
    return false;
}

sycl::queue& Runtime::stream(void* stream) {
    if (!stream) return compute_;
    auto* q = static_cast<sycl::queue*>(stream);
    if (!owns(q)) throw DeviceError("the stream is not a queue of the Xe runtime");
    return *q;
}

void Runtime::preload_kernels() {
    if (preload_.joinable()) return;
    preload_ = std::thread([this] {
        int built = 0, skipped = 0;
        for (const sycl::kernel_id& id : sycl::get_kernel_ids()) {
            if (preload_stop_.load(std::memory_order_relaxed)) return;
            try {
                (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(context_, {device_}, {id});
                ++built;
            } catch (const std::exception&) {   // none for this device, or a shape it does not run
                ++skipped;
            }
        }
        if (std::getenv("STRATA_TRACE"))
            std::fprintf(stderr, "strata trace: %d kernels built ahead, %d not for this GPU\n", built, skipped);
    });
}

Runtime::~Runtime() {
    preload_stop_.store(true, std::memory_order_relaxed);
    if (preload_.joinable()) preload_.join();
    try { finish(); }
    catch (const std::exception& e) { fatal(e.what()); }
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        stop_ = true;
    }
    watch_cv_.notify_all();
    watchdog_.join();
}

void Runtime::watch() {
    std::unique_lock<std::mutex> lock(watch_mutex_);
    while (!watch_cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stop_; })) {
        const auto now = std::chrono::steady_clock::now();
        for (const Waiter& w : waiters_) {
            if (now - w.since < kHangLimit) continue;
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now - w.since).count();
            std::fprintf(stderr, "Xe runtime: waiting for %s for %lld s (%zu waits in progress)\n", w.what,
                         (long long) seconds, waiters_.size());
            fatal("GPU wait exceeded the hang limit");
        }
    }
}

void Runtime::check_async() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_) std::rethrow_exception(error_);
}

sycl::event Runtime::copy_async(void* dst, const void* src, uint64_t bytes,
                               const std::vector<sycl::event>& dependencies) {
    check_async();
    return transfer_.memcpy(dst, src, bytes, dependencies);
}

sycl::event Runtime::compute_after(const std::vector<sycl::event>& dependencies) {
    check_async();
    return compute_.ext_oneapi_submit_barrier(dependencies);
}

void Runtime::wait(sycl::event event, const char* what) {
    // Blocking in the driver rather than polling keeps the wake-up latency off decode's per-token waits.  A timeout
    // cannot safely throw and free USM that an unfinished command owns, so a stuck wait ends the process instead.
    std::list<Waiter>::iterator self;
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        self = waiters_.insert(waiters_.end(), Waiter{std::chrono::steady_clock::now(), what});
    }
    struct Done {
        Runtime& r;
        std::list<Waiter>::iterator it;
        ~Done() { std::lock_guard<std::mutex> lock(r.watch_mutex_); r.waiters_.erase(it); }
    } done{*this, self};
    event.wait_and_throw();
    check_async();
}

void Runtime::finish(sycl::queue& queue) {
    if (!owns(&queue)) throw DeviceError("cannot drain a queue not owned by the Xe runtime");
    wait(queue.ext_oneapi_submit_barrier(), &queue == &compute_ ? "compute queue drain"
                                            : &queue == &transfer_ ? "transfer queue drain" : "engine stream drain");
}
void Runtime::finish() {
    finish(compute_);
    finish(transfer_);
    std::vector<sycl::queue*> streams;
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        for (auto& q : streams_) streams.push_back(&q);
    }
    for (auto* q : streams) finish(*q);
}

void Runtime::free(void* pointer) noexcept {
    if (!pointer) return;
    try {
        finish();
        sycl::free(pointer, context_);
    } catch (const std::exception& e) { fatal(e.what()); }
}

namespace {
uint64_t env_mib(const char* name) {
    const char* v = std::getenv(name);
    return v ? (uint64_t) std::strtoull(v, nullptr, 10) << 20 : 0;
}
}  // namespace

uint64_t vram_limit_bytes() {
    static const uint64_t v = env_mib("STRATA_VRAM_LIMIT_MIB");
    return v;
}
uint64_t max_alloc_limit_bytes() {
    static const uint64_t v = env_mib("STRATA_MAX_ALLOC_MIB");
    return v;
}

void apply_vram_limit(uint64_t& free, uint64_t& total) {
    const uint64_t lim = vram_limit_bytes();
    if (lim == 0 || lim >= total) return;
    const uint64_t used = total > free ? total - free : 0;
    total = lim;
    free = lim > used ? lim - used : 0;
}

uint64_t max_alloc_bytes() {
    const uint64_t dev = Runtime::get().device().get_info<sycl::info::device::max_mem_alloc_size>();
    const uint64_t lim = max_alloc_limit_bytes();
    return lim != 0 && lim < dev ? lim : dev;
}

DeviceInfo device_info(int ordinal) {
    if (ordinal != 0) throw DeviceError("only device ordinal 0 is supported");
    const auto& d = Runtime::get().device();
    DeviceInfo out;
    out.ordinal = ordinal;
    out.name = d.get_info<sycl::info::device::name>();
    out.backend = d.get_backend() == sycl::backend::ext_oneapi_level_zero ? "Level Zero"
                  : d.get_backend() == sycl::backend::ext_oneapi_cuda  ? "CUDA"
                  : d.get_backend() == sycl::backend::ext_oneapi_hip   ? "HIP"
                                                                       : "other";
    out.vendor_id = d.get_info<sycl::info::device::vendor_id>();
    out.device_id = d.has(sycl::aspect::ext_intel_device_id) ? d.get_info<sycl::ext::intel::info::device::device_id>() : 0;
    out.driver_version = d.get_info<sycl::info::device::driver_version>();
    out.platform_version = d.get_platform().get_info<sycl::info::platform::version>();
    out.total_bytes = d.get_info<sycl::info::device::global_mem_size>();
    if (d.has(sycl::aspect::ext_intel_free_memory)) {
        try {
            out.free_bytes = d.get_info<sycl::ext::intel::info::device::free_memory>();
            out.free_bytes_known = true;
        } catch (const sycl::exception&) { /* Optional telemetry; never substitute zero for unknown. */ }
    }
    uint64_t free = out.free_bytes_known ? out.free_bytes : out.total_bytes;
    apply_vram_limit(free, out.total_bytes);
    if (out.free_bytes_known) out.free_bytes = free;
    out.compute_units = d.get_info<sycl::info::device::max_compute_units>();
    out.integrated = integrated(d);
    out.subgroup_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    out.fp64 = d.has(sycl::aspect::fp64);
    out.host_usm = d.has(sycl::aspect::usm_host_allocations);
    out.device_usm = d.has(sycl::aspect::usm_device_allocations);
    return out;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison) : capacity_(bytes), ordinal_(ordinal) {
    if (ordinal != 0 || !bytes || bytes > std::numeric_limits<size_t>::max())
        throw DeviceError("DeviceArena requires nonzero bytes and device ordinal 0");
    auto& runtime = Runtime::get();
    // 4 KB aligned by hand: the CUDA adapter refuses an alignment its driver does not promise (cuMemAlloc gives
    // 256 bytes; UNSUPPORTED_ALIGNMENT whenever the pointer comes back unaligned).  Level Zero returns pages, so
    // the offset is 0 there.
    raw_ = sycl::malloc_device(bytes + 4095, runtime.device(), runtime.context());
    if (!raw_) throw DeviceError("device USM allocation failed for " + std::to_string(bytes) + " bytes");
    base_ = (void*) (((uintptr_t) raw_ + 4095) & ~(uintptr_t) 4095);
    try {
        // in 1 GiB pieces: a single memset past 4 GiB never completes on the B70 (see gpu.cpp)
        if (poison) {
            sycl::event e;
            for (uint64_t o = 0; o < bytes; o += 1ull << 30)
                e = runtime.compute().memset((uint8_t*) base_ + o, 0xff, std::min<uint64_t>(1ull << 30, bytes - o));
            runtime.wait(e, "arena poison fill");
        }
    } catch (...) { runtime.free(raw_); raw_ = base_ = nullptr; throw; }
}

DeviceArena::~DeviceArena() { Runtime::get().free(raw_); }

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (!align || (align & (align - 1))) throw DeviceError("alignment must be a power of two");
    const uintptr_t address = reinterpret_cast<uintptr_t>(base_) + used_;
    const uint64_t padding = (align - (address & (align - 1))) & (align - 1);
    if (padding > capacity_ - used_ || bytes > capacity_ - used_ - padding)
        throw DeviceError("DeviceArena capacity exceeded");
    const uint64_t offset = used_ + padding;
    used_ = offset + bytes;
    return static_cast<uint8_t*>(base_) + offset;
}
}  // namespace strata::core
