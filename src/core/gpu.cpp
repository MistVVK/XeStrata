// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/gpu.cpp - see include/strata/core/gpu.hpp.
#include "strata/core/gpu.hpp"
#include "strata/core/device.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>
#if __has_include(<sycl/ext/oneapi/experimental/ipc_memory.hpp>)
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#endif
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>

namespace strata::gpu {

namespace sx = sycl::ext::oneapi::experimental;

struct Event {
    std::optional<sycl::event> ev;
    bool timing = false;
};

struct Graph {
    sx::command_graph<sx::graph_state::executable> exec;
    size_t nodes;
};

struct VmemSegment {
    sx::physical_mem mem;
};

namespace {

thread_local std::string g_error;

// The live alloc_device blocks, so a refusal can say how much this process holds (the refusals seen so far came
// with ~26 GiB free in the runs that passed, cause unknown: bench/results/2026-09-30-xe-iq3xxs)
std::mutex g_live_mu;
std::map<const void*, size_t> g_live;
size_t g_live_bytes = 0;

std::string refusal_state() {
    size_t held = 0, n = 0;
    {
        std::lock_guard<std::mutex> lock(g_live_mu);
        held = g_live_bytes;
        n = g_live.size();
    }
    std::string free_mib = "unknown (set ZES_ENABLE_SYSMAN=1)";
    try {
        const auto& d = strata::core::Runtime::get().device();
        if (d.has(sycl::aspect::ext_intel_free_memory))
            free_mib = std::to_string(d.get_info<sycl::ext::intel::info::device::free_memory>() >> 20) + " MiB";
    } catch (const std::exception& e) { free_mib = std::string("unreadable: ") + e.what(); }
    return "; device free " + free_mib + ", this process holds " + std::to_string(held >> 20) + " MiB in " +
           std::to_string(n) + " blocks";
}

bool fail(const char* what, const std::exception& e) {
    g_error = std::string(what) + ": " + e.what();
    return false;
}

core::Runtime& rt() { return core::Runtime::get(); }

// A single memset of more than 4 GiB never completes on the B70; the same bytes in 1 GiB pieces do, and a 5 GiB memcpy
// and kernels addressing past 4 GiB work normally (bench/results/2026-09-30-xe-large-alloc).
constexpr size_t kPiece = size_t(1) << 30;
sycl::event memset_pieces(sycl::queue& queue, void* dst, int value, size_t bytes) {
    sycl::event e;
    for (size_t o = 0; o < bytes; o += kPiece) e = queue.memset((uint8_t*) dst + o, value, std::min(kPiece, bytes - o));
    return e;
}
// A stream is a queue of the device it was created on, whichever device is current (null: the current device's
// compute queue).
sycl::queue& q(Stream s) {
    if (!s) return rt().compute();
    core::Runtime* owner = core::Runtime::owner_of_stream(s);
    if (!owner) throw core::DeviceError("the stream is not a queue of the Xe runtime");
    return owner->stream(s);
}
core::Runtime& rt_of(Stream s) {
    if (!s) return rt();
    core::Runtime* owner = core::Runtime::owner_of_stream(s);
    return owner ? *owner : rt();
}

// the recordings in progress, by queue
std::mutex g_capture_mutex;
std::map<const sycl::queue*, sx::command_graph<sx::graph_state::modifiable>> g_captures;

}  // namespace

const char* last_error() { return g_error.c_str(); }

Stream stream_create() {
    try { return rt().create_stream(); }
    catch (const std::exception& e) { fail("stream_create", e); return nullptr; }
}

void stream_destroy(Stream s) {
    try { rt_of(s).destroy_stream(static_cast<sycl::queue*>(s)); }
    catch (const std::exception& e) { fail("stream_destroy", e); }
}

bool stream_sync(Stream s) {
    try { rt_of(s).finish(q(s)); return true; }
    catch (const std::exception& e) { return fail("stream_sync", e); }
}

bool stream_idle(Stream s) {
    try { return q(s).ext_oneapi_empty(); }
    catch (const std::exception& e) { return fail("stream_idle", e); }
}

bool device_sync() {
    try { rt().finish(); return true; }
    catch (const std::exception& e) { return fail("device_sync", e); }
}

void* alloc_device(size_t bytes) {
    try {
        void* p = sycl::malloc_device(bytes, rt().device(), rt().context());
        if (!p) {
            g_error = "alloc_device: " + std::to_string(bytes) + " bytes refused" + refusal_state();
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(g_live_mu);
        g_live[p] = bytes;
        g_live_bytes += bytes;
        return p;
    } catch (const std::exception& e) { fail("alloc_device", e); return nullptr; }
}

void* alloc_host(size_t bytes) {
    try {
        void* p = sycl::malloc_host(bytes, rt().context());
        if (!p) g_error = "alloc_host: " + std::to_string(bytes) + " bytes refused";
        return p;
    } catch (const std::exception& e) { fail("alloc_host", e); return nullptr; }
}

void free(const void* p) {
    {
        std::lock_guard<std::mutex> lock(g_live_mu);
        if (auto it = g_live.find(p); it != g_live.end()) {
            g_live_bytes -= it->second;
            g_live.erase(it);
        }
    }
    // freed by the device that allocated it, whichever is current
    core::Runtime* owner = core::Runtime::owner(p);
    (owner ? *owner : rt()).free(const_cast<void*>(p));
}

bool peer_supported(int a, int b) {
#ifdef SYCL_EXT_ONEAPI_INTER_PROCESS_COMMUNICATION
    try {
        return a != b && core::Runtime::at(a).device().has(sycl::aspect::ext_oneapi_ipc_memory) &&
               core::Runtime::at(b).device().has(sycl::aspect::ext_oneapi_ipc_memory);
    } catch (const std::exception& e) { return fail("peer_supported", e); }
#else
    (void) a; (void) b;
    g_error = "peer_supported: this SYCL runtime has no sycl_ext_oneapi_inter_process_communication";
    return false;
#endif
}

bool peer_alloc(int owner, int other, size_t bytes, PeerRegion* r) {
    *r = PeerRegion{};
#ifdef SYCL_EXT_ONEAPI_INTER_PROCESS_COMMUNICATION
    namespace ipc = sx::ipc_memory;
    try {
        core::Runtime& o = core::Runtime::at(owner);
        core::Runtime& t = core::Runtime::at(other);
        void* p = sycl::malloc_device(bytes, o.device(), o.context());
        if (!p) { g_error = "peer_alloc: " + std::to_string(bytes) + " bytes refused"; return false; }
        o.compute().memset(p, 0, bytes).wait();
        ipc::handle h = ipc::get(p, o.context());
        void* remote = ipc::open(h.data(), t.context(), t.device());
        ipc::put(h, o.context());
        if (!remote) { sycl::free(p, o.context()); g_error = "peer_alloc: the IPC handle did not open"; return false; }
        *r = PeerRegion{p, remote, owner, other};
        return true;
    } catch (const std::exception& e) { return fail("peer_alloc", e); }
#else
    (void) owner; (void) other; (void) bytes;
    g_error = "peer_alloc: this SYCL runtime has no sycl_ext_oneapi_inter_process_communication";
    return false;
#endif
}

void peer_free(PeerRegion* r) {
#ifdef SYCL_EXT_ONEAPI_INTER_PROCESS_COMMUNICATION
    try {
        if (r->remote) sx::ipc_memory::close(r->remote, core::Runtime::at(r->other).context());
        if (r->local) sycl::free(r->local, core::Runtime::at(r->owner).context());
    } catch (const std::exception& e) { fail("peer_free", e); }
#endif
    *r = PeerRegion{};
}

bool vmem_supported() {
    try { return rt().device().has(sycl::aspect::ext_oneapi_virtual_mem); }
    catch (const std::exception& e) { return fail("vmem_supported", e); }
}

size_t vmem_granularity() {
    try { return sx::get_mem_granularity(rt().device(), rt().context(), sx::granularity_mode::recommended); }
    catch (const std::exception& e) { fail("vmem_granularity", e); return 0; }
}

void* vmem_reserve(size_t bytes) {
    try { return reinterpret_cast<void*>(sx::reserve_virtual_mem(bytes, rt().context())); }
    catch (const std::exception& e) { fail("vmem_reserve", e); return nullptr; }
}

void vmem_free(void* va, size_t bytes) {
    try { sx::free_virtual_mem(reinterpret_cast<uintptr_t>(va), bytes, rt().context()); }
    catch (const std::exception& e) { fail("vmem_free", e); }
}

VmemSegment* vmem_map(void* va, size_t bytes) {
    try {
        auto* seg = new VmemSegment{sx::physical_mem(rt().device(), rt().context(), bytes)};
        try {
            seg->mem.map(reinterpret_cast<uintptr_t>(va), bytes, sx::address_access_mode::read_write);
        } catch (...) {
            delete seg;
            throw;
        }
        return seg;
    } catch (const std::exception& e) { fail("vmem_map", e); return nullptr; }
}

void vmem_unmap(void* va, size_t bytes, VmemSegment* seg) {
    try { sx::unmap(va, bytes, rt().context()); }
    catch (const std::exception& e) { fail("vmem_unmap", e); }
    delete seg;
}

VmemSegment* vmem_new(size_t bytes) {
    try { return new VmemSegment{sx::physical_mem(rt().device(), rt().context(), bytes)}; }
    catch (const std::exception& e) { fail("vmem_new", e); return nullptr; }
}

bool vmem_attach(VmemSegment* seg, void* va, size_t bytes) {
    try {
        seg->mem.map(reinterpret_cast<uintptr_t>(va), bytes, sx::address_access_mode::read_write);
        return true;
    } catch (const std::exception& e) { return fail("vmem_attach", e); }
}

bool vmem_detach(void* va, size_t bytes) {
    try {
        sx::unmap(va, bytes, rt().context());
        return true;
    } catch (const std::exception& e) { return fail("vmem_detach", e); }
}

void vmem_delete(VmemSegment* seg) { delete seg; }

bool is_host_usm(const void* p) {
    if (!p) return false;
    try {
        const core::Runtime* owner = core::Runtime::owner(p);
        return owner && sycl::get_pointer_type(p, owner->context()) == sycl::usm::alloc::host;
    }
    catch (const std::exception& e) { return fail("is_host_usm", e); }
}

bool copy(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return true;
    try { rt().wait(rt().compute().memcpy(dst, src, bytes), "synchronous copy"); return true; }
    catch (const std::exception& e) { return fail("copy", e); }
}

bool copy_async(void* dst, const void* src, size_t bytes, Stream s) {
    if (bytes == 0) return true;
    try { q(s).memcpy(dst, src, bytes); return true; }
    catch (const std::exception& e) { return fail("copy_async", e); }
}

bool copy_async_after(void* dst, const void* src, size_t bytes, Stream s, const Event* after, Event* done) {
    try {
        auto& queue = q(s);
        const sycl::event e = queue.submit([&](sycl::handler& h) {
            if (after != nullptr && after->ev) h.depends_on(*after->ev);
            if (bytes > 0) h.memcpy(dst, src, bytes);
        });
        if (done != nullptr) done->ev = e;
        return true;
    } catch (const std::exception& e) { return fail("copy_async_after", e); }
}

bool copy2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Stream s) {
    if (width == 0 || height == 0) return true;
    // a kernel rather than ext_oneapi_memcpy2d, which a SYCL graph recording does not accept: the verify window and
    // the drafter record this copy
    try {
        auto& queue = q(s);
        const bool words = ((uintptr_t) dst | (uintptr_t) src | dpitch | spitch | width) % 4 == 0;
        if (words) {
            const size_t w = width / 4, dp = dpitch / 4, sp = spitch / 4;
            auto* d = static_cast<uint32_t*>(dst);
            const auto* x = static_cast<const uint32_t*>(src);
            queue.parallel_for(sycl::range<2>(height, w), [=](sycl::id<2> id) { d[id[0] * dp + id[1]] = x[id[0] * sp + id[1]]; });
        } else {
            auto* d = static_cast<uint8_t*>(dst);
            const auto* x = static_cast<const uint8_t*>(src);
            queue.parallel_for(sycl::range<2>(height, width),
                               [=](sycl::id<2> id) { d[id[0] * dpitch + id[1]] = x[id[0] * spitch + id[1]]; });
        }
        return true;
    } catch (const std::exception& e) { return fail("copy2d_async", e); }
}

bool memset(void* dst, int value, size_t bytes) {
    if (bytes == 0) return true;
    try { rt().wait(memset_pieces(rt().compute(), dst, value, bytes), "synchronous memset"); return true; }
    catch (const std::exception& e) { return fail("memset", e); }
}

bool memset_async(void* dst, int value, size_t bytes, Stream s) {
    if (bytes == 0) return true;
    try { memset_pieces(q(s), dst, value, bytes); return true; }
    catch (const std::exception& e) { return fail("memset_async", e); }
}

bool mem_info(size_t* free_bytes, size_t* total_bytes) {
    try {
        const auto& d = rt().device();
        if (!d.has(sycl::aspect::ext_intel_free_memory)) {
            g_error = "mem_info: the device does not report free memory (set ZES_ENABLE_SYSMAN=1)";
            return false;
        }
        uint64_t free = d.get_info<sycl::ext::intel::info::device::free_memory>();
        uint64_t total = d.get_info<sycl::info::device::global_mem_size>();
        if (core::integrated_gpu(d)) core::apply_uma_limit(free);   // its memory is the system RAM
        core::apply_vram_limit(free, total);
        if (free_bytes) *free_bytes = (size_t) free;
        if (total_bytes) *total_bytes = (size_t) total;
        return true;
    } catch (const std::exception& e) { return fail("mem_info", e); }
}

bool device_speed(int device, int* units, int* khz) {
    try {
        const auto& d = core::Runtime::at(device < 0 ? core::current_device() : device).device();
        *units = (int) d.get_info<sycl::info::device::max_compute_units>();
        *khz = (int) d.get_info<sycl::info::device::max_clock_frequency>() * 1000;
        return true;
    } catch (const std::exception& e) { return fail("device_speed", e); }
}

bool host_register(void* p, size_t bytes) {
    try { sx::prepare_for_device_copy(p, bytes, rt().context()); return true; }
    catch (const std::exception& e) { return fail("host_register", e); }
}

void host_unregister(void* p) {
    try { sx::release_from_device_copy(p, rt().context()); }
    catch (const std::exception& e) { fail("host_unregister", e); }
}

bool launch_host(Stream s, std::function<void()> fn) {
    try {
        q(s).submit([&](sycl::handler& h) { h.host_task([fn = std::move(fn)] { fn(); }); });
        return true;
    } catch (const std::exception& e) { return fail("launch_host", e); }
}

bool event_create(Event** e, bool timing) {
    *e = new Event();
    (*e)->timing = timing;
    return true;
}

void event_destroy(Event* e) { delete e; }

bool event_record(Event* e, Stream s) {
    // an event without timing is a barrier where the runtime measured timestamps to slow the waits (measure_marks)
    try {
        auto& queue = q(s);
        e->ev = e->timing || !rt_of(s).marks_by_barrier() ? sx::submit_profiling_tag(queue)
                                                          : queue.ext_oneapi_submit_barrier();
        return true;
    }
    catch (const std::exception& ex) { return fail("event_record", ex); }
}

bool event_query(const Event* e) {
    if (!e->ev) return true;
    try {
        return e->ev->get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (const std::exception& ex) { return fail("event_query", ex); }
}

bool event_sync(const Event* e) {
    if (!e->ev) return true;
    try { rt().wait(*e->ev, "event sync"); return true; }
    catch (const std::exception& ex) { return fail("event_sync", ex); }
}

bool stream_wait_event(Stream s, const Event* e) {
    if (!e->ev) return true;
    try { q(s).ext_oneapi_submit_barrier({*e->ev}); return true; }
    catch (const std::exception& ex) { return fail("stream_wait_event", ex); }
}

bool event_elapsed_ms(float* ms, const Event* from, const Event* to) {
    if (!from->ev || !to->ev) { g_error = "event_elapsed_ms: an event was not recorded"; return false; }
    if (!from->timing || !to->timing) { g_error = "event_elapsed_ms: an event was not created for timing"; return false; }
    try {
        const auto a = from->ev->get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto b = to->ev->get_profiling_info<sycl::info::event_profiling::command_end>();
        *ms = (float) ((double) (int64_t) (b - a) / 1e6);
        return true;
    } catch (const std::exception& ex) { return fail("event_elapsed_ms", ex); }
}

bool begin_capture(Stream s) {
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        if (g_captures.count(&queue)) { g_error = "begin_capture: the stream is already recording"; return false; }
        auto it = g_captures.emplace(&queue, sx::command_graph<sx::graph_state::modifiable>(queue.get_context(),
                                                                                             queue.get_device())).first;
        it->second.begin_recording(queue);
        return true;
    } catch (const std::exception& e) { return fail("begin_capture", e); }
}

bool end_capture(Stream s, Graph** out) {
    *out = nullptr;
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&queue);
        if (it == g_captures.end()) { g_error = "end_capture: the stream is not recording"; return false; }
        auto graph = std::move(it->second);
        g_captures.erase(it);
        graph.end_recording(queue);
        const size_t nodes = graph.get_nodes().size();
        if (nodes == 0) { g_error = "end_capture: the recording holds ZERO nodes - the body submitted nothing"; return false; }
        *out = new Graph{graph.finalize(), nodes};
        return true;
    } catch (const std::exception& e) { return fail("end_capture", e); }
}

void abandon_capture(Stream s) {
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&queue);
        if (it == g_captures.end()) return;
        const auto graph = it->second;
        for (auto b = g_captures.begin(); b != g_captures.end();) {   // the queue and its open branches
            if (b->second == graph) {
                b->second.end_recording(*const_cast<sycl::queue*>(b->first));
                b = g_captures.erase(b);
            } else {
                ++b;
            }
        }
    } catch (const std::exception& e) { fail("abandon_capture", e); }
}

bool stream_fork(Stream from, Stream to) {
    try {
        auto& a = q(from);
        auto& b = q(to);
        {
            std::lock_guard<std::mutex> lock(g_capture_mutex);
            auto it = g_captures.find(&a);
            if (it != g_captures.end()) {
                if (g_captures.count(&b)) { g_error = "stream_fork: the branch is already recording"; return false; }
                auto graph = it->second;   // a handle: the branch records into the same graph
                graph.begin_recording(b);
                g_captures.emplace(&b, graph);
            }
        }
        const sycl::event e = a.ext_oneapi_submit_barrier();
        b.ext_oneapi_submit_barrier({e});
        return true;
    } catch (const std::exception& e) { return fail("stream_fork", e); }
}

bool stream_join(Stream into, Stream from) {
    try {
        auto& a = q(into);
        auto& b = q(from);
        const sycl::event e = b.ext_oneapi_submit_barrier();
        a.ext_oneapi_submit_barrier({e});
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&b);
        if (it != g_captures.end()) {
            it->second.end_recording(b);
            g_captures.erase(it);
        }
        return true;
    } catch (const std::exception& e) { return fail("stream_join", e); }
}

bool is_capturing(Stream s) {
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        return g_captures.count(&queue) != 0;
    } catch (const std::exception& e) { return fail("is_capturing", e); }
}

bool graph_launch(const Graph* g, Stream s) {
    if (!g) { g_error = "graph_launch: null graph"; return false; }
    try { q(s).ext_oneapi_graph(g->exec); return true; }
    catch (const std::exception& e) { return fail("graph_launch", e); }
}

size_t graph_nodes(const Graph* g) { return g ? g->nodes : 0; }

void graph_destroy(Graph* g) {
    if (!g) return;
    try { rt().finish(); } catch (const std::exception& e) { fail("graph_destroy", e); }
    delete g;
}

}  // namespace strata::gpu
