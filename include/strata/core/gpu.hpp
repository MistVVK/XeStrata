// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/gpu.hpp - the host-side GPU operations the engine uses, on the Xe runtime.
//
// The engine was written against the CUDA runtime: streams, device and mapped-pinned allocations, synchronous and
// stream-ordered copies, events and captured graphs, each call returning a status the caller checks.  These are the
// same operations with the same contracts on SYCL, as handles, so the engine's control flow and error handling stay
// as written:
//
//   Stream          a runtime queue (Runtime::create_stream), void* like cudaStream_t; null = the compute queue,
//                   which plays the CUDA legacy stream (the synchronous copies and memsets are ordered on it)
//   alloc_device    device USM
//   alloc_host      host USM: pinned and device-accessible at the same address, so device_pointer is the identity
//                   (cudaHostAlloc(Mapped) + cudaHostGetDevicePointer); any other pointer is refused
//   Event*          a queue position, a profiling tag: queried, waited on, waited for by another stream, timed
//   Graph*          an executable SYCL command graph; begin_capture/end_capture record what a stream is given
//                   (stream capture + instantiate), graph_launch replays it on a stream
//
// A failing call returns false (or null) and leaves its message in last_error(); SYCL's synchronous exceptions are
// caught here, asynchronous ones surface at the runtime's next wait.
#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>

namespace strata::gpu {

using Stream = void*;
struct Event;
struct Graph;

/// The message of the last failing call on this thread.
const char* last_error();
/// Issue #10: one line on an exception that left the engine: its text, the SYCL error code and Unified Runtime result
/// when it is a SYCL exception (error 40 is also what a kernel launch refused for device memory reports), the VRAM
/// free if it can still be read, and whether the GPU still answers (Runtime::device_status).
std::string failure_report(const std::exception& e);

/// `high_priority`: a stream the device may run ahead of the others (Runtime::create_stream).
Stream stream_create(bool high_priority = false);
/// A stream for copies and events alone (Runtime::create_copy_stream): the copy engine where the device has one.
Stream stream_create_copy();
void stream_destroy(Stream s);
bool stream_sync(Stream s);
/// What a poll found (stream_status, event_status).  Failed: the query itself failed, its message in last_error(); a
/// device error stays in its context, so every later query fails too and a wait that read it as Pending would spin to
/// its timeout or forever (issue #12: an illegal address on a later stage's CUDA GPU).
enum class Status { Done, Pending, Failed };
/// Done when every command submitted to `s` has completed (cudaStreamQuery == cudaSuccess).
Status stream_status(Stream s);
/// stream_status(s) == Done: a failed query reads as "not yet", for a caller that waits elsewhere.
bool stream_idle(Stream s);
bool device_sync();

void* alloc_device(size_t bytes);
/// The bytes this process holds through alloc_device on the current device.
size_t device_held();
/// Host USM.  A refusal while the pinned memory's hard reserve is held (pin_reserve_take) gives the reserve back and
/// tries once more.
void* alloc_host(size_t bytes);
/// Pinned memory as a budget (upstream 4f74de9cd): Windows (WDDM) lets a process page-lock about half the RAM, and the
/// expert arena, pinned until the driver refuses, left nothing for the few MiB of pinned buffers allocated after it
/// ("mapped staging allocation failed").  pin_reserve_take holds two reserves before the arena pins (by default on
/// Windows, any GPU: 640 MiB and a quarter of it, at least 64 MiB; STRATA_PIN_RESERVE_MIB=N sets the first on
/// any system, 0 none), a reserve that cannot be had being halved or skipped; the soft one goes back once the arena is
/// placed, the hard one at the first refused alloc_host or once the engine is up (pin_reserve_release).
void pin_reserve_take();
void pin_reserve_release_soft();
void pin_reserve_release();
template <class T> bool alloc_device(T** p, size_t bytes) { *p = static_cast<T*>(alloc_device(bytes)); return *p; }
template <class T> bool alloc_host(T** p, size_t bytes) { *p = static_cast<T*>(alloc_host(bytes)); return *p; }
/// Frees device or host USM after draining the runtime's queues (null is a no-op).
void free(const void* p);
/// The device address of host USM (itself); false for memory that is not host USM.
bool is_host_usm(const void* p);
inline bool device_pointer(void** d, const void* h) {
    *d = is_host_usm(h) ? const_cast<void*>(h) : nullptr;
    return *d != nullptr;
}
/// Whether the current GPU's kernels can read the host memory at `p` in place, as the PCIe share reads the missed
/// experts: host USM of its own runtime.  A CUDA or HIP GPU also reads host memory registered on it (host_register,
/// mapped at the same address: kernels_read_registered); a Level Zero GPU reads neither another context's host USM
/// nor a registration, which is for copies only.
bool kernels_read_host(const void* p);
bool kernels_read_registered();

/// Synchronous copy, ordered after the work already on the compute queue (the CUDA legacy stream).
bool copy(void* dst, const void* src, size_t bytes);
bool copy_async(void* dst, const void* src, size_t bytes, Stream s);
/// stream_wait_event(s, after), copy_async and event_record(done, s) as one submission: the copy waits for `after`
/// (null or not recorded: for nothing) and `done` marks the copy (an event without timing; null: not kept).  Three
/// submissions a copy filled the RTX 4070's stream when the prompt path queued its ring's 384 copies, and each copy
/// held its issuer for about 160 us (the 2026-10-09-xe-event-record record).
bool copy_async_after(void* dst, const void* src, size_t bytes, Stream s, const Event* after, Event* done);
/// `height` rows of `width` bytes between pitched buffers (cudaMemcpy2DAsync).
bool copy2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Stream s);
bool memset(void* dst, int value, size_t bytes);
bool memset_async(void* dst, int value, size_t bytes, Stream s);

bool mem_info(size_t* free_bytes, size_t* total_bytes);

/// Virtual memory (sycl_ext_oneapi_virtual_mem; the Level Zero and CUDA adapters, not HIP in intel/llvm 7.1.1): one
/// address range reserved once and backed by physical segments that can be given back to the driver and taken again,
/// every address staying the same (#533).  vmem_supported is the device's report; the sizes are multiples of
/// vmem_granularity().
struct VmemSegment;
bool vmem_supported();
size_t vmem_granularity();
void* vmem_reserve(size_t bytes);
void vmem_free(void* va, size_t bytes);
/// A new physical segment of `bytes` mapped read-write at `va`; null when the device has no memory for it.
VmemSegment* vmem_map(void* va, size_t bytes);
/// Unmaps `seg` from `va` and gives its memory back (the device must not be using it).
void vmem_unmap(void* va, size_t bytes, VmemSegment* seg);
/// The chunk form (the elastic K/V, vmm.hpp): a physical segment of `bytes` not mapped anywhere (null: no memory),
/// mapped read-write at a reserved `va`, unmapped from it (it keeps its bytes and can be mapped elsewhere), freed.
VmemSegment* vmem_new(size_t bytes);
bool vmem_attach(VmemSegment* seg, void* va, size_t bytes);
bool vmem_detach(void* va, size_t bytes);
void vmem_delete(VmemSegment* seg);
/// The device's compute units and maximum clock in kHz (cudaDevAttrMultiProcessorCount / ClockRate); device 0 only.
bool device_speed(int device, int* units, int* khz);

/// Registers pageable memory for fast device copies (cudaHostRegister).  Registered memory is a copy source or
/// destination only: kernels cannot address it (docs/XE.md, Runtime contract).
bool host_register(void* p, size_t bytes);
/// The same on engine device `device` instead of the current one.
bool host_register_on(int device, void* p, size_t bytes);
/// Whether engine device `a`'s kernels can access device `b`'s memory directly (sycl_ext_oneapi_peer_access: NVLink
/// or PCIe peer-to-peer, one backend); false for devices of different backends or without the extension.
bool peer_access(int a, int b);
/// Whether engine devices `a` and `b` are run by the same backend (Level Zero, CUDA, HIP): one maker's runtime, whose
/// host USM and registrations the other device's runtime may not take.
bool same_backend(int a, int b);
/// The device's name as the runtime reports it.
std::string device_name(int device);
void host_unregister_on(int device, void* p);
void host_unregister(void* p);

/// Runs `fn` on a host thread after the work already on `s`; later work on `s` waits for it (cudaLaunchHostFunc).
bool launch_host(Stream s, std::function<void()> fn);

/// `timing`: the event can be measured with event_elapsed_ms (cudaEventCreate); without it only ordered and waited on,
/// which is cheaper to record (cudaEventDisableTiming).
bool event_create(Event** e, bool timing = false);
void event_destroy(Event* e);
/// Marks the current end of `s`.
bool event_record(Event* e, Stream s);
/// Done when the marked position has been reached (and when nothing was recorded).
Status event_status(const Event* e);
/// event_status(e) == Done: a failed query reads as "not yet", for a caller that waits elsewhere.
bool event_query(const Event* e);
bool event_sync(const Event* e);
/// Makes later work on `s` wait for the marked position (cudaStreamWaitEvent).
bool stream_wait_event(Stream s, const Event* e);
/// Milliseconds between two completed events.
bool event_elapsed_ms(float* ms, const Event* from, const Event* to);

/// Starts recording the commands submitted to `s`; they are not executed.
bool begin_capture(Stream s);
/// Stops recording and finalizes the graph.  An empty recording fails.
bool end_capture(Stream s, Graph** out);
/// Stops a recording in progress and discards it (and the branches stream_fork opened on it).
void abandon_capture(Stream s);
/// A branch: what `to` is given next runs after everything `from` was given so far, beside what `from` is given
/// next.  While `from` records a graph, `to` records into the same graph until stream_join.
bool stream_fork(Stream from, Stream to);
/// The branch's end: what `into` is given next runs after everything `from` was given; `from` stops recording.
bool stream_join(Stream into, Stream from);
bool is_capturing(Stream s);
bool graph_launch(const Graph* g, Stream s);
size_t graph_nodes(const Graph* g);
/// Destroys a graph after draining the runtime's queues (null is a no-op).
void graph_destroy(Graph* g);

}  // namespace strata::gpu
