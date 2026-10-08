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
#include <functional>
#include <string>

namespace strata::gpu {

using Stream = void*;
struct Event;
struct Graph;

/// The message of the last failing call on this thread.
const char* last_error();

Stream stream_create();
void stream_destroy(Stream s);
bool stream_sync(Stream s);
/// True when every command submitted to `s` has completed (cudaStreamQuery == cudaSuccess).
bool stream_idle(Stream s);
bool device_sync();

void* alloc_device(size_t bytes);
void* alloc_host(size_t bytes);
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

/// Synchronous copy, ordered after the work already on the compute queue (the CUDA legacy stream).
bool copy(void* dst, const void* src, size_t bytes);
bool copy_async(void* dst, const void* src, size_t bytes, Stream s);
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
/// The device's compute units and maximum clock in kHz (cudaDevAttrMultiProcessorCount / ClockRate); device 0 only.
bool device_speed(int device, int* units, int* khz);

/// Registers pageable memory for fast device copies (cudaHostRegister).  Registered memory is a copy source or
/// destination only: kernels cannot address it (docs/XE.md, Runtime contract).
bool host_register(void* p, size_t bytes);
void host_unregister(void* p);

/// Runs `fn` on a host thread after the work already on `s`; later work on `s` waits for it (cudaLaunchHostFunc).
bool launch_host(Stream s, std::function<void()> fn);

bool event_create(Event** e);
void event_destroy(Event* e);
/// Marks the current end of `s`.
bool event_record(Event* e, Stream s);
/// True when the marked position has been reached (and when nothing was recorded).
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
