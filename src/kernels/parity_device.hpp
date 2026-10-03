// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/parity_device.hpp - device plumbing shared by the parity programs ported from CUDA.
//
// Each program keeps its fixtures, references and tolerances; this only replaces cudaMalloc, cudaMemcpy and
// stream synchronization with the Xe runtime.  Buffers live until the Device object goes away, and every copy
// back waits for the compute queue first, so a kernel launched on it (or synchronously) is finished before its
// output is read.
#pragma once

#include "strata/core/runtime.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace strata::parity {

// The compute queue as the kernels' `stream` argument.
inline void* stream() { return &core::Runtime::get().compute(); }

template<class T> void copy_in(T* device, const T* host, size_t count) {
    auto& runtime = core::Runtime::get();
    runtime.wait(runtime.copy_async(device, host, count * sizeof(T), {runtime.compute().ext_oneapi_submit_barrier()}),
                 "parity upload");
}

template<class T> void copy_out(T* host, const T* device, size_t count) {
    auto& runtime = core::Runtime::get();
    runtime.wait(runtime.copy_async(host, device, count * sizeof(T), {runtime.compute().ext_oneapi_submit_barrier()}),
                 "parity download");
}

struct Device {
    std::vector<std::unique_ptr<core::DeviceArena>> arenas;

    template<class T> T* alloc(size_t count) {
        arenas.push_back(std::make_unique<core::DeviceArena>(count * sizeof(T) > 0 ? count * sizeof(T) : 1));
        return static_cast<T*>(arenas.back()->base());
    }
    template<class T> T* upload(const std::vector<T>& host) {
        T* d = alloc<T>(host.size());
        copy_in(d, host.data(), host.size());
        return d;
    }
};

inline void sync() { core::Runtime::get().finish(core::Runtime::get().compute()); }

// Byte-level forms for programs that allocated with cudaMalloc: the buffers live until the process exits.
// The runtime is created first so that it outlives the allocations it frees.
inline Device& process_device() {
    core::Runtime::get();
    static Device device;
    return device;
}
inline void* alloc_bytes(size_t bytes) { return process_device().alloc<unsigned char>(bytes); }
// Host memory kernels read and write directly (CUDA's cudaHostAllocMapped): host USM, one address on both sides,
// kept until the process exits.
inline void* alloc_host_bytes(size_t bytes) {
    auto& rt = core::Runtime::get();
    void* p = sycl::malloc_host(bytes, rt.context());
    if (!p) throw core::DeviceError("parity: host USM allocation failed");
    return p;
}
inline void copy_bytes_in(void* device, const void* host, size_t bytes) {
    copy_in(static_cast<unsigned char*>(device), static_cast<const unsigned char*>(host), bytes);
}
inline void copy_bytes_out(void* host, const void* device, size_t bytes) {
    copy_out(static_cast<unsigned char*>(host), static_cast<const unsigned char*>(device), bytes);
}

}  // namespace strata::parity
