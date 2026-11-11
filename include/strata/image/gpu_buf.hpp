// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/gpu_buf.hpp - the image engine's device buffers, the parts every model's file had its own
// copy of: the stream's queue (q_of), an allocation the owner's list keeps (alloc, freed through release
// or the list's end), and its release.  `what` names the model in the error messages, as each file's copy did.
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::image {

inline sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

/// n values of T on the device (at least one, so a zero-length row still has a pointer), kept in `owned`; `zero`
/// clears them.  Throws with `what` (the model's name) when the allocation or the clearing fails.
template<class T> T* alloc(std::vector<void*>& owned, int64_t n, const char* what, bool zero = false) {
    T* p = nullptr;
    const size_t bytes = (size_t) std::max<int64_t>(n, 1) * sizeof(T);
    if (!gpu::alloc_device(&p, bytes))
        throw std::runtime_error(std::string(what) + ": GPU memory: " + gpu::last_error());
    if (zero && !gpu::memset(p, 0, bytes))
        throw std::runtime_error(std::string(what) + ": memset: " + gpu::last_error());
    owned.push_back(p);
    return p;
}

inline void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

}  // namespace strata::image
