// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/constraint_upload.hpp - the sampling constraints' device upload (the logit bias, the grammar's
// token masks), which verify.cpp and plain_model.cpp each had their own copy of: allocate the buffer
// once (it may be larger than this upload: the masks' room for a whole window), copy the bytes.  False on failure;
// the caller keeps its own error wording and its own tracking of the allocation.
#pragma once

#include "strata/core/gpu.hpp"

#include <cstddef>

namespace strata::core {

template<class T>
inline bool constraint_upload(T*& dev, const void* host, size_t alloc_bytes, size_t copy_bytes) {
    if (dev == nullptr && !strata::gpu::alloc_device(&dev, alloc_bytes)) return false;
    return strata::gpu::copy(dev, host, copy_bytes);
}

}  // namespace strata::core
