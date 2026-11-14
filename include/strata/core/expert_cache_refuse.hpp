#pragma once
// Tests only: make one of the segmented expert cache's driver calls fail on purpose, so a test can check what a shrink
// that the driver refused part way leaves behind (src/core/expert_cache.cpp).  The engine never sets it.

#include <cstdint>

namespace strata::core::ec_test {

/// Asked before each unmap and each release a shrink makes (`call` is "unmap" or "release", `seg` the segment's index).
/// True: that call counts as refused by the driver and is not made.  Null (the default, and always in the engine): every
/// call is made as usual.
extern bool (*refuse_segment_call)(const char* call, int64_t seg);

}  // namespace strata::core::ec_test
