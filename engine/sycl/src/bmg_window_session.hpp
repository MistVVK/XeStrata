#pragma once
#include <sycl/sycl.hpp>
namespace strata::core { struct SessionState; }
namespace strata::prefill {
void configure_bmg_window(core::SessionState& ss, sycl::queue& queue, bool single_unbatched);
}
