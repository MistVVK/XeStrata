#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>

namespace strata {
// Keep transfer ordering, context and async error handling, without allocating
// event timestamps that the prompt path never reads from its transfer queue.
inline std::unique_ptr<sycl::queue> make_prefill_copy_queue(const sycl::queue& compute) {
    return std::make_unique<sycl::queue>(compute.get_context(), compute.get_device(),
        dpct::exception_handler, sycl::property_list{sycl::property::queue::in_order{}});
}
}
