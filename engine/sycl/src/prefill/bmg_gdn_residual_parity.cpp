#include "bmg_gdn_residual.hpp"
#include "bmg_gdn_tiles.hpp"
#include "bmg_gdn_test_timing.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace {
template <class T> void fill(sycl::queue& q, T* values, size_t n, uint32_t seed) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        uint32_t u = (uint32_t(id[0]) + seed) * 2654435761u;
        u ^= u >> 15;
        values[id] = T(float(int(u % 4001) - 2000) * 0.001f);
    });
}

template <class T> bool identical(sycl::queue& q, const T* a, const T* b, size_t count) {
    auto* mismatches = sycl::malloc_device<uint32_t>(1, q);
    q.fill(mismatches, 0u, 1);
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
         using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
         if (sycl::bit_cast<Bits>(a[id]) != sycl::bit_cast<Bits>(b[id]))
             sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                              sycl::access::address_space::global_space>(*mismatches)
                 .fetch_add(1);
     }).wait_and_throw();
    uint32_t count_bad;
    q.memcpy(&count_bad, mismatches, sizeof(count_bad)).wait_and_throw();
    bool ok = count_bad == 0;
    sycl::free(mismatches, q);
    return ok;
}

bool compare(sycl::queue& q, int rows, bool with_low, bool timing) {
    constexpr int n = 2560, hc = 4;
    const int stride = hc * n + (with_low ? 16 : 0);
    const size_t residual_count = size_t(rows) * hc * n, image_count = size_t(rows) * stride;
    auto allocate = [&](size_t count) { return sycl::malloc_device<float>(count, q); };
    auto image = [&] {
        auto* x = sycl::malloc_device<uint16_t>(image_count, q);
        q.fill(x, uint16_t(0x7155), image_count);
        return x;
    };
    float *r = allocate(residual_count), *reference = allocate(residual_count);
    float *bo = allocate(size_t(rows) * n), *w = allocate(hc * n), *inj = allocate(rows * (hc + 2));
    float *rs = allocate(rows * hc), *rs_ref = allocate(rows * hc);
    uint16_t *hi = image(), *hi_ref = image();
    uint16_t *lo = with_low ? image() : nullptr, *lo_ref = with_low ? image() : nullptr;
    fill(q, r, residual_count, 3);
    fill(q, bo, size_t(rows) * n, 4);
    fill(q, w, hc * n, 9);
    fill(q, inj, rows * (hc + 2), 13);
    q.memcpy(reference, r, residual_count * sizeof(float));
    auto previous = [&] {
        strata::prefill::gr_write_norm_rs(reference, bo, inj, hc + 2, w, 1e-6f, rs_ref, hi_ref, rows, &q, lo_ref,
                                          stride);
    };
    auto tiled = [&] {
        strata::prefill::bmg_gdn_residual::write_norm(q, r, bo, inj, hc + 2, w, 1e-6f, rs, hi, lo, rows, stride);
    };
    previous();
    tiled();
    bool ok = identical(q, r, reference, residual_count) && identical(q, rs, rs_ref, rows * hc) &&
              identical(q, hi, hi_ref, image_count) && (!with_low || identical(q, lo, lo_ref, image_count));
    if (ok && timing)
        gdn_test::measure(q, "residual", rows, previous, tiled);
    for (void* ptr : std::array<void*, 11>{r, reference, bo, w, inj, rs, rs_ref, hi, hi_ref, lo, lo_ref})
        if (ptr)
            sycl::free(ptr, q);
    return ok;
}
} // namespace

int main(int argc, char** argv) {
    auto& q = dpct::get_in_order_queue();
    if (!strata::prefill::use_bmg_gdn_tiles(q, 8192))
        return 77;
    const bool timing = argc == 2 && std::string_view(argv[1]) == "--timing";
    for (int rows : {512, 1024, 1025, 8192}) {
        if (!compare(q, rows, false, timing))
            return 1;
    }
    return compare(q, 1025, true, false) ? 0 : 1;
}
