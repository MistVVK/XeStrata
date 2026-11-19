// sycl/src/core/dflash2_head_test.cpp - #5286: the DFlash2 draft head of a Q6_K output head that STRATA_Q6_PLANAR stores as planes.
// A random Q6_K matrix (the output head), converted to planes as native_head.cpp does at load; the draft vocabulary's rows are gathered with
// Dflash2Drafter::gather_draft_head and served with native_q6_k_mmvq, bitwise against the same rows of the whole head's result.  A Q8_0 head (never planar) is
// gathered as rows, byte for byte.  `--legacy` gathers the planar head as GGUF rows, the way dflash2.cpp did before the fix: its draft head is garbage and the
// test must FAIL (registered WILL_FAIL).  Needs one SYCL GPU; no model.
//   dflash2_head_test [--legacy]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/dflash2.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::Dflash2Drafter;

static int g_fail = 0;
static void check(bool ok, const char* what) { std::printf("  %-96s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

int main(int argc, char** argv) {
    const bool legacy = argc > 1 && std::strcmp(argv[1], "--legacy") == 0;
    sycl::queue* s = &dpct::get_in_order_queue();
    std::printf("device: %s%s\n", s->get_device().get_info<sycl::info::device::name>().c_str(), legacy ? " (legacy: GGUF-row gather of the planar head)" : "");
    const int n_in = 2560, n_out = 4096, ncols = 3, nsel = 1500;
    const size_t row_bytes = (size_t) (n_in / 256) * 210, wbytes = row_bytes * n_out;
    std::vector<uint8_t> hw(wbytes);
    uint32_t rs = 12345;
    for (auto& b : hw) b = (uint8_t) (rs = rs * 1664525u + 1013904223u, rs >> 13);
    for (size_t b = 0; b < wbytes / 210; ++b) { hw[b * 210 + 208] = 0x00; hw[b * 210 + 209] = 0x3c; }   // d = 1.0 (finite)
    std::vector<float> hx((size_t) ncols * n_in);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 7919) % 1000) / 500.f - 1.f;
    float* x = sycl::malloc_device<float>(hx.size(), *s);
    void* xq = sycl::malloc_device((size_t) ncols * (n_in / 32) * 36, *s);
    s->memcpy(x, hx.data(), hx.size() * 4).wait();
    strata::kernels::native_quantize_q8_1(x, xq, n_in, ncols, s);

    std::vector<int32_t> rows(nsel);
    for (int i = 0; i < nsel; ++i) rows[(size_t) i] = (int32_t) (((uint64_t) i * 2654435761ull + 777) % (uint64_t) n_out);
    int32_t* rd = sycl::malloc_device<int32_t>(nsel, *s);
    s->memcpy(rd, rows.data(), nsel * 4).wait();

    // the whole head, GGUF layout: the reference result
    void* w0 = sycl::malloc_device(wbytes, *s);
    s->memcpy(w0, hw.data(), wbytes).wait();
    float* y0 = sycl::malloc_device<float>((size_t) ncols * n_out, *s);
    strata::kernels::native_q6_k_mmvq(w0, xq, y0, n_in, n_out, ncols, s);
    s->wait();
    std::vector<float> ref((size_t) ncols * n_out);
    s->memcpy(ref.data(), y0, ref.size() * 4).wait();

    // the head as planes (what native_head.cpp keeps when STRATA_Q6_PLANAR is on)
    void* wp = sycl::malloc_device(wbytes, *s);
    s->memcpy(wp, hw.data(), wbytes).wait();
    wp = strata::kernels::native_q6_k_planar_adopt(wp, n_in, n_out, "test head");
    if (!strata::kernels::native_q6_k_is_planar(wp)) { std::printf("the head was not made planar (STRATA_Q6_PLANAR=0?): skipped\n"); return 77; }

    std::string err;
    uint8_t* dh = nullptr;
    if (legacy) {   // dflash2.cpp before #5286
        dh = sycl::malloc_device<uint8_t>((size_t) nsel * row_bytes, *s);
        strata::kernels::gather_rows((const uint8_t*) wp, (int64_t) row_bytes, rd, nsel, dh, s);
        s->wait();
    } else {
        dh = Dflash2Drafter::gather_draft_head(wp, 14, (int64_t) row_bytes, rd, nsel, s, err);
        check(dh != nullptr, "the planar head's draft rows are gathered");
        if (dh == nullptr) { std::printf("  %s\n", err.c_str()); return 1; }
        check(strata::kernels::native_q6_k_is_planar(dh), "the draft head is a planar matrix (the registry knows it)");
    }
    float* y1 = sycl::malloc_device<float>((size_t) ncols * nsel, *s);
    strata::kernels::native_q6_k_mmvq(dh, xq, y1, n_in, nsel, ncols, s);
    s->wait();
    std::vector<float> got((size_t) ncols * nsel);
    s->memcpy(got.data(), y1, got.size() * 4).wait();
    size_t bad = 0;
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < nsel; ++i)
            if (std::memcmp(&got[(size_t) c * nsel + i], &ref[(size_t) c * n_out + (size_t) rows[(size_t) i]], 4) != 0) ++bad;
    std::printf("  draft logits of %d rows x %d columns against the whole head's: %zu of %zu differ\n", nsel, ncols, bad, got.size());
    check(bad == 0, "the draft head's logits are bitwise the whole head's rows");
    if (!legacy) {
        strata::kernels::native_q6_k_unpack(dh);
        check(!strata::kernels::native_q6_k_is_planar(dh), "native_q6_k_unpack takes the draft head out of the planar registry");
        sycl::free(dh, *s);

        // a head that is not planar (Q8_0, 34 bytes per 32 values): gathered as rows, byte for byte
        const size_t rb8 = (size_t) (n_in / 32) * 34;
        std::vector<uint8_t> h8(rb8 * n_out);
        for (auto& b : h8) b = (uint8_t) (rs = rs * 1664525u + 1013904223u, rs >> 11);
        uint8_t* w8 = sycl::malloc_device<uint8_t>(h8.size(), *s);
        s->memcpy(w8, h8.data(), h8.size()).wait();
        uint8_t* d8 = Dflash2Drafter::gather_draft_head(w8, 8, (int64_t) rb8, rd, nsel, s, err);
        check(d8 != nullptr, "a Q8_0 head's draft rows are gathered");
        if (d8 != nullptr) {
            std::vector<uint8_t> g8((size_t) nsel * rb8);
            s->memcpy(g8.data(), d8, g8.size()).wait();
            size_t bad8 = 0;
            for (int i = 0; i < nsel; ++i)
                if (std::memcmp(&g8[(size_t) i * rb8], &h8[(size_t) rows[(size_t) i] * rb8], rb8) != 0) ++bad8;
            check(bad8 == 0, "the Q8_0 draft rows are the head's rows, byte for byte");
            sycl::free(d8, *s);
        }
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
