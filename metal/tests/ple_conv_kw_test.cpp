// metal/tests/ple_conv_kw_test.cpp - the PLE conv weight's load rule (src/engine/ple_conv_kw.h, #5387 / #5416 / #5818):
// F16 is read in place, F32 and BF16 are narrowed to F16, anything else is refused.  No GPU: plain arithmetic.
// Exit 0 = every check passed.
#include "../src/engine/ple_conv_kw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::metal_engine;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

int main() {
    // values exact in F16, BF16 and F32 (so every path must give the same F16 bits), plus one that F16 rounds
    const float v[] = {1.0f, -1.0f, 0.5f, 0.0f, -0.0f, 2.0f, 0.25f, -0.125f, 3.0f, 0.1f};
    const size_t n = sizeof v / sizeof v[0];
    std::vector<uint16_t> want(n);
    for (size_t i = 0; i < n; ++i) want[i] = ref::f32_to_f16(v[i]);

    check(ple_conv_kw_plan(ref::F16) == PleConvKw::file_f16, "F16: read in place");
    check(ple_conv_kw_plan(ref::F32) == PleConvKw::narrow_f32, "F32: narrowed at load");
    check(ple_conv_kw_plan(ref::BF16) == PleConvKw::narrow_bf16, "BF16: narrowed at load (was read as F16 bits before #5387)");
    for (int t : {(int) ref::Q8_0, (int) ref::Q4_K, (int) ref::IQ2_XS, (int) ref::Q2_0, 99})
        check(ple_conv_kw_plan(t) == PleConvKw::unsupported, "a quantized or unknown type is refused");

    // F32 tensor
    check(ple_conv_kw_elems(ref::F32, n * 4) == n, "F32: element count = bytes / 4");
    std::vector<uint16_t> got(n, 0xFFFF);
    check(ple_conv_kw_narrow(ref::F32, v, n, got.data()) && got == want, "F32 -> F16 bits");

    // BF16 tensor of the same values
    std::vector<uint16_t> b(n);
    for (size_t i = 0; i < n; ++i) b[i] = ref::f32_to_bf16(v[i]);
    check(ple_conv_kw_elems(ref::BF16, n * 2) == n, "BF16: element count = bytes / 2");
    std::fill(got.begin(), got.end(), 0xFFFF);
    check(ple_conv_kw_narrow(ref::BF16, b.data(), n, got.data()), "BF16 narrows");
    bool same = true;
    for (size_t i = 0; i < n; ++i) {
        const float a = ref::f16_to_f32(got[i]), e = ref::bf16_to_f32(b[i]);
        same = same && std::fabs(a - e) <= std::fabs(e) * 1e-3f;
    }
    check(same, "BF16 -> F16 keeps every value (within F16's rounding)");
    check(got[0] == 0x3C00 && got[1] == 0xBC00, "BF16 1.0 / -1.0 become F16 0x3C00 / 0xBC00");
    // the pre-#5387 reading: the BF16 bits themselves taken as F16 values
    check(ref::f16_to_f32(b[0]) == 1.875f && ref::f16_to_f32(b[2]) != 0.5f,
          "control: BF16 1.0 (0x3F80) read as F16 bits is 1.875, 0.5 is not 0.5 (the garbage of #5416 / #5818)");

    // F16 is not copied, other types are not narrowed
    check(ple_conv_kw_elems(ref::F16, n * 2) == n, "F16: element count = bytes / 2");
    check(!ple_conv_kw_narrow(ref::F16, want.data(), n, got.data()), "F16: no narrowing (the file's tensor is used)");
    check(!ple_conv_kw_narrow(ref::Q8_0, want.data(), n, got.data()) && ple_conv_kw_elems(ref::Q8_0, 34) == 0,
          "Q8_0: refused");

    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
