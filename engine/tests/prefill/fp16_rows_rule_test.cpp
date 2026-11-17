// fp16_rows_rule (include/strata/prefill/fp16_rows_rule.hpp): the tokens the FP16 expert path's buffers are sized for.  No GPU, no model.
#include <cstdio>

#include "strata/prefill/fp16_rows_rule.hpp"

using strata::prefill::fp16_buf_tokens;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const int64_t floor = 1024;
    // a fused layout: the tail below the floor only (the same size moe_bufs gives GU / H / Hq), at the floor and above, and a chunk below the floor keeps its own size
    CHECK(fp16_buf_tokens(8192, true, floor) == 1023);
    CHECK(fp16_buf_tokens(2048, true, floor) == 1023);
    CHECK(fp16_buf_tokens(1025, true, floor) == 1023);
    CHECK(fp16_buf_tokens(1024, true, floor) == 1023);
    CHECK(fp16_buf_tokens(1023, true, floor) == 1023);
    CHECK(fp16_buf_tokens(512, true, floor) == 512);
    CHECK(fp16_buf_tokens(1, true, floor) == 1);
    // a different floor (STRATA_PREFILL_STREAM_MIN, the CPU share's limit)
    CHECK(fp16_buf_tokens(8192, true, 2048) == 2047);
    CHECK(fp16_buf_tokens(2000, true, 2048) == 2000);
    // no fused layout (the Tesla P100: FP16 from the floor up; STRATA_PF_FUSED=0; a card the fused kernels do not cover): the full chunk at every size
    CHECK(fp16_buf_tokens(8192, false, floor) == 8192);
    CHECK(fp16_buf_tokens(2048, false, floor) == 2048);
    CHECK(fp16_buf_tokens(1024, false, floor) == 1024);
    CHECK(fp16_buf_tokens(1023, false, floor) == 1023);
    CHECK(fp16_buf_tokens(64, false, floor) == 64);
    if (failed) {
        std::fprintf(stderr, "fp16_rows_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("fp16_rows_rule_test: OK\n");
    return 0;
}
