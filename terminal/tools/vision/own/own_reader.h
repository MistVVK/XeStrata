// tools/vision/own/own_reader.h - strata-vision's reader loop on our own tower (`--tower own`): the same protocol as the mtmd one.
//
//   prints "READY <out_dim> own", then per stdin line  ENC <image> <output>  ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>";  QUIT ends.
//   The output file is int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd} then float32 [n_tokens][n_embd] (row i at x = i % nx, y = i / nx).
#pragma once

#include "cpu_caps.h"
#include "vit_geometry.h"

#include <string>

namespace strata_vit {

struct OwnReaderOptions {
    std::string mmproj;
    bool gpu = false;
    int threads = 0;          // CPU threads (<= 0: half the hardware threads, at most 16, as the mtmd reader)
    int max_tokens = 0;       // the helper's --max-tokens (0: the tower's own limit); a picture over it by more than an eighth is refused
    int min_tokens = 0;
};

// Test hook (STRATA_VIT_TEST_FAIL=gpu|load|warmup): our tower fails at that point before READY, as if there were no GPU, the load had failed, or the warm-up had
// run out of memory, so the server's GPU-to-CPU reader fallback can be checked on a box where ours would start.  True when `point` is the one named.
bool own_reader_inject_failure(const char* point);

// Runs the reader on stdin/stdout; returns the process exit code (0 after QUIT or end of input).  A failure before READY (no GPU, the load, the warm-up) is an
// "ERR ..." line on stdout and a non-zero exit code.
int own_reader_run(const OwnReaderOptions& o);

}  // namespace strata_vit
