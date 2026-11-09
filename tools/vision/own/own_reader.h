// tools/vision/own/own_reader.h - strata-vision's reader loop on our own tower (`--tower own`): the same protocol as the mtmd one.
//
//   prints "READY <out_dim>", then per stdin line  ENC <image> <output>  ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>";  QUIT ends.
//   The output file is int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd} then float32 [n_tokens][n_embd] (row i at x = i % nx, y = i / nx).
#pragma once

#include <string>

namespace strata_vit {

struct OwnReaderOptions {
    std::string mmproj;
    bool gpu = false;
    int threads = 0;          // CPU threads (<= 0: half the hardware threads, at most 16, as the mtmd reader)
    int max_tokens = 0;       // the helper's --max-tokens (0: the tower's own limit); a picture over it by more than an eighth is refused
    int min_tokens = 0;
};

// True when our tower can read this mmproj with this backend; otherwise `why` says what is missing (`--tower auto` falls back to mtmd then).
bool own_reader_supported(const std::string& mmproj, bool gpu, std::string& why);

// Runs the reader on stdin/stdout; returns the process exit code (0 after QUIT or end of input).
int own_reader_run(const OwnReaderOptions& o);

}  // namespace strata_vit
