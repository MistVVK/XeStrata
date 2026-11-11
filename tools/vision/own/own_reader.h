// tools/vision/own/own_reader.h - strata-vision's reader loop on our own tower (`--tower own`): the same protocol as the mtmd one.
//
//   prints "READY <out_dim> own", then per stdin line  ENC <image> <output>  ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>";  QUIT ends.
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
    // `auto` (the default): a failure before READY (no GPU, the load, the warm-up) prints nothing on stdout; own_reader_run returns kOwnStartFailed with the
    // reason in *start_error, and strata-vision starts mtmd's reader in its place.  Null: the failure is an "ERR ..." line, as with `--tower own`.
    std::string* start_error = nullptr;
};

// The tower when no --tower is given: STRATA_VISION_TOWER's value (env, may be null) if it is mtmd, own or auto, else auto; `note` gets a line for a value that
// is none of these.
std::string vision_tower_default(const char* env, std::string& note);

// own_reader_run's return when start_error is set and our tower did not reach READY
constexpr int kOwnStartFailed = 100;

// Test hook (STRATA_VIT_TEST_FAIL=gpu|load|warmup): our tower fails at that point before READY, as if there were no GPU, the load had failed, or the warm-up had
// run out of memory, so the fallback to mtmd can be checked on a box where ours would start.  True when `point` is the one named.
bool own_reader_inject_failure(const char* point);

// True when our tower can read this mmproj with this backend; otherwise `why` says what is missing (`--tower auto` falls back to mtmd then).
bool own_reader_supported(const std::string& mmproj, bool gpu, int max_tokens, std::string& why);      // max_tokens: --max-tokens (0: not given)

// Runs the reader on stdin/stdout; returns the process exit code (0 after QUIT or end of input).
int own_reader_run(const OwnReaderOptions& o);

}  // namespace strata_vit
