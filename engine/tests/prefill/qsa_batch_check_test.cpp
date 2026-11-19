// qsa_batch_problem (include/strata/kernels/qsa_batch_check.hpp): each failed predicate of the batch indexer append names itself (#5822).  No GPU.
#include <cstdio>
#include <limits>
#include <string>

#include "strata/kernels/qsa_batch_check.hpp"

using strata::kernels::QsaBatchArgs;
using strata::kernels::qsa_batch_problem;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
static bool has(const std::string& s, const char* w) { return s.find(w) != std::string::npos; }

static QsaBatchArgs good() {
    QsaBatchArgs a;
    a.idx_dim = a.want_dim = 128;
    a.idx_block = a.want_block = 4;
    a.n_rot = a.want_rot = 64;
    a.n = 402;
    a.p0 = 100817;
    a.max_cells = 196608;
    a.pos_base = 0;
    a.epsilon = 1e-6f;
    return a;
}

int main() {
    CHECK(qsa_batch_problem(good()).empty());
    { QsaBatchArgs a = good(); a.has_stream = false; CHECK(has(qsa_batch_problem(a), "no stream")); }
    { QsaBatchArgs a = good(); a.idx_dim = 64; CHECK(has(qsa_batch_problem(a), "geometry idx_dim 64")); }
    { QsaBatchArgs a = good(); a.p0 = -1; CHECK(has(qsa_batch_problem(a), "negative")); }
    // the case a prefix reuse can produce: positions past the session's cells; the text carries p0, n and the cells
    { QsaBatchArgs a = good(); a.p0 = 196500; a.n = 402;
      const std::string s = qsa_batch_problem(a);
      CHECK(has(s, "p0 196500") && has(s, "n 402") && has(s, "196608 cells")); }
    { QsaBatchArgs a = good(); a.max_cells = (int64_t) 1 << 32; a.p0 = 0; a.n = 1; CHECK(has(qsa_batch_problem(a), "max_cells")); }
    { QsaBatchArgs a = good(); a.pos_base = 6; CHECK(has(qsa_batch_problem(a), "multiple of the block 4")); }
    { QsaBatchArgs a = good(); a.epsilon = 0.0f; CHECK(has(qsa_batch_problem(a), "epsilon")); }
    { QsaBatchArgs a = good(); a.epsilon = std::numeric_limits<float>::quiet_NaN(); CHECK(has(qsa_batch_problem(a), "epsilon")); }
    { QsaBatchArgs a = good(); a.rope_problem = "freq_base must be positive"; CHECK(has(qsa_batch_problem(a), "RoPE scaling: freq_base must be positive")); }
    // the old text stays the start of every message
    { QsaBatchArgs a = good(); a.epsilon = -1.0f; CHECK(qsa_batch_problem(a).rfind("native QSA indexer (batch): bad geometry, positions, parameters or scaling", 0) == 0); }
    if (failed) { std::fprintf(stderr, "qsa_batch_check_test: %d failed\n", failed); return 1; }
    std::printf("qsa_batch_check_test: ok\n");
    return 0;
}
