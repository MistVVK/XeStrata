// draft_vocab_test: the DFlash2-alone start's draft-vocabulary choice (include/strata/spec/draft_vocab.hpp).
#include "strata/spec/draft_vocab.hpp"

#include <cstdio>
#include <set>
#include <string>

using strata::spec::df2_vocab_resolve;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main() {
    const std::set<std::string> files = {"/mtp/with/draft_vocab.bin", "/given.bin", "/pkg/data/draft_vocab.bin"};
    auto exists = [&](const std::string& p) { return files.count(p) > 0; };

    // the MTP folder has the subset: it is used, packaged default or not
    for (const char* pkg : {"", "/pkg/data/draft_vocab.bin"}) {
        auto r = df2_vocab_resolve("/mtp/with", "", pkg, false, exists);
        CHECK(r.ok && r.path == "/mtp/with/draft_vocab.bin" && !r.packaged && !r.whole_head);
    }
    // a file the person named wins over the folder's
    { auto r = df2_vocab_resolve("/mtp/with", "/given.bin", "/pkg/data/draft_vocab.bin", false, exists); CHECK(r.ok && r.path == "/given.bin" && !r.packaged); }
    // the Mac: the MTP folder has none, the build ships its default: that is used (what Metal did before the merge)
    { auto r = df2_vocab_resolve("/mtp/empty", "", "/pkg/data/draft_vocab.bin", false, exists); CHECK(r.ok && r.packaged && r.path == "/pkg/data/draft_vocab.bin"); }
    // CUDA / HIP / SYCL (no packaged default named): a missing file is the error it was
    { auto r = df2_vocab_resolve("/mtp/empty", "", "", false, exists); CHECK(!r.ok && r.why.find("/mtp/empty/draft_vocab.bin") != std::string::npos); }
    // no subset can be found at all (the build's default is missing too): the error, on every backend
    { auto r = df2_vocab_resolve("/mtp/empty", "", "/nowhere/draft_vocab.bin", false, exists); CHECK(!r.ok && !r.packaged); }
    // a file the person named that does not exist is their mistake: no silent fallback to the default
    { auto r = df2_vocab_resolve("/mtp/with", "/typo.bin", "/pkg/data/draft_vocab.bin", false, exists); CHECK(!r.ok && r.why.find("/typo.bin") != std::string::npos); }
    // the whole head asked for on purpose (STRATA_METAL_DRAFT_VOCAB=full) is not an error
    { auto r = df2_vocab_resolve("/mtp/empty", "", "", true, exists); CHECK(r.ok && r.whole_head); }

    if (g_fail == 0) std::printf("draft_vocab_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
