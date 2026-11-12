// staged_group_entry_default (include/strata/prefill/staged_group_rule.hpp): whether the staged walk (prompt chunks below 1,024 tokens) takes the batched group entry.
// CUDA build: on with nothing set; HIP build: off with nothing set (owner 2026-11-12, "8 yes for nvidia"); STRATA_PF_FUSED=1 turns it on and STRATA_PF_FUSED=0 off in
// both builds; any other non-empty value reads as off (as native_supported reads it); empty reads as unset.  No GPU, no model.
#include <cstdio>

#include "strata/prefill/staged_group_rule.hpp"

using strata::prefill::staged_group_entry_default;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // nothing set
    CHECK(staged_group_entry_default(false, nullptr) == true);    // CUDA: default on
    CHECK(staged_group_entry_default(true, nullptr) == false);    // HIP: default off (yesterday's path)
    CHECK(staged_group_entry_default(false, "") == true);         // an empty variable is unset
    CHECK(staged_group_entry_default(true, "") == false);
    // the switch, both ways, both builds
    CHECK(staged_group_entry_default(false, "1") == true);
    CHECK(staged_group_entry_default(true, "1") == true);         // HIP opt-in
    CHECK(staged_group_entry_default(false, "0") == false);       // CUDA opt-out
    CHECK(staged_group_entry_default(true, "0") == false);
    // any other non-empty value reads as off
    CHECK(staged_group_entry_default(false, "2") == false);
    CHECK(staged_group_entry_default(true, "yes") == false);
    CHECK(staged_group_entry_default(false, "10") == true);       // first character, as native_supported's v[0] == '1'
    if (failed) {
        std::fprintf(stderr, "staged_group_rule_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("staged_group_rule_test passed\n");
    return 0;
}
