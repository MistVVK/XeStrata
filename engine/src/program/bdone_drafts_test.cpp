// src/program/bdone_drafts_test.cpp - the draft counters of a --batch slot in its BDONE line (#5890; header-only, CPU).
#include "strata/program/bdone_drafts.hpp"

#include <cstdio>

using strata::program::bdone_drafts_text;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    CHECK(bdone_drafts_text(0, 0).empty());               // a slot that checked no proposal: the BDONE line is as it was
    CHECK(bdone_drafts_text(-1, 0).empty());
    CHECK(bdone_drafts_text(270, 203) == " drafts=270:203");
    CHECK(bdone_drafts_text(5, 0) == " drafts=5:0");      // proposals checked, none kept: the counters say so (not "0 drafts")
    CHECK(bdone_drafts_text(5, 9) == " drafts=5:5");      // never more kept than checked
    CHECK(bdone_drafts_text(5, -2) == " drafts=5:0");
    if (failed) return 1;
    std::printf("bdone_drafts_test: ok\n");
    return 0;
}
