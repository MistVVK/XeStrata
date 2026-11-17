// The Tesla P100 prompt-expert default (gp100_prompt_default.hpp): the FP16 path only with STRATA_PREFILL_MMQ unset, every visible card
// cc 6.0, and nothing in force that needs MMQ (STRATA_MMQ_WS=1, STRATA_EXPERT_NIBBLE=1, --peer-device).  Header-only, no GPU.
#include <cstdio>
#include <initializer_list>

#include "strata/prefill/gp100_prompt_default.hpp"

using strata::prefill::Gp100PromptInputs;
using strata::prefill::gp100_fp16_rule;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    Gp100PromptInputs p100;
    p100.all_cc60 = true;
    CHECK(gp100_fp16_rule(p100));   // one or more P100s, nothing set: the FP16 path

    Gp100PromptInputs other = p100;
    other.all_cc60 = false;         // any card that is not cc 6.0 (a P40, a V100, a mixed rig): MMQ as before
    CHECK(!gp100_fp16_rule(other));

    for (const char* v : {"1", "0", "2", ""}) {   // an explicit setting keeps its old meaning; the rule stands aside
        Gp100PromptInputs e = p100;
        e.env = v;
        CHECK(!gp100_fp16_rule(e));
    }

    Gp100PromptInputs ws = p100;
    ws.mmq_ws = true;
    CHECK(!gp100_fp16_rule(ws));

    Gp100PromptInputs nib = p100;
    nib.nibble = true;
    CHECK(!gp100_fp16_rule(nib));

    Gp100PromptInputs peer = p100;  // two P100s with --peer-device keep MMQ and the prompt peer
    peer.peer_device = true;
    CHECK(!gp100_fp16_rule(peer));

    CHECK(!strata::prefill::peer_device_requested().load());   // off until generate.cpp sees --peer-device
    strata::prefill::peer_device_requested().store(true);
    CHECK(strata::prefill::peer_device_requested().load());

    if (failed) {
        std::fprintf(stderr, "gp100_prompt_default_test: %d failed\n", failed);
        return 1;
    }
    std::printf("gp100_prompt_default_test: ok\n");
    return 0;
}
