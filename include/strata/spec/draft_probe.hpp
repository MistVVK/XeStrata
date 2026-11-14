// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/spec/draft_probe.hpp - which drafter an engine keeps, measured at its start (both engines').
//
// A model can have up to two drafters loaded: the MTP layer of its file and a DFlash2 pack (core/dflash2.hpp).  Which
// decodes faster - or whether no drafter at all does, where a verify row costs much (a slow PCIe link, experts read
// from the CPU) - depends on the machine, the engine and the model, and holding both costs VRAM the expert cache could
// have.  So the serve engines, at the end of their start and before READY, decode a short fixed text with each
// candidate ("arm": a drafter and the largest window it checks), in turn and up to twice over (one round once nothing
// can beat the default any more), and keep the faster: the losers are released from VRAM and RAM.  Nothing is
// written to disk: every start measures again (the user's decision, 2026-11-13).
//
// The measurement is shallow (a few hundred positions): the per-round window rule (DraftPolicy) adapts the window to
// the context's depth afterwards; the drafter itself is chosen once.  XESTRATA_DRAFT_PROBE: auto (the default:
// measure where there are two arms or more), 0 (do not measure: the engine keeps what --draft loads, as before),
// 1 (measure, the same as auto).  --draft mtp|dflash2|off and an explicit --spec are kept as given.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::spec {

enum class Drafter : int { None = 0, Mtp = 1, Df2 = 2 };

const char* drafter_name(Drafter d);

struct ProbeArm {
    Drafter drafter = Drafter::Mtp;
    int spec = 1;   ///< the largest window: the pending token and spec - 1 drafts (1: no drafts)
    bool static_table = false;   ///< DFlash2 with static table proposals
    int block = 0;               ///< DFlash2: 0 = 8 rows, 1 = 16 rows, 2 = adaptive
};

/// One run of an arm: the tokens it decoded and the time it took (ms), the drafts it checked and kept.
struct ProbeRun {
    int arm = 0;
    int64_t tokens = 0;
    double ms = 0;
    int64_t drafted = 0, accepted = 0;
};

struct ProbeVerdict {
    bool measured = false;      ///< every arm ran (else nothing changes: the engine keeps what it loaded)
    Drafter keep = Drafter::Mtp;   ///< the drafter the engine keeps
    bool df2_static = false;    ///< the winning DFlash2 form
    int df2_block = 0;          ///< that form's block mode
    int spec = 0;               ///< the kept drafter's window (0: as configured)
    std::vector<double> tps;    ///< each arm's tokens per second (the sum of its runs)
    std::string line;           ///< the start log's line: every arm's speed and the choice
};

class DraftProbe {
public:
    /// The rule: the default form (the first arm, the MTP at its configured window) stays unless another arm is
    /// faster in every run (each of its runs above each of the default's) and by `switch_margin` (and both arms'
    /// spread between runs) over the default's speed; of those, the fastest.  One drafter is kept: two within a
    /// margin on the short text were not alike at depth (B70, Qwen3.6-35B-A3B: tied at 120 tokens, DFlash2 14% slower
    /// on a 1K prompt).
    explicit DraftProbe(double margin = 0.03, double switch_margin = 0.10) : margin_(margin), switch_(switch_margin) {}

    /// The arms of a start: `mtp` / `df2`: the drafters loaded; `none_ok`: no drafter is a candidate (an engine whose
    /// --draft is auto); `spec`: the configured window, `spec_auto`: the window is a candidate too (--spec auto:
    /// spec and 2); `df2_spec`: DFlash2's window. Fewer than two arms: no measurement.
    static std::vector<ProbeArm> arms(bool mtp, bool df2, bool none_ok, int spec, bool spec_auto, int df2_spec,
                                      bool static_available = false, int wide_mode = 0, bool static_forced = false);

    /// The runs in the order they are made: each arm `rounds` times, interleaved (a b c a b c), after one warm-up
    /// run of the first arm (index -1 in the plan: not counted).
    static std::vector<int> plan(int n_arms, int rounds = 2);

    ProbeVerdict decide(const std::vector<ProbeArm>& arms, const std::vector<ProbeRun>& runs, int spec_default) const;
    /// After a round: no other arm can still beat the default (each has a run no faster than one of the default's),
    /// so the rest of the plan need not run (the start's time).
    static bool settled(const std::vector<ProbeArm>& arms, const std::vector<ProbeRun>& runs);

private:
    double margin_, switch_;
};

/// XESTRATA_DRAFT_PROBE (auto | 0 | 1): false when it is 0
bool probe_enabled();
/// The tokens each run decodes (XESTRATA_DRAFT_PROBE_TOKENS, default 48; at least 16)
int probe_tokens();
/// The warm-up run's tokens (it is not counted: the first requests pay the kernels' first launches)
constexpr int kProbeWarmTokens = 16;
/// The tokens a counted run decodes on a model that the warm-up decoded at `warm_tps`: about a second of decode,
/// 16 at least and probe_tokens() at most (GLM-5.3-Flash at 10 tok/s on the B70: the measurement took 128 s at 48)
int probe_run_tokens(double warm_tps);
/// The measurement's time once every arm has run once: past it, no further round starts (XESTRATA_DRAFT_PROBE_S,
/// default 20 s)
double probe_budget_s();
/// The plan's line `line` ("GEN <n> ...") with its token count set to n
std::string probe_line_tokens(const std::string& line, int n);

/// The probe's prompt from a model's vocabulary (its GGUF tokenizer.ggml.tokens): a fixed English text, each word one
/// token where the vocabulary has it as one (the byte-level BPE's leading-space mark U+0120 or SentencePiece's
/// U+2581, else the bare word); a word that is not one token is left out.  Empty when fewer than 16 words were found
/// (a vocabulary of another kind): the engine then does not measure.
std::vector<int32_t> probe_prompt(const std::vector<std::string>& vocab);

}  // namespace strata::spec
