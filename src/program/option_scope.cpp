// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/option_scope.cpp - see include/strata/program/option_scope.hpp.  The lists follow generate.cpp's
// options: one added there goes into one of them (or into plain_generate.cpp, where the other models take it).
#include "strata/program/option_scope.hpp"

#include <algorithm>
#include <iterator>

namespace strata::program {
namespace {

// Qwen3.8-Flash-Next's pack, PLE table, embedding and dense images, QSA, GR streams, MTP head norm, vision tower, and
// the switches and dumps of their kernels
const char* const kQwen38Only[] = {
    "--pack", "--ple-gguf", "--no-ple", "--ple-io", "--ple-row-cache", "--ple-inflight", "--ple-delay-us",
    "--ple-sync-submit", "--embd-gguf", "--native-dense-gguf", "--native-head-gguf", "--native-bf16",
    "--native-bf16-extra", "--native-ple-key", "--native-ple-postops", "--native-moe-combine", "--native-gdn",
    "--native-flash-attn-short", "--native-qsa", "--native-qsa-indexer", "--native-rope", "--native-router",
    "--gr-fp32-activations", "--gr-native-mmvf", "--no-fused-gr", "--no-fused-gdn", "--no-fast-attn",
    "--no-fast-select", "--mtp-hnorm", "--vision", "--shared-late", "--shared-expert-arena", "--keep-canonical",
    "--cpu-oracle-q8-0", "--dump-final-r", "--dump-halves", "--dump-mixed", "--dump-residual", "--dump-layers",
    "--prefill", "--prefill-experts", "--prefill-until", "--no-prefill-borrow",
    "--resident-experts", "--resident-cpu-experts", "--resident-budget-gib", "--expert-cache-per-layer",
    "--expert-cache-cpu-order",
};

// the engine's options every model could take, which the other models' path does not have yet
const char* const kNotYet[] = {
    "--slots", "--batch", "--batch-cpu-split", "--batch-groups", "--batch-mtp", "--no-batch-mtp",
    "--stream-token", "--check-logits", "--prompt-cache", "--prompt-cache-every", "--prompt-cache-root",
    "--prompt-cache-tail", "--conversation-save-checkpoints", "--conversation-save-compress",
    "--kv-grow", "--no-kv-grow", "--kv-resident", "--vram-reserve-later-mib", "--adapt-async", "--pcie-frac", "--pcie-mode", "--pool-tasks", "--stats", "--stage-timing",
    "--spec-split", "--no-spec-split", "--spec-oracle", "--spec-corrupt", "--suffix-draft",
    "--lookup-chain", "--lookup-chain-min", "--coupled-draft", "--no-coupled-draft", "--mtp", "--mtp-max-t",
    "--mtp-window", "--gpu-stages", "--expert-cache-device1",
    "--expert-cache-device2", "--expert-cache-device3", "--expert-cache-remote-placement", "--gpu-only-full",
    "--control-vector", "--control-vector-layer-range", "--control-vector-scaled", "--cvec-dir", "--cvec-mode",
    "--logits-stride", "--window-hashes", "--graph-only", "--no-capture", "--no-token-graph",
    "--no-pool", "--no-host-worker", "--no-hit-poke", "--no-publish-kernel", "--no-split-rows", "--pipeline-windows",
    "--sync-every-layer", "--short-read", "--yarn-attn-factor", "--yarn-beta-fast", "--yarn-beta-slow", "--yarn-ext-factor", "--turn-token", "--tail-role-token",
    "--aux-cpus", "--pool-cores", "--memory-limit-mib", "--conversation-cache-min-tokens",
    "--conversation-cache-park-gain", "--session-save-reclaim", "--prompt-cache-spread",
};

bool listed(const char* const* first, const char* const* last, const std::string& name) {
    return std::any_of(first, last, [&](const char* o) { return name == o; });
}

}  // namespace

OptionScope option_scope(const std::string& name) {
    if (listed(std::begin(kQwen38Only), std::end(kQwen38Only), name)) return OptionScope::qwen38_only;
    if (listed(std::begin(kNotYet), std::end(kNotYet), name)) return OptionScope::not_yet;
    return OptionScope::unknown;
}

}  // namespace strata::program
