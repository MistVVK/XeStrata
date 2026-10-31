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
// the switches and dumps of their kernels; and A/B switches of its engine's own parts (the shared expert's order, the
// canonical copies, the prompt path's borrowing and cut-off, the cache's per-layer and CPU-order experiments).
// --kv-resident streams the K/V of QSA, whose selection reads about 2,000 cells a query: the other models' dense
// attention reads every cell each token (GLM-5.3-Flash's DSA keeps its latent cells in RAM with --kv-ram instead).
// Its separate MTP pack's drafter (--mtp and its window and cap), its verifier's and token graph's switches and
// measurements (the per-stage counts, the 48 layer graphs, the stream-token path, the split verify window, the fetch
// branch's --pcie-mode, the pool's hit and publish experiments), and its prompt path's grouped products
// (--prefill-experts): the other models' engine has windows of its own, one graph a window size, one fetch over PCIe
const char* const kQwen38Only[] = {
    "--pack", "--ple-gguf", "--no-ple", "--ple-io", "--ple-row-cache", "--ple-inflight", "--ple-delay-us",
    "--ple-sync-submit", "--embd-gguf", "--native-dense-gguf", "--native-head-gguf", "--native-bf16",
    "--native-bf16-extra", "--native-ple-key", "--native-ple-postops", "--native-moe-combine", "--native-gdn",
    "--native-flash-attn-short", "--native-qsa", "--native-qsa-indexer", "--native-rope", "--native-router",
    "--gr-fp32-activations", "--gr-native-mmvf", "--no-fused-gr", "--no-fused-gdn", "--no-fast-attn",
    "--no-fast-select", "--mtp-hnorm", "--vision", "--shared-late", "--keep-canonical",
    "--cpu-oracle-q8-0", "--dump-final-r", "--dump-halves", "--dump-mixed", "--dump-residual", "--dump-layers",
    "--prefill", "--prefill-until", "--no-prefill-borrow", "--prefill-experts",
    "--expert-cache-per-layer", "--kv-resident",
    "--expert-cache-cpu-order",
    "--mtp", "--mtp-max-t", "--mtp-window", "--stream-token", "--check-logits", "--stats", "--stage-timing",
    "--spec-split", "--no-spec-split", "--spec-oracle", "--spec-corrupt", "--gpu-stages", "--gpu-only-full",
    "--logits-stride", "--window-hashes", "--graph-only", "--no-capture", "--no-token-graph", "--no-pool",
    "--no-hit-poke", "--no-publish-kernel", "--no-split-rows", "--sync-every-layer", "--pcie-mode",
    "--batch-cpu-split",
};

// the engine's options every model could take, which the other models' path does not have yet (2026-11-01)
const char* const kNotYet[] = {
    "--batch-groups",
    "--expert-cache-device1", "--expert-cache-device2", "--expert-cache-device3", "--expert-cache-remote-placement",
    "--pipeline-windows",
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
