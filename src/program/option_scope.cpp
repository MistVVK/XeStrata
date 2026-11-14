// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/option_scope.cpp - see include/strata/program/option_scope.hpp.  The list follows generate.cpp's
// options: one added there goes into plain_generate.cpp too, or into it when it rests on Qwen3.8-Flash-Next's own
// structure.
#include "strata/program/option_scope.hpp"

#include <algorithm>
#include <iterator>

namespace strata::program {
namespace {

// Qwen3.8-Flash-Next's pack, PLE table, embedding and dense images, QSA, GR streams, MTP head norm, vision tower, and
// the switches and dumps of their kernels; and A/B switches of its engine's own parts (the shared expert's order, the
// canonical copies, the prompt path's borrowing and cut-off, the cache's per-layer and CPU-order experiments).
// (--kv-resident is shared now: the plain engine streams its dense attention's K/V too, the last N cells a layer in
// VRAM and the rest read from RAM; GLM-5.3-Flash's DSA keeps --kv-ram.)
// Its separate MTP pack's drafter (--mtp and its window and cap), its verifier's and token graph's switches and
// measurements (the per-stage counts, the 48 layer graphs, the stream-token path, the split verify window, the fetch
// branch's --pcie-mode, the pool's hit and publish experiments), and its prompt path's grouped products
// (--prefill-experts): the other models' engine has windows of its own, one graph a window size, one fetch over PCIe
// --draft-fallback and --draft-mtp-spec rest on generate.cpp running DFlash2 and the MTP layer together (a static
// primary with the other as fallback, its own window): the plain engine keeps both loaded but picks one per request
// by its measured speed (--draft auto, strata/spec/drafter_policy.hpp) and sets the MTP's window with --spec, so the
// static knobs add nothing there.  --spec-min-p and --window-hashes are shared (shared_options.hpp).
const char* const kQwen38Only[] = {
    "--pack", "--ple-gguf", "--no-ple", "--ple-io", "--ple-row-cache", "--ple-inflight", "--ple-delay-us",
    "--ple-sync-submit", "--embd-gguf", "--native-dense-gguf", "--native-head-gguf", "--native-flash-attn-short",
    "--no-fused-gr", "--no-fused-gdn", "--no-fast-attn", "--mtp-hnorm", "--vision", "--shared-late", "--keep-canonical",
    "--cpu-oracle-q8-0", "--dump-final-r", "--dump-halves", "--dump-mixed", "--dump-residual", "--dump-layers",
    "--prefill", "--prefill-until", "--no-prefill-borrow", "--prefill-experts",
    "--expert-cache-per-layer",
    "--expert-cache-cpu-order",
    "--mtp", "--mtp-max-t", "--mtp-window", "--draft-fallback", "--draft-mtp-spec",
    "--stream-token", "--check-logits", "--stats", "--stage-timing",
    "--spec-split", "--no-spec-split", "--spec-oracle", "--spec-corrupt", "--gpu-stages", "--gpu-only-full",
    "--logits-stride", "--graph-only", "--no-capture", "--no-token-graph", "--no-pool",
    "--no-hit-poke", "--no-publish-kernel", "--no-split-rows", "--sync-every-layer", "--pcie-mode",
    "--batch-cpu-split",
};

bool listed(const char* const* first, const char* const* last, const std::string& name) {
    return std::any_of(first, last, [&](const char* o) { return name == o; });
}

}  // namespace

OptionScope option_scope(const std::string& name) {
    if (listed(std::begin(kQwen38Only), std::end(kQwen38Only), name)) return OptionScope::qwen38_only;
    return OptionScope::unknown;
}

}  // namespace strata::program
