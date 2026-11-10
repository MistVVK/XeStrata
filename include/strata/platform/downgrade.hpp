// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/platform/downgrade.hpp - the slower paths the engines fall back to: a library or module the build
// and the GPU promise (oneMath's products, oneDNN's or aotriton's attention, cuDNN's, comfy-kitchen's int8 product)
// that could not be taken where it was meant to run, so the engine's own path ran instead.  Each key is said once on
// stderr ("xestrata: downgrade KEY: why") and counted; the text engines hand the new keys to the server as INFO lines
// (`INFO downgrade=KEY`), which reports them (/v1/status) so a slow fallback is not silent (arcint's idea of
// counting its degraded paths, https://github.com/marfrit/arcint, Apache-2.0; written anew).  A choice made by
// measuring (the faster of two forms) or by what the GPU reports it lacks is not a downgrade.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::platform {

/// Notes that the path `key` (a short dotted name: "gemm.onemath", "attn.aotriton") fell back, `why` the reason; the
/// first time for a key it is said on stderr.  Thread-safe.
void downgrade(const char* key, const std::string& why);
/// The keys noted since the last call, in the order first noted (each once).
std::vector<std::string> downgrades_new();
/// The new keys (downgrades_new) as INFO lines on stdout (`INFO downgrade=KEY`), flushed: a text or image engine says
/// them from the thread that writes its answers, before READY and after each request, where no answer is half written.
void downgrades_say();
/// Every key noted and how often.
std::vector<std::pair<std::string, int64_t>> downgrades_all();

}  // namespace strata::platform
