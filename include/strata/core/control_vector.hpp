// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/control_vector.hpp - the control vectors' one loader, which the two engines had a copy of
// each: llama.cpp's common_control_vector_load (every file's direction.<l> times its scale, summed; layer 0 has
// none) and llama_adapter_cvec::apply with the projection-mode patch (project: the unit direction and its norm as
// the scale), into the tables kernels::cvec_upload takes, and uploaded.  The caller keeps its own session digest
// (the two engines' differ) and its own log line.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct ControlVectorLoad {
    std::vector<float> dir, scale;   // [layers][n_embd] and [layers], as cvec_upload takes them
    int first = 1, last = 0, single = -1, steered = 0;
    std::string summary;             // what INFO reports: mode:first-last[:singleL]
};

/// `files`: (path, scale) pairs.  `hint_expected`: the architecture the files should name (a mismatch is a warning,
/// not an error); `who`: the engine's name in that warning; `hint_named`: the warning also names the expected
/// architecture (generate's form).  `hc`: cvec_upload's head-count argument.  `mode`: 0 project, 1 add.
bool load_control_vectors(const std::vector<std::pair<std::string, float>>& files, int64_t L, int64_t N, int64_t hc,
                          int first, int last, int mode, int single, const std::string& hint_expected,
                          const std::string& who, bool hint_named, ControlVectorLoad& out, std::string& err);

}  // namespace strata::core
