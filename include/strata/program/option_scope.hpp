// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/option_scope.hpp - which of the engine's options a model other than Qwen3.8-Flash-Next takes.
//
// `strata` runs Qwen3.8-Flash-Next (qwen4exp) through generate.cpp and the models of plain_model.hpp (qwen35moe,
// qwen3moe) through plain_generate.cpp, which takes every option that does not rest on Qwen3.8-Flash-Next's own
// structure (its pack, PLE, QSA, GR streams, vision tower, and the switches of their kernels); one of those is refused
// with that reason.
#pragma once

#include <string>

namespace strata::program {

enum class OptionScope {
    unknown,       ///< not an option of the engine
    qwen38_only,   ///< Qwen3.8-Flash-Next's own structure: no other model has it
};

/// The scope of `name` (with its leading "--") for a model the plain path runs.  Options the plain path takes are
/// not asked about.
OptionScope option_scope(const std::string& name);

}  // namespace strata::program
