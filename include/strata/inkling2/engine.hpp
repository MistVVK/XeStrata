// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/inkling2/engine.hpp - Inkling-2-Mini's engine inside xestrata-engine (src/inkling2/ink_main.cpp).
#pragma once

namespace strata::inkling2 {

/// Runs the engine when --native names a .strata file (Inkling-2-Mini's v1.1 format) and returns its exit code; -1,
/// nothing done, for any other model (the GGUF engines take it).
int inkling_main(int argc, char** argv);

}  // namespace strata::inkling2
