// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/w8_choice.cpp - see include/strata/image/w8_choice.hpp.

#include "strata/image/w8_choice.hpp"

#include "dit_rows.hpp"

namespace strata::image {

int w8_here(void* stream) {
    return dit_rows::w8_at_open(3072, 12288, "the image DiTs (3072 -> 12288, SeedVR2's 7B's)", stream);
}

}  // namespace strata::image
