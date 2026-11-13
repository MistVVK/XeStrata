// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// include/strata/core/latent_share.hpp - the session's latent D/U pair on the device: the driver uploads it once
// (generate.cpp's LatentTab) and every window and the prompt path pull it in their own init, so the several
// verifiers of a serve session share one copy.  A non-latent pack never stores a pair: the accessors stay null.
// Written after edge0-strata's latent_share.hpp.
#pragma once

#include <cstdint>

namespace strata::core {

inline const uint8_t*& latent_shared_d() { static const uint8_t* p = nullptr; return p; }
inline const uint8_t*& latent_shared_u() { static const uint8_t* p = nullptr; return p; }
inline int& latent_shared_fmt() { static int f = 8; return f; }

}  // namespace strata::core