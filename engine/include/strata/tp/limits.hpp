#pragma once
#include <cstddef>

namespace strata::tp {
// Images use chunked broadcasts; other engine commands fit in one frame.
inline constexpr size_t max_prompt_bytes = 8 * 1024 * 1024;
}
