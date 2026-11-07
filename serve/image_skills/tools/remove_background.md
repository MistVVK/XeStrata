<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# remove_background

When: the user wants the subject cut out (a transparent PNG, a sticker, a product shot), or put on a plain color.

- `version` (BiRefNet's): `general` by default; `portrait` for a person; `matting` for hair, fur, glass, smoke and
  other soft or see-through edges; `hr` or `dynamic` for a picture larger than 2K; `lite` (or `lite-2k` for a large
  picture) only when the GPU has little memory.
- `background_color` for a plain background (`#ffffff` white); left out, the background is transparent.
- To put the subject on a new drawn background, edit_image instead (the image model draws the scene).
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
