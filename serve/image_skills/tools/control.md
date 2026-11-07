<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# control

When: the user gives a picture and wants a new image in its pose, its composition or its layout ("in this pose",
"with this composition", "draw this sketch properly", "the same room in another style").

- `type` by what is kept: `pose` for people's poses (a photo or a figure; nothing else of the picture is kept);
  `depth` for the composition and what is near or far; `canny` for every outline (a close copy of the shapes);
  `lineart` for a drawing's lines (`lineart_coarse` for rough ones); `hed` for soft outlines; `scribble` for a sketch
  or a doodle; `mlsd` for a room's or a building's straight lines; `gray` to keep the shading and recolor.
- The prompt describes the whole new image, as for generate_image: what it shows and its style. Nothing about the
  picture or the control.
- `strength` (with ControlNet, Qwen-Image-2.1): 1 by default; 0.5 to 0.8 lets the image depart from the picture more.
  `end` below 1 (0.6 to 0.8) keeps the composition and frees the details.
- MOSS-Image: `color` (where each thing is) and `depth` go through its color and depth adapter (`strength` the
  adapter's, `end` the share of the steps; no `start`). Its other kinds go in as a reference image.
- Other image models take the map as a reference image: strength and end do not apply there.
- To change part of the same picture, edit_image or edit_region instead; for the same picture larger, upscale_image.
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
