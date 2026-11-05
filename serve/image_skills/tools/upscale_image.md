<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# upscale_image

When: the user wants an image larger or finer: a wallpaper, a print, "in high resolution", "more detail".

- After the image is made as the user wants it, not instead of making it.
- `scale` 2 for a wallpaper or a print; 1.5 for a little larger.
- `redraw` (default true) draws the larger image again, adding detail; its `prompt` is the image's own (what it
  shows), not a request to enlarge it.
- "Larger, finer", "add detail", "high resolution": redraw. `strength` 0.35 by default; 0.25 keeps it closer to the
  original, 0.5 adds more detail and changes more.
- "Just larger", "keep it exactly as it is", a photo or a picture someone else made: `redraw` false (no prompt
  needed): nothing in it changes but its size.
- `method`: leave it `auto` (RealESRGAN, a sharper upscale than a plain resize). `lanczos` only when the user asks
  for a plain resize.
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
- To change what is in the picture, edit_image; to show more around it, extend_image.
