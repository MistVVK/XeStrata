<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# upscale_image

When: the user wants an image larger or finer: a wallpaper, a print, "in high resolution", "more detail".

- After the image is made as the user wants it, not instead of making it.
- `scale` 2 for a wallpaper or a print; 1.5 for a little larger.
- Without `redraw` (the default): the same picture, larger, its detail restored (SeedVR2) but nothing in it drawn
  anew; no prompt needed. "Just larger", "keep it exactly as it is", a photo or a picture someone else made.
- `redraw` true draws the larger image again, adding detail; its `prompt` is the image's own (what it shows), not a
  request to enlarge it. "Larger, finer", "add detail", "more detailed": redraw. `strength` 0.35 by default; 0.25
  keeps it closer to the original, 0.5 adds more detail and changes more.
- `method`: leave it `auto`: RealESRGAN where it is drawn again, SeedVR2 where it is not (it restores detail in one
  step, photos and illustrations alike). `lanczos` only when the user asks for a plain resize.
- SeedVR2's `variant`: `7b` (the default) for photos and video frames; `7b-sharp` for anime, blurred pictures, or
  when the user wants crisper detail.
- A result that says it is downloading: tell the user, and call it again later, in a new message, not now.
- To change what is in the picture, edit_image; to show more around it, extend_image.
