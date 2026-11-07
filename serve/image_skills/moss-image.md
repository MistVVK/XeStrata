<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# MOSS-Image

When: a MOSS-Image image: boxes (posters, signs, panels), an edit with a painted part or a named edit LoRA.

- Give generate_image a `layout` only when the user says where things or words go: a poster, a sign, several panels.
- With a layout, the prompt is a short description of the whole image.
- One item for each thing or text the user places; a text item has its exact words in `text`.
- `character` for a person or a character, `object` for a thing, `text` for words.
- A box marks where a thing is, not a frame to draw.
- Give a text item a box large enough to be read; `\n` breaks its line.
- Several panels: `summary.canvas` `multi_panel`; pictures put together: `collage`.
- edit_image with a painted part and more images than the one edited: first without `refine`. Only when the painted
  part comes out poor (blurred, broken, or without what the other images were to give it), edit the same image again
  (`image` the one edited before, not the result) with `refine` true.
- An edit LoRA the user names ("use armv6", "with the X LoRA"): edit_image's `edit_adapter`, the name as it is listed
  there. Never give one the user did not name. When the result says it cannot be used, tell the user why.
- moss-image-4-step draws in 4 steps without guidance: no negative prompt, no edits; the prompt as for MOSS-Image.
- control with `color` (where each thing is) or `depth` places people and things as the picture has them; describe
  each in the prompt. A map from a 3D mannequin or a render: `end` 0.5 (else its surface shows through). With
  moss-image-4-step keep `end` short (0.3); where the mannequin's surface still shows, use MOSS-Image's usual steps.
