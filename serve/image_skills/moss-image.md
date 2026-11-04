<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# MOSS-Image

When: a MOSS-Image image: a layout of boxes (posters, signs, panels) or an edit with a painted part.

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
