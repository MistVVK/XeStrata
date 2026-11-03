<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# MOSS-Image

- Give generate_image a `layout` only when the user says where things or words go: a poster, a sign, several panels.
- With a layout, the prompt is a short description of the whole image.
- One item for each thing or text the user places; a text item has its exact words in `text`.
- `character` for a person or a character, `object` for a thing, `text` for words.
- A box marks where a thing is, not a frame to draw.
- Give a text item a box large enough to be read; `\n` breaks its line.
- Several panels: `summary.canvas` `multi_panel`; pictures put together: `collage`.
