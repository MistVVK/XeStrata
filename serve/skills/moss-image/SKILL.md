---
name: moss-image
description: A MOSS-Image image placing things or words (posters, signs, panels).
metadata:
  kind: model
  for: moss-image
---

- Give generate_image a `layout` only when the user says where things or words go; the prompt is then a short
  description of the whole image.
- One item for each thing or text placed: `character` (a person), `object`, `text` (its exact words in `text`);
  `caption` how it looks. `bbox` [x_min, y_min, x_max, y_max] in the image's pixels from its top-left corner: where it
  is, not a frame to draw; a text box large enough to read; `\n` breaks its line.
- `summary.caption` the whole image; `summary.canvas` `multi_panel` for panels, `collage` for pictures put together.
- moss-image-sft-4-step: use for fast local edits as well as generation, without guidance or a negative prompt.
  Use 4 steps up to 1024², 6 above that, or 8 above 1536². For a whole-image style change, use the regular model.
- To combine two or more reference images, tell the user and switch to Qwen-Image-2.1 for the edit,
  even when MOSS-Image is selected: chinoll can overlay the references rather than combine them.
