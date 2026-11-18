---
name: control
description: A new image in a picture's pose, composition or layout ("in this pose", "this sketch drawn properly").
metadata:
  kind: tool
  tool: control
---

- `type`: `pose` (people's pose, nothing else kept), `depth` (composition), `canny` (every outline), `lineart` (a
  drawing's lines; `lineart_coarse` rough ones), `hed` (soft outlines), `scribble` (a sketch), `mlsd` (a room's
  straight lines), `normal` (surface orientation), `segmentation` (regions of objects and background),
  `gray` (shading), `none` (the image is a map already, one control returned).
- With `normal` or `segmentation`, describe the composition in the prompt: full body or close-up, and where people are.
- For Qwen-Image-2.1 lineart from a hand drawing, set `lineart_input: "drawing"`; for a photo leave it unset.
  Thin lines and hatching can otherwise be copied rather than followed. `none` passes a map without processing.
- The prompt describes the whole new image, nothing about the control. A person who must stay (their face, their
  clothes) in another pose: edit_image with references instead.
- ControlNet (Qwen-Image-2.1): `strength` (0 to 2) 1, 0.5 to 0.8 departs more; `end` (the share of the steps it
  follows) 1, 0.6 to 0.8 frees the details. For depth when the new subject differs from the map, use `strength` 0.6
  and `end` 1. `size` the picture's shape by default.
- MOSS-Image: `color` and `depth` by its adapter (`end` the share of the steps: 0.5 from a 3D mannequin).
  Its control LoRAs take pose, outlines, normals and segmentation. Other models take the map as a reference image.
