---
name: control
description: A new image in a picture's pose, sketch, composition or depth ("in this pose", "this sketch drawn properly", "the same layout as a Japanese room"): type, image, prompt.
metadata:
  kind: tool
  tool: control
---

Choose `control` when the user asks to follow a map or a picture's pose or layout in a new image, not `edit_image`
without a control. Choose the `type` the user asks for:

| User's words | `type` |
| --- | --- |
| Rough sketch, scribble or doodle | `scribble` |
| A finished line drawing to color | `lineart` |
| A straight-line or architectural line map | `mlsd` |
| Object or background regions, segmentation | `segmentation` |
| A color-coded map with a description for each color | `color` |
| Pose / depth or composition / outlines / surface orientation | `pose` / `depth` / `canny` / `normal` |

- Select a model that takes the requested control: Qwen-Image-2.1 uses ControlNet, MOSS-Image uses its color and depth
  adapter (and has other control LoRAs where offered). A map given as a reference is an approximation, not ControlNet.
  Set `model` to the compatible model when it is offered; tell the user if switching from the default. If no model
  takes the requested control directly, say what is available rather than claiming it was applied.
- `type: "color"` with `regions` stays `color` even when the map is already made: set `preprocess: false`, not
  `type: "none"`. Use the exact `#rrggbb` colors from the map.
- `hed` is soft outlines, `lineart_coarse` rough outlines, `gray` shading; `none` passes a map without regions as is.
- With `normal` or `segmentation`, describe the composition in the prompt: full body or close-up, and where people are.
- Qwen-Image-2.1 has no normal or segmentation ControlNet. Do not silently replace a requested `normal` or
  `segmentation` with `depth`: name a model that takes it, or explain the available reference-map alternative.
  MOSS-Image's M-LSD and gray maps are references; choose a model with direct control when one is offered.
- For Qwen-Image-2.1 lineart from a hand drawing, set `lineart_input: "drawing"`; for a photo leave it unset.
  Thin lines and hatching can otherwise be copied rather than followed. `none` passes a map without processing.
- The prompt describes the whole new image, nothing about the control. A person who must stay (their face, their
  clothes) in another pose: Qwen-Image-2.1 edit_image with `control: {"type":"pose", "image":<pose picture>}`.
  Keep the pose picture out of references; the edit's pose defaults are strength 0.7 and end 0.8.
- ControlNet (Qwen-Image-2.1): `strength` (0 to 2) 1, 0.5 to 0.8 departs more; `end` (the share of the steps it
  follows) 1, 0.6 to 0.8 frees the details. For depth when the new subject differs from the map, use `strength` 0.6
  and `end` 1. `size` the picture's shape by default.
- MOSS-Image: `color` and `depth` by its adapter (`end` the share of the steps: 0.5 from a 3D mannequin).
  For `color`, `regions` is `[{"color":"#rrggbb","prompt":"what to draw"}, ...]`; the color must exactly match the
  map. With a ready map set `preprocess: false`. The server turns the colors into positions, not color words. It rejects
  regions with centers too close to distinguish. For people, use a silhouette map (e.g. segmentation from a photo):
  a mannequin's joints can be copied. Overlapping shapes can make extra heads. Its control LoRAs take pose, outlines,
  normals and segmentation. Other models take the map as a reference image.
