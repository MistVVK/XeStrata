---
name: combine-tools
description: Two image or video tools or more - the plan's chart, calls run at once, recipes.
metadata:
  kind: tool
  tool: "*"
---
Plan: the first call's `plan`, a mermaid chart (the reply shows it):

```mermaid
flowchart LR
  in[attached photo] --> s1["find the hair (find_region)"] --> s2["make it blue (edit_image)"] --> out[result]
```

- `flowchart LR`; a box per step in the user's words, "what it does (tool)"; the inputs and the result boxes too;
  steps that need not wait for each other as branches; 10 steps at most.
- Ids s1, s2, ... on the tools' boxes; each call gives its box's id as `step`, and the page colors the box as it runs.
  Draw the chart again only when the plan changes.
- The calls of one reply run at once. Calls whose results are joined later share a `group`: one failing stops the
  rest of it. `use_text_gpu` true lends them the chat model's GPU: only when the user wants it fast, or two heavy
  steps of different image models stand side by side.

Recipes:

- A pose in words: edit_image. Another picture's pose: edit_image with that picture in `references` ("the person
  in image 1 in the pose of image 2"). A new image in an attached picture's pose: control `pose`.
- One part (hair, clothes, sky): edit_region; or find_region, then edit_image with its mask.
- Another background: edit_image ("on a beach"); a cut-out: remove_background.
- A full-body picture: generate_image, fix_image, then upscale_image.
- A large final image with a part changed: upscale_image first, then edit_region.
- Following things in a video: segment_video.
