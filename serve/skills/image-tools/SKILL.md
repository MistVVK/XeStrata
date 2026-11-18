---
name: image-tools
description: The image and video tools' rules, which always go with them.
metadata:
  kind: always
---
Plan first: two image or video steps or more get a chart of them as the first call's `plan` (combine-tools has more):

```mermaid
flowchart LR
  in[photo] --> s1["find the hat (find_region)"] --> s2["make it red (edit_image)"] --> out[result]
```

Choose the tool by the request, the first that fits:

1. One part changed by words ("her hair blue", "a smile", "a hat on the man on the right", "only the sky"):
   edit_region `{"image": "attached:1", "region": "hair", "prompt": "long blue hair"}`.
1. A thing removed, the words on a sign changed, or one person of several kept alone: edit_image on the whole
   picture, no mask, with Qwen-Image-2.1 where this setup has it (other models leave the thing there):
   `{"image": "attached:1", "model": "<this setup's qwen-image-2.1 model>", "prompt": "Remove the red boat and its
   reflection."}`; "Change the sign's text to \"CLOSED\".", "Keep only the woman in the middle, on a plain white
   background."
1. A thing added, held or worn (a bag in her hand, a swan on the lake), by words or from another picture: edit_image
   on the whole picture, no mask, the other picture in `references`: "The girl in image 1 holds the yellow bag from
   image 2 in her right hand. Keep everything else."
1. A part replaced by one from another picture (a face, hair, clothes, shoes): two calls, in two replies.
   1. find_region on the picture that changes, `{"image": "attached:1", "region": "face"}` (`hair`, `the shirt`,
      `feet`; one person of several: `the woman in the middle's face`). Clothes that cover more (a long dress for
      shorts): the body they will cover (`the girl's body below the neck`).
   1. edit_image `{"image": "attached:1", "mask": "<the mask link find_region gave>", "references": ["attached:2"],
      "prompt": "the face from image 2"}`; say what stays ("keep her black hair").
1. The same person in another picture's pose: edit_image `{"image": "attached:1", "control": {"type": "pose",
   "image": "attached:2"}}`, the pose in the prompt; not `references`.
1. A new image that follows a picture: control `{"image": "attached:1", "type": "...", "prompt": "the whole new image"}`.
   A sketch drawn properly: `scribble`. The same layout or depth with other things ("the same room, as a Japanese
   room", "this view in snow"): `depth`. Its outlines colored: `lineart`. A pose for a new person: `pose`.
1. The whole picture changed (a style, an age, the background): edit_image, the change as its prompt.
1. A size past 2048 a side (a 2670x1200 wallpaper, a print): two calls, in two replies.
   1. The picture at the same shape within 2048 a side, each side a multiple of 16: generate_image or edit_image with
      `"size": "2048x912"` for 2670x1200 (edit_image redraws an attached picture to it; extend_image keeps the
      picture and adds around it).
   1. upscale_image `{"image": "<its link>", "size": "2670x1200"}`.
1. Larger or finer: upscale_image. Wider or another shape: extend_image. A cut-out or a plain background:
   remove_background. Rough faces or hands: fix_image. Where something is: find_region.

For a named LoRA or control, set `model` to one that takes it; tell the user if it differs from the default.

Prompts:

- Short, from the user's words, in the model's prompt style. Add nothing they did not ask for (subjects, setting,
  style, light, quality tags) unless a skill says so, and drop nothing they asked for.
- Words the image shows go in double quotes, exactly as given.

Calls:

- The user's pictures are `attached:1`, `attached:2`, ... in attachment order; a tool's result gives its files' links
  (/v1/images/files/..., masks too). Pass only these, copied exactly. Never invent a link, in a call or in the
  answer.
- A tool given as one line: call it with the arguments its line names; its first call answers with the rest.
- The calls of one reply run at once: put together the steps that need no other step's result.
- Look at each result. Call again only when it came out wrong, with another prompt or region, at most twice; then
  tell the user what is wrong. Not the same call with another strength only.
- Do the steps the user asked for, and no finishing they did not ask for (no upscale, border or fix of your own).
- Ask first only when the user is consulting you, or the work is long (a long video).
- A result that says it is downloading: tell the user, and call it again later, in a new message.
