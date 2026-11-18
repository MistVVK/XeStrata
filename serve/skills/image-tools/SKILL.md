---
name: image-tools
description: The image and video tools' rules, which always go with them.
metadata:
  kind: always
---
Plan first: a request that needs two image or video steps or more gets a chart of them, as the first call's `plan`
(combine-tools has more):

```mermaid
flowchart LR
  in[photo] --> s1["find the hat (find_region)"] --> s2["make it red (edit_image)"] --> out[result]
```

Each step's own tool: one part (edit_region), larger (upscale_image), more around it (extend_image), a cut-out
(remove_background), faces or hands (fix_image), a picture's pose or layout for a new image (control). To keep a
person while changing to another picture's pose, edit_image with that picture as `control.image`, not a reference.
An explicit request for a map or control needs `control` (or `edit_image.control` for an edit), not an edit alone.
For a named LoRA or control, set `model` to one that supports it; tell the user if it differs from the default.
`edit_region` cannot combine its cropped part with a control map. Do not claim that combination was applied.
A tool given as
one line answers its first call with its arguments.

Prompts:

- Write the prompt briefly, from the user's words. Add nothing they did not ask for (subjects, setting, style, light,
  quality tags) unless a skill says so, and drop nothing they asked for.
- Words the image shows go in double quotes, exactly as given.

The tools:

- The user's pictures are `attached:1`, `attached:2`, and so on, in attachment order. Use these names or links a tool
  returned; never invent an external URL. Each result gives its files' links (/v1/images/files/..., masks and pose maps
  too): pass one by its link to the next tool's image, mask or references.
- Ask before going on only when the user is consulting you, or the work is long (a long video).
- The calls of one reply run at once: put together the steps that need no other step's result.
- Do the steps the user asked for, and no finishing they did not ask for (no upscale, border or fix of your own).
- Look at each result before the next step; call a tool again only when its result came out wrong.
- A result that says it is downloading: tell the user, and call it again later, in a new message.
