---
name: edit-region
description: One part changed or removed, the rest kept ("her hair blue", "a smile", "remove the boat"): region and prompt.
metadata:
  kind: tool
  tool: edit_region
---

- `region`: the part in short English words, as a label (`hair`, `sky`, `the left cup`); the prompt: what it
  becomes, in the model's prompt style ("long blue hair").
- Every instance is found; for one, say which: `the man's hair`, `the girl on the left`.
- `strength` 1 redraws; 0.6 keeps the shape.
- `crop_upscale`, how the cut-out is enlarged to be drawn: `seedvr2` for a small face, hand or thing whose shape
  must stay (`variant` 7b for photos, 7b-sharp for anime or blur, 3b on a smaller GPU); `lanczos` for a large part or for speed; else auto.
- `mask` instead of `region`: a mask find_region gave. Nothing found: find_region with other words.
