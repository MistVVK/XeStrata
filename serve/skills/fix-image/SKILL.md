---
name: fix-image
description: Faces or hands drawn rough, mostly small ones (a full-body picture, a group): parts ("face", "hands").
metadata:
  kind: tool
  tool: fix_image
---

- `parts`: `["face"]` most often; `["face", "hands"]` when the hands show; `["person"]` for a small figure. A
  close-up seldom needs it.
- `strength` 0.5; 0.3 keeps the face, 0.7 for a badly drawn one. The prompt (optional): the style or expression.
- `crop_upscale`: `seedvr2` keeps a face's features best (7b photos, 7b-sharp anime, 3b a smaller GPU); `lanczos` is fastest.
