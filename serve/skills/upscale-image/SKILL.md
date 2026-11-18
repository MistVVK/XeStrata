---
name: upscale-image
description: An image larger or finer (a wallpaper, a print, "more detail"): scale, redraw with prompt.
metadata:
  kind: tool
  tool: upscale_image
---

- After the image is as the user wants it. `scale` 2 (the default) for a wallpaper or a print, 1.5 a little larger.
- Without `redraw`: the same picture larger, its detail restored, nothing drawn anew ("just larger", a photo,
  someone else's picture). `redraw` true draws it again finer; its prompt is what the image shows; `strength` 0.35
  (0.25 closer, 0.5 more detail).
- `size` `"WxH"` instead of `scale`: that exact size, of the image's shape (a wallpaper's pixels); not with
  `redraw`. Another shape: image-tools' list.
- `method` auto; `lanczos` only for a plain resize. `variant` 7b for photos, 7b-sharp for anime, blur or crisper
  detail, 3b for a smaller GPU.
