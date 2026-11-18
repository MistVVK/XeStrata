---
name: upscale-video
description: An attached video larger or sharper ("upscale", "4K"): scale.
metadata:
  kind: tool
  tool: upscale_video
---

- A job on the server that answers at once: tell the user it runs, that they follow it in the Video tab and where the
  video will be; do not call again to check.
- `scale` 2 (1.5 a little); a long video at 2 takes long: say so. `method` auto; `lanczos` only for a plain resize.
  `variant` 7b for live action, 7b-sharp for anime or blur, 3b for a smaller GPU. The sound is kept; nothing is drawn anew.
