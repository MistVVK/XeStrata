---
name: guidance
description: Sampling guidance - finer detail asked for, or colors oversaturated or harsh.
metadata:
  kind: tool
  tool: guidance
---

- Most images need none: leave `guidance` out otherwise.
- detail_daemon: finer texture and small detail. skip_layer_guidance: crisper structure, some more time.
- apg: oversaturated or harsh colors; cfg_norm: the same, milder. cfg_zero_star: shapes that go wrong early; rare.
- Not a fix for a wrong subject or a broken face: change the prompt, or edit_image.
