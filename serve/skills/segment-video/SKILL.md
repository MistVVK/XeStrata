---
name: segment-video
description: Things in an attached video found, followed or masked ("track the car"): prompts, masks.
metadata:
  kind: tool
  tool: segment_video
---

- `prompts`: short English noun phrases, one thing each: `["person"]`, `["red car", "dog"]`; every instance is found
  on every frame. `masks` true for each object's own mask video.
- A job, as upscale_video's: tell the user, do not call again to check. To pick one object by pointing, or fix one
  missed: the Video tab's Find objects.
