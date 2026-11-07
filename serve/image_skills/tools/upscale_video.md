<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# upscale_video

When: the user attached a video and wants it larger or sharper: "upscale", "in 4K", "higher resolution", "restore".

- It starts a job on the server and answers at once: tell the user the job runs, that they follow it in the Video tab,
  and where the video will be. Do not call it again to see whether it is done.
- `scale` 2 by default; 1.5 for a little larger. A large video at 2 can take a long time: say so when it is long.
- `method`: leave it `auto` (SeedVR2, which restores detail; RealESRGAN where SeedVR2 does not fit). `lanczos` only
  when the user asks for a plain resize.
- SeedVR2's `variant`: `7b` (the default) for live action; `7b-sharp` for anime or a blurred video.
- The sound is kept. The video is not drawn anew: nothing in it changes but its size and detail.
