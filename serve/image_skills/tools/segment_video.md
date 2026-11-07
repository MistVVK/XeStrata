<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# segment_video

When: the user attached a video and wants things in it found, followed, cut out or masked: "track the car", "mask the
people", "where is the dog in this clip".

- `prompts`: short English noun phrases, one thing each, as SAM 3.1 reads them: `["person"]`, `["red car", "dog"]`.
  Every instance of each is found on every frame; a phrase that describes one of several ("the man on the left") finds
  them all. Translate the user's words into English.
- `masks` true when the user wants each object's own black-and-white mask video (for editing elsewhere); the video
  with the objects colored over it and one mask video of them all are always made.
- It starts a job on the server and answers at once: tell the user the job runs, that they follow it in the Video tab,
  and where the results will be. Do not call it again to see whether it is done.
- To pick one object by pointing, or to fix one that was missed or mixed up, send the user to the Video tab's Find
  objects: points and boxes on a paused frame.
