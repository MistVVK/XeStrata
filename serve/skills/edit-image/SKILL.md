---
name: edit-image
description: A change to an image or a masked part; taking a face, pose or clothes from other pictures.
metadata:
  kind: tool
  tool: edit_image
---

- The prompt says the change ("her coat red"); a model without edits (img2img) needs the whole picture described.
- `references`: other pictures' links, images 2, 3, ... after the edited image 1; name them in the prompt ("the pose
  of image 2", "the face from image 3"). Without them the user's attached pictures go in.
- `mask`: a mask's link (white changes): find_region's, or the one the user painted (set by itself).
- `strength` 1 redraws (the default; img2img's 0.87); 0.5 to 0.7 keeps the shapes. A face made into another person's: that person's picture as a
  reference, not fix_image.
- MOSS-Image with a mask and references: first without `refine`; `refine` true only when the part comes out poor
  (on the image edited before). `edit_adapter` only when the user names one.
- `crop_upscale`: as edit_region's.
