---
name: edit-image
description: A change to an image or a masked part; taking a face, pose or clothes from other pictures.
metadata:
  kind: tool
  tool: edit_image
---

- The prompt says the change ("her coat red"); a model without edits (img2img) needs the whole picture described.
- `references`: other pictures' links, images 2, 3, ... after the edited image 1; name them in the prompt ("the face
  from image 2"). Without them the user's other attached pictures go in, except the control image.
- `mask`: a mask's link (white changes): find_region's, or the one the user painted (set by itself). A part from
  another picture (a face, clothes, shoes; image-tools' list): find_region's mask and the picture in
  `references`, not fix_image.
- `control` keeps the person and takes another picture's pose: Qwen-Image-2.1, `{"type": "pose", "image":
  "attached:2"}`, never that picture in `references`. Describe the pose and say to keep the face, hair and clothes.
  Its pose defaults to strength 0.7 and end 0.8.
- `strength` 1 redraws (the default; img2img's 0.87); 0.5 to 0.7 keeps the shapes.
- Two or more reference images: tell the user and use Qwen-Image-2.1, even when MOSS-Image is selected.
- MOSS-Image with a mask and references: first without `refine`; `refine` true only when the part comes out poor
  (on the image edited before). `edit_adapter` only when the user names one.
- `crop_upscale`: leave it out (auto) unless the user asks; its values as edit_region's.
