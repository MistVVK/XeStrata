---
name: lora
description: Choosing LoRAs for generate_image, edit_image and control - which, how strong, and their block weights.
metadata:
  kind: tool
  tool: lora
---

- Use a LoRA only when the user names it, or asks for the character or the style it adds (its name, kind and trigger
  in `lora` say which). None otherwise.
- Write its trigger words in the prompt.
- A character LoRA (kind `character`, or a person's or creature's name): `scale` 1, and `blocks` `"character"` where
  `blocks` lists it (else leave `blocks` out). A lower strength takes the character away before the LoRA's style:
  keep 1. Write the style the picture should have, in the model's prompt style (the user's, else the model's own:
  `watercolor`, `photo`, `anime coloring`), so the LoRA brings the character and not its pictures' style.
- A style LoRA (kind `style`): `blocks` `"style"` where `blocks` lists it, else leave `blocks` out; `scale` 1.
  Describe the subject as the user asked.
- A kind not given: judge it by its name and trigger. A character's name (a person, a creature, a series'
  character) is a character LoRA; an artist's or a look's name, a style LoRA. When still unsure, leave `blocks` out.
- The user's own strength or block weights are used as they give them.
- A character LoRA with a style LoRA: both, the style's words left to the style LoRA.

- To keep a picture's composition while applying a style LoRA, use `control` with `depth` or `canny` and the LoRA.
