---
name: lora
description: Choosing LoRAs for generate_image and edit_image - which, how strong, and their block weights.
metadata:
  kind: tool
  tool: lora
---
- Use a LoRA only when the user names it, or asks for the character or the style it adds (its name, kind and trigger
  in `lora` say which). None otherwise.
- Write its trigger words in the prompt.
- A character LoRA (kind `character`, or a person's or creature's name): `blocks` `"character"`, `scale` about 0.7,
  and write the style the picture should have (the model's own or the user's: `anime coloring`, `watercolor`,
  `photo`), so the LoRA brings the character and not its pictures' style.
- A style LoRA (kind `style`): `blocks` `"style"`, `scale` 1; describe the subject as the user asked.
- A kind not given: judge it by its name and trigger; when unsure, `blocks` `"all"`.
- A name `blocks` does not list: leave `blocks` out.
- The user's own strength or block weights are used as they give them.
- A character LoRA with a style LoRA: both, the style's words left to the style LoRA.
