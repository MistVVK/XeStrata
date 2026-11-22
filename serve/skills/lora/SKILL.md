---
name: lora
description: Choosing LoRAs for generate_image, edit_image and control - which, how strong, and their block weights.
metadata:
  kind: tool
  tool: lora
---

- Use a LoRA only when the user names it, or asks for the character or the style it adds (its name, kind and trigger
  in `lora` say which). None otherwise.
- Put its trigger words at the start of the prompt. Set `model` to one listed for that LoRA; the default model can
  reject it. Tell the user if this changes the default model.
- Use these starting points for a character trained on blocks 8–15 and a style trained on all blocks. They are a
  guide from one character and a print style on Qwen-Image-2.1 Turbo, seeds 7 and 8, not a guarantee for every LoRA.
  Leave `blocks` out when applying them; the character LoRA already contains only the trained blocks.

  | LoRAs | Strength | Blocks on application |
  | --- | --- | --- |
  | Character only | 1.0; for a photo, 1.4 | all |
  | Style only | 1.0 | all |
  | Character and style | character 1.0, style 1.0 | all for both |

- With a character alone, write the desired style in the model's prompt style (the user's, else the model's own:
  `watercolor`, `photo`, `anime coloring`) so the LoRA brings the character and not its pictures' style.
- With both, try the style at 0.8 to 1.2 if the starting point does not fit. Describe the subject as requested.
  For a modern scene or night, say so in plain words, such as present-day cars and clothes, a dark night sky, and
  streetlights. A style trained on old pictures can bring their era and light.
- A strong style LoRA can add text or seals. Try `no text, no seal` in the negative prompt or lower its strength.
  Qwen-Image-2.1 Turbo at CFG 1 does not use the negative prompt; at higher CFG it did not remove every seal in the
  print-style comparison.
- A kind not given: judge it by its name and trigger. A character's name (a person, a creature, a series'
  character) is a character LoRA; an artist's or a look's name, a style LoRA. When still unsure, leave `blocks` out.
- The user's own strength or block weights are used as they give them.
- With a character and a style LoRA, leave the style's words to the style LoRA.

- To keep a picture's composition while applying a style LoRA, use `control` with `depth` or `canny` and the LoRA.
