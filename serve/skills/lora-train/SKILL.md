---
name: lora-train
description: Making a LoRA from the browser - when the user wants to train one, open the panel with train_lora and guide them.
metadata:
  kind: tool
  tool: train_lora
---

- When the user wants to make a LoRA ("I want a LoRA of this character", "a LoRA of this painter's style", "train a
  LoRA on these pictures"), call `train_lora`. Pass `model` if the user named an image model. The training panel opens
  in the chat; the user does the work there.
- Distilled models are not training bases. If the tool says it switched models, tell the user that the requested
  model is distilled for few steps and unsuitable for training, name the full-step base selected, and say the LoRA
  can also be used on the requested model. If the base is missing, give the setup command in the tool's error instead
  of saying training can start.
- Tell the user what the panel does, briefly: put the pictures in (drag and drop, several at once, or pick this
  chat's images), choose the kind (character, style, other), the name, the trigger word and the model, caption the
  pictures (the auto button writes them with the lora-caption rules; every caption can be edited), set the most
  steps (n) and how often a version is saved (k), and start.
- The captions follow the lora-caption skill: a character's own look is left to the trigger word (write what
  changes - clothes, pose, background - and the pictures' style); a style LoRA captions the content only, never the
  style's words. Read that skill when the user asks how to caption.
- Repeated signatures or seals can appear in a style LoRA even when captions name them. A negative prompt is not a
  reliable remedy. Check same-seed samples on subjects outside the dataset against the base model. For a style, include
  a present-day scene and a night scene to see whether the training era or daylight carries over.
- The training runs on the server and shares the GPU with the chat: it goes slower while the chat is busy and
  pauses for image jobs; the panel shows its state (running, slowed, paused and why). Do not wait for it and do not
  call train_lora again for the same request.
- When it ends, the panel shows a sample picture of every saved version (the same seed and prompts). The user picks
  one and registers it as a LoRA - the last version by default. The registered LoRA is in the model's list at once,
  with its kind, trigger and step; use it as the lora skill says.
- A cancelled training keeps the versions saved so far; the user can still pick and register one.
