<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# MTP draft acceptance across models and prompts — 2026-10-03

The MTP draft layer's acceptance had been measured on one prompt per model, and Swift 1.5 IQ2_XS stood out at 0.27 against 0.52–0.60 for the others.
This record measures every installed model on four prompts.
Machine: Arc Pro B70, i7-14700 ([the new machine's record](../2026-10-02-new-machine/README.md)).
Settings are setup's decode settings: `--spec 4 --spec-min-p 0.5`, the MTP draft layer with the CJK draft vocabulary, `--expert-cache auto`, greedy, and `--adapt-every 0`, so one run does not depend on the one before.
Each run generates up to 512 tokens and ends at the model's first end of turn (`--stop-eos`).
There is one run per model and prompt ([accept.sh](accept.sh), [logs](runs/)).

The prompts, as user turns with no system prompt, the model thinking as it chooses:

- chat: "Explain in three sentences why the sky is blue." ([ids](chat_ids.txt))
- ja: the same question in Japanese ([ids](ja_ids.txt))
- code: a Python `merge_sorted` with a docstring and doctests ([ids](code_ids.txt))
- math: a word problem with steps ([ids](math_ids.txt))

## Acceptance (accepted / drafted) and tokens per verify round

| Model | chat | ja | code | math |
| --- | --- | --- | --- | --- |
| Qwen IQ2_XS | 0.679, 2.47 | 0.663, 2.29 | 0.840, 3.31 | 0.834, 3.35 |
| Qwen Q2_0 | 0.605, 2.37 | 0.646, 2.37 | 0.856, 3.43 | 0.813, 3.35 |
| Qwen IQ3_XXS | 0.610, 2.35 | 0.711, 2.58 | 0.831, 3.32 | 0.832, 3.29 |
| Qwen IQ3_S | 0.595, 2.39 | 0.763, 2.59 | 0.825, 3.35 | 0.792, 3.22 |
| Coder IQ1_M | 0.521, 2.13 | 0.732, 2.77 | 0.802, 3.25 | 0.830, 3.37 |
| Swift IQ2_XS | 0.515, 2.06 | 0.686, 2.57 | 0.826, 3.29 | 0.799, 3.26 |
| Swift Q2_0 | 0.573, 2.26 | 0.682, 2.53 | 0.842, 3.29 | 0.840, 3.41 |
| Swift IQ3_XXS | 0.602, 2.43 | 0.683, 2.54 | 0.791, 3.16 | 0.836, 3.35 |

Generated tokens: chat 67–214, ja 189–273, code 512 (the cap) for every model, math 333–436.
Decode speed is in the [log](runs/accept.txt) (chat 50–70 tokens/s, code 66–89).

- Acceptance depends on the text more than on the model: 0.52–0.68 on the short chat answer, 0.65–0.76 in Japanese, 0.79–0.86 on code and arithmetic.
- Swift 1.5 is like the others. Its lowest figure, 0.515 on chat, comes from 67 tokens: it answered without thinking.
- Runs with a fixed number of tokens (as the earlier ones were) keep generating past the end of turn, and there the acceptance swings either way. Swift IQ3_XXS on chat with 256 fixed tokens gives 0.269 in four runs of the normal build and 0.271 with the ASan build ([log](runs/swift-iq3_xxs-chat-256.txt)). Its first 173 tokens are the same as the run that stops at the end of turn, which accepts 0.602. The 83 tokens after the end of turn took 79 rounds and accepted 2 of 220 drafts. Swift IQ2_XS on chat with 256 fixed tokens gave 0.717 instead: past its end of turn it repeated `<|im_start|>`, which the draft layer guesses. The earlier 0.27 was the same effect: that run ([log](../2026-10-02-new-machine/runs/chat-swift-iq2_xs-mtp-1.txt)) generated 96 fixed tokens and its end of turn came at the 69th, while the run here that stops there accepts 0.515.

## Seen on the way

- One run of the first pass (Swift IQ3_XXS, chat, 256 tokens) ended after its statistics with `free(): invalid next size (normal)`, a heap corruption found at exit. The same command three more times with the normal build and once with ASan and UBSan (host code) ran clean, so the cause is not known.
- One run of the first pass (IQ3_XXS, code) did not start: the output head's VRAM upload was refused with 27 GiB free, with `xe ... VM worker error: -16` in the kernel log 8 s before ([the arena record](../2026-10-03-arena-usm/README.md) has more on this).
