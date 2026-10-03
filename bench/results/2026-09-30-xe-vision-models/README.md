# Images on every model — 2026-09-30

Plan section 9.6: back to each model that runs (see [docs/XE.md](../../../docs/XE.md#models-on-the-b70)) with image requests. OrcaRouter IQ3_XXS was not downloaded, so it is not here. Swift 1.5 Q2_0 did not load at first; it was added later the same day, after the pack index v4 ([its record](../2026-09-30-xe-swift-q2_0/README.md)).

For each model, [vision_model_check.sh](vision_model_check.sh) starts the server with the CPU image encoder (`--vision cpu`, cap 300 image tokens, the model's own mmproj: the original's for Qwen and the Coder, Swift's for Swift) and sends, in order:

1. the landscape image of [the image record](../2026-09-30-xe-vision-cpu/README.md) with "What text is written in the image, and what shapes and colors are in it?" (temperature 0, 64 tokens)
2. the portrait image with the same question (a different image, a grid of the other shape)
3. both images in one request ([req-two.json](req-two.json)): "Describe each of the two images in one sentence."
4. the landscape image with "What text is written in the image? Answer briefly.", thinking off ([req-turn1.json](req-turn1.json))
5. the same conversation, the image sent again, with the answer and "What colour is the circle? Answer briefly."

It then runs requests 1 and 2 through llama-mtmd-cli at the pinned commit on the CPU, given the exact prompt the server renders ([as in the image record](../2026-09-30-xe-vision-cpu/README.md)), and compares the reasoning texts. The server uses MTP drafts; llama-mtmd-cli decodes plainly. Every result is in [summary.txt](summary.txt); each model's folder holds the replies, the reference texts, the prompt and the engine log.

| Model | Landscape vs llama-mtmd-cli (characters in common) | Portrait vs llama-mtmd-cli | Two images (prompt tokens) | Turn 1 | Turn 2 (prompt tokens reused) |
| --- | --- | --- | --- | --- | --- |
| [Qwen IQ2_XS](qwen-iq2_xs) | all 257 of the reference | 184 | 586 | "HELLO 4172" | "Red" (318) |
| [Qwen IQ3_XXS](qwen-iq3_xxs) | 148 | 217 | 586 | "HELLO 4172" | "Red" (318) |
| [Qwen IQ3_S](qwen-iq3_s) | 235 | 197 | 586 | "HELLO 4172" | "Red" (318) |
| [Qwen Q2_0](qwen-q2_0) | 185 | 145 | 586 | "HELLO 4172" | "Red" (333) |
| [Coder IQ1_M](coder-iq1_m) | all 235 of the reference | all 279 of the reference | 586 | "HELLO 4172" | "Red" (318) |
| [Swift IQ2_XS](swift-iq2_xs) | 135 | 21 | 586 | "HELLO 4172" | "Red" (318) |
| [Swift IQ3_XXS](swift-iq3_xxs) | 51 | 51 | 586 | "HELLO 4172" | "Red" (318) |
| [Swift Q2_0](swift-q2_0) | 94 | 105 | 586 | "HELLO 4172" | "Red" (333) |

- Every split from llama.cpp is between two wordings of the same thing. Within 64 tokens of reasoning the replies do not always reach the description yet (Qwen IQ3_S's landscape reply names none of the shapes, colours or text by then); where they do, they name them correctly, and several (as llama.cpp's own reply does on IQ2_XS) also mention black bars that the portrait image does not have. What every model answers is turns 1 and 2: the text "HELLO 4172" and the circle's colour. The earliest splits are Swift's. At its 21st character the portrait reply reads "identify the text, shapes, and" and the reference "describe the image: the text,". Swift IQ3_XXS takes, at the same point, "I need to identify…" for one image and "Let me carefully examine…" for the other, where the reference takes the opposite. llama.cpp gives no logits here, so how close these choices were is **unknown**.
- The two-image prompt is 300 + 220 image tokens plus the text on every model. Within 64 tokens of reasoning the models describe the first image; only Swift IQ2_XS reaches the second. On IQ2_XS, with thinking off and 200 tokens ([reply](qwen-iq2_xs/two-nothink.json)), the answer describes both: "a red circle next to a blue square, with the text "HELLO 4172" written below them" and "a green triangle positioned above the word "PELICAN"".
- Turn 2 sends the image again. The engine reuses the earlier conversation from its cache, image included (318 of the 355 prompt tokens; 333 on both Q2_0 models), and answers from it. That the encoder served the repeated image from its own cache (keyed by the image's hash) is by design and was not observed separately.
