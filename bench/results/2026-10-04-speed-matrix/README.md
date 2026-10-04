<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Speed by model and context length — 2026-10-04

The table of upstream's DETAILS.md (prompt and output speed, five models, 1K to 262K), measured on the development machine: Intel Arc Pro B70 (32 GB), i7-14700, DDR4-3200 in two channels, 91 GiB of RAM ([the machine's record](../2026-10-02-new-machine/README.md)).

- Engine: commit `58dbbe5`, the free build `build/llvm7` (intel/llvm built from source, `STRATA_PORTABLE=ON`), the compiler setup uses on this machine
- What setup writes for the context (images off): `--prefill auto`, the shipped expert profile, `--expert-cache auto`, MTP (`--spec 4 --spec-min-p 0.5`); `--max-context` 8192 for 1K and 4K (setup's smallest), else the tier's context; `--kv int8` above 8K; KV streaming (`--kv-resident 32768`) from 64K
- Greedy, 256 generated tokens, one run per cell, each a new process, after one discarded warm-up run (JIT). [run.sh](run.sh) runs one cell; every log is in [runs/](runs/), and [matrix.json](matrix.json) has every run ([collect.py](collect.py) reads them)
- Prompts: upstream's code-agent prompts are not public. [prompts.py](prompts.py) writes the same kind with the same token counts: a coding agent's system prompt, XeStrata's sources at `58dbbe5` as the files it has read, and a task about them. The engine reads all but the last token as the prompt (1,016 to 259,942 tokens)
- Paths are written as `<repo>`, `<data>` (the data folder) and `<record>`; the logs' line of prompt ids is replaced by its count

## Prompt (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 726 | 1,116 | 1,834 | 1,532 | 1,393 | 1,122 |
| **IQ2_XS** | 685 | 1,063 | 1,818 | 1,524 | 1,388 | 1,095 |
| **IQ3_XXS** | 577 | 862 | 1,759 | 1,488 | 1,356 | 1,099 |
| **IQ3_S** | 450 | 740 | 1,718 | 1,458 | 1,333 | 1,081 |
| **Coder** | 936 | 1,573 | 2,015 | 1,659 | 1,499 | 1,188 |

## Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 78.5 | 84.2 | 72.0 | 67.4 | 63.1 | 56.6 |
| **IQ2_XS** | 86.0 | 96.4 | 81.8 | 73.5 | 76.9 | 62.7 |
| **IQ3_XXS** | 70.7 | 71.9 | 77.0 | 70.1 | 69.4 | 59.1 |
| **IQ3_S** | 77.1 | 73.9 | 72.5 | 68.8 | 63.2 | 50.3 |
| **Coder** | 72.8 | 73.8 | 71.0 | 69.3 | 69.7 | 59.1 |

- Every run ended normally and wrote 256 tokens. The answers are a coding agent's plan and tool calls; the engine does not stop at the end of the turn, so the last tokens go on past `<|im_end|>`, as in upstream's runs.
- IQ3_XXS and IQ3_S at 262K, which upstream's 64 GB machine could not hold, ran here.
- Prompts: the chunk is 32,768 tokens from 32K up (`prompt chunk auto`). 32K is the fastest tier; past it the attention over the earlier chunks grows. The Coder holds all its 12,288 experts in VRAM, so it reads prompts fastest.
- Output moves with the text: speculative decoding writes more per round when more drafts are accepted (`spec_accept`, 0.65 to 0.86 here). IQ2_XS's 128K run (0.768) is faster than its 64K run (0.683) for that reason, and so on. One run per cell, so a difference of a few percent between neighbouring cells says nothing.
- The PCIe probe at start read 22.8 to 43.5 GB/s host to device from run to run (the 2026-10-02 record read 42.4; the link's errors are suspected). Every run kept the same share (`pcie_frac 0.55`), so the configuration did not change, but prompts stream experts over the link; how much a slow link cost a run is `unverified`.
- Not measured: the text speed with images on, and other GPUs.
