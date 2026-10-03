<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The expert arena: one mapping against host USM, measured again — 2026-10-03

The comparison of [the new machine's record](../2026-10-02-new-machine/README.md#the-expert-arena-and-the-pcie-share), on the build of this date (after the prompt-path work of 2026-10-03), to decide how the arena is kept.
Default: the arena is one mapping registered for device copies, so GPU kernels cannot read it and every expert missing from VRAM runs on the CPU.
Host USM (`STRATA_ARENA_HOST_USM=1`): one host USM block per layer, which the GPU's kernels read directly, here with `--pcie-mode kernel --pcie-frac 0.15` (the best setting of the earlier record).
IQ2_XS and IQ3_S, MTP with suffix drafts off, `--expert-cache 10000`, 128 tokens of the 20-token prompt, three rounds alternating ([h1.sh](h1.sh), [logs](runs/)).

| Model | One mapping (tok/s) | Host USM, kernel, 0.15 (tok/s) |
| --- | --- | --- |
| IQ2_XS | 70.75 / 70.57 / 66.44 | 72.80 / 72.38 / 72.45 |
| IQ3_S | 61.38 / 61.33 / 61.36 | 61.49 / 61.37 / 57.85 |

- IQ2_XS: host USM is 2.6–3.2% faster in the rounds without an outlier (one mapping's third round, 66.44, is one). 0.24 experts per layer go over PCIe; the CPU pool's call falls from 9.8 to 9.3 ms per round.
- IQ3_S: no difference (host USM's third round, 57.85, is an outlier).
- The same as on 2026-10-02 (+2–4% at 0.15, slower above it): the GPU's part of a round (about 20 ms waiting for the expert rings) is still the longer one, so moving experts from the CPU to the GPU helps little.
- Both settings repeat their own tokens across rounds; they differ from each other (another CPU/GPU split).

Whether to switch the default is left to the owner: the gain is small and model-dependent, and host USM changes how 33–50 GB of RAM are allocated.
One more thing bears on it: the output head's VRAM refusal at start (seen again three times on 2026-10-03, each with `xe ... VM worker error: -16` in the kernel log) has always been the first device allocation after the arena's mapping was registered, and host USM does not register the mapping.
Whether host USM avoids the refusal could not be told (below).

## Starts back to back

To see whether host USM avoids the refusal, [starts.sh](starts.sh) started the engine 40 times in each arena mode, alternating (IQ2_XS, one generated token), and counted refused uploads and `VM worker error` lines in the kernel log during each start ([log](runs/starts.txt)).
Neither mode failed in 40 starts, and the kernel logged no `VM worker error`.
The three refusals of the day came during other work: a server engine with conversation parking and KV streaming, a server started in a container, and an IQ3_XXS run (40 GB arena).
So this does not tell the modes apart. Whether memory pressure from other work plays a part (the xe driver's VM worker returning EBUSY while it rebinds the registered mapping) is a guess, `unverified`.

## Calibration on the B70

`tools/calibrate.py` (setup's `--calibrate`) ran to the end on a copy of the IQ2_XS config, so nothing was saved ([output](runs/calibrate.txt), 199 s).
It kept `--spec-min-p 0.70` (79.3 tok/s against 76.9 at 0.50 on its three prompts) and the default PCIe share.
With the default arena the GPU cannot read the experts, so the PCIe share changes nothing; its sweep went from 71.7 to 77.3 tok/s, which is the noise of one measurement and larger than the 3% a setting must win by.
