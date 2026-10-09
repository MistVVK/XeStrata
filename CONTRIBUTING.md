# Contributing to Project Maya

Reports and pull requests are welcome. These are what helps most.

## A speed report or a new machine

Open an issue with:

- **`./maya.sh --bench`** (Maya stopped) **then `./maya.sh --report`**, and attach `maya-report.txt`. Windows:
  `START-MAYA.bat --bench` / `--report`. The report includes the bench results.
- **A build from source without the installer** is fine too. Send the same things by hand:
  - the exact engine command and environment (`STRATA_GLM_*` settings);
  - GPUs (model, VRAM, PCIe generation and width as `nvidia-smi` shows them), CPU, RAM, disk;
  - model and context length;
  - the engine log's `glm fast:`, `glm split`, `glm stat:` and `glm spec:` lines.

  For numbers comparable with the README's table, measure the way `--bench` does: three questions of 256 tokens,
  greedy, thinking off, one short warm-up first; prefill (reading the prompt) at 2048 and 8192 tokens.

Name your Maya version (`VERSION`, or the commit). Say what you measured and what you didn't.

## A bug

Run `./maya.sh --report` right after the failure and attach `maya-report.txt`. If the engine stopped, add the end
of its log (the `.log` next to your `maya-*.json`).

## A pull request

- **Base it on `main`** and keep it to one topic.
- **Measure before and after** on your hardware, same harness, and put the numbers in the description. If the
  output changes (it can: which expert runs on the CPU or the GPU changes the last bits), say so.
- **New settings** are `STRATA_GLM_*` environment variables, off or at today's behaviour by default unless measured
  better, with a row in the README's Tuning table.
- **AMD (HIP) and other backends:** keep NVIDIA's code path unchanged (`#if defined(STRATA_USE_HIP)`). We can't test
  AMD ourselves, so we merge AMD changes on your measurements once NVIDIA builds and behaves the same.
- **A new CUDA call needs its AMD mapping.** The AMD build compiles the same sources with
  `include/strata/hip_compat/` forced in. A CUDA runtime or cuBLAS name without a `#define` there breaks every AMD build,
  as v1.0.21 did. `python tools/check_hip_compat.py` names any missing one.
- **Every push and pull request runs the checks** (GitHub Actions, `.github/workflows/tests.yml`), with no GPU needed:
  - the AMD-mapping check;
  - the server tests;
  - the installer and tools tests.
  
  A pull request is merged only when they pass.
- **Before merging, we build every NVIDIA target** and run on Tesla V100s (one GPU and two):
  - the engine's parity tests (`glm_*_parity --selftest`);
  - a decode / prefill A/B;
  - the server end to end.
- **Areas where help is welcome:**
  - decode speed when every expert fits in VRAM (kernel launches and host waits dominate there);
  - Windows with NVIDIA GPUs;
  - AMD;
  - consumer GPUs from 12 to 32 GB;
  - docs.

Every merged change ships in a release (`CHANGELOG.md`), with credit.
