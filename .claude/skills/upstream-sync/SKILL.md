---
name: upstream-sync
description: Bring upstream Strata's newest release tag into XeStrata on dev - sort its commits, merge, port the CUDA and HIP changes to the SYCL kernels, build the three modes - and make XeStrata at least as fast as that tag (at most 1% slower) on the Intel, NVIDIA and AMD GPUs, IQ3_XXS and four other model files. Only when the user asks (/upstream-sync).
disable-model-invocation: true
argument-hint: "[upstream TAG, e.g. v0.1.41]"
---

# Following upstream

upstream is Niko1221/Strata (the remote `upstream`).
A run brings one of its release tags into `dev` and ends when XeStrata is at least as fast as that tag on every GPU it can reach: the check of step 7 prints `ALL PASS`.

Rules for the whole run:

- The tag is `$ARGUMENTS` when given, else the newest: `git fetch upstream --tags; git tag -l 'v*' --sort=-v:refname | head -n 1`.
- Work and commits are on `dev` (Japanese commit messages, one logical change per commit, `git add` and `git commit` joined with `&&`). Nothing is pushed; a release is /release, not this.
- Only merged commits: upstream's open pull requests wait until upstream merges them.
- The GPUs: the Intel Arc Pro B70 and the RTX 4070 in the development machine, the RX 9060 XT in the AMD container. Where they are and how to run there: `LOCAL.md` in the repository root (kept out of git; when it is missing, ask the user and write it there).
  A GPU that cannot be reached (the container's host down, the card missing) is skipped: go on with the others, and report it and its cells as `unverified`.
- AMD work (builds, tests, runs) happens in the AMD container only, never on the development machine; the source goes in as a `git bundle` or a diff.
- A build that fails (any mode, any machine, XeStrata's or upstream's), and a fix that is not plain: stop and report it with the end of the log.
- Speed first: do not stop making a losing cell faster on your own judgement (a small expected gain, hard work, a compiler in the way). When every idea has been tried, report what was tried and the next ideas, and ask.
- Correctness is not traded: the FP64 references and the comparison with llama.cpp still hold, and no GPU gets slower to make another faster.
- Choose a path from what the device reports (AGENTS.md, Hardware independence), never from a GPU's name.

## 1. The starting point

```sh
git status --short                        # empty, on dev
git fetch upstream --tags
grep -n STRATA_UPSTREAM_VERSION CMakeLists.txt
```

The previous integration is the merge `upstream の Strata <version> を Xe に統合する`; its second parent is the last upstream commit `dev` has (`git log --merges --first-parent --format='%h %p %s' dev`).
upstream rewrote its history on 2026-10-06, so `git merge-base dev upstream/main` can be empty: the base is that parent (or, for history before it, its rewritten counterpart).
When the tag is already integrated, go to step 6.

## 2. Sort the commits

Write `.upstream-<version>.md` (add it to `.git/info/exclude`, as `.upstream-0140.md` is) with every commit of `<base>..<tag>` (`git log --no-merges --reverse`), one checkbox line each, in the groups of `.upstream-0140.md`:

| Group | What | Taken |
| --- | --- | --- |
| A | AMD, HIP | ported to the SYCL paths and measured on the RX 9060 XT; one only for a GPU there is none of (gfx11, gfx103x) is not ported, and the record says so |
| B | bench records | the records as they are |
| D | documents | what is true of XeStrata, into its documents (Japanese first) |
| E | engine features and fixes | all |
| M | several GPUs | all |
| N | NVIDIA, every generation (Pascal, Volta, Turing included) | ported to the SYCL kernels |
| P | speed paths | step 4 |
| S | upstream's `sycl/` | not taken: it stays deleted in the merge and is measured as a competitor (step 6) |
| T | serve, setup, tools | the merge |
| W | Windows | not taken |

Each line ends with what became of it (a commit, `N/A` and why, or the measurement that left it out).

## 3. The merge

```sh
tree=$(git merge-tree --write-tree --merge-base=<base> dev <tag>)   # resolve the conflicts it reports
git commit-tree -p dev -p <tag> -m "upstream の Strata <version> を Xe に統合する" <tree>
```

`sycl/` and the CUDA and HIP sources stay deleted (modify/delete conflicts are taken as deletes); their changes are ported by hand in steps 2 and 4.
Set `STRATA_UPSTREAM_VERSION` in `CMakeLists.txt`, and replace any old upstream hash a document names with the rewritten one.

## 4. Port and measure

Build folders are configured with Ninja and ccache (docs/DEVTOOLS.md, "Faster builds"); check the hits with `ccache -s`.
While porting, build and measure with the contrib-llvm build (`build/contrib`) only; the free and contrib-icpx modes come once at the end (step 5).

- Port related items as a group, each item behind an environment switch so the group can be split.
- A change that can be checked bitwise is checked by a probe alone; an opt-in feature is run once to see it works.
- A speed path is measured on all three GPUs and on each with its matrix units off (`STRATA_NO_XMX`, `STRATA_NO_BF16_MMA`, `STRATA_NO_INT8_MMA`): probes on the B70 and the RTX 4070 may run together, whole-engine A/B runs one card at a time.
- Kept when it is faster on one GPU and slower on none (chosen at run time from what the device reports where it helps only some); removed when it is faster nowhere. The numbers go into the commit and the record.
- Only when a group is not faster, split it with its switches.

## 5. The three modes

```sh
tools/lint/run.sh                          # the files that differ from HEAD
```

Then build and run CTest in `build/contrib`, `build/llvm7` (intel/llvm, without oneAPI's environment) and `build/xe` (icpx), and in the AMD container its contrib-llvm build with HIP (`STRATA_HIP_ARCHS`).
A warning or a mismatch only one mode shows is traced back to the commit that caused it and fixed there.
AVX-512 code touched: AGENTS.md's Intel SDE runs.

## 6. Against the tag

The engines, each built in a worktree of its own under `.claude/worktrees/` (or in the AMD container):

- XeStrata: `dev`'s HEAD, the contrib-llvm build; one GPU at a time with `ONEAPI_DEVICE_SELECTOR` (`level_zero:*`, `cuda:*`; `hip:*` in the container).
- upstream on the B70: the tag's `sycl/` port, built and run in its Docker image as its docs/INTEL.md says.
- upstream on the RTX 4070: the tag's CUDA build; on the RX 9060 XT: its HIP build; both as its documents say.

Each engine's own shipped expert profile. No other work on a machine while it measures.

The model files (their paths on each machine: `LOCAL.md`):

| File | Tiers |
| --- | --- |
| IQ3_XXS | `1k 4k 32k 64k 128k 262k` |
| Q2_0, IQ2_XS, IQ3_S, UD-Q4_K_XL | `1k 4k 32k` |

```sh
o=bench/results/<date>-upstream-<version>
python bench/results/2026-10-04-speed-matrix/prompts.py <pack> <ids>        # the prompts, once
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> \
  [PACK=... NATIVE=... PLE=... EXTRA="..."] TIERS="..." \
  .claude/skills/release/scripts/bench.sh $o/<file> <GPU>                    # per file and GPU
python .claude/skills/release/scripts/bench-collect.py $o/<file> xe-<commit> <tag>
```

`bench.sh` alternates the engines, two rounds per tier, after a warm-up each.
It stops upstream mid-run once its progress shows it at 10 tok/s or less (the prompt, from its `PP` lines after 30 s; the output, when 256 tokens at 10 tok/s and a minute have passed since the prompt) and does not run it again in that tier.

## 7. The check

```sh
python .claude/skills/upstream-sync/scripts/judge.py $o/IQ3_XXS $o/Q2_0 $o/IQ2_XS $o/IQ3_S $o/UD-Q4_K_XL
```

A cell (file, GPU, tier, prompt or output) passes when XeStrata's median is at most 1% below upstream's.
A cell upstream does not finish, finishes at 10 tok/s or less, or is stopped as that slow, is left out; one only XeStrata does not finish fails.

For each failing cell: find where the time goes (the GPU profilers of docs/DEVTOOLS.md, the engine's timing lines), read how upstream does that part, make XeStrata's faster (step 4's rules), and measure that cell again.
The run is done only when a full run of step 6 on the final HEAD, every file, tier and reachable GPU, gives `ALL PASS`: a cell fixed earlier can lose again to a later change.

## 8. The record and the report

On `dev`:

- `$o/README.md`: the machines, both engines' commits and builds, the method, the tables (`bench-collect.py`), the check's output, what the numbers say, and the GPUs skipped; `matrix.json` and the logs in `runs/` with this PC's paths written as `<repo>` and `<data>`. Step 4's measurements of kept and removed paths go into the same record.
- docs/XE.ja.md, then docs/XE.md: the upstream version and range brought in, what was not taken and why, linking the record.

Lint and commit. Report: the tag, what was taken and left out by group, the check's table, the GPUs skipped (`unverified`), and anything that needs the user.
