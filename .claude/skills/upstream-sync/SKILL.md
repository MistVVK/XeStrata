---
name: upstream-sync
description: Bring upstream Strata's newest release tag into XeStrata on dev - read its commits as ideas, port the shared changes and each idea to the SYCL paths and keep it where an A/B of its switch shows it faster (whether XeStrata already wins there or not), take output-changing approximations by upstream's own quality check, build and test the free and contrib-llvm modes - and end with XeStrata at least as fast as that tag (at most 1% slower) and as the previous XeStrata on the Intel, NVIDIA and AMD GPUs, on the generate path, the serve path and everyday use. Only when the user asks (/upstream-sync).
disable-model-invocation: true
argument-hint: "[upstream TAG, e.g. v0.1.42]"
---

# Following upstream

upstream is Niko1221/Strata (the remote `upstream`).
A run brings one of its release tags into `dev`.
Its work is upstream's ideas: each is read, ported to XeStrata's SYCL paths and kept where it makes XeStrata faster, whether XeStrata already beats upstream there or not (steps 2 to 5).
The tables against the tag and the previous XeStrata come last, as the check that nothing lost (steps 7 and 8); the run ends when that check prints `ALL PASS`.

Rules for the whole run:

- The tag is `$ARGUMENTS` when given, else upstream's newest: `git ls-remote --tags --refs upstream 'v*' | sed 's#.*refs/tags/##' | sort -V | tail -n 1`; not `git tag -l 'v*'`, which also lists other remotes' tags (Project Maya's `upstream-maya` has v1.0.x). Then `git fetch upstream --tags`.
- Work and commits are on `dev` (Japanese commit messages, one logical change per commit, `git add` and `git commit` joined with `&&`). Nothing is pushed; a release is /release, not this.
- Only merged commits: upstream's open pull requests wait until upstream merges them.
- The GPUs: the Intel Arc Pro B70 and the RTX 4070 in the development machine, the RX 9060 XT in the AMD container. Where they are and how to run there: `LOCAL.md` in the repository root (kept out of git; when it is missing, ask the user and write it there).
  A GPU that cannot be reached (the container's host down, the card missing) is skipped: go on with the others, and report it and its cells as `unverified`.
- AMD work (builds, tests, runs) happens in the AMD container only, never on the development machine; the source goes in as a `git bundle` or a diff.
- A build that fails (any mode, any machine, XeStrata's or upstream's), and a fix that is not plain: stop and report it with the end of the log.
- Speed first: do not stop making XeStrata faster on your own judgement (a small expected gain, hard work, a compiler in the way). When every idea has been tried, report what was tried and the next ideas, and ask.
- Correctness: an exact path keeps the FP64 references and the comparison with llama.cpp; an approximation that changes the output goes through step 5. No GPU gets slower to make another faster.
- Choose a path from what the device reports (AGENTS.md, Hardware independence), never from a GPU's name.
- Both engines: Qwen3.8-Flash-Next (and the Gyro packs) run on `src/program/generate.cpp`, the other models (Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B, Qwen3.5-9B, GLM-5.3-Flash) on `src/program/plain_generate.cpp` and `src/core/plain_model.cpp`, and the two do not differ in what they can do (the user's decision of 2026-10-31).
  upstream's changes land in `generate.cpp`; whatever of them is taken (a feature, a fix, a speed idea, an engine option) is taken into the plain engine too, in the same run, as a shared part both engines call (`src/core/`, `src/kernels/`, `src/prefill/`, `serve/`), not as a second copy.
  Only what rests on Qwen3.8-Flash-Next's own architecture stays with it: what `kQwen38Only` in `src/program/option_scope.cpp` lists (PLE, QSA, the separate MTP pack and their kernels).
  An engine option `generate.cpp` gains goes into the plain engine or into `kQwen38Only`.

## 1. The starting point

```sh
git status --short                        # empty, on dev
git fetch upstream --tags
grep -n STRATA_UPSTREAM_VERSION CMakeLists.txt
git tag -l 'xe*' | sort -V | tail -n 1    # the previous XeStrata (step 8)
```

The previous integration is the last commit of its run, which records the tag as its second parent and keeps the tree (step 9; `upstream の Strata <version> を Xe に統合する` before 2026-10-11); that parent is the last upstream commit `dev` has (`git log --merges --first-parent --format='%h %p %s' dev`).
upstream rewrote its history on 2026-10-06, so `git merge-base dev upstream/main` can be empty: the base is that parent (or, for history before it, its rewritten counterpart).
When the tag is already integrated, go to step 7.

## 2. Read upstream: the ideas

Write `.upstream-<version>.md` (add it to `.git/info/exclude`, as `.upstream-0140.md` is) in two parts.

**The commits.** Every commit of `<base>..<tag>` (`git log --no-merges --reverse`), one checkbox line each, with its group:

| Group | What | Taken |
| --- | --- | --- |
| A | AMD, HIP | as ideas (below) |
| B | bench records | not taken (AGENTS.md, Measurement records); read them for findings (below) |
| D | documents | what is true of XeStrata, into its documents (Japanese first); read upstream's investigation records for findings |
| E | engine features and fixes | all |
| M | several GPUs | all |
| N | NVIDIA, every generation (Pascal, Volta, Turing included) | as ideas (below) |
| P | speed paths | as ideas (below) |
| S | upstream's `sycl/` | not taken; its engine is what the B70 is compared with in steps 7 and 8 |
| T | serve, setup, tools | ported (step 3) |
| W | Windows | ported as the group it would otherwise be (E, P, T) is, onto XeStrata's Windows code (AGENTS.md, Operating systems); only a part for what XeStrata's Windows build does not have (a backend, upstream's own packaging) gets `N/A` |

Each line ends with what became of it: the idea it belongs to, a commit, or `N/A` and why.

**The ideas.** The commits of groups A, N and P (and any of E, M or T that changes a speed or a default) bound into ideas: one idea is one change of behaviour, however many commits it took upstream (the commit that adds it, the ones that tune it, the one that makes it the default).
For each idea:

| Field | What |
| --- | --- |
| Commits | the commits it binds |
| Kind | a kernel; a host-side policy (a default, a split, a cache or schedule decision); a feature |
| Where it is on | upstream's condition (CUDA only, not HIP, single GPU, an architecture, a size) and why: measured only there, or true only there (an instruction, a library, a hardware limit); from the commit messages, the code and upstream's records |
| Where it works | the setting it is for (a tier, the serve path's output, the follow-ups, a model file, experts that do not fit in VRAM) and upstream's measurement |
| Output | exact, or changes the output (step 5) |
| In XeStrata | to port (shared code, step 3; a speed path, step 4), or not applicable and why; for both engines, or for Qwen3.8-Flash-Next's alone and why (its architecture, `kQwen38Only`) |
| Result | step 4's and 5's numbers and what was kept |

A condition measured only on upstream's GPUs is not a reason to keep it: step 4 measures the idea on XeStrata's GPUs and puts its own condition, from what the device reports.
A host-side policy upstream keeps behind its backend (`#if` on CUDA, a check of the backend) is ported without that condition, under XeStrata's own (step 4).

**What earlier runs missed.** Earlier runs ported by hand, and the small fixes of setup, serve and tools were the easiest to miss (2026-10-11: eleven of them, from before v0.1.40.2).
List upstream's commits whose surviving lines XeStrata mostly lacks, over all of upstream's history, not only this range:

```sh
python .claude/skills/upstream-sync/scripts/carryover.py <tag> --ref dev --ledger .upstream-*.md
```

It compares the lines each commit added that upstream still has at the tag with the same file in `dev` (comments and short lines left out); a fix of one or two lines is not found this way.
Read each one listed: what XeStrata does not have and needs goes into the ledger (a line under its own heading) and is ported as in step 3; the rest gets `N/A` and why (CUDA or HIP only; for what XeStrata's Windows build does not have; XeStrata does it another way), so the next run's `--ledger` leaves it out.

**The findings.** upstream's records and community reports of this range (`bench/`, its investigation documents) said something about a GPU or a setting XeStrata runs on (an Intel Arc investigation, a card's bottleneck, a setting users run): a line each, with what XeStrata could take from it. They are worked like ideas in step 4 when they point at something to try.

**The licenses.** Every commit of the range that changes upstream's `LICENSE`, or adds code that names another copyright holder or license (an SPDX header, a notice such as ggml's MIT text, a vendored project), gets a line under its own heading with the holders and the license:

```sh
git diff --stat <base>..<tag> -- 'LICENSE*' 'NOTICE*'
git log -p --no-merges <base>..<tag> | grep -nE '^\+.*(SPDX-|Copyright|Permission is hereby granted)'
```

What step 3 takes from it keeps those holders and that license (AGENTS.md, Licenses); a holder or license XeStrata did not carry before also goes into `NOTICE` (and `LICENSES/`, `REUSE.toml`).

## 3. Port the shared changes

XeStrata's kernels are written from scratch (SYCL) and its prompt path and runtime have moved far from upstream's, so the tag cannot be merged (no `git merge`, no `git merge-tree`): every change taken is ported by hand onto `dev`, and the tag is recorded at the end, by a commit that keeps the tree as it is (step 9), as c3aa33ae (v0.1.40.2) and 0509e3e6 did.

- One commit per upstream change, or per idea (step 2), with upstream's hashes in the message: `... （upstream 1e6cec80、e57c9072）`.
- Groups E, M and T, and the parts of A, N and P that are shared code, are ported here or in step 4; `sycl/`, the CUDA and HIP sources and the HIP tables are not taken.
- Group W is ported as the group it would otherwise be, in `#if defined(_WIN32)` branches or `src/platform/` as AGENTS.md says; running it on Windows waits for a milestone (`unverified` until then).
- Each ledger line gets the commit that took it, or `N/A` and why (the code is not in XeStrata, or XeStrata does it another way).
  A line taken names the commit for each engine (`generate: <hash>, plain: <hash>`; one hash when one commit gives both the shared part), or `plain: N/A` and the architecture that keeps it from the plain engine.
- The documents XeStrata has (Japanese first) get what is true of XeStrata; upstream's new documents are not copied as files.
- <!-- upstream-name: off -->Upstream's name is read as XeStrata's while porting, as nothing a user sees says strata alone:
  a message or log head `"strata generate: ..."`, `"strata serve: ..."`, `"strata: ..."` becomes `"xestrata ...: ..."`;
  an environment variable `std::getenv("STRATA_X")` becomes `strata::env_get("X")` (include/strata/platform/env.hpp,
  which reads `XESTRATA_X` and then `STRATA_X`), and `os.environ.get("STRATA_X")` in serve/, setup.py and tools/ becomes
  `envvars.get("X")` (serve/envvars.py); a document or message names it `XESTRATA_X`;
  a program is named by its output name (`xestrata-engine`, `xestrata-image`, `xestrata-*`: CMakeLists.txt sets
  OUTPUT_NAME at its end, so a new `strata-*` target gets it there), while CMake's target names, options (`-DSTRATA_*`)
  and the namespace `strata` stay upstream's.
  `tools/lint/upstream_name.py` (run by `tools/lint/run.sh`) reports what was missed; a use that is upstream's on
  purpose (a field shared with upstream, a mark inside a file) goes into its ALLOWED list.<!-- upstream-name: on -->
- Once the ledger is done, set `STRATA_UPSTREAM_VERSION` in `CMakeLists.txt` to the tag and `XESTRATA_REVISION` back to 1 (AGENTS.md, "Packages and versions"), and replace any old upstream hash a document names with the rewritten one.
- Then bring the license classes (AGENTS.md, Licenses) to the new tag, in one commit (`ライセンスの表示を upstream <version> に合わせる`): `python tools/license_class.py --fix` rewrites the headers, so a file brought back to upstream's content loses XeStrata's line and goes back to MIT, and one XeStrata has changed for the first time gets XeStrata's line and `LGPL-3.0-or-later AND MIT`.
  What it reports and does not fix is fixed by hand: a `REUSE.toml` annotation (upstream's unchanged data files go under one with upstream's holders and MIT) and a file without a header. A new file of upstream's taken as it is gets a header with upstream's holders and MIT (the header is not a change); one taken and changed, both holders and `LGPL-3.0-or-later AND MIT`.
  A later commit of the run that changes an unchanged file of upstream's also changes its header; step 6's lint finds the ones missed.

## 4. Port each idea and measure it

Build folders are configured with Ninja and ccache (docs/DEVTOOLS.md, "Faster builds"); check the hits with `ccache -s`.
While porting, build and measure with the contrib-llvm build (`build/contrib-llvm`) only; the free mode comes once at the end (step 6).

Every idea is ported and measured, whether XeStrata wins or loses against upstream in its setting: an idea that made upstream faster can make XeStrata faster still, and a win only grows this way.

- Port it behind an environment switch (`XESTRATA_<NAME>=0|1` through `strata::env_get`, or the option upstream gave it), so it can be measured on and off and turned off later. Related ideas go in as a group, each behind its own switch.
- Its condition is XeStrata's: what the device reports (matrix units, memory, the PCIe link, the CPU pool's measured speed), not upstream's backend.
- An exact change that can be checked bitwise is checked by a probe; an opt-in feature is run once to see it works.
  A change that has to leave the answers as they are is checked by the answers' md5 (below), a cell that moved saying
  either that the change is not exact (step 5) or that it is a bug.
- A speed idea is measured with `scripts/ab_pair.py`: the same binary with its switch off (A) and on (B), in the setting the idea is for (its tier, the serve path, a conversation; `TIERS`, `MODES`, `CONVS`), once (`ROUNDS=1`) on one GPU, the one the idea is most likely to help (the user's decision of 2026-10-12).
  Both runs are XeStrata with one seed: they write one text unless the switch changes the output, so the difference is the switch's alone and a difference of 1 to 2% shows.

```sh
A="env XESTRATA_X=0 build/contrib-llvm/xestrata-engine" B="env XESTRATA_X=1 build/contrib-llvm/xestrata-engine" \
  PROFILE=data/expert-profile.bin DATA=<data> IDS=<ids> TIERS="32k" A_NAME=off B_NAME=on \
  python .claude/skills/upstream-sync/scripts/ab_pair.py generate|serve|usage <out> <GPU>
python .claude/skills/upstream-sync/scripts/judge.py <out>        # B against A per cell
```

- Faster on that GPU, or equal (within ±1%): kept, on by default.
  Slower: measured once on each of the two other GPUs; kept on where it is faster or equal there (chosen at run time from what the device reports, when it helps only some); opt-in where it is slower everywhere but upstream keeps it as an option; removed otherwise.
  The numbers go into the idea's line, the commit and the record.
- Probes on the B70 and the RTX 4070 may run together; whole-engine A/B runs one card at a time.
- An idea taken into the plain engine is measured there as well, by the same rule, on a plain model (Ornith-1.5 or Qwen3.6-35B-A3B), with `--no-pcie-split --adapt-every 0` where two outputs are compared (else the measured CPU/GPU split changes the output from run to run).

### The answers that must not move

An idea that must leave the answers as they are is checked by their md5, against the previous record's: a cell is one model at one length (Qwen3.8-Flash-Next and the plain models, at 1K and 32K) or one image model at one size (the four, at 512²), run on a fixed text with the settings that decide its answer (greedy, `--adapt-every 0 --pcie-frac 0`, setup's context for the length; the image models' 4 steps and seed 7), two rounds a cell (a cell that answers differently from run to run is one the watch cannot read alone).
A run keeps the answer's line and the logits it was chosen from, or the PNG; the md5s are of those, so the unify2 records' tables (`llm/*.out`, their `s-sum.py`) compare with a new record as they are.

```sh
DATA=<data> XE="CMD..." XE_PROFILE=<data>/data/expert-profile.bin [XE_IMAGE_EXE=... XE_LIB_DIRS=...] \
  python .claude/skills/upstream-sync/scripts/md5_watch.py $o/md5-<step> <GPU>      # every cell: ~30 min
python .claude/skills/upstream-sync/scripts/md5_judge.py $o/md5-<step> --prev $o/md5
```

`md5_judge.py` prints a line a cell: `SAME` (the same answer from the same logits bit for bit), `BITS` (the same answer, other logits: the arithmetic moved without moving it), `MIXED` (each record has an answer the other lacks), `CHANGED` (no answer in common), `FAILED` (no answer where the previous record's had one), `NEW` and `NOT COMPARED`.
A cell that moved is read by the KL of the logits (an LLM's: the previous record's distribution against this one's, over the rows both read the same text at) or by the pixels' difference (an image's), each against its own floor: `SAME` and `MIXED` are not failures, `BITS`, `CHANGED` and `FAILED` are (exit 1).
The cells cost more than an A/B: run them after an idea that has to be exact, or once for a bundle of such ideas, on the GPU whose answers the engine gives most steadily (the B70), and keep `$o/md5` as the next run's baseline (step 9).
The auto choices (`--expert-cache auto`, `--prefill auto`) can pick other slot counts between two starts and with them other bits: a cell that answers differently round to round shows as `MIXED`, and `EXTRA` takes setup's `--vram-free-mib` and `--prefill-cpu-share` to pin them.
`LOGITS=0` saves the dumps (32 MB a cell) and leaves a moved cell's KL unread; `MODELS`, `TIERS`, `IMAGE_MODELS` and `ROUNDS` narrow a step to the cells it touched.

### GPUs there is no machine for

An idea only for GPUs there is no machine for (NVIDIA Pascal, Volta and Turing; AMD gfx103x and gfx11) is ported too (the user's decision of 2026-10-09), on upstream's measurements:

- one upstream turns on by default there, with a measurement on such a GPU (its commit or its records): on by default on the same GPUs;
- one upstream keeps opt-in: opt-in, behind the same kind of switch.

It is chosen from what the device reports or from the compiled target (`__CUDA_ARCH__`, `__GFX11__`), never from a GPU's name, and it must not change the path of a GPU there is a machine for.
Its correctness is checked on what there is:

| GPUs | Check |
| --- | --- |
| NVIDIA Pascal, Volta, Turing | a contrib-llvm build with only that generation's code (`STRATA_CUDA_ARCHS=sm_60`, `sm_70`, `sm_75`) run on the RTX 4070 (its driver compiles the PTX; docs/BUILD.md did so for Pascal): CTest and the bitwise probes |
| AMD gfx103x, gfx11 | a build for them (`STRATA_HIP_ARCHS=gfx1030;gfx1100`) in the AMD container; a path without their own instructions forced on the RX 9060 XT by its switch; code with their own instructions (gfx11's WMMA) is checked by the build alone |
| GPUs without matrix units (Pascal, RDNA2, Intel without XMX) | the fallback paths with `XESTRATA_NO_XMX`, `XESTRATA_NO_BF16_MMA`, `XESTRATA_NO_INT8_MMA` on the three GPUs |

What could not run is `unverified`. These GPUs are not part of step 8's check: the record and the report list each such path with upstream's measurement and `unverified` on XeStrata's speed.
Paths an earlier merge left out for want of a machine (the `移さない` lines of `.upstream-0140.md`'s group A) are taken up the same way.

## 5. Approximations that change the output

An idea that changes the output (it skips experts, reads fewer of them, rounds coarser) is taken by upstream's own quality check, as a rule (the user's decision of 2026-10-11):

- Find upstream's check in its commit or record (KL divergence against the exact path, a task check, the models and contexts it ran) and its criterion.
- Run the same check on XeStrata's paths, on each GPU whose path the idea changes (the check's models and settings, XeStrata's binary with the switch off and on).
- Within upstream's criterion: on by default where upstream has it on by default (under XeStrata's own condition, step 4) and where step 4's A/B shows it faster; its switch turns it off. Beyond the criterion on a GPU, or no check upstream ran to follow: opt-in, and the report says why.
- Its numbers (the check's and the A/B's) go into the idea's line, the commit and the record; the documents name the switch that turns it off, as upstream's do.

Step 4's watch is the first of those checks, and the one that says where the output changed: the cells the idea touches are run before and after it, and the judge gives their KL (an LLM's) or pixels (an image's) against the floor two rounds of one build give (0 when they are bitwise equal).
Within the floor the movement is the engine's own noise, not the idea's; beyond it, and for a cell whose answer changed, upstream's check above decides.
An idea whose cells all come back `SAME` is exact: it goes in on by default, its switch turned off where it is slower (step 4).

When the rule does not fit an idea (upstream's check does not apply to XeStrata's path, or the loss is plain in use though within the criterion), ask the user.

## 6. Build and test

```sh
tools/lint/run.sh                          # the files that differ from HEAD
python tools/license_class.py              # every file's license class against the tag
```

The license check needs upstream's tag in the clone (step 1's fetch); without it, it is skipped.

As AGENTS.md says, with intel/llvm only, in the free and contrib-llvm modes:

- on the development machine, build and run CTest in `build/free` (without oneAPI's environment) and `build/contrib-llvm`, all of it, the slow tests too (`ctest -j$(nproc)`);
- in the AMD container, its contrib-llvm build with HIP (`STRATA_HIP_ARCHS`), and CTest likewise;
- the Windows cross build (docs/DEVTOOLS.md, Windows cross build), whole and linked: no new warnings;
- a change NVIDIA's device compile sees is run on the RTX 4070.

A warning or a mismatch only one mode shows is traced back to the commit that caused it and fixed there.
AVX-512 code touched: AGENTS.md's Intel SDE runs.
The answers' md5 (step 4) runs in the build the tables use, contrib-llvm: a cell of the free build is not compared with one of contrib-llvm's, their dense products being other libraries (oneMKL against XeStrata's own kernels) and so other bits.

## 7. Against the tag

The tables against the tag, on the three GPUs, with generate, the serve path and everyday use, for every model file: [references/measure.md](references/measure.md) has the engines, the files and tiers, and the commands.
They come after step 4 has taken upstream's ideas, as the check that nothing lost; a loss here is worked as in step 8.
The tables' runs follow upstream's text (`--spec-follow`), so they say nothing about the answers XeStrata itself gives: those are step 4's watch of the final HEAD, `$o/md5`, run once the tables are done (one more pass, half an hour or so) on the B70 — the next run's baseline, and what step 8 compares the answers against.

## 8. The check

The previous XeStrata is the newest `xe*` release before this run (step 1). Its records:

- its release record, `<data>/records/bench-results/<date>-release-<xe tag>/` (/release: IQ3_XXS on the generate path, the serve path at 32K and everyday use), for `IQ3_XXS`;
- the last merge's record, `<data>/records/bench-results/<date>-upstream-<version>/<file>`, for the other files (and for `IQ3_XXS` when no release record is newer).

```sh
python .claude/skills/upstream-sync/scripts/judge.py \
  --prev IQ3_XXS=<data>/records/bench-results/<date>-release-<xe tag> --prev Q2_0=<last merge>/Q2_0 ... \
  $o/IQ3_XXS $o/Q2_0 $o/IQ2_XS $o/IQ3_S $o/UD-Q4_K_XL $o/Gyro-M
```

Against the tag, a cell (file, GPU, tier, prompt or output) passes when XeStrata's median is at most 1% below upstream's.
The serve cells (GPU, tier, greedy or t07; prompt, follow-ups, output), read from `matrix-serve.json`, and the everyday cells (GPU, conversation; first token, prompt, output), read from `matrix-usage.json`, pass at most `SERVE_LIMIT` (`judge.py`, 5%) below upstream's, the prompt speed at 1%: each engine writes its own text there, so the other measures vary with the text from round to round (the rounds' seeds differ), up to 10% off a cell's median at temperature 0.7 on 2026-10-10.
A cell upstream does not finish, finishes at 10 tok/s or less, or is stopped as that slow, is left out; one only XeStrata does not finish fails.
A file whose experts do not fit in a machine's RAM budget, so that they are read from the SSD while it answers (SSD offload: UD-Q4_K_XL on the AMD container's 62 GB), is not run on that machine (the user's decision of 2026-10-09): as for GPUs there is no machine for (step 4), its paths are checked by the tests and the builds only, and the record lists its cells `unverified`.
The B70 is compared with upstream's `sycl/` port, as the release's speed table (/release) compares it (the user's decision of 2026-10-18; it replaces the decision of 2026-10-12, which measured XeStrata alone there): its engine is the tag's `sycl/` built in its container image (rootless Podman) and run through `.claude/skills/release/scripts/up-sycl.sh` (references/measure.md). Every GPU is judged against the tag and the previous XeStrata.

Against the previous XeStrata (`prev` lines), every cell both records have is judged: the prompt at 1%, the rest at `SERVE_LIMIT`, since the text changes with the version (on the generate path it is the tag's).
The previous XeStrata is not measured again: its record is the reference (the user's decision of 2026-10-12). A `prev` failure suspected of noise (another cell of the same run moved the other way, or upstream moved as much in the same run) is measured once more in the same setting; one still slower then, or not suspected of noise, is worked at once, the ideas' switches off one at a time with `ab_pair.py` (only the side that changed is measured: a greedy run of the same binary, settings and text is read from its record, AGENTS.md).

For each failing cell: find where the time goes (the GPU profilers of docs/DEVTOOLS.md, the engine's timing lines), read how upstream does that part (for a `prev` failure: what changed since the previous XeStrata, `git log <xe tag>..HEAD`), make XeStrata faster (step 4's rules), and measure that cell again.
The run is done only when a full run of step 7 on the final HEAD, every file and tier of the table (IQ3_XXS with 262K), the serve path, everyday use and every reachable GPU, gives `ALL PASS` with the `--prev` records: a cell fixed earlier can lose again to a later change.

The answers are judged against the previous XeStrata the same way: its record is the last merge's `md5/` (a run before this watch kept none, and then the answers have no previous record), and every cell both records have is `SAME`, `BITS`, `MIXED` or `CHANGED` (step 4).A cell that moved goes into the record's README and docs/XE.md with its KL or its pixels, and a cell whose KL is within its floor is the engine's own noise, not one to report: a release whose answers moved differently from the last one's is a change the user reads about there.
`MIXED` is listed too, as where the engine answers differently from run to run, so that the next run reads that cell knowing it.

## 9. The record and the report

XeStrata's own measurements are not shared (the user's decision of 2026-10-11): the record stays in the data folder, `$o` = `<data>/records/bench-results/<date>-upstream-<version>` (`LOCAL.md`), and never goes into git; upstream's records are not brought in, and `bench/XeStrataBenchResults` holds only the measurements of XeStrata others sent.

- `$o/README.md`: the machines, both engines' commits and builds, the method, the tables (`sync_collect.py`), the check's output (against the tag and the previous XeStrata), what the numbers say, and the GPUs skipped; `matrix.json` and the logs in `runs/` with this PC's paths written as `<repo>` and `<data>`. The ideas with step 4's and 5's numbers (kept, opt-in and removed) go into the same record.
- docs/XE.ja.md, then docs/XE.md: the upstream version and range brought in, what was not taken and why, the approximations on by default and their switches, without a link to the record (it is not in the repository).

Lint, and commit the documents as the run's last commit, with the tag as its second parent and the tree unchanged (as c3aa33ae records v0.1.40.2), so `dev`'s history shows where the run ended:

```sh
git add <the documents> && tree=$(git write-tree) && \
  c=$(git commit-tree -p HEAD -p <tag> -m "upstream の Strata <version> の取り込みを記録する" <tree>) && \
  git merge --ff-only "$c"
```

Its message says the range brought in (`<base>..<tag>` and the tag's commit) and that the commit records the tag as its second parent.

Report: the tag; the ideas, each with what was kept and its numbers; the approximations and their checks; what was not taken, by group; the check's table against the tag and the previous XeStrata; the GPUs skipped (`unverified`); the paths ported for GPUs there is no machine for; and anything that needs the user.
