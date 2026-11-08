---
name: release
description: Make a GitHub release of XeStrata in stages - the version, main fast-forwarded to dev, the tag (xe and the version, e.g. xe0.1.39); first the three contrib-llvm packages, the window's xestrata-gui deb and rpm and the Windows 7z published with the notes (English then Japanese) and the older releases marked as old, then a speed table against upstream on Intel, NVIDIA and AMD GPUs. Only when the user asks for a release (/release).
disable-model-invocation: true
argument-hint: "[VERSION, e.g. 0.1.39 or 0.1.39.1]"
---

# Releasing XeStrata

A release is the annotated tag `xe<VERSION>` on `main`, pushed to origin, and a published GitHub release on it, made in two stages:

- the preliminary release (先行報): the three contrib-llvm packages, the window's `xestrata-gui` deb and rpm, the Windows 7z, `SHA256SUMS` and the notes;
- the final release (確定報): the speed table against upstream on an Intel, an NVIDIA and an AMD GPU added to the notes.

Rules for the whole run:

- Invoking /release is the user's go-ahead for pushing `main` and the tag and for publishing and updating the release at each stage (the user's decision, 2026-10-09): do not stop to ask.
  Stop and report only when the scan of step 7 finds something not clearly public (then push nothing), a package build fails, or an engine of the speed table does not run.
- `gh` only as `gh release ...` (the one exception to "git commands only" in the user's global instructions); everything else with git.
  Every `gh release` names `--repo MistVVK/XeStrata`: without it gh takes the `upstream` remote (Niko1221/Strata).
  Never print or read gh's token (`gh auth status` shows the account; leave its token lines out).
- A release ships the contrib-llvm packages (the user's decision, 2026-10-09), the Windows 7z, a contrib-llvm build for Windows (the user's decision, 2026-10-14), and the Linux window's `xestrata-gui` for Ubuntu 26.04 and Fedora 44 (the user's decision, 2026-11-09; one package for every variant, free included; no Ubuntu 24.04 package). The free packages are built only when the user asks (`build-packages.sh xe<VERSION> free`), and not attached otherwise.
- The check of the packages and the 7z is their build alone (the user's decision, 2026-10-07): no CTest, no incus run, no Windows run, unless the user asks for them.
- Work and commits are on `dev` (Japanese commit messages, one logical change per commit, `git add` and `git commit` joined with `&&`). `dev` itself is never pushed; `main` is.
- A tag that has been pushed is never moved or deleted. A mistake after the push is fixed by a new version (the fourth number).

## 1. Check the starting point

```sh
git status --short --untracked-files=no   # must be empty, on dev
gh auth status 2>&1 | grep -v -i token    # logged in to github.com as MistVVK
podman info --format '{{.Version.Version}}' # the package builds run in rootless Podman
git fetch origin main --tags
git merge-base --is-ancestor main dev     # main must fast-forward to dev
git tag -l 'xe*' --sort=-v:refname | head -n 1; git ls-remote --tags origin 'xe*'
```

Stop and report if a tracked file has changes, gh is not logged in, Podman does not answer, or `main` has commits `dev` lacks.
Untracked files (a record not yet committed, for example) may stay: the packages are built from the tag, and the push takes only commits.
The newest `xe*` tag is the previous release (none before the first one).

## 2. The version

The version is `$ARGUMENTS` when given, else the one `CMakeLists.txt` makes from `STRATA_UPSTREAM_VERSION` and `XESTRATA_REVISION` (AGENTS.md, "Versions"); the tag is `xe<VERSION>` (the name the program shows, `STRATA_VERSION_NAME`).
It must not exist yet, locally or on origin.

- A release after integrating a new upstream version takes that version: the integration changes `STRATA_UPSTREAM_VERSION` and sets `XESTRATA_REVISION` back to 1 (`xe0.1.43.1-1`, the packages `0.1.43.1-1`).
- Another release on the same upstream version raises `XESTRATA_REVISION` by one: `xe0.1.43.1-1`, then `xe0.1.43.1-2` (packages `0.1.43.1-2`). Tags before 2026-10-13 have the earlier form (`xe0.1.40.2.1` was 0.1.40.2's second release; AGENTS.md, "Versions").

When the version changes, change `XESTRATA_REVISION` and every place that shows the version name (`git grep -n 'xe<OLD>'`: docs/XE), Japanese and English files in the same commit, and commit on `dev` (`版を xe<VERSION> にする`).
A new version makes conversations saved by `--conversation-save` under the old one unreadable (the engine version is part of their identity): the notes say so.

## 3. main and the tag

```sh
git fetch . dev:main                      # fast-forwards main to dev without checking it out
git tag -a xe<VERSION> -m "XeStrata xe<VERSION>" main
```

Both stay local until step 8.

## 4. The contrib-llvm packages

```sh
setsid nohup .claude/skills/release/scripts/build-packages.sh xe<VERSION> contrib >/dev/null 2>&1 </dev/null &
```

It builds the three contrib-llvm variants, the Windows 7z and the window's package from the tag one after another (ubuntu26.04 cuda13.1 and cuda12.4, fedora44 cuda13.4, windows cuda12.4: the 7z all cross-built in Ubuntu 26.04; ubuntu26.04 gui and fedora44 gui: a minute each; docs/BUILD.md), each in a clean container (about two hours), and gathers them with `SHA256SUMS` in `build/release/xe<VERSION>/`.
Wait on the PID in `build/release/xe<VERSION>/build-contrib.pid` (`kill -0`) and follow `summary-contrib.txt` (a Monitor that prints its new lines, re-armed every 30 minutes); never wait on `pgrep -f`, which matches the waiting shell.
On `FAILED`, report the end of the variant's log in `logs/`; the fix is an ordinary change on `dev`, after which delete the local tag (`git tag -d xe<VERSION>`) and go back to step 3 (only while the tag has not been pushed).

Steps 5 and 6 are done while it builds.

## 5. The release notes

Write `build/release/xe<VERSION>/NOTES.md`: the English section first, then the same content in Japanese (the user's decision, 2026-10-12), under a first line linking the two: `[English](#english) | [日本語](#日本語)`.

- What changed for users since the previous release tag (`git log --no-merges <previous>..xe<VERSION>`; for the first release, since `origin/main`), grouped by what a user notices, not a list of commits.
- How to install: README's install section, as it is there (the user's decision, 2026-10-11). On Linux two ways, in this order (the user's decision, 2026-10-19): `#### 1. Podman` (README's Podman part: the commands, `export PODMAN_COMPOSE_PROVIDER=podman-compose` among them, then its podman-compose, GPU and host lines), then `#### 2. apt / dnf`: the four numbered steps (download from the table with `SHA256SUMS`, `sha256sum -c SHA256SUMS --ignore-missing`, `sudo apt install ./…deb` or `sudo dnf install ./…rpm` with the release's file names, `xestrata --setup`, which shows the GPU makers' packages that are missing and the command that installs them: AGENTS.md, "Packages", the notes do not list them), then the `> [!NOTE]` that without an NVIDIA GPU either CUDA version works the same, then the package table (Intel, AMD (RDNA2 or later), NVIDIA (Turing or later): `xestrata-contrib-cuda13.1`, `cuda13.4` on Fedora; Intel, AMD (RDNA2 or later), NVIDIA Pascal and Volta: `xestrata-contrib-cuda12.4`), then the window's part (`xestrata-gui`: an extra, installed with `sudo apt install ./xestrata-gui_…deb` or `sudo dnf install ./xestrata-gui-…rpm` beside the main package, the distribution's .NET 10 runtime coming with it; `xestrata gui` opens it), then its Windows part (download `XeStrata-<VERSION>-windows-x64.7z`, start `xestrata.exe`; what it needs, the GPUs, where things go). A change to one goes into the other.
- When the version changed: saved conversations from another version are not read.
- That the release has no free packages: a free build is made from the source (docs/BUILD.md).

Shape the notes with the user's `i-have-adhd` skill (the user's decision, 2026-10-09): read its `SKILL.md` in the user's skills folder (`~/.claude/skills/i-have-adhd/` or `~/.codex/skills/i-have-adhd/`; it cannot be invoked as a tool) and follow its rules for the reader of the notes.
In short, for when it is not there:

- The first lines are what to do: which package to download for which GPU, and the install commands, as a numbered list.
- What changed, as what now works or is faster, in concrete terms (numbers where measured), ranked, at most five items a group.
- No preamble, recap or closing, no tangents; facts only, no plans or alternatives (the stage line excepted).

Each section starts with the warnings below (the user's decision, 2026-10-21; AGENTS.md, "Intel Graphics Compiler"), then a line of the stage: 先行報: upstream との速さの比較は後で足す / Preliminary: the speed comparison with upstream follows.
The first paragraph is the IGC bug, the second the Level Zero driver's: they stay while Ubuntu 26.04 or Debian 13 ships an IGC from 2.27 to 2.30, and while Ubuntu 26.04 ships a compute-runtime older than 26.22; the user decides when they go.
They name Podman, Fedora 44 and - on Ubuntu 24.04/26.04 with the HWE kernel - Intel's graphics PPA (xestrata --setup prints its commands) as the way out.

```markdown
> [!WARNING]
> **Intel Arc B-series, Lunar Lake and Panther Lake on Linux:** the Intel graphics compiler (IGC) 2.27 to 2.30, which Ubuntu 26.04 and Debian 13 ship, miscompiles some GPU code. XeStrata turns the affected optimization off by itself, at a cost of up to 10% in the decode of short prompts. For full speed, install with **Podman** (below; its image brings a fixed IGC), use **Fedora 44**, or - on Ubuntu 24.04/26.04 with the HWE kernel - update the driver from Intel's graphics PPA: `xestrata --setup` prints the commands.
>
> **Every Intel GPU, a separate bug:** a compute-runtime older than 26.22 (Ubuntu 26.04 ships 26.05) can refuse device memory while the card's VRAM is free: the engine then fails to start, and only a reboot brings the card back. This release makes the engine retry the cache in pieces when the driver refuses the one big allocation, and start with what the card gives. The way out is the same: update the driver (26.22 or later), or use Fedora 44 / Podman. On Ubuntu 24.04/26.04 with the HWE kernel, `xestrata --setup` stops before starting and prints the PPA commands; `--old-driver` starts anyway.
```

```markdown
> [!WARNING]
> **Linux の Intel Arc B シリーズ・Lunar Lake・Panther Lake:** Ubuntu 26.04 と Debian 13 が入れる Intel のグラフィックスのコンパイラー（IGC）2.27〜2.30 は、GPU のコードの一部を誤ってコンパイルします。XeStrata は該当する最適化を自動で切るので動きますが、短い prompt の decode が最大 10% 遅くなります。本来の速さで使うには、**Podman** で入れる（下の手順。中の IGC は直った版です）、**Fedora 44** を使う、または Ubuntu 24.04/26.04 で HWE カーネルなら Intel の graphics PPA でドライバーを更新してください（`xestrata --setup` がコマンドを出します）。
>
> **すべての Intel GPU に、別の不具合:** 26.22 より古い compute-runtime（Ubuntu 26.04 は 26.05 を入れます）は、カードの VRAM に空きがあっても GPU メモリの確保を断ることがあります。エンジンは起動に失敗し、カードは再起動しないと戻りません。この版から、エンジンは一括の確保を断られたときに expert cache を小分けで取り直し、確保できた分だけで起動します。抜け道は同じで、ドライバーを 26.22 以降に更新するか、Fedora 44 / Podman を使ってください。Ubuntu 24.04/26.04 で HWE カーネルなら、`xestrata --setup` は起動の前に止まって PPA のコマンドを出します（`--old-driver` でそのまま起動できます）。
```

Show the notes to the user.

## 6. The AMD machine

The AMD GPU is in another machine, so its part of the speed table (step 9) is prepared now; its container (step 9, "The container") is made there as soon as the Fedora 44 rpm is in `build/release/xe<VERSION>/` (the packages are built one after another, so it is there before `ALLDONE`), and it measures while the development machine waits for the packages.
Where the AMD machine is, where its Podman runs, its data folder and a folder with room for the container's work: `LOCAL.md` in the repository root, kept out of git; when it is missing, ask the user and write the answer there.
Copy over now what the container takes besides the rpm: the tag's tree (`git archive xe<VERSION>`, as a tar), upstream's tree at its newest release tag (`git ls-remote --tags --refs upstream 'v*' | sed 's#.*refs/tags/##' | sort -V | tail -n 1`; not `git tag -l 'v*'`, which also lists other remotes' tags (Project Maya's `upstream-maya` has v1.0.x)), and the prompts and continuations (step 9).

## 7. Scan what the push would publish

```sh
git fetch origin main
.claude/skills/release/scripts/prepush-scan.sh origin/main..main
```

It runs gitleaks over each commit's changes and looks in every added line for this PC's paths, user and host names, private IP addresses, e-mail addresses other than the public author address, and for recordings and files over 5 MB.
Judge each finding: third-party license texts, upstream's code and version numbers are public; a real secret or anything of this PC is not.
For anything not clearly public, stop and report it with the commit and file; do not push.

## 8. The preliminary release (先行報)

When `summary-contrib.txt` ends with `ALLDONE`: `build/release/xe<VERSION>/` holds the three contrib-llvm packages (two `.deb`, one `.rpm`), the two `xestrata-gui` packages (a `.deb`, an `.rpm`), the Windows `.7z` and `SHA256SUMS`, and `(cd build/release/xe<VERSION> && sha256sum -c SHA256SUMS)` passes.

```sh
git push origin main
git push origin xe<VERSION>
d=build/release/xe<VERSION>
gh release create xe<VERSION> --repo MistVVK/XeStrata --verify-tag --title "XeStrata xe<VERSION>" --notes-file "$d/NOTES.md" \
  "$d"/*.deb "$d"/*.rpm "$d"/*.7z "$d/SHA256SUMS"
gh release view xe<VERSION> --repo MistVVK/XeStrata --json url,isDraft
```

Retry a push only for a network failure (up to four times, waiting 2, 4, 8, 16 s).
`--verify-tag` makes gh refuse rather than create a tag of its own.
Report the release's URL, then make the development machine's container for step 9.
A wrong asset or note is fixed with `gh release upload --clobber`, `gh release delete-asset` or `gh release edit --notes-file`.

### The older releases

Once the new release is published (it is then the latest), mark every older one:

```sh
gh release list --repo MistVVK/XeStrata --limit 1000 --json tagName,isLatest -q '.[] | select(.isLatest | not) | .tagName' |
  while read -r t; do .claude/skills/release/scripts/mark-old.sh "$t"; done
```

`mark-old.sh` puts a warning at the top of a release's notes, in English and Japanese, that sends the reader to the latest release (`releases/latest`, so a note put in before stays right); a release that has it already is left as it is, so in practice only the previous release changes.

## 9. The speed table

The release against upstream's newest release tag, IQ3_XXS, on each maker's GPU, on Linux (the Windows 7z is not measured): the Intel Arc Pro B70 and the RTX 4070 in the development machine, the RX 9060 XT in the AMD machine (step 6).
The six tiers and the settings of the 2026-10-04-speed-matrix record (1K to 262K, what setup writes for each context, 256 greedy tokens), with one change (the user's decision of 2026-10-12): KV streaming (`--kv-resident`) only where setup writes it on the machine, from 64K when the RAM holds the KV cache beside the model, so a machine short of that RAM (the AMD machine: `LOCAL.md`) does not compare at those lengths with the releases up to xe0.1.42; its `prompts.py`, now `.claude/skills/release/scripts/prompts.py`, writes the prompts (`python .claude/skills/release/scripts/prompts.py <pack> <ids folder>`).
From 2026-10-16 (the user's decision) both engines follow the same continuation of each prompt (`--spec-follow <ids folder>/<tier>.follow`): an engine's own greedy text splits from another's within tens of tokens, and the drafts' acceptance on the text it splits into moved decode by up to 20%, so a table of each engine's own text measured which text it fell into.
The continuations are written once, with the prompts, by `scripts/follow.sh` (`XE=... DATA=<data> IDS=<ids folder> XE_PROFILE=... follow.sh`: XeStrata's 256 greedy tokens with the BF16 hyper-connection read) and kept, so every release compares on the same text; copy them with the prompts to the AMD machine. `bench.sh` refuses to run without them.
The tables of releases before xe0.1.44.1 timed each engine's own text and do not compare with later ones in decode. The serve and everyday tables below keep each engine's own text (`--spec-follow` is not for `--serve`).
The table's method stays fixed from release to release so the releases compare: the upstream merge's measurements (/upstream-sync) have scripts of their own, and a change to them does not change this table.

### The container

Every table of this step runs in a disposable Fedora 44 container of rootless Podman (the user's decision of 2026-11-05), one on each machine, made for the release by `scripts/podman-bench.sh` and removed with its images and its work folder once all three GPUs' tables are done (step 10), or as soon as something fails (made again from the start after the fix):

- XeStrata: the release's Fedora 44 rpm (`build/release/xe<VERSION>/xestrata-contrib-cuda13.4-*.rpm`) installed by `container/` as a user's Podman install gets it, with the GPU makers' libraries its `BUILD.json` records: the engine `/usr/lib64/xestrata/engine/xestrata-engine`, its tree (`serve/`, `setup.py`) `/usr/share/xestrata`, its profile `/usr/share/xestrata/data/expert-profile.bin`.
  One GPU at a time with `ONEAPI_DEVICE_SELECTOR` (`level_zero:*` for the B70, `cuda:*` for the RTX 4070).
- upstream: a copy of its newest release tag's tree, built in the same container (`scripts/podman/up_build.py`) by its own setup.py's functions: `build_engine` (CUDA, with NVIDIA's toolkit from its Fedora 44 repository) or `build_engine_hip` (HIP, with the development packages of the ROCm the image runs), and `build_image_engine` for its `sd-cli` (CUDA, HIP, Vulkan on Intel).
  On Intel its `sycl/` port, built by its `sycl/tools/build.sh` with the oneAPI DPC++ its docs/INTEL.md names (from Intel's repository; `AOT=<device>`: `LOCAL.md`) and run through `scripts/up-sycl.sh` with `UP_SYCL_DIRECT=1`: without its own image, but as its setup configures an Intel card (the arguments get what its `sycl/setup_intel.py` writes: the xe cards' expert profile, the streamed experts, KV streaming from 64K, the VRAM reserve, the prompt chunks; the engine the run-time switches of its `sycl/serve/strata-sycl.sh`).
- ComfyUI: its checkout's code copied in, its `models/` read where it is, and a venv in the container with the packages of the machine's venv for the GPU at the same versions (PyTorch's from PyTorch's index of its build).

Why one container for all three, not upstream's own `sycl/` image or the host's ComfyUI: the tables compare the engines, so they run on the same OS with the same GPU drivers' user space (Level Zero and the IGC, ROCm, the CUDA runtime; NVIDIA's driver libraries are the host's through CDI, the same for all three).
Upstream's `sycl/` image is another OS with its own IGC and compute-runtime; the tables of the releases before this change ran XeStrata and ComfyUI on each machine's own drivers, and do not compare with these one to one.

The machine's facts the commands take (`LOCAL.md`): `<data>`, a folder on a disk with room for the work (about 60 GB: the images' build, upstream's builds, ComfyUI's venvs), ComfyUI's checkout and its venv for each GPU, upstream's `AOT` device for the B70, and on the AMD machine where its Podman runs.
Set the machine's measuring mark (`LOCAL.md`) before `image` and remove it after `clean`; no other work on a machine while it measures (no build, no other engine).

```sh
o=<data>/records/bench-results/<date>-release-xe<VERSION>   # outside git (the user's decision of 2026-10-11)
s=<tag tree>/.claude/skills/release/scripts                 # the tag's tree: a worktree of xe<VERSION>, or its git archive on the AMD machine
export ID=xe<VERSION> WORK=<work>/xestrata-bench-xe<VERSION> MAKERS="intel nvidia"   # the AMD machine: MAKERS=amd
$s/podman-bench.sh image build/release/xe<VERSION>/xestrata-contrib-cuda13.4-*.rpm
DATA=<data> OUT=$o COMFY=<ComfyUI> MOUNTS="<ids> <ids-serve>" $s/podman-bench.sh start
$s/podman-bench.sh xe                                                # prints XE=... XE_PROFILE=... (also $WORK/XE.env)
AOT=<device> $s/podman-bench.sh upstream <upstream's tree> intel      # prints UP=... UP_PROFILE=... (also $WORK/upstream-intel/UP.env)
$s/podman-bench.sh upstream <upstream's tree> nvidia
COMFY=<ComfyUI> $s/podman-bench.sh comfy <its venv for the GPU>     # each GPU's
$s/podman-bench.sh run env XE=... UP=... ... bash $s/bench.sh $o B70   # each of the tables below, so
$s/podman-bench.sh clean                                             # after the last GPU, or after a failure
```

In the container, `XE`, `XE_PROFILE`, `XE_ROOT` and `XE_LIB_DIRS` as `xe` printed them (the installed engine `/usr/lib64/xestrata/engine/xestrata-engine` with the library folders setup gives it, oneMKL's among them: without them oneMath falls back to XeStrata's own kernels on Intel), and `PY=python3`; `UP`, `UP_PROFILE`, `UP_ROOT` (and `UP_TREE`, `UP_SYCL_DIRECT` for the sycl/ port) as `upstream` printed them, each maker's in its own `$WORK/upstream-<maker>/`.
`run` starts the command in the running container (`podman exec`): run each table's command through it, in the background as before (`setsid nohup ... &`), and follow its output; a run stopped from outside is stopped by `clean`, not by killing the `podman exec`.

```sh
$s/podman-bench.sh run env XE=... UP=... XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> bash $s/bench.sh $o B70
python $s/bench-collect.py $o xe<VERSION> <upstream tag>
```

`bench.sh` runs a warm-up per engine, then each tier twice per engine with the engines alternating; `bench-collect.py` writes `matrix.json` and the tables (the medians), one pair per GPU.
A cell that does not fit the GPU's or the machine's memory stays `-` with the reason in the record; an engine that does not start at all stops the run (report it).
`bench.sh` stops upstream once it is plainly at 10 tok/s or less; such a cell is `too slow`: not measurable, written 計測不能（低速）/ not measurable (too slow) in the notes, with a line saying upstream was stopped at 10 tok/s or less.

### serve and everyday use

Beside that table, two smaller ones on the same GPUs and engines, closer to what a user meets, by the upstream merge's scripts (`tools/bench_core.py`), cut down to about 40 minutes a GPU (the AMD machine measures meanwhile): `xestrata-engine --serve` at 32K, sampled at temperature 0.7, and the engines' own `serve/server.py` with the Japanese chat and the coding agent; two rounds each, the engines alternating, each engine writing its own text.
They go into `$o/serve/`, apart from the table's `matrix.json`; the prompts come from the merge's `sync_prompts.py` (its 32K is the table's 32K, and it writes the follow-up questions).

```sh
u=<tag tree>/.claude/skills/upstream-sync/scripts; r="$s/podman-bench.sh run env"
python $u/sync_prompts.py <pack> <ids-serve>                                   # once
$r XE=... UP=... XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids-serve> TIERS=32k MODES=t07 ROUNDS=2 \
  python3 $u/bench_serve.py $o/serve <GPU>
$r XE=... UP=... XE_ROOT=/usr/share/xestrata UP_ROOT=... XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=python3 \
  ROUNDS=2 CONVS="chat-ja agent" python3 $u/bench_usage.py $o/serve <GPU>
python $u/sync_collect.py $o/serve xe<VERSION> <upstream tag>                 # matrix-serve.json, matrix-usage.json
```

`XE` and `UP` are as above, and so are `XE_ROOT` and `UP_ROOT` (their `serve/server.py`); everyday use runs at a 262144 context with what setup writes for it.

### Gyro-M

And one more on the same GPUs and engines (the user's decision of 2026-10-22): agentionai's Gyro-M (the experts' trellis type TQ2_T), generate at 32K and 1K by the upstream merge's `bench_generate.py`, two rounds, the engines alternating and both following one text (`--spec-follow`, upstream's own first run); upstream reads its own pack through the merge's `up-pack.sh` (its references/measure.md, the files table).

```sh
G=<data>/models/Gyro/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf
$r XE=... UP="$u/up-pack.sh" UP_REAL="<upstream's command, as above>" XE_PROFILE=... UP_PROFILE=... DATA=<data> \
  IDS=<ids-serve> PACK=<data>/packs/gyro-m UP_PACK=<upstream's Gyro-M pack> NATIVE=$G PLE=$G TIERS="32k 1k" ROUNDS=2 \
  python3 $u/bench_generate.py $o/gyro-m <GPU>
python $u/sync_collect.py $o/gyro-m xe<VERSION> <upstream tag>                 # matrix.json, the tables
```

### image generation

And one more on the same GPUs (the user's decision of 2026-10-15; ComfyUI and Mage-Flow-Turbo from 2026-10-23): each engine's image generator as its own setup sets it up on the GPU, one 1024x1024 image from a fixed prompt and seed, of two models: Qwen-Image-2.1-Turbo (8 steps) and Mage-Flow-Turbo (4 steps).

- XeStrata and upstream, by the upstream merge's `bench_image.py` (its references/measure.md, "Image generation": XeStrata's resident engine measured warm, upstream's backend, how its `sd-cli` is built in its tree, the GPUs); two images per engine, alternating, after a warm-up each. upstream has Qwen-Image-2.1-Turbo alone (its `tools/image/qimage.py`): Mage-Flow-Turbo is XeStrata's alone (`IMAGE_MODEL=mage-flow-turbo UP_ROOT=none`), upstream's cell written as not in upstream.
- ComfyUI, by `bench_comfy.py` beside it, run after them: each image a server started, a warm-up image of another prompt and the measured one, as XeStrata's resident engine is measured. Its DiT as int8_convrot and, where ComfyUI has one (Mage-Flow-Turbo), as bf16, an arm each; its own text encoders (Qwen3-VL-8B int8_convrot, Qwen3-VL-4B bf16). Its copy and venvs in the container (`podman-bench.sh comfy`, "The container"); the commit is `$WORK/ComfyUI/COMMIT`. It prints no VAE time.
- About 15 minutes a GPU.

```sh
x="XE_ROOT=/usr/share/xestrata XE_IMAGE_EXE=/usr/lib64/xestrata/engine/xestrata-image XE_LIB_DIRS=$XE_LIB_DIRS"
$r $x UP_ROOT=... DATA=<data> QUANT=<Q8_0, Q6_K on a 12 GB card> UP_IMAGE_BACKEND=... \
  [XE_IMAGE_PCI=<PCI address> UP_IMAGE_GPU=N] PY=python3 ROUNDS=2 python3 $u/bench_image.py $o/image <GPU>
$r IMAGE_MODEL=mage-flow-turbo UP_ROOT=none $x DATA=<data> QUANT=Q8_0 [XE_IMAGE_PCI=<PCI address>] PY=python3 \
  ROUNDS=2 python3 $u/bench_image.py $o/image-mage <GPU>
$r COMFY=$WORK/ComfyUI VENV=<its venv for the GPU> [ONEAPI_DEVICE_SELECTOR=... | CUDA_VISIBLE_DEVICES=...] ROUNDS=2 \
  python3 $u/bench_comfy.py $o/image <GPU>
$r IMAGE_MODEL=mage-flow-turbo COMFY=$WORK/ComfyUI VENV=<...> ROUNDS=2 python3 $u/bench_comfy.py $o/image-mage <GPU>
python $u/sync_collect.py $o/image xe<VERSION> <upstream tag>                 # matrix-image.json, both in each
python $u/sync_collect.py $o/image-mage xe<VERSION> <upstream tag>
```

The record goes into `$o/`, in the data folder and never into git (XeStrata's own measurements are not shared, the user's decision of 2026-10-11): a README (the machines, the container (its Fedora 44 and the versions of the GPU libraries in it: the IGC, the compute-runtime, ROCm, the CUDA toolkit), the two engines' commits and builds, the method, the tables and what the numbers say), `matrix.json`, `serve/` (its `matrix-serve.json`, `matrix-usage.json` and runs), `gyro-m/` (its `matrix.json` and runs), `image/` and `image-mage/` (their `matrix-image.json`, runs and images, ComfyUI's commit), and the logs in `runs/` with this PC's paths written as `<repo>` and `<data>` (as the speed-matrix record does).
It is not committed; the release notes carry the tables.

## 10. The final release (確定報)

The final release is reached a GPU at a time (the user's decision, 2026-10-11): as soon as one GPU's tables are complete (its generate table, its serve and everyday table, its Gyro-M table and its image table), add them to the notes (English and Japanese sections) and update the release (`gh release edit`, below), without waiting for the other GPUs.

- The comparison section comes after the notes' other sections: its opening lines (the engines' versions, the model, the method), then the note on temperatures and what counts as equal (below) right after them, once, then the GPUs in a fixed order, the B70, the RTX 4070, the RX 9060 XT, each as its tables are done.
- While some are missing, one line under the done ones names them: B70・RTX 4070: 計測中（後で足す） / B70, RTX 4070: being measured (added later).
- The first update replaces the preliminary stage line with that section; when the last GPU is in, the line of the missing GPUs goes too.
- When all three GPUs' tables are in, remove the containers on both machines (`podman-bench.sh clean`, which ends with `clean: ... removed` and lists anything left), then the tag's tree each used (the worktree with `git worktree remove`, the archive on the AMD machine).
- Report the release's URL and the GPU's tables at each update.

The tables:

The notes' table is one per GPU, from `bench-collect.py`'s medians (the record's README keeps `bench-collect.py`'s own tables): a row per tier, named by the model's weight type and the length (`IQ3_XXS 1K`), and for prefill and then decode three columns (the LLM terms, in both languages; the user's decision, 2026-10-11): XeStrata's value, upstream's, and the difference (the value columns named with the engine and its version, `XeStrata xe0.1.43.1-1`, `upstream v0.1.41`), marked ✅ where XeStrata is faster and ❌ where it is slower: prefill as a ratio (`✅ 2.1 倍`, `✅ ×2.1`), decode as a percentage (`✅ +14%`; slower: `❌ −3%`).
Gyro-M's rows, from `gyro-m/`'s medians, come after IQ3_XXS's in the same generate table (`Gyro-M 32K`, `Gyro-M 1K`).
Within ±1% (either way) the two count as equal: `≒ 同等（−0.1%）` / `≈ equal (−0.1%)`, for prefill as well.
A value upstream was stopped in is 計測不能（低速） / not measurable (too slow), its difference `✅`, with the line saying upstream was stopped at 10 tok/s or less; a value that does not fit stays `-`.
Above each GPU's tables: the GPU (and its memory) and which build of upstream it ran (`sycl/`, CUDA, HIP).
Each table has a heading line with its temperature: **generate**（温度 0、トークン/秒） / **generate** (temperature 0, tokens/s), and **serve と通常使用**（温度 0.7） / **serve and everyday use** (temperature 0.7).

```markdown
**AMD Radeon RX 9060 XT (16 GB)**: upstream v0.1.41 は HIP 版

**generate**（温度 0、トークン/秒）

| 形式・長さ | prefill XeStrata xe0.1.43.1-1 | prefill upstream v0.1.41 | 差 | decode XeStrata xe0.1.43.1-1 | decode upstream v0.1.41 | 差 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| IQ3_XXS 1K | 509 | 295 | ✅ 1.7 倍 | 48.8 | 46.8 | ✅ +4% |
```

Below each GPU's table, its serve and everyday table from `serve/`'s medians, with the same columns (XeStrata, upstream, difference): a row per measure, prefill as a ratio, the rest as percentages, the wait and the first token better when shorter.
Equal there is the upstream merge's check (`judge.py`): prefill within ±1%, the rest within ±5%, both engines writing their own texts. A wait or a first token is judged as a speed, upstream's time over XeStrata's (372 ms against 354 ms: −4.8%, equal), and its column shows the time's own difference: `≒ 同等（+5%）` / `≈ equal (+5%)`.

```markdown
**serve と通常使用**（温度 0.7）

| 計測 | XeStrata xe0.1.43.1-1 | upstream v0.1.41 | 差 |
| --- | ---: | ---: | --- |
| serve 32K 最初の prefill (tok/s) | 839 | 380 | ✅ 2.2 倍 |
| serve 32K 会話の続きの待ち (ms) | 372 | 354 | ≒ 同等（+5%） |
| serve 32K decode (tok/s) | 45.3 | 40.3 | ✅ +12% |
| chat-ja 最初のトークン (s) | 0.68 | 1.05 | ✅ −35% |
| chat-ja decode (tok/s) | 42.8 | 41.0 | ≒ 同等（+4%） |
| agent 最初のトークン (s) | 5.74 | 11.97 | ✅ −52% |
| agent prefill (tok/s) | 691 | 334 | ✅ 2.1 倍 |
| agent decode (tok/s) | 46.2 | 45.0 | ≒ 同等（+3%） |
```

Below that, its image table from `image/`'s and `image-mage/`'s medians: a row per model and engine (XeStrata, upstream, ComfyUI's int8 and bf16), with the seconds of an image, of its sampling and of its VAE, which build each engine ran (its backend and the weights), and the difference of the image's seconds against XeStrata's as a ratio of the times, the other's over XeStrata's (`✅ 1.8 倍` / `✅ ×1.8` where XeStrata is faster; slower: `❌ 0.93 倍` / `❌ ×0.93`), within ±1% equal.
A time that was not measured is `-` (ComfyUI's VAE); upstream's Mage-Flow-Turbo is upstream にない / not in upstream.
Only upstream decides whether the release goes out (a release slower than upstream is not published): ComfyUI's rows are shown as they are.

```markdown
**画像の生成**（1024×1024、秒）

| モデル | エンジン | 1 枚 | サンプリング | VAE | XeStrata との差（1 枚） |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen-Image-2.1-Turbo（8 ステップ） | XeStrata xe0.2.2-1（Q8_0） | 7.7 | 6.9 | 0.4 | |
| | upstream v0.2.2（Vulkan、Q8_0） | 51.6 | 46.2 | 3.0 | ✅ 6.7 倍 |
| | ComfyUI 7a28332（int8） | 9.7 | 8.1 | - | ✅ 1.3 倍 |
| Mage-Flow-Turbo（4 ステップ） | XeStrata xe0.2.2-1（FP16） | 1.9 | 1.7 | 0.1 | |
| | upstream v0.2.2 | upstream にない | | | |
| | ComfyUI 7a28332（int8） | 1.9 | 1.5 | - | ≒ 同等（+0.5%） |
| | ComfyUI 7a28332（bf16） | 0.8 | 0.5 | - | ❌ 0.42 倍 |
```

Right after the section's opening lines, once (not repeated for each GPU), one note in each language section, with the temperatures and what counts as equal:

```markdown
> [!NOTE]
> - **generate**：温度 0（greedy）。差が ±1% 以内なら ≒ 同等です。
> - **serve と通常使用**：温度 0.7 で、それぞれのエンジンが自分の文を書くので、1 回ごとに中央値から 10% 前後ずれることがあります。差が prefill は ±1% 以内、待ち・decode・最初のトークンは ±5% 以内なら ≒ 同等です。
> - **画像の生成**：決まったプロンプトと種で 1 枚ずつ作った時間です。XeStrata と upstream はそれぞれの setup がその GPU に選ぶ作り方で、ComfyUI は起動して 1 枚作った後の 2 枚目です。差が ±1% 以内なら ≒ 同等です。
> - どれも、交互に 2 回ずつ測った中央値です。
```

```markdown
> [!NOTE]
> - **generate**: temperature 0 (greedy). Within ±1% counts as ≈ equal.
> - **serve and everyday use**: temperature 0.7, each engine writing its own text, so a run can land about 10% off the median. Within ±1% for prefill and ±5% for the wait, decode and first token counts as ≈ equal.
> - **image generation**: the time of an image from a fixed prompt and seed: XeStrata and upstream as their own setups set them up on the GPU, ComfyUI the second image after its start. Within ±1% counts as ≈ equal.
> - All are the median of two alternating runs.
```

```sh
gh release edit xe<VERSION> --repo MistVVK/XeStrata --notes-file build/release/xe<VERSION>/NOTES.md
```
