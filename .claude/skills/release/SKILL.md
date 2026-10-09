---
name: release
description: Make a GitHub release of XeStrata in stages - the version, main fast-forwarded to dev, the tag (xe and the version, e.g. xe0.1.39); first the three contrib-llvm packages published with the notes (Japanese then English), then a speed table against upstream on Intel, NVIDIA and AMD GPUs. Only when the user asks for a release (/release).
disable-model-invocation: true
argument-hint: "[VERSION, e.g. 0.1.39 or 0.1.39.1]"
---

# Releasing XeStrata

A release is the annotated tag `xe<VERSION>` on `main`, pushed to origin, and a published GitHub release on it, made in two stages:

- the preliminary release (先行報): the three contrib-llvm packages, `SHA256SUMS` and the notes;
- the final release (確定報): the speed table against upstream on an Intel, an NVIDIA and an AMD GPU added to the notes.

Rules for the whole run:

- Invoking /release is the user's go-ahead for pushing `main` and the tag and for publishing and updating the release at each stage (the user's decision, 2026-10-09): do not stop to ask.
  Stop and report only when the scan of step 7 finds something not clearly public (then push nothing), a package build fails, or an engine of the speed table does not run.
- `gh` only as `gh release ...` (the one exception to "git commands only" in the user's global instructions); everything else with git.
  Every `gh release` names `--repo MistVVK/XeStrata`: without it gh takes the `upstream` remote (Niko1221/Strata).
  Never print or read gh's token (`gh auth status` shows the account; leave its token lines out).
- A release ships the contrib-llvm packages only (the user's decision, 2026-10-09). The free packages are treated like contrib-icpx: built only when the user asks (`build-packages.sh xe<VERSION> free`), and not attached otherwise.
- The check of the packages is their build alone (the user's decision, 2026-10-07): no CTest, no incus run, unless the user asks for them.
- Work and commits are on `dev` (Japanese commit messages, one logical change per commit, `git add` and `git commit` joined with `&&`). `dev` itself is never pushed; `main` is.
- A tag that has been pushed is never moved or deleted. A mistake after the push is fixed by a new version (the fourth number).

## 1. Check the starting point

```sh
git status --short                        # must be empty, on dev
gh auth status 2>&1 | grep -v -i token    # logged in to github.com as MistVVK
docker info --format '{{.ServerVersion}}' # the package builds run in Docker
git fetch origin main --tags
git merge-base --is-ancestor main dev     # main must fast-forward to dev
git tag -l 'xe*' --sort=-v:refname | head -n 1; git ls-remote --tags origin 'xe*'
```

Stop and report if the tree is not clean, gh is not logged in, Docker does not answer, or `main` has commits `dev` lacks.
The newest `xe*` tag is the previous release (none before the first one).

## 2. The version

The version is `$ARGUMENTS` when given, else `project(XeStrata VERSION ...)` in `CMakeLists.txt`; the tag is `xe<VERSION>` (the name the program shows, `STRATA_VERSION_NAME`).
It must not exist yet, locally or on origin.

- A release after integrating a new upstream version takes that version (`STRATA_UPSTREAM_VERSION` changes with the integration, not here).
- Another release on the same upstream version adds a fourth number: `0.1.39` then `0.1.39.1`, `0.1.39.2`.

When the version changes, change `project(XeStrata VERSION ...)` and every place that shows the version name (`git grep -n 'xe<OLD>'`; commit `1ae5e9f7` lists them: README, docs/XE, setup.py, cmake/BUILD.json.in), Japanese and English files in the same commit, and commit on `dev` (`版を xe<VERSION> にする`).
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

It builds the three contrib-llvm variants from the tag one after another (ubuntu26.04 cuda13.1 and cuda12.4, fedora44 cuda13.4), each in a clean container (about an hour and a half), and gathers them with `SHA256SUMS` in `build/release/xe<VERSION>/`.
Wait on the PID in `build/release/xe<VERSION>/build-contrib.pid` (`kill -0`) and follow `summary-contrib.txt` (a Monitor that prints its new lines, re-armed every 30 minutes); never wait on `pgrep -f`, which matches the waiting shell.
On `FAILED`, report the end of the variant's log in `logs/`; the fix is an ordinary change on `dev`, after which delete the local tag (`git tag -d xe<VERSION>`) and go back to step 3 (only while the tag has not been pushed).

Steps 5 and 6 are done while it builds.

## 5. The release notes

Write `build/release/xe<VERSION>/NOTES.md`: the Japanese section first, then the same content in English.

- What changed for users since the previous release tag (`git log --no-merges <previous>..xe<VERSION>`; for the first release, since `origin/main`), grouped by what a user notices, not a list of commits.
- Which package to install: the table in README's install section (`xestrata-contrib-cuda13.1` / `cuda13.4` on Fedora; `xestrata-contrib-cuda12.4` for Volta), and that a contrib-llvm package only suggests the GPU makers' runtimes (Intel's Level Zero, with `libigc2 libigdfcl2` on Ubuntu, is installed by hand for an Intel GPU).
- When the version changed: saved conversations from another version are not read.
- How to check a download: `sha256sum -c SHA256SUMS`.
- That the release has no free packages: a free build is made from the source (docs/BUILD.md).

Shape the notes with the user's `i-have-adhd` skill (the user's decision, 2026-10-09): read its `SKILL.md` in the user's skills folder (`~/.claude/skills/i-have-adhd/` or `~/.codex/skills/i-have-adhd/`; it cannot be invoked as a tool) and follow its rules for the reader of the notes.
In short, for when it is not there:

- The first lines are what to do: which package to download for which GPU, and the install commands, as a numbered list.
- What changed, as what now works or is faster, in concrete terms (numbers where measured), ranked, at most five items a group.
- No preamble, recap or closing, no tangents; facts only, no plans or alternatives (the stage line excepted).

Each section starts with a line of the stage: 先行報: upstream との速さの比較は後で足す / Preliminary: the speed comparison with upstream follows.
Show the notes to the user.

## 6. The AMD engines

The AMD GPU is in another machine, so its part of the speed table (step 9) is prepared now and can run while the packages build.
Prepare the two engines in the AMD development container (where it is and how to run in it: `LOCAL.md` in the repository root, kept out of git; when it is missing, ask the user and write the answer there):

- XeStrata: bring the tag into the container's clone (it has no access to origin: a `git bundle` of `xe<VERSION>` copied in) and build the contrib-llvm mode with HIP for its GPU (`STRATA_HIP_ARCHS`, docs/BUILD.md) in a folder of the tag's own.
- upstream: its newest release tag (`git fetch upstream --tags; git tag -l 'v*' --sort=-v:refname | head -n 1`), built with HIP as its own documents say.

Run step 9's script there as soon as both are built; the development machine's GPUs wait until the packages are built (step 8).

## 7. Scan what the push would publish

```sh
git fetch origin main
.claude/skills/release/scripts/prepush-scan.sh origin/main..main
```

It runs gitleaks over each commit's changes and looks in every added line for this PC's paths, user and host names, private IP addresses, e-mail addresses other than the public author address, and for recordings and files over 5 MB.
Judge each finding: third-party license texts, upstream's code and version numbers are public; a real secret or anything of this PC is not.
For anything not clearly public, stop and report it with the commit and file; do not push.

## 8. The preliminary release (先行報)

When `summary-contrib.txt` ends with `ALLDONE`: `build/release/xe<VERSION>/` holds the three contrib-llvm packages (two `.deb`, one `.rpm`) and `SHA256SUMS`, and `(cd build/release/xe<VERSION> && sha256sum -c SHA256SUMS)` passes.

```sh
git push origin main
git push origin xe<VERSION>
d=build/release/xe<VERSION>
gh release create xe<VERSION> --repo MistVVK/XeStrata --verify-tag --title "XeStrata xe<VERSION>" --notes-file "$d/NOTES.md" \
  "$d"/*.deb "$d"/*.rpm "$d/SHA256SUMS"
gh release view xe<VERSION> --repo MistVVK/XeStrata --json url,isDraft
```

Retry a push only for a network failure (up to four times, waiting 2, 4, 8, 16 s).
`--verify-tag` makes gh refuse rather than create a tag of its own.
Report the release's URL, then prepare the development machine's engines for step 9.
A wrong asset or note is fixed with `gh release upload --clobber`, `gh release delete-asset` or `gh release edit --notes-file`.

## 9. The speed table

The release against upstream's newest release tag, IQ3_XXS, on each maker's GPU: the Intel Arc Pro B70 and the RTX 4070 in the development machine, the RX 9060 XT in the AMD container (step 6).
The tiers and the settings of `bench/results/2026-10-04-speed-matrix` and a 192K tier (1K to 262K, what setup writes for each context, 256 greedy tokens); `scripts/prompts.py` writes the prompts (`python .claude/skills/release/scripts/prompts.py <pack> <ids folder>`).

The engines, built from the tags (no package installed), each in a worktree of its own under `.claude/worktrees/`:

- XeStrata: the contrib-llvm build of `xe<VERSION>` (docs/BUILD.md, with `LD_LIBRARY_PATH` to intel/llvm's `lib`); one GPU at a time with `ONEAPI_DEVICE_SELECTOR` (`level_zero:*` for the B70, `cuda:*` for the RTX 4070).
- upstream on the B70: its `sycl/` port, built and run in its Docker image as its docs/INTEL.md says.
- upstream on the RTX 4070: its CUDA build, as its documents say.

Each engine's own shipped expert profile (`data/expert-profile.bin` of its tree).
No other work on a machine while it measures (no build, no other engine).

```sh
o=bench/results/<date>-release-xe<VERSION>
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> .claude/skills/release/scripts/bench.sh $o B70   # <data>: LOCAL.md
python .claude/skills/release/scripts/bench-collect.py $o xe<VERSION> <upstream tag>
```

`bench.sh` runs a warm-up per engine, then each tier twice per engine with the engines alternating; `bench-collect.py` writes `matrix.json` and the tables (the medians), one pair per GPU.
A cell that does not fit the GPU's or the machine's memory stays `-` with the reason in the record; an engine that does not start at all stops the run (report it).
`bench.sh` stops upstream once it is plainly at 10 tok/s or less; such a cell is `too slow`: not measurable, written 計測不能（低速）/ not measurable (too slow) in the notes, with a line saying upstream was stopped at 10 tok/s or less.

The record goes into `$o/` on `dev`: a README (the machines, the two engines' commits and builds, the method, the tables and what the numbers say), `matrix.json`, and the logs in `runs/` with this PC's paths written as `<repo>` and `<data>` (as the speed-matrix record does).
Lint and commit it on `dev` (it is not in the tag).

## 10. The final release (確定報)

Take the stage lines out of the notes, add the tables (Japanese and English sections) with the engines' versions, the GPUs and the model, and:

```sh
gh release edit xe<VERSION> --repo MistVVK/XeStrata --notes-file build/release/xe<VERSION>/NOTES.md
```

Report the release's URL and the table.
