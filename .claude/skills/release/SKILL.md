---
name: release
description: Make a GitHub release of XeStrata - the version, main fast-forwarded to dev, the tag (xe and the version, e.g. xe0.1.39), the five deb and rpm packages with SHA256SUMS, release notes in Japanese then English, and a draft release through gh. Only when the user asks for a release (/release).
disable-model-invocation: true
argument-hint: "[VERSION, e.g. 0.1.39 or 0.1.39.1]"
---

# Releasing XeStrata

A release is the annotated tag `xe<VERSION>` on `main`, pushed to origin, and a **draft** GitHub release on it with the release notes and the five packages plus `SHA256SUMS`.
Taking the draft out (publishing it) is the user's: they do it on the web page, or ask for `gh release edit xe<VERSION> --draft=false`.

Rules for the whole run:

- `gh` only as `gh release ...` (the one exception to "git commands only" in the user's global instructions); everything else with git.
  Never print or read gh's token (`gh auth status` shows the account; leave its token lines out).
- Pushing `main` and the tag, and creating the release, are outward-facing: do them only after the user's explicit go-ahead at step 8 of this run.
- The check is the package build alone (the user's decision, 2026-10-07): no CTest, no incus run, unless the user asks for them.
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

Both stay local until step 9.

## 4. The packages

```sh
setsid nohup .claude/skills/release/scripts/build-packages.sh xe<VERSION> >/dev/null 2>&1 </dev/null &
```

It builds the five variants from the tag one after another, each in a clean container (about two and a half hours), and gathers them with `SHA256SUMS` in `build/release/xe<VERSION>/`.
Wait on the PID in `build/release/xe<VERSION>/build.pid` (`kill -0`) and follow `summary.txt` (a Monitor that prints its new lines, re-armed every 30 minutes); never wait on `pgrep -f`, which matches the waiting shell.
On `FAILED`, report the end of the variant's log in `logs/`; the fix is an ordinary change on `dev`, after which delete the local tag (`git tag -d xe<VERSION>`) and go back to step 3.

## 5. The release notes

Write `build/release/xe<VERSION>/NOTES.md`: the Japanese section first, then the same content in English.

- What changed for users since the previous release tag (`git log --no-merges <previous>..xe<VERSION>`; for the first release, since `origin/main`), grouped by what a user notices, not a list of commits.
- Which package to install: the table in README's install section (`xestrata-free`; `xestrata-contrib-cuda13.1` / `cuda13.4` on Fedora; `xestrata-contrib-cuda12.4` for Volta), and that a contrib package only suggests the GPU makers' runtimes (Intel's Level Zero, with `libigc2 libigdfcl2` on Ubuntu, is installed by hand for an Intel GPU).
- When the version changed: saved conversations from another version are not read.
- How to check a download: `sha256sum -c SHA256SUMS`.

Facts only, no plans or alternatives. Show the notes to the user.

## 6. Scan what the push would publish

```sh
git fetch origin main
.claude/skills/release/scripts/prepush-scan.sh origin/main..main
```

It runs gitleaks over each commit's changes and looks in every added line for this PC's paths, user and host names, private IP addresses, e-mail addresses other than the public author address, and for recordings and files over 5 MB.
Judge each finding: third-party license texts, upstream's code and version numbers are public; a real secret or anything of this PC is not.
For anything not clearly public, stop and report it with the commit and file; do not push.

## 7. Check the assets

`build/release/xe<VERSION>/` holds five packages (three `.deb`, two `.rpm`) and `SHA256SUMS`; `(cd build/release/xe<VERSION> && sha256sum -c SHA256SUMS)` passes.

## 8. Ask for the go-ahead

Show the user: the tag and its commit, how many commits `main` adds on origin, the scan's result, the assets with their sizes, and the notes' path.
Go on only when they say so.

## 9. Push and create the draft

```sh
git push origin main
git push origin xe<VERSION>
d=build/release/xe<VERSION>
gh release create xe<VERSION> --verify-tag --draft --title "XeStrata xe<VERSION>" --notes-file "$d/NOTES.md" \
  "$d"/*.deb "$d"/*.rpm "$d/SHA256SUMS"
gh release view xe<VERSION> --json url,isDraft
```

Retry a push only for a network failure (up to four times, waiting 2, 4, 8, 16 s).
`--verify-tag` makes gh refuse rather than create a tag of its own.
Report the draft's URL.
A wrong asset or note in the draft is fixed with `gh release upload --clobber`, `gh release delete-asset` or `gh release edit --notes-file`.
