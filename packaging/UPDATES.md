# Updates: the signed feed, the component updater, the app updater

This page is for developers who build, sign and publish releases. Users update with [INSTALL.md](../docs/INSTALL.md#updating).
It covers the three updated parts, the feed format, what a PC checks, and the release steps.

Everything that downloads software on a user's PC sits behind one explicit step: `strata update` in the Terminal, or
"Update engine" or "Update Strata" in the app's Settings. Starting a model and serving never download anything
(`STRATA_NO_ACQUIRE` stays on for them; the update step is a separate Control process that is allowed to). Engines are
separate from the app and ship more often. An app-only update downloads no engine bytes, because the Windows and Linux installers are built
without the engine. An engine update downloads only the package for that PC's GPU.
**Models and saved tuning presets are never downloaded, replaced or deleted by any update.**

## The three things that are updated

| What | How | Signed with |
|---|---|---|
| Engine package (per platform / backend / variant) | Control's component updater: download with resume, verify, install beside the old one, test-start, switch | the feed key (Ed25519) |
| Control (Python runtime, `strata_control`, CLI, server) | the same updater: staged beside the running one, run once, then `components/control/active.json` points at it | the feed key |
| The desktop app | Tauri's signed updater in the shell, reached by the host through a token-protected loopback bridge | the Tauri updater key (minisign) |

Two separate private keys, both kept by the owner, neither in git, neither the same as Apple/Windows code signing:

* **feed key**: Ed25519, signs `strata-feed.json` and every engine package manifest.
* **Tauri updater key**: minisign (what `tauri signer generate` makes), signs the app installers.

<a id="the-feed"></a>
## The feed (`strata-feed.json` + `strata-feed.json.sig`)

`strata-feed.json.sig` is `{"alg": "ed25519", "key_id": "<16 hex>", "value": "<base64>"}` over the **exact bytes** of
`strata-feed.json` (no canonicalisation). The annotated example below illustrates the format; it is not a current signed feed.
Comments are for this document only; actual feed files must be strict JSON.

```jsonc
{
  "schema_version": 1, "feed": "strata-updates", "channel": "stable",
  "sequence": 12,                           // goes up with every feed; an older one than the last accepted is refused
  "generated_at": "2026-10-20T10:00:00Z", "expires_at": "2027-10-20T10:00:00Z",   // expires_at may be null
  "app": {                                  // exactly Tauri's static updater JSON, plus sha256/size per platform
    "version": "0.2.13", "notes": "...", "pub_date": "2026-10-20T10:00:00Z",
    "platforms": {"windows-x86_64": {"url": "app/Strata-0.2.13-windows-x64-setup.exe", "signature": "<minisign, base64>", "sha256": "...", "size": 71000000}}
  },
  "control": {"version": "0.1.59.1", "packages": [{"platform": "linux-x86_64", "format": "tar.gz", "url": "control/...tar.gz",
                                                  "size": 0, "sha256": "...", "installed_bytes": 0, "entry": "strata.py"}]},
  "engine": {"packages": [{
    "version": "0.1.59", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda13",
    "archs": ["sm_75", "sm_86", "sm_89", "sm_120", "compute_120"], "min_driver": {"nvidia": "580.65"}, "min_runtime": null,
    "build_id": "be885026f96a-cuda13.0", "format": "tar.gz", "url": "engine/strata-engine-0.1.59-linux-x86_64-cuda-cuda13.tar.gz",
    "size": 505568970,                      // bytes of the archive
    "sha256": "...",                        // of the archive
    "manifest_sha256": "...",               // of the package's engine-component.json (it holds every file's hash)
    "installed_bytes": 840893419, "notes": ""}]},
  "dsv41": {"packages": []}                 // optional: the separate DeepSeek engine packages, same entry shape as "engine"
  "drafters": {"packages": [{               // optional: DFlash2 drafter files Strata hosts (one pinned file each, not an archive)
    "name": "qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack", "version": "1", "url": "qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack",
    "size": 339694592, "sha256": "4df2e5ab...", "backends": ["metal"], "base": "incoai/Qwen3.8-Flash-Next-DFlash2@a22a1744", "licence": "Apache-2.0"}]}
}
```

The DeepSeek V4.1-Flash engine `strata-dsv41` reaches a PC in one of two ways. The engine package carries it beside `strata`
(`package_engine.py --require-dsv41`), or it is a separate package in the feed's `dsv41` member, packed with
`pack-engine --prefix strata-dsv41`. The separate package installs in `<data>/engine-dsv41/` beside the engine, never inside it
(`strata_control/dsv41_engine.py`), and is fetched when DeepSeek is chosen. `STRATA_DSV41_EXE` points setup at a build of your own.

**A Qwen4 model of another shape (`shape_engines`, optional).** An engine is compiled for one model geometry (`-DSTRATA_GEOMETRY=...`, `cmake/strata_geometry.cmake`) and refuses a GGUF of another one at load. A new-shape Qwen4 therefore ships with its own engine as a
separate component, picked by the model, and never in place of the Strata engine, so Flash-Next and every other model keep the engine they have (`strata_control/shape_engine.py`):

```jsonc
"shape_engines": {"packages": [{            // the same entry as "engine" (pack-engine writes it), plus the shape = the geometry's name the engine was compiled for
  "shape": "q4_lite", "version": "0.1.59", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda13", "archs": ["sm_120"], "min_driver": {"nvidia": "580.65"},
  "min_runtime": "0.1.58", "build_id": "...", "format": "tar.gz", "url": "strata-q4-lite-0.1.58-linux-x86_64-cuda-cuda13.tar.gz", "size": 0, "sha256": "...", "manifest_sha256": "...", "installed_bytes": 0}]},
"models": {"packages": [{"id": "qwen4-lite", "family": "qwen4exp_q4_lite", ...}]}     // the family names the shape: qwen4exp_<shape>
```

* Package: `package_engine.py --shape q4_lite ...` (the build made with `-DSTRATA_GEOMETRY=q4_lite`, or `-DSTRATA_GEOMETRY_FILE=` for a header that is not in the tree: use that header's name). The manifest's id is the shape's component (`strata-q4-lite`) and it carries `"shape"`; both are in the signed manifest. `sign-package`, then `pack-engine` (the archive is `strata-q4-lite-<version>-<platform>-<backend>[-<variant>]`, the entry names the shape), one folder of entries, `build-feed --shapes FOLDER`. `build-feed` refuses a model whose family is `qwen4exp_<shape>` when the feed has no engine for that shape; `verify-feed` lists the shape engines. Give the package `--min-runtime` of the Control that knows shape engines.
* A PC: the model's family (`qwen4exp_q4_lite`) makes setup (the app's Download and the Terminal's launch) fetch the engine of that shape through the signed feed when the model is chosen (resumable download, the archive's SHA-256, the manifest hash the feed names, the package signature, every file's hash, the shape in the manifest, then `strata --version` once and `strata --geometry`, which must print that shape: an engine compiled for another geometry is refused at install, and so is one that cannot answer the flag ("it does not say which model shape it was built for"; the Intel engine answers it since 0.2.14). Build the engine with `-DSTRATA_GEOMETRY_FILE=` of a header whose label is the shape name: `q4geo_synth.py --from-gguf ... --label q4_lite`), into `<components>/shapes/q4_lite/engine/<version>-<platform>-<backend>/`. The Strata engine's folder is not read or written. The model's run config says `"exe": "component:strata-q4-lite"`; every start looks it up there and checks every file. `strata update` brings an installed shape engine up to date beside the older one (the newest two are kept) and never installs one by itself. Starting never downloads.
* Older Controls: they ignore the member (no `engine` entry, no `dsv41` entry), they read only `strata-engine` manifests (a shape package copied into their engine folder is refused: its id is another component), and the model's family is one they do not know, so they list it as "needs a newer Strata" and never offer it.
* A Mac does not offer a model of its own shape yet (that install path has no shape step).

The `models` member (optional) is the model list: a new model can ship as a signed feed update, with no Control package. Each entry is one model:

```jsonc
"models": {"packages": [{
  "id": "qwen4-30b", "title": "Qwen4 30B", "by": "Qwen",  "about": "one plain sentence",
  "licence": "Apache 2.0", "licence_url": "https://huggingface.co/Qwen/Qwen4-30B",     // the url must be https
  "family": "qwen4exp",                     // the GGUF architecture the engine runs (`strata --families`): qwen4exp (the Flash-Next install path) or qwen35moe (the Qwen3.6 one)
  "repo": "unsloth/Qwen4-30B-GGUF", "revision": "<40 hex>",                              // the pinned commit
  "quants": [{"name": "Q4_K_M", "about": "4-bit", "ram_gb": 24, "arena_gb": 17.0,      // setup's recommendation for this size; arena_gb optional
              "files": [{"name": "Qwen4-30B-Q4_K_M-00001-of-00002.gguf", "path": "Q4_K_M/Qwen4-30B-Q4_K_M-00001-of-00002.gguf", "size": 123, "sha256": "..."}]}],
  "mmproj": {"name": "mmproj-F16.gguf", "size": 123, "sha256": "..."}, "pictures": true, // pictures need the mmproj
  "drafter": "mtp",                         // mtp | none
  "context": 131072,                        // the default context (informational)
  "min_engine": "0.2.12",                   // the first engine that reads it; setup refuses an older one as for a built-in model
  "kv_bytes_per_token": 90112, "mtp_drafts": 2, "state_gb": 0.07                         // qwen35moe only: the cache figures setup shows (kv_bytes_per_token is required there)
}]}
```

Control keeps a model only if the whole entry is well-formed (ids, 40-hex revision, https licence url, plain `.gguf` file names, positive sizes, 64-hex SHA-256, no file reused by two sizes);
a bad entry is left out with one log line and the rest of the feed and the other models work. A model is added to setup's own tables, so the menus list it after the built-in ones and
the install is the built-in one: the revision and the file pins go where `download` looks them up, the SHA-256 is checked after the download and a wrong file is kept as `.bad`. A `qwen4exp`
size is one `.gguf` or numbered parts `...-00001-of-0000N.gguf` (up to 9) of one folder; a `qwen35moe` size is one `.gguf`. A model whose family the installed engine does not run is listed as not
available on this engine and never offered; a family this Control does not know is the same. The last verified feed's list is kept in `<components>/.updates/feed-models.json` (the signed
feed and its signature, checked again when read), so an offline PC still sees it; a newer feed without the member withdraws the list, an older (replayed) feed is ignored. `STRATA_FEED_MODELS=0`
turns the whole thing off. Build it with `feed_tool.py model-files FILE... [--repo-dir Q4_K_M]` (sizes and hashes of the real files) and `build-feed --models FOLDER` (the folder holds the
`<id>.entry.json` files; a malformed one stops the build); `verify-feed` lists the models and exits 3 if Control would drop one. Older Controls (0.2.11 and before) ignore the member.

The `drafters` member lists the tuned DFlash2 drafter (`strata_control/drafting.py` TUNED_NAME: Strata's fine-tuned copy of incoai/Qwen3.8-Flash-Next-DFlash2, Apache-2.0,
modified by Strata). Control fetches it for the Mac sets that have a measured entry (`SET_DEFAULTS`: UD-Q4_K_XL and UD-IQ4_XS-v31) during setup, `strata launch` and
`strata drafting fetch`: it reads this signed feed, takes the entry named TUNED_NAME that lists `metal`, and downloads `url` (resumable) into `<data>/dflash2/`. The size and SHA-256
are pinned in the code too, so the feed can move the file but a feed that lists other bytes is ignored. No feed answer (offline, host down) or no such entry: nothing is said and the
model keeps the MTP. Make the entry with `feed_tool.py pack-drafter PACK --out RELEASE/drafters --version 1 --backends metal [--url-prefix ""]` (copies the file flat and writes
`<name>.entry.json`) and pass the folder to `build-feed --drafters RELEASE/drafters`; upload the pack beside the feed in the same release. Older Controls ignore the member.

The member holds two entries (one `pack-drafter` call each, both folders' `.entry.json` files in the same `--drafters` folder): `qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack` (339,694,592 bytes) with
`--backends metal,cuda,hip`, and `qwen3.8-flash-next-dflash2.strata1.q4_0-bf16sel.pack` (522,458,112 bytes, sha256 34ca17fe...) with `--backends sycl` for the Intel Arc engine, which reads only bf16
selector codebooks. Control asks the feed for the file name its table entry (`SET_DEFAULTS`) names and for its own backend, so no platform fetches the other's file.

`url` is relative to the feed's own location (or an absolute `https://` address; a loopback `http://` is allowed so a local test
feed works; a local feed folder may only name files inside itself).
For a GitHub release the assets are flat: pack with `--url-prefix ""` and upload every file beside the feed.

### What a PC checks, in order

1. the signature of the feed (key ids it trusts: `strata_control/update_keys.json`, plus `STRATA_UPDATE_TRUSTED_KEYS` for a test setup, which
   the CLI announces);
2. `channel`, `expires_at`, and `sequence` not lower than the last accepted feed (the downgrade guard for a replayed old feed);
3. the engine package that fits **this** PC (`feed.select_engine`): platform, backend (CUDA / HIP / SYCL / Metal as detected),
   CUDA `archs` against the card (a `compute_NN` PTX entry covers newer cards), HIP `archs` against the card's gfx name,
   `min_driver` against the NVIDIA driver (this is how CUDA 13 beats CUDA 12 on a new driver and CUDA 12 is chosen on an old one),
   `min_runtime` against Control; newest version wins, and equal versions prefer the build that needs the newer driver;
4. never older than what is installed (an update never downgrades; only `strata components use` / `rollback` go back, to an engine that is already installed).

### What an engine update does

download (resume, progress) -> SHA-256 of the archive = the feed's -> unpack into `components/.s/<4 hex>/` (short on purpose: Windows paths are limited to 259 characters unless long paths are on; files are written through extended-length paths) (names and links checked,
total size limited to the feed's `installed_bytes`) -> the manifest is the one the feed names (`manifest_sha256`) and says what the feed says
-> the manifest's own signature verifies -> every file's hash -> fits this PC -> **test-start from the staging folder through the real start
path** (isolated owner runtime and profile, `component:strata-engine`, an installed model, one-token probe, stop) -> only then rename
into `components/engine/` and re-resolve to make sure a start now picks it. Any failure leaves `engine/` exactly as it was.
Every older engine is kept until the person deletes it (Settings > Engine, `strata components remove`); `state.json` `engine.previous` is the one the last
switch left. A new engine becomes the one in use and clears any pin. With no model installed yet (first run)
the test is `--version` only, and the report says so.

Not while an engine is serving: the check uses the activity lease `strata serve` and the owner runtime already hold. `--wait SECONDS` waits;
a terminal asks whether to wait; scripts get exit 7. The download itself may run first; the check is repeated right before the test start and
before the switch.

Saved tuning presets: when the engine identity changed (version, build id, manifest hash, backend) `tuning/recheck.json` gets a
`pending` entry (`engine_changed`, from/to identities). Presets are not opened. The tuner offers the ~2 minute check for them and calls
`preset_marks.clear(id)`; rolling back to the engine they were measured on clears the debt again.

### Commands

    strata update [--check] [--app] [--no-engine] [--no-control] [--feed URL|DIR] [--wait S] [--force] [--model-config RUN.json]
    strata components update            # the same
    strata components rollback          # back to the previous engine (an engine switch; nothing is downloaded)
    strata components engines [--output json]   # every installed engine (Qwen and DeepSeek families), sizes, which one is in use: what Settings > Engine shows
    strata components use <version|folder|newest>   # switch to an installed engine; `newest` goes back to following updates (no pin)
    strata components remove <version|folder>       # delete an installed engine; never the one in use or one a running program uses
    strata components engine            # shows "update available (...)" from the last check, no network

An update whose engine is already installed (the person switched to an older one) downloads nothing and makes it the engine in use again.
A version the person switched away from is never hidden from the update. The DeepSeek engine (`engine-dsv41/`) has its own list and its own choice
(the `.chosen` file there, read by the app, `strata launch` and setup); an update to a newer DeepSeek engine clears it. The two families never touch
each other's files or state (`strata_control/engine_library.py`).

Exit codes: 0 done or nothing to do, 5 refused (signature, hash, downgrade, no trusted key, acquiring forbidden),
6 failed with the old version still in use (network, test start), 7 an engine is serving, 130 stopped.
`--output json` / `--output jsonl` (progress events, last line is the result) are what the app's host uses.

## How the owner makes the real keys (once)

On an offline machine, outside any git checkout:

    python packaging/feed_tool.py keygen --out D:\strata-keys          # feed-key.json (SECRET) + feed-key.pub.json
    npx tauri signer generate -w D:\strata-keys\tauri-updater.key      # asks for a password; prints the public key

Back both up (a lost feed key means every installed PC needs a build with a new key; a lost Tauri key means the same for the app).
Then, in git, once, in the commit that ships the first signed release (the release key id is `5d1bc23abbeffd0d`):

* the `keys` entry from `feed-key.pub.json` into `strata_control/update_keys.json`;
* the Tauri public key into `apps/desktop/src-tauri/tauri.conf.json` (`plugins.updater.pubkey`) and into `update_keys.json` as `app_minisign_public`
  (the Terminal verifies the app installer with it).

Only release keys live in `update_keys.json`. A tester or local-test key is never added there: it is passed with `STRATA_UPDATE_TRUSTED_KEYS`
(a file of extra public keys; `packaging/make_local_feed.py` writes one as `test-trust.json`), so private tester builds keep working
without their key being part of a release build. A build whose `update_keys.json` is empty (a source checkout from before the keys) refuses every
feed with "this build trusts no update key".
A key rotation is a release that carries both public keys; the old private key signs that release, the new one signs the next.

## Releasing

Keys live only on the release PC (`feed-key.json`, `tauri-updater.key`; the Tauri key has no password). Every build box builds WITHOUT a key:
the default `tauri.conf.json` has `createUpdaterArtifacts: false`, so `tauri build` needs no signing key and makes the same installer bytes.
All signing happens on the release PC, on the finished files.

**What is built (per OS)**

* *The installer*, one per OS (0.2.13). The Windows and Linux ones have no engine inside: built with `--config src-tauri/tauri.update.conf.json`, which drops `engine-bundle`. The Mac dmg keeps its small Metal engine (one Mac variant).
  Windows: `Strata-0.2.13-windows-x64-setup.exe`; Linux: `Strata-0.2.13-linux-x86_64.AppImage`; Mac: `Strata-0.2.13-macos-arm64.dmg` for people and
  `Strata.app.tar.gz` as the updater archive (the updater cannot install from a dmg). The Windows and Linux files are both the first download and the
  file `latest.json` names; the first start of the app downloads the engine for the card (the first-run card), or installs a file the person already has.
  Why: an app update must not carry an engine, and the right engine depends on the card. Engines come per card from the signed feed
  (`feed.select_engine`: platform, backend, CUDA `archs` and `min_driver`, HIP `archs`; `tests/terminal/test_release_matrix.py`).

**Steps**

    python packaging/feed_tool.py sign-package <engine package folder> --key <feed-key.json>        # every engine package (the Mac app bundles its UNSIGNED copy)
    python packaging/feed_tool.py pack-engine  <engine package folder> --out RELEASE/engine --url-prefix=
    python packaging/feed_tool.py pack-control <control folder> --version 0.1.59.1 --platform P --out RELEASE/control --url-prefix=   # --format zip for windows-x86_64
    # --url-prefix= (not --url-prefix "") because PowerShell 5.1 drops an empty argument. The Control version is the VERSION file
    # (0.1.59.1), not the app version: a Control package newer than the installed one is installed by `strata update`.
    # Pack Linux and Mac packages on Linux / the Mac or on Windows with this tool: it sets the execute bit from the file header either way.
    cd apps/desktop && npx tauri build --bundles nsis --config src-tauri/tauri.update.conf.json    # Windows installer (no engine inside)
    # Linux: tauri build --bundles appimage (no engine inside) is the AppImage
    # on the release PC, per updater artifact (Windows setup .exe, Linux .AppImage, Mac Strata.app.tar.gz):
    npx tauri signer sign -f <path to tauri-updater.key> <artifact>                                 # writes <artifact>.sig
    python packaging/feed_tool.py set-app-platform --app-json RELEASE/tauri-latest.json --target windows-x86_64 --version 0.2.13 --archive Strata-0.2.13-windows-x64-setup.exe --sig Strata-0.2.13-windows-x64-setup.exe.sig --notes "..."
    python packaging/feed_tool.py set-app-platform --app-json RELEASE/tauri-latest.json --target linux-x86_64   --version 0.2.13 --archive Strata-0.2.13-linux-x86_64.AppImage --sig Strata-0.2.13-linux-x86_64.AppImage.sig
    python packaging/feed_tool.py set-app-platform --app-json RELEASE/tauri-latest.json --target darwin-aarch64 --version 0.2.13 --archive Strata.app.tar.gz --sig Strata.app.tar.gz.sig
    python packaging/feed_tool.py build-feed --out RELEASE --sequence N --key <feed-key.json> --engine RELEASE/engine --control RELEASE/control --app-json RELEASE/tauri-latest.json [--dsv41 RELEASE/dsv41] [--shapes RELEASE/shapes] [--drafters RELEASE/drafters]
    python packaging/feed_tool.py verify-feed RELEASE --pub <feed-key.pub.json>

`set-app-platform` checks each `.sig` against the public key in `tauri.conf.json` over the exact bytes, so a file signed with another key, or changed after
signing, stops the release. The Mac dmg is not in `latest.json` (the updater uses `Strata.app.tar.gz`) and needs no Tauri signature; it is downloaded by hand.

### The Mac app

`tauri build` signs its installers with the Tauri key before anything else touches them. On a Mac that is too early: `sign-and-notarize.sh` then re-signs every
Mach-O in the app with the Developer ID and notarises and staples it, which changes the bytes. The `Strata.app.tar.gz` and `.sig` that `tauri build` wrote hold the
**unsigned, un-notarised** app, and the updater would install exactly that on a user's Mac. So the in-app Mac update is built from the finished app instead:

1. `sign-and-notarize.sh <dmg> --updater-archive` (on the Mac) packs the signed, notarised, stapled `.app` into `Strata.app.tar.gz` (the app folder at the root, the layout
   Tauri's bundler makes), unpacks it again and checks `codesign`, the staple and `spctl`. It never reads the Tauri key.
2. Move the `.tar.gz` to the release PC and sign it there with `npx tauri signer sign -f <key> Strata.app.tar.gz` (or `TAURI_SIGNING_PRIVATE_KEY_PATH` and the password variable).
   The key stays on the PC and in the owner's safe storage; nothing in this tree reads or stores it, and the Mac needs no node and no key.
3. `feed_tool.py set-app-platform` checks that signature against the public key in `tauri.conf.json` over the exact archive bytes, checks the archive holds `<Name>.app` at its root,
   and writes `url`, `signature`, `sha256` and `size` of the darwin entry. A signature that was made for the pre-notarisation file does not verify, so it cannot be published by mistake.
4. `build-feed --app-json` and `verify-feed` as above. Upload the notarised `Strata.dmg`, `Strata.app.tar.gz`, and the `latest.json` from the feed folder (`app/latest.json`).

The Windows installer and the Linux AppImage need no extra step: their `.sig` files from `tauri build` are the ones to pass to `set-app-platform`. Re-signing the Mac app after the archive
was made (a second `sign-and-notarize.sh` run) makes a new archive: sign and add it again.

`build-feed` also writes `RELEASE/install-manifest.txt`, the file the one-line installers read: one line per Control package,
`platform version url sha256 size`, taken from the `*.entry.json` files in `RELEASE/control`. The Control package is the Terminal that `install.sh` unpacks (it must hold `bin/strata`).
`install.sh` unpacks with `tar -xzf` and `install.ps1` with `Expand-Archive`, so `windows-x86_64` must be packed with `--format zip` and the other platforms as `tar.gz`;
`build-feed` leaves a pack in the wrong format out of the manifest and says so. Pack Control for every platform you want the one-line install to cover (`install.sh` knows `linux-x86_64`, `linux-aarch64` and `macos-arm64`, `install.ps1` `windows-x86_64`);
a platform with no line gets "this release has no Terminal for <platform>". Upload `install.sh` and `install.ps1` from `packaging/` unchanged: their built-in address is the release folder.

**Linux ARM64 (DGX Spark) has its own feed file.** A Control up to 0.1.52 refuses a feed that names a platform it does not know (`the feed's control entry names an unknown platform`), so the `linux-aarch64` entries are not in `strata-feed.json`, which every released client reads. They are in `strata-feed-linux-aarch64.json` (+ `.sig`, same format, same key, its own `sequence`, aarch64 entries only), uploaded to the same release. A Linux ARM64 Terminal reads that file by default, from a folder or a release address (`strata_control.feed.feed_name()`); a source that names its own `.json` file is used as given. Until every client skips unknown platforms, each release must publish both files, and the aarch64 file's sequence must never go down.

The feed is hosted as the files of a GitHub release of `Niko1221/Strata`. The built-in address is
`https://github.com/Niko1221/Strata/releases/latest/download` (`strata_cli/update_cmd.py` `DEFAULT_FEED` adds `/strata-feed.json`; `install.sh`
and `install.ps1` use the folder). GitHub assets are flat, so build the packages with `--url-prefix ""` and upload every file of the release folder
as an asset: `strata-feed.json`, `strata-feed.json.sig`, `install-manifest.txt`, `install.sh`, `install.ps1`, the engine and Control archives and the
app installer with `latest.json` (the Tauri updater endpoint). `STRATA_FEED` (installers) and `--feed` / `STRATA_UPDATE_FEED` (`strata update`) override the
address. Nothing in this tree uploads anything.

Publish as a normal release (latest): a pre-release is not found by `latest/download`, so the installers, `strata update` and the app updater would get a 404. The `latest` release must carry every file listed above, so a later hotfix release repeats them all. The app updater reads
`https://github.com/Niko1221/Strata/releases/latest/download/latest.json`.

## Testing locally

Everything above runs against a folder or a loopback server: `tests/terminal/test_update_feed.py` (signatures, hashes, downgrades, selection),
`tests/terminal/test_updater.py` (end to end against a loopback HTTP feed with stub engines: update, test-start, switch; a failing test start
keeps the old engine; an interrupted download resumes; rollback; Control; the app installer's minisign check), `apps/host/tests/component-updates.test.mjs`
(the Settings buttons). For a real engine on a box, build a package with `packaging/package_engine.py`, serve it from a local feed made by `packaging/make_local_feed.py`, and run `strata update --feed` against it.

### Release gate: a feed model installed from a real file

The model list (`models` member) has a gate with a real file, not a mock: `tools/feed_model_install_gate.py` writes a small synthetic qwen4exp GGUF
(`tools/q4geo_synth.py`), signs a models-only feed with a throw-away key, serves the file from a loopback Hugging Face stand-in, and runs `setup.py` in the
app's mode (no engine, no pip) against it. It fails when setup raises (`KeyError: 'mmproj'` for a model with no picture reader), fetches the Flash-Next draft
layer for `drafter: none`, writes a second copy of the experts instead of a RAM budget, or keeps the shipped 48 x 512 expert profile for a model of another shape;
a second run must change nothing. It runs as `tools/test_feed_model_install_realfile.py` in `python tools/check` (py-tools) when a llama.cpp checkout is
given (`STRATA_TEST_LLAMA_CPP`, for its gguf-py), and is SKIPPED with that reason otherwise: set the variable wherever the release gate runs. With
`--full --tokenizer-from <real shard> --engine <engine folder built for that shape>` it also starts the engine from the run config and sends one chat request.
Run it before every release that touches `setup.py`, `strata_control/feed_models.py`, `strata_control/setup_config.py` or the feed schema, and before a Qwen4 day.

## History / older measurements

- **Keys:** the feed key and the Tauri updater key were created once and the feed key's public half was committed in `strata_control/update_keys.json`
  with the first signed release (0.2.0-beta). Terminals from before that build trust no key and refuse every feed.
- **Engine-less installers:** an app update once carried an engine. The Windows and Linux installers now drop `engine-bundle`
  (`tauri.update.conf.json`), so one `latest.json` entry per OS serves every card variant.
