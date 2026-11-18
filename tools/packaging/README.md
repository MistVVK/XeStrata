# Strata packaging: targets, recipes and locks

This page is for developers who build the desktop app, the Terminal archive and the engine packages. To install Strata, use
[docs/INSTALL.md](../../docs/INSTALL.md). It describes the current release (app and Terminal 0.2.16, Control 0.1.62, engines 0.1.62).
How packages are signed and published is in [UPDATES.md](UPDATES.md), and the package format is in [ENGINE-COMPONENT.md](ENGINE-COMPONENT.md).

Release assets show what was published; the platform guides and retained reports show what was tested. A published package does not
qualify every GPU, model or feature on its platform.

## Quick start

| I want to | See |
| --- | --- |
| Build the desktop app | [Desktop app](#desktop-app) |
| Build an engine package | [ENGINE-COMPONENT.md](ENGINE-COMPONENT.md#building-a-package) |
| Build the Terminal archive | [The Terminal archive](#the-terminal-archive) |
| Sign and publish a release | [UPDATES.md](UPDATES.md#releasing) |
| Test an update against a local feed | [UPDATES.md](UPDATES.md#testing-locally) |

## The target matrix

Each row is an OS, CPU ABI and backend. A row is `qualified` only after the acceptance checklist
([below](#what-a-platform-must-produce-before-it-is-qualified)) has run on it.

| Target | App artifact | Engine package recipe | Status |
| --- | --- | --- | --- |
| `macos-arm64-metal` | `.app` and `.dmg` (Tauri) | Metal `strata`, bundled with `strata-vision` (`packaging/build_vision_mac.sh`) | Built. Self-test passes from the mounted DMG. Inference measurements are in [MAC.md](../../docs/MAC.md). Signing: `packaging/mac/sign-and-notarize.sh` ([mac/README.md](mac/README.md)). |
| `darwin-arm64` Terminal | self-contained `.tar.gz` | any | Built and verified. Runs from a moved directory with no system Python. |
| `windows-x86_64-cuda` | NSIS `.exe` (Tauri) | `cuda13` and `cuda12` | Built and tested on a Windows PC with an NVIDIA card: silent install with a System32-only PATH, a real model, an agent chat. |
| `windows-x86_64-hip` | NSIS `.exe` (Tauri) | `tools/hip/build_windows.bat` | Published. Strix Halo inference is recorded in [STRIX_HALO.md](../../docs/STRIX_HALO.md). This does not qualify every discrete Radeon card ([AMD_HIP.md](../../docs/AMD_HIP.md)). |
| `linux-x86_64-cuda` | AppImage (Tauri) | `packaging/build_cuda_linux.sh`, built in a glibc 2.35 chroot | `cuda13` built and tested on RTX 3060 and RTX A4000 (app in the browser, a real model, a picture). `cuda12` built. |
| `linux-x86_64-hip` | AppImage (Tauri) | `packaging/build_hip_linux.sh` | `rx9070` (gfx1200 / gfx1201) built and tested on a Radeon AI PRO R9700. `rx7900` (gfx1100 / 1101 / 1102) built, not run on an RX 7000 card. `rx6000` (gfx1030 / 1031) built, not tested on a real RX 6000 card. |
| `linux-aarch64-cuda` | none (no AppImage for arm64) | `packaging/build_cuda_linux_aarch64.sh`, built on the DGX Spark (`cuda13`, `sm_121`) | The engine package and the Terminal archive built and tested on one DGX Spark ([DGX_SPARK_LINUX.md](../../docs/DGX_SPARK_LINUX.md)). Not tried on any other arm64 machine. |
| `linux-x86_64-sycl` | none | `packaging/build_sycl_linux.sh` (one JIT package for Alchemist and Battlemage) | Engine package built and tested on an Arc A750. Battlemage is supported and untested by us. |
| `windows-x86_64-sycl` | none | none | Gated. No native Windows SYCL path exists upstream. |
| `macos-x86_64-*` | none | none | Unsupported. The macOS engine is Apple Silicon only. |

Cross-compilation is not attempted for the desktop app. Tauri bundles are per OS, and the Windows installer is built on Windows.
The Python contracts, the owner transport, the checks and the engine build recipes are portable.

## The private Node runtime

The desktop bundle ships a pinned standalone Node beside the App Host, so an end user supplies no runtime. It is not committed
(about 117 MB, an input rather than source). Fetch and verify it:

```sh
apps/desktop/scripts/fetch-node.sh                 # host platform
apps/desktop/scripts/fetch-node.sh --target linux-x64
apps/desktop/scripts/fetch-node.sh --check         # verify what is installed
```

The script refuses an archive whose SHA-256 differs from the digest recorded for that target. The digests come from nodejs.org's
`SHASUMS256.txt` for the pinned version. To bump the version, change `STRATA_NODE_VERSION`, add the new digests and run `--check`.
`fetch-node.ps1` is the Windows equivalent.

## Desktop app

```sh
cd apps/desktop
./scripts/fetch-node.sh                                                      # the sidecar (once)
npm install --prefix ../host && npm --prefix ../host run build                # the UI
cargo tauri build --bundles app                                              # macOS: Strata.app
./scripts/make-dmg.sh                                                        # macOS DMG (headless; Tauri's default step needs a Finder session)
cargo tauri build --bundles nsis                                             # Windows, on Windows: per-user NSIS installer
cargo tauri build --bundles appimage deb                                     # Linux, on Linux
```

The Windows installer is per-user and needs no elevation (`installMode: currentUser` in `tauri.conf.json`). Models, profiles and
sessions live outside the install directory (`~/.strata` and the platform state directories), so uninstalling the app does not remove them.

Build boxes build without a signing key: the default `tauri.conf.json` has `createUpdaterArtifacts: false`. Signing happens on the
release PC ([UPDATES.md](UPDATES.md#releasing)).

## Strata Control and the engine in the installed app

The installer carries Control, and on a Mac the Metal engine too. Neither fetches anything when it runs.

**Control** (`<install>/control`, built by `apps/desktop/scripts/stage-control.py`) is the same bundle as the Terminal archive:
the pinned CPython 3.12.15, the hash-locked wheels, `strata_control`, `strata_cli` and `serve`, plus a `BUNDLE.json` with the
interpreter and lock digests. The host finds it beside itself or through `STRATA_CONTROL_DIR`, runs it on its own interpreter
(`PYTHONNOUSERSITE=1`, no `PYTHONHOME` or `PYTHONPATH`) and starts a model with `strata profile create|use-config` and
`strata runtime start`. The engine then runs under the owner runtime (loopback TCP and a 256-bit token on Windows, a private
socket elsewhere; a Job Object or process group; leases), and a second client such as the Terminal can attach to the same instance.

The app starts each engine with `runtime start --die-with-pid <host>`, so the engine ends with the host. The host ends with the shell
(a kill-on-close Job Object on Windows; the host watches its parent everywhere). The app is a single instance: a second launch focuses
the open window. One app can run several engines (profiles `app-1`, `app-2`, ...). Every close path is in [docs/CLOSING.md](../../docs/CLOSING.md).

**The engine.** The Windows and Linux installers carry no engine. On the first start the app shows the engine for the card with its size
and one Download button (with progress and resume), or installs an engine file the person already has ("I already have the file").
The Mac dmg keeps its small Metal engine in `<install>/engine-bundle/<package folder>`. Either way the engine ends up in the per-user
component folder, after the package is checked against its manifest (`%LOCALAPPDATA%\Strata\components`, `$XDG_DATA_HOME/strata/components`, or `STRATA_COMPONENTS_DIR`),
where Control finds it like any engine that `strata update` installed:

    components/engine/<version>-<platform>-<backend>[-<variant>]/
        engine-component.json     the manifest (see ENGINE-COMPONENT.md)
        strata[.exe]  *.dll | *.so   the binary and the GPU runtime libraries beside it

A run config names it as `"exe": "component:strata-engine"`. At start, `strata_control/engine_component.py` picks the package for this PC
(platform, then the backend its GPU needs, found with nvidia-smi, rocminfo or sycl-ls, or `STRATA_ENGINE_BACKEND`), checks every
file's size and SHA-256 and puts the package folder first on the library path. A mismatch is `EngineIntegrityError` and nothing starts.

When no package matches, `strata serve` ends with exit 5 and "No engine for this PC was found". It never fetches. Before that point
everything is a warning: `strata launch` says it could not find an engine, asks whether to go on, downloads the model and ends with
the next step (exit 0, nothing started). The app shows the same warning with an "Update engine" button in Settings.
`packaging/package_engine.py` makes a local package from a build folder.

**No acquisition at run time.** `STRATA_NO_ACQUIRE=1` (set by the Tauri shell for the host, and by the host for Control) makes
`strata_control/acquisition.py` refuse pip, downloads and compiles. On the host side `server/acquisition.mjs` refuses the llama.cpp
and cloudflared fetches (a connector placed in `components/cloudflared/` is used as it is). Models the user chooses to download are a
separate, explicit action. `apps/host/tests/no-acquire.test.mjs` fails on any outbound request or installer process.
`apps/desktop/scripts/installed-flow.mjs` drives an installed app end to end.

## The Terminal archive

The Terminal bundles a private Python and its dependencies for `windows-x86_64`, `linux-x86_64`, `linux-aarch64` (DGX Spark) and `darwin-arm64`
(`package_terminal.py --target ...`, built on the target OS).

- The interpreter is python-build-standalone CPython 3.12.15, digest-checked at build time.
- Packages come from `packaging/locks/runtime-<target>-cp312.txt` and are installed with
  `pip install --require-hashes --no-deps --only-binary=:all:` at build time only. `lock_wheels.py` makes the lock; review its diff when a pin changes.
- Wrappers: `bin/strata.cmd` and `bin/strata-direct.ps1` on Windows (never a `strata.ps1`: PowerShell would run it before `strata.cmd`, and the default execution policy refuses it), `bin/strata` on POSIX. `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`, `setup.sh`
  and `update.sh` use a `python/` folder beside them when there is one.
- `smoke_terminal.py <archive>` extracts an archive into a clean environment (a path with a space, a PATH with no Python) and runs
  `--version`, `doctor`, `profile list`, `install` and `serve --engine mock`.
- Check each archive's `BUNDLE.json` and lock digests. An old archive is not upgraded when the source lock changes.

## What a platform must produce before it is qualified

1. The app launches and the window serves the supplied UI from the host's origin.
2. The owner transport works: one owner per installation, a second client attaches, a stale owner can be recovered, and no client can stop a process it does not own.
3. Real inference on a supported pack, with the engine build hash and the model hashes recorded.
4. A qualification run on that platform, with raw evidence kept. The Mac-first run in October 2026 is the earlier example: [history/docs/work/QUALIFICATION-KIT.md](../../docs/history/docs/work/QUALIFICATION-KIT.md).

## Release checks and gaps

- CI covers scripts, Terminal checks, Metal builds and Docker images ([.github/workflows](../../.github/workflows)). It does not replace native desktop
  packaging, signing or real-GPU acceptance on every target. Some recipes still run by hand.
- Signing needs the owner's credentials, which are never in this tree. The feed and every engine package are signed with the feed key
  (`feed_tool.py`); the app installers with the Tauri updater key ([UPDATES.md](UPDATES.md)). The Mac app and Terminal archive are signed and
  notarised on a Mac that holds the Developer ID certificate. No Windows code-signing certificate is configured, so Windows may warn about the installer.
- Check the licenses and provenance of bundled fonts and assets when packaging. The early D05 checklist lacked redistribution evidence;
  treat that as an open checklist finding, not an audit of every release asset.

## History / older measurements

- **Tauri config:** an early draft used `perMachine`, `oneClick` and `deleteAppDataOnUninstall`. The Tauri schema rejects them; the
  current config uses `installMode: currentUser`.
- **Repository size:** `apps/desktop/src-tauri/target` and the Node sidecar were briefly committed, which left about 324 MB in the git object
  store. They are untracked and ignored now. History was not rewritten.
- **Mac packaging tests:** the first DMG self-test (8/8) and the engine's 17 kernel-parity suites ran on an Apple M4 GPU in a 16 GiB Mac, with service stand-ins for signing.
  Later inference measurements on larger Macs are in [MAC.md](../../docs/MAC.md).
- **Intel Arc:** official since 0.1.40.2. The 2026-10 B70 tests used the source build, not the engine package. `sycl/setup_intel.py` (Docker) remains the source-build route.
