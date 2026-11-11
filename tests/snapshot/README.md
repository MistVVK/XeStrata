# Config snapshot

For every **machine class x model set x entry point** this records what the engine would be started with, and a test fails on any difference.

Nothing real is used: no GPU, no model file (the model files are empty), no engine (a text stub), no download (the drafter fetch is a recorder), no network,
no real user folder (every run has its own scratch HOME, data folder and install folder, removed afterwards).

## Run it

```
python -m pytest tests/snapshot -q -p no:cacheprovider      # the gate: about a minute; needs Python 3.11+ and Node 24
STRATA_SNAPSHOT_UPDATE=1 python -m pytest tests/snapshot     # after an intended change: rewrites golden/, then review the git diff of golden/
python -m tests.snapshot.generate --out DIR                  # write the snapshot of this tree somewhere else (to compare two branches)
python -m tests.snapshot.report --golden DIR_A --against DIR_B   # Markdown: where the entry points differ, what changed between A and B
```

`STRATA_SNAPSHOT_SKIP_APP=1` skips the two app entry points by name when Node is not installed (the test prints what it skipped).
The same command works on Windows, Linux and macOS; the golden files hold no path, user name or time of the machine that made them.

## What is simulated

* **machine classes** (`matrix.py`): NVIDIA all-resident (RTX PRO 5000 72 GB), NVIDIA 12 GB streaming (RTX 3060), DGX Spark (linux-aarch64, GB10),
  Strix Halo Windows (gfx1151), Radeon R9700 (gfx1201, Linux), Intel Arc A750 (SYCL), Mac M-series small (16 GB) and large (128 GB).
  A class sets the engine family, the card architecture, the card memory, the platform and the hardware report the Terminal's catalog and the app's host read.
* **model sets**: IQ3_S, UD-Q4_K_XL, UD-IQ4_XS-v31.
* **entry points**: `setup`, `launch`, `app_plain`, `app_models`.

## Entry points

| entry | how it is driven | what is real | what is replaced |
|---|---|---|---|
| `launch` | `strata launch --model <set> --yes --no-tui` through `strata_cli.cli.main` | the whole command: settings, `Runner.run()`, the drafter step, config derivation and writing | the engine download, the model download (`ensure_model` returns the stand-in below), tuning, the server start, the drafter download |
| `app_plain` | `apps/host/server/native-control.mjs` `operation('load', {model})` in Node (`app_driver.mjs`) | the host's whole Load code, and Control's `drafting plan` / `drafting fetch` (run through `control_shim.py` on the same simulated machine) | the installer process (catalog, hardware), Control's model list / engine / profile save / start, the drafter download |
| `app_models` | the same with `values: {drafting: 'auto'}` (what the Models screen's Load passes) | same | same |
| `setup` | `strata_control.setup_config.config_for` (qwen / unsloth), the moved last stretch of `setup.py` main(), called with inputs built by `setup_driver.py` | the whole run-config construction: engine arguments, speed switches, cards, vision section | what `setup.py` decides before it (the cards and RAM come from the class, the context, budget, low-RAM, vision and resident choices from setup's own pure functions), the probes of the machine (`ProbeLib`), the model reader |

For `launch`, `app_plain` and `app_models` the last step is the same code the server runs when it starts the engine: `serve.server.engine_args(cfg, exe)` and
`serve.server.child_env(cfg)`.  The record has the run config the entry point left (`config_file`), the argument list (`engine.argv`), the variables the start adds to
the environment (`engine.env_set`), the notes the start prints, the drafter downloads that were asked for (`fetches`) and, for the app, the calls the host made to Control.

### The `setup` entry point and the stand-in

`setup.py` main() is one long function that probes the GPU, resolves or builds an engine, downloads and hashes the model and packs it.  Its last stretch (from
the engine arguments to just before `write_setup_config()`) is now `strata_control/setup_config.py`: `config_for(SetupInputs, lib)` for Qwen and Unsloth.
It is the same lines, moved (CLEAN-1A, item 1a.7); `setup.py` builds the inputs and calls it.  `lib` is the module
that holds the helper functions (`setup.py` today; they move into `strata_control` in step 5.1).  The snapshot passes `ProbeLib`, `setup.py` with the functions that
look at the machine (nvidia-smi, the disk kind, the compute mode, a saved calibration, the image setup) answered by the simulated class.

The record of `setup` has `inputs` (what was assumed, so a reviewer sees it), `config_file` (what `config_for` returns, before `write_setup_config()` merges an
earlier config), `engine` (`engine_args` / `child_env` of that config, as for the other entry points) and `setup_says` (the notes setup prints on the way).
The classes whose setup is another script (Intel Arc: `sycl/setup_intel.py`; Mac: `metal/setup_mac.py`) are `not_driven` with the reason.

The launch and app entry points still start from `World.base_config`, a hand-built stand-in for setup's output, not from `config_for`.  It differs from what
`config_for` makes (it has no `--vision`, `--kv-resident`, `--ple-io ram`, `--reserve-mib` ...).  Differences between launch / app_plain / app_models can never come
from it (all three start from the same file), but the `setup` record is the one that shows what a real install writes.  Replacing the stand-in by `config_for`'s output
changes the launch and app goldens and is a separate decision.

## The gate: one launch policy (CLEAN-1A)

In every machine class x model set, `strata launch`, the app's plain Load and the app's Models-screen Load start the engine with the same arguments and the same
environment, and ask for the same drafter downloads.  `test_entry_points_agree.py` applies the allow-list in `allowlist.py` to the golden files; the list is the
only place a difference is allowed, and each entry has its reason.  Today:

| difference | where | why |
|---|---|---|
| `api_key`, `aliases`, `port` (18500 / 8080), `fit_max_tokens`, `mcp` | app vs launch | the app's own serving settings |
| `log` | app vs launch | the app's install folder, the Terminal's launch folder |
| `engine_ready_s` | app vs launch | the app's wait for the engine; the Terminal derives its own |
| `args_derived`, `args_written` | launch only | launch's bookkeeping copies of `args` |
| `drafting: auto` | Models-screen Load only (intended) | that Load saves the person's Drafting choice |

**Request slots.**  On NVIDIA 72 GB the plain Load used to write `parallel: 3` and start `--batch 3 --batch-mtp`; that was only because it ran the MTP (no drafter had been fetched).
Following Auto it runs DFlash2, which answers one request at a time, so the app writes no `parallel` there, as the Models-screen Load always did.  The golden files pin it:
no app entry writes `parallel` or `--batch-mtp` in any cell.  The app's own default of several request slots (`apps/host/server/native-parallel.mjs`) is for runs that are not
DFlash2 and is not reached in this matrix.

## Adding a class or a set

Add a row to `matrix.py`, run the update command, review the new golden file.  A golden file for something not in the matrix, or a matrix cell without one, fails
`test_every_golden_file_belongs_to_the_matrix`.

## CI

No workflow is added in step 0 (step 2.5/2.7).  A job needs `actions/setup-python` (3.11+, `pip install -r requirements.txt`) and `actions/setup-node` (24), then
`python -m pytest tests/snapshot -q`.

## Known limits

* The record is of a model that is installed and has **no drafter on disk yet**: the state a first Load / launch meets.  The state "drafter already downloaded" is not in the matrix.
* `engine_args` and `child_env` are called as the server calls them, but the server's other start-time steps (component engine resolution, vision reader choice, effort end arguments) are not.
* Host-OS code paths (a Windows-only branch) run on the host the test runs on; the golden files were checked to be identical when made on Windows and on Linux.
