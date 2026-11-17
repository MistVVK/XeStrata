# The 0.2.16 layout move

This folder holds the script that moves the repository's top level into six folders, and the checks used to test it. It carries
the script only. The move itself is made once, in a short window after every branch of the release is merged, as one commit of
renames and path edits.

## The layout

| folder | holds |
|---|---|
| `engine/` | the C++/CUDA/HIP/SYCL/Metal engine: `src`, `include`, `cmake`, `metal`, `sycl`, `spark`, `third_party`, `CMakeLists.txt`, `CMakePresets.json`, the C++ tests and benches |
| `app/` | the desktop app and its host (was `apps/`) |
| `terminal/` | the Terminal, in the same layout as the installed Terminal: `strata.py`, `setup.py`, `strata_cli`, `strata_control`, `serve`, `integrations`, `bin`, `data`, `ref`, `crates`, the Python tools (`terminal/tools`) and the Python tests |
| `docs/` | the documentation, the translated READMEs (`docs/lang/`), `history/`, the benchmark results |
| `tools/` | the repository's own checks (`tools/check` and the docs and binary checks), `packaging`, `docker`, `research`, `nix`, `BENCH.bat`, `bench.sh`, this folder |
| `.github/` | CI, `CONTRIBUTING.md`, `SECURITY.md` |

Root files: `README.md`, `SUPPORTERS.md`, `LICENSE`, `AGENTS.md`, `setup.sh`, `START-HERE.bat`, `.gitignore`, `.gitattributes`. `SETUP.bat`,
`UPDATE.bat` and `update.sh` are removed: `setup.py --update` now does the git pull itself, and the first run in the new layout
carries an old clone's state into `terminal/` (`.venv` by `START-HERE.bat` / `setup.sh`, the rest by `setup.py`).

## Running it

The script moves the tree it lives in, so run a copy:

```sh
cp -r tools/layout /tmp/layout            # any folder outside the checkout
git clone <repo> wt && cd wt              # or a clean worktree of the head to move
python -I /tmp/layout/move_0216.py .      # renames with git mv, then the path edits; nothing is committed
```

`--plan` prints the moves and stops. `--no-edits` makes the renames only. The script writes `map0216-edits.log` (one line per edited
file and kind) and `map0216-prose.tsv` (paths named in comments and page text, not edited) beside the checkout.

It stops with exit code 3 and a list when an explicit edit no longer matches. That means the tree changed in a way the edit list
does not know yet. Fix the edit in `move_0216_edits.py` and replay from a clean checkout. A replay is deterministic.

## What it changes (a replay on the integration head of 11-17 03:30)

- 74 `git mv` (pure renames, so `git log --follow` and blame keep working) and 3 removals;
- 1,427 path edits in 308 files. By kind:
  - root-relative path tokens in CI, Docker, `.gitattributes`, `tools/check` and the release scripts;
  - Markdown and HTML links, recomputed from each page's old and new place;
  - Python joins from a root that moved with its folder;
  - relative paths in the app's scripts and tests;
  - `.gitignore`;
  - GitHub `blob/main` links;
  - explicit edits for code that computes a root or names a moved path in a way a rewrite cannot know;
  - tables of repository paths inside test files.

Paths after a shell variable are only rewritten when the variable is the checkout (`$repo`, `$ROOT`, `%SRC%` and similar). A
toolkit's `$CUDA/bin` and a package stage's `$STAGE/metal/kernels` are left alone.

## Checks (the dry runs of 11-16 and 11-17)

Each check runs on a clean integration head (before) and on the same head with the move replayed (after).

| check | before | after the move |
|---|---|---|
| `tools/check` py-tests, py-serve, py-tools, docs (Linux, 128 threads), head of 11-17 02:30 | 3,789 / 1,807 / 1,724 / 6 passed, 0 failed | 3,784 / 1,807 / 1,724 / 6 passed, 0 failed (5 tests fewer on that run: `tests/core/test_tqp_ref.py` sat among the C++ tests and was no longer collected; the script now moves it to `terminal/tests/core/`, and a collect-only count gives 3,861 on both trees) |
| `tools/check` cmake-cpu, host-typecheck, host-test, host-build, docs (Linux, 12 threads) | cmake-cpu 120 passed, host-test 1,851 passed, 0 failed | cmake-cpu 121 passed, host-test 1,851 passed, 0 failed |
| `tools/check` snapshot (that box has a real Intel card) | 25 of 64 fail | the same 25 fail |
| the Windows test files that touch the start files and Windows paths, plus every Windows-only test file, one pytest process per file (`checks/win_start_tests.py`) | 614 passed, 1 failed, 12 skipped | the same: 614 passed, 1 failed, 12 skipped (the failure is in the integration head itself) |
| the Terminal archive for Windows (`checks/archive_lists.py`, dry run of 11-16) | 557 members | 557 members, the same names; 7 members differ in content (the files the move edits) |
| CUDA 13.0 (sm_75 / 86 / 120) and CUDA 12.9 (sm_60 / 70 / 75 / 86): `cmake -S engine`, `strata` and the picture readers | | built; `tools/check_no_mmq.py` clean on both engines |
| BASE-05 on the CUDA 13 engine (RTX PRO 5000): 9 prompts, first-token logits | | identical to the baseline of the unmoved head, 9 of 9 |
| HIP: `cmake -S engine`, gfx1200 and gfx1201, engine, both picture readers and the two programs in `terminal/tools/hip` | | built; the programs compile as they did before the move |
| SYCL: `cmake -S engine/sycl` with icpx, engine compile | | passed (117 build steps) |
| Metal: `tools/packaging/mac/build_engine.sh`, engine + picture reader + the `metal/` folder staged | | passed (Apple M5 Max) |

### The path scan

`checks/path_scan.py` finds paths that a program builds from separate string segments (`join(here, '..', '..', 'serve', 'web')`) or
as a relative include (`#include "../../include/..."`), which the move script cannot rewrite by itself. Run it on the moved tree
after committing the clean head and the moved tree as two throwaway commits:

```sh
python -I checks/path_scan.py <worktree> --explained checks/path_scan_explained.txt     # --before HEAD~1 --after HEAD by default
```

It resolves every run of quoted segments that holds `..`, every `"../x"` string, and every `${VAR}/path` in the CMake files, from the file's old place and its new place.
A hit is a target that exists before the move and is gone after it. It exits with 1 while a hit is left. A hit is either a missed edit
(add it to `move_0216_edits.py`) or explained (a line in `checks/path_scan_explained.txt`, with the reason). The replay of a
release head is done only when it reports no hit.

Run each Windows test file in its own process. In one combined pytest run, `serve/winjob.py` and `serve/lifeline.py` keep the
`subprocess.Popen` that an earlier test may have replaced, and 7 tests fail in both trees for that reason only.

## The file splits (`split/`), after the move

`split/split_server_run.py <terminal folder> <work folder>` splits `serve/server.py` into `serve/server_parts/` by pure moves, on a
checkout that has the layout move. A name moves unchanged only when nothing can tell the difference:

- no test replaces it (`split/patched_py.py` lists the names tests patch, assign or `setattr`, under every spelling and alias of the module);
- nothing rebinds it at run time: no `global` statement, no write through `globals()` or the module's `__dict__`, no assignment to the
  module from other non-test code (a computed key stops the script);
- everything it uses is movable too, or an import the new module repeats.

The moved text is copied unchanged, the original re-exports the names, and the script checks the result itself: every moved
definition's syntax tree equals the original's, every other statement is unchanged, no name is lost or doubled. It also lists the
process-start sites of the moved code in the one test table that counts them. On the head of 11-17 it moves 230 of 299 names (2,495
of 11,227 lines, 22%). `split/names_of.py` prints a module's names and their kinds, to compare before and after;
`split/closure_py.py` is the same analysis without writing anything. `setup.py` is not split: the Terminal archive, the Control stage,
the updater and the release tools read it as one file. The C++ files are only planned (`split/cpp_funcs.py`, `split/big_blocks.py`
list their function blocks).

## Not done by the script

- The text of comments and pages that name old paths (`map0216-prose.tsv`). Each needs a reader; the public pages also need the
  plain-words pass.
- Builds the dry run could not run: Docker images, the Tauri app bundle, Nix.
- Do not run the locale generators `app/host/scripts/make-{fix,gh,pick,r3,tune,ui}-locales.py`: their output files have hand
  edits the generators do not know. The move renames them unchanged.
