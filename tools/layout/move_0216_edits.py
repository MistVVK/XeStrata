"""MAP-0216 (from MAP-0215): the explicit path edits of move_0216.py.

EDITS: (file AFTER the move, old text, new text, expected count): exact replacements.
REGEX_EDITS: (glob over the files AFTER the move, regex, replacement, minimum total count over the glob).
"""
EDITS = []
REGEX_EDITS = []


def rep(path, old, new, count=1):
    EDITS.append((path, old, new, count))


# ------------------------------------------------------------------------------------------- the start files (root)
# a source checkout runs from terminal/ (setup.py's ROOT: .venv, engine/, the run configs); a folder that has no
# terminal/ (an older layout) keeps running where it is
# MAP-0216 (Q7, owner 11-16): a clone made before 0.2.16 has its .venv at the root; it moves into terminal/ here, before Python
# starts (Python cannot move the folder it runs from); setup.py's move_over_old_layout() moves the rest of the old state
rep("setup.sh", 'cd "$(dirname "$0")" || exit 1',
    'cd "$(dirname "$0")" || exit 1\n# the Terminal\'s sources and everything setup writes live in terminal/ (0.2.16 layout); an older clone\'s .venv moves there once\n'
    'if [ -f terminal/setup.py ]; then\n  [ -e terminal/.venv ] || { [ -x .venv/bin/python ] && mv .venv terminal/.venv; }\n  cd terminal || exit 1\nfi')
rep("START-HERE.bat", 'cd /d "%~dp0"',
    'cd /d "%~dp0"\r\nrem the Terminal\'s sources and everything setup writes live in terminal\\ (0.2.16 layout); an older clone\'s .venv moves there once\r\n'
    'if exist "terminal\\setup.py" (\r\n  if not exist "terminal\\.venv" if exist ".venv\\Scripts\\python.exe" move ".venv" "terminal\\.venv" >nul\r\n  cd /d "%~dp0terminal"\r\n)')
rep("START-HERE.bat", "SETUP.bat and UPDATE.bat run this file, so they use it too.", "(START-HERE.bat --setup, --update)")
rep("setup.sh", "update.sh runs this file, so it uses it too.", "(./setup.sh --setup, --update)")
rep("tools/BENCH.bat", 'cd /d "%~dp0"', 'cd /d "%~dp0..\\terminal"')
rep("tools/bench.sh", 'cd "$(dirname "$0")" || exit 1', 'cd "$(dirname "$0")/../terminal" || exit 1')

# ------------------------------------------------------------------------------------------------ tools/check
rep("tools/check", 'HOST = os.path.join(ROOT, "app", "host")','HOST = os.path.join(ROOT, "app", "host")\nTERMINAL = os.path.join(ROOT, "terminal")\nENGINE = os.path.join(ROOT, "engine")')
rep("tools/check", "        rc, _ = run_cmd(cmd, ROOT, env, max(30, t - (time.time() - t0)), log)\n",
    "        rc, _ = run_cmd(cmd, ENGINE, env, max(30, t - (time.time() - t0)), log)\n")
rep("tools/check", 'step_pytest("py-tests", ["tests"], ["--ignore=" + os.path.join(ROOT, "terminal", "tests", "snapshot")])',
    'step_pytest("py-tests", ["terminal/tests"], ["--ignore=" + os.path.join(ROOT, "terminal", "tests", "snapshot")])')
rep("tools/check", 'step_pytest("py-serve", ["serve"])', 'step_pytest("py-serve", ["terminal/serve"])')

# ------------------------------------------------------------------------------------------------ packaging
TERM_ROOT = 'REPO = Path(__file__).resolve().parents[2]\nROOT = REPO / "terminal"          # the Terminal\'s source tree: the same layout as the archive'
for f in ["package_terminal.py", "lock_wheels.py", "monitor_bundle.py", "make_local_feed.py", "make_model_release.py",
          "stage_tuning.py", "feed_tool.py", "package_engine.py"]:
    rep("tools/packaging/" + f, "ROOT = Path(__file__).resolve().parents[1]", TERM_ROOT)
rep("tools/packaging/package_terminal.py", 'LOCKS = ROOT / "packaging" / "locks"', 'LOCKS = REPO / "tools" / "packaging" / "locks"')
rep("tools/packaging/lock_wheels.py", 'LOCKS = ROOT / "packaging" / "locks"', 'LOCKS = REPO / "tools" / "packaging" / "locks"')
rep("tools/packaging/package_terminal.py", "            src = ROOT / item\n", "            src = source_of(item)\n")
rep("terminal/tests/terminal/test_bundle_imports.py", "src, dst = ROOT / item, cls.root / item", "src, dst = pt.source_of(item), cls.root / item")
rep("tools/packaging/package_terminal.py", 'shutil.copy2(source / "metal" / "tools" / f"{name}.py", folder / "metal" / "tools" / f"{name}.py")',
    'shutil.copy2(engine_part(source) / "metal" / "tools" / f"{name}.py", folder / "metal" / "tools" / f"{name}.py")')
rep("tools/packaging/package_terminal.py", "        shutil.copytree(source / name, folder / name, ignore=shutil.ignore_patterns(\"__pycache__\", \"*.pyc\"))",
    "        shutil.copytree(engine_part(source) / name, folder / name, ignore=shutil.ignore_patterns(\"__pycache__\", \"*.pyc\"))")
rep("tools/packaging/package_terminal.py", "def add_installer(folder: Path, source: Path = ROOT) -> dict:",
    "def engine_part(source: Path) -> Path:\n    \"\"\"Where the engine-side installer files (metal/tools, third_party/strata-gguf-py) are: engine/ of the checkout, or a tree in the\n    archive's layout (a test's fake tree, a staged Terminal).\"\"\"\n    return REPO / \"engine\" if Path(source).resolve() == ROOT.resolve() else source\n\n\ndef source_of(item: str) -> Path:\n    \"\"\"The source of one PAYLOAD item: the archive keeps the Terminal's paths; docs/ is the repository's.\"\"\"\n    return (REPO if item.startswith(\"docs/\") else ROOT) / item\n\n\ndef add_installer(folder: Path, source: Path = ROOT) -> dict:")
rep("tools/packaging/package_terminal.py", '"lock": {"file": lock.relative_to(ROOT).as_posix()', '"lock": {"file": lock.relative_to(REPO).as_posix()')
rep("tools/packaging/package_terminal.py", 'text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")',
    'text = (REPO / "engine" / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")')
rep("tools/packaging/monitor_bundle.py", "shutil.copyfile(ROOT / LICENSE_NOTICE, stage / LICENSE_NOTICE)", "shutil.copyfile(REPO / LICENSE_NOTICE, stage / LICENSE_NOTICE)")
rep("tools/packaging/make_local_feed.py", 'sys.path.insert(0, str(ROOT / "packaging"))', 'sys.path.insert(0, str(REPO / "tools" / "packaging"))')
rep("tools/packaging/make_local_feed.py", 'text = (ROOT / "packaging" / name).read_text', 'text = (REPO / "tools" / "packaging" / name).read_text')
rep("tools/packaging/make_model_release.py", 'TEMPLATES = ROOT / "packaging" / "templates" / "model-release"', 'TEMPLATES = REPO / "tools" / "packaging" / "templates" / "model-release"')
rep("tools/packaging/feed_tool.py", 'conf = Path(__file__).resolve().parents[1] / "apps" / "desktop" / "src-tauri" / "tauri.conf.json"',
    'conf = Path(__file__).resolve().parents[2] / "app" / "desktop" / "src-tauri" / "tauri.conf.json"')
rep("tools/packaging/build_cuda_windows.py", "REPO = Path(__file__).resolve().parents[1]", "REPO = Path(__file__).resolve().parents[2]")
rep("tools/docker/fetch-engines.py", "REPO_ROOT = Path(__file__).resolve().parent.parent", "REPO_ROOT = Path(__file__).resolve().parents[2]")
# build scripts one level deeper: the checkout is ../.. now; the engine is configured from engine/
for f in ["build_hip_linux.sh", "build_split_cuda.sh", "build_split_hip.sh"]:
    rep("tools/packaging/" + f, '$(cd "$(dirname "$0")/.." && pwd)', '$(cd "$(dirname "$0")/../.." && pwd)')
rep("tools/packaging/mac/build_engine.sh", 'ROOT=$(cd "$(dirname "$0")/../.." && pwd)', 'ROOT=$(cd "$(dirname "$0")/../../.." && pwd)')
rep("tools/packaging/build_vision_windows.bat", 'for %%I in ("%~dp0..") do set "SRC=%%~fI"', 'for %%I in ("%~dp0..\\..") do set "SRC=%%~fI"')
rep("engine/spark/build_engine.sh", 'repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)', 'repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)')
REGEX_EDITS.append(("tools/packaging/*.sh", r"(cmake[^\n]*? -S )\.(?=[ \\\n])", r"\1engine", 1))
REGEX_EDITS.append(("engine/spark/build_engine.sh", r"(-S )\.(?=[ \\\n])", r"\1engine", 1))
REGEX_EDITS.append(("tools/packaging/shape_build.sh", r"\$1/include/", r"$1/engine/include/", 1))

# ------------------------------------------------------------------------------------------------ the app
rep("app/desktop/scripts/stage-control.py", 'sys.path.insert(0, str(ROOT / "packaging"))', 'sys.path.insert(0, str(ROOT / "tools" / "packaging"))')
rep("app/desktop/scripts/stage-control.py", 'command = [sys.executable, str(ROOT / "packaging" / "package_terminal.py"), "--out", str(work)]',
    'command = [sys.executable, str(ROOT / "tools" / "packaging" / "package_terminal.py"), "--out", str(work)]')
rep("app/desktop/scripts/app-close-matrix.py", "ROOT = Path(__file__).resolve().parents[3]\nsys.path.insert(0, str(ROOT))",
    'ROOT = Path(__file__).resolve().parents[3] / "terminal"\nsys.path.insert(0, str(ROOT))')
rep("app/host/server/index.mjs", "// `HERE` is apps/host/server, so the repository root is three levels up (server -> host -> apps).\nconst REPO_ROOT = CONTROL ? CONTROL.dir : resolve(HERE, '..', '..', '..');",
    "// `HERE` is app/host/server, so the repository root is three levels up (server -> host -> app); the Terminal's sources are in its terminal/.\nconst REPO_ROOT = CONTROL ? CONTROL.dir : resolve(HERE, '..', '..', '..', 'terminal');")
rep("app/host/server/native-control.mjs", "function repo(){if(packaged)return installRoot();const path=realpathSync(preferences.repo);",
    "function repo(){if(packaged)return installRoot();let path=realpathSync(preferences.repo);if(existsSync(join(path,'terminal','setup.py')))path=join(path,'terminal');")
for f in ["agent-lost-e2e.mjs", "context-e2e.mjs"]:
    rep("app/host/scripts/" + f, "const REPO = join(import.meta.dirname, '..', '..', '..');", "const REPO = join(import.meta.dirname, '..', '..', '..', 'terminal');")

# ------------------------------------------------------------------------------------------------ the Terminal's code
# setup.py: the engine's sources (CMake, third_party, sycl, metal) are ../engine in a checkout; the archive has its few in ROOT
SETUP_HEAD = '''

def engine_src() -> Path:
    """The engine's sources: ../engine in a source checkout (0.2.16 layout); ROOT itself in a tree that has them there (an
    older checkout, a test's tree) or has none (an installed Terminal keeps its few engine-side files, metal/tools and
    third_party/strata-gguf-py, in ROOT).  A function, like home(): ROOT is the one thing a caller can point elsewhere."""
    if (ROOT / "CMakeLists.txt").is_file() or not (ROOT.parent / "engine" / "CMakeLists.txt").is_file():
        return ROOT
    return ROOT.parent / "engine"


def strata_folder() -> Path:
    """The folder a person sees as "Strata" (the checkout, or the install): the data folder Strata-data sits next to it."""
    return ROOT.parent if engine_src() != ROOT else ROOT
'''
rep("terminal/setup.py", "ROOT = Path(__file__).resolve().parent\n", "ROOT = Path(__file__).resolve().parent\n" + SETUP_HEAD)
rep("terminal/setup.py", '(ROOT / "metal" / "tools").resolve()', '(engine_src() / "metal" / "tools").resolve()')
rep("terminal/setup.py", 'VENDORED_GGUF_PY = ROOT / "third_party" / "strata-gguf-py"', 'VENDORED_GGUF_PY = engine_src() / "third_party" / "strata-gguf-py"')
rep("terminal/setup.py", 'VENDORED_LLAMA = ROOT / "third_party" / "strata-llama"', 'VENDORED_LLAMA = engine_src() / "third_party" / "strata-llama"')
rep("terminal/setup.py", 'cmake_build(ROOT, ROOT / "build-hip", "strata",', 'cmake_build(engine_src(), ROOT / "build-hip", "strata",')
rep("terminal/setup.py", 'cmake_build(ROOT, bdir, "strata",', 'cmake_build(engine_src(), bdir, "strata",')
rep("terminal/setup.py", 'home().parent / "Strata-data"', '(strata_folder() if home() == ROOT else home()).parent / "Strata-data"', 4)
rep("terminal/setup.py", "home().parent / 'Strata-data'", "(strata_folder() if home() == ROOT else home()).parent / 'Strata-data'")
# the tools' GGUF library: the vendored gguf-py is in engine/third_party in a checkout
rep("terminal/tools/_paths.py", 'VENDORED = ENGINE / "third_party" / "strata-gguf-py"',
    'def _vendored() -> Path:\n    """ENGINE/third_party/strata-gguf-py (an installed Terminal, a test\'s tree), else the source checkout\'s engine/third_party (0.2.16 layout)."""\n'
    '    own = ENGINE / "third_party" / "strata-gguf-py"\n    beside = ENGINE.parent / "engine" / "third_party" / "strata-gguf-py"\n'
    '    return beside if not own.is_dir() and beside.is_dir() else own\n\n\nVENDORED = _vendored()')
rep("terminal/tools/_paths.py", 'cands = [os.environ.get("STRATA_GGUF_PY"), ENGINE / "third_party" / "strata-gguf-py",',
    'cands = [os.environ.get("STRATA_GGUF_PY"), _vendored(),')
rep("terminal/tools/_paths.py", 'for c in (ENGINE / "third_party" / "strata-gguf-py" / "converter",', 'for c in (_vendored() / "converter",')
# the version of a source tree: engine/CMakeLists.txt
rep("terminal/strata_cli/__init__.py", '(root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")',
    '(root / "CMakeLists.txt" if (root / "CMakeLists.txt").is_file() else root.parent / "engine" / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")')

# CI: the presets live in engine/; the serve tests are terminal/serve; setup.py is imported from terminal/
REGEX_EDITS.append((".github/workflows/*.yml", r"cmake --preset ", "cmake -S engine --preset ", 2))
REGEX_EDITS.append((".github/workflows/*.yml", r"discover -s serve ", "discover -s terminal/serve ", 1))
rep(".github/workflows/check.yml", "python -c \"import setup; print(repr(setup.cpu_info()))\"", "python -c \"import sys; sys.path.insert(0, \x27terminal\x27); import setup; print(repr(setup.cpu_info()))\"")
rep("tools/check", 'step_pytest("py-tools", ["terminal/tools/test_*.py"])', 'step_pytest("py-tools", ["terminal/tools/test_*.py", "tools/test_*.py"])')
# what setup.py downloads into ROOT/third_party (stable-diffusion.cpp, the Vulkan SDK) is in terminal/ now
rep(".gitignore", "/engine/third_party/*.zip\n", "/engine/third_party/*.zip\n/terminal/third_party/*.zip\n")
rep(".gitignore", "/engine/third_party/llama.cpp\n", "/engine/third_party/llama.cpp\n/terminal/third_party/llama.cpp\n")
# repository checks: folder lists and joins that are not ROOT-anchored
rep("tools/settings_register.py", 'ENGINE_DIRS = ("src", "include", "engine/metal/src", "engine/sycl/src")', 'ENGINE_DIRS = ("engine/src", "engine/include", "engine/metal/src", "engine/sycl/src")')
rep("tools/check_vendored.py", 'os.path.join("third_party", ', 'os.path.join("engine", "third_party", ', 4)
# a generated page: its template carries the links the link pass fixed in the page itself
rep("tools/bench_results_index.py", "[docs](../../docs/README.md) and the [README](../../README.md); how to measure is in [BENCHMARK_PROTOCOL.md](../../docs/BENCHMARK_PROTOCOL.md).", "[docs](../README.md) and the [README](../../README.md); how to measure is in [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md).")

# engine CMake: the checks it runs are in tools/, the HIP tuners, the vision sources and the i-quant fixture writer in terminal/tools/
REGEX_EDITS.append(("engine/cmake/backends/*.cmake", r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/tools/(check_isa_guard|check_no_mmq|check_cuda_spills)\.py", r"${CMAKE_CURRENT_SOURCE_DIR}/../tools/\1.py", 5))
REGEX_EDITS.append(("engine/cmake/backends/hip.cmake", r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/tools/vision/", r"${CMAKE_CURRENT_SOURCE_DIR}/../terminal/tools/vision/", 1))
REGEX_EDITS.append(("engine/cmake/strata_gpu_engine.cmake", r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/tools/iq_fixture\.py", r"${CMAKE_CURRENT_SOURCE_DIR}/../terminal/tools/iq_fixture.py", 1))
REGEX_EDITS.append(("engine/cmake/strata_gpu_engine.cmake", r"add_executable\((tune_hipblaslt|tune_rocblas) tools/hip/", r"add_executable(\1 ../terminal/tools/hip/", 2))
rep("engine/src/core/amd_community_test.cpp", "std::string(\"tools/hip/\")", "std::string(\"../terminal/tools/hip/\")")

# app/host tests that build a repository root from several parts
rep("app/host/tests/kit-copy.test.mjs", "resolve(repo,'serve/web/components.css')", "resolve(repo,'terminal/serve/web/components.css')")
rep("app/host/tests/fit-label-rule.test.mjs", "const repo = resolve(root, '../..');", "const repo = resolve(root, '../..', 'terminal');")
rep("app/host/tests/own-path-fit.test.mjs", "const repo = resolve(root, '../..');", "const repo = resolve(root, '../..', 'terminal');")
rep("app/host/tests/staging-scripts-syntax.test.mjs", "resolve(import.meta.dirname, '..', '..', '..', 'packaging')", "resolve(import.meta.dirname, '..', '..', '..', 'tools', 'packaging')")
rep("app/host/tests/timeskip-real.test.mjs", "const REPO = resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..', '..');", "const REPO = resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..', '..', 'terminal');")
rep("app/host/tests/timeskip-real.test.mjs", "join(REPO, 'apps', 'host', 'server', 'timeskip-revise.mjs')", "join(REPO, '..', 'app', 'host', 'server', 'timeskip-revise.mjs')")
# the snapshot driver gets the Terminal root (STRATA_SNAPSHOT_REPO = terminal/) and loads the app host beside it
rep("terminal/tests/snapshot/app_driver.mjs", "join(repo, 'apps/host/server/", "join(repo, '../app/host/server/", 2)

# tools/check: the Python steps run from terminal/ (the tests start tools by path and import tools.x, as from the old root)
rep("tools/check", '"snapshot")])', '"snapshot")], cwd=TERMINAL)')
rep("tools/check", 'step_pytest("py-serve", ["terminal/serve"])', 'step_pytest("py-serve", ["terminal/serve"], cwd=TERMINAL)')
rep("tools/check", '["terminal/tools/test_*.py", "tools/test_*.py"])', '["terminal/tools/test_*.py", "tools/test_*.py"], cwd=TERMINAL)')
rep("tools/check", 'step_pytest("py-optional", ["terminal/tools/dflash2", "terminal/tools/trellis"])(ctx)', 'step_pytest("py-optional", ["terminal/tools/dflash2", "terminal/tools/trellis"], cwd=TERMINAL)(ctx)')
rep("tools/check", 'step_pytest("snapshot", ["terminal/tests/snapshot"], need_node=True)', 'step_pytest("snapshot", ["terminal/tests/snapshot"], need_node=True, cwd=TERMINAL)')

# the image build (tools/docker/build.sh + tools/docker/Dockerfile): the context keeps the repository layout of what it copies
rep("tools/docker/build.sh", 'repo=$(CDPATH= cd -- "$here/.." && pwd -P)', 'repo=$(CDPATH= cd -- "$here/../.." && pwd -P)')
rep("tools/docker/build.sh", '"$repo/CMakeLists.txt"', '"$repo/engine/CMakeLists.txt"')
rep("tools/docker/build.sh", 'paths="packaging bin strata.py strata_control strata_cli serve requirements.txt requirements-monitor.txt docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md data bench/prompts/standard-v1 setup.py ref/load.py CMakeLists.txt metal/tools/metal_repack.py metal/tools/stream_experts.py"', 'paths="tools/packaging terminal/bin terminal/strata.py terminal/strata_control terminal/strata_cli terminal/serve terminal/requirements.txt terminal/requirements-monitor.txt docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md terminal/data terminal/bench/prompts/standard-v1 terminal/setup.py terminal/ref/load.py engine/CMakeLists.txt engine/metal/tools/metal_repack.py engine/metal/tools/stream_experts.py"')
rep("tools/docker/build.sh", "':(glob)tools/**/*.py'", "':(glob)terminal/tools/**/*.py'")
rep("tools/docker/build.sh", "tools_py=$(find tools -name '*.py')", "tools_py=$(find terminal/tools -name '*.py')")
rep("tools/docker/Dockerfile", "RUN python3 /src/packaging/package_terminal.py", "RUN python3 /src/tools/packaging/package_terminal.py")
rep("tools/docker/Dockerfile", "COPY terminal-src/packaging/package_engine.py terminal-src/packaging/binary_deps.py", "COPY terminal-src/tools/packaging/package_engine.py terminal-src/tools/packaging/binary_deps.py")

# two shipped tools whose ROOT joins must work in both layouts (the installed Terminal has docs/ and bench/ inside ROOT)
rep("terminal/tools/calibrate.py", 'sorted((ROOT.parent / "docs").glob("*.md"))', 'sorted((ROOT / "docs" if (ROOT / "docs").is_dir() else ROOT.parent / "docs").glob("*.md"))')
rep("terminal/tools/strata_bench.py", 'ROOT.parent / "docs" / "bench-results"', '(ROOT.parent / "docs" / "bench-results" if (ROOT.parent / "docs" / "bench-results").is_dir() else ROOT / "bench" / "results")')


# ================================================================================ MAP-0216: found on the 0.2.16 integration tree
# (each was a path use the MAP-0215 dry run missed or one that appeared since; see MAP-0216.md)
# setup.py's engine fingerprint (#31: a pull that changes the sources rebuilds the engine) hashed ROOT / "src" ...: with ROOT =
# terminal/ it hashed nothing, so a changed engine source would never rebuild.  The engine parts are read from engine_src(), the
# reader's tower from ROOT (terminal/tools/vision); each part's paths stay relative to its own top, so the hash of an unchanged
# tree is the same as before the move (no needless rebuild after the pull).
rep("terminal/setup.py", r"""    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))""",
    r"""    for part in parts:
        top = ROOT if part.startswith("tools/") else engine_src()
        base = top / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(top).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))""")
# the SYCL engine's CTest fixture writer and the Metal engine's data default: STRATA_ROOT is engine/ there
rep("engine/sycl/CMakeLists.txt", "${STRATA_ROOT}/tools/iq_fixture.py", "${STRATA_ROOT}/../terminal/tools/iq_fixture.py")
rep("engine/metal/CMakeLists.txt", 'STRATA_METAL_DATA_DEFAULT="${STRATA_ROOT}/data")', 'STRATA_METAL_DATA_DEFAULT="${STRATA_ROOT}/../terminal/data")')

# release build scripts one level deeper (MAP-0215 had the `cd "$(dirname "$0")/.."` form; these use `CDPATH= cd --`)
for f in ["build_cuda_linux.sh", "build_cuda_linux_aarch64.sh", "build_sycl_linux.sh"]:
    rep("tools/packaging/" + f, 'repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)', 'repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)')
# the SYCL and Metal engines are configured from engine/sycl and engine/metal (a bare folder name is not mapped by the token pass)
rep("tools/packaging/build_sycl_linux.sh", "cmake -S sycl ", "cmake -S engine/sycl ")
rep("tools/packaging/mac/build_engine.sh", '"$CMAKE" -S metal ', '"$CMAKE" -S engine/metal ')
REGEX_EDITS.append((".github/workflows/*.yml", r"cmake -S (sycl|metal) ", r"cmake -S engine/\1 ", 3))
# the Mac engine PACKAGE keeps its layout (metal/kernels ... beside the executable): the token pass rewrote two texts that name it
rep("tools/packaging/mac/build_engine.sh",
    "#   engine/metal/kernels/*.metal           engine/metal/engine-msl/*.metal          engine/metal/data/{expert-profile.bin,draft_vocab.bin}",
    "#   metal/kernels/*.metal           metal/engine-msl/*.metal          metal/data/{expert-profile.bin,draft_vocab.bin}")
rep("tools/packaging/mac/build_engine.sh", 'echo "engine/metal/$n in the package is not the header text"', 'echo "metal/$n in the package is not the header text"')

# ---- round 5 (owner PC, the Windows test files of round 4)
# the app's Control stage: the installer files come from the Terminal's tree (setup.py, ref/, tools/ are in terminal/), and
# package_terminal's engine_part() takes the engine-side files from engine/ only when it is given its own ROOT
rep("app/desktop/scripts/stage-control.py", "    return pt.add_installer(folder, ROOT)", "    return pt.add_installer(folder, pt.ROOT)")
# tests that read an engine source or a docs page beside setup.py
rep("terminal/tools/test_setup_amd.py", 'Path(setup.__file__).resolve().parent / "src/program/generate.cpp"',
    'Path(setup.__file__).resolve().parent.parent / "engine" / "src/program/generate.cpp"')
rep("terminal/tools/test_setup_amd.py", 'Path(setup.__file__).resolve().parent / "docs/TROUBLESHOOTING.md"',
    'Path(setup.__file__).resolve().parent.parent / "docs/TROUBLESHOOTING.md"')
# tests of the start files: BENCH.bat / bench.sh are in tools/, START-HERE.bat at the root; SETUP.bat, UPDATE.bat and update.sh are gone (Q6)
rep("terminal/tests/terminal/test_bench_standard.py", '(self.ROOT / name).read_text(encoding="utf-8"))', '(self.ROOT.parent / "tools" / name).read_text(encoding="utf-8"))')
rep("terminal/tests/terminal/test_packaging_pins.py", '        self.assertIn("START-HERE.bat", (ROOT / "SETUP.bat").read_text(encoding="utf-8"))\n        self.assertIn("START-HERE.bat", (ROOT / "UPDATE.bat").read_text(encoding="utf-8"))\n', "")
rep("terminal/tests/terminal/test_packaging_pins.py", '        self.assertIn("setup.sh", (ROOT / "update.sh").read_text(encoding="utf-8"))\n', "")
rep("terminal/tools/test_cyrillic_paths.py", 'for name in ("START-HERE.bat", "bin/strata.cmd", "bin/strata.ps1"):', 'for name in ("../START-HERE.bat", "bin/strata.cmd", "bin/strata.ps1"):')

# ---- round 6 (p1006, tools/check on the round-4 tree: 27 py-tests, 40 py-tools and 2 host-test failures; each fixed here)
PATH_SPANS = []      # (file, start text, end text, minimum count): quoted repository paths inside the span are mapped (move_0216.py)

# shipped code that reads the engine's version or sources: both layouts (an installed Terminal has CMakeLists.txt in its own folder,
# a 0.2.16 checkout has it in ../engine)
rep("terminal/strata_control/updater.py", '(root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")',
    '(root / "CMakeLists.txt" if (root / "CMakeLists.txt").is_file() else root.parent / "engine" / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")')
rep("terminal/tools/strata_bench.py", '(ROOT.parent / "engine" / "CMakeLists.txt").read_text(encoding="utf-8")',
    '(ROOT / "CMakeLists.txt" if (ROOT / "CMakeLists.txt").is_file() else ROOT.parent / "engine" / "CMakeLists.txt").read_text(encoding="utf-8")')
for f in ["terminal/tools/image/sdcpp_dit/apply.py", "terminal/tools/image/sdcpp_vae/apply.py"]:
    rep(f, "REPO = HERE.parents[2]\n",
        "REPO = HERE.parents[2] if (HERE.parents[2] / \"src\").is_dir() else HERE.parents[3] / \"engine\"   # the engine's sources (../engine in a 0.2.16 checkout)\n")
rep("terminal/tools/gen_strata_quant_tables.py", 'HEADER = ROOT / "third_party" / "strata-ggml"',
    'HEADER = (ROOT if (ROOT / "third_party").is_dir() else ROOT.parent / "engine") / "third_party" / "strata-ggml"')
# the image build helper imports the Terminal's packages
rep("tools/docker/fetch-engines.py", "sys.path.insert(0, str(REPO_ROOT))\n", "sys.path.insert(0, str(REPO_ROOT / \"terminal\"))\n")
# the SYCL release build: its build folder is the engine's (engine/sycl/setup_intel.py and strata-sycl.sh take engine/ as their repo now)
rep("tools/packaging/build_sycl_linux.sh", "build=${BUILD:-$repo/build-sycl}", "build=${BUILD:-$repo/engine/build-sycl}")

# tests that read engine sources by a table of engine-relative paths
rep("terminal/tools/test_qwen4_intake.py", 'return (ROOT / rel).read_text(encoding="utf-8")', 'return (ROOT.parent / "engine" / rel).read_text(encoding="utf-8")')
rep("terminal/tests/terminal/test_shape_engine.py", "        src = (ROOT / rel).read_text(encoding=\"utf-8\")\n        start = src.index('std::strcmp(argv[i], \"--version\") == 0')",
    "        src = (ROOT.parent / \"engine\" / rel).read_text(encoding=\"utf-8\")\n        start = src.index('std::strcmp(argv[i], \"--version\") == 0')")
rep("terminal/tests/split/test_split_engine.py", "text = (ROOT / self.HIP_PEER_FILE).read_text(", 'text = (ROOT.parent / "engine" / self.HIP_PEER_FILE).read_text(')
rep("terminal/tests/split/test_split_engine.py", "p.relative_to(ROOT).as_posix()", 'p.relative_to(ROOT.parent / "engine").as_posix()', 2)
rep("terminal/tests/sycl/test_setup_intel_prefill.py", 'ROOT / "build-sycl-aot" / "strata"', 'si.ROOT / "build-sycl-aot" / "strata"', 2)
# tests that read repository files by a table of repository paths
PATH_SPANS.append(("terminal/tools/test_engine_targets.py", "SCRIPTS = {", "}\n", 2))
rep("terminal/tools/test_engine_targets.py", "            text = (ROOT / rel).read_text(encoding=\"utf-8\")\n            self.assertNotIn(\"--target strata strata_kq\"",
    "            text = (ROOT.parent / rel).read_text(encoding=\"utf-8\")\n            self.assertNotIn(\"--target strata strata_kq\"")
PATH_SPANS.append(("terminal/tools/test_no_llama_download.py", "SCRIPTS = [", "]\n", 8))
rep("terminal/tools/test_no_llama_download.py", 'text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")', 'text = (ROOT.parent / rel).read_text(encoding="utf-8", errors="replace")')
rep("terminal/tests/terminal/test_install_scripts_env.py", 'for name in ("README.md", "README.de.md", "docs/INSTALL.md", "docs/TERMINAL_LAUNCH.md"):\n            text = (ROOT / name)',
    'for name in ("README.md", "docs/lang/README.de.md", "docs/INSTALL.md", "docs/TERMINAL_LAUNCH.md"):\n            text = (ROOT.parent / name)')
rep("terminal/tests/terminal/test_launcher_runnable.py", '("docker/entrypoint.sh", "python_usable")', '("../tools/docker/entrypoint.sh", "python_usable")')
rep("terminal/tests/terminal/test_tui2_023_versions.py", 'first("CMakeLists.txt", ', 'first("../engine/CMakeLists.txt", ')
rep("terminal/tests/terminal/test_tui2_023_versions.py", 'first("apps/desktop/src-tauri/Cargo.toml", ', 'first("../app/desktop/src-tauri/Cargo.toml", ')
rep("terminal/tests/terminal/test_tui2_023_versions.py",
    'for name in ("README.md", "README.de.md", "README.es.md", "README.fr.md", "README.ja.md", "README.pl.md", "README.pt-BR.md", "README.zh-CN.md"):\n            text = (ROOT / name)',
    'for name in ("README.md", "docs/lang/README.de.md", "docs/lang/README.es.md", "docs/lang/README.fr.md", "docs/lang/README.ja.md", "docs/lang/README.pl.md", "docs/lang/README.pt-BR.md", "docs/lang/README.zh-CN.md"):\n            text = (ROOT.parent / name)')
rep("terminal/tests/terminal/test_tui_stress_launch.py", 'os.path.join(root, "docs", "TERMINAL_LAUNCH.md")', 'os.path.join(root, "..", "docs", "TERMINAL_LAUNCH.md")')
rep("terminal/tests/terminal/test_speed_switches.py", "from './apps/host/server/native-settings.mjs'", "from '../app/host/server/native-settings.mjs'")
rep("terminal/tests/terminal/test_terminal_commit.py", 'text.index("RUN python3 /src/packaging/package_terminal.py")', 'text.index("RUN python3 /src/tools/packaging/package_terminal.py")')
rep("terminal/tools/test_quant_header_names.py", "'QUANT=include/strata/quant/strata-quant-common.h'", "'QUANT=engine/include/strata/quant/strata-quant-common.h'")
# the image build's context keeps the repository layout of what it copies (tools/docker/build.sh)
rep("terminal/tests/terminal/test_docker_files.py", '(ctx / "terminal-src" / "tools" / f"{name}.py")', '(ctx / "terminal-src" / "terminal" / "tools" / f"{name}.py")')
rep("terminal/tests/terminal/test_docker_files.py", '(ctx / "terminal-src" / "metal" / "tools" / f"{name}.py")', '(ctx / "terminal-src" / "engine" / "metal" / "tools" / f"{name}.py")')
rep("terminal/tests/terminal/test_docker_files.py", "        for item in payload:\n            self.assertTrue(any(item == p",
    "        for item in payload:\n            item = item if item.startswith(\"docs/\") else \"terminal/\" + item     # the payload is the Terminal's; build.sh lists repository paths\n            self.assertTrue(any(item == p")
# the app's IME test reads the web page's copy of the rule
rep("app/host/tests/ime-fields.test.mjs", "resolve(root, '../../serve/web/ime.js')", "resolve(root, '../../terminal/serve/web/ime.js')")

# ---- round 7 (m1002: the Mac engine package build of round 6 stopped at strata-vision, "src/tools/tools/vision does not exist")
# scripts that find the checkout from their own folder, one level deeper now
rep("tools/packaging/build_vision_mac.sh", 'cmake -S "$here/../tools/vision"', 'cmake -S "$here/../../terminal/tools/vision"')
rep("tools/docker/publish.sh", 'repo=$(CDPATH= cd -- "$here/.." && pwd -P)', 'repo=$(CDPATH= cd -- "$here/../.." && pwd -P)')

# ---- round 8 (m1002: strata-vision of round 7 stopped at 'stb_image.h' file not found; Windows per-file run of round 7: 1 failure)
# the picture reader's CMake files reach the engine's vendored sources by relative path; the reader is in terminal/tools/vision now
rep("terminal/tools/vision/own/CMakeLists.txt", '"${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb"', '"${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine/third_party/stb"')
rep("terminal/tools/vision/CMakeLists.txt", '"${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/strata-llama"', '"${CMAKE_CURRENT_SOURCE_DIR}/../../../engine/third_party/strata-llama"')
# tests that read an engine source beside setup.py (the round-5 edit took the first one in test_setup_amd.py)
REGEX_EDITS.append(("terminal/tools/test_*.py", r'Path\(setup\.__file__\)\.resolve\(\)\.parent / "(src|include)/', r'Path(setup.__file__).resolve().parent.parent / "engine" / "\1/', 1))

# ---- round 9 (p1006, tools/check on the round-6 tree: py-tests 3655/0 as before; py-tools 5 and host-test 1 failures, fixed here)
rep("terminal/tools/test_engine_targets.py", """'. "$repo/packaging/engine_targets.sh"'""", """'. "$repo/tools/packaging/engine_targets.sh"'""")
for f in ["terminal/tools/test_qimage.py", "terminal/tools/test_sdcpp_dit.py", "terminal/tools/test_sdcpp_vae.py"]:
    rep(f, "(ROOT / src).read_bytes()", '(ROOT.parent / "engine" / src).read_bytes()')      # the decoders' sources are the engine's
# the app's dev checkout (preferences.repo) is the repository; the Terminal's files are under repo(), which maps it to terminal/
rep("app/host/server/native-control.mjs", "preferences.repo,'serve/server.py')", "repo(),'serve/server.py')")
rep("app/host/server/native-control.mjs", "preferences.repo,'engine/.previous')", "repo(),'engine/.previous')")
rep("app/host/server/native-control.mjs", "preferences.repo,'setup.py')", "repo(),'setup.py')")
# the host test that starts the app host from the checkout (it crashed Node with an assertion when its cwd did not exist)
rep("app/host/tests/tune-and-timeskip-routes.test.mjs", "join(REPO, 'apps/host/dist')", "join(REPO, 'app/host/dist')")
rep("app/host/tests/tune-and-timeskip-routes.test.mjs", "cwd: join(REPO, 'apps/host'),", "cwd: join(REPO, 'app/host'),")

# ---- round 10 (l620, py-tools of round 9: 3 failures; the path tables now map only paths and files, not "cmake" as a word)
rep("terminal/tools/test_qimage.py", "for f in (ROOT / mod.HIP_COMPAT).iterdir():", 'for f in (ROOT.parent / "engine" / mod.HIP_COMPAT).iterdir():')
# the round-8 edit of the picture reader's CMake file: the test names its text
rep("terminal/tools/test_no_llama_download.py", """'set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/strata-llama"'""",
    """'set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../engine/third_party/strata-llama"'""")

# ---- round 11 (l620, round 10 on bb59dc86e: tests added since the first base read repository paths from the Terminal root)
rep("terminal/tests/terminal/test_engine_deps.py", 'for cmake in ("metal/CMakeLists.txt", "tools/vision/CMakeLists.txt"):\n            self.assertIn(f\'CMAKE_OSX_DEPLOYMENT_TARGET "{floor}"\', (ROOT / cmake)',
    'for cmake in ("engine/metal/CMakeLists.txt", "terminal/tools/vision/CMakeLists.txt"):\n            self.assertIn(f\'CMAKE_OSX_DEPLOYMENT_TARGET "{floor}"\', (ROOT.parent / cmake)')
rep("terminal/tests/terminal/test_engine_deps.py", 'for script, times in (("packaging/mac/build_engine.sh", 1), ("apps/desktop/scripts/make-dmg.sh", 1),\n                              ("packaging/mac/sign-and-notarize.sh", 1)):\n            text = (ROOT / script)',
    'for script, times in (("tools/packaging/mac/build_engine.sh", 1), ("app/desktop/scripts/make-dmg.sh", 1),\n                              ("tools/packaging/mac/sign-and-notarize.sh", 1)):\n            text = (ROOT.parent / script)')

# ---- round 13 (p1006 host-test on 8e6b18397: the web locale tests added since read serve/web/locales by path segments)
rep("app/host/tests/german-sie.test.mjs", "join(here, '..', '..', '..', 'serve', 'web', 'locales', 'de.json')", "join(here, '..', '..', '..', 'terminal', 'serve', 'web', 'locales', 'de.json')")
# found by the Mac build of the rehearsal (m1002, 11-17 03:32): the Metal CMake file's parity test names the reference decoder by ${STRATA_ROOT}/ref/...; ref/ is the Terminal's now
rep("engine/metal/CMakeLists.txt", "${STRATA_ROOT}/ref/tqp/tqp_ref_c.c", "${STRATA_ROOT}/../terminal/ref/tqp/tqp_ref_c.c", 2)
# found by checks/path_scan.py on 19f8643c1: two sources of the Terminal's tools include engine headers by a relative path (#5060 added the second)
rep("terminal/tools/hip/tune_hipblaslt.cpp", '#include "../../include/strata/kernels/bf16_bits.hpp"', '#include "../../../engine/include/strata/kernels/bf16_bits.hpp"')
rep("terminal/tools/vision/strata_vision.cpp", '#include "../../include/strata/core/kpack_path.hpp"', '#include "../../../engine/include/strata/core/kpack_path.hpp"')
for f in ["locale-complete", "locale-reviewed", "french-typography-all"]:
    rep("app/host/tests/" + f + ".test.mjs", "join(here, '..', '..', '..', 'serve', 'web', 'locales')", "join(here, '..', '..', '..', 'terminal', 'serve', 'web', 'locales')")
