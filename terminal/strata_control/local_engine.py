"""`strata components add-build FOLDER` (#5114): an engine built on this computer (`./setup --build`) as an engine package of its own.

The released engines are signed packages from the feed; a PC they cannot run on (a CPU without AVX, #5114) can build its own. This turns such a
build folder into a package beside the others: the entrypoint must run (`strata --version` within ADD_TIMEOUT_S; an engine that dies, for example
with an illegal instruction, is refused with that reason), its family (backend) and card architectures come from its BUILD.json (or --backend),
a family this PC cannot run is refused, the files are copied into <components>/engine/<version>-<platform>-<backend>-local/ with a manifest
that lists every file's SHA-256 and says "local": true and no signature. It becomes the engine in use by the person's own choice (the same
switch as `strata components use`). A local package is never picked by itself: only while it is the chosen one (engine_component.resolve).
`strata components use newest` goes back to the released engines and `strata components remove <folder>` deletes it.
STRATA_REQUIRE_SIGNED_COMPONENTS=1 refuses it like any unsigned package. Nothing is downloaded and nothing outside the components folder changes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from strata_control import engine_component as ec
from strata_control.targets import backend_supported, host_platform

ADD_TIMEOUT_S = 10.0
LOCAL_SUFFIX = "-local"
#: the files a build folder contributes: the entrypoint, its runtime libraries beside it, the picture readers, its build record
_LIB = re.compile(r"\.(so(\.\d+)*|dll|dylib)$", re.IGNORECASE)
_HELPERS = ("strata-vision", "strata-vision.exe", "strata-vision-hip", "strata-vision-hip.exe", "strata-vision-sycl", "strata-vision-sycl.exe")


class AddBuildError(ValueError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _entrypoint(folder: Path) -> Path:
    for name in ("strata.exe", "strata") if os.name == "nt" else ("strata",):
        if (folder / name).is_file():
            return folder / name
    raise AddBuildError(f"{folder} has no engine program (strata{'.exe' if os.name == 'nt' else ''}): give the folder `./setup --build` wrote it to", 2)


def _version_of(exe: Path, folder: Path) -> str:
    from strata_control.childenv import quiet_crash_env
    env = quiet_crash_env(dict(os.environ))               # a Strata program: no debugger dialog or backtrace on a crash (childenv)
    var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    env[var] = str(folder) + os.pathsep + env.get(var, "")
    try:
        done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", timeout=ADD_TIMEOUT_S, env=env, cwd=str(folder),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise AddBuildError(f"the engine did not answer `--version` within {ADD_TIMEOUT_S:g} seconds, so it was not added", 5) from None
    except OSError as exc:
        raise AddBuildError(f"the engine could not be run ({exc}), so it was not added", 5) from None
    if done.returncode != 0:
        why = ""
        if done.returncode < 0 and -done.returncode == getattr(signal, "SIGILL", 4):
            why = "it stopped on an instruction this processor does not have (built for another CPU)"
        elif os.name == "nt" and done.returncode & 0xFFFFFFFF == 0xC000001D:
            why = "it stopped on an instruction this processor does not have (built for another CPU)"
        else:
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or [f"exit code {done.returncode}"]
            why = f"`--version` failed: {tail[0][:200]}"
        raise AddBuildError(f"the engine does not run on this computer: {why}. It was not added.", 5)
    text = f"{done.stdout}\n{done.stderr}"
    m = re.search(r"\b(\d+\.\d+\.\d+(?:\.\d+)?)\b", text)
    if not m:
        raise AddBuildError("the engine answered `--version` without a version number, so it was not added", 5)
    return m.group(1)


def _build_record(folder: Path) -> dict:
    try:
        data = json.loads((folder / "BUILD.json").read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def backend_of(build: dict) -> str | None:
    """The engine family a BUILD.json describes: its "backend", else the toolkit it names (cuda / hip / rocm / sycl / metal)."""
    value = str(build.get("backend") or "").strip().lower()
    if value:
        return {"rocm": "hip", "amd": "hip", "nvidia": "cuda", "intel": "sycl"}.get(value, value)
    for key, family in (("cuda", "cuda"), ("hip", "hip"), ("rocm", "hip"), ("sycl", "sycl"), ("oneapi", "sycl"), ("metal", "metal")):
        if build.get(key):
            return family
    return None


def archs_of(build: dict, backend: str) -> list[str]:
    out = []
    for item in build.get("archs") or []:
        text = str(item).strip()
        if backend == "cuda" and text.isdigit():
            out.append(f"sm_{text}")
        elif text:
            out.append(text)
    return out


def _files(folder: Path, exe: Path) -> list[Path]:
    picked = [exe]
    for path in sorted(folder.iterdir()):
        if path == exe or not path.is_file():
            continue
        if _LIB.search(path.name) or path.name in _HELPERS or path.name == "BUILD.json":
            picked.append(path)
    return picked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_build(root: Path, folder: Path, *, backend: str | None = None, detected: str | None = None) -> dict:
    """Copy the build in `folder` into the components folder `root` as a local, unsigned package; returns {"folder", "version", "backend", "archs",
    "files", "bytes"}. Nothing is switched here (the caller chooses it with the same switch as `strata components use`). AddBuildError: refused,
    nothing written."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise AddBuildError(f"{folder} is not a folder", 2)
    exe = _entrypoint(folder)
    build = _build_record(folder)
    family = (backend or backend_of(build) or "").strip().lower()
    if not family:
        raise AddBuildError("the build does not say which kind of graphics card it is for (no BUILD.json): add --backend cuda, hip, sycl, metal or cpu", 2)
    target = host_platform()
    ok, why = backend_supported(family, target)
    if not ok:
        raise AddBuildError(f"a {family} engine cannot run on this computer ({why})", 5)
    if detected and family not in (detected, "cpu"):
        raise AddBuildError(f"this is a {family} engine, and this computer's graphics card needs a {detected} engine", 5)
    version = _version_of(exe, folder)
    archs = archs_of(build, family)
    if not archs and family == "cuda":
        archs = list(ec.cuda_archs_present())            # built here, for this PC's cards (setup --build builds for them)
    elif not archs and family == "hip":
        archs = list(ec.hip_archs_present())
    if not archs and family != "cpu":
        raise AddBuildError("the build does not say which cards it has code for (no archs in BUILD.json), and this PC's cards could not be read", 2)
    name = f"{version}-{target}-{family}{LOCAL_SUFFIX}"
    engines = ec.engine_dir(Path(root))
    final = engines / name
    if final.exists():
        raise AddBuildError(f"a local engine {version} is already added ({final}); remove it first with `strata components remove {name}`", 2)
    engines.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".add-", dir=engines))
    try:
        entries = []
        for src in _files(folder, exe):
            dst = staging / src.name
            shutil.copy2(src, dst)
            entries.append({"path": src.name, "sha256": _sha256(dst), "bytes": dst.stat().st_size})
        manifest = {"schema_version": 2, "id": ec.COMPONENT_ID, "name": ec.COMPONENT_ID, "version": version, "platform": target,
                    "backend": family, "archs": archs, "min_driver": {}, "build_id": "local:" + str(build.get("source") or "unknown"),
                    "min_runtime": None, "entrypoint": exe.name, "files": entries, "signature": None, "local": True,
                    "notes": "built on this computer (strata components add-build); not signed by Strata"}
        (staging / ec.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        component = ec.read_manifest(staging)
        ec.verify(component, full=True)
        os.replace(staging, final)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {"folder": name, "version": version, "backend": family, "archs": manifest["archs"], "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries), "path": str(final)}
