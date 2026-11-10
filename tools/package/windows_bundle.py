#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/package/windows_bundle.py - the Windows 7z's folder and archive, from what tools/package/windows.sh built.

    windows_bundle.py --src DIR --engine DIR --sycl DIR --launcher DIR --nuget DIR --python-zip ZIP --wheels DIR
                      --llama DIR --jxl DIR --version V --cuda-archs A;B --out DIR

The folder (XeStrata-<version>-windows-x64/):

    xestrata.exe, xestrata-cli.exe and their .NET libraries   the launchers (launcher)
    app/                  setup.py, serve/, tools/, data/ and gguf-py, read-only, with the PACKAGED record
    app/engine/           the engine, its SYCL runtime (intel/llvm's), libjxl's cjxl.exe and djxl.exe, and BUILD.json
    python/               Python's embeddable distribution with the packages setup and the server use
    NOTICE, COPYING, COPYING.LESSER, THIRD-PARTY-NOTICES.md, licenses/

Every program and library in it (.exe, .dll, .pyd) must belong to a component of COMPONENTS or to a Python package
of WHEELS, and none may be a non-free runtime (the Visual C++ runtime the user installs, NVIDIA's libraries the driver
brings): the script stops otherwise, so nothing goes out without its notice.  THIRD-PARTY-NOTICES.md lists every
component, its version, license and source, and licenses/ holds their texts."""
import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import engine_notices     # beside this script
import launcher_notices

# The SYCL runtime's DLLs a Windows engine loads (sycl-jit.dll, the runtime compiler, is not: 120 MB the engine does
# not use).  ze_loader.dll comes with Intel's driver, nvcuda.dll and nvml.dll with NVIDIA's, amdhip64_7.dll with AMD's.
SYCL_DLLS = ("sycl9.dll", "ur_win_proxy_loader.dll", "ur_adapter_cuda.dll", "ur_adapter_hip.dll",
             "ur_adapter_level_zero.dll", "ur_adapter_level_zero_v2.dll", "xptifw.dll")
ENGINE_EXES = ("xestrata-engine.exe", "xestrata-device.exe", "xestrata-image.exe", "xestrata-vision-cpu.exe")
# oneMath and its oneMKL backend (the engine's build folder's bin/), which loads oneMKL where the user installed it
ONEMATH_DLLS = ("onemath.dll", "onemath_blas_mklgpu.dll", "onemath_blas_cublas.dll", "onemath_blas_rocblas.dll")
# AMD GPUs' image attention: the engine's module that opens aotriton (on an AMD GPU only), aotriton's library (built
# in the engine's build, liblzma linked in) and its kernels' images, which it finds beside it
AOTRITON_DLLS = ("xestrata_sdpa_aotriton.dll", "aotriton_v2.dll")
# Intel GPUs' image attention: oneDNN (built in the engine's build, onednn/bin/)
ONEDNN_DLLS = ("dnnl.dll",)
# what a Windows PC brings (the Visual C++ runtime, which xestrata.exe checks for) and what must never be carried
FORBIDDEN = ("vcruntime*.dll", "msvcp*.dll", "concrt*.dll", "ucrtbase*.dll", "api-ms-win-*.dll", "nvcuda.dll",
             "nvml.dll", "cublas*.dll", "cudart*.dll", "mkl_*.dll", "amdhip64*.dll", "amd_comgr*.dll", "rocblas*.dll",
             "*hipblaslt*.dll")
# the Python packages the 7z carries (requirements.txt, and colorama, which tqdm takes on
# Windows): a wheel not named here stops the script
WHEELS = ("numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil", "pypdf", "pypdfium2",
          "jsonschema", "markupsafe", "certifi", "charset-normalizer", "charset_normalizer", "idna", "urllib3",
          "typing_extensions", "attrs", "jsonschema_specifications", "referencing", "rpds_py", "colorama", "pip")

# The Windows engine's parts (engine_notices.py's PARTS.json names): what the 7z's engine is built with
WIN_PARTS = {"onemath", "onemath_rocblas", "onemath_cublas", "onemkl", "cuda", "hip", "onednn", "aotriton",
             "cudnn_frontend", "cutlass", "xgrammar"}
# the parts of engine_notices.ROWS the 7z carries, as COMPONENTS' rows; those it does not carry (the contrib engine's
# oneMKL and NVIDIA's libraries, where the user installed them) go to NOTICES.json only
ENGINE_ROWS = engine_notices.rows_for(WIN_PARTS, linux=False)

# name, version (a key of `versions`, or the text), license, source, license texts (paths relative to `roots`' keys, or
# URLs), the files in the folder it covers (globs relative to its top)
COMPONENTS = [
    ("XeStrata", "version", "LGPL-3.0-or-later (with MIT parts: NOTICE)", "https://github.com/MistVVK/XeStrata",
     [], [*launcher_notices.own_files("win-x64"), *(f"app/engine/{e}" for e in ENGINE_EXES),
          f"app/engine/{AOTRITON_DLLS[0]}"]),
    *((r["name"], r["version"], r["license"], r["source"], r["texts"],
       r.get("win", []) + ([f"app/engine/{d}" for d in SYCL_DLLS] if r["version"] == "intel-llvm" else []))
      for r in ENGINE_ROWS if r.get("bundled", True)),
    *launcher_notices.components("win-x64"),       # the launchers' parts (shared with the xestrata-gui package)
    ("Python (embeddable distribution)", "python", "PSF-2.0 and the licenses its LICENSE.txt names",
     "https://www.python.org/", ["python:LICENSE.txt"], ["python/*.exe", "python/*.dll", "python/*.pyd"]),
    ("libjxl's cjxl and djxl (statically linked with Highway, Brotli, skcms, libpng, zlib, libjpeg-turbo, giflib, "
     "libwebp and sjpeg)", "jxl", "BSD-3-Clause; the libraries under their own licenses (Apache-2.0, BSD-3-Clause, "
     "MIT, zlib, IJG, libpng)", "https://github.com/libjxl/libjxl", ["jxl:licenses/LICENSE.*"],
     ["app/engine/cjxl.exe", "app/engine/djxl.exe"]),
]

# the program's files from the source tree (as cmake/packaging.cmake installs them on Linux)
APP_FILES = ("setup.py", "chat.py", "requirements.txt")
APP_DIRS = ("serve", "tools", "ref", "data", "third_party/main/outfit", "third_party/main/katex",
            "third_party/main/mermaid", "third_party/main/quantscope", "third_party/main/py3langid")
APP_SKIP = ("__pycache__", "test_*", "mcp_fake_server.py", "lint", "vision", "package")


def say(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    say(f"windows_bundle: {msg}")
    sys.exit(1)


def copy_tree(src: Path, dst: Path, skip=()) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*skip) if skip else None)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


nuget_version = launcher_notices.nuget_version


def unbundle_runtime(site: Path) -> None:
    """A wheel's own copy of the Visual C++ runtime (numpy's numpy.libs/msvcp140-<hash>.dll, renamed by delvewheel so
    it loads beside another copy) removed, and its binaries' imports of it renamed back to the runtime's own name, so
    they load the one Windows has (the Visual C++ Redistributable xestrata.exe checks for): the name is rewritten in
    place, the rest of the field filled with NULs."""
    for lib in sorted(site.glob("*.libs/*.dll")):
        m = re.fullmatch(r"(?i)(msvcp140(?:_\w+)?|vcruntime140(?:_1)?|concrt140)-[0-9a-f]+\.dll", lib.name)
        if not m:
            continue
        old, new = lib.name.encode(), (m.group(1) + ".dll").encode()
        for f in site.rglob("*"):
            if f.suffix.lower() in (".pyd", ".dll") and f != lib:
                data = f.read_bytes()
                if old + b"\0" in data:
                    f.write_bytes(data.replace(old + b"\0", new + b"\0" * (len(old) - len(new) + 1)))
                    say(f"{f.relative_to(site)}: imports {new.decode()} instead of {lib.name}")
        lib.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("src", "engine", "sycl", "launcher", "nuget", "wheels", "llama", "rocm", "jxl", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--python-zip", type=Path, required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--cuda-archs", required=True, help="the engine's NVIDIA architectures, sm_60;sm_61;...")
    ap.add_argument("--hip-archs", required=True, help="the engine's AMD architectures, gfx1030;gfx1100;...")
    ap.add_argument("--mkl-version", required=True, help="the oneMKL the engine was built with, 2026.2.0")
    a = ap.parse_args()

    top_name = f"XeStrata-{a.version}-windows-x64"
    top = a.out / top_name
    shutil.rmtree(top, ignore_errors=True)
    top.mkdir(parents=True)

    # ---- the launchers
    for f in a.launcher.iterdir():
        if f.suffix == ".pdb":
            continue
        (copy_tree(f, top / f.name) if f.is_dir() else shutil.copy2(f, top / f.name))

    # ---- the program
    app = top / "app"
    app.mkdir()
    for name in APP_FILES:
        shutil.copy2(a.src / name, app / name)
    for d in APP_DIRS:
        copy_tree(a.src / d, app / d, APP_SKIP)
    copy_tree(a.src / "serve" / "web", app / "serve" / "web")
    gguf = app / "third_party" / "main" / "llama.cpp"
    copy_tree(a.llama / "gguf-py", gguf / "gguf-py", ("__pycache__", "tests"))
    shutil.copy2(a.llama / "LICENSE", gguf / "LICENSE")
    (app / "PACKAGED").write_text(json.dumps({"engine": "engine", "version": a.version}) + "\n")

    # ---- the engine
    eng = app / "engine"
    eng.mkdir()
    for e in ENGINE_EXES:
        shutil.copy2(a.engine / e, eng / e)
    for d in SYCL_DLLS:
        shutil.copy2(a.sycl / "bin" / d, eng / d)
    for d in ONEMATH_DLLS:
        shutil.copy2(a.engine / "bin" / d, eng / d)
    for d in ONEDNN_DLLS:
        shutil.copy2(a.engine / "onednn" / "bin" / d, eng / d)
    for x in ("cjxl.exe", "djxl.exe"):   # the images' JPEG XL out and in (serve/images.py cjxl, djxl)
        shutil.copy2(a.jxl / "bin" / x, eng / x)
    aot = a.engine / "_deps" / "aotriton" / "aotriton"
    shutil.copy2(a.engine / AOTRITON_DLLS[0], eng / AOTRITON_DLLS[0])
    shutil.copy2(aot / "lib" / AOTRITON_DLLS[1], eng / AOTRITON_DLLS[1])
    shutil.copytree(aot / "lib" / "aotriton.images", eng / "aotriton.images")
    archs = [x for x in a.cuda_archs.split(";") if x]
    (eng / "BUILD.json").write_text(json.dumps({
        "source": "package", "backend": "xe", "package": top_name, "version": a.version, "license": "contrib",
        "cuda_archs": archs, "hip_archs": [x for x in a.hip_archs.split(";") if x],
        # oneMKL's pip packages for the Python the 7z carries (setup names them), the version the engine was built with,
        # without the SYCL runtime they depend on (the 7z's)
        "runtime": {"onemkl_pip": [f"onemkl-sycl-blas=={a.mkl_version}", f"mkl=={a.mkl_version}", "tbb==2023.*",
                                   f"onemkl-license=={a.mkl_version}"]}, "lib_dirs": []},
        indent=1) + "\n")

    # ---- Python: the embeddable distribution without its Visual C++ runtime (the user's, which xestrata.exe checks)
    # and its ._pth (which would leave the script's folder off sys.path: setup.py imports serve/ from beside it), and
    # the wheels unpacked into its site-packages
    py = top / "python"
    with zipfile.ZipFile(a.python_zip) as z:
        z.extractall(py)
    for f in [*py.glob("vcruntime*.dll"), *py.glob("python*._pth")]:
        f.unlink()
    site = py / "Lib" / "site-packages"
    site.mkdir(parents=True)
    wheels = sorted(a.wheels.glob("*.whl"))
    for w in wheels:
        name = w.name.split("-")[0].lower()
        if name not in WHEELS:
            fail(f"{w.name} is not one of the packages the 7z carries (WHEELS)")
        with zipfile.ZipFile(w) as z:
            z.extractall(site)
    unbundle_runtime(site)

    # ---- notices
    for name in ("NOTICE", "COPYING", "COPYING.LESSER"):
        shutil.copy2(a.src / name, top / name)
    lic = top / "licenses"
    lic.mkdir()
    versions = {"version": a.version, "dotnet": "10", "llama.cpp": "the commit setup.py pins",
                "outfit": "third_party/main/outfit", "quantscope": "bfdf205", "py3langid": "0.4.0",
                "python": a.python_zip.stem.split("-")[1],
                "jxl": a.jxl.parent.name.removeprefix("jxl-"),   # windows.sh's jxl-<version>/x64-windows-static
                "rocm": "ROCm 7.1 (Ubuntu 26.04's)"}             # tools/package/win_rocm.sh's
    sycl_rec = json.loads((a.sycl / "XESTRATA.json").read_text())
    versions["intel-llvm"] = f"{sycl_rec['tag']} ({sycl_rec['commit'][:12]}, with XeStrata's patches)"
    # the fork's tag the engine's build fetched (CMakeLists.txt pins it)
    m = re.search(r"tags/([\w.-]+)\.tar\.gz", (a.engine / "CMakeCache.txt").read_text())
    versions["onemath"] = m.group(1) if m else "the archive CMakeLists.txt pins"
    versions.update(launcher_notices.nuget_versions(a.nuget, "win-x64", a.launcher))
    roots = {"sycl": a.sycl, "python": py, "llama": a.llama, "src": a.src, "nuget": a.nuget, "rocm": a.rocm,
             "jxl": a.jxl, "onemath": a.engine / "_deps" / "strata_onemath-src"}
    rows = []
    covered: set = set()
    files = [p.relative_to(top).as_posix() for p in top.rglob("*") if p.is_file()]
    for name, ver_key, license_, source, texts, globs in COMPONENTS:
        version = versions.get(ver_key, ver_key)
        slug = "".join(c if c.isalnum() else "-" for c in name.split(" (")[0].lower()).strip("-")
        kept = []
        for t in texts:
            if t.startswith("http"):
                url = t.format(avalonia=versions["avalonia"])
                data = fetch(url)
                dst = lic / f"{slug}-{Path(url).name}"
                dst.write_bytes(data)
                kept.append(dst)
                continue
            root, rel = t.split(":", 1)
            base = roots[root]
            if root == "nuget":
                pkg, rest = rel.split("/", 1)
                base, rel = base / pkg / nuget_version(a.nuget, pkg, launcher_notices.published(a.launcher)), rest
            found = sorted(base.glob(rel))
            if not found:
                fail(f"{name}: no license text {t}")
            for text in found:
                dst = lic / f"{slug}-{text.name}"
                shutil.copy2(text, dst)
                kept.append(dst)
        for g in globs:
            covered.update(f for f in files if fnmatch.fnmatch(f, g))
        rows.append((name, version, license_, source, [k.relative_to(top).as_posix() for k in kept]))
    for dist in sorted(site.glob("*.dist-info")):
        meta = (dist / "METADATA").read_text(errors="replace")
        field = lambda key: next((ln.split(":", 1)[1].strip() for ln in meta.splitlines()  # noqa: E731
                                  if ln.startswith(key + ":")), "")
        # by name, or anything in the licenses/ folder (PEP 639: pypdfium2 keeps PDFium's and its parts' texts there
        # under their own names)
        wheel_texts = [f for f in dist.rglob("*") if f.is_file() and (f.name.upper().startswith(
            ("LICEN", "COPYING", "NOTICE", "AUTHORS")) or f.relative_to(dist).parts[0] == "licenses")]
        if not wheel_texts:
            fail(f"{dist.name}: no license text in the wheel")
        kept = []
        for text in wheel_texts:
            # a text below licenses/'s own folders (pip's of the packages it vendors, all named LICENSE) by its path
            parts = text.relative_to(dist).parts
            dst = lic / f"python-{field('Name').lower()}-{'-'.join(parts[1:]) if len(parts) > 2 else text.name}"
            shutil.copy2(text, dst)
            kept.append(dst.relative_to(top).as_posix())
        record = (dist / "RECORD").read_text().splitlines()
        covered.update(f"python/Lib/site-packages/{ln.split(',')[0]}" for ln in record)
        rows.append((f"{field('Name')} (Python package)", field("Version"),
                     field("License-Expression") or field("License") or "see its license text",
                     field("Home-page") or f"https://pypi.org/project/{field('Name')}/", kept))

    # ---- what may not go out: a program or library of no component, and a non-free runtime
    binaries = [f for f in files if f.lower().endswith((".exe", ".dll", ".pyd"))]
    bad = [f for f in binaries if any(fnmatch.fnmatch(Path(f).name.lower(), g) for g in FORBIDDEN)]
    if bad:
        fail("non-free runtime in the folder: " + ", ".join(bad))
    unknown = [f for f in binaries if f not in covered]
    if unknown:
        fail("programs or libraries of no component (COMPONENTS in this script): " + ", ".join(unknown))

    lines = ["# Third-party notices", "",
             f"XeStrata {a.version} for Windows carries the components below. XeStrata's own code is under the LGPL "
             "and its parts from other projects keep their licenses "
             "(`NOTICE`, `COPYING`, `COPYING.LESSER`); its source is at https://github.com/MistVVK/XeStrata. "
             "The license texts are in `licenses/`.", "",
             "The Microsoft Visual C++ runtime, the GPU makers' drivers and NVIDIA's libraries are not part of it: "
             "they come with Windows' Visual C++ Redistributable and the drivers.", "",
             "| Component | Version | License | Source | Texts |", "| --- | --- | --- | --- | --- |"]
    for name, version, license_, source, kept in rows:
        lines.append(f"| {name} | {version} | {license_} | {source} | {', '.join(f'`{k}`' for k in kept)} |")
    (top / "THIRD-PARTY-NOTICES.md").write_text("\n".join(lines) + "\n")
    # the same for the server's About > Licenses (engine_notices.py's form, the texts read in), with the parts the
    # engine opens where the user installed them; not the Python packages, which the server reads where they are

    def read(k: str) -> dict:
        return {"name": Path(k).name, "text": (top / k).read_text(encoding="utf-8", errors="replace")}

    notices = [engine_notices.entry({"name": name, "license": license_, "source": source}, version,
                                    [read(k) for k in kept])
               for name, version, license_, source, kept in rows[1:len(COMPONENTS)]]   # not XeStrata's own
    notices += [engine_notices.entry(r, r["version"], []) for r in ENGINE_ROWS if not r.get("bundled", True)]
    (eng / "NOTICES.json").write_text(json.dumps(notices, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    archive = a.out / f"{top_name}.7z"
    archive.unlink(missing_ok=True)
    subprocess.run(["7z", "a", "-t7z", "-mx=9", "-mmt=on", "-bd", str(archive), top_name], cwd=a.out, check=True,
                   stdout=subprocess.DEVNULL)
    say(f"{archive} ({archive.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
