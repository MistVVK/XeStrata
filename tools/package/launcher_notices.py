#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/package/launcher_notices.py - the launcher's window and the NuGet packages it is built of, with their licenses:
the Windows 7z's (windows_bundle.py takes COMPONENTS from here) and the xestrata-gui package's notices alike.

    launcher_notices.py --launcher DIR --nuget DIR --version V --out DIR

writes the xestrata-gui package's THIRD-PARTY-NOTICES.md, notices.json and licenses/ to DIR (its
/usr/share/doc/xestrata-gui/), from
what `dotnet publish -r linux-x64` put in the launcher's folder DIR and the NuGet cache it restored to.  Every program
and library there must belong to a component, or it stops: nothing goes out without its notice.  The .NET runtime is
not one of them: the distribution's dotnet-runtime-10.0 package brings it.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def windows(target: str) -> bool:
    return target.startswith("win")


def own_files(target: str) -> list:
    """The launcher's own files (XeStrata's, under its license)."""
    if windows(target):
        return ["xestrata.exe", "xestrata-cli.exe", "xestrata.dll", "xestrata-cli.dll", "Xestrata.Launcher.Core.dll"]
    return ["xestrata-gui", "xestrata-gui.dll", "Xestrata.Launcher.Core.dll"]


def components(target: str) -> list:
    """The launcher's parts from other projects for `target` (win-x64, linux-x64), as windows_bundle.py's COMPONENTS:
    name, version (a key of nuget_versions' answer, or the text), license, source, license texts (nuget:PACKAGE/PATH or
    URLs), the files in the launcher's folder it covers (globs)."""
    native, lib = ("win32", "*.dll") if windows(target) else ("linux", "*.so")
    skia = lib.replace("*", "libSkiaSharp")
    harfbuzz = lib.replace("*", "libHarfBuzzSharp")
    rows = [
        (".NET application host", "dotnet", "MIT", "https://github.com/dotnet/runtime",
         ["https://raw.githubusercontent.com/dotnet/runtime/main/LICENSE.TXT",
          "https://raw.githubusercontent.com/dotnet/runtime/main/THIRD-PARTY-NOTICES.TXT"], []),
        ("Avalonia", "avalonia", "MIT", "https://github.com/AvaloniaUI/Avalonia",
         ["https://raw.githubusercontent.com/AvaloniaUI/Avalonia/{avalonia}/licence.md"], ["Avalonia*.dll"]),
    ]
    if windows(target):
        rows.append(("ANGLE (Avalonia's build)", "avalonia.angle.windows.natives", "BSD-3-Clause",
                     "https://github.com/AvaloniaUI/angle", ["nuget:avalonia.angle.windows.natives/LICENSE"],
                     ["av_libglesv2.dll"]))
    rows += [
        ("SkiaSharp and Skia", "skiasharp",
         "MIT; Skia BSD-3-Clause and the licenses of the parts it carries (FreeType FTL, libpng, zlib, expat, "
         "libjpeg-turbo, libwebp, Adobe's DNG SDK, piex: its notices; of the DNG SDK its code only, none of its "
         "documentation, whose terms are not free, is included)",
         "https://github.com/mono/SkiaSharp", ["nuget:skiasharp/LICENSE.txt",
                                               f"nuget:skiasharp.nativeassets.{native}/THIRD-PARTY-NOTICES.txt"],
         ["SkiaSharp.dll", skia]),
        ("HarfBuzzSharp and HarfBuzz", "harfbuzzsharp", "MIT; HarfBuzz MIT (its notices)",
         "https://github.com/mono/SkiaSharp", ["nuget:harfbuzzsharp/LICENSE.txt",
                                               f"nuget:harfbuzzsharp.nativeassets.{native}/THIRD-PARTY-NOTICES.txt"],
         ["HarfBuzzSharp.dll", harfbuzz]),
        ("FSharp.Core", "fsharp.core", "MIT", "https://github.com/dotnet/fsharp",
         ["https://raw.githubusercontent.com/dotnet/fsharp/main/License.txt"],
         ["FSharp.Core.dll", "*/FSharp.Core.resources.dll"]),
        ("MicroCom.Runtime", "microcom.runtime", "MIT", "https://github.com/kekekeks/MicroCom",
         ["https://raw.githubusercontent.com/kekekeks/MicroCom/master/LICENSE"], ["MicroCom.Runtime.dll"]),
        ("Tmds.DBus.Protocol" + (" (Avalonia's Linux part, not used on Windows)" if windows(target) else ""),
         "tmds.dbus.protocol", "MIT", "https://github.com/tmds/Tmds.DBus",
         ["https://raw.githubusercontent.com/tmds/Tmds.DBus/main/COPYING"], ["Tmds.DBus.Protocol.dll"]),
    ]
    return rows


def published(launcher: Path) -> dict:
    """The NuGet packages' versions the launcher was published with (its *.deps.json, from the lock files)."""
    found = {}
    for deps in launcher.glob("*.deps.json"):
        for key, lib in json.loads(deps.read_text()).get("libraries", {}).items():
            if lib.get("type") == "package":
                name, version = key.split("/", 1)
                found[name.lower()] = version
    return found


def nuget_version(nuget: Path, package: str, pinned: dict | None = None) -> str:
    """The version of `package`: the one the launcher was published with (`pinned`, published()'s), else the newest in
    the NuGet cache."""
    if pinned and package in pinned:
        return pinned[package]
    folder = nuget / package
    versions = sorted((p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else [],
                      key=lambda v: [int(x) if x.isdigit() else x for x in v.replace("-", ".").split(".")])
    if not versions:
        sys.exit(f"launcher_notices: no NuGet package {package} in {nuget}")
    return versions[-1]


def nuget_versions(nuget: Path, target: str, launcher: Path) -> dict:
    """The versions of the NuGet packages `components` names, as the launcher in `launcher` was published with."""
    pinned = published(launcher)
    names = {c[1] for c in components(target)} - {"dotnet"}
    return {"dotnet": "10", **{n: nuget_version(nuget, n, pinned) for n in names}}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("launcher", "nuget", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    target = "linux-x64"
    lic = a.out / "licenses"
    shutil.rmtree(lic, ignore_errors=True)
    lic.mkdir(parents=True)
    versions = nuget_versions(a.nuget, target, a.launcher)
    files = [p.relative_to(a.launcher).as_posix() for p in a.launcher.rglob("*") if p.is_file()]
    covered = {f for f in files if f in own_files(target)}
    rows: list = [("XeStrata's launcher", a.version, "LGPL-3.0-or-later", "https://github.com/MistVVK/XeStrata", [])]
    for name, ver_key, license_, source, texts, globs in components(target):
        slug = "".join(c if c.isalnum() else "-" for c in name.split(" (")[0].lower()).strip("-")
        kept = []
        for t in texts:
            if t.startswith("http"):
                url = t.format(avalonia=versions["avalonia"])
                dst = lic / f"{slug}-{Path(url).name}"
                dst.write_bytes(fetch(url))
            else:
                pkg, rel = t.removeprefix("nuget:").split("/", 1)
                src = a.nuget / pkg / nuget_version(a.nuget, pkg, published(a.launcher)) / rel
                if not src.is_file():
                    sys.exit(f"launcher_notices: {name}: no license text {t}")
                dst = lic / f"{slug}-{src.name}"
                shutil.copy2(src, dst)
            kept.append(f"licenses/{dst.name}")
        covered.update(f for f in files for g in globs if fnmatch.fnmatch(f, g))
        rows.append((name, versions.get(ver_key, ver_key), license_, source, kept))
    binaries = [f for f in files if f.endswith((".dll", ".so")) or f in own_files(target)]
    unknown = [f for f in binaries if f not in covered]
    if unknown:
        sys.exit("launcher_notices: libraries of no component (components() in this script): " + ", ".join(unknown))
    lines = ["# Third-party notices", "",
             f"XeStrata's launcher window {a.version} (the xestrata-gui package) carries the components below. "
             "XeStrata's own code is under the LGPL (/usr/share/doc/xestrata-*/: NOTICE, COPYING, COPYING.LESSER); "
             "its source is at https://github.com/MistVVK/XeStrata. The license texts are in `licenses/`.", "",
             "The .NET runtime is not part of it: the distribution's dotnet-runtime-10.0 package brings it.", "",
             "| Component | Version | License | Source | Texts |", "| --- | --- | --- | --- | --- |"]
    for name, version, license_, source, kept in rows:
        lines.append(f"| {name} | {version} | {license_} | {source} | {', '.join(f'`{k}`' for k in kept)} |")
    (a.out / "THIRD-PARTY-NOTICES.md").write_text("\n".join(lines) + "\n")
    # the same for the server's About > Licenses (engine_notices.py's form, the texts read in), without the launcher's
    # own row
    (a.out / "notices.json").write_text(json.dumps(
        [{"name": name, "version": version, "license": license_, "source": source, "free": True, "bundled": True,
          "texts": [{"name": Path(k).name, "text": (a.out / k).read_text(encoding="utf-8", errors="replace")}
                    for k in kept]} for name, version, license_, source, kept in rows[1:]],
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
