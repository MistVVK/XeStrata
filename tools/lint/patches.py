#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check the final changed lines of third-party patches against their upstream style.

Optional --source PROJECT=PATH names a local upstream git clone or release archive.
No source is modified.
"""
import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {
    "intel-llvm": ("ea269d5f7189ab7ade0ada0706d973c5f5ee5f65", ROOT / ".tools/intel-llvm/src"),  # v7.2.1, untagged
    "oneDNN": ("v3.14.2", ROOT / ".tools/src/oneDNN"),
    "comfy-kitchen": ("v0.2.41", ROOT / ".tools/src/comfy-kitchen"),
    "aotriton": ("0.15b", ROOT / ".tools/src/aotriton"),
}


def git(source, *args, check=True):
    return subprocess.run(["git", "-C", str(source), *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check)


def update_ownership(old, new, owned, ours):
    """Carry ownership across a patch, marking only its inserted or replaced lines."""
    result = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i, j, first, last in matcher.get_opcodes():
        if tag == "equal":
            result.extend(owned[i:j])
        elif tag in ("replace", "insert"):
            result.extend([ours] * (last - first))
    return result


def owned_spans(owned):
    spans = []
    for i, marked in enumerate(owned, 1):
        if marked:
            if spans and spans[-1][1] == i - 1:
                spans[-1] = (spans[-1][0], i)
            else:
                spans.append((i, i))
    return spans


def check_project(name, source, ref, clang_format):
    patches = sorted((ROOT / "third_party/main" / name / "patches").glob("*.patch"))
    if not patches:
        return True
    archive = None
    if source.is_file() and ref in source.name and tarfile.is_tarfile(source):
        archive = tarfile.open(source, "r:gz")
        members = {"/".join(m.name.split("/")[1:]): m for m in archive.getmembers() if m.isfile()}
    elif not (source / ".git").exists() or git(source, "rev-parse", "--verify", ref, check=False).returncode:
        print(f"== patch lint {name}: skipped (upstream source at {ref} not available at {source})", flush=True)
        return True
    paths = set()
    for patch in patches:
        result = subprocess.run(["git", "apply", "--numstat", str(patch)], capture_output=True, text=True, check=True)
        paths.update(line.split("\t", 2)[2] for line in result.stdout.splitlines())
    print(f"== patch lint {name}: {len(patches)} patches, {len(paths)} files", flush=True)
    ok = True
    with tempfile.TemporaryDirectory(prefix="xestrata-patch-lint-") as temp:
        tree = Path(temp)
        ownership = {}
        configs = {Path(".clang-format"), Path(".clang-format-ignore"), Path(".cmake-format.py")}
        for path in paths:
            for parent in Path(path).parents:
                configs.update((parent / ".clang-format", parent / ".clang-format-ignore", parent / ".cmake-format.py"))
        for path in sorted(paths | {str(c) for c in configs}):
            if archive:
                member = members.get(path)
                data = archive.extractfile(member).read() if member else None
            else:
                result = git(source, "show", f"{ref}:{path}", check=False)
                data = result.stdout if result.returncode == 0 else None
            if data is None:
                continue  # A patch can add a new file; not every directory has its own style file.
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if path in paths:
                ownership[path] = [False] * len(data.splitlines())
        for patch in patches:
            header = patch.read_text().split("\ndiff --git ", 1)[0]
            upstream = bool(re.search(r"(?m)^upstream commit: [0-9a-f]{40}$", header))
            if upstream:
                print(f"== patch lint {name}/{patch.name}: upstream backport, skipped", flush=True)
            before = {path: (tree / path).read_text().splitlines(keepends=True)
                      if (tree / path).is_file() else [] for path in paths}
            subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
            for path in paths:
                target = tree / path
                after = target.read_text().splitlines(keepends=True) if target.is_file() else []
                ownership[path] = update_ownership(before[path], after, ownership.get(path, []), not upstream)
        for path in sorted(paths):
            target = tree / path
            if not target.is_file():
                continue
            after = target.read_text()
            spans = owned_spans(ownership[path])
            if not spans:
                continue
            if target.suffix in (".cpp", ".cc", ".c", ".hpp", ".h", ".cl"):
                style = next((parent / ".clang-format" for parent in target.parents
                              if (parent == tree or tree in parent.parents) and (parent / ".clang-format").is_file()),
                             None)
                if not clang_format:
                    print(f"== clang-format {name}/{path}: skipped (clang-format not installed)")
                elif not style:
                    print(f"== clang-format {name}/{path}: skipped (upstream has no style file)")
                else:
                    args = [clang_format, "--dry-run", "--Werror"]
                    args.extend(f"--lines={first}:{last}" for first, last in spans)
                    result = subprocess.run([*args, str(target)], capture_output=True, text=True)
                    if result.returncode:
                        print(result.stderr.replace(str(tree) + "/", name + "/"), end="")
                        ok = False
            if target.name == "CMakeLists.txt" or target.suffix == ".cmake":
                cmake_lint = shutil.which("cmake-lint")
                configs = [parent / ".cmake-format.py" for parent in target.parents
                           if parent == tree or tree in parent.parents]
                config = next((c for c in configs if c.is_file()), None)
                if not cmake_lint:
                    print(f"== cmake-lint {name}/{path}: skipped (cmake-lint not installed)")
                elif not config:
                    print(f"== cmake-lint {name}/{path}: skipped (upstream has no lint configuration)")
                else:
                    args = [cmake_lint, "--suppress-decorations", "-c", str(config)]
                    result = subprocess.run([*args, "--", str(target)], capture_output=True, text=True)
                    diagnostics = []
                    for line in result.stdout.splitlines():
                        match = re.match(re.escape(str(target)) + r":(\d+),\d+: ", line)
                        if match and any(first <= int(match.group(1)) <= last for first, last in spans):
                            diagnostics.append(line.replace(str(tree) + "/", name + "/"))
                    if diagnostics:
                        print("\n".join(diagnostics))
                        ok = False
                    if result.returncode and not re.search(re.escape(str(target)) + r":\d+,\d+: ", result.stdout):
                        print(result.stdout + result.stderr, end="")
                        ok = False
            lines = after.splitlines()
            additions = "\n".join(line for first, last in spans for line in lines[first - 1:last]) + "\n"
            if shutil.which("codespell"):
                result = subprocess.run(["codespell", "--config", str(ROOT / "tools/lint/codespell.cfg"), "-"],
                                        input=additions, capture_output=True, text=True)
                if result.returncode:
                    print(f"== codespell {name}/{path}\n{result.stdout}{result.stderr}", end="")
                    ok = False
            else:
                print(f"== codespell {name}/{path}: skipped (codespell not installed)")
    if archive:
        archive.close()
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="PROJECT=PATH")
    parser.add_argument("patches", nargs="*")
    args = parser.parse_args()
    sources = {name: path for name, (_, path) in PROJECTS.items()}
    for item in args.source:
        name, sep, path = item.partition("=")
        if not sep or name not in PROJECTS:
            parser.error(f"unknown source: {item}")
        sources[name] = Path(path).resolve()
    selected = {Path(p).parts[2] for p in args.patches}
    for name in sorted(selected - PROJECTS.keys()):
        print(f"== patch lint {name}: skipped (no upstream source or pinned version configured)")
    formatter = shutil.which("clang-format")
    if not formatter:
        candidates = sorted(Path("/opt/intel/oneapi/compiler").glob("*/bin/compiler/clang-format"))
        formatter = str(candidates[-1]) if candidates else None
    ok = True
    for name, (ref, _) in PROJECTS.items():
        if not selected or name in selected:
            ok = check_project(name, sources[name], ref, formatter) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
