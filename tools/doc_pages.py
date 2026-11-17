"""Which Markdown files are current pages and which are archived history.

One list, used by the release check (tools/release_manifest.py) and the link check (tools/check_links.py), so the two
agree about what "current" means.  Standard library only.
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folders that are never scanned: dependencies, build output, version control, other people's code.
SKIP_DIRS = {".git", "node_modules", "third_party", "dist", "target", "build", "__pycache__", ".venv", "venv"}

# Archived material: dated reports, finished campaigns, release notes, measurement logs.  They keep the versions and
# the links they had when they were written.  (Folders end in "/"; "docs/release-notes-" is a file-name prefix.)
ARCHIVE_PREFIXES = (
    "docs/history/",
    "docs/bench-results/",
    "phaseA-tests/",
    "docs/release-notes-",
)


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def is_archived(relpath):
    return any(relpath.startswith(p) for p in ARCHIVE_PREFIXES)


def _skipped(relpath):
    parts = relpath.split("/")
    return any(p in SKIP_DIRS for p in parts[:-1]) or (len(parts) > 1 and parts[0].startswith("wt-"))


def _git_markdown(root):
    """The .md files git knows about (tracked, plus untracked files that are not ignored), or None outside a git tree.
    Ignored build trees (for example a llama.cpp copy under build-*/_deps) are therefore never scanned."""
    try:
        r = subprocess.run(["git", "-C", root, "ls-files", "-z", "-co", "--exclude-standard", "--", "*.md"],
                           capture_output=True, text=True, encoding="utf-8", timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [p for p in r.stdout.split("\0") if p and os.path.isfile(os.path.join(root, p))]


def markdown_files(root=None):
    """Every .md file under the repository as (relative path, archived?), sorted."""
    root = root or ROOT
    listed = _git_markdown(root)
    if listed is not None:
        return sorted((r, is_archived(r)) for r in listed if not _skipped(r))
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (d.startswith("wt-") and base == root)]
        for name in files:
            if name.endswith(".md"):
                r = os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/")
                out.append((r, is_archived(r)))
    return sorted(out)


def current_pages(root=None):
    return [r for r, archived in markdown_files(root) if not archived]
