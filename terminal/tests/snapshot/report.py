"""Read golden files and print, as Markdown, where the entry points differ and (with --against) what changed between two sets of golden files.

  python -m tests.snapshot.report                          (the golden files of this tree)
  python -m tests.snapshot.report --golden A --against B   (A = this tree's, B = another tree's, e.g. the branch f-0213-df2ux)
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .matrix import ENTRY_POINTS, MACHINES, MODEL_SETS

GOLDEN = Path(__file__).resolve().parent / "golden"

#: config keys and variables that differ between the Terminal and the app by design: where things live and who they belong to (not what the engine runs).
IDENTITY_KEYS = {"port", "api_key", "aliases", "log", "mcp", "engine_ready_s", "fit_max_tokens"}


def load(root: Path):
    out = {}
    for m in MACHINES:
        for s in MODEL_SETS:
            p = root / m.key / f"{s.key}.json"
            if p.is_file():
                out[(m.key, s.key)] = json.loads(p.read_text(encoding="utf-8"))["entries"]
    return out


def flags(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.append((argv[i], argv[i + 1]))
            i += 2
        else:
            out.append((argv[i], None))
            i += 1
    return out


def drafter(rec) -> str:
    eng = rec.get("engine")
    if not eng:
        return "not driven" if rec.get("not_driven") else "?"
    a = eng["argv"]
    if "--dspark" in a:
        return "DSpark " + a[a.index("--dspark") + 1]
    if "--draft" in a:
        pack = a[a.index("--draft-model") + 1]
        return "DFlash2 %s conf %s spec %s" % ("tuned" if "strata1" in pack else "stock", a[a.index("--draft-conf") + 1], a[a.index("--spec") + 1])
    return "MTP" if "--mtp" in a else "none"


def table(data) -> list[str]:
    lines = ["| machine class | set | Terminal launch | app plain Load | app Models-screen Load | same? |", "|---|---|---|---|---|---|"]
    for m in MACHINES:
        for s in MODEL_SETS:
            e = data.get((m.key, s.key))
            if not e:
                continue
            la, pl, mo = drafter(e["launch"]), drafter(e["app_plain"]), drafter(e["app_models"])
            lines.append(f"| {m.key} | {s.key} | {la} | {pl} | {mo} | {'yes' if la == pl == mo else '**NO**'} |")
    return lines


def differences(data, a="launch", b="app_plain"):
    kinds = collections.defaultdict(list)
    for (mk, sk), e in data.items():
        A, B = e[a], e[b]
        fa, fb = set(flags(A["engine"]["argv"])), set(flags(B["engine"]["argv"]))
        for x in sorted(fa - fb, key=str):
            kinds[("argument", f"only in {a}: {' '.join(v for v in x if v)}")].append(f"{mk}/{sk}")
        for x in sorted(fb - fa, key=str):
            kinds[("argument", f"only in {b}: {' '.join(v for v in x if v)}")].append(f"{mk}/{sk}")
        ea, eb = A["engine"]["env_set"], B["engine"]["env_set"]
        for k in sorted(set(ea) | set(eb)):
            if ea.get(k) != eb.get(k):
                kinds[("environment", f"{k}: {a} {ea.get(k, '<unset>')} | {b} {eb.get(k, '<unset>')}")].append(f"{mk}/{sk}")
        ca, cb = A["config_file"], B["config_file"]
        for k in sorted(set(ca) | set(cb)):
            if k in ("args", "args_derived", "args_written"):
                continue
            if ca.get(k, "<absent>") != cb.get(k, "<absent>"):
                kind = "identity/location" if k in IDENTITY_KEYS else "config key"
                kinds[(kind, f"{k}: {a} {short(ca.get(k, '<absent>'))} | {b} {short(cb.get(k, '<absent>'))}")].append(f"{mk}/{sk}")
        if [f["file"] for f in A.get("fetches") or []] != [f["file"] for f in B.get("fetches") or []]:
            kinds[("download", f"drafter fetched: {a} vs {b}")].append(f"{mk}/{sk}")
    return kinds


def short(v) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    s = s.replace("<ROOT>", "").replace("/local/Strata/launch/", "launch/")
    return s if len(s) < 48 else s[:45] + "..."


def render_differences(data, a, b) -> list[str]:
    kinds = differences(data, a, b)
    lines = [f"### {a} vs {b}", "", "| kind | difference | cells |", "|---|---|---|"]
    for (kind, text), cells in sorted(kinds.items()):
        shown = ", ".join(cells[:3]) + (f" ... ({len(cells)})" if len(cells) > 3 else "")
        lines.append(f"| {kind} | {text} | {shown} |")
    if not kinds:
        lines.append("| - | none | - |")
    return lines + [""]


def changed(a, b) -> list[str]:
    cnt = collections.defaultdict(list)
    for key in sorted(a):
        for entry in ENTRY_POINTS:
            ra, rb = a[key].get(entry), (b.get(key) or {}).get(entry)
            if ra != rb:
                parts = sorted(k for k in set(ra or {}) | set(rb or {}) if (ra or {}).get(k) != (rb or {}).get(k))
                cnt[(entry, tuple(parts))].append("%s/%s" % key)
    lines = ["| entry | what changed | cells |", "|---|---|---|"]
    for (entry, parts), cells in sorted(cnt.items()):
        lines.append(f"| {entry} | {', '.join(parts)} | {len(cells)}: {', '.join(cells[:4])}{' ...' if len(cells) > 4 else ''} |")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--against")
    a = ap.parse_args(argv)
    data = load(Path(a.golden))
    print("## Drafter each entry point starts the engine with\n")
    print("\n".join(table(data)) + "\n")
    print("## Where the entry points differ\n")
    for x, y in (("launch", "app_plain"), ("launch", "app_models"), ("app_models", "app_plain")):
        print("\n".join(render_differences(data, x, y)))
    if a.against:
        other = load(Path(a.against))
        print("## Changes between the two sets of golden files\n")
        print("\n".join(changed(data, other)) + "\n")
        print("## The other set's drafter table\n")
        print("\n".join(table(other)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
