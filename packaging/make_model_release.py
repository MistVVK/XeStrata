#!/usr/bin/env python3
"""A model release, planned: what would be built, signed and uploaded for a new model, from the intake report.  A DRY RUN.

    python packaging/make_model_release.py --intake qwen4_intake.json --id qwen4-x --app-version 0.2.12 --sequence 123 \
        --prev-upload C:/rc214e/FINAL2/UPLOAD [--revision <HF commit>] [--engine-version 0.1.58] [--case A|B] [--write OUTDIR]

What it does: reads tools/qwen4_intake.py's JSON, takes the verdict (case A: the model runs on the engine as it is; case B: the
engines have to be rebuilt for its shape; case C is refused) and prints the plan in the order of the 0.2.11 release
(phaseA-tests/rel0211: final_rc1.py, publish211.sh, PUBLISH-PLAN.md):

    case A  the feed (a `models` entry, the feed re-signed at the next sequence). No app rebuild: the app lists feed models
            from the feed (the Settings update check is enough). `--with-app` adds the app steps when a release has app text anyway.
            Engines, DeepSeek packages and Controls stay as they are: same files, same names.
    case B  the engine of the model's shape, as its own component (`--shape`, package_engine.py --shape, feed member shape_engines: strata_control/shape_engine.py),
            plus the feed. The Strata engine, the DeepSeek packages, the Controls and the app are not rebuilt: a Control of 0.2.13 or later already installs and runs
            a shape engine, and the model's family `qwen4exp_<shape>` makes it fetch it.

Each step shows where it runs, a rough time, the command (printed, never run) and its outputs with sizes.  Sizes are those of the
previous release's files (`--prev-upload`), so they are estimates for what gets rebuilt and exact for what stays.

What it never does: sign, read or name a key, build, upload, tag, push or publish.  The steps that need the release key print
`<FEED-KEY>` for the owner to fill in at that moment.  `--execute` is refused.  With `--write OUTDIR` (a new, empty folder) it also
writes, as plain local files: the plan (md and json), the model's feed entry (DRAFT), and the release notes, model card and checklist
filled from the report with every measured number left as `[FILL: ...]` (publish211.sh refuses a body that still has one).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The release templates (the notes and the model card; forms with {{keys}} this script fills and [FILL: ...] left for a person). A release-staff
#: tool's data: the Terminal, Control and app packages do not carry packaging/. --templates names another folder (one that also holds CHECKLIST.md, say).
TEMPLATES = ROOT / "packaging" / "templates" / "model-release"
TEMPLATE_NAMES = ("RELEASE-NOTES.md", "MODEL-CARD.md", "CHECKLIST.md")
sys.path.insert(0, str(ROOT))                         # strata_control.shape_engine (the shape names)

# artifact names in a release upload folder (phaseA-tests/rel0211/PUBLISH-PLAN.md, scripts/relfinal/expect_rc1.py)
ENGINE_RE = re.compile(r"^strata-engine-(?P<v>\d[\d.]*)-(?P<plat>.+?)\.(?:tar\.gz|zip)$")
DSV41_RE = re.compile(r"^strata-dsv41-(?P<v>\d[\d.]*)-(?P<plat>.+?)\.(?:tar\.gz|zip)$")
CONTROL_RE = re.compile(r"^strata-control-(?P<v>\d[\d.]*)-(?P<plat>.+?)\.(?:tar\.gz|zip)$")
APP_RE = re.compile(r"^(Strata-\d[\d.]*-(windows-x64-setup\.exe|linux-x86_64\.AppImage|macos-arm64\.dmg)(\.sig)?|Strata\.app\.tar\.gz(\.sig)?)$")
FEED_NAMES = ["strata-feed.json", "strata-feed.json.sig", "strata-feed-linux-aarch64.json", "strata-feed-linux-aarch64.json.sig"]
SMALL_NAMES = ["SHA256SUMS.txt", "latest.json", "install.sh", "install.ps1", "install-manifest.txt"]

# where each platform's Qwen engine is built (phaseA-tests/rel0211/RELEASE-STATE.md "Build plan" and the box rules)
# (the Q4-P8 rehearsal of 2026-11-08, phaseA-tests/qwen4/RESULTS-Q4-P8.md section 5: five variants in 37 minutes)
BUILD_BOX = {"linux-x86_64-cuda": "l620", "linux-x86_64-hip": "p620", "linux-x86_64-sycl": "p1006", "windows": "owner PC",
             "macos": "m1002", "linux-aarch64": "the Spark (dgxspark)"}
BOX_NOTE = {"l620": "jobq; the CUDA build with its packaging takes about 15 min",
            "p620": "jobq; /data has little room, a package is about 4 GB",
            "p1006": "jobq",
            "owner PC": "jobq --lock heavy, 20 min per job at most (split the release set); never while a model runs",
            "m1002": "builds the unsigned Mac package (jobq); the owner signs it on his Mac",
            "the Spark (dgxspark)": "off limits unless the coordinator says"}


def gib(n: int | None) -> str:
    return "?" if n is None else ("%.2f GiB" % (n / 2**30) if n >= 2**30 else "%.1f MiB" % (n / 2**20))


def box_for(plat: str) -> str:
    for key, box in BUILD_BOX.items():
        if plat.startswith(key) or key in plat:
            return box
    return "see the 0.2.11 build plan"


def scan_upload(folder: Path | None) -> dict:
    out = {"engine": [], "dsv41": [], "control": [], "app": [], "feed": [], "small": [], "other": []}
    if folder is None:
        return out
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        n, size = p.name, p.stat().st_size
        for key, rx in (("engine", ENGINE_RE), ("dsv41", DSV41_RE), ("control", CONTROL_RE), ("app", APP_RE)):
            m = rx.match(n)
            if m:
                out[key].append({"name": n, "size": size, "plat": m.groupdict().get("plat", ""), "version": m.groupdict().get("v", "")})
                break
        else:
            out["feed" if n in FEED_NAMES else "small" if n in SMALL_NAMES else "other"].append({"name": n, "size": size})
    return out


def renamed(name: str, old: str, new: str) -> str:
    return name.replace("-%s-" % old, "-%s-" % new, 1)


def upload_minutes(nbytes: int) -> str:
    """The 0.2.10 table: 8.7 GiB took 12 to 70 minutes between 200 and 20 Mbit/s."""
    if not nbytes:
        return "-"
    lo, hi = nbytes * 8 / 200e6 / 60, nbytes * 8 / 20e6 / 60
    return "%d to %d min (200 to 20 Mbit/s)" % (max(1, round(lo)), max(1, round(hi)))


def plan(report: dict, set_index: int, a, prev: dict) -> dict:
    s = report["sets"][set_index]
    v = s.get("verdict", {})
    case = a.case or v.get("case")
    if case not in ("A", "B"):
        raise SystemExit("case %s: %s. A model release is planned for case A or B only (run tools/qwen4_intake.py)."
                         % (case, v.get("reading", "no verdict in the report")))
    with_app = bool(a.with_app)                         # a feed model needs no app rebuild (the app lists it from the feed), case B included: the shape engine is fetched by the Control
    steps: list[dict] = []
    changed: list[dict] = []        # files that are new or different in the upload folder
    kept: list[dict] = []

    def step(sid, what, where, minutes, cmds, outputs=None, owner=False, note=None):
        steps.append({"id": sid, "what": what, "where": where, "minutes": minutes, "commands": cmds, "outputs": outputs or [],
                      "owner_only": owner, "note": note})

    warn: list[str] = []
    files = s["files"]
    if any(not f.get("sha256") for f in files):
        warn.append("some files in the report have no sha256: run the intake with --sha256 (local files) or on the Hugging Face repository (the LFS hash) before the entry is final")
    if not a.revision:
        warn.append("no --revision: the entry needs the pinned Hugging Face commit")
    entry = model_entry(s, a)

    n = 1
    step("%d" % n, "Re-read the files at the pinned revision (headers only) and keep the report", "owner PC, python only", 1,
         ["python tools/qwen4_intake.py --hf %s --rev %s%s --out intake-pinned.json" % (a.hf_repo or "<REPO>", a.revision or "<COMMIT>",
                                                                                         (" --dir " + a.hf_dir) if a.hf_dir else "")]
         if (a.hf_repo or not a.revision) else ["python tools/qwen4_intake.py <the folder> --sha256 --out intake-final.json"],
         ["intake-pinned.json"])
    n += 1
    if case == "B":
        shape = a.shape
        comp = "strata-" + shape.replace("_", "-")
        for e in prev["engine"]:
            if e["name"].endswith((".tar.gz", ".zip")):
                new = renamed(e["name"], e["version"], a.engine_version).replace("strata-engine-", comp + "-", 1)
                changed.append({"name": new, "size": e["size"], "est": True, "why": "the engine of the shape %s, its own component" % shape})
        kept += [{"name": e["name"], "size": e["size"], "why": "the Strata engine is not touched: Flash-Next and every other model keep it"} for e in prev["engine"]]
        kept += [{"name": e["name"], "size": e["size"], "why": "DeepSeek packages are not touched (check that no shared kernel header changed)"} for e in prev["dsv41"]]
        kept += [{"name": e["name"], "size": e["size"], "why": "no Control rebuild: a Control of 0.2.13 or later already knows shape engines"} for e in prev["control"]]
        by_box: dict[str, list] = {}
        for e in prev["engine"]:
            by_box.setdefault(box_for(e["plat"]), []).append(e)
        step("%d" % n, "A geometry header for the model's shape, made from the file", "owner PC (python)", 2,
             ["# the shape fields the intake listed:"] + ["#   %s is %s (kernels built for %s)" % (d["key"], d["got"], d.get("kernels_built_for", d.get("limit")))
                                                          for d in v.get("shape_fields_that_differ", [])]
             + ["python tools/q4geo_synth.py --from-gguf <first shard> --header <geo.hpp> --label %s" % shape,
                "# every build passes -DSTRATA_GEOMETRY_FILE=<geo.hpp> (Linux packaging scripts: STRATA_CMAKE_EXTRA=\"-DSTRATA_GEOMETRY_FILE=<geo.hpp>\"); give the engines their own version %s" % a.engine_version],
             note="the engine built for this shape refuses Flash-Next files, so it ships as its own component named after the shape (%s) and never in place of the Strata engine (owner, 2026-11-08; RESULTS-SHAPE.md)" % comp)
        n += 1
        for box, es in sorted(by_box.items()):
            step("%d" % n, "Build the %s engine packages on %s" % (shape, box), "%s (%s)" % (box, BOX_NOTE.get(box, "see the 0.2.11 build plan")), 25 if box.startswith("owner") else 15 if box == "l620" else 10,
                 ["# per package, as in phaseA-tests/rel0211/scripts (build_*.sh), with the shape's header; then, instead of the usual package_engine.py call:",
                  "python packaging/package_engine.py --src <stage> --backend <backend> --version %s --shape %s --min-runtime <the Control that knows shape engines> ...   # the usual --arch/--variant/--tree/--min-driver of that platform" % (a.engine_version, shape),
                  "python packaging/feed_tool.py sign-package <package-folder> --key <FEED-KEY>   # OWNER'S KEY, not run here",
                  "python packaging/feed_tool.py pack-engine <package-folder> --out <shape-packs> --url-prefix=   # the archive is %s-<version>-<platform>-<backend>, the entry names the shape" % comp],
                 [renamed(e["name"], e["version"], a.engine_version).replace("strata-engine-", comp + "-", 1) + "  (" + gib(e["size"]) + ", estimated)" for e in es])
            n += 1
        step("%d" % n, "Mac: the owner signs and notarises the unsigned %s engine on his Mac, then it is signed into the feed" % shape, "the owner's Mac, about 45 min of his time", 45,
             ["bash sign-and-notarize.sh --engine strata-engine-%s-macos-arm64-metal-unsigned-folder.tar.gz   # the code is generic: the entrypoint comes from the manifest (the package's id is %s)" % (a.engine_version, comp),
              "python packaging/feed_tool.py sign-package <the signed folder> --key <FEED-KEY>   # OWNER'S KEY, not run here", "python packaging/feed_tool.py pack-engine <the signed folder> --out <shape-packs> --url-prefix="],
             [comp + "-%s-macos-arm64-metal.tar.gz" % a.engine_version], owner=True,
             note="a Mac installs a model of its own shape like every other platform (Terminal and app): the Mac engine of the shape is part of the release; after the Mac package exists and before build-feed")
    else:
        kept += [{"name": e["name"], "size": e["size"], "why": "same file, same name; the feed entry is reused"} for e in prev["engine"] + prev["dsv41"] + prev["control"]]

    step("%d" % n, "The model's feed entry (a `models` member entry, DRAFT)", "owner PC, python only", 2,
         ["python packaging/make_model_release.py --intake <report> --id %s ... --write <OUTDIR>   # writes models/%s.entry.json (the feed's schema; build-feed refuses every [FILL: ...] left in it) and models/%s.model.json (the older draft)" % (a.id, a.id, a.id)],
         ["models/%s.entry.json" % a.id], note="the entry follows strata_control/feed.py (packaging/UPDATES.md): fill the [FILL: ...] values (about, licence_url, ram_gb) by hand or with --about/--licence-url/--ram-gb, then `feed_tool.py build-feed --models <OUTDIR>/models`")
    n += 1
    if with_app:
        for e in prev["app"]:
            nm = re.sub(r"Strata-\d[\d.]*-", "Strata-%s-" % a.app_version, e["name"])
            changed.append({"name": nm, "size": e["size"], "est": True, "why": "app rebuilt with the Models page text"})
        step("%d" % n, "App text for the model (Models page) and the app bundles", "owner PC (Windows installer), a Linux box (AppImage), m1002 (Mac, unsigned; the owner signs)", 40,
             ["# edit apps/host/src (the model's name and one line), then the 0.2.11 app recipe: scripts/relwin/win_installer_rc1.ps1, the AppImage and Mac scripts",
              "python packaging/feed_tool.py set-app-platform --app-json latest.json --target <target> --archive <installer> --sig <installer.sig> --version %s" % a.app_version],
             [c["name"] for c in changed if APP_RE.match(c["name"])])
        n += 1
        changed.append({"name": "latest.json", "size": None, "est": True, "why": "updater json at %s" % a.app_version})
    else:
        kept += [{"name": e["name"], "size": e["size"], "why": "no app change"} for e in prev["app"]]
    if with_app:
        mac_extra = ""
        step("%d" % n, "Mac: the owner signs and notarises the unsigned Mac files on his Mac", "the owner's Mac, about 45 min of his time", 45,
             ["bash sign-and-notarize.sh Strata-%s-macos-arm64.dmg%s --updater-archive --out signed" % (a.app_version, mac_extra)],
             ["the signed folder (dmg, Strata.app.tar.gz and its .sig)"], owner=True)
        n += 1
    feed_cmd = ("python packaging/feed_tool.py build-feed --out <RELEASE> --key <FEED-KEY> --sequence %d --engine <engine-entries> --dsv41 <dsv41-entries> "
                "--drafters <drafter-entries> --models <models-folder> --control <control-entries>%s%s   # OWNER'S KEY; `--models` is Q4-PREP-B's change"
                % (a.sequence, " --shapes <shape-entries>" if case == "B" else "", " --app-json latest.json" if with_app else ""))
    step("%d" % n, "Sign the feeds at sequence %d (the shared one and linux-aarch64)" % a.sequence, "owner PC, with the owner's key at that moment", 3,
         [feed_cmd, "python packaging/feed_tool.py verify-feed <RELEASE> --pub <FEED-KEY.pub.json>"],
         list(FEED_NAMES), owner=True)
    for x in FEED_NAMES:
        changed.append({"name": x, "size": None, "est": False, "why": "signed at sequence %d" % a.sequence})
    n += 1
    changed.append({"name": "SHA256SUMS.txt", "size": None, "est": False, "why": "recomputed"})
    step("%d" % n, "Assemble the upload folder and check it", "owner PC, python only", 5,
         ["python phaseA-tests/rel0211/scripts/relfinal/final_rc1.py --mode real --work <FINAL> --key <FEED-KEY> --pub <FEED-KEY.pub.json>   # 0.2.11's assembly: hard links, SHA256SUMS; copy it for this version first",
          "python phaseA-tests/rel0211/scripts/relfinal/check_sums_rc1.py --upload <UPLOAD>",
          "python phaseA-tests/rel0211/scripts/relfinal/check_feeds_rc1.py --shared <UPLOAD>/strata-feed.json --arm <UPLOAD>/strata-feed-linux-aarch64.json --upload <UPLOAD> --pub <FEED-KEY.pub.json> --src <repo> --seq %d" % a.sequence],
         ["the upload folder"])
    n += 1
    up = sum(c["size"] or 0 for c in changed)
    step("%d" % n, "Publish (the owner's word first): draft release, upload, feed LAST, then flip Latest", "owner PC", None,
         ["bash phaseA-tests/rel0211/publish211.sh   # the dry run of the 0.2.11 publish; copy it for this release (tag, sequence, folders) and print its plan",
          "# the printed plan: gh release create <TAG> --draft ...; gh release upload ... (packages, Controls, installers, install files, strata-feed.json LAST); gh release edit <TAG> --draft=false --latest"],
         ["upload: %s in %d files, %s" % (gib(up), len(changed), upload_minutes(up))], owner=True)
    n += 1
    step("%d" % n, "Verify from the public URLs", "any box", 20,
         ["download every asset back and sha256sum -c SHA256SUMS.txt", "python packaging/feed_tool.py verify-feed <public strata-feed.json> --pub <FEED-KEY.pub.json>",
          "a clean-home install and `strata update --check` on a Linux box: the sequence is %d and the model is listed" % a.sequence], [])
    return {"id": a.id, "case": case, "set": s["files"][0]["name"], "steps": steps, "changed": changed, "kept": kept, "entry": entry,
            "warnings": warn, "upload_bytes": up}


def entry_family(s: dict, a) -> str:
    """The family the feed entry carries: the GGUF architecture, and for a model of its own shape (case B) `qwen4exp_<shape>`, which makes a Control fetch that shape's engine
    (strata_control/shape_engine.py) and makes an older Control, which does not know the family, leave the model out."""
    arch = s.get("architecture")
    shape = getattr(a, "shape", None)
    if shape and arch == "qwen4exp":
        from strata_control import shape_engine
        return shape_engine.family_of(shape)
    return arch


def model_entry(s: dict, a) -> dict:
    d = s.get("draft_catalog_entry", {})
    g = s.get("general", {})
    geo = s.get("geometry") or {}
    ctx = (geo.get("keys") or {}).get("context_length")
    return {
        "_status": "DRAFT: made from the intake report by packaging/make_model_release.py; field names follow the Q4-PREP-B brief and are to be checked against strata_control/feed.py",
        "id": a.id, "title": a.title or g.get("name") or d.get("title"), "by": d.get("by"),
        "licence": {"name": g.get("license"), "url": g.get("license.link") or "[FILL: licence URL]"},
        "family": entry_family(s, a), "hf": {"repo": a.hf_repo or "[FILL: repository]", "revision": a.revision or "[FILL: pinned commit]"},
        "files": [{"name": f["name"], "size": f["size"], "sha256": f.get("sha256")} for f in s["files"]],
        "mmproj": ({"name": s["projectors"][0]["file"], "size": s["projectors"][0]["size"]} if s.get("projectors") else None),
        "drafter": "mtp" if s.get("mtp", {}).get("in_file") else "none",
        "default_context": a.default_context if a.default_context else "[FILL: default context]", "context_trained": ctx,
        "setup": d.get("models", {}),
        "min_engine": a.min_engine or a.engine_version or "[FILL: minimum engine version]",
        "pictures": bool(s.get("projectors") and all(p["supported"] for p in s["projectors"])),
    }


def feed_entry(s: dict, a) -> dict:
    """The model's entry in the feed's `models` member, in the schema strata_control/feed.py reads (packaging/UPDATES.md). Every value the report cannot know
    (about, licence, licence_url, ram_gb, quant name) is a "[FILL: ...]" text, which `feed_tool.py build-feed --models` refuses: it is filled by hand, then built."""
    g = s.get("general", {})
    d = s.get("draft_catalog_entry", {})
    files = []
    for f in s["files"]:
        item = {"name": f["name"].rsplit("/", 1)[-1], "size": f["size"], "sha256": f.get("sha256") or "[FILL: sha256]"}
        if "/" in f["name"]:
            item["path"] = f["name"]
        files.append(item)
    first = s["files"][0]["name"]
    quant = a.quant or a.hf_dir or (first.rsplit("/", 1)[0] if "/" in first else "[FILL: quant name]")
    e = {"id": a.id, "title": a.title or g.get("name") or d.get("title") or a.id, "by": a.by or d.get("by") or "[FILL: who made the model]",
         "about": a.about or "[FILL: one plain sentence]",
         "licence": a.licence or g.get("license") or "[FILL: licence]",
         "licence_url": a.licence_url or g.get("license.link") or "[FILL: https licence url]",
         "family": entry_family(s, a), "repo": a.hf_repo or "[FILL: repository]", "revision": a.revision or "[FILL: pinned commit, 40 hex]",
         "quants": [{"name": quant, "about": a.quant_about or "[FILL: one plain line]", "ram_gb": a.ram_gb if a.ram_gb else "[FILL: ram_gb, measured]", "files": files}],
         "pictures": bool(s.get("projectors") and all(p["supported"] for p in s["projectors"])),
         "drafter": "mtp" if s.get("mtp", {}).get("in_file") else "none",
         "context": a.default_context if a.default_context else "[FILL: default context]",
         "min_engine": a.min_engine or a.engine_version or "[FILL: minimum engine version]"}
    if s.get("projectors"):
        pr = s["projectors"][0]
        e["mmproj"] = {"name": pr["file"], "size": pr["size"], "sha256": pr.get("sha256") or "[FILL: sha256]"}
    if s.get("architecture") == "qwen35moe":
        for k in ("kv_bytes_per_token", "state_gb"):
            if d.get(k) is not None:
                e[k] = d[k]
    return e


def builtin_sizes() -> set | None:
    """The built-in size names (setup.py MODELS), which a feed size may not take: None when setup.py cannot be read here."""
    import contextlib
    import io
    sys.path.insert(0, str(ROOT))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import setup as setup_mod
        return set(setup_mod.MODELS)
    except (Exception, SystemExit):                       # noqa: BLE001 - the launch line then names both possible spellings
        return None


def launch_lines(entry: dict, sizes: set | None) -> list[str]:
    """The `strata launch` lines for each size of the feed entry, with the size name a person types: the quant's name, or <quant>-<id> when that
    name is a built-in size (strata_control/feed_models.py _flash_next takes the same decision)."""
    out = []
    mid = entry.get("id")
    for q in entry.get("quants") or []:
        name = str(q.get("name"))
        if sizes is None:
            keys = [name, "%s-%s" % (name, mid)]
        else:
            keys = ["%s-%s" % (name, mid) if name in sizes else name]
        for key in keys:
            out.append("strata launch --family %s --model %s" % (mid, key.lower()))
            out.append("strata launch --family %s --model %s --model-path <the folder with the files>   # files already on the computer (an Intel Arc card: always)" % (mid, key.lower()))
    if sizes is None:
        out.append("# setup.py could not be read here: the size is the first spelling unless its quant name is a built-in size")
    return out


def render_md(p: dict, a) -> str:
    out = ["# Model release plan: %s (case %s)" % (p["id"], p["case"]), "",
           "DRY RUN. Nothing was built, signed, uploaded or published. The set: `%s`." % p["set"], ""]
    for w in p["warnings"]:
        out.append("- WARNING: " + w)
    out.append("")
    for st in p["steps"]:
        out.append("## Step %s: %s%s" % (st["id"], st["what"], "   [OWNER]" if st["owner_only"] else ""))
        out.append("Where: %s. Time: %s." % (st["where"], ("about %d min" % st["minutes"]) if st["minutes"] else "the upload time below"))
        if st.get("note"):
            out.append("Note: " + st["note"])
        out.append("")
        out += ["    " + c for c in st["commands"]]
        if st["outputs"]:
            out.append("")
            out += ["- " + o for o in st["outputs"]]
        out.append("")
    out.append("## Files in the upload folder")
    out.append("")
    out.append("| file | size | why |")
    out.append("|---|---:|---|")
    for c in p["changed"]:
        out.append("| %s | %s%s | %s |" % (c["name"], gib(c["size"]) if c["size"] else "small", " (est.)" if c.get("est") and c["size"] else "", c["why"]))
    out.append("")
    out.append("Kept from the previous release (same files, not uploaded again): %d files, %s." % (len(p["kept"]), gib(sum(k["size"] or 0 for k in p["kept"]))))
    out.append("")
    out.append("New to upload: %s in %d files; %s." % (gib(p["upload_bytes"]), len(p["changed"]), upload_minutes(p["upload_bytes"])))
    if p.get("launch"):
        out.append("")
        out.append("## How a person starts it (the size name as the Terminal takes it)")
        out.append("")
        out += ["    " + line for line in p["launch"]]
    return "\n".join(out) + "\n"


def fill(template: str, values: dict) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))), template)


def template_values(s: dict, p: dict, a) -> dict:
    g = s.get("general", {})
    rows = "\n".join("| %s | %s | %s |" % (f["name"], ("%.1f GB" % (f["size"] / 1e9)), (f.get("sha256") or "[FILL: sha256]")[:16] + ("..." if f.get("sha256") else ""))
                     for f in s["files"])
    return {"model_title": a.title or g.get("name") or s["files"][0]["name"], "app_version": a.app_version, "engine_version": a.engine_version or "[FILL: engine version]",
            "sequence": a.sequence, "case": p["case"], "file_rows": rows, "total_gb": "%.1f" % (s["size_bytes"] / 1e9),
            "n_files": len(s["files"]), "licence": g.get("license") or "[FILL: licence]", "licence_url": g.get("license.link") or "[FILL: licence URL]",
            "family": s.get("architecture"), "repo": a.hf_repo or "[FILL: repository]", "revision": a.revision or "[FILL: pinned commit]",
            "drafter": "MTP (in the file)" if s.get("mtp", {}).get("in_file") else "none", "id": a.id,
            "pictures": "yes" if p["entry"]["pictures"] else "no", "tool_calls": s.get("chat_template", {}).get("tool_call_format", "[FILL]")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intake", type=Path, required=True, help="the JSON report of tools/qwen4_intake.py")
    ap.add_argument("--set", type=int, default=0, help="which set of the report (default 0)")
    ap.add_argument("--id", required=True, help="the model's id in the feed")
    ap.add_argument("--app-version", required=True)
    ap.add_argument("--engine-version", help="case B: the version of the new shape engine")
    ap.add_argument("--shape", help="case B: the name of the geometry the engines are compiled for (q4_lite): the engine ships as its own component strata-q4-<shape> and the model's family is qwen4exp_<shape>")
    ap.add_argument("--min-engine", help="the minimum engine version for the model (case A: the engine that runs it)")
    ap.add_argument("--sequence", type=int, required=True, help="the next feed sequence")
    ap.add_argument("--prev-upload", type=Path, help="the previous release's upload folder (names and sizes)")
    ap.add_argument("--case", choices=["A", "B"], help="override the report's verdict")
    ap.add_argument("--revision", help="the pinned Hugging Face commit")
    ap.add_argument("--hf-repo")
    ap.add_argument("--hf-dir")
    ap.add_argument("--default-context", type=int)
    ap.add_argument("--title", help="the model's title in the feed entry and the notes (default: the file's general.name, e.g. 'Qwen3.8-Flash-Next GSQ-RCO')")
    ap.add_argument("--by", help="the feed entry's `by` (who made the model)")
    ap.add_argument("--about", help="the feed entry's one plain sentence")
    ap.add_argument("--licence", help="the licence name (default: the file's general.license)")
    ap.add_argument("--licence-url", help="the https address of the licence (default: general.license.link)")
    ap.add_argument("--quant", help="the quant's name in the entry (default: --hf-dir, or the folder of the files)")
    ap.add_argument("--quant-about", help="one plain line about the quant")
    ap.add_argument("--ram-gb", type=float, help="the RAM the quant needs, measured (the intake's estimate is not used)")
    ap.add_argument("--no-app", action="store_true", help="case B: no app change in this release (case A has none unless --with-app)")
    ap.add_argument("--with-app", action="store_true", help="plan the app steps (Models page text, installers) too; a case-A feed model needs none")
    ap.add_argument("--templates", type=Path, default=TEMPLATES,
                    help="the folder with the templates: RELEASE-NOTES.md, MODEL-CARD.md and, if there, CHECKLIST.md (default: packaging/templates/model-release)")
    ap.add_argument("--checklist", type=Path, help="the release checklist template by path (an internal work list, not in the repo); filled as CHECKLIST.md beside the "
                                                    "notes and the model card, and used instead of a CHECKLIST.md in --templates")
    ap.add_argument("--write", type=Path, metavar="OUTDIR", help="also write the plan, the entry and the filled templates into this new folder")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    if a.execute:
        print("REFUSED: make_model_release.py is a dry run. It signs, builds and uploads nothing; the steps that need the release key are printed for the owner.")
        return 2
    if a.checklist is not None and not a.checklist.is_file():
        raise SystemExit("--checklist: no file %s" % a.checklist)
    report = json.loads(a.intake.read_text(encoding="utf-8"))
    if not 0 <= a.set < len(report["sets"]):
        raise SystemExit("the report has %d set(s)" % len(report["sets"]))
    s = report["sets"][a.set]
    case = a.case or s.get("verdict", {}).get("case")
    if case == "B" and not a.engine_version:
        raise SystemExit("case B builds the engine of the new shape: give --engine-version")
    if case == "B":
        from strata_control import shape_engine
        if not a.shape or not shape_engine.valid(a.shape):
            raise SystemExit("case B: give --shape, the name of the geometry the engines are compiled for (like q4_lite: lower case, digits and _); that engine ships as its own component")
    elif a.shape:
        raise SystemExit("--shape is for case B (a model of a new shape); case A runs on the engines as they are")
    if case == "A" and a.write and not a.min_engine:
        raise SystemExit("case A: give --min-engine, the first engine version that runs the model (the entry needs it, and build-feed refuses an entry that still says [FILL])")
    prev = scan_upload(a.prev_upload)
    if a.prev_upload is None:
        print("note: no --prev-upload, so sizes of the files that stay or get rebuilt are unknown", file=sys.stderr)
    p = plan(report, a.set, a, prev)
    p["launch"] = launch_lines(feed_entry(s, a), builtin_sizes())
    if a.json:
        print(json.dumps(p, indent=1))
    else:
        print(render_md(p, a))
    if a.write:
        if a.write.exists() and any(a.write.iterdir()):
            raise SystemExit("--write needs a new or empty folder: %s" % a.write)
        (a.write / "models").mkdir(parents=True, exist_ok=True)
        (a.write / "PLAN.md").write_text(render_md(p, a), encoding="utf-8")
        (a.write / "plan.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
        (a.write / "models" / ("%s.model.json" % a.id)).write_text(json.dumps(p["entry"], indent=1), encoding="utf-8")
        (a.write / "models" / ("%s.entry.json" % a.id)).write_text(json.dumps(feed_entry(s, a), indent=1) + "\n", encoding="utf-8")
        vals = template_values(s, p, a)
        for name in TEMPLATE_NAMES:
            t = a.templates / name if a.templates else None
            if name == "CHECKLIST.md" and a.checklist is not None:
                t = a.checklist
            if t is not None and t.is_file():
                (a.write / name).write_text(fill(t.read_text(encoding="utf-8"), vals), encoding="utf-8")
            else:
                print("note: no template %s in %s (give --templates)" % (name, a.templates), file=sys.stderr)
        print("written to %s" % a.write, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
