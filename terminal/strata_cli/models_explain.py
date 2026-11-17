"""`strata models explain NAME` (#4887): where strata launch looks for a model's files and what it finds there.

Read only: it lists folders and reads file sizes, nothing is downloaded, moved or written. The lookup is Catalog.model_folder_trace, the one strata launch
uses (Catalog.model_folder returns its pick), and the data folder is the one the launch reads (the installed run config's, else the data folder every command
agrees on), so what this prints is what the launch does."""

from __future__ import annotations

import difflib
from pathlib import Path

RULE_WORDS = {"exact": "setup's spelling", "lower": "the lower-case tag of the run config", "case": "the same name in other letter case",
              "fallback": "none of them is there: setup's spelling, where a download would go"}


def _parse(name: str, family: str | None) -> tuple[str | None, str]:
    if "/" in name and family is None:
        fam, _, size = name.partition("/")
        return fam.strip() or None, size.strip()
    return family, name.strip()


def explain(name: str, *, family: str | None = None, data_dir: str | None = None, catalog=None) -> dict:
    """The lookup for one model as a dict (the JSON of --output json). ValueError: a name the model table does not know (with the closest match)."""
    if catalog is None:
        from strata_cli.launch.catalog import Catalog
        catalog = Catalog()
    fam, size = _parse(name, family)
    fam, size = catalog.resolve(size, fam)
    tag = catalog.tag_of(fam, size)
    if data_dir:
        base, where = Path(data_dir).expanduser(), "given with --data-dir"
    else:
        base = Path(catalog.data_dir_for(fam, size))
        installed = catalog.installed_config(tag)
        where = f"from the installed run config {installed}" if installed is not None and base != Path(catalog.data_dir()) else "the data folder strata launch uses"
    trace = catalog.model_folder_trace(base, fam, size)
    folder = Path(trace["folder"])
    files = []
    if folder.is_dir():
        try:
            for f in sorted(folder.glob("*.gguf")):
                if f.is_file():
                    files.append({"name": f.name, "bytes": f.stat().st_size, "picture_reader": f.name.lower().startswith("mmproj")})
        except OSError:
            pass
    others: list[str] = []
    models = base / "models"
    if trace["rule"] == "fallback":
        try:
            names = sorted(e.name for e in models.iterdir() if e.is_dir())
        except OSError:
            names = []
        near = difflib.get_close_matches(tag, [n.lower() for n in names], n=5, cutoff=0.3)
        others = [n for n in names if n.lower() in near] + [n for n in names if n.lower() not in near]
        others = others[:10]
    weights = sum(f["bytes"] for f in files if not f["picture_reader"])
    return {"family": fam, "model": size, "tag": tag, "data_dir": str(base), "data_dir_from": where, "models_dir_exists": models.is_dir(),
            "tried": [{"path": str(t["path"]), "rule": t["rule"], "exists": bool(t["exists"])} for t in trace["tried"]],
            "folder": str(folder), "rule": trace["rule"], "found": trace["rule"] != "fallback",
            "gguf_files": files, "weights_bytes": weights, "other_folders": others}


def words(r: dict) -> str:
    lines = [f"{r['family']}/{r['model']} (folder tag {r['tag']})",
             f"Data folder: {r['data_dir']} ({r['data_dir_from']})"]
    if not r["models_dir_exists"]:
        lines.append(f"There is no models folder in it ({Path(r['data_dir']) / 'models'}).")
    lines.append("Looked for:")
    for t in r["tried"]:
        lines.append(f"  {t['path']}  {'there' if t['exists'] else 'not there'}  ({RULE_WORDS[t['rule']]})")
    if r["found"]:
        lines.append(f"Uses: {r['folder']} ({RULE_WORDS[r['rule']]})")
        if r["gguf_files"]:
            lines.append(f"Model files: {len([f for f in r['gguf_files'] if not f['picture_reader']])}, {r['weights_bytes'] / 1e9:.1f} GB")
            for f in r["gguf_files"]:
                lines.append(f"  {f['name']}  {f['bytes'] / 1e9:.2f} GB" + ("  (picture reader)" if f["picture_reader"] else ""))
        else:
            lines.append("The folder has no .gguf files, so strata launch will download the model.")
    else:
        lines.append(f"Not found. strata launch would download the model into {r['folder']}.")
        if r["other_folders"]:
            lines.append("Folders that are there: " + ", ".join(r["other_folders"]))
    return "\n".join(lines)
