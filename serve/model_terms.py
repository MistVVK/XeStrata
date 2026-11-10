# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/model_terms.py - the terms of the models a server runs (About > Model terms, GET /v1/licenses/models).

The terms come from one place, the tables setup downloads the models by: setup.py's MODEL_TERMS (the text models, their
picture readers and the translation models) and tools/image/qimage.py's TERMS (the image models and the image tools'
weights).  Each repository's entry names its license as the license's text titles it and pins the files that hold the
text; fetch() puts them in the user's data folder (terms/), their size and SHA-256 checked, when setup or the image
tools download the model, so the page reads them offline.  Nothing is summarized: the page shows each file whole.

A model of the user's own (setup's --gguf, an image LoRA) has no entry: the page says its terms are those of whoever
made the file, with the address it was fetched from where the server fetched it (record_lora_source)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path

from serve import places

ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()


def terms_dir() -> Path:
    return places.user_dir("data") / "terms"


def _setup():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import setup
    return setup


def _qimage():
    tools = str(ROOT / "tools" / "image")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import qimage
    return qimage


def table() -> dict[str, dict]:
    """Every repository's terms: setup's and the image tools'."""
    return {**_setup().MODEL_TERMS, **_qimage().TERMS}


def revision(repo: str) -> str | None:
    return _setup().HF_REVISIONS.get(repo) or _qimage().REVISIONS.get(repo)


def _host_path(src: str) -> tuple[str, str]:
    host, repo = src.split(":", 1)
    return host, repo


def file_url(src: str, rev: str, path: str) -> str:
    """Where the text is fetched from: Hugging Face (HF_ENDPOINT, as setup's downloads) or GitHub, at its commit."""
    host, repo = _host_path(src)
    if host == "gh":
        return f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or "https://huggingface.co"
    return f"{endpoint}/{repo}/resolve/{rev}/{path}"


def page_url(src: str, rev: str, path: str) -> str:
    """The text's page for a person, at its commit."""
    host, repo = _host_path(src)
    return f"https://{'github.com' if host == 'gh' else 'huggingface.co'}/{repo}/blob/{rev}/{path}"


def text_path(src: str, rev: str, path: str) -> Path:
    host, repo = _host_path(src)
    return terms_dir() / host / repo.replace("/", "__") / rev / path


def _read_ok(src: str, rev: str, path: str, size: int, sha: str) -> bytes | None:
    p = text_path(src, rev, path)
    try:
        data = p.read_bytes()
    except OSError:
        return None
    return data if len(data) == size and hashlib.sha256(data).hexdigest() == sha else None


def fetch(repos, say: Callable[[str], None] | None = None, timeout: float = 60, terms: dict | None = None) -> list[str]:
    """The terms' texts of `repos` (repository names) that are not here yet, downloaded and checked: the names of
    those that could not be (no network, a changed file).  A failure leaves the model usable: the page then links the
    text's page instead."""
    terms = table() if terms is None else terms   # setup.py passes its own (run as a script, it is not "setup")
    failed = []
    for repo in dict.fromkeys(repos):
        for src, rev, path, size, sha in terms.get(repo, {}).get("texts", ()):
            if _read_ok(src, rev, path, size, sha) is not None:
                continue
            try:
                req = urllib.request.Request(file_url(src, rev, path), headers={"User-Agent": "xestrata-setup"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read(size + 1)
                if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                    raise ValueError("not the pinned file (size or SHA-256)")
                dst = text_path(src, rev, path)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
            except (OSError, ValueError) as e:
                failed.append(f"{repo}: {path} ({e})")
                if say:
                    say(f"the terms of {repo} ({page_url(src, rev, path)}) could not be fetched: {e}")
    return failed


class TermsFetch:
    """The server's fetch of the texts a page found missing (a model set up before setup fetched them): on a thread
    of its own, each repository tried once a run, so an offline PC asks no more than once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tried: set = set()
        self.thread: threading.Thread | None = None

    def start(self, repos) -> bool:
        """Whether a fetch runs now (started for `repos` not tried yet, or one already running)."""
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return True
            todo = [r for r in dict.fromkeys(repos) if r not in self.tried]
            if not todo:
                return False
            self.tried.update(todo)
            self.thread = threading.Thread(target=fetch, args=(todo,), kwargs={"timeout": 20}, daemon=True,
                                           name="xestrata-terms-fetch")
            self.thread.start()
            return True


def entry(repo: str, role: str, **extra) -> dict:
    """One model's row for the page: its repository, revision, license, the model card's license field, and the texts
    (each whole, or None where it is not here: its page instead)."""
    t = table().get(repo, {})
    texts = []
    for src, rev, path, size, sha in t.get("texts", ()):
        data = _read_ok(src, rev, path, size, sha)
        texts.append({"name": path, "source": src.split(":", 1)[1], "url": page_url(src, rev, path),
                      "text": data.decode("utf-8", errors="replace") if data is not None else None})
    return {"role": role, "repo": repo, "by": repo.split("/", 1)[0], "card": f"https://huggingface.co/{repo}",
            "revision": revision(repo), "license": t.get("license"), "card_license": t.get("card"),
            "texts": texts, **extra}


def own_file(role: str, name: str, source: str | None = None) -> dict:
    """A file of the user's own: its terms are its maker's."""
    return {"role": role, "own": True, "name": name, "source": source, "texts": []}


# ------------------------------------------------------------------------------------------- LoRAs' sources
def _sources_path() -> Path:
    return terms_dir() / "lora-sources.json"


def record_lora_source(files, url: str) -> None:
    """The address an image LoRA the server fetched came from, by its file's path (About > Model terms)."""
    with _lock:
        p = _sources_path()
        try:
            got = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            got = {}
        got.update({str(Path(f).resolve()): url for f in files})
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, p)


def lora_sources() -> dict[str, str]:
    try:
        return json.loads(_sources_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------- what runs
def text_model_rows(config_path: str | None, vision: bool) -> list[dict]:
    """The text model's rows, from its run config (the family setup wrote it for), and its picture reader's."""
    if not config_path:
        return []
    s = _setup()
    try:
        ch = s.choices_from_config(Path(config_path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return []
    fam = s.FAMILIES.get(ch["family"]) or {}
    if ch["family"] == s.CUSTOM:
        return [own_file("model", str(ch.get("model") or "your own files"))]
    rows = [entry(repo, role, title=fam.get("title"), maker=fam.get("by"))
            for role, repo in s.family_repos(ch["family"], ch.get("model"))
            if role == "model" or vision]
    if ch.get("esp") == "on" and s.ESP_VECTOR.is_file():   # the speed projection, Qwen3.8-Flash-Next's license
        lic = s.ESP_VECTOR.parent / "LICENSE"
        rows.append({"role": "speed_projection", "repo": None, "name": s.ESP_VECTOR.name,
                     "license": "Qwen Community License 1.0 (third_party/nonfree/experimental-speed-projection)",
                     "texts": [{"name": "LICENSE", "source": None, "url": None,
                                "text": lic.read_text(encoding="utf-8", errors="replace")}] if lic.is_file() else []})
    return rows


def _image_files() -> dict[str, str]:
    """Every image file setup and the tools download: where it is kept (its folder and name, qimage.local_path) -> its
    repository (qimage's FILES).  The folder tells apart files of the same name (each model's mmproj-F16.gguf)."""
    q = _qimage()
    out: dict[str, str] = {}
    for name, spec in q.FILES.items():
        if isinstance(spec, tuple) and len(spec) >= 2:
            try:
                local = q.local_path(Path("/"), name)
            except (KeyError, IndexError, TypeError):
                continue
            out.setdefault("/".join(local.parts[-2:]), spec[0])
    return out


def image_rows(cfgs: list[dict], aux_root: Path | None, loras: list[tuple[str, Path]]) -> list[dict]:
    """The image models' rows: each file a config names that setup downloaded (by its name), the image tools' weights
    that are here, and the LoRAs of the models' folders."""
    q = _qimage()
    by_name = _image_files()
    rows, seen = [], set()

    def add(repo: str, role: str, **extra) -> None:
        if (repo, role) not in seen:
            seen.add((repo, role))
            rows.append(entry(repo, role, **extra))

    for cfg in cfgs:
        for key, val in cfg.items():
            if isinstance(val, str) and (repo := by_name.get("/".join(Path(val).parts[-2:]))):
                add(repo, "image", model=cfg.get("model_name"), part=key)
    if aux_root is not None:
        for name, a in q.AUX.items():
            if name.endswith("-license"):
                continue
            out = Path(aux_root) / a["out"]
            spec = q.FILES.get(a["file"])
            if out.is_file() and isinstance(spec, tuple):
                agreed = None
                if a.get("consent"):   # image_aux.accept's mark: "<date> <time> <where>", then the license
                    mark = Path(aux_root) / f"{name}.accepted"
                    agreed = (mark.read_text(encoding="utf-8").split("\n", 1)[0] if mark.is_file() else None)
                add(spec[0], "tool", tool=name, agreed=agreed)
    sources = lora_sources()
    for model, f in loras:
        rows.append({**own_file("lora", f.name, sources.get(str(f.resolve()))), "model": model})
    return rows


def translation_rows(model) -> list[dict]:
    return [entry(model.repo, "translation", title=model.name)] if model is not None else []
