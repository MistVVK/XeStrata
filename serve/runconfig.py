# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/runconfig.py - #564: the web page's Settings view of the run config (xestrata-<model>.json).

A short list of documented keys can be read and changed from XeStrata's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as xestrata-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.

A few keys can also be set for every model, in the user's settings (settings.json beside the configs,
serve/places.py): GLOBAL lists them.  A model's config overrides the settings key by key (effective), so a model can
turn off or change what the others use.  The settings file is setup's as well (the data folder, the installs): a
change rewrites only the keys it names.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.parse
from pathlib import Path

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", or ("sampling", check) for a key
# of the "sampling" block, ("arg", flag) for an engine option kept in "args".
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy; without it the model card's: 1.0 thinking, "
     "0.7 not)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool",
     "Shorten a max_tokens that does not fit the context (on by default); off: answer 400 like llama.cpp"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("script_guard", ("enum", ["off", "ja", "en", "ko", "zh-cn", "zh-tw"]),
     "Keep other languages' characters out of the answers of requests that name none: ja, en "
     "(English and the like: no CJKV), ko, zh-cn or zh-tw; off: as the model writes"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0",
     "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    ("search_engine", ("enum", ["searxng", "deepseek"]),
     "Web search: searxng (the server at search_url) or deepseek (DeepSeek's search API, billed per search)"),
    ("search_url", "url",
     "Web search: the address of a SearXNG-compatible search server (http://127.0.0.1:8888); empty in a model's "
     "config: no search for that model"),
    ("deepseek_api_key", "secret",
     "Web search with DeepSeek: its API key (write-only: never shown; the queries go to DeepSeek)"),
]
SPEC = {k: kind for k, kind, _ in EDITABLE}
# the keys settings.json may hold for every model
GLOBAL = ("search_engine", "search_url", "deepseek_api_key", "script_guard")
GLOBAL_ONLY = ("deepseek_api_key",)                  # ... and the ones only it holds (a secret: one place)


def _arg(cfg: dict, flag: str):
    args = cfg.get("args")
    a: list = args if isinstance(args, list) else []
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() else v
    return None


def value_of(cfg: dict, key: str):
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if isinstance(kind, tuple) and kind[0] == "arg":
        return _arg(cfg, kind[1])
    return cfg.get(key)


def effective(cfg: dict, settings: dict, key: str):
    """The value a GLOBAL key has for this model: the config's when it has the key (an empty one included), else the
    settings'."""
    return cfg[key] if key in cfg else settings.get(key)


def view(cfg: dict, path: str | Path, settings: dict | None = None, settings_path: str | Path | None = None) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies).  A GLOBAL key also
    gives the settings' value and the one in effect, with where it comes from ("model", "global" or None)."""
    settings = settings or {}
    out = []
    for key, kind, help_ in EDITABLE:
        k = kind[0] if isinstance(kind, tuple) else kind
        row = {"key": key, "value": value_of(cfg, key), "help": help_,
               "kind": "number" if k in ("sampling", "arg", "int>=0", "num>=0") else k,
               **({"choices": kind[1]} if k == "enum" else {}), "scopes": ["model"]}
        if k == "secret":                                # its value never leaves the server: only whether it is set
            row.update({"value": None, "scopes": ["global"], "global": None, "set": bool(settings.get(key))})
        elif key in GLOBAL:
            row.update({"scopes": ["global", "model"], "global": settings.get(key),
                        "effective": effective(cfg, settings, key),
                        "from": "model" if key in cfg else "global" if settings.get(key) is not None else None})
        out.append(row)
    return {"file": Path(path).name, "keys": out,
            **({"global_file": str(settings_path)} if settings_path is not None else {}),
            "note": "Saved to the run config (or the settings, for every model); used from the next start of the "
                    "model."}


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "secret":
        if not isinstance(v, str) or not 8 <= len(v.strip()) <= 512 or any(c.isspace() for c in v.strip()):
            raise ValueError(f"{key}: expected an API key (one word, no spaces)")   # never the value itself
        return v.strip()
    if k == "url":
        if not isinstance(v, str) or len(v) > 2048:
            raise ValueError(f"{key}: expected an http or https address, not {v!r}")
        v = v.strip()
        u = urllib.parse.urlsplit(v)
        if v and (u.scheme not in ("http", "https") or not u.hostname):
            raise ValueError(f"{key}: expected an http or https address (http://127.0.0.1:8888), not {v!r}")
        return v.rstrip("/")
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        return _number(key, v, whole=True)
    if k == "num>=0":
        return _number(key, v)
    if k == "arg":
        return _number(key, v, whole=True)
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict, scope: str = "model") -> tuple[dict, list[str]]:
    """The config (or with scope "global", the settings) with `changes` ({key: value or None}) made, and the keys
    that changed.  Every value is checked first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    if scope not in ("model", "global"):
        raise ValueError(f'scope: expected "model" or "global", not {scope!r}')
    if scope == "global" and any(k not in GLOBAL for k in changes):
        raise ValueError(f"only {', '.join(GLOBAL)} can be set for every model")
    if scope == "model" and any(k in GLOBAL_ONLY for k in changes):
        raise ValueError(f"{', '.join(k for k in changes if k in GLOBAL_ONLY)} is set for every model only "
                         '(scope "global")')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    if scope == "global":                              # empty there is the same as unset (only a model's config can
        checked = {k: None if v == "" else v for k, v in checked.items()}   # turn a key off for itself)
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        if value_of(new, key) == v:
            continue
        kind = SPEC[key]
        if isinstance(kind, tuple) and kind[0] == "sampling":
            s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
            sub = key.split(".", 1)[1]
            if v is None:
                s.pop(sub, None)
            else:
                s[sub] = v
            if s:
                new["sampling"] = s
            else:
                new.pop("sampling", None)
        elif isinstance(kind, tuple) and kind[0] == "arg":
            a = list(new.get("args") or [])
            flag = kind[1]
            if flag in a[:-1]:
                i = a.index(flag)
                if v is None:
                    del a[i:i + 2]
                else:
                    a[i + 1] = str(v)
            elif v is not None:
                a += [flag, str(v)]
            new["args"] = a
        elif v is None:
            new.pop(key, None)
        else:
            new[key] = v
        changed.append(key)
    return new, changed


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = path.with_name(path.name + ".bak")
    shutil.copyfile(path, bak)
    os.chmod(bak, 0o600)                               # #33: the config and its copy hold the api_key: this user's only
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=1))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return bak


def save_settings(path: str | Path, settings: dict) -> None:
    """The user's settings written whole and readable by this user only, as setup writes them (no .bak: setup keeps
    none either)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(json.dumps(settings, indent=1))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_settings(path: str | Path) -> dict:
    """The user's settings; none yet is {}.  One that cannot be read is a ValueError (OSError), never taken for
    empty: writing it back would lose setup's keys."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    s = json.loads(text)
    if not isinstance(s, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return s


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
