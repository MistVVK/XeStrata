"""DeepSeek models that this Strata no longer runs: how a saved config, a folder, a file or a typed name is recognised, and the words
a person sees.

Strata ran DeepSeek-V4.1-Flash up to 0.2.13 on an engine of its own (strata-dsv41).  This version does not: a DeepSeek config is
refused before anything starts and before any file of it is written, its model files are never touched, and the words name where
they are and how large, so the person can decide.  Every check here is local and bounded (config fields, a file name, the first
bytes of one GGUF, the sizes of the files directly in one or two folders): it answers at once, also offline, and never raises.
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping

#: the family and arch names setup and the server used for DeepSeek-V4.1-Flash
FAMILY = "deepseek41"
ENGINE_NAMES = ("strata-dsv41", "strata-dsv41.exe")
#: the model sizes setup and `strata launch` offered under --family deepseek41 (lower case); "UD-IQ1_M" and "UD-IQ1_S" are not here: they are also
#: Unsloth Qwen names
MODEL_NAMES = frozenset({"ist-das-2.00bpw", "ist-das-2.25bpw", "ist-das-1.75bpw", "ista-das-coder", "ud-q8_k_xl",
                         "deepseek-v4.1-flash", "deepseek41", "deepseek"})
#: the config file stem setup wrote (strata-deepseek-v4.1-flash-<size>.json) and the model_name in it
TAG = "deepseek-v4.1-flash"
#: the published file names of the DeepSeek sets (unsloth, ISTA-DAS, the Engram table, the DSpark drafter)
_FILE = re.compile(r"^(?:dspark-)?DeepSeek-V4[._]?1-Flash[-_].*\.gguf$", re.I)
_DIR_ENTRIES = 200            # entries looked at in one folder: a bounded look, never a walk
_GGUF_HEAD = 1 << 20          # bytes of a GGUF read to find general.architecture

DRAFT = "[DRAFT] "            # the sentences marked with it wait for the owner's words (par/OWNER-WORDING-1111.md)

#: D1: a start of a saved DeepSeek model (the app, `strata start`, a saved launch); approved by the owner 11-11 (par/OWNER-WORDING-1111.md)
REFUSED = "{name} is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started."
FILES_STAY = "Its files are still on this computer: {size} in {folder}. Nothing was changed or deleted."
NEXT = ("Choose a Qwen model to run instead. If you no longer need the DeepSeek files, you can delete that folder "
                "to free the space.")
#: D2: the model list's line for a DeepSeek row
ROW = DRAFT + "DeepSeek - no longer supported."
ROW_FILES = DRAFT + "DeepSeek - no longer supported. {size} in {folder}."
#: D3: DeepSeek asked for by name (`strata launch --model`, `--family deepseek41`, setup, a DeepSeek-only flag)
NOT_OFFERED = DRAFT + "Strata no longer offers DeepSeek models. Run `strata launch` to choose one of the Qwen models."
#: D4: the person's own DeepSeek files (`strata launch --model-path`, the own-files screen, `strata models import`)
OWN_FILES = DRAFT + ("These are DeepSeek model files ({size} in {folder}). This version of Strata no longer runs DeepSeek models; "
                     "nothing in the folder was changed. Choose a Qwen model instead.")
OWN_FILES_NOSIZE = DRAFT + ("These are DeepSeek model files (in {folder}). This version of Strata no longer runs DeepSeek models; "
                            "nothing in the folder was changed. Choose a Qwen model instead.")
#: D5: the server started directly with a DeepSeek config or tokenizer
SERVER = DRAFT + ("[strata] this config runs a DeepSeek model ({why}). This version of Strata no longer runs DeepSeek models, so the "
                  "server stops here. The model files were not changed.")
#: D6: setup's update and its list of installed models
LEFT_AS_IS = DRAFT + "{name}: a DeepSeek model, left as it is (this version of Strata no longer runs DeepSeek models)."

WHY = {"arch": "arch deepseek41", "engine": "engine strata-dsv41", "tokenizer": "DeepSeek tokenizer", "gguf": "DeepSeek model file",
       "name": "DeepSeek model name"}


# ----------------------------------------------------------------------------------------------------------- recognising
def is_name(word: Any) -> bool:
    """A model, size or family name typed or saved for DeepSeek (`--model IST-DAS-2.00bpw`, `--family deepseek41`, a saved choice)."""
    if not isinstance(word, str) or not word.strip():
        return False
    w = word.strip().lower()
    return w in MODEL_NAMES or w.startswith(TAG) or w.startswith("deepseek-v4")


def is_file_name(name: Any) -> bool:
    """A published DeepSeek-V4.1-Flash file name (a shard, the Engram table, the DSpark drafter)."""
    return isinstance(name, str) and bool(_FILE.match(Path(name).name))


def gguf_arch(path: Any) -> str | None:
    """general.architecture of a GGUF file, read from its first megabyte (None: not a GGUF, unreadable, or not found early)."""
    try:
        with open(path, "rb") as f:
            head = f.read(_GGUF_HEAD)
    except (OSError, TypeError, ValueError):
        return None
    if len(head) < 24 or head[:4] != b"GGUF":
        return None
    try:
        pos = 8 + 8                                    # magic, version, tensor count
        (n_kv,) = struct.unpack_from("<Q", head, pos)
        pos += 8
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(min(n_kv, 64)):
            (klen,) = struct.unpack_from("<Q", head, pos)
            pos += 8
            key = head[pos:pos + klen].decode("utf-8", "replace")
            pos += klen
            (vtype,) = struct.unpack_from("<I", head, pos)
            pos += 4
            if vtype == 8:                             # a string
                (slen,) = struct.unpack_from("<Q", head, pos)
                pos += 8
                value = head[pos:pos + slen].decode("utf-8", "replace")
                pos += slen
                if key == "general.architecture":
                    return value
            elif vtype in sizes:
                pos += sizes[vtype]
            else:                                      # an array before the architecture: stop looking
                return None
            if pos > len(head):
                return None
    except (struct.error, ValueError):
        return None
    return None


def _tokenizer_pre(cfg: Mapping) -> str | None:
    tok = cfg.get("tokenizer")
    if not isinstance(tok, str) or not tok:
        return None
    try:
        with open(Path(tok) / "tokenizer.json", "rb") as f:
            data = f.read(_GGUF_HEAD)
        pre = json.loads(data.decode("utf-8")).get("pre")
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return pre if isinstance(pre, str) else None


def config_reason(cfg: Any) -> str | None:
    """Why a run config is a DeepSeek one ("arch", "engine", "tokenizer", "name"), or None.  Reads the config's fields and, only when it
    names no arch, the first megabyte of its tokenizer.json."""
    if not isinstance(cfg, Mapping):
        return None
    arch = cfg.get("arch")
    if isinstance(arch, str) and arch.strip().lower().startswith("deepseek"):
        return "arch"
    exe = cfg.get("exe")
    if isinstance(exe, str) and re.split(r"[\\/]", exe)[-1].lower() in ENGINE_NAMES:
        return "engine"
    if isinstance(cfg.get("model_name"), str) and is_name(cfg["model_name"]):
        return "name"
    if not arch:
        pre = _tokenizer_pre(cfg)
        if pre and pre.lower().startswith("deepseek"):
            return "tokenizer"
    return None


def read_config(path: Any) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def config_file_reason(path: Any) -> str | None:
    """config_reason of a run config file (an unreadable file is not DeepSeek's: the usual errors say what is wrong with it)."""
    stem = Path(str(path)).stem.lower() if path else ""
    if stem.startswith("strata-" + TAG):
        return "name"
    return config_reason(read_config(path))


def folder_reason(path: Any) -> str | None:
    """Why a folder or file the person points at holds a DeepSeek model ("name": a published file name, "gguf": a GGUF whose architecture
    is DeepSeek's), or None.  Looks at most at _DIR_ENTRIES names and reads the head of at most one GGUF."""
    try:
        p = Path(path)
        if p.is_file():
            if is_file_name(p.name):
                return "name"
            arch = gguf_arch(p)
            return "gguf" if arch and arch.lower().startswith("deepseek") else None
        if not p.is_dir():
            return None
        first = None
        with os.scandir(p) as it:
            for i, entry in enumerate(it):
                if i >= _DIR_ENTRIES:
                    break
                if is_file_name(entry.name):
                    return "name"
                if first is None and entry.name.lower().endswith(".gguf") and entry.is_file():
                    first = entry.path
        if first:
            arch = gguf_arch(first)
            return "gguf" if arch and arch.lower().startswith("deepseek") else None
    except OSError:
        return None
    return None


# ------------------------------------------------------------------------------------------------------ where and how large
def _top_files(folder: Path) -> tuple[int, int]:
    """(bytes, files) of the files directly in `folder` (at most _DIR_ENTRIES entries; symlinks followed, folders not entered)."""
    total = count = 0
    try:
        with os.scandir(folder) as it:
            for i, entry in enumerate(it):
                if i >= _DIR_ENTRIES:
                    break
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                        count += 1
                except OSError:
                    continue
    except OSError:
        pass
    return total, count


def _pack_model_dir(pack: Path) -> Path | None:
    """The model folder a DeepSeek pack names (its ds4.txt line `model_dir <folder>`)."""
    try:
        with open(pack / "ds4.txt", "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 400:
                    break
                if line.startswith("model_dir "):
                    return Path(line[len("model_dir "):].strip())
    except OSError:
        return None
    return None


def config_folders(cfg: Mapping) -> list[Path]:
    """The folders that hold a DeepSeek config's model: the model folder its pack names, then the pack itself (existing ones only)."""
    args = [str(x) for x in (cfg.get("args") or []) if isinstance(x, (str, int, float))]
    out: list[Path] = []
    for flag in ("--pack", "--model"):
        if flag in args and args.index(flag) + 1 < len(args):
            p = Path(args[args.index(flag) + 1])
            model_dir = _pack_model_dir(p) if flag == "--pack" else None
            for q in (model_dir, p if p.is_dir() else p.parent if p.is_file() else None):
                if q is not None and q.is_dir() and q not in out:
                    out.append(q)
    return out


def where(folders: Iterable[Path]) -> tuple[str, int] | None:
    """("<folder>[ and <folder>]", bytes) of the files directly in these folders, or None when none is there."""
    folders = list(folders)
    total, names = 0, []
    for f in folders:
        size, count = _top_files(f)
        if count:
            total += size
            names.append(str(f))
    return (" and ".join(names), total) if names else None


def size_words(n: int) -> str:
    gb = max(n, 0) / 1e9
    return f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"


# ----------------------------------------------------------------------------------------------------------------- words
_LABELS = {"ist-das-2.00bpw": "ISTA-DAS 2.00 bpw", "ist-das-2.25bpw": "ISTA-DAS 2.25 bpw", "ist-das-1.75bpw": "ISTA-DAS 1.75 bpw",
           "ista-das-coder": "ISTA-DAS Coder", "ud-q8_k_xl": "Unsloth UD-Q8_K_XL", "ud-iq1_s": "Unsloth UD-IQ1_S", "ud-iq1_m": "Unsloth UD-IQ1_M"}


def title(cfg: Mapping | None = None, path: Any = None) -> str:
    """The model's name as setup showed it ("DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw"), from the config file's name when it has setup's form."""
    stem = Path(str(path)).stem.lower() if path else ""
    head = "strata-" + TAG + "-"
    if stem.startswith(head):
        size = stem[len(head):]
        return "DeepSeek-V4.1-Flash " + _LABELS.get(size, size.upper())
    return "DeepSeek-V4.1-Flash"


def refusal(cfg: Mapping | None, name: str | None = None) -> str:
    """D1: the whole refusal for a saved DeepSeek model: what happened, where its files are and how large (when that can be read), the
    next step."""
    title = name or (cfg or {}).get("model_name") or "This model"
    lines = [REFUSED.format(name=title)]
    found = where(config_folders(cfg)) if isinstance(cfg, Mapping) else None
    if found:
        lines.append(FILES_STAY.format(size=size_words(found[1]), folder=found[0]))
    lines.append(NEXT)
    return " ".join(lines)


def row_line(cfg: Mapping | None) -> str:
    """D2: the model list's line for a DeepSeek row."""
    found = where(config_folders(cfg)) if isinstance(cfg, Mapping) else None
    return ROW_FILES.format(size=size_words(found[1]), folder=found[0]) if found else ROW


def own_files(path: Any) -> str:
    """D4: the person pointed at DeepSeek files."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    found = where([folder])
    return OWN_FILES.format(size=size_words(found[1]), folder=found[0]) if found else OWN_FILES_NOSIZE.format(folder=folder)


def server_line(reason: str) -> str:
    """D5."""
    return SERVER.format(why=WHY.get(reason, reason))


def left_as_is(name: str) -> str:
    """D6."""
    return LEFT_AS_IS.format(name=name)
