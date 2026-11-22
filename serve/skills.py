# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/skills.py - the chat's skills, in Agent Skills' form: a folder whose SKILL.md starts with a front matter that
names the skill and says when to read it (`name`, `description`), its body the skill.

Two kinds, read the same way:

- XeStrata's own (serve/skills/): the image tools' rules that always go with them (`image-tools`), the prompt styles,
  the image models' and the tools' skills.  Their `metadata` says what each is for (`kind`: always, style, model, tool;
  `for`: the family or model, or the tool, it goes with).  They can be switched off, not changed or deleted.
- The user's (the data folder's skills/, or the config's "skills_dir"): added, changed, deleted and switched on or off
  from the web page.  A chat request of the web page's (xestrata_skills) lists those switched on.

The chat's model sees the list (a skill's name and when to read it) in load_skill's description and reads the bodies
it needs with load_skill, several at once; a conversation carries no body it does not read.  The image tools' own
part of the description (their rules, what this setup can do, which skills go with which model and tool) is the image
tools' (serve/images.py ImageTools.skill_part); load_skill joins it to the user's skills, so there is one list and one
way to read a skill.

A user's skill comes from outside XeStrata: its body reaches the model marked as the user's, as a tool's result does.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BUILTIN = Path(__file__).resolve().parent / "skills"
TOOL = "load_skill"
# Agent Skills' rules for a name and a description; a body (and a SKILL.md) of a user's at most MAX_BODY bytes
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
NAME_MAX = 64
DESCRIPTION_MAX = 1024
MAX_BODY = 64 * 1024
STATE = ".state.json"   # the skills switched off, built-in and the user's: {"off": [name, ...]}
KINDS = ("always", "catalog", "style", "model", "tool", "user")   # the order skills read together are joined in


class SkillError(ValueError):
    """A skill that cannot be added, changed or read: the message says why."""


@dataclass
class Skill:
    name: str
    description: str
    body: str
    builtin: bool
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind") or ("tool" if self.meta.get("tool") else "user"))


def parse(text: str, where: str = "SKILL.md") -> tuple[dict, str]:
    """SKILL.md -> (its front matter, its body without a license comment).  SkillError when the front matter is missing
    or is not a mapping."""
    m = re.match(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", text, flags=re.S)
    if not m:
        raise SkillError(f"{where}: it starts with a front matter (--- name: ... description: ... ---)")
    try:
        front = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        raise SkillError(f"{where}: the front matter is not YAML ({str(e).splitlines()[0]})") from None
    if not isinstance(front, dict):
        raise SkillError(f"{where}: the front matter is a mapping (name: ..., description: ...)")
    body = re.sub(r"\A\s*<!--.*?-->\s*", "", text[m.end():], flags=re.S).strip()
    return front, body


def check(front: dict, body: str) -> tuple[str, str]:
    """A skill's name and description, as Agent Skills require them -> (name, description); SkillError otherwise."""
    name, desc = front.get("name"), front.get("description")
    if not isinstance(name, str) or not NAME.fullmatch(name) or len(name) > NAME_MAX:
        raise SkillError(f"name: lowercase letters, digits and single hyphens, at most {NAME_MAX} characters "
                         "(e.g. my-skill)")
    if not isinstance(desc, str) or not desc.strip() or len(desc) > DESCRIPTION_MAX:
        raise SkillError(f"description: what the skill does and when to read it, 1 to {DESCRIPTION_MAX} characters")
    if not body:
        raise SkillError("the skill has no body below its front matter")
    return name, " ".join(desc.split())


def _read(path: Path, builtin: bool) -> Skill:
    front, body = parse(path.read_text(encoding="utf-8"), str(path))
    name, desc = check(front, body)
    if name != path.parent.name:
        raise SkillError(f"{path}: its name {name!r} is not its folder's")
    meta = front.get("metadata")
    return Skill(name, desc, body, builtin,
                 {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {})


def _read_external(path: Path) -> Skill:
    """A built-in skill read for GET /v1/skills.  Nine of the files on disk have ': ' in an unquoted description,
    which YAML reads as a mapping and the chat's _read() skips.  Quote those two values here instead of changing a
    file or the chat's reader: the external endpoint serves every built-in, and nothing else changes."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", text, flags=re.S)
    if not m:
        raise SkillError(f"{path}: it starts with a front matter")
    lines = []
    for line in m.group(1).splitlines():
        f = re.match(r"^(\s*)(name|description):\s+(.*?)\s*$", line)
        lines.append(f"{f.group(1)}{f.group(2)}: {json.dumps(f.group(3))}"
                     if f and not f.group(3).startswith(("'", '"')) else line)
    try:
        front = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as e:
        raise SkillError(f"{path}: the front matter is not YAML ({str(e).splitlines()[0]})") from None
    if not isinstance(front, dict):
        raise SkillError(f"{path}: the front matter is not a mapping")
    body = re.sub(r"\A\s*<!--.*?-->\s*", "", text[m.end():], flags=re.S).strip()
    name, desc = check(front, body)
    if name != path.parent.name:
        raise SkillError(f"{path}: its name {name!r} is not its folder's")
    meta = front.get("metadata")
    return Skill(name, desc, body, True,
                 {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {})


def default_dir() -> Path:
    """The user's skills when the config names no "skills_dir": the data folder setup uses (the settings' data_dir,
    else its default) - setup writes "skills_dir" into each model's config, so this is for an older config."""
    from serve import places
    try:
        data = json.loads(places.settings_path().read_text(encoding="utf-8")).get("data_dir")
    except (OSError, ValueError):
        data = None
    if not data:
        root = Path(__file__).resolve().parents[1]
        packaged = not os.access(root, os.W_OK)
        data = places.user_dir("data") if packaged else root.parent / "XeStrata-data"
    return Path(data) / "skills"


class Skills:
    """XeStrata's skills and the user's: read from their folders when they change (each SKILL.md's time), the user's
    added, changed, deleted and switched on or off here."""

    def __init__(self, user_dir: Path | str | None, builtin_dir: Path = BUILTIN):
        self.user_dir = Path(user_dir) if user_dir else None
        self.builtin_dir = builtin_dir
        self.lock = threading.Lock()
        self._cache: dict[Path, tuple[float, Skill | None, str]] = {}

    # ---- reading
    def _scan(self, root: Path | None, builtin: bool) -> list[tuple[Skill | None, str, Path]]:
        if root is None or not root.is_dir():
            return []
        out = []
        for d in sorted(root.iterdir()):
            f = d / "SKILL.md"
            if not d.is_dir() or d.name.startswith(".") or not f.is_file():
                continue
            try:
                t = f.stat().st_mtime
            except OSError:
                continue
            got = self._cache.get(f)
            if got is None or got[0] != t:
                try:
                    got = (t, _read(f, builtin), "")
                except (OSError, UnicodeError, SkillError) as e:
                    got = (t, None, str(e))
                self._cache[f] = got
            out.append((got[1], got[2], f))
        return out

    def all(self) -> list[Skill]:
        """Every skill that reads, XeStrata's first; a user's with a built-in skill's name is left out."""
        with self.lock:
            own = [s for s, _, _ in self._scan(self.builtin_dir, True) if s is not None]
            names = {s.name for s in own}
            user = [s for s, _, _ in self._scan(self.user_dir, False) if s is not None and s.name not in names]
        return own + user

    def external_all(self) -> list[Skill]:
        """Every built-in skill on disk, for the external endpoint only.  Unlike all(), this includes one whose
        front matter's unquoted description has ': ' in it; the chat's list and _cache are left alone."""
        if not self.builtin_dir.is_dir():
            return []
        out = []
        for d in sorted(self.builtin_dir.iterdir()):
            f = d / "SKILL.md"
            if not d.is_dir() or d.name.startswith(".") or not f.is_file():
                continue
            try:
                out.append(_read_external(f))
            except (OSError, UnicodeError, SkillError):
                continue
        return out

    def external_get(self, name: str) -> Skill | None:
        return next((s for s in self.external_all() if s.name == name), None)

    def broken(self) -> list[dict]:
        """The user's SKILL.md files that do not read, and why (the web page's list shows them)."""
        with self.lock:
            return [{"name": f.parent.name, "error": err} for s, err, f in self._scan(self.user_dir, False)
                    if s is None]

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.all() if s.name == name), None)

    def off(self) -> set[str]:
        if self.user_dir is None:
            return set()
        try:
            got = json.loads((self.user_dir / STATE).read_text(encoding="utf-8")).get("off")
        except (OSError, ValueError, AttributeError):
            return set()
        return {n for n in got if isinstance(n, str)} if isinstance(got, list) else set()

    def on(self, name: str) -> bool:
        return name not in self.off()

    def enabled(self, builtin: bool | None = None) -> list[Skill]:
        off = self.off()
        return [s for s in self.all() if s.name not in off and (builtin is None or s.builtin == builtin)]

    def body(self, name: str) -> str:
        """A built-in skill's body, "" when there is none or it is switched off (the image tools read theirs here)."""
        s = self.get(name)
        return s.body if s is not None and s.builtin and self.on(name) else ""

    def listing(self) -> dict:
        """GET /skills: every skill, whether it is on and whether it is XeStrata's, and the user's that do not read."""
        off = self.off()
        return {"dir": str(self.user_dir) if self.user_dir else None,
                "skills": [{"name": s.name, "description": s.description, "builtin": s.builtin,
                            "on": s.name not in off, "kind": s.kind, "chars": len(s.body)} for s in self.all()],
                "broken": self.broken()}

    def text(self, name: str) -> str:
        """A user's SKILL.md as it is on the disk (the web page edits it); SkillError for one that is not the user's."""
        s = self.get(name)
        if s is None or s.builtin or self.user_dir is None:
            raise SkillError(f"{name}: not one of the skills added here")
        return (self.user_dir / name / "SKILL.md").read_text(encoding="utf-8")

    # ---- writing (the user's skills; a built-in skill is switched off, never changed)
    def _need_dir(self) -> Path:
        if self.user_dir is None:
            raise SkillError("this server keeps no skills of its own (the config names no skills_dir)")
        self.user_dir.mkdir(parents=True, exist_ok=True)
        return self.user_dir

    def save(self, text: str, old: str | None = None) -> Skill:
        """A user's SKILL.md written (a new skill, or `old` changed; a new name renames it) -> the skill."""
        if len(text.encode("utf-8")) > MAX_BODY:
            raise SkillError(f"SKILL.md: at most {MAX_BODY // 1024} KiB")
        front, body = parse(text)
        name, _ = check(front, body)
        root = self._need_dir()
        with self.lock:
            builtin = {s.name for s, _, _ in self._scan(self.builtin_dir, True) if s is not None}
            if name in builtin:
                raise SkillError(f"{name}: one of XeStrata's skills has that name; give yours another")
            if old is not None and old != name and not (root / old / "SKILL.md").is_file():
                raise SkillError(f"{old}: not one of the skills added here")
            if (old is None or old != name) and (root / name).exists():
                raise SkillError(f"{name}: a skill of that name is here already")
            d = root / name
            if old is not None and old != name:
                (root / old).rename(d)
            d.mkdir(exist_ok=True)
            tmp = d / "SKILL.md.tmp"
            tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            os.replace(tmp, d / "SKILL.md")
            self._cache.pop(d / "SKILL.md", None)
        if old is not None and old != name and old in self.off():   # a renamed skill keeps its switch
            self._set_off(old, False)
            self._set_off(name, True)
        return _read(d / "SKILL.md", False)

    def delete(self, name: str) -> None:
        s = self.get(name)
        if s is None or s.builtin or self.user_dir is None:
            raise SkillError(f"{name}: not one of the skills added here (XeStrata's can be switched off only)")
        d = self.user_dir / name
        with self.lock:
            for f in sorted(d.rglob("*"), reverse=True):
                f.unlink() if f.is_file() or f.is_symlink() else f.rmdir()
            d.rmdir()
            self._cache.pop(d / "SKILL.md", None)
        self._set_off(name, False)

    def switch(self, name: str, on: bool) -> None:
        if self.get(name) is None:
            raise SkillError(f"{name}: there is no such skill")
        self._set_off(name, not on)

    def _set_off(self, name: str, off: bool) -> None:
        root = self._need_dir()
        with self.lock:
            names = self.off()
            (names.add if off else names.discard)(name)
            tmp = root / (STATE + ".tmp")
            tmp.write_text(json.dumps({"off": sorted(names)}), encoding="utf-8")
            os.replace(tmp, root / STATE)

    def import_file(self, data: bytes, filename: str = "SKILL.md") -> Skill:
        """A SKILL.md, or a zip of a skill's folder (its SKILL.md taken; the rest is not read by load_skill), added."""
        if filename.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
            try:
                z = zipfile.ZipFile(io.BytesIO(data))
                found = sorted((n for n in z.namelist() if n.rsplit("/", 1)[-1] == "SKILL.md"), key=len)
                if not found:
                    raise SkillError("the zip holds no SKILL.md")
                info = z.getinfo(found[0])
                if info.file_size > MAX_BODY:
                    raise SkillError(f"SKILL.md: at most {MAX_BODY // 1024} KiB")
                data = z.read(info)
            except zipfile.BadZipFile:
                raise SkillError("the file is not a zip that can be read") from None
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise SkillError("SKILL.md: UTF-8 text") from None
        return self.save(text)


def wrap_user(s: Skill) -> str:
    """A user's skill as load_skill returns it: marked as the user's, not XeStrata's."""
    return (f"<skill name=\"{s.name}\" source=\"added by this server's user, not XeStrata\">\n{s.body}\n</skill>")


# ---------------------------------------------------------------------------------- skills for an external caller
# GET /v1/skills serves the built-in skills through one conversion that lives here, in the shared skills module: no
# skill name or model name is special-cased.  The file on disk stays the chat's.  Every rule below is a pattern on the
# text, applied once, in a fixed order; the setup's model ids and the request's base URL are data passed in by serve.

_PLACEHOLDER = re.compile(r"<this setup's [^<>]*?model>", re.I)
# A1: a chat's attachment notation, and the sentence that describes it.
_ATTACHED = re.compile(r"\battached:(\d+)\b")
_ATTACHED_SAY = re.compile(r"The user's pictures are `attached:1`, `attached:2`, \.\.\. in attachment order")
# A2: the chat's implicit extra pictures, the chat's saved mask link.
_EXTRA_PICS = re.compile(r"Without them the user's other attached pictures go in, except the control image\.")
_MASK_LINK = re.compile(r"the marked picture and the mask's link")
# A4: the chat's plan / step / group / use_text_gpu / stub call arguments and same-reply calls.
_PLAN_SAY = re.compile(r"two image or video steps or more get a chart of them as the first call's `plan` "
                       r"\(combine-tools has more\)")
_PLAN_IDS = re.compile(r"each call gives its box's id as `step`, and the page colors the box as it runs")
_GROUP = re.compile(r"The calls of one reply run at once\. Calls whose results are joined later share a `group`: "
                    r"one failing stops the rest of it\. `use_text_gpu` true lends them the chat model's GPU: only "
                    r"when the user wants it fast, or two heavy steps of different image models stand side by side\.")
_STUB = re.compile(r"A tool given as one line: call it with the arguments its line names; its first call "
                   r"answers with the rest\.")
_REPLY = re.compile(r"The calls of one reply run at once: put together the steps that need no other step's result\.")
_TWO_REPLIES = re.compile(r"\bin two replies\b")
_CHAIN = re.compile(r"Each step takes the link the step before it returned as its `image`\.")
# A5/A6: a download message, the Video tab and the chat's training panel.
_DOWNLOAD = re.compile(r"A result that says it is downloading: tell the user, and call it again later, in a new "
                       r"message\.")
_VIDEO_UP = re.compile(r"tell the user it runs, that they follow it in the Video tab and where the video will be; "
                       r"do not call again to check\.")
_VIDEO_SEG = re.compile(r"A job, as upscale_video's: tell the user, do not call again to check\. To pick one object "
                        r"by pointing, or fix one missed: the Video tab's Find objects\.")
_PANEL = re.compile(r"The training panel opens in the chat; the user does the work there\.")
_PANEL_HELP = re.compile(r"Tell the user what the panel does, briefly:")
_PANEL_WAIT = re.compile(r"Do not wait for it and do not call train_lora again for the same request\.")
_PANEL_PICK = re.compile(r"The user picks one and registers it as a LoRA")
_SETUP = re.compile(r"\bthis setup\b(?!\s*\()")


def external_models(images: Any) -> dict:
    """The setup's image model ids in config order, grouped by family: the facts A3 resolves a placeholder with."""
    gens = list(images.gens.values()) if images is not None else []
    families: dict[str, list[str]] = {}
    for g in gens:
        family = str(getattr(g, "family", None) or g.cfg.get("family") or "qwen-image")
        families.setdefault(family, []).append(g.model)
    return {"ids": [g.model for g in gens], "default": gens[0].model if gens else "", "families": families}


def _family_ids(ctx: dict, asked: str) -> list[str]:
    """The ids of the family a placeholder names: `qwen-image-2.1` is the `qwen-image` family.  No id is guessed."""
    keys = []
    for f in ctx["families"]:
        if f and (asked == f or asked.startswith(f + "-") or f.startswith(asked + "-")):
            keys.append(f)
    return ctx["families"][max(keys, key=len)] if keys else []


def _contract(ctx: dict, base: str) -> str:
    """The one contract block every external skill carries: the setup's real ids and the A1-A6 REST wording."""
    ids = ctx["ids"]
    qwen = _family_ids(ctx, "qwen-image-2.1")
    if ids:
        models = "Image models in this setup: " + ", ".join(f"`{i}`" for i in ids)
        models += f"; default `{ctx['default']}`." if ctx["default"] else "."
    else:
        models = "This setup has no image models."
    if qwen:
        models += " Qwen-Image-2.1 ids here: " + ", ".join(
            f"`{i}`" + (" (default of this family)" if n == 0 else "") for n, i in enumerate(qwen)) + "."
    else:
        models += " This setup has no Qwen-Image-2.1 model; do not guess a name."
    return "\n\n".join([
        "## External REST API",
        f"These skills are served from {base} by GET /v1/skills. Use this REST contract, not the chat's call rules.",
        models,
        "- Input images: send the bytes as multipart `image` (`image[]` for several) or as JSON `image` / `images` "
        "with a `data:` or `http(s)` URL. A relative link `/v1/images/files/<name>` is resolved against " + base
        + "; fetch it and send the bytes, or pass a link you already have as a URL. There is no general "
        "`POST /v1/images/files` upload (that POST answers 404): get a link from a `response_format=url` response of "
        "`/v1/images/edits`, `/v1/images/upscale`, `/v1/images/segmentations`, `/v1/images/remove-background`, "
        "`/v1/images/controls` or `/v1/videos/files`.",
        "- Result links: a response's `url` is absolute; a chat tool's `/v1/images/files/...` is relative. Copy "
        "links exactly, never invent one. Only the newest files are kept, so an old link can 404. Masks and control "
        "maps are saved in the same place. `POST /v1/images/segmentations` returns a mask inline (`data[].mask`: "
        "base64 PNG with `mask_format=png`, COCO RLE with `mask_format=rle`), not a link.",
        "- Requests with no dependency may be sent in parallel. When one depends on another, use the first "
        "response's link (as fetched bytes) for the next request; the server has no `group` and no failure cascade. "
        "A tool's definition is `GET /v1/skills/<name>`; keep at most two retries on a bad result and change the "
        "prompt or region, not only `strength`. The chat's two-stage stub call is chat only.",
        "- A 503 with `code: weights_downloading` is not success: wait `Retry-After` seconds (default 30) and "
        "resend the same request. A video job returns an id: poll `GET /v1/videos/<id>` for `status` / `progress`, "
        "then `GET /v1/videos/<id>/content` (or `?variant=overlay|mask|object_<id>`); do not re-POST the job.",
    ])


def externalize(text: str, ctx: dict, base: str) -> str:
    """One SKILL.md -> its external form: the front matter kept, each A1-A6 rule applied once, A7 left alone."""
    m = re.match(r"\A(\ufeff?---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z))(.*)\Z", text, flags=re.S)
    head, body = (m.group(1), m.group(2)) if m else ("", text)

    def placeholder(match: re.Match) -> str:
        asked = match.group(0)[1:-1].rsplit(" model", 1)[0].replace("this setup's", "").strip().lower()
        ids = _family_ids(ctx, asked)
        if not ids:
            return "no such model in this setup" + ("; image models: " + ", ".join(ctx["ids"]) if ctx["ids"] else "")
        return ids[0]   # the family's id in config order, so the example stays a valid JSON string

    body = _PLACEHOLDER.sub(placeholder, body)
    rules = (
        (_ATTACHED_SAY, "Input pictures are the links you pass, in order"),
        (_ATTACHED, r"<input image \1 link>"),
        (_EXTRA_PICS, "Every input picture is explicit: pass the extras in `image[]` / `images`."),
        (_MASK_LINK, "the marked picture and the mask (a link in the chat, `data[].mask` inline in a REST "
                     "`/v1/images/segmentations` response)"),
        (_PLAN_SAY, "a chart of them drawn for the plan"),
        (_PLAN_IDS, "the ids are for your own tracking"),
        (_GROUP, "Requests with no dependency may be sent in parallel; when one depends on another, use the first "
                 "response's link (as bytes) for the next request. There is no server-side `group` or failure "
                 "cascade, and `use_text_gpu` is not an API argument."),
        (_STUB, "A tool's definition is `GET /v1/skills/<name>` and its arguments are the REST contract."),
        (_REPLY, "Requests with no dependency may be sent in parallel."),
        (_TWO_REPLIES, "in two consecutive requests"),
        (_CHAIN, "Each step takes the link the step before it returned as its `image` (fetch a relative link and "
                 "send the bytes)."),
        (_DOWNLOAD, "A 503 with `code: weights_downloading` is not success: wait `Retry-After` seconds (default 30) "
                    "and resend the same request."),
        (_VIDEO_UP, "POST returns a job id; poll `GET /v1/videos/<id>` for `status` / `progress`, then "
                    "`GET /v1/videos/<id>/content` for the file. Do not re-POST the job."),
        (_VIDEO_SEG, "A job, as upscale_video's: poll `GET /v1/videos/<id>` and use `GET .../content` with "
                     "`?variant=overlay|mask|object_<id>`; do not re-POST. Pick an object by pointing, or fix one "
                     "missed, with the segmentation job's `prompts`."),
        (_PANEL, "The chat's training panel runs on the server's LoRA training API: `POST /v1/lora/jobs` and the "
                 "actions under `/v1/lora/jobs/<id>`."),
        (_PANEL_HELP, "The panel's steps, each a REST request (`POST /v1/lora/jobs`: a multipart form with `images` "
                      "and the fields `name`, `kind`, `trigger`, `model`; `POST .../caption` writes the captions and "
                      "`PUT .../captions` edits them; `POST .../start` takes `steps` (n) and `save_every` (k)):"),
        (_PANEL_WAIT, "Do not wait for it and do not re-POST: a second `POST /v1/lora/jobs` makes a second job, and "
                      "`.../start` refuses a job that is queued or running. Read its `status` and `progress` with "
                      "`GET /v1/lora/jobs/<id>`."),
        (_PANEL_PICK, "Pick one and register it with `POST /v1/lora/jobs/<id>/register` (JSON `step` and `name`; "
                      "without `step` it takes the last version)"),
        (_SETUP, f"this setup ({base})"),
    )
    for pattern, replacement in rules:
        body = pattern.sub(replacement, body)
    return head + body.rstrip() + "\n\n" + _contract(ctx, base) + "\n"


class SkillTools:
    """load_skill for a chat request (a hub run_with_mcp runs: template_tools, routes, call, settings): its description
    the image tools' part when they are offered (their rules, what this setup can do, their skills' list), the built-in
    skills of the other tools offered (`tools`: their names, as the translation tool's skill), and the user's skills
    switched on when the request asks for them (`user`); it reads the named skills, several at once."""

    def __init__(self, skills: Skills, images=None, user: bool = False, tools: tuple = ()):
        self.skills, self.images, self.user, self.tools = skills, images, user, tuple(tools)
        self.settings = {"max_rounds": 4}

    def _index(self) -> dict[str, tuple[str, str]]:
        """{name: (kind, the line that says when to read it)} of the skills this request may read."""
        out: dict[str, tuple[str, str]] = {}
        if self.images is not None:
            for name, kind, when in self.images.skill_index():
                out[name] = (kind, when)
        for s in self.skills.enabled(builtin=True):     # the other tools' own skills
            if s.kind == "tool" and s.meta.get("tool") in self.tools:
                out.setdefault(s.name, ("tool", s.description))
        if self.user:
            for s in self.skills.enabled(builtin=False):
                out.setdefault(s.name, ("user", s.description))
        return out

    def template_tools(self, exclude=()) -> list[dict]:
        index = self._index()
        if not index or TOOL in exclude:
            return []
        head = self.images.skill_part() if self.images is not None else ""
        lines = "\n".join(f"- {name}: {when}" for name, (_, when) in index.items())
        text = ("Read the skills a request needs (several at once) before acting on it."
                + (f"\n\n{head}" if head else "") + f"\n\nSkills:\n{lines}")
        return [{"name": TOOL, "description": text,
                 "parameters": {"type": "object", "properties": {
                     "names": {"type": "array", "items": {"type": "string", "enum": list(index)}}},
                     "required": ["names"]}}]

    def routes(self) -> dict:
        return {TOOL: (None, TOOL)} if self._index() else {}

    def _other(self, name: str) -> bool:
        """A skill of a tool that is not the image tools'."""
        s = self.skills.get(name)
        return s is not None and s.builtin and s.meta.get("tool") in self.tools

    def call(self, name: str, arguments, cancel=None) -> dict:
        t0 = time.monotonic()
        index = self._index()
        names = arguments.get("names") if isinstance(arguments, dict) else None
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names or any(n not in index for n in names):
            text = f"error: names: some of {', '.join(index)}"
            return {"ok": False, "text": text, "chars": len(text), "truncated": False,
                    "ms": round((time.monotonic() - t0) * 1000), "server": None, "tool": TOOL, "images": []}
        picked = [n for kind in KINDS for n in dict.fromkeys(names) if index[n][0] == kind]
        # a tool's skill read: the tool itself declared from the next round on (ImageTools.unlock)
        given = []
        for n in picked:
            s = self.skills.get(n) if index[n][0] == "tool" else None
            tool = s.meta.get("tool") if s is not None else None
            if tool and tool not in self.tools and self.images is not None and self.images.unlock(tool):
                given.append(tool)
        parts = []
        for n in picked:
            if index[n][0] == "user":
                s = self.skills.get(n)
                parts.append(wrap_user(s) if s is not None else f"{n}: no longer here")
            elif (text := self.skills.body(n) if self.images is None or self._other(n) else self.images.skill_body(n)):
                parts.append(text if text.startswith("# ") else f"# {n}\n\n{text}")
        body = "\n\n".join(p for p in parts if p)
        if len(picked) > 1:
            body = "Where these differ, the later one wins.\n\n" + body
        if given:
            body += f"\n\nYou can call {', '.join(given)} from now on."
        return {"ok": True, "text": body, "chars": len(body), "truncated": False,
                "ms": round((time.monotonic() - t0) * 1000), "server": None, "tool": TOOL, "images": []}
