# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/lora_caption.py - a LoRA's training images captioned by the chat's model: each image beside its
caption, `<image stem>.txt`, one line `<trigger>, <caption>`, as tools/image/mf_pre.py reads it (docs/LORA_TRAIN.md
"The caption policy").

    python tools/image/lora_caption.py --dir DIR --kind character --trigger WORD [--keep "teal bob hair, freckles"]
        [--style booru|natural] [--style-words "ukiyo-e, woodblock"] [--model-skill anima|magetrail]
        [--server http://127.0.0.1:8095] [--overwrite]

The server is XeStrata's (serve), its chat model one that reads images (the config's "vision").  Each image goes to
its /v1/chat/completions with the lora-caption skill (serve/skills/lora-caption: a character LoRA's caption writes
everything but the character - its clothes, the place, the picture's style - so the style is learned with those words
and not with the trigger; a style LoRA's the other way round) and the prompt style of the model to be trained
(serve/skills: booru's Danbooru tags for Anima and MageTrail, natural's sentences for the others; --model-skill adds
the model's own rules).  `--keep` names the character's own look, the part the caption leaves to the trigger.  A
caption already there is kept unless --overwrite.  XESTRATA_API_KEY is sent where the server asks for a key.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve import skills as skills_api  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")   # mf_pre.py's QI_EXTS
SIDE = 1024          # the longest side sent (the server reads a picture smaller still)
TIMEOUT_S = 600      # one image's answer, the model's first load included


def skill(name: str) -> str:
    """A built-in skill's body (serve/skills/<name>/SKILL.md)."""
    _, body = skills_api.parse((skills_api.BUILTIN / name / "SKILL.md").read_text(encoding="utf-8"))
    return body


def picture(path: Path) -> str:
    """The image as a data URL, at most SIDE pixels a side (JPEG), turned by its EXIF."""
    try:
        from PIL import Image
        from PIL.ImageOps import exif_transpose
    except ImportError:
        return f"data:image/{path.suffix.lstrip('.').lower()};base64," + base64.b64encode(path.read_bytes()).decode()
    with Image.open(path) as im:
        im = exif_transpose(im).convert("RGB")
        im.thumbnail((SIDE, SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tidy(text: str, trigger: str, kind: str, style_words: str = "") -> str:
    """The model's answer as one caption line: its thinking and fences out, the first line that says something, the
    trigger first."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = [ln.strip().strip("`").strip() for ln in text.splitlines()]
    line = next((ln for ln in lines if ln and not ln.startswith("```")), "")
    line = line.strip('"').strip()
    if not line.lower().startswith(trigger.lower()):
        line = f"{trigger}, {line}" if line else trigger
    front, _ = skills_api.parse((skills_api.BUILTIN / "lora-caption" / "SKILL.md").read_text(encoding="utf-8"))
    prefix, content = line[:len(trigger)], line[len(trigger):]
    for pattern in front["metadata"]["reading_exclude"]:
        content = re.sub(pattern, "", content, flags=re.I)
    if kind == "style":
        patterns = front["metadata"]["style_exclude"] + [
            r"(?:in (?:the )?style of )?" + re.escape(word.strip()) + r"(?: style)?"
            for word in style_words.split(",") if word.strip()]
        # Only remove style-only clauses at the end, not an anime poster or a watercolor in the scene.
        while match := re.search(r",\s*([^,]+?)\s*$", content):
            clause = match.group(1).strip().rstrip(".!?").strip()
            if not any(re.fullmatch(pattern, clause, flags=re.I) for pattern in patterns):
                break
            content = content[:match.start()]
    content = re.sub(r"\b(?:reading|reads|saying)\s*(?=,|[.!?]|$)", "", content, flags=re.I)
    content = re.sub(r"\b(?:and|with)\s*,", ",", content, flags=re.I)
    content = re.sub(r"\ba showing\b", "showing", content, flags=re.I)
    content = re.sub(r"\bin a with\b", "with", content, flags=re.I)
    content = re.sub(r"(?:,\s*)?\b(?:with|and)\s*([.!?]?)$", r"\1", content, flags=re.I)
    content = re.sub(r"\s+,", ",", content)
    content = re.sub(r",(?:\s*,)+", ",", content)
    content = re.sub(r",\s*([.!?]|$)", r"\1", content)
    line = prefix + content
    return " ".join(line.split())


def ask(server: str, system: str, user: str, image: str) -> str:
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [{"type": "text", "text": user},
                                                      {"type": "image_url", "image_url": {"url": image}}]}],
            # a caption is a line: a model that thinks first (Qwen3.8) spent the tokens on it and wrote none
            "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("XESTRATA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        out = json.load(r)
    return out["choices"][0]["message"].get("content") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="the training images' folder")
    ap.add_argument("--kind", required=True, choices=("character", "style", "other"))
    ap.add_argument("--trigger", required=True, help="the trigger word every caption starts with")
    ap.add_argument("--keep", default="", help="what the LoRA must learn (a character's own look), left to the trigger")
    ap.add_argument("--style-words", default="", help="comma-separated names and features of the style to learn")
    ap.add_argument("--style", default="natural", choices=("natural", "booru"),
                    help="the trained model's prompt style: booru for Anima and MageTrail, natural for the others")
    ap.add_argument("--model-skill", default="", help="the trained model's own skill (anima, magetrail), added")
    ap.add_argument("--server", default="http://127.0.0.1:8095")
    ap.add_argument("--overwrite", action="store_true", help="caption again an image that has its .txt")
    a = ap.parse_args()

    # the prompt style and the model's rules first, the caption's last: where they differ (a model's quality tags),
    # the caption's win
    system = "\n\n".join(p for p in (
        "Prompt style:\n" + skill(a.style), skill(a.model_skill) if a.model_skill else "",
        "Where the rules above differ from these, these win.\n\n" + skill("lora-caption")) if p)
    user = f"Kind: {a.kind}. Trigger word: {a.trigger}."
    if a.keep:
        user += f" What the LoRA must learn (leave it out): {a.keep}."
    if a.kind == "style" and a.style_words:
        user += f" Style names and features to learn (leave them out): {a.style_words}."
    files = sorted(p for p in Path(a.dir).iterdir() if p.suffix.lower() in EXTS and p.is_file())
    if not files:
        print(f"lora_caption: no image in {a.dir}", file=sys.stderr)
        return 1
    done = kept = 0
    for f in files:
        cap = f.with_suffix(".txt")
        if cap.exists() and not a.overwrite:
            kept += 1
            continue
        try:
            sidecar = f.with_suffix(".json")
            note = json.loads(sidecar.read_text(encoding="utf-8")).get("note", "") if sidecar.exists() else ""
            image_user = user + (f" Image notes (context only; check against the picture): {note}" if note else "")
            line = tidy(ask(a.server, system, image_user, picture(f)), a.trigger, a.kind, a.style_words)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            print(f"lora_caption: {f.name}: {e}", file=sys.stderr)
            return 1
        cap.write_text(line + "\n", encoding="utf-8")
        print(f"{f.name}: {line}")
        done += 1
    print(f"lora_caption: {done} captioned, {kept} kept as they were")
    return 0


if __name__ == "__main__":
    sys.exit(main())
