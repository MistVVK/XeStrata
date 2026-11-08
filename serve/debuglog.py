# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/debuglog.py - the debug chat log ("debug_chat_log": true, --debug-chat-log; off by default).

For looking into a conversation afterwards, the server writes what it saw and did into one folder
(places.user_dir("state") / "debug-chat-log"):

- requests-YYYY-MM-DD.jsonl: one line a chat request - its body, the prompt the engine read (each picture's image
  tokens as one marker with its key and count), the server's own tool rounds (the calls and the results the model
  read), the answer (reasoning, text, calls), usage and timings, an error.
- img/<sha256>.<ext>: the pictures, once each: the ones a request sent (its body names the file instead of the data
  URL) and the ones a tool made.
- chats/<chat id>.json: the web page's own record of a chat, as the browser keeps it (POST /v1/xestrata/debug/chat).

When the folder grows past its limit ("debug_chat_log_mib", 1024 by default) the oldest days' lines go first, then the
pictures no line or chat names.  A write that fails is said in the server's log once and the request goes on."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from pathlib import Path

DATA_URL = re.compile(r"^data:image/([a-z0-9.+-]+);base64,", re.I)
EXT = {"jpeg": "jpg", "svg+xml": "svg"}
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class DebugLog:
    def __init__(self, folder: Path, limit_mib: float):
        self.folder, self.limit = Path(folder), int(max(1.0, limit_mib) * (1 << 20))
        self.lock = threading.Lock()
        self.warned = False

    def _say(self, e: OSError) -> None:
        if not self.warned:
            self.warned = True
            print(f"[xestrata] debug chat log: could not write to {self.folder} ({e.strerror or e}); the requests go "
                  "on", flush=True)

    def picture(self, data: bytes, ext: str) -> str:
        """The picture saved once by its hash: its path in the folder ("img/<sha256>.<ext>")."""
        name = f"img/{hashlib.sha256(data).hexdigest()}.{EXT.get(ext.lower(), ext.lower())}"
        path = self.folder / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return name

    def scrub(self, value):
        """`value` with each picture's data URL saved and replaced by {"picture": "img/..."}."""
        if isinstance(value, str):
            m = DATA_URL.match(value)
            if m:
                try:
                    return {"picture": self.picture(base64.b64decode(value[m.end():], validate=False), m.group(1))}
                except (ValueError, OSError):
                    return {"picture": None, "bytes": len(value)}
            return value
        if isinstance(value, dict):
            return {k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        return value

    def request(self, record: dict) -> None:
        """One request's line, then the folder kept under its limit."""
        try:
            with self.lock:
                line = json.dumps(self.scrub(record), ensure_ascii=False, default=str)
                self.folder.mkdir(parents=True, exist_ok=True)
                with open(self.folder / f"requests-{time.strftime('%Y-%m-%d')}.jsonl", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self._trim()
        except OSError as e:
            self._say(e)

    def chat(self, chat_id: str, record) -> bool:
        """The web page's record of a chat, in place of the one before.  False: not a chat id."""
        if not CHAT_ID.match(chat_id or ""):
            return False
        try:
            with self.lock:
                data = json.dumps(self.scrub(record), ensure_ascii=False, default=str)
                path = self.folder / "chats" / f"{chat_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(data, encoding="utf-8")
                tmp.replace(path)
                self._trim()
        except OSError as e:
            self._say(e)
        return True

    def _trim(self) -> None:
        files = [p for p in self.folder.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= self.limit:
            return
        days = sorted(p for p in files if p.parent == self.folder and p.name.startswith("requests-"))
        chats = sorted((p for p in files if p.parent.name == "chats"), key=lambda p: p.stat().st_mtime)
        for p in days[:-1] + chats[:-1]:              # the oldest first; today's lines and the newest chat stay
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            if total <= self.limit:
                break
        named = set()
        for p in (q for q in self.folder.rglob("*") if q.is_file() and q.suffix in (".jsonl", ".json")):
            named |= set(re.findall(r"img/[0-9a-f]{64}\.[a-z0-9.+-]+", p.read_text(encoding="utf-8", errors="replace")))
        for p in (self.folder / "img").glob("*") if (self.folder / "img").is_dir() else ():
            if f"img/{p.name}" not in named:
                p.unlink(missing_ok=True)
