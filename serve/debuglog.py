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

The days before today's and the chats not saved for a day are packed with xz (requests-YYYY-MM-DD.jsonl.xz,
chats/<chat id>.json.xz) by a thread of their own, at the start and when a request is the first of a new day: the
packed file is read back and checked before the plain one goes, and a chat saved again is written plain and its .xz
removed.  The pictures stay as they are (their formats are packed already).

When the folder grows past its limit ("debug_chat_log_mib", 1024 by default, counted on the files as they are on the
disk, the packed ones packed) the oldest days' lines go first, then the oldest chats, then the pictures no line or chat
names.  A write or a packing that fails is said in the server's log once and the request goes on; a file that could not
be packed stays plain."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
import os
import re
import threading
import time
from pathlib import Path

DATA_URL = re.compile(r"^data:image/([a-z0-9.+-]+);base64,", re.I)
EXT = {"jpeg": "jpg", "svg+xml": "svg"}
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
DAY = re.compile(r"^requests-(\d{4}-\d{2}-\d{2})\.jsonl(\.xz)?$")
CHUNK = 1 << 20
CHAT_IDLE_S = 86400          # a chat not saved for this long is packed


def _read_chunks(f):
    while True:
        b = f.read(CHUNK)
        if not b:
            return
        yield b


def _text(path: Path) -> str:
    """A log file's text, a packed one unpacked."""
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


class DebugLog:
    def __init__(self, folder: Path, limit_mib: float):
        self.folder, self.limit = Path(folder), int(max(1.0, limit_mib) * (1 << 20))
        self.lock = threading.Lock()
        self.said: set[str] = set()
        self.day = time.strftime("%Y-%m-%d")
        self.packing = self.again = False

    def _say(self, e: Exception, what: str = "write to") -> None:
        if what not in self.said:
            self.said.add(what)
            print(f"[xestrata] debug chat log: could not {what} {self.folder} ({getattr(e, 'strerror', None) or e}); "
                  "the requests go on", flush=True)

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
                day = time.strftime("%Y-%m-%d")
                with open(self.folder / f"requests-{day}.jsonl", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                new_day, self.day = day != self.day, day
                if not (new_day or self.packing):   # else trimmed after the packing, on the packed sizes
                    self._trim()
        except OSError as e:
            self._say(e)
            return
        if new_day:
            self.pack_soon()

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
                path.with_name(path.name + ".xz").unlink(missing_ok=True)   # only the newest record stays
                if not self.packing:
                    self._trim()
        except OSError as e:
            self._say(e)
        return True

    def pack_soon(self) -> None:
        """pack() on a thread of its own; a call while it runs makes it go round once more."""
        with self.lock:
            if self.packing:
                self.again = True
                return
            self.packing = True
        threading.Thread(target=self._pack_loop, name="debug-chat-log-xz", daemon=True).start()

    def _pack_loop(self) -> None:
        while True:
            try:
                self.pack()
            except Exception as e:                  # noqa: BLE001 - never takes the server down; said once
                self._say(e, "pack the files in")
            with self.lock:
                if not self.again:
                    self.packing = False
                    return
                self.again = False

    def pack(self) -> None:
        """The days before today's and the chats not saved for a day, packed with xz, then the folder kept under its
        limit.  The packing is done outside the lock; only the swap of the files is inside it."""
        today = time.strftime("%Y-%m-%d")
        todo = []
        if self.folder.is_dir():
            todo += [p for p in self.folder.glob("requests-*.jsonl")
                     if (m := DAY.match(p.name)) and m.group(1) < today]
        chats = self.folder / "chats"
        if chats.is_dir():
            old = time.time() - CHAT_IDLE_S
            todo += [p for p in chats.glob("*.json") if p.stat().st_mtime < old]
        for p in sorted(todo):
            try:
                self._pack_one(p, append=p.parent == self.folder)
            except (OSError, lzma.LZMAError, EOFError) as e:
                self._say(e, "pack the files in")
        try:
            with self.lock:
                self._trim()
        except OSError as e:
            self._say(e)

    def _pack_one(self, src: Path, append: bool) -> None:
        """`src` into `src`.xz, read back and checked, then `src` removed.  append: after the streams `src`.xz holds
        already (a day's lines written after it was packed); else in place of it (a chat's newer record)."""
        dst = src.with_name(src.name + ".xz")
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            st = src.stat()
            append = append and dst.exists()
            old = hashlib.sha256()
            if append:
                with lzma.open(dst, "rb") as f:
                    for b in _read_chunks(f):
                        old.update(b)
            want, mine = old.copy(), hashlib.sha256()
            comp = lzma.LZMACompressor()
            with open(src, "rb") as f, open(tmp, "wb") as out:
                if append:
                    with open(dst, "rb") as prev:
                        for b in _read_chunks(prev):
                            out.write(b)
                for b in _read_chunks(f):
                    want.update(b)
                    mine.update(b)
                    out.write(comp.compress(b))
                out.write(comp.flush())
            if append and old.digest() == mine.digest():
                tmp.unlink()                        # packed before, its plain left behind: keep the .xz as it is
            else:
                got = hashlib.sha256()
                with lzma.open(tmp, "rb") as f:
                    for b in _read_chunks(f):
                        got.update(b)
                if got.digest() != want.digest():
                    raise lzma.LZMAError(f"{tmp.name} does not read back as {src.name}")
            with self.lock:
                now = src.stat()
                if (now.st_size, now.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
                    return                          # written again meanwhile: packed on a later round
                if tmp.exists():
                    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))  # the oldest go first in _trim
                    tmp.replace(dst)
                src.unlink()
        except FileNotFoundError:
            pass                                    # trimmed or saved again meanwhile
        finally:
            tmp.unlink(missing_ok=True)

    def _trim(self) -> None:
        files = [p for p in self.folder.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= self.limit:
            return
        days = sorted((p for p in files if p.parent == self.folder and DAY.match(p.name)),
                      key=lambda p: (DAY.match(p.name).group(1), p.name))   # type: ignore[union-attr]
        chats = sorted((p for p in files if p.parent.name == "chats" and p.name.endswith((".json", ".json.xz"))),
                       key=lambda p: p.stat().st_mtime)
        for p in days[:-1] + chats[:-1]:              # the oldest first; today's lines and the newest chat stay
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            if total <= self.limit:
                break
        named = set()
        for p in (q for q in self.folder.rglob("*")
                  if q.is_file() and q.name.endswith((".jsonl", ".json", ".jsonl.xz", ".json.xz"))):
            try:
                text = _text(p)
            except (lzma.LZMAError, EOFError):
                text = ""
            named |= set(re.findall(r"img/[0-9a-f]{64}\.[a-z0-9.+-]+", text))
        for p in (self.folder / "img").glob("*") if (self.folder / "img").is_dir() else ():
            if f"img/{p.name}" not in named:
                p.unlink(missing_ok=True)
