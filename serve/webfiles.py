# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Which files the server hands out below /web/ from folders: the page's translations, the vendored KaTeX and mermaid.

The flat files (app.js, app.css, ...) are served by serve/server.py itself. The two folders here hold more than one
level, so each has its own whitelist: a name pattern and the types it may have. Nothing outside these folders is ever
read, a name with a backslash, a colon or a percent sign is refused, and the resolved path must stay inside the folder.
The translations are serve/web/locales; KaTeX is another project's (MIT), so it lives in third_party/main/katex and is
served at upstream's URL, /web/vendor/katex/; mermaid (MIT, third_party/main/mermaid) at /web/vendor/mermaid/, its
script alone.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
KATEX = Path(__file__).resolve().parents[1] / "third_party" / "main" / "katex"
MERMAID = Path(__file__).resolve().parents[1] / "third_party" / "main" / "mermaid"
MERMAID_PREFIX = "/web/vendor/mermaid/"

# /web/locales/<lang>.json   e.g. de.json, pt-BR.json
LOCALE = re.compile(r"^/web/locales/(?P<name>[a-z]{2,3}(?:-[A-Z]{2})?\.json)$")
# /web/vendor/katex/dist/... : the script, the style sheet and the fonts only
VENDOR_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".woff2": "font/woff2"}
VENDOR_PREFIX = "/web/vendor/katex/"


def subtree_file(path: str, web: Path = WEB, katex: Path = KATEX, mermaid: Path = MERMAID):
    """(file, content type) for a /web/locales or /web/vendor path, or None when it is not served."""
    if any(c in path for c in ("\\", ":", "%", "\0")) or ".." in path:
        return None
    m = LOCALE.match(path)
    if m:
        f = web / "locales" / m.group("name")
        return (f, "application/json; charset=utf-8") if f.is_file() else None
    if path == MERMAID_PREFIX + "dist/mermaid.min.js":
        f = mermaid / "dist" / "mermaid.min.js"
        return (f, VENDOR_TYPES[".js"]) if f.is_file() else None
    if path.startswith(VENDOR_PREFIX):
        vendor = katex.resolve()
        f = (vendor / path[len(VENDOR_PREFIX):]).resolve()
        ctype = VENDOR_TYPES.get(f.suffix)
        if ctype and f.is_relative_to(vendor) and f.is_file():
            return f, ctype
    return None
