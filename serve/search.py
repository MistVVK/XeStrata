# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/search.py - web search for a request that asks for it, through one of three search engines.

`search_engine` (the model's config, else the user's settings: serve/runconfig.py GLOBAL) chooses it:

- "searxng" (the default): a SearXNG-compatible search server at `search_url`.  Any server that answers SearXNG's
  `GET /search?q=...&format=json` works.  The parameters some such servers add (max_results, include_raw_content,
  include_domains, exclude_domains) are sent as well; a server that does not know them ignores them, and the results
  are cut and filtered here too.
- "deepseek": DeepSeek's search API (POST /v1/tools/search, billed per search that finds something), with the
  user's `deepseek_api_key` from the settings.
- "tinyfish": TinyFish's search API (free up to a daily allowance), with the user's `tinyfish_api_key` from the
  settings; the results' page texts come from TinyFish's fetch API.

An API key is never logged nor written into a result.  All three give the same results ([{"url", "title", "content",
"raw_content"?}]) and the same notes.

`SearchTools` is a hub run_with_mcp runs (template_tools, routes, call, settings), as the image tools are: the model
calls web_search, and fetch_page for a page whose text the search did not bring.  Every source a call returned is
kept (`sources`), for the citations each API answers with.  A failure (the search server down, a timeout, nothing
found) is a result that starts with "error:", so the model can say so instead of the request failing.

fetch_page goes to an address the model chose, so it connects only to public addresses: the host's addresses are
checked after the name is resolved, the connection goes to the address that was checked (a name that resolves again
to another address cannot slip in), and every redirect is checked the same way.  The search server itself is the
user's choice and may be on this PC.
"""
from __future__ import annotations

import html.parser
import http.client
import ipaddress
import json
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# results, and the characters of a page's text, by the request's search_context_size (OpenAI's low / medium / high)
CONTEXT = {"low": (3, 2000), "medium": (5, 4000), "high": (8, 8000)}
RESULT_CHARS = 24000                     # all of one call's result for the model
SEARCH_TIMEOUT_S = 20                    # a server asks its engines in parallel: about 1-3 s, more with page texts
FETCH_TIMEOUT_S = 15
FETCH_BYTES = 2 << 20                    # a page read for its text: 2 MiB of HTML is far more than its words
REDIRECTS = 5
AGENT = "XeStrata (web search)"

TOOLS = [
    {"name": "web_search",
     "description": "Search the web. Returns the top results with their titles, addresses, snippets and, where the "
                    "search server brings it, the page's text. Use it for facts that may be recent or that you are "
                    "not sure of, and cite the addresses you use.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "the search words, in the language the pages are likely in"}},
         "required": ["query"]}},
    {"name": "fetch_page",
     "description": "Read one web page's text (an address from web_search, or one the user gave).",
     "parameters": {"type": "object", "properties": {
         "url": {"type": "string", "description": "the page's http or https address"}},
         "required": ["url"]}},
]


class SearchError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ page text
class _Text(html.parser.HTMLParser):
    """The words of an HTML page and its title: scripts, styles and the head left out, blocks on lines of their own."""
    SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header",
             "footer", "pre", "blockquote", "table", "ul", "ol", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.skip = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        elif not self.skip:
            self.parts.append(data)


def page_text(body: bytes, content_type: str) -> tuple[str, str]:
    """(title, text) of a fetched page."""
    m = re.search(r"charset=([\w-]+)", content_type, re.I)
    meta = None if m else re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', body[:4096], re.I)
    charset = m.group(1) if m else meta.group(1).decode("ascii") if meta else "utf-8"
    try:
        text = body.decode(charset, "replace")
    except LookupError:
        text = body.decode("utf-8", "replace")
    if "html" not in content_type.lower() and not re.match(r"\s*<", text):
        return "", text.strip()
    p = _Text()
    p.feed(text)
    words = "\n".join(re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(p.parts).split("\n"))
    return " ".join("".join(p.title).split()), re.sub(r"\n{3,}", "\n\n", words).strip()


# ------------------------------------------------------------------------------------------------ public fetch
def public_address(host: str, port: int) -> str:
    """An address of `host` to connect to, when every address the name has is a public one; else SearchError."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise SearchError(f"{host} cannot be resolved ({e.strerror or e})") from None
    addrs = []
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        mapped = getattr(ip, "ipv4_mapped", None)
        if not (mapped or ip).is_global or (mapped or ip).is_multicast:
            raise SearchError(f"{host} is not a public address ({ip}): pages on this PC or its network are not read")
        addrs.append(str(ip))
    if not addrs:
        raise SearchError(f"{host} has no address")
    return addrs[0]


class _PinnedHTTPS(http.client.HTTPSConnection):
    """An HTTPS connection to an address already checked, with the certificate checked for the host's name."""

    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self.ip = ip

    def connect(self):
        sock = socket.create_connection((self.ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)  # type: ignore[attr-defined]


def fetch_public(url: str, max_bytes: int = FETCH_BYTES, timeout: float = FETCH_TIMEOUT_S) -> tuple[str, str, bytes]:
    """(final url, content type, body) of a public http(s) page, redirects followed and each one checked."""
    for _ in range(REDIRECTS + 1):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise SearchError(f"not an http or https address: {url!r}")
        port = u.port or (443 if u.scheme == "https" else 80)
        ip = public_address(u.hostname, port)
        conn = (_PinnedHTTPS(u.hostname, ip, port, timeout) if u.scheme == "https"
                else http.client.HTTPConnection(ip, port, timeout=timeout))
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        try:
            conn.request("GET", path, headers={"Host": u.netloc.rsplit("@", 1)[-1], "User-Agent": AGENT,
                                               "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"})
            r = conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and r.getheader("Location"):
                url = urllib.parse.urljoin(url, r.getheader("Location") or "")
                continue
            if r.status >= 400:
                raise SearchError(f"{url} answered HTTP {r.status}")
            return url, r.getheader("Content-Type") or "", r.read(max_bytes)
        except (OSError, http.client.HTTPException) as e:
            raise SearchError(f"{url} could not be read ({getattr(e, 'strerror', None) or e})") from None
        finally:
            conn.close()
    raise SearchError(f"{url}: more than {REDIRECTS} redirects")


# ------------------------------------------------------------------------------------------------ the search server
def _domain_ok(url: str, include: list, exclude: list) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()

    def under(d):
        d = d.lower().lstrip(".")
        return host == d or host.endswith("." + d)
    return (not include or any(under(d) for d in include)) and not any(under(d) for d in exclude)


class Searxng:
    """A SearXNG-compatible search server."""
    name = "searxng"

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def search(self, query: str, count: int, include=(), exclude=(), locale=None) -> tuple[list[dict], list[str]]:
        return search(self.base, query, count, True, include, exclude), []


class DeepSeek:
    """DeepSeek's search API (api-docs.deepseek.com/guides/search): a query of 1 to 128 characters with operators
    (site:, -site:), up to 50 results, each with a summary for a model and, asked, the page's text."""
    name = "deepseek"
    URL = "https://api.deepseek.com/v1/tools/search"
    QUERY_CHARS = 128
    ERRORS = {401: "DeepSeek refused the API key (deepseek_api_key in the settings)",
              402: "the DeepSeek account has no balance left for searches",
              429: "DeepSeek's search limit was reached; try again later"}

    def __init__(self, key: str):
        self._key = key

    def __repr__(self) -> str:                           # never the key, in a log or a traceback
        return "DeepSeek(search API)"

    def search(self, query: str, count: int, include=(), exclude=(), locale=None) -> tuple[list[dict], list[str]]:
        # `locale` is not used: DeepSeek's only language control (the query's lang:) filters, it does not prefer
        notes = []
        if len(query) > self.QUERY_CHARS:
            query = query[:self.QUERY_CHARS]
            notes.append(f"the query was cut to DeepSeek's {self.QUERY_CHARS} characters: {query!r}")
        ops = " ".join([*(f"site:{d}" for d in include), *(f"-site:{d}" for d in exclude)])
        if ops and len(query) + 1 + len(ops) <= self.QUERY_CHARS:   # else the domains are filtered here only
            query = f"{query} {ops}"
        body = json.dumps({"query": query, "count": max(1, min(count, 50)), "need_content": True}).encode()
        for attempt in (0, 1):                           # 408, 429 and 5xx once more, as DeepSeek's guide says
            req = urllib.request.Request(self.URL, body, {"Authorization": f"Bearer {self._key}",
                                                          "Content-Type": "application/json", "User-Agent": AGENT})
            try:
                with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as r:
                    got = json.loads(r.read(16 << 20))
                break
            except urllib.error.HTTPError as e:
                if attempt == 0 and (e.code in (408, 429) or e.code >= 500):
                    time.sleep(1.0)
                    continue
                raise SearchError(self.ERRORS.get(e.code, f"DeepSeek's search answered HTTP {e.code}")) from None
            except (OSError, ValueError) as e:
                raise SearchError(f"DeepSeek's search cannot be reached ({getattr(e, 'reason', None) or e})") \
                    from None
        out, have = [], set()
        results = got.get("results") if isinstance(got, dict) else None
        for r in results if isinstance(results, list) else []:
            url = r.get("url") if isinstance(r, dict) else None
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            # DeepSeek's result URLs end in numbered fragments of its own (".../specifications.html#1#1", seen
            # 2026-11-02): the page's address is the part before them
            url = re.sub(r"(#\d+)+$", "", url)
            if url in have or not _domain_ok(url, include, exclude):
                continue
            have.add(url)
            out.append({"url": url, "title": str(r.get("title") or url),
                        "content": str(r.get("summary") or r.get("snippet") or ""),
                        **({"raw_content": r["content"]} if isinstance(r.get("content"), str) else {})})
            if len(out) >= count:
                break
        return out, notes


class TinyFish:
    """TinyFish's search API (docs.tinyfish.ai/search-api), with the results' page texts from its fetch API in one
    request (docs.tinyfish.ai/fetch-api: up to 10 addresses, the text as Markdown).  Both are free up to a daily
    allowance (12,000 searches, 1,000 fetched pages), so a search here takes `count` pages of it."""
    name = "tinyfish"
    SEARCH_URL = "https://api.search.tinyfish.ai"
    FETCH_URL = "https://api.fetch.tinyfish.ai"
    FETCH_URLS = 10
    FETCH_PAGE_MS = 12000                                # each page's limit at TinyFish: the call waits a little more
    ERRORS = {401: "TinyFish refused the API key (tinyfish_api_key in the settings)",
              402: "the TinyFish account's daily allowance is used up and it has no balance left",
              429: "TinyFish's search limit (requests per minute) was reached; try again later"}

    def __init__(self, key: str):
        self._key = key

    def __repr__(self) -> str:                           # never the key, in a log or a traceback
        return "TinyFish(search API)"

    def _call(self, req: urllib.request.Request, what: str, timeout: float):
        for attempt in (0, 1):                           # 429 and 5xx once more
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read(16 << 20))
            except urllib.error.HTTPError as e:
                if attempt == 0 and (e.code == 429 or e.code >= 500):
                    time.sleep(1.0)
                    continue
                raise SearchError(self.ERRORS.get(e.code, f"TinyFish's {what} answered HTTP {e.code}")) from None
            except (OSError, ValueError) as e:
                raise SearchError(f"TinyFish's {what} cannot be reached ({getattr(e, 'reason', None) or e})") \
                    from None
        return None

    def search(self, query: str, count: int, include=(), exclude=(), locale=None) -> tuple[list[dict], list[str]]:
        params = {"query": query, **(locale or {})}      # location / language: TinyFish resolves the one not given
        if include:
            params["include_domains"] = ",".join(include)
        if exclude:
            params["exclude_domains"] = ",".join(exclude)
        req = urllib.request.Request(f"{self.SEARCH_URL}?{urllib.parse.urlencode(params)}",
                                     headers={"X-API-Key": self._key, "Accept": "application/json",
                                              "User-Agent": AGENT})
        got = self._call(req, "search", SEARCH_TIMEOUT_S)
        out, have = [], set()
        results = got.get("results") if isinstance(got, dict) else None
        for r in results if isinstance(results, list) else []:
            url = r.get("url") if isinstance(r, dict) else None
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            if url in have or not _domain_ok(url, include, exclude):
                continue
            have.add(url)
            out.append({"url": url, "title": str(r.get("title") or url), "content": str(r.get("snippet") or "")})
            if len(out) >= count:
                break
        notes = []
        if out:
            try:
                self._texts(out[:self.FETCH_URLS])
            except SearchError as e:                     # the snippets still answer
                notes.append(f"the pages' texts were not fetched: {e}")
        return out, notes

    def _texts(self, found: list[dict]) -> None:
        """The pages' texts as raw_content, from one fetch request (a page TinyFish could not read keeps none)."""
        body = json.dumps({"urls": [r["url"] for r in found], "format": "markdown",
                           "per_url_timeout_ms": self.FETCH_PAGE_MS}).encode()
        req = urllib.request.Request(self.FETCH_URL, body, {"X-API-Key": self._key, "Accept": "application/json",
                                                            "Content-Type": "application/json", "User-Agent": AGENT})
        got = self._call(req, "page fetch", self.FETCH_PAGE_MS / 1000 + 8)
        by_url = {r["url"]: r for r in found}
        results = got.get("results") if isinstance(got, dict) else None
        for p in results if isinstance(results, list) else []:
            r = by_url.get(p.get("url")) if isinstance(p, dict) else None
            if r is not None and isinstance(p.get("text"), str) and p["text"].strip():
                r["raw_content"] = p["text"]


def search(base: str, query: str, count: int, raw: bool, include=(), exclude=(),
           timeout: float = SEARCH_TIMEOUT_S) -> list[dict]:
    """The search server's results for `query`: [{"url", "title", "content", "raw_content"?}], at most `count`."""
    params = {"q": query, "format": "json", "max_results": count}
    if raw:
        params["include_raw_content"] = 1
    if include:
        params["include_domains"] = ",".join(include)
    if exclude:
        params["exclude_domains"] = ",".join(exclude)
    req = urllib.request.Request(f"{base}/search?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read(16 << 20))
    except urllib.error.HTTPError as e:
        hint = " (SearXNG answers JSON only with \"json\" in its search.formats)" if e.code == 403 else ""
        raise SearchError(f"the search server answered HTTP {e.code}{hint}") from None
    except (OSError, ValueError) as e:
        raise SearchError(f"the search server at {base} cannot be reached ({getattr(e, 'reason', None) or e})") \
            from None
    out = []
    results = body.get("results") if isinstance(body, dict) else None
    for r in results if isinstance(results, list) else []:
        url = r.get("url") if isinstance(r, dict) else None
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if not _domain_ok(url, include, exclude):
            continue
        out.append({"url": url, "title": str(r.get("title") or url), "content": str(r.get("content") or ""),
                    **({"raw_content": r["raw_content"]} if isinstance(r.get("raw_content"), str) else {})})
        if len(out) >= count:
            break
    return out


def locale(user_location, accept_language: str | None) -> dict:
    """Where the search is for: {"location": country, "language": code}, either or neither.  The request's
    user_location (OpenAI's Chat {"approximate": {"country"}}, its Responses' and Anthropic's {"country"}) first, its
    country alone; else the first language of Accept-Language (a browser's, as the web page's chat sends), with its
    region when it names one."""
    if isinstance(user_location, dict):
        inner = user_location.get("approximate")
        country = (inner if isinstance(inner, dict) else user_location).get("country")
        if isinstance(country, str) and re.fullmatch(r"[A-Za-z]{2}", country):
            return {"location": country.upper()}
    best, best_q = None, 0.0
    for part in (accept_language or "").split(","):
        tag, _, q = part.strip().partition(";q=")
        try:
            weight = float(q) if q else 1.0
        except ValueError:
            continue
        if tag and tag != "*" and weight > best_q:
            best, best_q = tag, weight
    subtags = (best or "").replace("_", "-").split("-")
    if not re.fullmatch(r"[A-Za-z]{2,3}", subtags[0]):
        return {}
    region = next((t for t in subtags[1:] if re.fullmatch(r"[A-Za-z]{2}", t)), None)
    return {"language": subtags[0].lower(), **({"location": region.upper()} if region else {})}


class SearchTools:
    """web_search and fetch_page for one request (the hub run_with_mcp runs).  `engine`: Searxng, DeepSeek or
    TinyFish.
    `options`: the request's search settings - "context" (low / medium / high), "include" and "exclude" (domains),
    "max_uses", "locale" (locale() above)."""

    def __init__(self, engine: Searxng | DeepSeek | TinyFish, options: dict | None = None):
        o = options or {}
        self.engine = engine
        self.count, self.chars = CONTEXT.get(o.get("context") or "medium", CONTEXT["medium"])
        self.include = [d for d in o.get("include") or [] if isinstance(d, str) and d]
        self.exclude = [d for d in o.get("exclude") or [] if isinstance(d, str) and d]
        self.locale = o.get("locale") or {}
        self.max_uses = o.get("max_uses") if isinstance(o.get("max_uses"), int) and o["max_uses"] > 0 else None
        self.settings = {"max_rounds": 4}
        self.sources: list[dict] = []           # every result returned, in order, once per address
        self.searches: list[dict] = []          # {"id", "query", "sources", "ok"}: one per web_search call
        self.calls = 0

    def template_tools(self, exclude=()) -> list[dict]:
        return [dict(t) for t in TOOLS if t["name"] not in exclude]

    def routes(self) -> dict:
        return {t["name"]: (None, t["name"]) for t in self.template_tools()}

    def _keep(self, items: list[dict]) -> None:
        have = {s["url"] for s in self.sources}
        self.sources += [{"url": s["url"], "title": s["title"]} for s in items if s["url"] not in have]

    def call(self, name: str, arguments, cancel: threading.Event | None = None) -> dict:
        t0 = time.monotonic()
        args = arguments if isinstance(arguments, dict) else {}
        found: list[dict] = []
        truncated = False
        try:
            if self.max_uses is not None and self.calls >= self.max_uses:
                raise SearchError(f"not run: this request allows {self.max_uses} search tool call(s) (max_uses)")
            self.calls += 1
            if name == "web_search":
                query = str(args.get("query") or "").strip()
                if not query:
                    raise SearchError("web_search needs a query")
                found, notes = self.engine.search(query, self.count, self.include, self.exclude, self.locale)
                if not found:
                    raise SearchError(f"nothing was found for {query!r}")
                lines = [f"({n})" for n in notes]
                for i, r in enumerate(found, 1):
                    body = (r.get("raw_content") or "").strip()
                    lines.append(f"[{i}] {r['title']}\n{r['url']}\n{r['content'].strip()}" +
                                 (f"\n--- page text ---\n{body[:self.chars]}" if body else ""))
                    truncated = truncated or len(body) > self.chars
                text = f"Results for {query!r}:\n\n" + "\n\n".join(lines)
            elif name == "fetch_page":
                url = str(args.get("url") or "").strip()
                final, ctype, body = fetch_public(url)
                title, words = page_text(body, ctype)
                if not words:
                    raise SearchError(f"{final} has no text to read")
                found = [{"url": final, "title": title or final, "content": ""}]
                limit = self.chars * 2
                truncated = len(words) > limit
                text = f"{title}\n{final}\n\n{words[:limit]}"
            else:
                raise SearchError(f"there is no tool named {name!r}")
            ok = True
        except SearchError as e:
            text, ok = f"error: {e}", False
        if len(text) > RESULT_CHARS:
            text, truncated = text[:RESULT_CHARS], True
        if truncated:
            text += "\n[cut to fit: ask fetch_page for one page's text]"
        self._keep(found)
        if name == "web_search":
            self.searches.append({"query": str(args.get("query") or ""), "ok": ok,
                                  "sources": [{"url": s["url"], "title": s["title"]} for s in found]})
        return {"ok": ok, "text": text, "chars": len(text), "truncated": truncated,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name,
                "sources": [{"url": s["url"], "title": s["title"]} for s in found]}


def cited(text: str, sources: list[dict]) -> list[dict]:
    """The sources whose address the answer writes, each with where: [{"url", "title", "start", "end"}]."""
    out = []
    for s in sources:
        at = text.find(s["url"])
        if at >= 0:
            out.append({**s, "start": at, "end": at + len(s["url"])})
    return sorted(out, key=lambda c: c["start"])
