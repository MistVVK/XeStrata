"""serve/test_web_start_question.py - /?q= fills in the question and never sends it (#4221).

Opening /?q=<text> used to put the text in the box and call send() on load, so any web page could link to
http://127.0.0.1:<port>/?q=... and make the model answer, with the API key and the MCP tools saved in that browser,
without the person pressing Send.  The page's own start() runs here under Node against a fake page and a fake credentialed
backend with a request spy: after navigation and after a reload there are zero generation, tool or chat requests, the
question is in the box and the box has focus; only an explicit Send (the composer's own handler) reaches the backend.
Skipped without node.

    python -m unittest serve.test_web_start_question -v
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"


def between(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


HARNESS = r"""
const calls = [];                       // every request the page makes to its backend, in order
const sends = [];                       // every time the page's send() runs
const focus = [];
const input = {value: "", style: {}, scrollHeight: 40, focus() { focus.push("input"); }};
const page = {url: new URL(%(url)s), hash: ""};
const location = {
  get search() { return page.url.search; },
  get pathname() { return page.url.pathname; },
  get hash() { return page.hash; },
};
const history = {replaceState(_s, _t, url) { page.url = new URL(url, page.url); }};
const innerHeight = 800;
const $ = (id) => (id === "input" ? input : {focus() {}, set value(v) {}, get value() { return ""; }});
// a credentialed backend: the saved API key and the MCP tools are what a send would use
async function fetch(path, opts) { calls.push({path: String(path), method: (opts && opts.method) || "GET", auth: !!(opts && opts.headers && opts.headers.Authorization)}); return {ok: true, json: async () => ({})}; }
const StrataI18n = {lang: "en", setLanguage: async () => {}};
const store = {get: (_k, d) => d};
const settings = {chatList: false};
const urlLang = null;
const tab = "chat";
function renderEmptyChat() {}
function initChats() {}
async function setChatList() {}
function setBusy() {}
function renderChat() {}
function poll() {}
function fitHeader() {}
const requestAnimationFrame = () => {};
async function loadHealth() { await fetch("health"); }
async function loadMcp() { await fetch("v1/mcp"); }
function showTab(name) { focus.push("tab:" + (name || "chat")); }
async function send() { sends.push(input.value); await fetch("v1/chat/completions", {method: "POST", headers: {Authorization: "Bearer saved-key"}}); }
%(autosize)s
%(start)s
(async () => {
  await start();
  await new Promise((r) => setTimeout(r, 20));        // let the health and tool-list fetches settle
  const first = {input: input.value, url: page.url.pathname + page.url.search, calls: calls.slice(), sends: sends.length, focus: focus.slice()};
  // reload: the browser asks for the address it now has (the ?q= was removed from it by the first start)
  input.value = ""; calls.length = 0; sends.length = 0; focus.length = 0;
  await start();
  await new Promise((r) => setTimeout(r, 20));
  const second = {input: input.value, calls: calls.slice(), sends: sends.length};
  console.log(JSON.stringify({first, second}));
})();
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class StartQuestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = APP.read_text(encoding="utf-8")
        cls.start = between(cls.src, "async function start() {", "\nstart();")
        m = re.search(r"function autosize\(\)[^\n]*", cls.src)
        assert m, "autosize() is gone"
        cls.autosize = m.group(0)

    def run_page(self, url):
        js = HARNESS % {"url": json.dumps(url), "start": self.start, "autosize": self.autosize}
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_navigation_fills_the_question_and_sends_nothing(self):
        r = self.run_page("http://127.0.0.1:8080/?q=Delete%20my%20files")["first"]
        self.assertEqual(r["input"], "Delete my files")
        self.assertEqual(r["sends"], 0)
        generation = [c for c in r["calls"] if c["method"] != "GET" or "chat" in c["path"] or "completions" in c["path"] or "tool" in c["path"]]
        self.assertEqual(generation, [], "no model or tool request fires on navigation")
        self.assertFalse([c for c in r["calls"] if c["auth"]], "nothing went out with the saved key")
        self.assertIn("input", r["focus"], "the box is focused")
        self.assertEqual(r["url"], "/", "the question is taken out of the address")

    def test_reload_after_that_sends_nothing_and_does_not_refill(self):
        r = self.run_page("http://127.0.0.1:8080/?q=hello")["second"]
        self.assertEqual(r["sends"], 0)
        self.assertEqual(r["input"], "")
        self.assertEqual([c for c in r["calls"] if c["method"] != "GET"], [])

    def test_no_question_in_the_address_changes_nothing(self):
        r = self.run_page("http://127.0.0.1:8080/")["first"]
        self.assertEqual(r["input"], "")
        self.assertEqual(r["sends"], 0)

    def test_long_and_odd_questions_are_filled_as_text(self):
        for q in ["%3Cscript%3Ealert(1)%3C%2Fscript%3E", "a" * 5000, "%E4%B8%AD%E6%96%87%20x"]:
            r = self.run_page(f"http://127.0.0.1:8080/?q={q}")["first"]
            self.assertEqual(r["sends"], 0)
            self.assertTrue(r["input"])

    def test_the_page_never_calls_send_from_the_address(self):
        # the only things that call send() are the composer's own submit and the Enter key
        text = self.src.splitlines()
        lines = [n for n, line in enumerate(text, 1) if re.search(r"\bsend\(\)", line.split("//")[0])]
        self.assertTrue(lines)
        for n in lines:
            line = text[n - 1]
            self.assertTrue("onsubmit" in line or "preventDefault" in line, f"app.js:{n} calls send() outside a person's action: {line.strip()}")
        self.assertNotIn("startQuestion) { $(\"input\").value = startQuestion; send()", self.src)


if __name__ == "__main__":
    unittest.main()
