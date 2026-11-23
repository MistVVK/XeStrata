# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/test_web_sampling.py - the web page sends no sampling of its own until the user changes it (0.1.45, #2685).

The page used to send temperature 0.6 / top_p 0.95 / top_k 20 with every request, so the server's model-card defaults
(thinking 1.0, not thinking 0.7 + presence penalty) never applied to it.  Pins, with node, the page's own functions:
untouched (the shown defaults, or the old stored 0.6) sends nothing, a changed value is sent whole, 0 is greedy.
The thinking level the model takes by default is no reasoning_effort either (its own default applies).
Skipped without node.

    python -m unittest serve.test_web_sampling -v
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"


def slice_between(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebSampling(unittest.TestCase):
    defaults: str
    thinking: str
    shared: str
    body: str

    @classmethod
    def setUpClass(cls):
        src = APP.read_text(encoding="utf-8")
        cls.defaults = slice_between(src, "const DEFAULTS", "let settings")
        cls.thinking = slice_between(src, "function thinkingInfo", "function thinkingButtons")
        cls.shared = slice_between(src, "function sharedDefaults", "async function saveShared")
        cls.body = slice_between(src, "  if (!(settings.temperature > 0)) {", "  if (settings.seed)")

    def run_js(self, settings: dict, pieces: str, health: dict | None = None):
        js = (self.defaults + "function projectionLoaded() { return false; }\n"
              f"const health = {json.dumps(health or {})};\n" + self.thinking +
              self.shared +
              f"\nconst settings = {json.dumps(settings)};\nconst body = {{}};\n{pieces}\n"
              "console.log(JSON.stringify({body, shared: sharedDefaults(settings)}));")
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def sampling(self, **kw):
        s = {"thinking": "high", "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max": "", "seed": "", "esp": True}
        s.update(kw)
        r = self.run_js(s, self.body)
        return r["body"], r["shared"]

    def test_untouched_sends_nothing(self):
        for t in (1.0, 0.6):                       # the shown default, and the old stored default
            body, shared = self.sampling(temperature=t)
            self.assertEqual(body, {})
            self.assertEqual(shared, {})

    def test_changed_values_are_sent_whole(self):
        body, shared = self.sampling(temperature=0.3)
        self.assertEqual(body, {"temperature": 0.3, "top_p": 0.95, "top_k": 20})
        self.assertEqual(shared["temperature"], 0.3)
        body, _ = self.sampling(top_k=40)
        self.assertEqual(body, {"temperature": 1.0, "top_p": 0.95, "top_k": 40})
        body, _ = self.sampling(temperature=0.6, top_p=0.9)
        self.assertEqual(body["temperature"], 0.6)

    def test_zero_is_greedy(self):
        body, shared = self.sampling(temperature=0)
        self.assertEqual(body, {"temperature": 0})
        self.assertEqual(shared["temperature"], 0)

    def test_thinking_is_sent_by_the_model_own_levels(self):
        glm = {"thinking": {"levels": ["none", "low", "high", "max"], "default": "max"}}
        onoff = {"thinking": {"levels": ["none", "on"], "default": "on"}}
        for health, level, sent in ((glm, "", None), (glm, "max", None), (glm, "high", "high"),
                                    (glm, "medium", None),      # not GLM's: its default
                                    (onoff, "on", None), (onoff, "none", "none"), (onoff, "low", None)):
            s = {"thinking": level, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max": "", "seed": "",
                 "esp": True}
            shared = self.run_js(s, "", health)["shared"]
            self.assertEqual(shared.get("reasoning_effort"), sent, (health, level))

    def test_the_old_default_is_shown_as_the_new_one(self):
        src = APP.read_text(encoding="utf-8")
        self.assertRegex(src, r"if \(samplingUntouched\(settings\)\) settings\.temperature = DEFAULTS\.temperature;")
        self.assertIsNone(re.search(r"temperature: 0\.6", src))


if __name__ == "__main__":
    unittest.main()
