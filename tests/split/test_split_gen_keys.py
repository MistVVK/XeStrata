"""The GEN keys serve/server.py can send an engine, against the keys the all-cards split engine's parser (src/split/tools/sfnx.cpp, parse_gen) accepts.

A key the split engine does not know ends the request with "GEN: unknown key ..." (an HTTP 503 in the middle of a reply: pen_from was found that way on the AMD package
gate). Every key the server can send is classified here; a new one fails this test until it is sent to the split engine safely, dropped for it, or refused with a 400."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# keys the split engine's parse_gen accepts (some only note that they change nothing: pin, pcie_frac, spec_min_p, pipe, aux_cpus, cvec)
ACCEPTED = {"temperature", "top_p", "top_k", "min_p", "penalty_repeat", "penalty_freq", "penalty_present", "penalty_last_n", "seed", "keep", "ckpt", "pin",
            "pcie_frac", "spec_min_p", "pipe", "aux_cpus", "cvec", "img_keys"}
# keys the server does not send to a split engine: pen_from is dropped (StrataEngine.pen_from_key), logprobs / allowed / logit_bias are refused with a 400 first
# (refuse_for_split, logit_bias.validate_request)
DROPPED = {"pen_from"}
REFUSED = {"logprobs", "allowed", "logit_bias"}
# keys only other engines are sent: the prefill-only request of serve/split.py,
# and hold (the thinking budget's GEN hold=N:ID): the server sends it only to an engine whose READY line says "hold" (a --batch engine: "holdb" too), which is the
# single-card engine (src/program/generate.cpp prints "READY <ctx> stop hold holdb"); the split engine says "READY <ctx> stop" (sfnx.cpp), so it is never sent it, and
# the server also refuses it for a split engine by name (Service._run).  Sent to a split engine anyway, parse_gen would end the request with "GEN: unknown key hold".
OTHER_ENGINES = {"prefill", "hold"}


def server_keys() -> set[str]:
    text = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'keys \+= f?" (\w+)=', text))
    keys |= set(re.findall(r'keys \+= " (\w+)="', text))
    for tup in re.findall(r'for k in \(((?:"\w+",? ?)+)\):\n\s+v = tune\.get\(k\)', text):
        keys |= set(re.findall(r'"(\w+)"', tup))
    keys |= set(re.findall(r'f" (pen_from)=', text))
    keys |= set(re.findall(r'" (prefill)=1', text))
    keys |= set(re.findall(r'f" (cvec)=', text))
    keys |= set(re.findall(r'"(logit_bias)="', (ROOT / "serve" / "logit_bias.py").read_text(encoding="utf-8")))
    keys |= set(re.findall(r'" (\w+)=" \+', (ROOT / "serve" / "logit_bias.py").read_text(encoding="utf-8")))
    keys |= set(re.findall(r'keys \+= f" (\w+)=', text))
    keys |= set(re.findall(r'keys0 \+ f" (\w+)=', text))   # generate_batched adds the thinking budget's hold per leg (P2b)
    return keys


def parser_keys() -> set[str]:
    text = (ROOT / "src" / "split" / "tools" / "sfnx.cpp").read_text(encoding="utf-8")
    body = text.split("static GenReq parse_gen", 1)[-1] if "static GenReq parse_gen" in text else text
    return set(re.findall(r'k == "(\w+)"', body))


class SplitGenKeys(unittest.TestCase):
    def test_every_key_the_server_can_send_is_classified(self):
        sent = server_keys()
        known = ACCEPTED | DROPPED | REFUSED | OTHER_ENGINES
        self.assertEqual(sorted(sent - known), [], "a GEN key the server sends that nobody classified for the split engine")
        for must in ("pen_from", "logprobs", "allowed", "temperature", "penalty_last_n", "seed"):
            self.assertIn(must, sent, f"{must}: the parse of serve/server.py no longer finds it (update this test)")

    def test_hold_is_only_for_an_engine_that_says_hold_and_the_split_engine_never_does(self):
        """The server sends `hold` only to an engine whose READY line says "hold" (the --batch slots: "holdb"); the split engine's READY line says neither, and
        its parser does not know the key (it would end the request: "GEN: unknown key hold")."""
        split = (ROOT / "src" / "split" / "tools" / "sfnx.cpp").read_text(encoding="utf-8")
        readys = re.findall(r'say\("READY "[^;]*;', split)
        self.assertTrue(readys, "the split engine's READY line is no longer found (update this test)")
        for line in readys:
            self.assertNotIn("hold", line)
        self.assertNotIn("hold", parser_keys())
        gen = (ROOT / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
        self.assertIn('"READY %lld stop hold holdb\\n"', gen)      # the engine that does take it says both
        self.assertIn('"hold" in f[2:]', (ROOT / "serve" / "server.py").read_text(encoding="utf-8"))   # and the server reads the word, not the engine's name

    def test_the_keys_sent_to_a_split_engine_are_accepted_by_its_parser(self):
        parser = parser_keys()
        self.assertEqual(sorted(ACCEPTED - parser), [], "a key the split engine is sent but its parse_gen does not know")

    def test_pen_from_is_not_sent_to_a_split_engine_and_is_to_the_others(self):
        from serve import server

        split = object.__new__(server.StrataEngine)
        split.split_engine = True
        other = object.__new__(server.StrataEngine)
        other.split_engine = False
        self.assertEqual(split.pen_from_key(100, 130), "")
        self.assertEqual(other.pen_from_key(100, 130), " pen_from=99")
        self.assertEqual(other.pen_from_key(100, 100), "")

    def test_a_split_engine_name_is_found_by_its_executable(self):
        from serve import server

        for exe, want in (("/x/strata-split-4-amd", True), ("/x/strata-split-2-sm86", True), ("C:/x/strata-split-2", True), ("/x/strata", False),
                          ("/x/strata-vision", False)):
            e = object.__new__(server.StrataEngine)
            e.split_engine = __import__("os").path.basename(exe).startswith("strata-split")
            self.assertEqual(e.split_engine, want, exe)

    def test_logprobs_and_logit_bias_are_refused_with_a_400_for_a_split_engine(self):
        from serve import logit_bias, server

        e = object.__new__(server.StrataEngine)
        e.split_engine = True
        with self.assertRaises(ValueError) as cm:
            server.refuse_for_split(e, {"logprobs": True, "top_logprobs": 3})
        self.assertIn("--split-engine off", str(cm.exception))
        server.refuse_for_split(e, {"temperature": 0.7})          # an ordinary request goes on
        with self.assertRaises(ValueError) as cm:
            logit_bias.validate_request({"logit_bias": {"5": 1.0}}, e, 1000)
        self.assertIn("--split-engine off", str(cm.exception))
        other = object.__new__(server.StrataEngine)
        other.split_engine = False
        server.refuse_for_split(other, {"logprobs": True, "top_logprobs": 3})   # other engines unchanged


if __name__ == "__main__":
    unittest.main()
