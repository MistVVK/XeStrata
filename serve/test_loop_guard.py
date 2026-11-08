# SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The loop guard (serve/loop_guard.py): what it calls a loop and what not, and what Service.run does with one -
against an engine that continues an answer from wherever the prompt says it got to.  The detection and policy cases
are transcribed from Project Maya's serve/test_loop_guard.py (249f914de, afd465ed1; MIT).

    python -m unittest serve.test_loop_guard -v
"""
from __future__ import annotations

import json
import random
import re
import threading
import unittest

from serve import loop_guard as LG
from serve.server import (IM_END, ByteTokenizer, Detokenizer, Service, check_request_sampling,
                          loop_guard_setting)


class Words:
    """A stand-in for a BPE vocabulary: a word with its leading space, a run of punctuation or of white space."""
    PIECE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

    def __init__(self):
        self.ids: dict[str, int] = {}

    def __call__(self, text: str) -> list[int]:
        return [self.ids.setdefault(p, len(self.ids)) for p in self.PIECE.findall(text)]


def first_loop(ids):
    det = LG.LoopDetector()
    for i, t in enumerate(ids):
        loop = det.push(t)
        if loop is not None:
            return i, loop
    return None


LONG_BLOCK = " ".join(random.Random(11).choices(
    "check the metric logs then the deploy diff and the gc pauses before the p99 of each host".split(), k=600)) + "\n"

LEGIT = {
    "markdown table": "| id | name | score |\n|---|---|---|\n" + "".join(
        f"| {i} | user{i * 7 % 13} | {(i * 37) % 100} |\n" for i in range(60)),
    "a long block twice": "Draft:\n" + LONG_BLOCK + "Final:\n" + LONG_BLOCK + "Done.",
    "JSON array": json.dumps(list(range(300))),
    "one function name": "".join(f"result = process_item(items[{i}], options)\n" for i in range(80)),
    "similar code lines": "".join(f"    x[{i}] = a[{i}] * w[{i % 4}] + b[{i}];\n" for i in range(64)),
    "a short line three times": "Hello there, world!\n" * 3 + "Done.",
    "a short row of zeros": "board = [" + ", ".join(["0"] * 30) + "]\n",
}


class Detection(unittest.TestCase):
    def test_real_loops_fire(self):
        w = Words()
        for name, head, block, n in (("number sequence", "The values are: ", "1017, 245, 26, 100, 47, 187, ", 40),
                                     ("one token", "Result: ", "0 ", 300),
                                     ("a long plan", "Let me plan. ", LONG_BLOCK, 3)):
            with self.subTest(name):
                got = first_loop(w(head) + w(block) * n)
                self.assertIsNotNone(got)
                self.assertGreaterEqual(got[1].repeats + got[1].period, LG.MIN_SPAN)

    def test_legit_repetition_does_not_fire(self):
        for name, text in LEGIT.items():
            with self.subTest(name):
                self.assertIsNone(first_loop(Words()(text)))
                self.assertIsNone(first_loop(list(text.encode())))     # byte by byte, as the tests' tokenizer


class Policy(unittest.TestCase):
    loop = list(range(100, 120)) * 12

    def fire(self, g, ids):
        return next((a for a in (g.push(t) for t in ids) if a is not None), None)

    def test_answer_penalty_then_stop(self):
        g = LG.LoopGuard()
        a = self.fire(g, self.loop)
        self.assertEqual((a.kind, a.where, a.period), (LG.PENALTY, "answer", 20))
        g.resume(list(range(100, 120)), a)
        self.assertEqual(self.fire(g, self.loop).kind, LG.STOP)

    def test_thinking_penalties_then_close(self):
        g = LG.LoopGuard(in_think=True)
        kinds = []
        for _ in range(4):
            a = self.fire(g, self.loop * 2)
            kinds.append(a.kind)
            if a.kind != LG.PENALTY:
                break
            g.resume([], a)
        self.assertEqual(kinds, [LG.PENALTY] * LG.THINK_PENALTIES + [LG.CLOSE_THINKING])
        # a model whose thinking the server cannot close (Harmony): never closed, never stopped
        g = LG.LoopGuard(in_think=True, can_close=False)
        for _ in range(5):
            a = self.fire(g, self.loop * 2)
            self.assertEqual(a.kind, LG.PENALTY)
            g.resume([], a)

    def test_the_thinking_followed_through_the_text(self):
        tok = ByteTokenizer()
        g = LG.LoopGuard(in_think=True, pieces=lambda: Detokenizer(tok).push)
        for t in tok.encode("planning\n</think>\n\n"):
            g.push(t)
        self.assertFalse(g.in_think)
        g.rewind(tok.encode("still planning"))                   # what was kept: back in the thinking
        self.assertTrue(g.in_think)

    def test_never_inside_a_tool_call(self):
        tok = ByteTokenizer()
        call = '<tool_call>\n<function=read>\n<parameter=path>\nsrc/a.py\n</parameter>\n</function>\n</tool_call>\n'
        g = LG.ToolSafeGuard(pieces=lambda: Detokenizer(tok).push)
        self.assertIsNone(self.fire(g, tok.encode(call * 6)))      # whole calls again and again: left alone
        g = LG.ToolSafeGuard(pieces=lambda: Detokenizer(tok).push)
        inner = '<tool_call>\n<function=write>\n<parameter=content>\n' + "0 1 2 3 4 5 6 7 8 9\n" * 30
        a = self.fire(g, tok.encode(inner))
        self.assertEqual((a.kind, a.drop, a.where), (LG.PENALTY, 0, "tool call"))   # a loop in one call: penalty

    def test_penalty_sampling(self):
        g = LG.LoopGuard()
        a = self.fire(g, self.loop)
        g.resume([1] * 50, a)
        smp, n = g.sampling({"temperature": 0.7, "repetition_penalty": 1.3}, 50, 4000)
        self.assertEqual((smp["repetition_penalty"], smp["penalty_last_n"], smp["temperature"]), (1.3, 128, 0.7))
        self.assertEqual(n, 2 * 20 + LG.PENALTY_EXTRA)
        self.assertEqual(g.sampling({"top_k": 5}, 50 + n, 4000 - n), ({"top_k": 5}, 4000 - n))


class LoopingEngine:
    """Writes `head` and then `block` until its limit; with a repetition penalty (`steered`) or once the prompt holds
    a closed thinking (`after_close`) it writes `answer` and the end of the turn instead."""

    def __init__(self, tok, head, block, answer, steered=True):
        self.tok, self.head, self.block, self.answer, self.steered = tok, head, block, answer, steered
        self.calls: list[tuple[list[int], int, dict]] = []
        self.max_context = 100000
        self.start = None

    def generate(self, ids, max_new, sampling, cancel, **kw):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        if self.start is None:
            self.start = len(ids)
        written = self.tok.decode(ids[self.start:])
        if (self.steered and (sampling or {}).get("repetition_penalty", 1) > 1) or "</think>" in written:
            out = self.tok.encode(self.answer) + self.tok.encode(IM_END, parse_special=True)
        else:
            text = "" if written else self.head
            out = self.tok.encode(text + self.block * 400)
        yield from out[:max_new]


class Serving(unittest.TestCase):
    block = "Wait, I should check the second step again. "

    def run_service(self, engine, thinking, guard=True, max_new=6000):
        tok = engine.tok
        prompt = '<|im_start|>user\nsolve<|im_end|>\n<|im_start|>assistant\n' + ("<think>\n" if thinking else "")
        svc = Service(engine, tok, None)
        svc.answer_room = False
        svc.loop_guard = guard
        out = list(svc.run(tok.encode(prompt, parse_special=True), thinking, [], max_new, {}, threading.Event()))
        text = {k: "".join(ev.text or "" for kind, ev in out if kind == "event" and ev.kind == k)
                for k in ("reasoning", "content")}
        return svc, text, out[-1][1]

    def test_an_answer_loop_is_steered_out(self):
        engine = LoopingEngine(ByteTokenizer(), "Answer: ", self.block, "The step is right.")
        svc, text, done = self.run_service(engine, False)
        self.assertEqual(done["finish"], "stop")
        self.assertTrue(text["content"].startswith("Answer: " + self.block))
        self.assertTrue(text["content"].endswith("The step is right."))
        self.assertLess(text["content"].count(self.block), 4)       # the repeats past the held tail are gone
        kept = text["content"][:-len("The step is right.")]
        self.assertEqual(engine.calls[1][0], engine.calls[0][0] + list(kept.encode()))   # on from what was kept
        self.assertEqual(engine.calls[1][2]["repetition_penalty"], LG.LOOP_PENALTY)
        self.assertEqual(svc.loop_guard_counts["penalised"], 1)

    def test_a_loop_that_comes_back_ends_the_answer(self):
        engine = LoopingEngine(ByteTokenizer(), "Answer: ", self.block, "unused", steered=False)
        svc, text, done = self.run_service(engine, False)
        self.assertEqual(done["finish"], "stop")
        self.assertLess(text["content"].count(self.block), 8)
        self.assertEqual(svc.loop_guard_counts["stopped"], 1)

    def test_a_thinking_loop_is_closed_and_answered(self):
        engine = LoopingEngine(ByteTokenizer(), "Let me see. ", self.block, "Four.", steered=False)
        svc, text, done = self.run_service(engine, True)
        self.assertEqual((done["finish"], text["content"].strip()), ("stop", "Four."))
        self.assertEqual(svc.loop_guard_counts, {"detected": 3, "penalised": 2, "thinking_closed": 1, "stopped": 0})

    def test_off_is_the_engine_as_it_writes(self):
        engine = LoopingEngine(ByteTokenizer(), "Answer: ", self.block, "unused")
        _, text, done = self.run_service(engine, False, guard=False, max_new=3000)
        self.assertEqual((done["finish"], len(engine.calls)), ("length", 1))
        self.assertEqual(len(text["content"]), 3000)

    def test_settings(self):
        self.assertTrue(loop_guard_setting({}, env={}))
        self.assertFalse(loop_guard_setting({"loop_guard": False}, env={}))
        self.assertTrue(loop_guard_setting({"loop_guard": False}, env={"XESTRATA_LOOP_GUARD": "1"}))
        self.assertFalse(loop_guard_setting({}, env={"XESTRATA_LOOP_GUARD": "off"}))
        req = {"loop_guard": "false"}
        check_request_sampling(req)
        self.assertIs(req["loop_guard"], False)
        with self.assertRaises(ValueError):
            check_request_sampling({"loop_guard": "maybe"})


if __name__ == "__main__":
    unittest.main()
