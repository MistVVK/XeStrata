"""The thinking budget's hold (GEN hold=N:ID, lane AGENT-TURN P2): the budget's second engine pass continues from the
live session instead of reading the whole reasoning again.

The stub engine below behaves like the real one where it matters: it decodes in windows of several tokens, commits a
window before it sends its tokens, sees a STOP only at a window's end (and can run one more window before it does), and
keeps as its live session the prompt plus every token it sent but the last.  A prompt that the live session is not a
prefix of falls back to the checkpoint at the previous prompt's last <|im_start|>.  With the hold it stops exactly at
the hold and commits nothing past it, as serve_hold_remaining() does in the engine."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import HOLD_KEY, REASONING_WRAP_UP, ByteTokenizer, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ThinkTokenizer(ByteTokenizer):
    """The byte tokenizer with <think> / </think> as one token each, as the real tokenizer has them."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")


class WindowEngine:
    THOUGHT = "Let me think step by step about two plus two. " * 4
    ANSWER = "The answer is 4."

    def __init__(self, tok, windows=(3, 2, 4), overrun=True, can_hold=True, honours_hold=True, thought=None, can_hold_batch=False):
        self.tok, self.max_context = tok, 4096
        self.windows, self.overrun = windows, overrun
        self.can_hold, self.honours_hold = can_hold, honours_hold
        self.can_hold_batch = can_hold_batch   # READY says holdb: the --batch slots end at the hold too (P2b)
        self.batch = self.multi = 0
        self.thought = thought if thought is not None else self.THOUGHT
        self.start = tok.encode("<|im_start|>", parse_special=True)[0]
        self.think_end = tok.encode("</think>", parse_special=True)[0]
        self.end = tok.encode("<|im_end|>", parse_special=True)
        self.wrap = tok.encode(REASONING_WRAP_UP, parse_special=True)
        self.live: list[int] = []
        self.checkpoints: list[int] = []
        self.turn: list[int] | None = None
        self.last: dict = {}
        self.passes: list[dict] = []

    def _model(self, ids):
        """The tokens the model writes after `ids` (greedy, deterministic)."""
        full = self.tok.encode(self.thought + "</think>\n\n" + self.ANSWER, parse_special=True) + self.end
        if self.turn is not None and ids[:len(self.turn)] == self.turn:
            so_far = ids[len(self.turn):]
            if so_far[-len(self.wrap):] == self.wrap:
                return self.tok.encode(self.ANSWER) + self.end
            if full[:len(so_far)] == so_far:
                return full[len(so_far):]
        self.turn = list(ids)
        return full

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        ids = list(ids)
        if self.live and len(self.live) <= len(ids) - 1 and ids[:len(self.live)] == self.live:
            reused = len(self.live)
        else:
            reused = max([c for c in self.checkpoints if c <= len(ids) - 1 and c <= len(self.live)
                          and ids[:c] == self.live[:c]] or [0])
        hold = (sampling or {}).get(HOLD_KEY) if self.honours_hold else None
        row = {"prompt": len(ids), "reused": reused, "read": len(ids) - reused, "hold": hold}
        self.passes.append(row)
        last_start = max(i for i, t in enumerate(ids) if t == self.start)
        out = self._model(ids)
        hold_n = hold[0] if hold else -1
        sent: list[int] = []
        finish = "length"
        w = 0
        try:
            k = 0
            while k < len(out):
                cap = min(max_new, hold_n) if hold_n > 0 else max_new
                if len(sent) >= cap:
                    break
                window = out[k:k + self.windows[w % len(self.windows)]]
                w += 1
                if hold_n > 0:   # the engine's bound: rows up to the hold unless </think> comes first
                    room = cap - len(sent)
                    if self.think_end in window[:room]:
                        room = max_new - len(sent)
                    window = window[:room]
                else:
                    window = window[:max_new - len(sent)]
                k += len(window)
                pending = list(window)
                while pending:
                    if cancel.is_set():
                        finish = "cancel"
                        return
                    t = pending.pop(0)
                    sent.append(t)
                    if t == self.think_end:
                        hold_n = -1
                    if t in self.end:
                        finish = "stop"
                        return
                    yield t
            if hold_n > 0 and len(sent) >= hold_n and len(sent) < max_new:
                finish = "hold"
        except GeneratorExit:
            # the STOP is seen at the window's end: the window was committed and sent whole, and one more window may
            # have run before the STOP arrived
            sent.extend(pending)
            if self.overrun and k < len(out):
                more = out[k:k + self.windows[w % len(self.windows)]]
                if hold_n > 0:   # with the hold the engine never runs past it (its bound ends the request there)
                    more = more[:max(0, hold_n - len(sent))]
                sent.extend(more)
            finish = "hold" if hold_n > 0 and len(sent) >= hold_n else "cancel"
            raise
        finally:
            self.live = ids + sent[:-1] if sent else ids[:-1]
            if last_start not in self.checkpoints:
                self.checkpoints.append(last_start)
            row.update(sent=len(sent), finish=finish)
            self.last = {"finish": finish, "reused": reused, "prompt_read": len(ids) - reused, "generated": len(sent)}


class HoldCase(unittest.TestCase):
    def start(self, **kw):
        self.tok = ThinkTokenizer()
        self.engine = WindowEngine(self.tok, **kw)
        svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def chat(self, **extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 400, **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            msg = json.loads(r.read().decode())["choices"][0]["message"]
        return msg.get("reasoning_content"), msg.get("content")

    def reply(self, budget, **kw):
        self.start(**kw)
        return self.chat(reasoning_budget_tokens=budget), self.engine.passes

    def expected(self, budget, thought=WindowEngine.THOUGHT):
        return thought[:budget] + REASONING_WRAP_UP.split("</think>")[0], WindowEngine.ANSWER


class ThinkingHold(HoldCase):
    def test_without_the_hold_the_second_pass_reads_the_reasoning_again(self):
        """Today's behaviour, kept for an engine that does not say "hold": the engine ran past the cut, so the second
        pass falls back to the checkpoint before the assistant turn and reads the reasoning again."""
        got, passes = self.reply(20, can_hold=False)
        self.assertEqual(got, self.expected(20))
        self.assertEqual(len(passes), 2)
        self.assertIsNone(passes[0]["hold"])
        wrap = len(self.tok.encode(REASONING_WRAP_UP, parse_special=True))
        self.assertGreater(passes[1]["read"], 20 + wrap)

    def test_the_hold_makes_the_second_pass_continue_from_the_live_session(self):
        got, passes = self.reply(20)
        self.assertEqual(got, self.expected(20))
        self.assertEqual(len(passes), 2)
        think_end = self.tok.encode("</think>", parse_special=True)[0]
        self.assertEqual(passes[0]["hold"], (20, think_end))
        self.assertEqual((passes[0]["sent"], passes[0]["finish"]), (20, "hold"))
        wrap = len(self.tok.encode(REASONING_WRAP_UP, parse_special=True))
        # all but the last reasoning token is reused; the read is that token and the wrap-up
        self.assertEqual(passes[1]["reused"], passes[0]["prompt"] + 20 - 1)
        self.assertEqual(passes[1]["read"], 1 + wrap)
        self.assertIsNone(passes[1]["hold"])   # the budget is spent: the answer pass has none

    def test_same_answer_with_and_without_the_hold_for_every_budget(self):
        for budget in (1, 2, 5, 19, 20, 21, 100, 183):
            with self.subTest(budget=budget):
                got_hold, _ = self.reply(budget)
                got_old, _ = self.reply(budget, can_hold=False)
                self.assertEqual(got_hold, got_old)
                self.assertEqual(got_hold, self.expected(budget))

    def test_a_cut_that_is_not_clean_at_the_hold_takes_one_more_matching_pass(self):
        """A character split across tokens at the hold: the server does not wrap there; the next pass (a hold of one
        more token) continues from the live session, and the wrap-up comes at the same clean point as without the hold."""
        thought = "Two plus two: " + "é" * 10 + " is four. " * 8   # two bytes per e-acute
        budget = len("Two plus two: ".encode()) + 3                       # inside the third e-acute
        got_hold, passes = self.reply(budget, thought=thought)
        got_old, _ = self.reply(budget, thought=thought, can_hold=False)
        self.assertEqual(got_hold, got_old)
        self.assertEqual(len(passes), 3)
        self.assertEqual(passes[0]["finish"], "hold")
        self.assertEqual(passes[1]["hold"][0], 1)
        self.assertEqual(passes[1]["reused"], passes[0]["prompt"] + passes[0]["sent"] - 1)
        self.assertEqual(passes[1]["read"], 1)
        self.assertEqual(passes[2]["reused"], passes[1]["prompt"] + passes[1]["sent"] - 1)

    def test_thinking_that_ends_before_the_budget_runs_in_one_pass(self):
        got, passes = self.reply(500)
        self.assertEqual(got, (WindowEngine.THOUGHT, WindowEngine.ANSWER))
        self.assertEqual(len(passes), 1)
        self.assertEqual(passes[0]["finish"], "stop")

    def test_an_engine_that_says_hold_but_ignores_it_gives_the_same_answer(self):
        """A new server with an engine that does not act on the key: the server cuts and STOPs as before."""
        got, passes = self.reply(20, honours_hold=False)
        self.assertEqual(got, self.expected(20))
        self.assertEqual(len(passes), 2)

    def test_no_budget_no_hold(self):
        self.start()
        self.assertEqual(self.chat(), (WindowEngine.THOUGHT, WindowEngine.ANSWER))
        self.assertEqual(len(self.engine.passes), 1)
        self.assertIsNone(self.engine.passes[0]["hold"])

    def test_a_batch_engine_whose_slots_do_not_say_holdb_gets_no_hold(self):
        self.start()
        self.engine.batch = 4
        self.assertEqual(self.chat(reasoning_budget_tokens=20), self.expected(20))
        self.assertTrue(all(p["hold"] is None for p in self.engine.passes))

    def test_new_server_old_engine_no_token_no_hold_for_a_batch_engine(self):
        """Mixed versions, new server + old engine: an engine that says neither hold nor holdb (before P2), and one that says hold but
        not holdb (P2 without P2b), get no key when they run --batch; the budget then cuts and STOPs as it always did."""
        for kw in ({"can_hold": False}, {"can_hold": True}, {"can_hold": False, "can_hold_batch": True}):
            with self.subTest(**kw):
                self.start(**kw)
                self.engine.batch = 4
                self.assertEqual(self.chat(reasoning_budget_tokens=20), self.expected(20))
                self.assertTrue(all(p["hold"] is None for p in self.engine.passes))

    def test_a_batch_engine_that_says_holdb_gets_the_hold(self):
        """P2b: a --batch engine whose slots end at the hold (READY holdb) gets the key like a solo engine: the second pass reads the last
        reasoning token and the wrap-up, as in the solo case."""
        self.start(can_hold_batch=True)
        self.engine.batch = 4
        self.assertEqual(self.chat(reasoning_budget_tokens=20), self.expected(20))
        passes = self.engine.passes
        self.assertEqual(len(passes), 2)
        self.assertEqual(passes[0]["hold"], (20, self.tok.encode("</think>", parse_special=True)[0]))
        wrap = len(self.tok.encode(REASONING_WRAP_UP, parse_special=True))
        self.assertEqual((passes[1]["reused"], passes[1]["read"]), (passes[0]["prompt"] + 20 - 1, 1 + wrap))

    def test_holdb_does_not_matter_without_batch_and_a_multi_engine_gets_none(self):
        self.start(can_hold_batch=True)
        self.engine.multi = 2
        self.assertEqual(self.chat(reasoning_budget_tokens=20), self.expected(20))
        self.assertTrue(all(p["hold"] is None for p in self.engine.passes))


class HoldKey(unittest.TestCase):
    def test_the_gen_key(self):
        self.assertEqual(StrataEngine.sampling_keys({HOLD_KEY: (512, 248069)}), " hold=512:248069")

    def test_only_the_servers_own_value_is_sent(self):
        for v in ([512, 1], (0, 1), (5, -1), (5,), ("5", 1), (True, 1), None, 7):
            with self.subTest(v=v):
                self.assertNotIn("hold=", StrataEngine.sampling_keys({HOLD_KEY: v}))


if __name__ == "__main__":
    unittest.main()
