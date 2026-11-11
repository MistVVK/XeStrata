"""Tests for tools/force_check.py's neutral-writer gate (cut, gen-neutral, gate_report), without an engine or a GPU: a stand-in writer that speaks
llama-server's /health and /completion with token ids, and synthetic scored arms for the two-direction report.

    python -m unittest tools.test_force_check_neutral
"""
from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "force_check.py"
sys.path.insert(0, str(ROOT / "tools"))
import force_check as FC  # noqa: E402

# A llama-server stand-in: -m MODEL --port P ...; /health -> ok; /completion -> n_predict ids derived from the prompt ids (so the test can
# check that the ids went in unchanged), and it records every request it got.
FAKE_WRITER = textwrap.dedent('''
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    model = args[args.index("-m") + 1]
    log = model + ".requests.jsonl"
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send({"status": "ok"})
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            open(log, "a").write(json.dumps(req) + "\\n")
            ids = req["prompt"]
            toks = [(sum(ids) + 7 * i) % 1000 for i in range(req["n_predict"])]
            self._send({"content": "", "tokens": toks, "timings": {"prompt_per_second": 1.0}})
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def synthetic_arm(rng, p_mis, p_big, nreq=48, n=128):
    d = {}
    for q in range(nreq):
        rows = []
        for _ in range(n):
            if rng.random() < p_mis:
                big = rng.random() < p_big
                rows.append((True, rng.uniform(1.0, 3.0) if big else rng.uniform(0.0, 0.4), 1.0))
            else:
                rows.append((False, 0.0, 2.0))
        d["r%d" % q] = rows
    return d


class Cut(unittest.TestCase):
    def test_cuts_at_the_fixed_offsets(self):
        with tempfile.TemporaryDirectory() as t:
            ids = list(range(1000, 1000 + 300))
            Path(t, "ids.txt").write_text(",".join(map(str, ids)))
            r = subprocess.run([sys.executable, str(TOOL), "cut", str(Path(t, "ids.txt")), str(Path(t, "c.jsonl")), "--sizes", "50,100", "--n", "3"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [json.loads(l) for l in Path(t, "c.jsonl").read_text().splitlines()]
            self.assertEqual([x["id"] for x in rows], ["p50-0", "p50-1", "p50-2", "p100-0", "p100-1", "p100-2"])
            for x in rows:
                size, i = (int(v) for v in x["id"][1:].split("-"))
                off = (i * 7919 + size) % (len(ids) - size - 1)
                self.assertEqual(x["ids"], ids[off: off + size])
                self.assertEqual(x["split"], "held")

    def test_a_cut_longer_than_the_file_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            Path(t, "ids.txt").write_text(",".join(map(str, range(40))))
            r = subprocess.run([sys.executable, str(TOOL), "cut", str(Path(t, "ids.txt")), str(Path(t, "c.jsonl")), "--sizes", "64", "--n", "1"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)


class GenNeutral(unittest.TestCase):
    def test_the_writer_gets_the_ids_and_its_tokens_become_the_set(self):
        with tempfile.TemporaryDirectory() as t:
            fake = Path(t, "fake_writer.py")
            fake.write_text(FAKE_WRITER)
            model = Path(t, "model.gguf")
            model.write_text("not a model")
            corpus = Path(t, "c.jsonl")
            rows = [{"id": "p8-%d" % i, "kind": "p8", "split": "held", "ids": [i, i + 1, i + 2, 5, 6, 7, 8, 9]} for i in range(3)]
            rows.append({"id": "x", "kind": "p8", "split": "train", "ids": [1, 2]})   # not held: left out
            corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = Path(t, "set.json")
            writer = '"%s" "%s"' % (sys.executable, fake) if os.name == "nt" else "%s %s" % (sys.executable, fake)
            r = subprocess.run([sys.executable, str(TOOL), "gen-neutral", str(out), "--corpus", str(corpus), "--writer", writer,
                                "--writer-model", str(model), "--writer-port", str(free_port()), "--per-kind", "16", "--new", "5",
                                "--writer-wait", "60"], capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            st = json.loads(out.read_text())
            self.assertEqual(st["writer"], "neutral")
            self.assertEqual([it["id"] for it in st["items"]], ["p8-0", "p8-1", "p8-2"])
            for it, row in zip(st["items"], rows):
                self.assertEqual(it["prompt"], row["ids"])
                self.assertEqual(it["cont"], [(sum(row["ids"]) + 7 * i) % 1000 for i in range(5)])
            reqs = [json.loads(l) for l in Path(str(model) + ".requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(reqs), 3)
            for q in reqs:   # greedy, token ids in, no prompt cache, ids back
                self.assertEqual((q["temperature"], q["top_k"], q["cache_prompt"], q["return_tokens"], q["n_predict"]), (0.0, 1, False, True, 5))
                self.assertIsInstance(q["prompt"], list)

    def test_a_writer_that_dies_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            corpus = Path(t, "c.jsonl")
            corpus.write_text(json.dumps({"id": "a", "kind": "k", "ids": [1, 2, 3]}) + "\n")
            dead = Path(t, "dead.py")
            dead.write_text("import sys; sys.exit(3)\n")
            writer = '"%s" "%s"' % (sys.executable, dead) if os.name == "nt" else "%s %s" % (sys.executable, dead)
            r = subprocess.run([sys.executable, str(TOOL), "gen-neutral", str(Path(t, "s.json")), "--corpus", str(corpus), "--writer", writer,
                                "--writer-model", str(Path(t, "m.gguf")), "--writer-port", str(free_port()), "--writer-wait", "30"],
                               capture_output=True, text=True, timeout=120)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("writer ended before it was ready", r.stdout + r.stderr)


class GateReport(unittest.TestCase):
    def test_equal_arms_pass_and_both_directions_are_reported(self):
        rng = random.Random(11)
        stock = [synthetic_arm(rng, 0.035, 0.0), synthetic_arm(rng, 0.035, 0.0)]
        cand = [synthetic_arm(rng, 0.035, 0.0), synthetic_arm(rng, 0.035, 0.0)]
        r = FC.gate_report(stock, cand)
        self.assertNotEqual(r["verdict"], "REJECT")
        self.assertEqual(len(r["cand_vs_stock"]), 2)
        self.assertEqual(len(r["stock_vs_cand_reported"]), 2)

    def test_a_worse_candidate_is_rejected(self):
        rng = random.Random(12)
        stock = [synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300), synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300)]
        cand = [synthetic_arm(rng, 0.02, 0.0, nreq=60, n=300), synthetic_arm(rng, 0.02, 0.0, nreq=60, n=300)]
        self.assertEqual(FC.gate_report(stock, cand)["verdict"], "REJECT")

    def test_the_reversed_direction_does_not_decide(self):
        rng = random.Random(13)
        stock = [synthetic_arm(rng, 0.02, 0.0, nreq=60, n=300), synthetic_arm(rng, 0.02, 0.0, nreq=60, n=300)]
        cand = [synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300), synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300)]
        r = FC.gate_report(stock, cand)   # the candidate is better: it passes; stock against it would be rejected
        self.assertEqual(r["verdict"], "PASS")
        self.assertTrue(all(x["verdict"] == "REJECT" for x in r["stock_vs_cand_reported"]))

    def test_one_rejected_candidate_arm_rejects(self):
        rng = random.Random(14)
        stock = [synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300), synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300)]
        cand = [synthetic_arm(rng, 0.007, 0.0, nreq=60, n=300), synthetic_arm(rng, 0.0075, 0.5, nreq=60, n=300)]
        self.assertEqual(FC.gate_report(stock, cand)["verdict"], "REJECT")


class GateRefusesEngineText(unittest.TestCase):
    def test_a_stock_written_set_is_refused_by_gate(self):
        with tempfile.TemporaryDirectory() as t:
            Path(t, "s.json").write_text(json.dumps({"config": "x", "items": []}))
            Path(t, "cfg.json").write_text(json.dumps({"exe": "none", "args": []}))
            r = subprocess.run([sys.executable, str(TOOL), "gate", str(Path(t, "cfg.json")), str(Path(t, "s.json")), str(Path(t, "o")),
                                "--cand-env", "STRATA_X=1"], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("neutral writer", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
