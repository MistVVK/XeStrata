#!/usr/bin/env python3
"""Streamed output must reach only the client that asked for it (#4451, #2892).

Many interleaved streaming clients, each with its own random canary in its prompt and the model asked to repeat it.
Some clients disconnect at random points (before the first byte, during the prompt read, after a few chunks).  Every
text a client receives (content and reasoning) is searched for canaries: any canary that is not the client's own is a
cross-delivery.  Run against any OpenAI-compatible Strata server; the server must be one you started (check the
listener first on a shared box).

    python research/tools/live-probes-2026-10/stream_ownership_probe.py --url http://127.0.0.1:8080 --clients 6 --rounds 20 --seed 1 --out result.json

Exit code 0: no cross-delivery.  1: at least one.  2: the run itself failed (no server, no completed client).
Nothing here sends anything but chat requests; it never reads another client's connection.
"""
import argparse
import http.client
import json
import random
import re
import sys
import threading
import time
import urllib.parse

CANARY = re.compile(r"\b[A-Z]{3}-\d{4}\b")
FILLER_WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et "
                "dolore magna aliqua").split()


def make_canary(rnd: random.Random, used: set) -> str:
    while True:
        c = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + "-" + f"{rnd.randrange(10000):04d}"
        if c not in used:
            used.add(c)
            return c


def prompt_for(canary: str, filler_words: int, rnd: random.Random) -> str:
    filler = " ".join(rnd.choice(FILLER_WORDS) for _ in range(filler_words))
    head = (filler + "\n\n") if filler else ""
    return (f"{head}Your code word is {canary}. Write the code word {canary} {rnd.choice((20, 30, 40))} times, "
            "separated by single spaces, and write nothing else.")


USED: set = set()          # every canary this run has issued: a canary-looking string the model made up is not a cross-delivery


class Result:
    def __init__(self, canary, plan):
        self.canary, self.plan = canary, plan
        self.content, self.reasoning, self.chunks, self.error, self.finished, self.status = "", "", 0, None, False, None

    @property
    def text(self):
        return self.reasoning + "\n" + self.content

    def foreign(self):
        return sorted({c for c in CANARY.findall(self.text) if c != self.canary and c in USED})


def one_client(host, port, body, res: Result, stop_after_chunks, stop_after_s, timeout):
    """Streams the reply; leaves (closes the socket) after `stop_after_chunks` chunks or `stop_after_s` seconds."""
    t0 = time.monotonic()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        if stop_after_s is not None and stop_after_chunks is None:      # leave before any answer
            time.sleep(stop_after_s)
            return
        resp = conn.getresponse()
        res.status = resp.status
        if resp.status != 200:
            res.error = f"HTTP {resp.status}"
            resp.read()
            return
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                res.finished = True
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                res.content += d.get("content") or ""
                res.reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
                res.chunks += 1
            if stop_after_chunks is not None and res.chunks >= stop_after_chunks:
                return
            if stop_after_s is not None and time.monotonic() - t0 >= stop_after_s:
                return
    except Exception as e:                       # noqa: BLE001 - a dropped connection is a result, not a crash
        res.error = f"{type(e).__name__}: {e}"
    finally:
        try:
            conn.close()
        except Exception:                        # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--clients", type=int, default=6, help="interleaved clients per round")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--disconnect", type=float, default=0.5, help="probability that a client leaves early")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--thinking", type=float, default=0.5, help="share of clients that ask for reasoning (effort 'low')")
    ap.add_argument("--filler-max", type=int, default=3000, help="at most this many filler words in a prompt (a longer prompt read)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    host, port = u.hostname, u.port or 80
    rnd = random.Random(a.seed)
    used, results, started = USED, [], time.time()
    for rd in range(a.rounds):
        threads, batch = [], []
        for k in range(a.clients):
            canary = make_canary(rnd, used)
            leave = rnd.random() < a.disconnect
            plan = {"round": rd, "leave": leave}
            filler = rnd.choice((0, 0, 200, a.filler_max)) if a.filler_max else 0
            body = {"messages": [{"role": "user", "content": prompt_for(canary, filler, rnd)}], "stream": True,
                    "max_tokens": a.max_tokens, "reasoning_effort": "low" if rnd.random() < a.thinking else "none"}
            plan.update(filler_words=filler, effort=body["reasoning_effort"])
            chunks = after_s = None
            if leave:
                mode = rnd.choice(("before_answer", "early", "mid"))
                plan["mode"] = mode
                if mode == "before_answer":
                    after_s = rnd.choice((0.05, 0.3, 1.0, 2.5))
                elif mode == "early":
                    chunks = rnd.randrange(1, 4)
                else:
                    chunks = rnd.randrange(5, 40)
            res = Result(canary, plan)
            batch.append(res)
            t = threading.Thread(target=one_client, args=(host, port, body, res, chunks, after_s, a.timeout), daemon=True)
            threads.append(t)
        for t in threads:
            t.start()
            time.sleep(rnd.choice((0.0, 0.02, 0.1)))
        for t in threads:
            t.join(a.timeout + 30)
        results.extend(batch)
        bad = [r for r in batch if r.foreign()]
        print(f"round {rd + 1}/{a.rounds}: {len(batch)} clients, {sum(1 for r in batch if r.finished)} finished, "
              f"{sum(1 for r in batch if r.plan['leave'])} left early, {len(bad)} with a foreign canary", flush=True)
    crossed = [r for r in results if r.foreign()]
    finished = [r for r in results if r.finished]
    # a finished client of a plain request should have its own canary in its text; a missing one is not a cross-delivery
    # but is reported (the model did not follow the instruction, or the reply was empty / replaced)
    own_missing = [r for r in finished if r.canary not in r.text]
    summary = {"clients": len(results), "finished": len(finished), "left_early": sum(1 for r in results if r.plan["leave"]),
               "errors": sum(1 for r in results if r.error and r.plan["leave"] is False),
               "cross_delivery": len(crossed), "finished_without_own_canary": len(own_missing),
               "seconds": round(time.time() - started, 1), "seed": a.seed,
               "crossed": [{"canary": r.canary, "foreign": r.foreign(), "plan": r.plan, "excerpt": r.text[:300]}
                           for r in crossed[:20]],
               "samples": [{"canary": r.canary, "plan": r.plan, "chunks": r.chunks, "excerpt": r.text[:200]}
                           for r in (own_missing[:3] + finished[:3])]}
    print(json.dumps({k: v for k, v in summary.items() if k not in ("crossed", "samples")}), flush=True)
    for c in summary["crossed"]:
        print("CROSS-DELIVERY", json.dumps(c), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    if not finished:
        return 2
    return 1 if crossed else 0


if __name__ == "__main__":
    sys.exit(main())
