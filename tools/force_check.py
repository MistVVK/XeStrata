#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""force_check.py: the token-forcing check after SpecPilot (arXiv 2611.02501, Li et al., Apache-2.0).

A candidate engine/build/setting is judged on a FIXED text: the engine scores prompt + continuation teacher-forced
(every position's head logits come out of the verify windows with the head in FP32: XESTRATA_LOGPOS /
XESTRATA_LOGPOS_TOPK, the prompt cache off), and the checker counts where the scorer's argmax is not the forced token.
Nothing here depends on a drafter; it serves any class N change (Q5_0 experts, tree verify, a new kernel ...).

  force_check.py gen   CFG SET.json --corpus held.jsonl [--kinds code,json] [--per-kind 4] [--offset 8] [--new 300]
[--env K=V ...]         one engine with CFG's normal decode flags; greedy continuations of the corpus prompts (rows
with "ids") -> SET.json   force_check.py score CFG SET.json OUT.tsv [--path decode|prompt] [--env K=V ...] [--arg "--
flag v" ...]         --path decode (default): one engine with CFG's own flags (drafter, spec windows, adaptive tiers
...), prompt cache off; every request decodes the         forced continuation: the verify windows take the forced
token in place of the argmax (XESTRATA_FOLLOW), so drafts are accepted when they match it,         and the FP32 head
logits of every valid row come out (XEXESTRATA_FOLLOW_LOGPOS, top-2).  --path prompt: the whole text is read as a
prompt through the         windows (XESTRATA_LOGPOS, --short-read), no decode features.  OUT.tsv.meta.json = per item
line ranges.         --env XESTRATA_FORCE_FAULT=7:0.3 is the test fault (adds 0.3 to every 7th token's logit in the
scored rows).   force_check.py check --stock A.tsv B.tsv [...] --cand C.tsv [--delta D] [--json out.json]         the
two rules of the paper against the stock arms (>= 2 stock arms: the noise): exit 0 pass, 1 reject, 2 inconclusive
force_check.py selftest      the statistics on synthetic rows (no engine)

An "arm" = one scored TSV.  Which engine produced the continuation and which scored it is the caller's choice:   - a
change of the scorer (a different expert format, kernel, KV type): same SET.json, `score` with the candidate
build/flags;   - a change of the decode path (tree verify, a drafter): `gen` with the candidate, `score` with the
STOCK scorer (the paper's way). Stock arms = the same procedure with the stock build, at least twice (a second `gen`,
or a second `score`).

Rules (paper):  mismatch = the scorer's argmax is not the forced token; margin = log p(argmax) - log p(forced token).
rate rule : candidate mismatch rate within +0.5 points of stock at 95% (one-sided bootstrap over requests, paired on
the same positions).               REJECT if the 95% lower bound of (cand - stock) > 0.5 points; PASS if the 95% upper
bound <= 0.5 points; else INCONCLUSIVE (more tokens).   event rule: delta = the largest margin among stock's first
departures per request, rounded UP to a multiple of 1/8; a large-margin event is a               mismatch with margin
> 2 delta.  REJECT if the candidate has more events than stock by an exact one-sided binomial test (p < 0.05,
exposure-weighted).  (The paper also audits events stock reproduces on its own paths; that audit is not done here,
events are counted raw.)"""
import argparse
import json
import math
import os
import pathlib
import random
import signal
import subprocess
import sys

MEASURED = {"XESTRATA_LOGPOS", "XESTRATA_LOGPOS_TOPK", "XESTRATA_FOLLOW", "XESTRATA_FOLLOW_LOGPOS",
            "XESTRATA_ROUTE_TAIL_SKIP", "XESTRATA_FORCE_FAULT"}

SCORER_DROP_VALUE = {"--draft", "--draft-model", "--draft-fallback", "--mtp-draft-vocab", "--spec", "--draft-conf", "--draft-cap", "--mtp", "--spec-min-p",  # noqa: E501
                     "--prompt-cache", "--max-context", "--prefill", "--short-read", "--prompt-cache-every", "--conversation-cache-mib", "--suffix-draft"}  # noqa: E501
SCORER_DROP_FLAG = {"--serve", "--vision"}


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, cfg, args, env_extra, log):
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("STRATA_") or k in MEASURED:
                env.pop(k)
        env.update(cfg.get("env", {}))
        env.update(env_extra)
        if cfg.get("lib_dirs"):
            env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
        self.errf = open(log, "wb")
        self.p = subprocess.Popen([cfg["exe"], "--serve"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errf, text=True,  # noqa: E501
                                  cwd=cfg.get("cwd"), env=env, bufsize=1, start_new_session=True)

    def ready(self):
        for line in self.p.stdout:
            if line.startswith("READY"):
                return True
        return False

    def gen(self, ids, n):
        self.p.stdin.write("GEN %d %s\n" % (n, ",".join(map(str, ids))))
        self.p.stdin.flush()
        toks = []
        for line in self.p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return toks, line.split()
            elif line.startswith("ERR"):
                raise SystemExit("engine: " + line.strip())
        raise SystemExit("the engine closed its output")

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=300)
        except Exception:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)   # only our own session
            except Exception:
                self.p.kill()
        self.errf.close()


def fresh_profile(cfg, args, tag):
    """cfg "profile_src": a fresh copy of the expert profile per engine start (the engine may rewrite it), and the
    args with the copy's path (upstream 13eedf552)."""
    src = cfg.get("profile_src")
    if not src or "--expert-profile" not in args:
        return args
    import shutil
    dst = os.path.abspath(tag) + ".profile.bin"
    shutil.copyfile(src, dst)
    args = list(args)
    args[args.index("--expert-profile") + 1] = dst
    return args


def scorer_args(cfg, total_tokens, extra):
    base, out, i = list(cfg["args"]), [], 0
    while i < len(base):
        x = base[i]
        if x in SCORER_DROP_FLAG:
            i += 1
        elif x in SCORER_DROP_VALUE:
            i += 2
        else:
            out.append(x)
            i += 1
    return out + ["--prompt-cache", "0", "--max-context", str(total_tokens + 512), "--short-read", str(total_tokens + 64), "--prefill", "512"] + extra  # noqa: E501


def kv_env(lst):
    d = {}
    for kv in lst or []:
        k, _, v = kv.partition("=")
        d[k] = v
    return d


# ------------------------------------------------------------------------------------------------ gen / score
def cmd_gen(a):
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    rows = [json.loads(ln) for ln in open(a.corpus) if ln.strip()]
    rows = [r for r in rows if r.get("split", "held") == "held"]
    kinds = a.kinds.split(",") if a.kinds else sorted({r["kind"] for r in rows})
    sel = []
    for k in kinds:
        sel += [r for r in rows if r["kind"] == k][a.offset:a.offset + a.per_kind]
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit("only %.0f GB available (< %d): not starting an engine" % (mem_available_gb(), a.min_free_gb))
    eng = Engine(cfg, fresh_profile(cfg, list(cfg["args"]), a.out), kv_env(a.env), a.out + ".engine.log")
    items = []
    try:
        if not eng.ready():
            raise SystemExit("the engine ended before READY, see %s.engine.log" % a.out)
        for r in sel:
            toks, f = eng.gen(r["ids"], a.new)
            items.append({"id": r["id"], "kind": r["kind"], "prompt": r["ids"], "cont": toks})
            print("%s: %d prompt, %d forced tokens" % (r["id"], len(r["ids"]), len(toks)), flush=True)
    finally:
        eng.close()
    json.dump({"config": a.config, "env": a.env or [], "items": items}, open(a.out, "w"))
    return 0


def cmd_score(a):
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.set))
    items = st["items"]
    total = max(len(i["prompt"]) + len(i["cont"]) for i in items)
    out = pathlib.Path(a.out).resolve()
    if out.exists():
        out.unlink()
    env = kv_env(a.env)
    extra = [x for s in (a.arg or []) for x in s.split()]
    if a.path == "decode":
        # the real decode/verify windows (drafter, spec windows, adaptive tiers, whatever the config has) with the forced continuation standing in  # noqa: E501
        # for the argmax (XESTRATA_FOLLOW, one line per request in decode order); rows come from XEXESTRATA_FOLLOW_LOGPOS  # noqa: E501
        follow = str(out) + ".follow.txt"
        with open(follow, "w") as ff:
            for it in items:
                ff.write(",".join(map(str, it["cont"])) + chr(10))
        env.update({"XESTRATA_FOLLOW": follow, "XEXESTRATA_FOLLOW_LOGPOS": str(out), "XESTRATA_LOGPOS_TOPK": "2"})
        args = list(cfg["args"])
        # XeStrata: the forcing is in the serial decode/verify window loop (a pipelined window is prepared ahead of the
        # verdict it belongs to), so the decode path runs without --pipeline-windows
        if "--pipeline-windows" in args:
            i = args.index("--pipeline-windows")
            del args[i:i + 2]
        args += ["--pipeline-windows", "0"]
        if "--prompt-cache" in args:
            i = args.index("--prompt-cache")
            del args[i:i + 2]
        args += ["--prompt-cache", "0"] + extra
    else:
        env.update({"XESTRATA_LOGPOS": str(out), "XESTRATA_LOGPOS_TOPK": "2", "XESTRATA_ROUTE_TAIL_SKIP": env.get("XESTRATA_ROUTE_TAIL_SKIP", "0")})  # noqa: E501
        args = scorer_args(cfg, total, extra)
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit("only %.0f GB available (< %d): not starting an engine" % (mem_available_gb(), a.min_free_gb))
    args = fresh_profile(cfg, args, str(out))
    eng = Engine(cfg, args, env, str(out) + ".engine.log")

    def nlines():
        return sum(1 for _ in open(out)) if out.exists() else 0
    meta = []
    try:
        if not eng.ready():
            raise SystemExit("the engine ended before READY, see %s.engine.log" % out)
        for it in items:
            ids = it["prompt"] + it["cont"]
            n0 = nlines()
            if a.path == "decode":
                toks, f = eng.gen(it["prompt"], len(it["cont"]))
                if toks != it["cont"]:
                    print("%s: WARNING the engine emitted %d tokens, %d differ from the forced continuation" % (
                        it["id"], len(toks), sum(1 for x, y in zip(toks, it["cont"]) if x != y) + abs(len(toks) - len(it["cont"]))), flush=True)  # noqa: E501
            else:
                toks, f = eng.gen(ids, 1)
            n1 = nlines()
            if n1 - n0 < len(it["cont"]) // 2:
                raise SystemExit("%s: only %d positions scored (the prompt took the batched path? raise --short-read; decode path: XESTRATA_FOLLOW not in this build?)" % (it["id"], n1 - n0))  # noqa: E501
            meta.append({"id": it["id"], "kind": it["kind"], "first_line": n0, "end_line": n1, "prompt_len": len(it["prompt"]), "ids": ids})  # noqa: E501
            print("%s: %d positions" % (it["id"], n1 - n0), flush=True)
    finally:
        eng.close()
    json.dump({"set": a.set, "config": a.config, "path": a.path, "env": a.env or [], "args": args, "items": meta}, open(str(out) + ".meta.json", "w"))  # noqa: E501
    return 0


# ------------------------------------------------------------------------------------------------ check
def load_arm(tsv):
    """-> {request id: [(is_mismatch, margin, top2_margin), ...]} over the continuation positions"""
    meta = json.load(open(tsv + ".meta.json"))
    lines = open(tsv).read().splitlines()
    out = {}
    for it in meta["items"]:
        ids, P = it["ids"], it["prompt_len"]
        rows = []
        for ln in lines[it["first_line"]:it["end_line"]]:
            f = ln.split("\t")
            pos, tgt = int(f[0]), int(f[1])
            if pos < P - 1 or pos + 1 >= len(ids):
                continue
            if ids[pos + 1] != tgt:
                raise SystemExit("%s: line for position %d scores token %d, the set has %d" % (it["id"], pos, tgt, ids[pos + 1]))  # noqa: E501
            lp_t, am, lp_am = float(f[2]), int(f[3]), float(f[4])
            lp2 = None
            for x in f[8:10]:
                i, _, v = x.partition(":")
                if int(i) != am:
                    lp2 = float(v)
                    break
            rows.append((am != tgt, max(0.0, lp_am - lp_t), (lp_am - lp2) if lp2 is not None else float("nan")))
        out[it["id"]] = rows
    return out


def ceil8(x):
    return max(0.125, math.ceil(x * 8.0 - 1e-12) / 8.0)


def binom_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def check(stock, cand, delta=None, seed=1, boots=4000):
    """stock: list of arms, cand: one arm (each {request: rows}); the requests are the same. -> dict"""
    ids = [i for i in cand if all(i in s for s in stock)]
    # delta: largest margin at stock's first departure per request (every stock arm), rounded up to 1/8
    if delta is None:
        firsts = []
        for s in stock:
            for i in ids:
                for mm, mg, _ in s[i]:
                    if mm:
                        firsts.append(mg)
                        break
        delta = ceil8(max(firsts)) if firsts else 0.125

    def counts(arm):
        return [(len(arm[i]), sum(1 for r in arm[i] if r[0]), sum(1 for r in arm[i] if r[0] and r[1] > 2 * delta)) for i in ids]  # noqa: E501
    sc = [counts(s) for s in stock]
    cc = counts(cand)
    n_s = [sum(c[0] for c in a) for a in sc]
    n_c = sum(c[0] for c in cc)

    def rate(cs):
        return 100.0 * sum(c[1] for c in cs) / max(1, sum(c[0] for c in cs))
    r_stock = [rate(a) for a in sc]
    r_cand = rate(cc)
    r_s = sum(r_stock) / len(r_stock)
    rng = random.Random(seed)
    diffs = []
    for _ in range(boots):
        pick = [rng.randrange(len(ids)) for _ in ids]
        dn = sum(cc[j][0] for j in pick)
        dc = 100.0 * sum(cc[j][1] for j in pick) / max(1, dn)
        ds = sum(100.0 * sum(a[j][1] for j in pick) / max(1, sum(a[j][0] for j in pick)) for a in sc) / len(sc)
        diffs.append(dc - ds)
    diffs.sort()
    lo, hi = diffs[int(0.05 * boots)], diffs[int(0.95 * boots) - 1]
    if lo > 0.5:
        rate_v = "REJECT"
    elif hi <= 0.5:
        rate_v = "PASS"
    else:
        rate_v = "INCONCLUSIVE"
    ev_c = sum(c[2] for c in cc)
    ev_s = sum(sum(c[2] for c in a) for a in sc)
    N_s = sum(n_s)
    p0 = n_c / (n_c + N_s)
    n = ev_c + ev_s
    p = binom_tail_ge(ev_c, n, p0) if n > 0 else 1.0
    event_v = "REJECT" if p < 0.05 else "PASS"
    # stock-vs-stock spread (the noise): per-arm rates and large-margin events
    return {"requests": len(ids), "tokens_cand": n_c, "tokens_stock_arms": n_s, "delta": delta, "mismatch_rate_stock_arms_pct": [round(x, 3) for x in r_stock],  # noqa: E501
            "mismatch_rate_cand_pct": round(r_cand, 3), "diff_pts": round(r_cand - r_s, 3), "diff_ci90_pts": [round(lo, 3), round(hi, 3)],  # noqa: E501
            "rate_rule": rate_v, "events_cand": ev_c, "events_stock_arms": [sum(c[2] for c in a) for a in sc], "events_p": round(p, 5),  # noqa: E501
            "event_rule": event_v, "verdict": "REJECT" if "REJECT" in (rate_v, event_v) else ("PASS" if rate_v == "PASS" else "INCONCLUSIVE")}  # noqa: E501


def cmd_check(a):
    stock = [load_arm(t) for t in a.stock]
    cand = load_arm(a.cand)
    if len(stock) < 2:
        print("warning: one stock arm only: the noise between stock runs is not measured", file=sys.stderr)
    r = check(stock, cand, a.delta)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return {"PASS": 0, "REJECT": 1, "INCONCLUSIVE": 2}[r["verdict"]]


def cmd_selftest(a):
    rng = random.Random(7)

    def arm(p_mis, p_big, nreq=60, n=300):
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
    s1, s2, s3 = arm(0.007, 0.0), arm(0.007, 0.0), arm(0.007, 0.0)
    ok = check([s1, s2], s3)
    bad = check([s1, s2], arm(0.02, 0.0))
    ev = check([s1, s2], arm(0.0075, 0.5))
    print("stock vs stock:", ok["verdict"], ok["diff_pts"], ok["diff_ci90_pts"], "events p", ok["events_p"])
    print("rate +1.3 pts:", bad["verdict"], bad["diff_pts"], bad["diff_ci90_pts"])
    print("more large-margin events:", ev["verdict"], ev["events_cand"], ev["events_stock_arms"], ev["events_p"])
    assert ok["verdict"] != "REJECT" and bad["verdict"] == "REJECT" and ev["event_rule"] == "REJECT", "selftest failed"
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("config")
    g.add_argument("out")
    g.add_argument("--corpus", required=True)
    g.add_argument("--kinds", default="")
    g.add_argument("--per-kind", type=int, default=4)
    g.add_argument("--offset", type=int, default=8)
    g.add_argument("--new", type=int, default=300)
    g.add_argument("--env", action="append")
    g.add_argument("--min-free-gb", type=int, default=20)
    s = sp.add_parser("score")
    s.add_argument("config")
    s.add_argument("set")
    s.add_argument("out")
    s.add_argument("--path", choices=["decode", "prompt"], default="decode",
                   help="decode: the real verify windows with the forced tokens (XESTRATA_FOLLOW); "
                        "prompt: the text read as a prompt (XESTRATA_LOGPOS)")
    s.add_argument("--env", action="append")
    s.add_argument("--arg", action="append")
    s.add_argument("--min-free-gb", type=int, default=20)
    c = sp.add_parser("check")
    c.add_argument("--stock", nargs="+", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--delta", type=float)
    c.add_argument("--json")
    sp.add_parser("selftest")
    a = ap.parse_args()
    return {"gen": cmd_gen, "score": cmd_score, "check": cmd_check, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
