#!/usr/bin/env python3
"""force_check.py: the token-forcing check (round 0217), after SpecPilot (arXiv 2611.02501, Li et al., Apache-2.0, "token forcing").

A candidate engine/build/setting is judged on a FIXED text: the engine scores prompt + continuation teacher-forced (every position's head logits come
out of the verify windows with the head in FP32: STRATA_LOGPOS / STRATA_LOGPOS_TOPK, the prompt cache off), and the checker counts where the scorer's
argmax is not the forced token.  Nothing here depends on a drafter; it serves any class N change (Q5_0 experts, tree verify, a new kernel ...).

  force_check.py gen   CFG SET.json --corpus held.jsonl [--kinds code,json] [--per-kind 4] [--offset 8] [--new 300] [--env K=V ...]
        one engine with CFG's normal decode flags; greedy continuations of the corpus prompts (rows with "ids") -> SET.json
  force_check.py score CFG SET.json OUT.tsv [--path decode|prompt] [--env K=V ...] [--arg "--flag v" ...]
        --path decode (default): one engine with CFG's own flags (drafter, spec windows, adaptive tiers ...), prompt cache off; every request decodes the
        forced continuation: the verify windows take the forced token in place of the argmax (STRATA_FOLLOW), so drafts are accepted when they match it,
        and the FP32 head logits of every valid row come out (STRATA_FOLLOW_LOGPOS, top-2).  --path prompt: the whole text is read as a prompt through the
        windows (STRATA_LOGPOS, --short-read), no decode features.  OUT.tsv.meta.json = per item line ranges.
        --env STRATA_FORCE_FAULT=7:0.3 is the test fault (adds 0.3 to every 7th token's logit in the scored rows).
  force_check.py check --stock A.tsv B.tsv [...] --cand C.tsv [--delta D] [--json out.json]
        the two rules of the paper against the stock arms (>= 2 stock arms: the noise): exit 0 pass, 1 reject, 2 inconclusive
  force_check.py selftest      the statistics on synthetic rows (no engine)

  THE GATE FOR A CHANGE OF THE PROMPT ARITHMETIC (a prompt kernel, an expert format's prompt path; owner 11-10), a neutral writer:
  force_check.py cut   IDS.txt CORPUS.jsonl [--sizes 4096,8192,20000] [--n 16]
        prompt cuts of one long id file (N windows per size at fixed offsets) -> a corpus
  force_check.py gen-neutral SET.json --corpus CORPUS.jsonl --writer /path/llama-server --writer-model SHARD1.gguf [--writer-args "-ngl 99 --cpu-moe"]
                             [--writer-lib DIR ...] [--per-kind 16] [--new 128]
        the continuations written by a program that is NEITHER arm: llama.cpp's llama-server on the same GGUF, greedy, the prompt passed as token ids
        (the ids `score` feeds the engine), the generated ids read back (return_tokens) -> SET.json marked "writer": "neutral"
  force_check.py gate  CFG SET.json OUTDIR --cand-env K=V [--cand-env ...] [--env K=V ...] [--arg "..."]
        scores the neutral set through stock twice (stockA, stockB) and the candidate twice (candA, candB), --path decode, then:
        PASS = each candidate arm against the two stock arms PASSES the rate rule and the event rule (check below); REJECT if any arm is rejected;
        otherwise INCONCLUSIVE.  The reversed direction (each stock arm against the candidate arms) is reported in gate.json, not required.
        Exit 0 pass, 1 reject, 2 inconclusive.  Size: 48 prompts (16 each at 4K / 8K / 20K) x 128 tokens = ~6,000 scored tokens an arm.
        Why (11-10, UD-IQ4_XS-v31 on an R9700 and an RTX A6000, IQ3_XXS on the R9700): with the STOCK engine writing the text, the
        rule measures the distance from the writer, not quality: (1) the FP16 prompt path, the reference of the KL gates, failed it on stock's
        text by more than the kernel under test (+1.66 points, 90% CI [+1.17, +2.15], against the kernel's +1.14 [+0.62, +1.69], 12 prompts);
        (2) with the CANDIDATE writing the text the stock engine failed by the same margin (+1.69 [+0.52, +2.93] and +1.56, REJECT, while the
        candidate passed against stock); (3) with llama.cpp writing, the candidate passed in every pairing (R9700 v31 -0.39 [-0.85, +0.07] and
        -0.33; R9700 IQ3_XXS -0.16 [-0.75, +0.42] twice; A6000 v31 48 prompts +0.21 [-0.05, +0.49] and +0.16), stock vs candidate reversed
        PASS or INCONCLUSIVE.  Stock A and stock B are never byte-identical through the decode windows.
  The stock-written form (gen with the stock engine, score, check) stays as a DIAGNOSTIC: it measures how far a candidate's arithmetic is
  from the engine that wrote the text, and is not the gate for prompt-arithmetic changes.

An "arm" = one scored TSV.  Which engine produced the continuation and which scored it is the caller's choice:
  - a change of the prompt arithmetic (a prompt kernel, an expert format's prompt path): `gate` on a `gen-neutral` set (above);
  - a change of the decode path (tree verify, a drafter): `gen` with the candidate, `score` with the STOCK scorer (the paper's way).
Stock arms = the same procedure with the stock build, at least twice (a second `gen`, or a second `score`).

Rules (paper):  mismatch = the scorer's argmax is not the forced token; margin = log p(argmax) - log p(forced token).
  rate rule : candidate mismatch rate within +0.5 points of stock at 95% (one-sided bootstrap over requests, paired on the same positions).
              REJECT if the 95% lower bound of (cand - stock) > 0.5 points; PASS if the 95% upper bound <= 0.5 points; else INCONCLUSIVE (more tokens).
  event rule: delta = the largest margin among stock's first departures per request, rounded UP to a multiple of 1/8; a large-margin event is a
              mismatch with margin > 2 delta.  REJECT if the candidate has more events than stock by an exact one-sided binomial test (p < 0.05,
              exposure-weighted).  (The paper also audits events stock reproduces on its own paths; that audit is not done here, events are counted raw.)
"""
import argparse, json, math, os, random, signal, subprocess, sys, pathlib

SCORER_DROP_VALUE = {"--draft", "--draft-model", "--draft-fallback", "--mtp-draft-vocab", "--spec", "--draft-conf", "--draft-cap", "--mtp", "--spec-min-p",
                     "--prompt-cache", "--max-context", "--prefill", "--short-read", "--prompt-cache-every", "--conversation-cache-mib", "--suffix-draft"}
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
            if k.startswith("STRATA_"):
                env.pop(k)
        env.update(cfg.get("env", {}))
        env.update(env_extra)
        if cfg.get("lib_dirs"):
            env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
        self.errf = open(log, "wb")
        self.p = subprocess.Popen([cfg["exe"], "--serve"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errf, text=True,
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
    """cfg "profile_src": a fresh copy of the expert profile per engine start (the engine may rewrite it) -> args with the copy's path"""
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
    return out + ["--prompt-cache", "0", "--max-context", str(total_tokens + 512), "--short-read", str(total_tokens + 64), "--prefill", "512"] + extra


def kv_env(lst):
    d = {}
    for kv in lst or []:
        k, _, v = kv.partition("=")
        d[k] = v
    return d


# ------------------------------------------------------------------------------------------------ gen / score
def cmd_gen(a):
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
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
        # the real decode/verify windows (drafter, spec windows, adaptive tiers, whatever the config has) with the forced continuation standing in
        # for the argmax (STRATA_FOLLOW, one line per request in decode order); rows come from STRATA_FOLLOW_LOGPOS
        follow = str(out) + ".follow.txt"
        with open(follow, "w") as ff:
            for it in items:
                ff.write(",".join(map(str, it["cont"])) + chr(10))
        env.update({"STRATA_FOLLOW": follow, "STRATA_FOLLOW_LOGPOS": str(out), "STRATA_LOGPOS_TOPK": "2"})
        args = list(cfg["args"])
        if "--prompt-cache" in args:
            i = args.index("--prompt-cache")
            del args[i:i + 2]
        args += ["--prompt-cache", "0"] + extra
    else:
        env.update({"STRATA_LOGPOS": str(out), "STRATA_LOGPOS_TOPK": "2", "STRATA_ROUTE_TAIL_SKIP": env.get("STRATA_ROUTE_TAIL_SKIP", "0")})
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
                        it["id"], len(toks), sum(1 for x, y in zip(toks, it["cont"]) if x != y) + abs(len(toks) - len(it["cont"]))), flush=True)
            else:
                toks, f = eng.gen(ids, 1)
            n1 = nlines()
            if n1 - n0 < len(it["cont"]) // 2:
                raise SystemExit("%s: only %d positions scored (the prompt took the batched path? raise --short-read; decode path: STRATA_FOLLOW not in this build?)" % (it["id"], n1 - n0))
            meta.append({"id": it["id"], "kind": it["kind"], "first_line": n0, "end_line": n1, "prompt_len": len(it["prompt"]), "ids": ids})
            print("%s: %d positions" % (it["id"], n1 - n0), flush=True)
    finally:
        eng.close()
    json.dump({"set": a.set, "config": a.config, "path": a.path, "env": a.env or [], "args": args, "items": meta}, open(str(out) + ".meta.json", "w"))
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
                raise SystemExit("%s: line for position %d scores token %d, the set has %d" % (it["id"], pos, tgt, ids[pos + 1]))
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
        return [(len(arm[i]), sum(1 for r in arm[i] if r[0]), sum(1 for r in arm[i] if r[0] and r[1] > 2 * delta)) for i in ids]
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
    return {"requests": len(ids), "tokens_cand": n_c, "tokens_stock_arms": n_s, "delta": delta, "mismatch_rate_stock_arms_pct": [round(x, 3) for x in r_stock],
            "mismatch_rate_cand_pct": round(r_cand, 3), "diff_pts": round(r_cand - r_s, 3), "diff_ci90_pts": [round(lo, 3), round(hi, 3)],
            "rate_rule": rate_v, "events_cand": ev_c, "events_stock_arms": [sum(c[2] for c in a) for a in sc], "events_p": round(p, 5),
            "event_rule": event_v, "verdict": "REJECT" if "REJECT" in (rate_v, event_v) else ("PASS" if rate_v == "PASS" else "INCONCLUSIVE")}


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


# ------------------------------------------------------------------------------------------------ the neutral writer (the gate, owner 11-10)
def cmd_cut(a):
    """prompt cuts of one long id file (comma-separated token ids): N windows of each size at fixed offsets (the cuts kl_ab.py and the
    11-10 gate runs used: offset = (i * 7919 + size) mod (len - size - 1)) -> a corpus for gen / gen-neutral"""
    big = [int(x) for x in open(a.ids).read().strip().split(",") if x.strip()]
    n = 0
    with open(a.out, "w") as f:
        for size in [int(s) for s in a.sizes.split(",") if s]:
            if size >= len(big):
                raise SystemExit("cut: the id file has %d ids, a cut of %d does not fit" % (len(big), size))
            for i in range(a.n):
                off = (i * 7919 + size) % max(1, len(big) - size - 1)
                f.write(json.dumps({"id": "p%d-%d" % (size, i), "kind": "p%d" % size, "split": "held", "ids": big[off: off + size]}) + "\n")
                n += 1
    print("cut: %d prompts -> %s" % (n, a.out))
    return 0


def http_json(url, obj=None, timeout=60.0):
    import urllib.request
    data = None if obj is None else json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def cmd_gen_neutral(a):
    """greedy continuations written by a NEUTRAL program (neither arm): llama.cpp's llama-server on the same GGUF.  The prompt goes in as token
    ids (the ids `score` feeds the engine), temperature 0 / top_k 1, no prompt cache, the generated ids back with return_tokens."""
    import shlex, time
    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
    rows = [r for r in rows if r.get("split", "held") == "held"]
    kinds = a.kinds.split(",") if a.kinds else sorted({r["kind"] for r in rows})
    sel = []
    for k in kinds:
        sel += [r for r in rows if r["kind"] == k][a.offset:a.offset + a.per_kind]
    def split(s):   # (Windows: shlex's non-POSIX mode keeps the quotes around a token)
        return [x[1:-1] if len(x) > 1 and x[0] == x[-1] == '"' else x for x in shlex.split(s, posix=os.name != "nt")]
    cmd = split(a.writer) + ["-m", a.writer_model, "--port", str(a.writer_port), "--host", "127.0.0.1",
                             "-c", str(a.writer_ctx), "-np", "1"] + split(a.writer_args or "")
    env = dict(os.environ)
    env.update(kv_env(a.writer_env))
    if a.writer_lib:
        k = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[k] = os.pathsep.join(a.writer_lib + [env.get(k, "")])
    log = open(a.out + ".writer.log", "w")
    log.write("# " + " ".join(cmd) + "\n")
    log.flush()
    kw = {"creationflags": 0x00000200} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, **kw)
    url = "http://127.0.0.1:%d" % a.writer_port
    items = []
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                raise SystemExit("gen-neutral: the writer ended before it was ready, see %s.writer.log" % a.out)
            try:
                if http_json(url + "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t0 > a.writer_wait:
                raise SystemExit("gen-neutral: the writer was not ready after %d s" % a.writer_wait)
            time.sleep(1.0)
        for r in sel:
            res = http_json(url + "/completion", {"prompt": r["ids"], "n_predict": a.new, "temperature": 0.0, "top_k": 1, "cache_prompt": False,
                                                  "return_tokens": True, "stream": False}, timeout=3600)
            toks = [int(x) for x in res.get("tokens", [])]
            if not toks:
                raise SystemExit("gen-neutral: %s: the writer returned no token ids (a llama-server without return_tokens?)" % r["id"])
            items.append({"id": r["id"], "kind": r["kind"], "prompt": r["ids"], "cont": toks})
            print("%s: %d prompt, %d forced tokens (writer)" % (r["id"], len(r["ids"]), len(toks)), flush=True)
    finally:
        try:
            if os.name == "nt":
                p.terminate()
            else:
                os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=60)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        log.close()
    json.dump({"config": "writer: " + " ".join(cmd), "writer": "neutral", "env": [], "items": items}, open(a.out, "w"))
    return 0


def gate_report(stock, cand, delta=None):
    """stock, cand: two (or more) scored arms each, on a NEUTRAL writer's text.  The gate: every candidate arm against the stock arms (check());
    REJECT if any is REJECT, PASS if all PASS, else INCONCLUSIVE.  The reversed direction (every stock arm against the candidate arms) is
    reported beside it, not required."""
    fwd = [check(stock, c, delta) for c in cand]
    rev = [check(cand, s, delta) for s in stock]
    vs = [r["verdict"] for r in fwd]
    verdict = "REJECT" if "REJECT" in vs else ("PASS" if all(v == "PASS" for v in vs) else "INCONCLUSIVE")
    return {"verdict": verdict, "cand_vs_stock": fwd, "stock_vs_cand_reported": rev}


def cmd_gate(a):
    st = json.load(open(a.set))
    if st.get("writer") != "neutral" and not a.allow_engine_writer:
        raise SystemExit("gate: %s was not written by gen-neutral; the gate needs a neutral writer's text (an engine-written set measures the "
                         "distance from that engine: use score + check as a diagnostic, or --allow-engine-writer)" % a.set)
    os.makedirs(a.outdir, exist_ok=True)
    arms = {"stockA": a.env or [], "stockB": a.env or [], "candA": (a.env or []) + a.cand_env, "candB": (a.env or []) + a.cand_env}
    for name, env in arms.items():
        out = os.path.join(a.outdir, name + ".tsv")
        if os.path.exists(out) and os.path.exists(out + ".meta.json"):
            print("%s: kept (%s exists)" % (name, out), flush=True)
            continue
        cmd_score(argparse.Namespace(config=a.config, set=a.set, out=out, path=a.path, env=env, arg=a.arg, min_free_gb=a.min_free_gb))
    load = lambda n: load_arm(os.path.join(a.outdir, n + ".tsv"))
    r = gate_report([load("stockA"), load("stockB")], [load("candA"), load("candB")], a.delta)
    print(json.dumps(r, indent=1))
    json.dump(r, open(os.path.join(a.outdir, "gate.json"), "w"), indent=1)
    return {"PASS": 0, "REJECT": 1, "INCONCLUSIVE": 2}[r["verdict"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("config"); g.add_argument("out"); g.add_argument("--corpus", required=True)
    g.add_argument("--kinds", default=""); g.add_argument("--per-kind", type=int, default=4); g.add_argument("--offset", type=int, default=8)
    g.add_argument("--new", type=int, default=300); g.add_argument("--env", action="append"); g.add_argument("--min-free-gb", type=int, default=20)
    s = sp.add_parser("score"); s.add_argument("config"); s.add_argument("set"); s.add_argument("out")
    s.add_argument("--path", choices=["decode", "prompt"], default="decode", help="decode: the real verify windows with the forced tokens (STRATA_FOLLOW); prompt: the text read as a prompt (STRATA_LOGPOS)")
    s.add_argument("--env", action="append"); s.add_argument("--arg", action="append"); s.add_argument("--min-free-gb", type=int, default=20)
    c = sp.add_parser("check"); c.add_argument("--stock", nargs="+", required=True); c.add_argument("--cand", required=True)
    c.add_argument("--delta", type=float); c.add_argument("--json")
    sp.add_parser("selftest")
    u = sp.add_parser("cut"); u.add_argument("ids"); u.add_argument("out"); u.add_argument("--sizes", default="4096,8192,20000")
    u.add_argument("--n", type=int, default=16)
    w = sp.add_parser("gen-neutral"); w.add_argument("out"); w.add_argument("--corpus", required=True)
    w.add_argument("--writer", required=True, help="the writer's command line, e.g. /path/llama-server (its -m/--port/--host/-c/-np are added)")
    w.add_argument("--writer-model", required=True, help="the GGUF the engine reads (shard 1 of a split set)")
    w.add_argument("--writer-args", default="", help="more writer arguments, e.g. \"-ngl 99 --cpu-moe -fa on\"")
    w.add_argument("--writer-env", action="append"); w.add_argument("--writer-lib", action="append", default=[])
    w.add_argument("--writer-port", type=int, default=18471); w.add_argument("--writer-ctx", type=int, default=32768)
    w.add_argument("--writer-wait", type=int, default=900)
    w.add_argument("--kinds", default=""); w.add_argument("--per-kind", type=int, default=16); w.add_argument("--offset", type=int, default=0)
    w.add_argument("--new", type=int, default=128)
    t = sp.add_parser("gate"); t.add_argument("config"); t.add_argument("set"); t.add_argument("outdir")
    t.add_argument("--cand-env", action="append", required=True, help="the candidate's setting(s), K=V")
    t.add_argument("--env", action="append", help="settings of both arms"); t.add_argument("--arg", action="append")
    t.add_argument("--path", choices=["decode", "prompt"], default="decode"); t.add_argument("--delta", type=float)
    t.add_argument("--min-free-gb", type=int, default=20); t.add_argument("--allow-engine-writer", action="store_true")
    a = ap.parse_args()
    return {"gen": cmd_gen, "score": cmd_score, "check": cmd_check, "selftest": cmd_selftest, "cut": cmd_cut, "gen-neutral": cmd_gen_neutral,
            "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
