"""Serve-API test of the DFlash2 drafter: runs `strata --serve` on one arm and a scenario of greedy requests, writes the tokens and the engine's DONE figures.

    python serve_test.py --exe ENGINE --arm df2 --scenario scenario.json --out out_df2.json --log arm.log -- <the engine arguments, as in the run config>

The wire is the engine's own: stdin `GEN <max_new> <id>,<id>,...`, stdout `T <id>` ... `DONE <n> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <resume> ...`
(serve/server.py is the Python client).  A scenario is {"requests": [{"name", "ids": [...]} | {"name", "after": "<name>", "follow": [...]}, ...], "max_new": N}:
a request with `after` continues that request's conversation (its prompt + its output + `follow`), which is what a chat client sends for the next turn.
Pure standard library: it runs on any box that has the engine.

compare mode:  python serve_test.py --compare nd.json mtp.json df2.json
prints, per request, the first token where the arm differs from the no-draft arm (the engine's windows are not bitwise equal to one-token decode at near-ties,
for the MTP as well; the DFlash2 arm must not be worse than the MTP arm), the drafts accepted / offered, and the figures the park test needs.
"""
import json
import subprocess
import sys
import threading
import time


def run(exe, args, scenario, log_path, ready_timeout=900, req_timeout=600):
    log = open(log_path, "ab")
    proc = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    lines = []
    cv = threading.Condition()

    def pump():
        for ln in proc.stdout:
            with cv:
                lines.append(ln.rstrip("\n"))
                cv.notify_all()
        with cv:
            lines.append(None)
            cv.notify_all()

    threading.Thread(target=pump, daemon=True).start()
    pos = 0

    def next_line(timeout):
        nonlocal pos
        end = time.time() + timeout
        with cv:
            while pos >= len(lines):
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError("no line from the engine")
                cv.wait(left)
            ln = lines[pos]
            pos += 1
            return ln

    while True:
        ln = next_line(ready_timeout)
        if ln is None:
            raise RuntimeError("the engine ended before READY")
        if ln.startswith("READY"):
            break
    done = {}
    results = []
    for req in scenario["requests"]:
        if "after" in req:
            prev = done[req["after"]]
            ids = prev["ids"] + prev["out"] + req.get("follow", [])
        else:
            ids = list(req["ids"])
        proc.stdin.write("GEN %d %s\n" % (req.get("max_new", scenario.get("max_new", 100)), ",".join(str(int(t)) for t in ids)))
        proc.stdin.flush()
        out, stats, resume = [], {}, None
        while True:
            ln = next_line(req_timeout)
            if ln is None:
                raise RuntimeError("the engine ended during %s" % req["name"])
            if ln.startswith("T "):
                out.append(int(ln.split()[1]))
            elif ln.startswith("RESUME "):
                resume = int(ln.split()[1])
            elif ln.startswith("ERR"):
                raise RuntimeError("engine: " + ln)
            elif ln.startswith("DONE"):
                f = ln.split()
                stats = {"produced": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]), "finish": f[5],
                         "accepted": int(f[6]), "offered": int(f[7]), "resume": int(f[8])}
                break
        done[req["name"]] = {"ids": ids, "out": out}
        results.append({"name": req["name"], "prompt_len": len(ids), "out": out, "resume": resume, **stats})
        print("%-14s prompt %5d resume %5s tokens %3d drafts %s/%s  %.1f tok/s" % (
            req["name"], len(ids), resume, len(out), stats.get("accepted"), stats.get("offered"),
            1000.0 * len(out) / max(1e-9, stats.get("decode_ms", 0.0))), flush=True)
    try:
        proc.stdin.close()
        proc.wait(60)
    except Exception:
        proc.kill()
    return results


def compare(nd_path, mtp_path, df2_path):
    nd, mtp, df2 = (json.load(open(p)) for p in (nd_path, mtp_path, df2_path))
    worst = 0
    print("%-14s %7s %9s %9s %12s %12s" % ("request", "tokens", "mtp diff@", "df2 diff@", "mtp drafts", "df2 drafts"))
    for a, b, c in zip(nd["results"], mtp["results"], df2["results"]):
        def first(x, y):
            n = min(len(x), len(y))
            return next((i for i in range(n) if x[i] != y[i]), None)
        fm, fd = first(a["out"], b["out"]), first(a["out"], c["out"])
        print("%-14s %7d %9s %9s %12s %12s" % (a["name"], len(a["out"]), fm, fd, "%d/%d" % (b["accepted"], b["offered"]), "%d/%d" % (c["accepted"], c["offered"])))
    return worst


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--compare":
        compare(*argv[1:4])
        sys.exit(0)
    opts = {}
    i = 0
    while i < len(argv) and argv[i] != "--":
        opts[argv[i].lstrip("-")] = argv[i + 1]
        i += 2
    engine_args = argv[i + 1:]
    scenario = json.load(open(opts["scenario"]))
    res = run(opts["exe"], engine_args, scenario, opts.get("log", "serve_test.log"))
    json.dump({"arm": opts.get("arm"), "results": res}, open(opts["out"], "w"))
    print("SERVE_TEST_DONE")
