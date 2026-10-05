# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/test_strata_mcp.py - tests for tools/strata_mcp.py (Strata's MCP server), standard library only.

    python -m unittest tools/test_strata_mcp.py -v

Setup and the model server are fakes (a temporary Strata folder with a stand-in setup.py and serve/server.py), so
nothing is downloaded and no GPU is used: the JSON-RPC handshake, argument checks, install / start / stop with real
background processes, status against a fake HTTP server, and that the built-in model table still matches setup.py.
"""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata_mcp as M  # noqa: E402

FAKE_SETUP = r'''
import json, sys, time
from pathlib import Path
def main():
    root = Path(__file__).resolve().parent
    (root / "setup-args.json").write_text(json.dumps(sys.argv[1:]))
    print("Strata - fake setup", flush=True)
    print("\n=== Step 1: checking your PC ===", flush=True)
    print("  [ok] GPU: fake", flush=True)
    print("\n=== Step 5: downloading Qwen3.8-Flash-Next Q2_0 ===", flush=True)
    for pct in (10, 20, 30):
        print(f"\r  model.gguf:  {pct * 0.5:.2f} / 50.00 GB ({pct}%)   ", end="", flush=True)
    print(flush=True)
    if (root / "slow").exists():
        end = time.time() + 30
        while not (root / "go").exists() and time.time() < end:
            time.sleep(0.1)
    if (root / "fail").exists():
        print("\n  [X]  not enough free disk space", flush=True)
        print("       use --models-dir on a bigger drive", flush=True)
        sys.exit(1)
    a = sys.argv
    model = a[a.index("--model") + 1].lower()
    cfg = {"exe": sys.executable, "args": ["--max-context", a[a.index("--context") + 1]], "port": 8095,
           "model_name": "qwen3.8-flash-next-" + model, "log": str(root / f"xestrata-{model}.log")}
    (root / f"xestrata-{model}.json").write_text(json.dumps(cfg))
    print("All set.", flush=True)


if __name__ == "__main__":
    main()
'''

FAKE_SERVER = r'''
import json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
a = sys.argv
port = int(a[a.index("--port") + 1])
cfg = json.loads(Path(a[a.index("--config") + 1]).read_text())
root = Path(__file__).resolve().parent.parent
print("loading the model (fake) ...", flush=True)
if cfg.get("crash"):
    print("engine failed: out of memory (fake)", flush=True)
    sys.exit(3)
time.sleep(cfg.get("delay", 0))
state = {"loaded": True}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def js(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self.js(200, {"status": "ok", "service": "strata", "model": cfg["model_name"], "max_context": 32768,
                          "images": False, "api_key": False, "loaded": state["loaded"]})
        elif p == "/status":
            self.js(200, {"busy": bool(cfg.get("busy")), "queued": 0})
        elif p == "/v1/status":
            self.js(200, {"service": "strata", "engine": "0.1.32", "uptime_s": 5, "activity": {"requests": 0},
                          "machine": {"gpu": {"name": "Fake GPU", "used_mib": 11000, "total_mib": 12227},
                                      "ram": {"used_gib": 40.0, "total_gib": 63.1}}})
        else:
            self.js(404, {"error": {"message": "not found"}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        p = self.path.split("?")[0]
        if p == "/unload":
            state["loaded"] = False
            with open(root / "unloaded.txt", "a") as f:
                f.write(str(port) + "\n")
            self.js(200, {"status": "unloaded"})
        elif p == "/v1/chat/completions":
            with open(root / "bench-request.json", "w") as f:
                json.dump(body, f)
            self.js(200, {"model": cfg["model_name"],
                          "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                          "usage": {"prompt_tokens": 30, "completion_tokens": body.get("max_tokens", 1)},
                          "timings": {"predicted_per_second": 61.5, "prompt_per_second": 400.0}})
        else:
            self.js(404, {})
httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
def term(*_):
    print("[strata] stopped (fake)", flush=True)
    os._exit(0)
signal.signal(signal.SIGTERM, term)
print(f"ready: http://127.0.0.1:{port}/v1", flush=True)
httpd.serve_forever()
'''


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for(cond, seconds=30.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


FAKE_HW = {"os": "TestOS 1", "ram_gb": 64.0, "cpu": {"name": "Fake CPU", "cores": 12, "avx2": True, "avx512": True},
           "gpus": [{"index": 0, "name": "Intel Arc B580", "vram_gb": 11.9, "pci": "0000:03:00.0", "vendor": "intel",
                     "integrated": False, "problem": None, "usable": True}]}


class FakeRoot(unittest.TestCase):
    """A temporary Strata folder: fake setup.py and serve/server.py, its own settings file (data folder)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "XeStrata"
        (self.root / "serve").mkdir(parents=True)
        (self.root / "setup.py").write_text(FAKE_SETUP)
        (self.root / "serve" / "server.py").write_text(FAKE_SERVER)
        self.env_backup = {k: os.environ.get(k) for k in ("XDG_CONFIG_HOME",)}
        os.environ["XDG_CONFIG_HOME"] = str(base / "config")
        self.strata = M.Strata(self.root)
        self.strata.python_override = sys.executable
        self.strata.health_timeout = 1.0
        self.strata._hardware = lambda: dict(FAKE_HW)
        self.strata.disk_free = lambda p: 1000.0
        self.tools = M.Tools(self.strata)
        self.extra = []

    def tearDown(self):
        for p in self.extra:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        srv = self.strata.state("server")
        if srv and M.proc_alive(srv.get("pid"), srv.get("ident")):
            M.proc_kill_tree(srv["pid"], srv["ident"])
            M.wait_gone(srv["pid"], srv["ident"], 10)
        job = self.strata.state("install")
        if job and M.proc_alive(job.get("pid"), job.get("ident")):
            M.proc_kill_tree(job["pid"], job["ident"])
            M.wait_gone(job["pid"], job["ident"], 10)
        for p in self.strata.children:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        for k, v in self.env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def call(self, name, args=None):
        """A tools/call through the JSON-RPC layer: (result dict, isError)."""
        srv = M.McpServer(self.strata, io.BytesIO())
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args or {}}})
        res = r["result"]
        return json.loads(res["content"][0]["text"]), res["isError"]

    def write_config(self, tag="q2_0", port=8095, **extra):
        cfg = {"exe": sys.executable, "args": ["--max-context", "32768"], "port": port,
               "model_name": f"qwen3.8-flash-next-{tag}", "log": str(self.root / f"xestrata-{tag}.log"), **extra}
        (self.root / f"xestrata-{tag}.json").write_text(json.dumps(cfg))
        (self.root / f"xestrata-{tag}.log").write_text("engine line 1\nengine line 2\n")
        return cfg


# ------------------------------------------------------------------------------------------------ the protocol
class Protocol(FakeRoot):
    def run_lines(self, msgs):
        inp = io.BytesIO(("\n".join(m if isinstance(m, str) else json.dumps(m) for m in msgs) + "\n").encode())
        out = io.BytesIO()
        M.McpServer(self.strata, out).serve(inp)
        return {m.get("id"): m for m in (json.loads(x) for x in out.getvalue().decode().splitlines())}

    def test_handshake_list_call(self):
        r = self.run_lines([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "t", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "strata_models", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "ping"},
        ])
        init = r[1]["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertIn("tools", init["capabilities"])
        self.assertEqual(init["serverInfo"]["name"], "xestrata")
        names = [t["name"] for t in r[2]["result"]["tools"]]
        self.assertEqual(names, ["strata_status", "strata_models", "strata_install", "strata_start", "strata_stop",
                                 "strata_logs", "strata_benchmark", "strata_connect_info"])
        for t in r[2]["result"]["tools"]:
            self.assertEqual(t["inputSchema"]["type"], "object")
            self.assertFalse(t["inputSchema"]["additionalProperties"])
            self.assertGreater(len(t["description"]), 60)
        call = r[3]["result"]
        self.assertFalse(call["isError"])
        self.assertIn("families", call["structuredContent"])
        self.assertEqual(json.loads(call["content"][0]["text"])["recommendation"]["model"], "IQ3_XXS")
        self.assertEqual(r[4]["result"], {})
        self.assertNotIn(None, r)                       # the notification got no answer

    def test_older_protocol_and_errors(self):
        r = self.run_lines([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            "{not json",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "rm_rf", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 3, "method": "no/such"},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "strata_logs", "arguments": []}},
        ])
        self.assertEqual(r[1]["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(r[None]["error"]["code"], -32700)
        self.assertEqual(r[2]["error"]["code"], -32602)
        self.assertEqual(r[3]["error"]["code"], -32601)
        self.assertTrue(r[4]["result"]["isError"])
        self.assertNotIn("structuredContent", r[4]["result"])   # 2024-11-05 has no structured results

    def test_real_stdio_process(self):
        """The script itself over a pipe: stdout carries only JSON-RPC lines."""
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "strata_logs", "arguments": {"source": "setup"}}}]
        p = subprocess.run([sys.executable, str(HERE / "strata_mcp.py"), "--root", str(self.root)],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                           timeout=60, env=dict(os.environ))
        lines = [json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual(sorted(m["id"] for m in lines), [1, 2, 3])
        self.assertEqual(len(next(m for m in lines if m["id"] == 2)["result"]["tools"]), 8)
        self.assertIn("ready", p.stderr)

    def test_not_a_strata_folder(self):
        p = subprocess.run([sys.executable, str(HERE / "strata_mcp.py"), "--root", self.tmp.name], input="",
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2)


# ------------------------------------------------------------------------------------------------ validation
class Validation(FakeRoot):
    def assertRejected(self, tool, args, text):
        res, err = self.call(tool, args)
        self.assertTrue(err, f"{tool} {args} was accepted: {res}")
        self.assertIn(text, res["error"])

    def test_install_arguments(self):
        self.assertRejected("strata_install", {"model": "Q9_X"}, "model must be one of")
        self.assertRejected("strata_install", {"family": "llama"}, "family must be one of")
        self.assertRejected("strata_install", {"context": 12345}, "context must be one of")
        self.assertRejected("strata_install", {"gpu": -1}, "between")
        self.assertRejected("strata_install", {"gpu": "1; calc"}, "whole number")
        self.assertRejected("strata_install", {"gpus": "0,1"}, "unknown argument")
        self.assertRejected("strata_install", {"port": 80}, "between")
        self.assertRejected("strata_install", {"backend": "hip"}, "unknown argument")
        self.assertRejected("strata_install", {"command": "setup.py --build"}, "unknown argument")
        self.assertRejected("strata_install", {"family": "swift", "model": "IQ3_S"}, "has no IQ3_S")
        self.assertRejected("strata_install", {"family": "unsloth", "model": "UD-Q4_K_XL", "vision": "yes"},
                            "images are not available")
        self.assertRejected("strata_install", {"vision": "yes\nrm"}, "control characters")

    def test_data_dir_paths(self):
        self.assertRejected("strata_install", {"data_dir": "models"}, "absolute")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "x" / ".." / "XeStrata-x")}, "'..'")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "Documents")}, "named XeStrata")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "nope" / "XeStrata-data")},
                            "does not exist")
        self.assertRejected("strata_install", {"data_dir": "//server/share/XeStrata-data"}, "network")
        self.assertRejected("strata_install", {"data_dir": "/etc/XeStrata-data"}, "system folder")
        ok = [self.root.parent / "XeStrata-data", self.root / "models-here", self.root.parent / "XeStrata-big"]
        for d in ok:
            self.assertEqual(M.norm(self.tools.check_data_dir(str(d))), M.norm(d))

    def test_other_tools_arguments(self):
        self.assertRejected("strata_start", {"model": "../../etc/passwd"}, "not valid")
        self.assertRejected("strata_start", {"model": "q2_0"}, "no model is installed")
        self.write_config("q2_0")
        self.assertRejected("strata_start", {"model": "iq3_s"}, "no installed model 'iq3_s'")
        self.assertRejected("strata_start", {"wait_seconds": 100000}, "between")
        self.assertRejected("strata_logs", {"lines": 0}, "between")
        self.assertRejected("strata_logs", {"source": "/etc/passwd"}, "source must be one of")
        self.assertRejected("strata_benchmark", {"max_tokens": 100000}, "between")
        self.assertRejected("strata_stop", {"force": "yes"}, "true or false")

    def test_engine_log_outside_is_not_read(self):
        self.write_config("q2_0", log=str(Path(self.tmp.name).parent / "elsewhere.log"))
        res, err = self.call("strata_logs", {"source": "engine", "model": "q2_0"})
        self.assertTrue(err)
        self.assertIn("outside the Strata folder", res["error"])
        self.write_config("q2_0")
        res, err = self.call("strata_logs", {"source": "engine"})
        self.assertFalse(err)
        self.assertEqual(res["lines"], ["engine line 1", "engine line 2"])


# ------------------------------------------------------------------------------------------------ install
class Install(FakeRoot):
    def wait_job(self):
        self.assertTrue(wait_for(lambda: not self.strata.install_job()["running"], 60))

    def test_plan_then_install(self):
        res, err = self.call("strata_install", {})
        self.assertFalse(err)
        self.assertIn("Nothing was done yet", res["summary"])
        self.assertEqual((res["plan"]["family"], res["plan"]["model"], res["plan"]["context"]),
                         ("qwen", "IQ3_XXS", 32768))
        self.assertFalse((self.root / "setup-args.json").exists())
        self.assertIsNone(self.strata.install_job())

        res, err = self.call("strata_install", {"model": "Q2_0", "context": 65536, "vision": "no", "confirm": True})
        self.assertFalse(err, res)
        self.assertIn("started in the background", res["summary"])
        self.wait_job()
        args = json.loads((self.root / "setup-args.json").read_text())
        self.assertEqual(args[:2], ["--yes", "--no-start"])
        for pair in (["--family", "qwen"], ["--model", "Q2_0"], ["--context", "65536"], ["--vision", "no"]):
            i = args.index(pair[0])
            self.assertEqual(args[i:i + 2], pair)
        res, err = self.call("strata_install", {})
        self.assertEqual(res["install"]["result"], "installed")
        self.assertEqual(res["install"]["exit_code"], 0)
        self.assertEqual(res["next"], "strata_start")
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertEqual([m["model"] for m in st["installed"]["models"]], ["q2_0"])

    def test_progress_while_running_and_cancel(self):
        (self.root / "slow").write_text("")
        res, err = self.call("strata_install", {"family": "coder", "confirm": True})
        self.assertFalse(err, res)
        self.assertTrue(wait_for(lambda: (self.strata.install_job() or {}).get("download") is not None, 30))
        res, err = self.call("strata_install", {"model": "IQ3_S", "confirm": True})   # a second call: progress only
        self.assertFalse(err)
        self.assertIn("an install is running", res["summary"])
        self.assertEqual(res["install"]["step"], "5/7: downloading Qwen3.8-Flash-Next Q2_0")
        self.assertEqual(res["install"]["download"]["percent"], 30)
        res, err = self.call("strata_start", {})
        self.assertTrue(err)
        self.assertIn("install is still running", res["error"])
        res, err = self.call("strata_install", {"cancel": True})
        self.assertIn("stopped", res["summary"])
        self.assertFalse(self.strata.install_job()["running"])

    def test_failure_is_reported(self):
        (self.root / "fail").write_text("")
        self.call("strata_install", {"model": "Q2_0", "confirm": True})
        self.wait_job()
        res, _ = self.call("strata_install", {})
        self.assertEqual(res["install"]["result"], "failed")
        self.assertIn("not enough free disk space - use --models-dir on a bigger drive", res["install"]["error"])

    def test_disk_space_is_checked(self):
        self.strata.disk_free = lambda p: 10.0
        res, err = self.call("strata_install", {"model": "Q2_0"})
        self.assertIn("not enough free disk space", res["plan"]["disk_short"])
        res, err = self.call("strata_install", {"model": "Q2_0", "confirm": True})
        self.assertTrue(err)
        self.assertIsNone(self.strata.install_job())


# ------------------------------------------------------------------------------------------------ start / stop / status
class StartStop(FakeRoot):
    def test_start_status_bench_stop(self):
        port = free_port()
        self.write_config("q2_0", port=port, delay=1.5)
        res, err = self.call("strata_start", {"wait_seconds": 30})
        self.assertFalse(err, res)
        self.assertIn("Strata is running", res["summary"])
        pid = res["server"]["pid"]

        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn(f"running on port {port}", st["summary"])
        srv = st["running"][0]
        self.assertEqual(srv["engine_version"], "0.1.32")
        self.assertEqual(srv["gpu"]["total_mib"], 12227)
        self.assertIn("this MCP server", srv["started_by"])

        res, err = self.call("strata_start", {})            # already running: nothing new
        self.assertIn("already running", res["summary"])

        b, err = self.call("strata_benchmark", {"max_tokens": 64})
        self.assertFalse(err, b)
        self.assertEqual(b["output_tok_s"], 61.5)
        sent = json.loads((self.root / "bench-request.json").read_text())
        self.assertEqual((sent["temperature"], sent["max_tokens"], sent["reasoning_effort"]), (0, 64, "none"))

        c, _ = self.call("strata_connect_info", {})
        self.assertEqual(c["openai_base_url"], f"http://127.0.0.1:{port}/v1")

        logs, _ = self.call("strata_logs", {"source": "server"})
        self.assertTrue(any(line.startswith("ready:") for line in logs["lines"]))

        res, err = self.call("strata_stop", {})
        self.assertFalse(err, res)
        self.assertIn("Strata stopped", res["summary"])
        self.assertEqual(res["engine_unloaded"], "unloaded")
        self.assertEqual((self.root / "unloaded.txt").read_text().split(), [str(port)])
        ident = M.proc_identity(pid)
        self.assertIsNone(ident)
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertEqual(st["running"], [])
        res, _ = self.call("strata_stop", {})
        self.assertEqual(res["summary"], "Strata is not running")

    def test_start_reports_a_crash(self):
        self.write_config("q2_0", port=free_port(), crash=True)
        res, err = self.call("strata_start", {"wait_seconds": 30})
        self.assertTrue(err)
        self.assertIn("out of memory (fake)", res["error"])
        self.assertEqual(self.strata.state("server"), {})

    def test_loading_then_ready(self):
        port = free_port()
        self.write_config("q2_0", port=port, delay=3)
        res, err = self.call("strata_start", {"wait_seconds": 0})
        self.assertIn("loading", res["summary"])
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn("loading", st["summary"])
        self.assertTrue(wait_for(lambda: self.strata.probe(port) is not None, 30))

    def test_stop_while_busy_needs_force(self):
        port = free_port()
        self.write_config("q2_0", port=port, busy=True)
        self.call("strata_start", {"wait_seconds": 30})
        res, err = self.call("strata_stop", {})
        self.assertTrue(err)
        self.assertIn("force=true", res["error"])
        res, err = self.call("strata_stop", {"force": True})
        self.assertFalse(err, res)

    def test_external_server_is_only_unloaded(self):
        port = free_port()
        self.write_config("q2_0", port=port)
        p = subprocess.Popen([sys.executable, str(self.root / "serve" / "server.py"), "--config",
                              str(self.root / "xestrata-q2_0.json"), "--port", str(port)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.extra.append(p)
        self.assertTrue(wait_for(lambda: self.strata.probe(port) is not None, 30))
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn("outside this MCP server", st["running"][0]["started_by"])
        res, err = self.call("strata_start", {})
        self.assertIn("started outside this MCP server", res["summary"])
        res, err = self.call("strata_stop", {})
        self.assertFalse(err, res)
        self.assertIn("only its model was unloaded", res["summary"])
        time.sleep(0.5)
        self.assertIsNone(p.poll())                      # the server itself was not touched

    def test_never_kills_an_unrelated_process(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.extra.append(other)
        # a stale record: this process id, but another start time (a reused id)
        self.strata.save_state("server", {"pid": other.pid, "ident": "linux:1",
                                          "port": free_port(), "model": "q2_0"})
        res, _ = self.call("strata_stop", {})
        self.assertEqual(res["summary"], "Strata is not running")
        self.assertFalse(M.proc_terminate(other.pid, "linux:1"))
        time.sleep(0.3)
        self.assertIsNone(other.poll())

    def test_status_sees_other_programs_on_the_port(self):
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def log_message(self, *a):
                pass
        httpd = HTTPServer(("127.0.0.1", 0), H)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            self.write_config("q2_0", port=port)
            st, _ = self.call("strata_status", {"hardware": False})
            self.assertEqual(st["running"], [])
            self.assertEqual(st["ports_used_by_other_programs"][0]["port"], port)
            res, err = self.call("strata_start", {})
            self.assertTrue(err)
            self.assertIn("used by another program", res["error"])
        finally:
            httpd.shutdown()
            httpd.server_close()


# ------------------------------------------------------------------------------------------------ helpers
class Helpers(unittest.TestCase):
    def test_progress_from_setup_output(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "setup.log"
            f.write_text("=== Step 4: the Strata engine ===\n  [ok] engine\n=== Step 5: downloading X ===\n"
                         "\r  a.gguf:   1.00 / 50.00 GB (2%)   \r  a.gguf:  25.00 / 50.00 GB (50%)   \n")
            p = M.install_progress(M.read_tail(f, 50))
        self.assertEqual(p["step"], "5/7: downloading X")
        self.assertEqual(p["download"], {"file": "a.gguf", "done_gb": 25.0, "total_gb": 50.0, "percent": 50})
        self.assertNotIn("error", p)

    def test_builtin_table_matches_setup(self):
        s = M.Strata(HERE.parent)
        S = s.setup_module()
        self.assertIsNotNone(S, "the repository's setup.py did not import")
        self.assertEqual(list(S.MODELS), list(M.FALLBACK_MODELS))
        for m, d in S.MODELS.items():
            for k in ("download_gb", "ram_gb", "arena_gb"):
                self.assertEqual(d[k], M.FALLBACK_MODELS[m][k], f"{m} {k}")
            self.assertEqual(d.get("families", ("qwen", "swift")),
                             M.FALLBACK_MODELS[m].get("families", ("qwen", "swift")))
        self.assertEqual(list(S.FAMILIES), list(M.FALLBACK_FAMILIES))
        for f, d in S.FAMILIES.items():
            self.assertEqual(d["tag"], M.FALLBACK_FAMILIES[f]["tag"])
        self.assertEqual(list(S.CONTEXTS), M.FALLBACK_CONTEXTS)

    def test_unsloth_sizes(self):
        """UD-IQ4_XS is a regular size, listed first (the family's default); UD-Q4_K_XL stays experimental."""
        s = M.Strata(HERE.parent)
        for models in (s.tables()[0], M.FALLBACK_MODELS):
            self.assertEqual(s.sizes_of(models, "unsloth"), ["UD-IQ4_XS", "UD-Q4_K_XL"])
            self.assertFalse(models["UD-IQ4_XS"].get("experimental"))
            self.assertTrue(models["UD-Q4_K_XL"].get("experimental"))
        fam = next(f for f in s.model_table({"ram_gb": 63.7, "gpus": []}) if f["family"] == "unsloth")
        self.assertFalse(fam["experimental"])
        self.assertTrue(fam["images"])
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual((by["UD-IQ4_XS"]["experimental"], by["UD-IQ4_XS"]["images"]), (False, True))
        self.assertEqual((by["UD-Q4_K_XL"]["experimental"], by["UD-Q4_K_XL"]["images"]), (True, False))
        self.assertFalse(by["UD-IQ4_XS"]["on_this_pc"].startswith("experimental"))
        self.assertTrue(by["UD-Q4_K_XL"]["on_this_pc"].startswith("experimental"))

    def test_recommendation_follows_ram(self):
        s = M.Strata(HERE.parent)
        gpu = [{"index": 0, "name": "x", "vram_gb": 24.0, "usable": True, "vendor": "intel"}]
        cases = {96: ("qwen", "IQ3_XXS", 131072), 48: ("qwen", "Q2_0", 131072), 32: ("coder", "IQ1_M", 131072)}
        for ram, want in cases.items():
            r = s.recommend({"ram_gb": ram, "gpus": gpu, "cpu": {"avx2": True}})
            self.assertEqual((r["family"], r["model"], r["context"]), want, ram)
        r = s.recommend({"ram_gb": 16, "gpus": [{**gpu[0], "vram_gb": 8.0}], "cpu": {"avx2": True}})
        self.assertIsNone(r["model"])
        r = s.recommend({"ram_gb": 64, "gpus": [], "cpu": {"avx2": True}})
        self.assertIsNone(r["model"])
        r = s.recommend({"ram_gb": 64, "gpus": [{**gpu[0], "vram_gb": 12.0}], "cpu": {"avx2": True}})
        self.assertEqual(r["context"], 32768)
        r = s.recommend({"ram_gb": 64, "gpus": [{**gpu[0], "vram_gb": 0.0, "integrated": True}], "cpu": {"avx2": True}})
        self.assertEqual(r["model"], "IQ3_XXS")         # the processor's own graphics: it runs, slowly
        self.assertTrue(any("own graphics" in n for n in r["notes"]))


if __name__ == "__main__":
    unittest.main()
