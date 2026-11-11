"""tools/test_run_strata_image.py - tools/image/gate/run_strata_image.py with a stand-in for strata-image: the command
line it builds (the three model files, the gate prompt, size, seed, 8 steps, output, report, dump), the files it writes
(p<i>_s<seed>.png / .log / .json, results.jsonl) and the record (stage times under run_set.py's names, memory per stage,
rc and the error tail of a failed picture).  POSIX only (the runner uses os.wait4).

    python -m pytest tools/test_run_strata_image.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "image" / "gate" / "run_strata_image.py"
sys.path.insert(0, str(RUNNER.parent))
from run_set import PROMPTS  # noqa: E402

STAND_IN = r'''#!/usr/bin/env python3
import json, sys
a = sys.argv[1:]
open(sys.argv[0] + ".args", "a").write(json.dumps(a) + "\n")
v = lambda k: a[a.index(k) + 1]
if v("--seed") == "13":
    print("strata-image: something broke"); sys.exit(1)
open(v("-o"), "wb").write(b"\x89PNG fake")
st = [{"name": n, "wall_ms": ms, "gpu_ms": -1, "ours_gb": 1.5, "driver_gb": 2.0, "host_gb": 0.5, "note": ""}
      for n, ms in (("open", 700.0), ("encode", 150.0), ("steps", 9800.0), ("decode", 1200.0), ("png", 30.0))]
json.dump({"plan": "resident", "stages": st}, open(v("--report"), "w"))
print("STRATA-IMAGE PASS")
'''


@unittest.skipIf(os.name != "posix", "the runner uses os.wait4")
class RunStrataImage(unittest.TestCase):
    def test_runs_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            exe = t / "fake-strata-image"
            exe.write_text(STAND_IN)
            exe.chmod(0o755)
            models = t / "models"
            models.mkdir()
            r = subprocess.run([sys.executable, "-I", "-B", str(RUNNER), str(t), "set1", "512x512", str(models), "1-2",
                                "42,13", "--exe", str(exe), "--dump", "--min-free-gb", "0"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("RUN-OK", r.stdout)
            out = t / "pics" / "set1"
            rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
            self.assertEqual([(x["prompt"], x["seed"], x["rc"]) for x in rows],
                             [(1, 42, 0), (1, 13, 1), (2, 42, 0), (2, 13, 1)])
            ok = rows[0]
            self.assertEqual((ok["open_s"], ok["text_encode_s"], ok["sampling_s"], ok["vae_decode_s"], ok["png_s"]),
                             (0.7, 0.15, 9.8, 1.2, 0.03))
            self.assertEqual((ok["steps_ours_gb"], ok["decode_driver_gb"], ok["plan"]), (1.5, 2.0, "resident"))
            self.assertIn("something broke", rows[1]["error"])
            self.assertTrue((out / "p1_s42.png").is_file() and (out / "p2_s42.json").is_file())
            args = [json.loads(x) for x in Path(str(exe) + ".args").read_text().splitlines()]
            a = args[0]
            v = lambda k: a[a.index(k) + 1]
            self.assertEqual(v("--prompt"), PROMPTS[1])
            self.assertEqual((v("--width"), v("--height"), v("--seed"), v("--steps")), ("512", "512", "42", "8"))
            self.assertEqual(Path(v("--dit")).name, "qwen-image-2.1-turbo-Q8_0.gguf")
            self.assertEqual(Path(v("--vae")).name, "qwen_image_2.1_vae_bf16.safetensors")
            self.assertEqual(Path(v("--te")).name, "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")
            self.assertEqual(Path(v("--dump")).name, "dump_p1_s42")
            self.assertEqual(Path(v("-o")), out / "p1_s42.png")


if __name__ == "__main__":
    unittest.main()
