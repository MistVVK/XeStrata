"""tools/test_te_tokenizer.py - our Qwen-Image text-encoder tokenizer (src/image/te/qwen2_tok.cpp) against the token ids
stable-diffusion.cpp 2988060 built for the same prompts.

The expected ids were recorded from sd.cpp itself (tools/image/gate/te_tok_collect.py: a stage-hook build of sd-cli that
dumps the ids LLMEmbedder::encode_prompt hands its text encoder) for every case of tools/image/gate/te_tok_cases.py and
stored in tools/image/gate/data/te_tok_expected.json.  The test needs no GPU, no sd.cpp and no model file: it compiles
the tokenizer with the system C++ compiler and gives it only the merges those cases use
(tools/image/gate/data/te_tok_merges.txt, with their ranks in the full 151,387-merge list).

    python -m pytest tools/test_te_tokenizer.py"""
import base64
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tools" / "image" / "gate" / "data"
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


@unittest.skipUnless(CXX, "no C++ compiler")
class TokenizerMatchesSdcpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.exe = d / "strata-te-tokens"
        r = subprocess.run([CXX, "-std=c++17", "-O2", "-I", str(ROOT / "include"), "-I", str(ROOT / "src" / "image" / "te"),
                            str(ROOT / "src" / "image" / "te" / "qwen2_tok.cpp"),
                            str(ROOT / "src" / "image" / "te" / "qwen2_tok_main.cpp"), "-o", str(cls.exe)],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise AssertionError("build failed:\n" + r.stderr[-4000:])
        cls.cases = json.loads((DATA / "te_tok_expected.json").read_text())
        lines = [f"{c['name']} {base64.b64decode(c['prompt_b64']).hex()}" for c in cls.cases]
        (d / "cases.txt").write_text("\n".join(lines) + "\n", newline="\n")      # read byte for byte: no CR on Windows
        r = subprocess.run([str(cls.exe), "run", str(DATA / "te_tok_merges.txt"), str(d / "cases.txt"), str(d / "out.txt")],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise AssertionError("run failed:\n" + r.stderr[-4000:])
        cls.got = {}
        for line in (d / "out.txt").read_text().splitlines():
            parts = line.split(" ", 2)
            cls.got[parts[0]] = parts[1:]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_case_set_is_wide(self):
        names = [c["name"] for c in self.cases]
        self.assertGreaterEqual(len(names), 380)
        for prefix in ("gate_", "long_", "special_", "gguf_literal_", "w_", "u_", "e_", "l_", "x_", "r_"):
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)

    def test_every_case_matches(self):
        bad = []
        for c in self.cases:
            g = self.got.get(c["name"])
            if g is None:
                bad.append((c["name"], "no output"))
                continue
            if "error" in c:
                if g[0] != "error":
                    bad.append((c["name"], "sd.cpp refused the prompt, we did not"))
                continue
            if g[0] != "ok":
                bad.append((c["name"], f"we refused: {g[1]}"))
                continue
            n_sys, ids, w = g[1].split(" ")
            ids = [int(x) for x in ids.split(",")] if ids else []
            want_w = [struct.unpack("<I", struct.pack("<f", x))[0] for x in c["weights"]]
            got_w = [int(x, 16) for x in w.split(",")] if w else []
            if int(n_sys) != c["n_system"]:
                bad.append((c["name"], f"n_system {n_sys} vs {c['n_system']}"))
            elif ids != c["ids"]:
                k = next((i for i, (a, b) in enumerate(zip(ids, c["ids"])) if a != b), min(len(ids), len(c["ids"])))
                bad.append((c["name"], f"ids differ at {k}: ours {ids[k:k + 5]} sd.cpp {c['ids'][k:k + 5]} "
                                       f"({len(ids)} vs {len(c['ids'])} ids)"))
            elif got_w != want_w:
                bad.append((c["name"], "weights differ"))
        self.assertEqual(bad, [], f"{len(bad)} of {len(self.cases)} cases differ")


if __name__ == "__main__":
    unittest.main()
