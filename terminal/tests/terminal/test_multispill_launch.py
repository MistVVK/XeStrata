"""Two small cards on a layer split (MULTISPILL, round 0212): what the launcher says when a card runs out of memory.

The final line of a failed start is the engine's own reason, not the picture reader's harmless "ggml_cuda_init: ... no CUDA-capable device" that it prints when the
cards are full and it runs on the processor; the next step does not tell the person to change the context or the number of requests.  Nothing here starts an engine."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strata_cli.launch import runner as R
from strata_control import vram_fit

# the log of the 2 x 8 GB start that failed (HYBRID, srv118, 2 x RTX PRO 6000 held to 8 GB free by hogs, 128K): the engine's reason, then the picture reader's line
SERVE_STOP = [
    "strata serve: layer split, CUDA1 prompt path: prefill: device buffers for a chunk of 512 tokens do not fit (5 of 97247 MiB free at the failure; 555 MiB taken, the failing buffer wanted 2 MiB)",
    "strata serve: the GPU has too little free VRAM for the prompt path: turn images off (setup: --vision no), close other programs using the GPU, use a shorter context, or read prompts in smaller chunks (--prefill 512)",
    "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected",
]
CACHE_STOP = [
    "strata generate: layer split, CUDA1: 0.17 GiB free of 94.97, room for experts 0.00 GiB",
    "strata generate: layer split, CUDA1 expert cache: no room",
    "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected",
]


class TheEnginesReason(unittest.TestCase):
    def test_the_reason_is_the_engines_not_the_picture_readers(self):
        self.assertEqual(vram_fit.engine_stop_reason(SERVE_STOP),
                         "the GPU has too little free VRAM for the prompt path: turn images off (setup: --vision no), close other programs using the GPU, "
                         "use a shorter context, or read prompts in smaller chunks (--prefill 512)")
        self.assertEqual(vram_fit.engine_stop_reason(CACHE_STOP), "layer split, CUDA1 expert cache: no room")

    def test_other_stops_and_the_walks_own_lines_are_not_a_memory_stop(self):
        self.assertIsNone(vram_fit.engine_stop_reason(["strata generate: something else is wrong", "ggml_cuda_init: failed to initialize CUDA"]))
        self.assertIsNone(vram_fit.engine_stop_reason([]))
        self.assertIsNone(vram_fit.engine_stop_reason(None))
        walk = ["strata generate: short of GPU memory: CUDA1 has 2.53 GiB free and a layer split card needs about 9.1 GiB: each later card loads only the dense weights of its own layers",
                "strata serve: CUDA1: 293 MiB of VRAM free with everything loaded - LOW: its window graphs are captured on first use"]
        self.assertIsNone(vram_fit.engine_stop_reason(walk), "what the walk did and a LOW warning say nothing about why a start stopped")

    def test_the_last_memory_line_wins(self):
        lines = ["strata generate: layer split, CUDA1 expert cache: no room", "strata serve: the GPU has too little free VRAM for the prompt path: x"]
        self.assertTrue(vram_fit.engine_stop_reason(lines).startswith("the GPU has too little free VRAM"))


class TheLastLine(unittest.TestCase):
    def explain(self, lines, **result):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            cfg = Path(tmp) / "run.json"
            cfg.write_text(json.dumps({"exe": "e", "args": [], "log": str(log)}), encoding="utf-8")
            return R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 1, **result}, cfg)

    def test_a_prompt_path_stop_shows_the_engines_line_and_a_next_step_that_keeps_the_context(self):
        text, nxt = self.explain(SERVE_STOP)
        self.assertIn("too little free VRAM for the prompt path", text)
        self.assertNotIn("ggml_cuda_init", text)
        self.assertNotIn("no CUDA-capable device", text)
        self.assertIn("were not changed", nxt, "the context and the requests stay what the person chose")
        self.assertIn("fewer cards", nxt)
        self.assertNotIn("--context 16k", nxt, "the old advice asked for a smaller context")

    def test_a_stage_cache_stop_shows_the_engines_line(self):
        text, _ = self.explain(CACHE_STOP)
        self.assertIn("layer split, CUDA1 expert cache: no room", text)
        self.assertNotIn("ggml_cuda_init", text)

    def test_a_stop_for_another_reason_is_worded_as_before(self):
        text, nxt = self.explain(["strata generate: cudaMalloc failed", "unsupported quant type 99"])
        self.assertIn("unsupported quant type 99", text)
        self.assertNotIn("were not changed", nxt)


if __name__ == "__main__":
    unittest.main()
