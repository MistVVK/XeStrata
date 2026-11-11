"""tools/image/gate/test_stage_hooks.py - TEST-ONLY: stage_hooks.py on a fake sd.cpp tree that holds only the lines it
edits: every hook lands once, the token-id dump sits right before LLMEmbedder's text-encoder call, a second run changes
nothing.

    python -m pytest tools/image/gate/test_stage_hooks.py"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_hooks as S  # noqa: E402

FILES = {
    "core/rng_philox.hpp": "#include <vector>\nstruct R {\n" + S.NOISE_OLD + "        return {};\n    }\n};\n",
    "pipeline/image.cpp": ('#include "core/rng.hpp"\n'
                           "        auto cond                        = sd->get_learned_condition(condition_params);\n"
                           "            sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd->rng);\n"
                           "                sd::Tensor<float> image = sd->decode_first_stage(final_latents[i]);\n"),
    "pipeline/diffusion_engine.cpp": ('#include "core/rng.hpp"\n'
                                      "            auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);\n"
                                      "            if (output_opt.empty()) {\n"),
    "conditioning/conditioner.hpp": ('#include "core/util.h"\n'
                                     "        auto hidden_states = llm->compute(n_threads,\n"
                                     "                                          input_ids,\n"
                                     "                                          sd::Tensor<float>(),\n"
                                     + S.TE_CALL),
}


class StageHooks(unittest.TestCase):
    def run_hooks(self, src: Path, dst: Path):
        old = sys.argv
        sys.argv = ["stage_hooks.py", str(src), str(dst)]
        try:
            S.main()
        finally:
            sys.argv = old

    def test_hooks_once_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "src_tree", Path(td) / "dst_tree"
            for rel, text in FILES.items():
                p = src / "src" / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(text.encode())
            self.run_hooks(src, dst)
            cond = (dst / "src" / "conditioning" / "conditioner.hpp").read_bytes().decode()
            self.assertEqual(cond.count("own5::dump_ids(tokens, prompt_template_encode_start_idx);\n" + S.TE_CALL), 1)
            self.assertEqual(cond.count('#include "own5_hooks.hpp"'), 1)
            # the other llm->compute call (no attention mask) is left alone
            self.assertEqual(cond.count("own5::dump_ids"), 1)
            hooks = (dst / "src" / "own5_hooks.hpp").read_text()
            self.assertIn("inline void dump_ids(const std::vector<int>& ids, int start)", hooks)
            before = {p: p.read_bytes() for p in dst.rglob("*") if p.is_file()}
            self.run_hooks(src, dst)
            after = {p: p.read_bytes() for p in dst.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual((src / "src" / "conditioning" / "conditioner.hpp").read_bytes().decode(),
                             FILES["conditioning/conditioner.hpp"])


if __name__ == "__main__":
    unittest.main()
