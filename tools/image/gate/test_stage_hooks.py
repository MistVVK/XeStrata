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
                                      "            if (output_opt.empty()) {\n"
                                      "    auto latents                      = first_stage_model->diffusion_to_vae_latents(x);\n"),
    "conditioning/conditioner.hpp": ('#include "core/util.h"\n'
                                     "        auto hidden_states = llm->compute(n_threads,\n"
                                     "                                          input_ids,\n"
                                     "                                          sd::Tensor<float>(),\n"
                                     + S.TE_CALL),
    "model/diffusion/qwen_image_2_1.hpp": ('#include "model/diffusion/qwen_image.hpp"\n'
                                           "                pe_data          = finish_rope_pe(std::move(embedding));\n"),
    "model/diffusion/qwen_image_2_1_native.hpp": "            std::vector<float> pe_all = finish_rope_pe(std::move(emb));\n",
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
            self.assertEqual(cond.count("own5::dump_ids(tokens, weights, prompt_template_encode_start_idx);\n" + S.TE_CALL), 1)
            self.assertEqual(cond.count('#include "own5_hooks.hpp"'), 1)
            # the other llm->compute call (no attention mask) is left alone
            self.assertEqual(cond.count("own5::dump_ids"), 1)
            eng = (dst / "src" / "pipeline" / "diffusion_engine.cpp").read_bytes().decode()
            self.assertEqual(eng.count("own5::dit_in_hook(diffusion_params.x, diffusion_params.timesteps);\n"
                                       "            auto output_opt"), 1)
            self.assertEqual(eng.count('own5::dump("vae_in", latents.data()'), 1)
            img = (dst / "src" / "pipeline" / "image.cpp").read_bytes().decode()
            self.assertEqual(img.count('own5::dump("vae_out", image.data()'), 1)
            q21 = (dst / "src" / "model" / "diffusion" / "qwen_image_2_1.hpp").read_bytes().decode()
            self.assertEqual(q21.count('own5::pe_hook(pe_data, "stock");'), 1)
            nat = (dst / "src" / "model" / "diffusion" / "qwen_image_2_1_native.hpp").read_bytes().decode()
            self.assertEqual(nat.count('own5::pe_hook(pe_all, "native");'), 1)
            hooks = (dst / "src" / "own5_hooks.hpp").read_text()
            self.assertIn("inline void dump_ids(const std::vector<int>& ids, const std::vector<float>& weights, int start)", hooks)
            before = {p: p.read_bytes() for p in dst.rglob("*") if p.is_file()}
            self.run_hooks(src, dst)
            after = {p: p.read_bytes() for p in dst.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual((src / "src" / "conditioning" / "conditioner.hpp").read_bytes().decode(),
                             FILES["conditioning/conditioner.hpp"])


if __name__ == "__main__":
    unittest.main()
