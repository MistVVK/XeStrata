"""tools/image/gate/stage_hooks.py - TEST-ONLY (OWN-5 gate harness, never in a shipped build path): make a copy of a
pinned stable-diffusion.cpp tree with stage dump / replace hooks, driven by environment variables:

  OWN5_DUMP_DIR=<dir>   write noise.bin (the initial noise), cond.bin (text-encoder hidden states, c_crossattn),
                        vel_<k>.bin (the DiT output of model call k = 0, 1, ...), latent.bin (the latent before VAE
                        decode); each .bin is float32 with a .shape text file beside it
  OWN5_NOISE_DIR=<dir>  take the initial noise from <dir>/noise_<seed>.bin instead of PhiloxRNG
  OWN5_COND_IN=<file>   replace c_crossattn by this file (same shape)
  OWN5_LATENT_IN=<file> replace the final latent by this file before VAE decode (same shape)
  OWN5_EXIT_AFTER=<k>   exit(0) right after model call k was dumped (no picture)
  (with OWN5_DUMP_DIR)  also tokens.bin: the text encoder's input token ids (int32) as LLMEmbedder::encode_prompt
                        builds them, and tokens.start: prompt_template_encode_start_idx (the system tokens cond drops)

Without the variables the copy behaves like the original.  Usage: python3 stage_hooks.py <src tree> <dst tree>"""
import shutil
import sys
from pathlib import Path

HOOKS = r'''// own5_hooks.hpp - TEST-ONLY stage hooks (tools/image/gate/stage_hooks.py)
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/tensor.hpp"
namespace own5 {
inline const char* env(const char* k) { const char* v = std::getenv(k); return v && *v ? v : nullptr; }
inline void dump(const char* name, const float* p, size_t n, const std::vector<int64_t>& shape) {
    const char* d = env("OWN5_DUMP_DIR");
    if (!d) return;
    std::string base = std::string(d) + "/" + name;
    if (FILE* f = std::fopen((base + ".bin").c_str(), "wb")) { std::fwrite(p, sizeof(float), n, f); std::fclose(f); }
    if (FILE* f = std::fopen((base + ".shape").c_str(), "w")) {
        for (auto s : shape) std::fprintf(f, "%lld ", (long long) s);
        std::fprintf(f, "\n");
        std::fclose(f);
    }
}
inline void dump_ids(const std::vector<int>& ids, int start) {
    const char* d = env("OWN5_DUMP_DIR");
    if (!d) return;
    std::string base = std::string(d) + "/tokens";
    if (FILE* f = std::fopen((base + ".bin").c_str(), "wb")) { std::fwrite(ids.data(), sizeof(int), ids.size(), f); std::fclose(f); }
    if (FILE* f = std::fopen((base + ".start").c_str(), "w")) { std::fprintf(f, "%d\n", start); std::fclose(f); }
}
inline void load(const char* file, float* p, size_t n, const char* what) {
    FILE* f = std::fopen(file, "rb");
    size_t got = f ? std::fread(p, sizeof(float), n, f) : 0;
    if (f) std::fclose(f);
    if (got != n) { std::fprintf(stderr, "OWN5: %s %s missing or short (%zu of %zu)\n", what, file, got, n); std::abort(); }
    std::fprintf(stderr, "OWN5: %s from %s\n", what, file);
}
inline void tensor_hook(sd::Tensor<float>& t, const char* name, const char* in_var) {
    if (t.numel() == 0) return;
    if (const char* in = env(in_var)) load(in, t.data(), (size_t) t.numel(), name);
    dump(name, t.data(), (size_t) t.numel(), t.shape());
}
inline void vel_hook(const sd::Tensor<float>& t) {
    static int k = 0;
    std::string name = "vel_" + std::to_string(k);
    dump(name.c_str(), t.data(), (size_t) t.numel(), t.shape());
    if (const char* e = env("OWN5_EXIT_AFTER")) {
        if (k >= std::atoi(e)) { std::fprintf(stderr, "OWN5: exit after model call %d\n", k); std::fflush(nullptr); std::exit(0); }
    }
    ++k;
}
}  // namespace own5
'''

NOISE_OLD = "    std::vector<float> randn(uint32_t n) override {\n"
NOISE_NEW = NOISE_OLD + r'''        if (const char* own5_dir = std::getenv("OWN5_NOISE_DIR")) {   // OWN-5 test hook
            std::string p = std::string(own5_dir) + "/noise_" + std::to_string(this->seed) + ".bin";
            std::vector<float> r(n);
            FILE* f = std::fopen(p.c_str(), "rb");
            size_t got = f ? std::fread(r.data(), sizeof(float), n, f) : 0;
            if (f) std::fclose(f);
            if (got != n) { std::fprintf(stderr, "OWN5: noise file %s missing or short\n", p.c_str()); std::abort(); }
            this->offset += 1;
            return r;
        }
'''

# LLMEmbedder::encode_prompt's text-encoder call (conditioning/conditioner.hpp; the one with attention_mask)
TE_CALL = ("        auto hidden_states = llm->compute(n_threads,\n"
           "                                          input_ids,\n"
           "                                          attention_mask,\n"
           "                                          image_embeds,\n"
           "                                          out_layers,\n")


def edit(path: Path, old: str, new: str, count=1):
    t = path.read_text()
    if new in t:
        return
    if t.count(old) != count:
        raise SystemExit(f"{path}: expected {count} x {old!r}")
    path.write_text(t.replace(old, new))


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not dst.exists():
        shutil.copytree(src, dst, symlinks=True)
    (dst / "src" / "own5_hooks.hpp").write_text(HOOKS)
    h = dst / "src" / "core" / "rng_philox.hpp"
    edit(h, "#include <vector>\n", "#include <vector>\n#include <cstdio>\n#include <cstdlib>\n#include <string>\n")
    edit(h, NOISE_OLD, NOISE_NEW)
    img = dst / "src" / "pipeline" / "image.cpp"
    edit(img, '#include "core/rng.hpp"\n', '#include "core/rng.hpp"\n#include "own5_hooks.hpp"\n')
    edit(img, "        auto cond                        = sd->get_learned_condition(condition_params);\n",
         "        auto cond                        = sd->get_learned_condition(condition_params);\n"
         '        own5::tensor_hook(cond.c_crossattn, "cond", "OWN5_COND_IN");\n')
    edit(img, "            sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd->rng);\n",
         "            sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd->rng);\n"
         '            own5::tensor_hook(noise, "noise", "OWN5_NOISE_IN_UNUSED");\n')
    edit(img, "                sd::Tensor<float> image = sd->decode_first_stage(final_latents[i]);\n",
         '                own5::tensor_hook(const_cast<sd::Tensor<float>&>(final_latents[i]), "latent", "OWN5_LATENT_IN");\n'
         "                sd::Tensor<float> image = sd->decode_first_stage(final_latents[i]);\n")
    eng = dst / "src" / "pipeline" / "diffusion_engine.cpp"
    edit(eng, '#include "core/rng.hpp"\n', '#include "core/rng.hpp"\n#include "own5_hooks.hpp"\n')
    edit(eng, "            auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);\n"
              "            if (output_opt.empty()) {\n",
         "            auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);\n"
         "            own5::vel_hook(output_opt);\n"
         "            if (output_opt.empty()) {\n")
    cond = dst / "src" / "conditioning" / "conditioner.hpp"
    edit(cond, '#include "core/util.h"\n', '#include "core/util.h"\n#include "own5_hooks.hpp"\n')
    edit(cond, TE_CALL, "        own5::dump_ids(tokens, prompt_template_encode_start_idx);\n" + TE_CALL)
    print("stage hooks applied:", dst)


if __name__ == "__main__":
    main()
