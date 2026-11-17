"""tools/image/sdcpp_dit/apply.py - put Strata's own DiT matrix products (src/image/dit/dit_gemm.cu: our weight decoders,
f16 activations, cuBLAS / hipBLAS GemmEx with f32 accumulation) into the native Qwen-Image-2.1 runner of a
stable-diffusion.cpp checkout at the pinned commit.  Step 4, piece D1: TEST BUILDS ONLY for now (build_variant.py
--dit); setup does not apply it, so no user build changes.

    python tools/image/sdcpp_dit/apply.py third_party/stable-diffusion.cpp      (after sdcpp_native/apply.py)

What it does:
  * copies src/image/dit/dit_gemm.cu, src/image/te/te_dequant.cu and the headers they use into src/strata_dit/;
  * CMakeLists.txt: a static library strata_dit (CUDA or HIP language, its own flags) linked into the
    stable-diffusion library, with the cuBLAS / hipBLAS ggml links already; SD_STRATA_DIT defined;
  * the native runner (src/model/diffusion/qwen_image_2_1_native.hpp): with --model-args strata_dit_gemm=1 (and
    the native runner on), every product the runner sends through its gemm() - img_in and the four per block - runs
    on dit_gemm instead of a ggml_mul_mat graph; without the argument the runner is unchanged.  On an error the
    runner goes back to ggml's product for the rest of the run (test-build log line, not user text)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if (HERE.parents[2] / "src").is_dir() else HERE.parents[3] / "engine"   # the engine's sources (../engine in a 0.2.16 checkout)
MARK = "// strata: own DiT GEMMs"

FILES = [
    ("src/image/dit/dit_gemm.cu", "dit_gemm.cu"),
    ("src/image/te/te_dequant.cu", "te_dequant.cu"),
    ("include/strata/image/dit_gemm.hpp", "include/strata/image/dit_gemm.hpp"),
    ("include/strata/image/te_dequant.hpp", "include/strata/image/te_dequant.hpp"),
    ("include/strata/kernels/f16_bits.hpp", "include/strata/kernels/f16_bits.hpp"),
    ("include/strata/kernels/bf16_bits.hpp", "include/strata/kernels/bf16_bits.hpp"),
]
HIP_COMPAT = "include/strata/hip_compat"

NAT_INCLUDE_OLD = '#include "ggml-alloc.h"\n'
NAT_INCLUDE_NEW = NAT_INCLUDE_OLD + "#ifdef SD_STRATA_DIT\n#include <cmath>\n#include <vector>\n#include \"strata/image/dit_gemm.hpp\"  " + MARK + "\n#endif\n"
NAT_FLAG_OLD = "        bool fuse_qkv   = true;\n"
NAT_FLAG_NEW = NAT_FLAG_OLD + ("        bool strata_gemm = false;   " + MARK + " (--model-args strata_dit_gemm=1)\n"
                               "        int strata_groups = 15;     // test builds, STRATA_DIT_GROUPS: 1 blocks + img_in, 2 timestep MLP +"
                               " modulation + norm_out, 4 txt_in, 8 head\n")
NAT_CTOR_OLD = "            gbuf.resize(64 << 20);\n"
NAT_CTOR_NEW = NAT_CTOR_OLD + ("#ifdef SD_STRATA_DIT\n"
                               "            strata_gemm = model_args != nullptr && std::strstr(model_args, \"strata_dit_gemm=1\") != nullptr;\n"
                               "            if (const char* gs = getenv(\"STRATA_DIT_GROUPS\")) strata_groups = std::atoi(gs);\n"
                               "#endif\n")
NAT_GEMM_OLD = """        void gemm(G& g, ggml_tensor* weight, ggml_tensor* src, ggml_tensor* dst, int64_t n) {
            ggml_tensor* y = ggml_mul_mat(g.ctx, weight, rows(g.ctx, src, n));
"""
NAT_GEMM_NEW = """        void gemm(G& g, ggml_tensor* weight, ggml_tensor* src, ggml_tensor* dst, int64_t n) {
#ifdef SD_STRATA_DIT
            if (strata_gemm && (strata_groups & 1)) {   """ + MARK + """: y[n][N] packed into dst, as the ggml_mul_mat node below writes it
                const std::string e = strata::image::dit_gemm(qi21_stream(runtime_backend), (uint32_t) weight->type,
                                                              weight->data, weight->ne[0], weight->ne[1],
                                                              (const float*) src->data, (int64_t) (src->nb[1] / sizeof(float)),
                                                              n, (float*) dst->data, weight->ne[1]);
                if (e.empty()) return;
                LOG_WARN("strata DiT GEMM (test build): %s; ggml's product from here on", e.c_str());
                strata_gemm = false;
            }
#endif
            ggml_tensor* y = ggml_mul_mat(g.ctx, weight, rows(g.ctx, src, n));
"""
# D2: the prelude (timestep MLP, modulation, norm_out.linear), txt_in and the output head through dit_gemm too; the
# element-wise steps between them stay small ggml graphs.  Scratch in `gu` from float offset kscr() on (run() puts the
# graphs' own unbound nodes at its start).
NAT_HELPERS_OLD = "        // ---------------------------------------------------------------- the transformer blocks\n"
NAT_HELPERS_NEW = """#ifdef SD_STRATA_DIT
        """ + MARK + """: helpers for the prelude / txt_in / head
        // past what run() may give a graph's own intermediates (up to rows x C + a C-vector), 256-float aligned
        size_t kscr() const { return ((size_t) std::max(t_cap, l_cap) * C + 2 * C + 255) / 256 * 256; }
        bool sgemm(ggml_tensor* weight, const float* xin, int64_t ldx, int64_t n, float* y, int64_t ldy) {
            const std::string e = strata::image::dit_gemm(qi21_stream(runtime_backend), (uint32_t) weight->type,
                                                          weight->data, weight->ne[0], weight->ne[1], xin, ldx, n, y, ldy);
            if (!e.empty()) LOG_WARN("strata DiT GEMM (test build): %s", e.c_str());
            return e.empty();
        }
        float* scr(size_t off) { return (float*) gu->data + kscr() + off; }
        ggml_tensor* sview(G& g, int64_t ne0, int64_t ne1, size_t off) {
            return ggml_view_2d(g.ctx, gu, ne0, ne1, ne0 * sizeof(float), (kscr() + off) * sizeof(float));
        }
#endif
""" + NAT_HELPERS_OLD
NAT_PRELUDE_OLD = """            ggml_backend_tensor_set(timein, tv, 0, sizeof(tv));
            {
                G g = begin();
                ggml_tensor* te = ggml_timestep_embedding(g.ctx, timein, 256, 10000);
"""
NAT_PRELUDE_NEW = """            ggml_backend_tensor_set(timein, tv, 0, sizeof(tv));
#ifdef SD_STRATA_DIT
            if (strata_gemm && (strata_groups & 2)) {   """ + MARK + """: timestep MLP, modulation and the final scale
                const size_t o1 = 512, o2 = o1 + 2 * C, o3 = o2 + 2 * C, o4 = o3 + 2 * C;
                { G g = begin(); ggml_tensor* te = ggml_timestep_embedding(g.ctx, timein, 256, 10000);
                  bind(te, gu, kscr() * sizeof(float)); run(g, {te}); }
                bool ok = sgemm(w("time_text_embed.timestep_embedder.linear_1.weight"), scr(0), 256, 2, scr(o1), C);
                { G g = begin(); ggml_tensor* s1 = ggml_silu(g.ctx, sview(g, C, 2, o1));
                  bind(s1, gu, (kscr() + o2) * sizeof(float)); run(g, {s1}); }
                ok = ok && sgemm(w("time_text_embed.timestep_embedder.linear_2.weight"), scr(o2), C, 2, scr(o3), C);
                { G g = begin(); ggml_tensor* s2 = ggml_silu(g.ctx, sview(g, C, 2, o3));
                  bind(s2, gu, (kscr() + o4) * sizeof(float)); run(g, {s2}); }
                ok = ok && sgemm(w("modulation.1.weight"), scr(o4), C, 2, (float*) mod->data, 4 * C);
                ok = ok && sgemm(w("norm_out.linear.weight"), scr(o4), C, 1, (float*) scale_out->data, C);
                if (!ok) {
                    fail("the strata DiT GEMMs failed (test build)");
                    return QwenImage21Runner::compute(n_threads, inputs);
                }
            } else
#endif
            {
                G g = begin();
                ggml_tensor* te = ggml_timestep_embedding(g.ctx, timein, 256, 10000);
"""
NAT_TXT_OLD = """                {
                    // txt_in: zero-centered RMSNorm (weight + 1) -> in_layer -> GELU -> out_layer, into x
"""
NAT_TXT_NEW = """#ifdef SD_STRATA_DIT
                if (strata_gemm && (strata_groups & 4)) {   """ + MARK + """: txt_in
                    const size_t o2 = (size_t) Lt * C, o3 = 2 * (size_t) Lt * C;
                    { G g = begin();
                      ggml_tensor* t  = ggml_rms_norm(g.ctx, rows(g.ctx, ctx_in, Lt), 1e-6f);
                      ggml_tensor* nw = ggml_scale_bias(g.ctx, w("txt_in.text_norm.weight"), 1.f, 1.f);
                      t = ggml_mul(g.ctx, t, nw);
                      bind(t, gu, kscr() * sizeof(float)); run(g, {t}); }
                    bool ok = sgemm(w("txt_in.in_layer.weight"), scr(0), config.context_dim, Lt, scr(o2), C);
                    { G g = begin(); ggml_tensor* s = ggml_gelu(g.ctx, sview(g, C, Lt, o2));
                      bind(s, gu, (kscr() + o3) * sizeof(float)); run(g, {s}); }
                    ok = ok && sgemm(w("txt_in.out_layer.weight"), scr(o3), C, Lt, (float*) x->data, C);
                    if (!ok) {
                        fail("the strata DiT GEMMs failed (test build)");
                        return QwenImage21Runner::compute(n_threads, inputs);
                    }
                } else
#endif
""" + NAT_TXT_OLD
NAT_HEAD_OLD = """            {
                G g = begin();
                ggml_tensor* t = ggml_norm(g.ctx, rows(g.ctx, x, T), 1e-6f);
"""
NAT_HEAD_NEW = """#ifdef SD_STRATA_DIT
            if (strata_gemm && (strata_groups & 8)) {   """ + MARK + """: the output head
                { G g = begin();
                  ggml_tensor* t = ggml_norm(g.ctx, rows(g.ctx, x, T), 1e-6f);
                  t = ggml_mul(g.ctx, t, ggml_scale_bias(g.ctx, scale_out, 1.f, 1.f));
                  bind(t, gu, kscr() * sizeof(float)); run(g, {t}); }
                if (!sgemm(w("proj_out.weight"), scr(0), C, T, (float*) outp->data, config.out_channels)) {
                    fail("the strata DiT GEMMs failed (test build)");
                    return QwenImage21Runner::compute(n_threads, inputs);
                }
            } else
#endif
""" + NAT_HEAD_OLD
# Test builds only: STRATA_DIT_TRACE_DIR=<dir> writes, for one DiT call, the residual stream after each block (txt_NN.bin
# for the text-prefix pass, img_NN.bin for the image pass; float32 rows x 4096), the prelude's outputs (mod.bin,
# scale.bin, txtin.bin, imgin.bin), the velocity rows (out.bin), and trace.txt: per block the largest |value| of every
# GEMM input (qkv, to_out, gate_up, out) and attention input (q, k, v).  Changes no value (it only reads, after a sync).
TRACE_HELPERS = """#ifdef SD_STRATA_DIT
        """ + MARK + """: trace (test builds; STRATA_DIT_TRACE_DIR)
        const char* trace_dir = getenv("STRATA_DIT_TRACE_DIR");
        const int trace_block = getenv("STRATA_DIT_TRACE_BLOCK") ? std::atoi(getenv("STRATA_DIT_TRACE_BLOCK")) : -1;
        void trace_file(const std::string& name, ggml_tensor* t, size_t bytes) {
            ggml_backend_synchronize(runtime_backend);   // tensor_get copies on another stream: wait for this one
            std::vector<char> buf(bytes);
            ggml_backend_tensor_get(t, buf.data(), 0, bytes);
            if (FILE* f = std::fopen((std::string(trace_dir) + "/" + name).c_str(), "wb")) { std::fwrite(buf.data(), 1, bytes, f); std::fclose(f); }
        }
        float trace_max(ggml_tensor* t, size_t count, bool half) {
            ggml_backend_synchronize(runtime_backend);
            std::vector<uint16_t> h16;
            std::vector<float> f32;
            float m = 0.f;
            if (half) {
                h16.resize(count);
                ggml_backend_tensor_get(t, h16.data(), 0, count * 2);
                for (uint16_t v : h16) { const float a = std::fabs(ggml_fp16_to_fp32(v)); if (!(a <= m)) m = a; }
            } else {
                f32.resize(count);
                ggml_backend_tensor_get(t, f32.data(), 0, count * 4);
                for (float v : f32) { const float a = std::fabs(v); if (!(a <= m)) m = a; }
            }
            return m;
        }
        void trace_line(const std::string& s) {
            if (FILE* f = std::fopen((std::string(trace_dir) + "/trace.txt").c_str(), "a")) { std::fprintf(f, "%s\\n", s.c_str()); std::fclose(f); }
        }
#endif
"""
TR_QKV_OLD = """                                   1e-6f);
                if (rc) break;
                const size_t pbytes = (size_t) L * D * sizeof(ggml_fp16_t);   // one head's prefix rows
"""
TR_QKV_NEW = """                                   1e-6f);
                if (rc) break;
#ifdef SD_STRATA_DIT
                if (trace_dir && !prefix && i == trace_block) {   // block detail: the qkv GEMM output, Q / K / V as attention reads them
                    if (wqkv != nullptr) trace_file("blk_qkv.bin", gu, (size_t) n * 3 * C * sizeof(float));
                    trace_file("blk_q16.bin", q16, (size_t) H * n * D * (use_wmma ? 2 : 4));
                    trace_file("blk_k16.bin", kw, (size_t) H * kvt * D * 2);
                    trace_file("blk_v16.bin", vw, (size_t) H * kvt * D * 2);
                }
                if (trace_dir) {   """ + MARK + """: trace
                    char tl[256];
                    std::snprintf(tl, sizeof tl, "%s %d qkv_in %.6g q %.6g k %.6g v %.6g", prefix ? "txt" : "img", i,
                                  trace_max(h, (size_t) n * C, false), trace_max(q16, (size_t) H * n * D, use_wmma),
                                  trace_max(kw, (size_t) H * kvt * D, true), trace_max(vw, (size_t) H * kvt * D, true));
                    trace_line(tl);
                }
#endif
                const size_t pbytes = (size_t) L * D * sizeof(ggml_fp16_t);   // one head's prefix rows
"""
TR_MID_OLD = """                rc = res_ln(X, Hh, m1, t1, Hh, m2);
                if (rc) break;
"""
TR_MID_NEW = TR_MID_OLD + """#ifdef SD_STRATA_DIT
                if (trace_dir && !prefix && i == trace_block) {   // block detail: attention output, stream, LN2 output
                    trace_file("blk_att.bin", att, (size_t) n * C * sizeof(float));
                    trace_file("blk_x_mid.bin", x, (size_t) n * C * sizeof(float));
                    trace_file("blk_ln2.bin", h, (size_t) n * C * sizeof(float));
                }
                if (trace_dir) {
                    char tl[160];
                    std::snprintf(tl, sizeof tl, "%s %d to_out_in %.6g gate_up_in %.6g", prefix ? "txt" : "img", i,
                                  trace_max(att, (size_t) n * C, false), trace_max(h, (size_t) n * C, false));
                    trace_line(tl);
                }
#endif
"""
TR_SWI_OLD = """                rc = qi21_swiglu(s, (const float*) gu->data, (float*) mlp->data, (int) n, (int) F);
                if (rc) break;
"""
TR_SWI_NEW = TR_SWI_OLD + """#ifdef SD_STRATA_DIT
                if (trace_dir && !prefix && i == trace_block) trace_file("blk_act.bin", mlp, (size_t) n * F * sizeof(float));
                if (trace_dir) {
                    char tl[96];
                    std::snprintf(tl, sizeof tl, "%s %d out_in %.6g", prefix ? "txt" : "img", i, trace_max(mlp, (size_t) n * F, false));
                    trace_line(tl);
                }
#endif
"""
TR_END_OLD = "                rc = res_ln(X, Hh, m3, t3, last ? nullptr : Hh, m0);\n"
TR_END_NEW = TR_END_OLD + """#ifdef SD_STRATA_DIT
                if (trace_dir && rc == 0) {
                    char fn[32];
                    std::snprintf(fn, sizeof fn, "%s_%02d.bin", prefix ? "txt" : "img", i);
                    trace_file(fn, x, (size_t) n * C * sizeof(float));
                }
#endif
"""
TR_MOD_OLD = "            if (qi21_tanh(qi21_stream(runtime_backend), (const float*) mod->data, (float*) gates->data,\n"
TR_MOD_NEW = """#ifdef SD_STRATA_DIT
            if (trace_dir) { trace_file("mod.bin", mod, 8 * C * sizeof(float)); trace_file("scale.bin", scale_out, C * sizeof(float)); }
#endif
""" + TR_MOD_OLD
TR_TXT_OLD = "                if (!blocks_pass(Lt, 1, pe_txt, 0)) {\n"
TR_TXT_NEW = """#ifdef SD_STRATA_DIT
                if (trace_dir) trace_file("txtin.bin", x, (size_t) Lt * C * sizeof(float));
#endif
""" + TR_TXT_OLD
TR_IMG_OLD = "            if (!blocks_pass(T, 0, pe_img, Lt)) {\n"
TR_IMG_NEW = """#ifdef SD_STRATA_DIT
            if (trace_dir) trace_file("imgin.bin", x, (size_t) T * C * sizeof(float));
#endif
""" + TR_IMG_OLD
TR_OUT_OLD = "            ggml_backend_tensor_get(outp, o.data(), 0, o.size() * sizeof(float));\n"
TR_OUT_NEW = TR_OUT_OLD + """#ifdef SD_STRATA_DIT
            if (trace_dir) trace_file("out.bin", outp, o.size() * sizeof(float));
#endif
"""
CMAKE_OLD = "target_include_directories(${SD_LIB} PUBLIC . src include)\n"
CMAKE_NEW = """if (SD_CUDA OR SD_HIPBLAS)  # """ + MARK.lstrip("/ ") + """: src/strata_dit (tools/image/sdcpp_dit/apply.py)
    set(STRATA_DIT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/strata_dit)
    set(STRATA_DIT_SRC ${STRATA_DIT_DIR}/dit_gemm.cu ${STRATA_DIT_DIR}/te_dequant.cu)
    if (SD_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_library(strata_dit STATIC ${STRATA_DIT_SRC})
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_target_properties(strata_dit PROPERTIES CUDA_ARCHITECTURES native)
        endif()
        if (GGML_STATIC AND NOT WIN32)
            target_link_libraries(strata_dit PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        elseif (GGML_STATIC)
            target_link_libraries(strata_dit PRIVATE CUDA::cudart_static CUDA::cublas)
        else()
            target_link_libraries(strata_dit PRIVATE CUDA::cudart CUDA::cublas)
        endif()
    else()
        if (AMDGPU_TARGETS AND NOT GPU_TARGETS)
            set(GPU_TARGETS ${AMDGPU_TARGETS})
        endif()
        if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
        endif()
        enable_language(HIP)
        find_package(hip REQUIRED)
        find_package(hipblas REQUIRED)
        set_source_files_properties(${STRATA_DIT_SRC} PROPERTIES LANGUAGE HIP)
        add_library(strata_dit STATIC ${STRATA_DIT_SRC})
        target_link_libraries(strata_dit PRIVATE hip::host roc::hipblas)
        set_target_properties(strata_dit ${SD_LIB} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    endif()
    target_include_directories(strata_dit PRIVATE ${STRATA_DIT_DIR} ${STRATA_DIT_DIR}/include)
    target_compile_features(strata_dit PRIVATE cxx_std_17)
    set_target_properties(strata_dit PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${SD_LIB} PRIVATE strata_dit)
    target_include_directories(${SD_LIB} PRIVATE ${STRATA_DIT_DIR}/include)
    target_compile_definitions(${SD_LIB} PRIVATE SD_STRATA_DIT)
endif()
""" + CMAKE_OLD


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise SystemExit(f"{path}: the expected code is not there (another stable-diffusion.cpp commit, or no native runner?)")
    path.write_bytes(text.replace(old, new).encode("utf-8"))


def copy_sources(dst: Path) -> None:
    for src, rel in FILES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / src, dst / rel)
    hc = dst / HIP_COMPAT
    hc.mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / HIP_COMPAT).iterdir()):
        if f.is_file():
            shutil.copyfile(f, hc / f.name)


def apply(sd: Path) -> None:
    sd = Path(sd)
    nat = sd / "src" / "model" / "diffusion" / "qwen_image_2_1_native.hpp"
    if not nat.is_file():
        raise SystemExit(f"{nat}: the native runner is not applied (tools/image/sdcpp_native/apply.py first)")
    copy_sources(sd / "src" / "strata_dit")
    edit(nat, NAT_INCLUDE_OLD, NAT_INCLUDE_NEW)
    edit(nat, NAT_FLAG_OLD, NAT_FLAG_NEW)
    edit(nat, NAT_CTOR_OLD, NAT_CTOR_NEW)
    edit(nat, NAT_GEMM_OLD, NAT_GEMM_NEW)
    # helpers and trace helpers in one edit: a second apply finds the whole new text and changes nothing
    edit(nat, NAT_HELPERS_OLD, NAT_HELPERS_NEW.replace(NAT_HELPERS_OLD, TRACE_HELPERS + NAT_HELPERS_OLD))
    edit(nat, NAT_PRELUDE_OLD, NAT_PRELUDE_NEW)
    edit(nat, NAT_TXT_OLD, NAT_TXT_NEW)
    edit(nat, NAT_HEAD_OLD, NAT_HEAD_NEW)
    for old, new in ((TR_QKV_OLD, TR_QKV_NEW), (TR_MID_OLD, TR_MID_NEW), (TR_SWI_OLD, TR_SWI_NEW), (TR_END_OLD, TR_END_NEW),
                     (TR_MOD_OLD, TR_MOD_NEW), (TR_TXT_OLD, TR_TXT_NEW), (TR_IMG_OLD, TR_IMG_NEW), (TR_OUT_OLD, TR_OUT_NEW)):
        edit(nat, old, new)
    edit(sd / "CMakeLists.txt", CMAKE_OLD, CMAKE_NEW)


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        apply(Path(sys.argv[1]))
        print("Strata DiT GEMMs applied (native runner, --model-args strata_dit_gemm=1)")
    else:
        raise SystemExit(__doc__)
