"""Which Qwen3.6-35B-A3B (GGUF architecture qwen35moe) files the Qwen3.6 engine can load: a mirror of the three checks its loader makes
(src/qwen35/q36_engine.cu, load()), so that "use my own files" names family qwen36 only for a file the engine runs and refuses the others
with the plain "not supported" words before anything is set up.

* routed experts (``*_exps``): gate and up the same type, that type one the grouped expert kernels take for gate/up (STRATA_GU_FMTS) and the
  down projection one they take for down (STRATA_D_FMTS), in the release build (STRATA_Q6K_EXPERTS=ON: packaging/build_*.sh) - the loader's
  native_expert_supported (src/kernels/cuda/iq_kernels.cu);
* the matrices it loads with mat() (attention, the shared expert, the GDN projections, the MTP block's eh_proj, output): a type the native
  MMVQ takes (native_mmvq_supported, src/kernels/cuda/native_mmvq.cu);
* the vectors it loads with f32() (norms, router, the GDN state parameters, the MTP norms): F32, F16, BF16 or Q8_0 (upload_f32);
* token_embd: any type with a known block geometry (row_bytes).

The prompt path's fused, int8 and FP16 kernels are chosen per type at run time and fall back to these, so they refuse no file; a tensor
that fails one of the checks above stops the engine at load. tests/terminal/test_q36_types.py keeps these sets equal to the CUDA source text.
"""

from __future__ import annotations

import re

# GGML type numbers (tools/gguf_reader.py GGML_TYPES)
EXPERT_GATE_UP = frozenset({16, 17, 18, 21, 22, 23, 29, 42, 12, 13, 14, 6, 2, 3, 8, 40, 144, 145, 147, 148, 149})   # STRATA_GU_FMTS (Q6K build)
EXPERT_DOWN = frozenset({20, 23, 42, 7, 6, 2, 3, 8, 12, 13, 14, 40, 144, 146, 147, 148, 149})                      # STRATA_D_FMTS (Q6K build)
MATRIX = frozenset({2, 6, 7, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29})                             # native_mmvq_supported
VECTOR = frozenset({0, 1, 30, 8, 11, 12, 13, 14, 23})                                                                              # upload_f32: F32 F16 BF16 Q8_0, Q3_K-Q6_K, IQ4_XS

_VECTOR_NAMES = re.compile(r"(attn_norm|post_attention_norm|ssm_alpha|ssm_beta|ssm_dt|ssm_a|ssm_conv1d|ssm_norm|attn_q_norm|attn_k_norm|"
                           r"ffn_gate_inp|ffn_gate_inp_shexp|nextn\.enorm|nextn\.hnorm|nextn\.shared_head_norm)(\.weight|\.bias)?$")
_MATRIX_NAMES = re.compile(r"(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp|"
                           r"nextn\.eh_proj)\.weight$")


def kind_of(name: str) -> str:
    """'expert_gate_up' | 'expert_down' | 'matrix' | 'vector' | 'embedding' | 'other' (a tensor the loader does not read)."""
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "matrix"
    if name == "output_norm.weight":
        return "vector"
    tail = name.split(".", 2)[2] if name.startswith("blk.") and name.count(".") >= 2 else name
    if tail in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
        return "expert_gate_up"
    if tail == "ffn_down_exps.weight":
        return "expert_down"
    if _VECTOR_NAMES.fullmatch(tail):
        return "vector"
    if _MATRIX_NAMES.fullmatch(tail):
        return "matrix"
    return "other"


def problems(types: dict[str, int], known_types=None) -> list[str]:
    """The tensors of one Qwen3.6 file set (name -> GGML type, every shard together) the loader would refuse, as "name: TYPE (what)" lines;
    empty when the engine loads them all. `known_types`: the type numbers with a block geometry (default: tools/gguf_reader.py's table)."""
    if known_types is None:
        from tools.gguf_reader import BLOCK_GEOMETRY, GGML_TYPES
        known_types = {t for t, n in GGML_TYPES.items() if n in BLOCK_GEOMETRY}
    out: list[str] = []
    rules = {"expert_gate_up": (EXPERT_GATE_UP, "routed expert gate/up"), "expert_down": (EXPERT_DOWN, "routed expert down"),
             "matrix": (MATRIX, "matrix"), "vector": (VECTOR, "vector")}
    for name, t in sorted(types.items()):
        kind = kind_of(name)
        if kind == "embedding":
            if t not in known_types:
                out.append(f"{name}: type {t} (token embedding)")
        elif kind in rules and t not in rules[kind][0]:
            out.append(f"{name}: type {t} ({rules[kind][1]})")
    gates = {n.rsplit(".", 2)[0]: t for n, t in types.items() if n.endswith("ffn_gate_exps.weight")}
    for n, t in types.items():
        if n.endswith("ffn_up_exps.weight") and gates.get(n.rsplit(".", 2)[0]) not in (None, t):
            out.append(f"{n}: gate and up of the routed experts differ in type")
    return out


#: setup.py OWN_PATH_FAMILIES["qwen36"]["models"] and its file name pattern (tests/terminal/test_q36_types.py keeps them the same)
FAMILY = "qwen36"
SIZES = ("UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0")
FILE = "Qwen3.6-35B-A3B-{q}.gguf"


def size_for(file_name: str, types: dict[str, int]) -> tuple[str, bool]:
    """(the catalog size setup installs this file as, whether that is the published file's own name). A file under another name is
    taken as the size whose routed experts it is closest to: Q8_0 gate/up -> Q8_0, Q6_K -> UD-Q6_K_XL, anything else -> UD-Q4_K_XL (setup
    uses the size only for its RAM and context advice; the file is read as it is)."""
    for q in SIZES:
        if file_name == FILE.format(q=q):
            return q, True
    gate = {t for n, t in types.items() if n.endswith("ffn_gate_exps.weight")}
    if gate == {8}:
        return "Q8_0", False
    if gate == {14}:
        return "UD-Q6_K_XL", False
    return "UD-Q4_K_XL", False
