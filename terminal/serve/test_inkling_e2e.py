"""Comprehensive unit & E2E tests for Inkling-2-Mini serving, dialect, o200k tokenizer, shape_engine,
and live `strata-inkling` binary (--selfcheck, --info, --bench, --serve at c=1,4,8,32,64 concurrent slots + Vision)."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from serve import frontend, inkling_dialect, inkling_kind
from serve.server_parts import engines as srv_engines
from strata_control import shape_engine
import strata_tokenizer
import inkling_tokenizer


def _find_inkling_exe() -> Path | None:
    env_bin = os.environ.get("STRATA_INKLING_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    repo = ROOT.parent
    for cand in (
        repo / "build-ink" / "strata-inkling.exe",
        repo / "build-ink" / "strata-inkling",
        repo / "engine" / "build" / "src" / "inkling2" / "strata-inkling.exe",
        repo / "engine" / "build" / "src" / "inkling2" / "strata-inkling",
    ):
        if cand.is_file():
            return cand
    return None


def _build_mini_inkling_strata(out_path: Path, vision_path: Path) -> None:
    """Build a valid tiny .strata v1.1 Inkling-2-Mini model + Vision shard for fast C++ E2E testing."""
    import numpy as np
    import zstandard as zstd

    PAGE = 4096
    MAGIC = b"STRATA\x00\x01"

    def align4k(n: int) -> int:
        return ((n + PAGE - 1) // PAGE) * PAGE

    # Tiny valid Inkling2 geometry: 6 phys layers (core [2, 4) x 2 passes = 8 virtual visits, block_size=2),
    # hidden=256, inter=128, n_head=4, n_kv=2, head_dim=64, d_rel=16, rel_extent=64, swa_win=64,
    # n_experts=4, top_k=2, vocab_padded=256, vocab_unpadded=250
    D, I, H, KV, HD = 256, 128, 4, 2, 64
    D_REL, REL_EXT, SWA_WIN = 16, 64, 64
    N_PHYS, CORE_S, CORE_E, N_PASS, RANK = 6, 2, 4, 2, 16
    N_EXP, TOP_K = 4, 2
    V_PAD, V_UNP = 256, 250
    LOCAL_IDS = [0, 2, 3, 4]  # global layers: 1, 5
    rng = np.random.default_rng(12345)

    def make_f32(shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    s1_buf = bytearray()
    trunk_meta = []

    def add_t(name: str, arr: np.ndarray, fmt: str = "F32", target_list=None):
        lst = trunk_meta if target_list is None else target_list
        off = len(s1_buf)
        if fmt == "F32":
            raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        elif fmt == "BF16":
            u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
            bf16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
            raw = bf16.tobytes()
        elif fmt == "Q8_ROW":
            a2 = arr.reshape(arr.shape[0], -1).astype(np.float32)
            sc = np.maximum(np.max(np.abs(a2), axis=1) / 127.0, 1e-8).astype(np.float32)
            q = np.clip(np.round(a2 / sc[:, None]), -127, 127).astype(np.int8)
            raw = sc.tobytes() + q.tobytes()
        else:
            raise ValueError(fmt)
        s1_buf.extend(raw)
        lst.append({"name": name, "format": fmt, "shape": list(arr.shape), "offset": off, "bytes": len(raw)})

    add_t("model.llm.embed.weight", make_f32((V_PAD, D), 0.1), "BF16")
    add_t("model.llm.embed_norm.weight", np.ones((D,), np.float32), "F32")
    add_t("model.llm.norm.weight", np.ones((D,), np.float32), "F32")
    add_t("model.llm.unembed.weight", make_f32((V_PAD, D), 0.1), "Q8_ROW")
    add_t("model.llm.attn_res.output_query", make_f32((D,), 0.02), "F32")
    core_len = CORE_E - CORE_S
    add_t("model.llm.middle_cycle_cross_pass_queries", make_f32((2 * core_len, D), 0.02), "F32")
    add_t("model.llm.middle_cycle_cross_pass_kv_queries", make_f32((2 * core_len, D), 0.02), "F32")

    q_dim, kv_dim = H * HD, KV * HD
    for il in range(N_PHYS):
        p = f"model.llm.layers.{il}"
        is_swa = il in LOCAL_IDS
        if il > 0:
            add_t(f"model.llm.attn_res.attn_queries.{il - 1}", make_f32((D,) if is_swa else (3, D), 0.02), "F32")
        add_t(f"model.llm.attn_res.mlp_queries.{il}", make_f32((D,), 0.02), "F32")
        if is_swa:
            add_t(f"{p}.attn_norm.weight", np.ones((D,), np.float32), "F32")
            add_t(f"{p}.attn.k_sconv.weight", make_f32((kv_dim, 1, 4), 0.02), "F32")
            add_t(f"{p}.attn.v_sconv.weight", make_f32((kv_dim, 1, 4), 0.02), "F32")
        else:
            add_t(f"{p}.q_input_norm.weight", np.ones((D,), np.float32), "F32")
            add_t(f"{p}.k_input_norm.weight", np.ones((D,), np.float32), "F32")
            add_t(f"{p}.v_input_norm.weight", np.ones((D,), np.float32), "F32")
            add_t(f"{p}.attn.k_sconv.static_w", make_f32((4, kv_dim), 0.02), "F32")
            add_t(f"{p}.attn.v_sconv.static_w", make_f32((4, kv_dim), 0.02), "F32")
            add_t(f"{p}.attn.k_sconv.weight_proj", make_f32((KV * 4, D), 0.02), "BF16")
            add_t(f"{p}.attn.v_sconv.weight_proj", make_f32((KV * 4, D), 0.02), "BF16")
        add_t(f"{p}.attn.wq_du.weight", make_f32((q_dim, D), 0.04), "BF16")
        add_t(f"{p}.attn.wk_dv.weight", make_f32((kv_dim, D), 0.04), "BF16")
        add_t(f"{p}.attn.wv_dv.weight", make_f32((kv_dim, D), 0.04), "BF16")
        add_t(f"{p}.attn.wr_du.weight", make_f32((H * D_REL, D), 0.04), "BF16")
        add_t(f"{p}.attn.wo_ud.weight", make_f32((D, q_dim), 0.04), "BF16")
        add_t(f"{p}.attn.gate_attn.weight", make_f32((q_dim, D), 0.04), "BF16")
        add_t(f"{p}.attn.attn_sink_proj.weight", make_f32((H, D), 0.02), "F32")
        add_t(f"{p}.attn.rel_logits_proj.proj", make_f32((D_REL, REL_EXT), 0.02), "F32")
        add_t(f"{p}.attn.q_l2_clip.scale", np.ones((H, HD), np.float32) + make_f32((H, HD), 0.05), "F32")
        add_t(f"{p}.attn.rel_l2_clip.scale", np.ones((H, D_REL), np.float32) + make_f32((H, D_REL), 0.05), "F32")
        add_t(f"{p}.attn_sconv.weight", make_f32((D, 1, 4), 0.02), "F32")
        add_t(f"{p}.mlp_sconv.weight", make_f32((D, 1, 4), 0.02), "F32")
        add_t(f"{p}.mlp_norm.weight", np.ones((D,), np.float32), "F32")
        add_t(f"{p}.mlp.gate.weight", make_f32((N_EXP, D), 0.05), "F32")
        add_t(f"{p}.mlp.gate.bias", make_f32((N_EXP,), 0.01), "F32")
        if CORE_S <= il < CORE_E:
            for cn in ("attn", "k", "v", "mlp"):
                add_t(f"{p}.controllers.{cn}.down_weight", make_f32((RANK, 2 * D), 0.02), "F32")
                add_t(f"{p}.controllers.{cn}.gain_weight", make_f32((D, RANK), 0.02), "F32")
                add_t(f"{p}.controllers.{cn}.forget_weight", make_f32((D, RANK), 0.02), "F32")

    # Embedded DSpark drafter (1 layer tiny for fast test)
    dspark_meta = []
    D_RANK = 32
    add_t("dspark.fuse_weight", np.zeros((4, 4), np.float32), "F32", dspark_meta)
    add_t("dspark.mask_embed", make_f32((D,), 0.02), "F32", dspark_meta)
    add_t("dspark.norm.weight", np.ones((D,), np.float32), "F32", dspark_meta)
    add_t("dspark.markov_head.dflare_norm.weight", np.ones((D,), np.float32), "F32", dspark_meta)
    add_t("dspark.markov_head.markov_w1.weight", make_f32((V_PAD, D_RANK), 0.02), "F32", dspark_meta)
    add_t("dspark.markov_head.rnn_proj.weight", make_f32((3 * D_RANK, 2 * D_RANK + 2 * D), 0.02), "F32", dspark_meta)
    add_t("dspark.markov_head.rnn_proj.bias", np.zeros((3 * D_RANK,), np.float32), "F32", dspark_meta)
    add_t("dspark.markov_head.hidden_out.weight", make_f32((D, D_RANK), 0.02), "F32", dspark_meta)
    add_t("dspark.markov_head.hidden_out.bias", np.zeros((D,), np.float32), "F32", dspark_meta)
    for il in range(1):
        p = f"dspark.layers.{il}"
        add_t(f"{p}.dflare_norm.weight", np.ones((D,), np.float32), "F32", dspark_meta)
        add_t(f"{p}.input_layernorm.weight", np.ones((D,), np.float32), "F32", dspark_meta)
        add_t(f"{p}.post_attention_layernorm.weight", np.ones((D,), np.float32), "F32", dspark_meta)
        add_t(f"{p}.self_attn.q_proj.weight", make_f32((q_dim, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.self_attn.k_proj.weight", make_f32((kv_dim, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.self_attn.v_proj.weight", make_f32((kv_dim, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.self_attn.k_proj_ctx.weight", make_f32((kv_dim, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.self_attn.v_proj_ctx.weight", make_f32((kv_dim, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.self_attn.o_proj.weight", make_f32((D, q_dim), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.self_attn.q_norm.weight", np.ones((HD,), np.float32), "F32", dspark_meta)
        add_t(f"{p}.self_attn.k_norm.weight", np.ones((HD,), np.float32), "F32", dspark_meta)
        add_t(f"{p}.self_attn.k_norm_ctx.weight", np.ones((HD,), np.float32), "F32", dspark_meta)
        add_t(f"{p}.mlp.gate_proj.weight", make_f32((D * 2, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.mlp.up_proj.weight", make_f32((D * 2, D), 0.03), "BF16", dspark_meta)
        add_t(f"{p}.mlp.down_proj.weight", make_f32((D, D * 2), 0.03), "BF16", dspark_meta)

    # Section 4: SF4 (layers 0..4) + MX8 (layer 5) experts
    s4_buf = bytearray()
    dir_entries = []
    for il in range(N_PHYS):
        use_mx8 = (il == N_PHYS - 1)
        for eid in range(N_EXP):
            off = len(s4_buf)
            if not use_mx8:
                hdr = struct.pack("<4f", 0.5, 6.0, 0.5, 6.0)
                w13_p = rng.integers(0, 256, size=(2 * I, D // 2), dtype=np.uint8).tobytes()
                w13_s = np.full((2 * I, D // 16), 0x38, dtype=np.uint8).tobytes()
                dn_off = 16 + len(w13_p) + len(w13_s)
                w2_p = rng.integers(0, 256, size=(D, I // 2), dtype=np.uint8).tobytes()
                w2_s = np.full((D, I // 16), 0x38, dtype=np.uint8).tobytes()
                raw = hdr + w13_p + w13_s + w2_p + w2_s
                code_id = 15
            else:
                w13_w = rng.integers(0, 120, size=(2 * I, D), dtype=np.uint8).tobytes()
                w13_s = np.full((2 * I, D // 32), 125, dtype=np.uint8).tobytes()
                dn_off = len(w13_w) + len(w13_s)
                w2_w = rng.integers(0, 120, size=(D, I), dtype=np.uint8).tobytes()
                w2_s = np.full((D, I // 32), 125, dtype=np.uint8).tobytes()
                raw = w13_w + w13_s + w2_w + w2_s
                code_id = 16
            pad = align4k(len(raw)) - len(raw)
            body = raw + b"\x00" * pad
            s4_buf.extend(body)
            dir_entries.append(struct.pack("<HBBHHQIIII", eid, code_id, code_id, 0, 0, off, len(body), dn_off, 0, 0))

    # 3x loop-weighted vram_load_order (core layers 2..3 first!)
    order = [il * N_EXP + e for il in range(CORE_S, CORE_E) for e in range(N_EXP)] + \
            [il * N_EXP + e for il in list(range(0, CORE_S)) + list(range(CORE_E, N_PHYS)) for e in range(N_EXP)]
    s3_bytes = b"".join(dir_entries) + np.array(order, dtype=np.uint16).tobytes()
    s5_bytes = json.dumps({"version": 11, "build": "MINI-TEST"}).encode("utf-8")

    manifest = {
        "format": "strata-v1.1",
        "build": "MINI-TEST",
        "arch": "inkling2_loop",
        "config": {
            "eos_token_id": 249,
            "image_token_id": 248,
            "text_config": {
                "num_hidden_layers": N_PHYS,
                "hidden_size": D,
                "intermediate_size": I,
                "num_attention_heads": H,
                "num_key_value_heads": KV,
                "head_dim": HD,
                "d_rel": D_REL,
                "rel_extent": REL_EXT,
                "sliding_window_size": SWA_WIN,
                "sconv_kernel_size": 4,
                "n_layers_per_attn_res_block": 2,
                "n_routed_experts": N_EXP,
                "num_experts_per_tok": TOP_K,
                "vocab_size": V_PAD,
                "unpadded_vocab_size": V_UNP,
                "rms_norm_eps": 1e-6,
                "qk_l2_clip_threshold": 22.627417,
                "rel_l2_clip_threshold": 8.0,
                "log_scaling_n_floor": 8192,
                "log_scaling_alpha": 0.1,
                "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0,
                "route_scale": 8.0,
                "final_logit_softcapping": 30.0,
                "local_layer_ids": LOCAL_IDS,
                "loop_config": {"core_start": CORE_S, "core_end": CORE_E, "num_passes": N_PASS, "gate_rank": RANK},
            },
            "mtp_config": {
                "num_hidden_layers": 1,
                "hidden_size": D,
                "num_attention_heads": H,
                "num_key_value_heads": KV,
                "head_dim": HD,
                "intermediate_size": D * 2,
                "block_size": 8,
                "markov_rank": D_RANK,
                "sliding_window_size": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1e6,
                "local_layer_ids": [0],
                "target_layer_ids": [1, 3, 5, 8],
            },
        },
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "trunk_tensors": trunk_meta,
        "dspark_tensors": dspark_meta,
    }
    s0_comp = zstd.ZstdCompressor(level=3).compress(json.dumps(manifest).encode("utf-8"))

    def write_strata(path: Path, profile: int, flags: int, sec_map: dict[int, tuple[bytes, int]]):
        with open(path, "wb") as f:
            f.write(b"\x00" * PAGE)
            entries = []
            for sid in range(6):
                if sid in sec_map and len(sec_map[sid][0]) > 0:
                    data, sflags = sec_map[sid]
                    pos = f.tell()
                    if pos % PAGE:
                        f.write(b"\x00" * (PAGE - (pos % PAGE)))
                        pos = f.tell()
                    f.write(data)
                    entries.append(struct.pack("<QQ32sIIQ", pos, len(data), hashlib.sha256(data).digest(), sid, sflags, 0))
                else:
                    entries.append(struct.pack("<QQ32sIIQ", 0, 0, hashlib.sha256(b"").digest(), sid, 0, 0))
            pos = f.tell()
            if pos % PAGE:
                f.write(b"\x00" * (PAGE - (pos % PAGE)))
            frame = bytearray(PAGE)
            frame[0:8] = MAGIC
            struct.pack_into("<IHH", frame, 8, flags, profile, 6)
            for idx, e in enumerate(entries):
                frame[0x10 + idx * 64 : 0x10 + (idx + 1) * 64] = e
            f.seek(0)
            f.write(frame)

    write_strata(out_path, 0, 1, {0: (s0_comp, 0), 1: (bytes(s1_buf), 0), 3: (s3_bytes, 1), 4: (bytes(s4_buf), 0), 5: (s5_bytes, 0)})

    # Build tiny Vision shard (1 ViT layer, hidden=64, heads=4, patch=20, merge=2, grid=4, out_hidden=256)
    VD, VH, VLAY, VGRID = 64, 4, 1, 4
    vs1 = bytearray()
    vmeta = []

    def add_v(name: str, arr: np.ndarray):
        off = len(vs1)
        u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
        bf16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
        vs1.extend(bf16)
        vmeta.append({"name": name, "dtype": "bfloat16", "shape": list(arr.shape), "offset": off, "bytes": len(bf16)})

    add_v("model.visual.backbone.embeddings.class_embedding", make_f32((VD,), 0.02))
    add_v("model.visual.backbone.embeddings.cls_position_embedding", make_f32((VD,), 0.02))
    add_v("model.visual.backbone.embeddings.patch_embedding.weight", make_f32((VD, 1200), 0.02))
    add_v("model.visual.backbone.embeddings.position_embedding", make_f32((VGRID * VGRID, VD), 0.02))
    add_v("model.visual.backbone.ln_pre.weight", np.ones((VD,), np.float32))
    add_v("model.visual.backbone.ln_pre.bias", np.zeros((VD,), np.float32))
    p = "model.visual.backbone.layers.0"
    for nm, sh, val in [
        ("ln_1.weight", (VD,), 1.0), ("ln_1.bias", (VD,), 0.0),
        ("ln_2.weight", (VD,), 1.0), ("ln_2.bias", (VD,), 0.0),
        ("ls_1.gamma", (VD,), 0.1), ("ls_2.gamma", (VD,), 0.1),
    ]:
        add_v(f"{p}.{nm}", np.full(sh, val, np.float32))
    for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
        add_v(f"{p}.self_attn.{nm}.weight", make_f32((VD, VD), 0.03))
        add_v(f"{p}.self_attn.{nm}.bias", np.zeros((VD,), np.float32))
    add_v(f"{p}.mlp.c_fc.weight", make_f32((VD * 4, VD), 0.03))
    add_v(f"{p}.mlp.c_fc.bias", np.zeros((VD * 4,), np.float32))
    add_v(f"{p}.mlp.c_proj.weight", make_f32((VD, VD * 4), 0.03))
    add_v(f"{p}.mlp.c_proj.bias", np.zeros((VD,), np.float32))
    add_v("model.visual.proj_in.weight", make_f32((VD * 4, VD * 4), 0.03))
    add_v("model.visual.proj_out.weight", make_f32((D, VD * 4), 0.03))

    vman = {
        "format": "strata-v1.1-vision",
        "arch": "inkling2_vision",
        "vision_config": {
            "hidden_size": VD, "num_hidden_layers": VLAY, "num_attention_heads": VH,
            "native_patch_size": 20, "adapter_merge_size": 2, "posemb_grid_size": VGRID,
            "projection_hidden_dim": VD * 4, "text_hidden_size": D, "mlp_ratio": 4.0, "layer_norm_eps": 1e-5,
        },
        "tensors": vmeta,
    }
    vs0 = zstd.ZstdCompressor(level=3).compress(json.dumps(vman).encode("utf-8"))
    write_strata(vision_path, 3, 1, {0: (vs0, 0), 1: (bytes(vs1), 0)})


class TestInklingServing(unittest.TestCase):
    def test_shape_engine_registration(self):
        self.assertEqual(shape_engine.INKLING_SHAPE, "inkling2")
        self.assertTrue(shape_engine.exe_name("inkling2").startswith("strata-inkling"))
        self.assertEqual(shape_engine.component_id("inkling2"), "strata-inkling")

    def test_o200k_tokenizer_and_ignore_merges(self):
        self.assertEqual(strata_tokenizer.pre_tokenizer_kind("inkling2"), "o200k")
        self.assertEqual(strata_tokenizer.pre_tokenizer_kind("o200k"), "o200k")
        self.assertEqual(strata_tokenizer.pre_tokenizer_kind("gpt-4o"), "o200k")
        # Build minimal o200k tokenizer where "hello" is in vocab directly (ignore_merges!)
        b2u = strata_tokenizer.BYTE_TO_UNICODE
        base_vocab = [b2u[b] for b in range(256)] + ["hello", "<|content_thinking|>", "<|content_model_end_sampling|>"]
        types = [1] * 257 + [3, 3]
        tok = strata_tokenizer.Tokenizer(base_vocab, [], types, pre="inkling2")
        self.assertTrue(tok.ignore_merges)
        self.assertEqual(tok.encode("hello"), [256])
        self.assertEqual(tok.encode("<|content_thinking|>hello<|content_model_end_sampling|>", parse_special=True), [257, 256, 258])

    def test_inkling_dialect_streaming_and_tool_recovery(self):
        # Streaming chunk-by-chunk across split special tokens!
        sample = (
            "<|message_model|><|content_thinking|>Let me check Tokyo weather.<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>"
            "{\"name\":\"get_weather\",\"args\":{\"city\":\"Tokyo\",\"units\":\"C\"}}<|end_message|>"
            "<|message_model|><|content_text|>Checking now.<|end_message|><|content_model_end_sampling|>"
        )
        p = frontend.InklingOutputParser(
            thinking=True,
            tools=[{"name": "get_weather", "parameters": {"type": "object"}}],
        )
        events = []
        for i in range(0, len(sample), 5):
            events.extend(p.feed(sample[i : i + 5]))
        events.extend(p.finish())
        reason = "".join(e.text for e in events if e.kind == "reasoning")
        content = "".join(e.text for e in events if e.kind == "content")
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(reason, "")
        self.assertIn("Checking now.", content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, {"city": "Tokyo", "units": "C"})

        # Malformed XML tool call recovery with special chars < > &
        xml_sample = (
            "<|message_model|><|content_xml|>"
            "<tool_call name=\"run_py\"><param name=\"code\">if a < b && c > 0: print(1)</param></tool_call>"
            "<|end_message|>"
        )
        p2 = frontend.InklingOutputParser(
            thinking=False,
            tools=[{"name": "run_py", "parameters": {"type": "object"}}],
        )
        ev2 = p2.feed(xml_sample) + p2.finish()
        calls2 = [e.call for e in ev2 if e.kind == "tool_call"]
        self.assertEqual(len(calls2), 1)
        self.assertEqual(calls2[0].name, "run_py")
        self.assertIn("a < b && c > 0", calls2[0].arguments["code"])

    def test_reasoning_effort_and_parallel_64_args(self):
        # Inkling-2-Mini has no thinking mode: default is off ("none") and every input normalizes to "none".
        for k in ("none", "off", "minimal", "low", "medium", "high", "max"):
            self.assertEqual(inkling_dialect.normalize_effort(k), "none")
        cfg = {"arch": "inkling2_loop", "exe": "component:strata-inkling", "args": ["--native", "m.strata"], "parallel": 64}
        self.assertEqual(srv_engines.parallel_args(cfg, list(cfg["args"])), ["--parallel", "64"])
        self.assertTrue(inkling_kind.is_inkling(cfg))
        self.assertIn("--parallel", inkling_kind.engine_args(cfg, announce=False))
        self.assertIn("64", inkling_kind.engine_args(cfg, announce=False))

    def test_strata_inkling_binary_c1_4_8_32_64_and_vision(self):
        exe = _find_inkling_exe()
        if exe is None:
            self.skipTest("strata-inkling binary not built yet")

        # 1. --version & standalone --selfcheck
        r_ver = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True)
        self.assertTrue(r_ver.stdout.strip().startswith("strata-inkling "))
        r_sc = subprocess.run([str(exe), "--selfcheck", "--cpu"], capture_output=True, text=True, check=True)
        self.assertIn("PASS", r_sc.stdout)

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            m_path = td_p / "mini.strata"
            v_path = td_p / "mini-vision.strata"
            _build_mini_inkling_strata(m_path, v_path)

            # 2. --info & --selfcheck --verify-hash
            r_info = subprocess.run([str(exe), "--info", str(m_path), "--vision", str(v_path)], capture_output=True, text=True, check=True)
            info = inkling_kind.parse_info(r_info.stdout)
            self.assertEqual(info["arch"], "inkling2_loop")
            self.assertEqual(info["drafter"], "dspark")
            self.assertEqual(info["images"], "1")
            self.assertEqual(info["parallel"], "64")
            subprocess.run([str(exe), "--selfcheck", "--verify-hash", "--native", str(m_path), "--cpu"], check=True, capture_output=True)

            # 3. Verify c = 1 solo vs c = 4, 8, 32, 64 concurrent slots via `--serve` (FP16 & FP8!)
            def run_serve_cmds(cmds: list[str], parallel: int, kv: str = "fp16", vision: Path | None = None) -> list[str]:
                args = [str(exe), "--serve", "--native", str(m_path), "--parallel", str(parallel),
                        "--kv-dtype", kv, "--max-context", "256", "--draft", "off", "--cpu"]
                if vision is not None:
                    args += ["--vision", str(vision)]
                proc = subprocess.run(args, input="\n".join(cmds + ["QUIT"]) + "\n",
                                      capture_output=True, text=True, timeout=120, check=True)
                return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

            # Solo reference for 2 distinct prompts
            solo_a = run_serve_cmds(["GEN 3 temperature=0 10,20,30,40"], parallel=1, kv="fp16")
            toks_a = [int(ln.split()[1]) for ln in solo_a if ln.startswith("T ")]
            solo_b = run_serve_cmds(["GEN 3 temperature=0 50,60,70,80"], parallel=1, kv="fp16")
            toks_b = [int(ln.split()[1]) for ln in solo_b if ln.startswith("T ")]
            self.assertEqual(len(toks_a), 3)
            self.assertEqual(len(toks_b), 3)

            # Test c = 4, 8, 32, 64 concurrent slots simultaneously!
            for par, kv in [(4, "fp16"), (8, "fp16"), (32, "fp8"), (64, "fp8")]:
                cmds = ["HELLO", "STATS"]
                for s in range(par):
                    prompt = "10,20,30,40" if (s % 2 == 0) else "50,60,70,80"
                    cmds.append(f"GENERATE req{s} 3 temperature=0 {prompt}")
                lines = run_serve_cmds(cmds, parallel=par, kv=kv)
                self.assertTrue(any(ln.startswith("HELLO ok ") for ln in lines))
                self.assertTrue(any(ln.startswith("STATS ") for ln in lines))
                got: dict[str, list[int]] = {}
                dones: set[str] = set()
                for ln in lines:
                    parts = ln.split()
                    if parts[0] == "T" and len(parts) == 3:
                        got.setdefault(parts[1], []).append(int(parts[2]))
                    elif parts[0] == "DONE" and len(parts) >= 3 and parts[1].startswith("req"):
                        dones.add(parts[1])
                self.assertEqual(len(dones), par, f"parallel={par} missing DONEs")
                for s in range(par):
                    exp = toks_a if (s % 2 == 0) else toks_b
                    if kv == "fp16":
                        self.assertEqual(got[f"req{s}"], exp, f"parallel={par} FP16 slot {s} mismatch")
                    else:
                        self.assertEqual(len(got[f"req{s}"]), 3, f"parallel={par} FP8 slot {s} short")

            # 4. Test native BGEN / BT / BDONE protocol at c = 4 + DSpark=ON vs DSpark=OFF bit-exact parity!
            bgen_cmds = [f"BGEN {s} 3 temperature=0 10,20,30,40" for s in range(4)]
            blines = run_serve_cmds(bgen_cmds, parallel=4, kv="fp16")
            self.assertEqual(sum(1 for ln in blines if ln.startswith("BADM ")), 4)
            self.assertEqual(sum(1 for ln in blines if ln.startswith("BDONE ")), 4)

            # DSpark=ON vs DSpark=OFF bit-exact greedy parity (exercises ShortConv partial-accept rollback!)
            proc_ds = subprocess.run(
                [str(exe), "--serve", "--native", str(m_path), "--parallel", "1",
                 "--kv-dtype", "fp16", "--max-context", "256", "--draft", "dspark", "--cpu"],
                input="GEN 6 temperature=0 10,20,30,40\nQUIT\n",
                capture_output=True, text=True, timeout=60, check=True,
            )
            proc_off = subprocess.run(
                [str(exe), "--serve", "--native", str(m_path), "--parallel", "1",
                 "--kv-dtype", "fp16", "--max-context", "256", "--draft", "off", "--cpu"],
                input="GEN 6 temperature=0 10,20,30,40\nQUIT\n",
                capture_output=True, text=True, timeout=60, check=True,
            )
            ds_toks = [int(ln.split()[1]) for ln in proc_ds.stdout.splitlines() if ln.startswith("T ")]
            off_toks = [int(ln.split()[1]) for ln in proc_off.stdout.splitlines() if ln.startswith("T ")]
            self.assertEqual(len(off_toks), 6)
            self.assertEqual(ds_toks, off_toks, f"DSpark=ON {ds_toks} != DSpark=OFF {off_toks}")

            # 5. Vision end-to-end with a generated 45x35 RGBA PNG image!
            from PIL import Image
            img_p = td_p / "test.png"
            Image.new("RGBA", (45, 35), (120, 200, 80, 128)).save(img_p)
            vlines = run_serve_cmds([f"GEN 2 temperature=0 image={inkling_kind.encode_path(img_p)} 10,248,20"],
                                    parallel=1, kv="fp16", vision=v_path)
            self.assertTrue(any(ln.startswith("IMG 0 1 2 2") for ln in vlines), f"Missing IMG line in {vlines}")
            self.assertEqual(sum(1 for ln in vlines if ln.startswith("T ")), 2)

            # 6. Full server.StrataEngine zero-HoL concurrent multiplexing at c = 8 (FP16), c = 32 (FP8), c = 64 (FP8)!
            import concurrent.futures
            import threading
            import serve.server as srv

            for par, kv in [(8, "fp16"), (32, "fp8"), (64, "fp8")]:
                eng = srv.StrataEngine(
                    str(exe),
                    ["--serve", "--native", str(m_path), "--parallel", str(par),
                     "--kv-dtype", kv, "--max-context", "256", "--draft", "off", "--cpu"],
                )
                try:
                    self.assertTrue(eng.inkling_engine)
                    self.assertEqual(eng.batch, par)
                    self.assertTrue(eng.slot_read)

                    def run_one(idx: int) -> list[int]:
                        prompt = [10, 20, 30, 40] if (idx % 2 == 0) else [50, 60, 70, 80]
                        cancel = threading.Event()
                        toks = []
                        for tok in eng.generate(prompt, 3, {"temperature": 0.0}, cancel):
                            if tok is not None:
                                toks.append(int(tok))
                        return toks

                    with concurrent.futures.ThreadPoolExecutor(max_workers=par) as pool:
                        futs = [pool.submit(run_one, i) for i in range(par)]
                        results = [f.result(timeout=60) for f in futs]

                    for idx, got_toks in enumerate(results):
                        if kv == "fp16":
                            exp = toks_a if (idx % 2 == 0) else toks_b
                            self.assertEqual(got_toks, exp, f"StrataEngine parallel={par} req {idx} mismatch")
                        else:
                            self.assertEqual(len(got_toks), 3, f"StrataEngine parallel={par} FP8 req {idx} short")
                finally:
                    eng.close()


if __name__ == "__main__":
    unittest.main()
