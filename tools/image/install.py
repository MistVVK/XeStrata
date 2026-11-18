# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/image/install.py - image generation (Qwen-Image-2.1-Turbo, Qwen-Image-2.1, Mage-Flow, HunyuanImage 3.0 or
MOSS-Image) for an existing XeStrata install: downloads and checks the model files for the image engine setup compiled
with the text engine (engine/xestrata-image), and writes the "images" section into a run config (or prints it), as
setup.py --images does inside the normal setup (setup.py's setup_images).

    python tools/image/install.py --config ~/.config/xestrata/xestrata-iq3_xxs.json
    python tools/image/install.py --quant Q6_K --print
    python tools/image/install.py --config ~/.config/xestrata/xestrata-iq3_xxs.json --model mage-flow-turbo

--gpu-policy auto|keep|shrink|unload: what the text model does during an image job (serve/images.py).
--quant: Qwen-Image-2.1's image model's size, lowered (after its text encoder's) when the files would not fit the RAM
the config's text model leaves, unless --ssd-offload on; Mage-Flow's comes in one size (bf16); HunyuanImage 3.0's as
w4a8 (its default) or int8, which the RAM lowers to w4a8 (it has no text encoder, so only its own size is lowered).
--model qwen-image-turbo|qwen-image-base|mage-flow-turbo|mage-flow|mage-flow-base|magetrail[-uncensored]|
hunyuan-image-3-distil|moss-image-sft|moss-image-base[-uncensored]: the image model (setup.py --images), or several,
comma-separated (the first the default, the others named by a request's "model"), or no (image generation taken out of
--config); -uncensored: its text encoder with the refusal direction removed (HunyuanImage 3.0 has none: it reads its own
prompt).  The text encoder is Q8_0, or lowered first; MOSS-Image's Qwen3.5-9B is the chat model's own file when the
config's model is Qwen3.5-9B of the same maker (shared, its prompts encoded by the chat engine).

The weights are under their own licenses, which setup prints when it installs them (tools/image/qimage.py's
license_note): Qwen-Image-2.1's under the Qwen Research License, Mage-Flow's and MageTrail's under MIT, and
HunyuanImage 3.0's under the Tencent Hunyuan Community License Agreement, which does not apply in the European Union,
the United Kingdom or South Korea: https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE; MOSS-Image's
under Apache 2.0.  Their full texts: the web app's About > Model terms.
--dit-q8, --dit-keep: Mage-Flow's or MOSS-Image's image model in 8 or 16 bits, and Mage-Flow's not in use kept in RAM
(as setup's --image-dit-q8 and --image-dit-keep).
--gpu auto|same|PCI: the GPU of the image jobs (setup's --image-gpu); --sam3-license agree|no: Meta's SAM License for
SAM 3.1, which finds a part of a picture by words (setup's --image-sam3-license; asked when not given).
--tools yes|all|no: the editing tools' weights downloaded now (all: the versions a request chooses as well), or each
on its first use (setup's --image-tools; asked when not given).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import qimage as QI  # noqa: E402  (beside this file)
from serve import option_values  # noqa: E402
import setup as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quant", choices=["Q4_K_M", "Q6_K", "Q8_0", *QI.HUNYUANIMAGE3_QUANTS])
    ap.add_argument("--auto-largest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--models-dir", help="where the model files go (default: the data folder's models)")
    ap.add_argument("--config", help="a run config (xestrata-<model>.json) to write the \"images\" section into")
    ap.add_argument("--gpu-policy", choices=option_values.GPU_POLICIES, default=S.OPTION["image-gpu-policy"]["default"])
    ap.add_argument("--ssd-offload", choices=["on", "off"],
                    help="on: keep the size even when the RAM is short (the rest read from the SSD)")
    ap.add_argument("--model", default=QI.DEFAULT_MODEL,
                    help="qwen-image-turbo (default: Qwen-Image-2.1-Turbo, 8 steps), qwen-image-base "
                         "(Qwen-Image-2.1, 40 steps), mage-flow-turbo (Mage-Flow-Turbo, 4 steps), mage-flow (its RL "
                         "model, 20 steps), mage-flow-base (its base model, 30 steps), magetrail (30 steps) or "
                         "hunyuan-image-3-distil (HunyuanImage 3.0 Instruct-Distil, 8 steps, guidance 2.5, its own "
                         "tokenizer and system prompt: no text encoder), moss-image-sft (MOSS-Image-SFT, 50 steps, "
                         "guidance 3) or moss-image-base (its base model), or several, comma-separated (the first the "
                         "default); -uncensored: the text encoder (Qwen3-VL-8B, Mage-Flow's 4B, MOSS-Image's "
                         "Qwen3.5-9B) with the refusal direction removed")
    ap.add_argument("--lora-training", action="store_true", default=None,
                    help="prepare the full-step bases for LoRA training (setup's --image-lora-training)")
    ap.add_argument("--dit-q8", choices=["auto", "on", "off"],
                    help="Mage-Flow, MOSS-Image: its DiT 16-bit in VRAM when it fits there with the text encoder, "
                         "else 8 bits (auto, the default); on always 8 bits, off always 16-bit")
    ap.add_argument("--gpu", metavar="auto|same|PCI", help="the GPU of the image jobs, as setup's --image-gpu")
    ap.add_argument("--sam3-license", choices=["agree", "no"], help="as setup's --image-sam3-license")
    ap.add_argument("--tools", choices=["yes", "all", "no"], help="as setup's --image-tools")
    ap.add_argument("--dit-keep", choices=["on", "off"],
                    help="Mage-Flow: keep the DiT not in use (the generation's or the edit's) in RAM (on, the default)")
    ap.add_argument("--print", action="store_true", help="print the images section instead of writing it")
    ap.add_argument("--json", action="store_true",
                    help="setup.py's --json output (one JSON object a line, for the Windows launcher)")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    if a.model == "no":   # image generation taken out of the config
        if not a.config:
            ap.error("--model no needs --config")
        off = Path(a.config)
        cfg = json.loads(off.read_text(encoding="utf-8-sig"))
        cfg["images"] = cfg["image_models"] = None
        S.write_config(off, cfg)
        S.ok(f"{off.name}: image generation taken out (once the server starts again)")
        return 0
    try:
        QI.parse_choices(a.model)
    except ValueError as e:
        ap.error(f"--model {a.model}: {e}")
    gpu = S.gpu_info() or {}
    if a.models_dir is None:
        data, _ = S.data_folder(None)
        a.models_dir = str(data / "models")
    ns = argparse.Namespace(images=a.model, image_quant=a.quant, image_gpu_policy=a.gpu_policy, yes=a.yes,
                            image_ssd_offload=a.ssd_offload, image_dit_q8=a.dit_q8, image_auto_largest=a.auto_largest,
                            image_dit_keep=a.dit_keep, image_gpu=a.gpu, image_sam3_license=a.sam3_license,
                            image_lora_training=a.lora_training, image_tools=a.tools)
    path = Path(a.config) if a.config else None
    cfg = json.loads(path.read_text(encoding="utf-8-sig")) if path is not None and path.is_file() else {}
    # the RAM the model was set up to leave free (setup's --ram-reserve-gib, its engine's --ram-reserve-mib)
    rr = S.choices_from_config(path).get("ram_reserve_gib") if cfg and path is not None else None
    ns.ram_reserve_gib = rr
    eng = Path(cfg["exe"]).parent if cfg.get("exe") else ROOT / "engine"     # the text engine's folder
    # the RAM the text model of that config keeps (its size by the name setup gave it), none without a config; a
    # model of PLAIN_FAMILIES (`xestrata-engine --native`) as setup_plain estimates it, its GGUF the file a text encoder
    # can share (MOSS-Image's Qwen3.5-9B)
    args = cfg.get("args") or []
    native = Path(args[args.index("--native") + 1]) if "--native" in args[:-1] else None
    name = str(cfg.get("model_name", "")).lower()
    text_gb: float | None
    # setup's family where the config names it; a config written before 2026-11-09 by its model name
    fam = S.PLAIN_FAMILIES.get(str(cfg.get("family"))) or \
        next((f for f in sorted(S.PLAIN_FAMILIES.values(), key=lambda f: len(f["name"]), reverse=True)
              if name.startswith(f["name"] + "-")), None)
    if fam is not None and native is not None:
        text_gb = S.DENSE_RAM_GB if fam.get("dense") else \
            (native.stat().st_size / 1e9 if native.exists() else 0.0) + 8
    else:
        model = next((m for m in sorted(S.MODELS, key=len, reverse=True) if m.lower() in name), None)
        text_gb = S.text_ram_gb(args, S.ram_gb(), model) if cfg else None
    got = S.setup_images(ns, gpu, Path(a.models_dir), eng, text_gb, native)
    if got is None:
        return 1
    section, more = got
    if a.print or not a.config:
        print(json.dumps({"images": section, **({"image_models": more} if more else {})}, indent=1))
        return 0
    assert path is not None
    cfg["images"] = section
    cfg["image_models"] = more or None
    S.place_images(ns, cfg)            # the card of the image jobs (--gpu), as setup places it
    S.write_config(path, cfg)
    S.ok(f"{path.name}: image generation added (POST /v1/images/generations once the server starts again)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
