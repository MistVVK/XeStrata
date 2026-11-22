# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: the comfyui_controlnet_aux contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
"""serve/image_controls.py - the composition control (the chat's `control`, the Images API's `control` field and
POST /v1/images/controls): a control image made from a picture by the engine's CONTROL line (xestrata-image's
cnet_prep: canny, depth, pose and the others), then given to the image model.

    job["control"] = parse_control(req["control"], cfg, picture)   # the request's, checked
    prepare(svc, gen, job, cancel)                                   # before the drawing: the map, then the model's way

How the map reaches the model depends on what it has (capability):
  controlnet  Qwen-Image-2.1: alibaba-pai's Fun ControlNet Union beside its DiT (the IMG line's CNET:, CTRL:, CTRLS:),
              its strength and the share of the schedule it is on (start and end, as sigmas by percent_sigma).
  moss        MOSS-Image's color and depth (darask0's adapter, image_aux's "moss-control": its LoRA at the control's
              strength and its widened input projection, the IMG line's MCTL:, MCOLOR:, MDEPTH:, MCTLENDS:): a color
              map (SAM 3's instances of COCO's things and stuff, each a flat color of its own: COLORMAP) and a depth
              map (Depth Anything V2 Small's), either or both (a list of the two), each on for its share of the steps
              (end); a start is not trained, and is refused.
  reference   Mage-Flow's edits, HunyuanImage 3.0 and MOSS-Image with an edit adapter (its other kinds): no ControlNet
              is published for them, so the map goes in as one more reference image and the prompt asks for its pose,
              outlines or depth (an edit's instruction, as with Qwen-Image's edits without ControlNet); strength, start
              and end do not apply and are refused.
Models without edits (MageTrail, Anima) take no control (MOSS-Image without its edit adapter: color and depth alone).

Each kind's network weights (Depth Anything V2 Small, HED, line art, M-LSD, YOLOX-L with DWPose) and the Fun ControlNet
Union are image_aux's: fetched the first time a request needs them (AuxPending until then).  The map is made in a GPU
turn of its own (the engine's AUXNEED control, then CONTROL), before the drawing's.
"""
from __future__ import annotations

import base64
import bisect
import contextlib
import io
import re
import struct
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from serve import image_aux
from serve import images as img
from serve import loras as lora_api

KINDS = ("canny", "depth", "pose", "lineart", "lineart_coarse", "hed", "scribble_hed", "scribble", "scribble_xdog",
         "mlsd", "gray", "normal", "segmentation")
MOSS_KINDS = ("color", "depth")   # MOSS-Image's own control (its adapter)
MOSS_CONTROL = "moss-control"     # image_aux's name of darask0's adapter
# the color map's concepts, as the adapter's training maps had them (COCO panoptic's things and its main stuff): SAM 3
# finds each, its instances a color each
COLOR_CONCEPTS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "sky", "wall", "floor", "ceiling", "road", "pavement", "grass", "tree", "building", "house", "mountain", "rock",
    "sea", "river", "water", "sand", "snow", "dirt", "fence", "window", "door", "curtain", "rug", "shelf", "cabinet",
    "bridge", "light", "mirror", "pillow", "blanket", "table", "counter", "stairs", "flower", "food", "paper")
ALIASES = {"openpose": "pose", "dwpose": "pose", "softedge": "hed", "grayscale": "gray", "line_art": "lineart",
           "depth_anything": "depth", "edges": "canny", "lines": "mlsd", "map": "none"}
UNION = "cnet-union"          # image_aux's name of the Fun ControlNet Union
RESOLUTION = 512              # the shorter side a picture is processed at (the preprocessors' own default)
MAX_RESOLUTION = 2048
# what the prompt asks of the map, for a model that takes it as a reference image ({i}: its place among them)
FOLLOW = {"pose": "the pose of the person in image {i}", "depth": "the composition and depth of image {i}",
          "canny": "the outlines of image {i}", "lineart": "the line drawing of image {i}",
          "lineart_coarse": "the line drawing of image {i}", "hed": "the outlines of image {i}",
          "scribble_hed": "the sketch in image {i}", "scribble": "the sketch in image {i}",
          "scribble_xdog": "the sketch in image {i}", "mlsd": "the straight lines and layout of image {i}",
          "gray": "the shading and composition of image {i}", "normal": "the surface orientation of image {i}",
          "segmentation": "the objects and their layout in image {i}", "none": "the structure of image {i}"}


def mode_of(cfg: dict, kind: str) -> str | None:
    """How the model of the config section `cfg` takes a control of `kind`: "controlnet", "moss", "reference" or
    None."""
    caps = img._qimage().family_capabilities(cfg.get("family"))
    if caps.get("control") == "controlnet":
        return "controlnet" if kind not in image_aux.CONTROLNET_REFUSED_KINDS else None
    if caps.get("control") == "moss" and kind in MOSS_KINDS:
        return "moss"
    if kind in caps.get("control_loras", {}) and img.can_edit(cfg):
        return "lora"
    if caps.get("control_reference") and img.can_edit(cfg) and kind != "color":
        return "reference"
    return None


def capability(cfg: dict) -> str | None:
    """How the model takes its control images, as /health says it: "controlnet", "moss" (its color and depth adapter,
    the other kinds as references where it edits), "reference" or None."""
    if img._qimage().family_capabilities(cfg.get("family")).get("control") == "moss":
        return "moss"
    return mode_of(cfg, "canny")


def kinds_for(cfg: dict) -> list[str]:
    return [k for k in KINDS + ("color",) if mode_of(cfg, k)]


def weight_names(kind: str, mode: str | None) -> list[str]:
    """image_aux's weights a control of `kind` reaches the model of `mode` through: the kind's networks (W:, W2:),
    then the ControlNet."""
    q = image_aux._qimage()
    extra = ([UNION] if mode == "controlnet" else
             [f"moss-control-{q.family_capabilities('moss-image')['control_loras'][kind]}"] if mode == "lora" else [])
    return list(q.CNET_KINDS.get(kind, ())) + extra


def _num(v: dict, key: str, lo: float, hi: float, default: float) -> float:
    return img.parse_number(v.get(key), f"control.{key}", lo, hi, default)


def _flag(v: dict, key: str, default: bool) -> bool:
    return img.parse_flag(v.get(key), f"control.{key}", default)


def _as_map(v: dict, default: bool) -> bool:
    """preprocess false means the picture is already the map; a form's "false" says the same as JSON's False."""
    flag = img.parse_flag(v.get("preprocess"), "control.preprocess", None)
    return default if flag is None else not flag


def decode_image(v) -> bytes:
    """A control image given inline: base64, or a data: URL of it."""
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if not isinstance(v, str) or not v:
        raise img.ImageRequestError("control.image: a picture (base64, or a data: URL)")
    try:
        return img.decode_inline_image(v, strict=True)
    except ValueError:
        raise img.ImageRequestError("control.image: not base64") from None


def _picture(v: dict, picture: bytes | None) -> bytes:
    if picture is None and v.get("image") is not None:
        picture = decode_image(v["image"])
        # the same pass a multipart file, an http(s) URL or a path gets from the server: WebP to PNG and EXIF's
        # orientation.  A data: URL or bare base64 must not reach the engine as unnormalised bytes
        from serve.server import Vision
        picture = Vision.normalize(picture)
    if not picture:
        raise img.ImageRequestError("control.image: the picture to take the composition from")
    try:
        w, h = img.image_dims(picture)
    except Exception:   # noqa: BLE001 - any picture the dims reader does not take
        raise img.ImageRequestError("control.image: not a picture (PNG, JPEG, WebP, GIF or BMP)") from None
    if w < 16 or h < 16:
        raise img.ImageRequestError("control.image: at least 16 x 16 pixels")
    return picture


def _kind(v: dict) -> str:
    kind = str(v.get("type") or v.get("kind") or "").lower()
    return ALIASES.get(kind, kind)


def _params(v: dict, kind: str) -> dict:
    """A kind's settings from the request; Canny's and M-LSD's defaults and limits as comfyui_controlnet_aux's
    (Apache-2.0)."""
    p: dict = {}
    if kind == "canny":
        p["low"] = _num(v, "low_threshold", 0, 255, 100)
        p["high"] = _num(v, "high_threshold", 0, 255, 200)
    elif kind == "scribble_xdog":
        p["threshold"] = _num(v, "threshold", 1, 255, 12)
    elif kind in ("hed", "scribble_hed"):
        p["safe"] = _flag(v, "safe", True)
    elif kind == "mlsd":
        p["score"] = _num(v, "score_threshold", 0.01, 2.0, 0.1)
        p["distance"] = _num(v, "distance_threshold", 0.01, 20.0, 0.1)
    elif kind == "pose":
        for part in ("body", "hands", "face"):
            p[part] = _flag(v, f"detect_{part}", True)
    return p


def _regions(value) -> list[dict]:
    """The colors of a map and their descriptions; the colors name pixels, not words for the model."""
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise img.ImageRequestError("control.regions: a list of 1 to 8 {color, prompt} objects")
    out: list[dict] = []
    for r in value:
        if not isinstance(r, dict) or set(r) != {"color", "prompt"}:
            raise img.ImageRequestError("control.regions: each region needs color and prompt")
        color, prompt = r["color"], r["prompt"]
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise img.ImageRequestError("control.regions.color: the map's exact #rrggbb color")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000 or "\0" in prompt:
            raise img.ImageRequestError("control.regions.prompt: a non-empty description of at most 2000 characters")
        if any(x["color"] == color.lower() for x in out):
            raise img.ImageRequestError("control.regions.color: each color only once")
        out.append({"color": color.lower(), "prompt": prompt.strip()})
    return out


def region_prompt(png: bytes, regions: list[dict]) -> str:
    """Locate each exact color by its centroid. Give the model positions, never the map's color names.

    Use the axis with the greatest minimum separation, so vertically stacked regions use top-to-bottom even where
    their horizontal coordinates coincide. Refuse coincident centroids rather than silently assign the wrong region.
    """
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    h, w = a.shape[:2]
    points = []
    for region in regions:
        color = tuple(bytes.fromhex(region["color"][1:]))
        y, x = np.nonzero(np.all(a == color, axis=2))
        if not len(x):
            raise img.ImageRequestError(f"control.regions: color {region['color']} is not in the color map")
        points.append((float(x.mean()) / w, float(y.mean()) / h))
    if len(points) == 1:
        return f"The subject is {regions[0]['prompt']}."
    gaps = []
    for axis in (0, 1):
        values = sorted(p[axis] for p in points)
        gaps.append(min(b - a for a, b in zip(values, values[1:])))
    axis = 0 if gaps[0] >= gaps[1] else 1
    if gaps[axis] < 0.03:
        raise img.ImageRequestError("control.regions: their centers are too close to assign distinct positions")
    order = sorted(range(len(points)), key=lambda i: points[i][axis])
    positions: tuple[str, ...] = (("left", "right") if axis == 0 else ("top", "bottom"))
    if len(order) == 3:
        positions = (("left", "center", "right") if axis == 0 else ("top", "middle", "bottom"))
    elif len(order) > 3:
        edge = "left" if axis == 0 else "top"
        positions = tuple(f"position {i + 1} from the {edge}" for i in range(len(order)))
    return " ".join(f"The subject at the {position} of the image: {regions[i]['prompt']}."
                    for position, i in zip(positions, order))


def _picture_samples(data: bytes):
    """Bounded, unfiltered samples: interpolation would invent colors along a map's boundaries."""
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((256, 256), Image.Resampling.NEAREST)
        return np.asarray(im).reshape(-1, 3)


def _drawing(data: bytes) -> bool:
    """A mostly white, neutral image with a small amount of dark ink; uncertain pictures stay photos."""
    import numpy as np
    a = _picture_samples(data).astype(np.int16)
    lo, hi = a.min(axis=1), a.max(axis=1)
    return bool((lo >= 240).mean() >= 0.75 and 0.005 <= (hi < 160).mean() <= 0.25
                and (hi - lo <= 12).mean() >= 0.98)


def _color_map(data: bytes) -> bool:
    """A few exact flat colors cover almost all pixels; photographs go through the preprocessor."""
    import numpy as np
    _, counts = np.unique(_picture_samples(data), axis=0, return_counts=True)
    return bool(2 <= len(counts) and np.sort(counts)[-8:].sum() >= counts.sum() * 0.995)


def _moss(items: list, cfg: dict, picture: bytes | None) -> dict:
    """MOSS-Image's color and depth: one control or a list of the two (each its own picture; the multipart
    control_image the first's), the adapter's strength (the first's, or the only one given) -> {"type": "color",
    "depth" or "color+depth", "mode": "moss", "maps": [{"type", "image", "end", "resolution", "as_map"}],
    "strength"}."""
    maps: list[dict[str, Any]] = []
    strength: float | None = None
    regions = None
    for i, v in enumerate(items):
        if not isinstance(v, dict):
            raise img.ImageRequestError("control: an object {type, image, strength, end, ...}, or a list of them")
        kind = _kind(v)
        if v.get("backend") not in (None, "moss"):
            raise img.ImageRequestError("control.backend: color and depth together need the moss backend")
        if v.get("regions") is not None:
            if kind != "color":
                raise img.ImageRequestError("control.regions: give the regions on the color map")
            regions = _regions(v["regions"])
        if kind not in MOSS_KINDS:
            raise img.ImageRequestError(f"control.type: MOSS-Image's control takes color and depth together (a list); "
                                        f"{kind!r} goes alone (a reference image)")
        if any(m["type"] == kind for m in maps):
            raise img.ImageRequestError(f"control: {kind} twice")
        start = img.parse_number(v.get("start"), "control.start", 0.0, 1.0, 0.0)
        if start != 0:
            raise img.ImageRequestError("control.start: MOSS-Image's control adapter is trained from the first step "
                                        "only (an end is taken)")
        if v.get("strength") is not None:
            s = _num(v, "strength", 0.0, 2.0, 1.0)
            if strength is not None and s != strength:
                raise img.ImageRequestError("control.strength: one for the adapter (color and depth share it)")
            strength = s
        image = _picture(v, picture if i == 0 else None)
        as_map = _as_map(v, kind == "color" and _color_map(image))
        maps.append({"type": kind, "image": image,
                     "end": _num(v, "end", 0.0, 1.0, 1.0), "as_map": as_map,
                     "resolution": int(_num(v, "resolution", 64, MAX_RESOLUTION, RESOLUTION))})
    return {"type": "+".join(m["type"] for m in maps), "mode": "moss", "maps": maps,
            "strength": 1.0 if strength is None else strength, "image": maps[0]["image"], "regions": regions}


def parse_control(v: Any, cfg: dict, picture: bytes | None = None) -> dict:
    """A request's `control` (an object, or a kind's name; MOSS-Image's color and depth: a list of them too) with its
    picture (control.image, or the multipart control_image given as `picture`) -> {"type", "image", "mode",
    "strength", "start", "end", "resolution", "params"} (MOSS-Image's: _moss's).  type "none", or preprocess false:
    the picture is already the control map.  ImageRequestError for a model without control, a kind it does not take, a
    field out of range, or (for a model that takes the map as a reference) strength, start or end."""
    if isinstance(v, str):
        v = {"type": v}
    if isinstance(v, list):
        if not v or len(v) > 2:
            raise img.ImageRequestError("control: an object, or a list of MOSS-Image's color and depth")
        if mode_of(cfg, _kind(v[0]) if isinstance(v[0], dict) else "") != "moss":
            raise img.ImageRequestError("control: a list is MOSS-Image's color and depth; this model takes one")
        return _moss(v, cfg, picture)
    if not isinstance(v, dict):
        raise img.ImageRequestError("control: an object {type, image, strength, start, end, ...}")
    if capability(cfg) is None:
        raise img.ImageRequestError("control: this image model takes no control image (Qwen-Image-2.1 has ControlNet; "
                                    "MOSS-Image has color and depth; Mage-Flow's edits, HunyuanImage 3.0 and "
                                    "MOSS-Image with edits take the other kinds as a reference image)")
    kind = _kind(v)
    if kind != "none" and kind not in KINDS + MOSS_KINDS:
        raise img.ImageRequestError(f"control.type: one of {', '.join(kinds_for(cfg))} or none (the image a map "
                                    "already)")
    mode = mode_of(cfg, kind)
    backend = v.get("backend")
    if backend is not None:
        if kind != "depth" or mode != "moss" or backend not in ("moss", "lora"):
            raise img.ImageRequestError("control.backend: moss or lora, for MOSS-Image depth only")
        if backend == "lora":
            if not img.can_edit(cfg):
                raise img.ImageRequestError("control.backend lora: this MOSS-Image model has no edit adapter")
            mode = "lora"
    if mode is None:
        raise img.ImageRequestError(f"control.type {kind}: not this image model's (it takes "
                                    f"{', '.join(kinds_for(cfg)) or 'none'})")
    if mode == "moss":
        return _moss([v], cfg, picture)
    if v.get("regions") is not None:
        raise img.ImageRequestError("control.regions: MOSS-Image's color map only")
    as_map = _as_map(v, False)
    if as_map and mode != "lora":
        kind = "none"
    out: dict[str, Any] = {"type": kind, "image": _picture(v, picture), "mode": mode, "as_map": as_map,
                           "resolution": int(_num(v, "resolution", 64, MAX_RESOLUTION, RESOLUTION)),
                           "params": _params(v, kind)}
    lineart_input = v.get("lineart_input")
    if lineart_input is not None and (mode != "controlnet" or kind not in ("lineart", "lineart_coarse") or as_map):
        raise img.ImageRequestError("control.lineart_input: photo or drawing for a preprocessed ControlNet lineart")
    if mode == "controlnet" and kind in ("lineart", "lineart_coarse"):
        if lineart_input not in (None, "photo", "drawing"):
            raise img.ImageRequestError("control.lineart_input: photo or drawing")
        out["lineart_input"] = lineart_input or ("drawing" if _drawing(out["image"]) else "photo")
    if mode == "lora":
        out["strength"] = _num(v, "strength", 0.0, 2.0, 1.0)
        for key, default in (("start", 0.0), ("end", 1.0)):
            if v.get(key) is not None and _num(v, key, 0.0, 1.0, default) != default:
                raise img.ImageRequestError(f"control.{key}: scheduled control LoRA is not available")
    elif mode == "controlnet":
        out["strength"] = _num(v, "strength", 0.0, 2.0, 1.0)
        out["start"] = _num(v, "start", 0.0, 1.0, 0.0)
        out["end"] = _num(v, "end", 0.0, 1.0, 1.0)
        if out["start"] > out["end"]:
            raise img.ImageRequestError("control.start: at most control.end")
        out["inpaint"] = _flag(v, "inpaint", True)
    else:
        for key in ("strength", "start", "end"):
            if v.get(key) is not None:
                raise img.ImageRequestError(f"control.{key}: this image model takes the control image as a reference "
                                            "image (no ControlNet): its strength and schedule do not apply")
    return out


def gather(req: dict) -> dict:
    """A request's control from its parts: `control` (an object, a kind's name, or an object as JSON text: a
    multipart form's) and a form's control_<field> fields (control_type, control_strength...), merged into one
    object; the request as it is without any."""
    import json
    c = req.get("control")
    if isinstance(c, str) and c.strip().startswith("{"):
        try:
            c = json.loads(c)
        except ValueError:
            raise img.ImageRequestError("control: an object, or its JSON") from None
    flat = {k[len("control_"):]: v for k, v in req.items() if k.startswith("control_") and k != "control_image"}
    if c is None and not flat:
        return req
    if isinstance(c, list):   # MOSS-Image's color and depth: the list as it is
        return {**{k: v for k, v in req.items() if not (k.startswith("control_") and k != "control_image")},
                "control": c}
    if isinstance(c, str):
        c = {"type": c}
    c = {**(c if isinstance(c, dict) else {}), **flat}
    return {**{k: v for k, v in req.items() if not (k.startswith("control_") and k != "control_image")}, "control": c}


def parse_job(req: dict, cfg: dict, files: list[bytes] | None = None, mask: bytes | None = None,
              picture: bytes | None = None, edit: bool = False) -> dict:
    """A generations or edits request with `control` -> its job (images.parse_request's or parse_edit_request's) with
    job["control"] (parse_control's; `picture` the multipart control_image).  A model that takes the map as a reference
    image draws an edit: the request's images, then the control picture in the map's place (prepare puts the map
    there); its size by default the first image's shape.  ControlNet's generation without a size takes the control
    picture's shape at about 1024 x 1024."""
    value = req.get("control")
    if ((edit or files) and isinstance(value, dict) and _kind(value) == "pose"
            and mode_of(cfg, "pose") == "controlnet"):
        value = dict(value)
        if value.get("strength") is None:
            value["strength"] = 0.7
        if value.get("end") is None:
            value["end"] = 0.8
    c = parse_control(value, cfg, picture)
    r = {k: v for k, v in req.items() if k not in ("control", "control_image")}
    files = list(files or [])
    if not files and r.get("size") in (None, "", "auto"):
        r["size"] = "{}x{}".format(*img.ref_size(*img.image_dims(c["image"])))
    if c["mode"] == "moss" and r.get("strength") is not None and not files:
        raise img.ImageRequestError("strength: an edit's or img2img's; the control's strength is control.strength")
    if c["mode"] == "lora":
        if files or mask is not None or r.get("edit_adapter") not in (None, ""):
            raise img.ImageRequestError("control: a control LoRA with another edit adapter or mask is not available; "
                                        "call control with the picture to make a new controlled image, or remove "
                                        "control to edit the original")
        if str(cfg.get("model_name") or "").startswith("moss-image-base"):
            raise img.ImageRequestError("control: the control LoRA has not been checked with MOSS-Image Base")
        if not cfg.get("vision"):
            raise img.ImageRequestError("control: the MOSS-Image vision encoder is required (run setup again)")
        if r.get("steps") is None and r.get("num_inference_steps") is None:
            # 4-step's turbo LoRA takes 4..8 steps, not the usual 30: parse_request would refuse the control job
            r["steps"] = min(30, img.model_defaults(cfg)["steps"]) if cfg.get("turbo_lora") else 30
        if r.get("guidance_scale") is None and r.get("true_cfg_scale") is None:
            # the 4-step LoRA allows no negative prompt and a scale of 1, its config's value
            r["guidance_scale"] = img.model_defaults(cfg)["guidance"] if cfg.get("turbo_lora") else 4.0
        r.setdefault("uncond_skip_layers", [])
        job = img.parse_edit_request(r, [c["image"]], {**cfg, "edit_format": "ai-toolkit"})
    elif c["mode"] == "reference":
        job = img.parse_edit_request(r, files + [c["image"]], cfg, mask)
    elif edit or files:
        job = img.parse_edit_request(r, files, cfg, mask)
    else:
        job = img.parse_request(r, cfg)
    if c.get("regions") and (job.get("layout") or job.get("enhance")):
        raise img.ImageRequestError("control.regions: not with layout or prompt enhancement")
    job["control"] = c
    return job


def control_words(c: dict) -> str:
    """CONTROL's words of a control's parameters (xestrata-image's)."""
    p = c.get("params") or {}
    words = f" RES:{int(c.get('resolution') or RESOLUTION)}"
    if "low" in p:
        words += f" LOW:{p['low']!r} HIGH:{p['high']!r}"
    if "threshold" in p:
        words += f" THR:{p['threshold']!r}"
    if "safe" in p:
        words += f" SAFE:{int(p['safe'])}"
    if "score" in p:
        words += f" SCORE:{p['score']!r} DIST:{p['distance']!r}"
    for part in ("body", "hands", "face"):
        if part in p:
            words += f" {part.upper()}:{int(p[part])}"
    return words


def weights(gen, names: list[str]) -> list[str]:
    """The paths of image_aux's `names`, or AuxPending while any is downloading (its download started)."""
    root = image_aux.aux_dir(gen.cfg)
    got = [image_aux.ensure(n, root, gen._say) for n in names]
    pending = [g for g in got if g["status"] != "ready"]
    if pending:
        raise img.AuxPending(pending)
    return [g["path"] for g in got]


def _png(rgba: bytes, w: int, h: int) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.frombytes("RGBA", (w, h), rgba).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _lineart_map(png: bytes, drawing: bool) -> bytes:
    """A smooth white-on-black outline: raw HED for a photo, or the drawing itself."""
    from PIL import Image, ImageFilter, ImageOps
    import statistics

    with Image.open(io.BytesIO(png)) as pic:
        rgba = pic.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        gray = ImageOps.grayscale(Image.alpha_composite(background, rgba).convert("RGB"))
    if drawing:
        w, h = gray.size
        border = [gray.getpixel((x, y)) for x in range(w) for y in (0, h - 1)]
        border += [gray.getpixel((x, y)) for y in range(h) for x in (0, w - 1)]
        if statistics.median(border) >= 128:
            gray = ImageOps.invert(gray)
    gray = gray.filter(ImageFilter.MaxFilter(3))
    if drawing:
        gray = gray.filter(ImageFilter.GaussianBlur(0.5))
    out = io.BytesIO()
    gray.convert("RGB").save(out, "PNG")
    return out.getvalue()


# RGB histograms of the leftmost control in thedeoxen's normal card images/figure.jpg, revision
# fd4eb2c19e31a606d97bab5b9e59cad8426dcfd8 (350 x 512 crop).  Matching and smoothing kept both a full-body
# composition and a close-up in RTX 4070 checks; BAE alone and matching without smoothing did not.
_NORMAL_TARGET = (
    'c-rmOcWh2k7{~E*->RybEv-^iC_$*I5ix=q31ZLI$g7Aw6I5G!HKbIHNbHCh2{nRHn<84HViS}`qAg;Di19snf8ve5{7c;AljrVx'
    '?itTH#{Ajlz@=DBe7s^wlt38-AyBcrVikm-6lx+EHMCy`%}^f=(E#<(86D6QT@ZtKbV6V0gWhP3e(0g&Ug(DL7=|S6564{0z-Wxb'
    'EKI@##Y8L>yFVW(NWf$yW1h~B!V0X?H&*MJ1euLAq+^L}!Y*XtD7NB0G8J#)i>)8WQ{kTo;lwx$Qk;bp48$B9!x|bW$k&+1#raQv'
    'W4_WT2WWs?Vjc~Xjf3Ub4>e{>gfY*fj9J*inAcQ(SK_)x##F6sOyw|RmRB&QJ<qlZFs7X!b<TT>Xy_`4<2xbvz&H0|3RYklrmH;t'
    'F<93IV2HR;SJ&)u6q+IgF^E8{<N4nP2-Q9IdMOl87w_;E5Ah9mk%RL%h*Q{!1K5MjIF5~K$89>lUgvC^Ey7~8rERkXY9o7nt<Eh|'
    '+uF9Df)VOd3F>nL)gMP{e}Jyr{uZa>SjFz@zizZal={EjZ>sSS4w}kDNT|k8pvIYF{+`X{71gbcpYn&VatB@0RM4^AryHBH$}Rq0'
    '$7NIprFAX9>wddmQ7Y-aFx6Xa)p1+pmi}s|aT<41Dbz~5!$t}kMPHAnfX#WaKLy)QVwxC9&Z3dhNI01^+D8&wJ|DkHVtc{I-4w`8'
    'EZ~RI`N18;44yB<9UMgjKiH7>p5^^P#6)~UKIiAb!6f8yt}!tf)7cM4UF7iGY>dMgM4}^VArNmV`+e-j5+q>+ViAod=!ANB!@v3w'
    'cM=a{17<3}BqJU@(M{vfj=}HLRV`u|eC_k_!Do%jLVQ3zZr~~|YD{J+zh!EC+A(|@=dc}nb>AgqXk6R-FX%nzHNJP_h-Aw#+{P^w'
    ';Rzn;`KS8UHC#uY?tP9Ec&vAx(Kk-&xvPpcf4syMeal|Itg_^YEz1MlZ?ErDe$P^wuB(1*xe8R4uf~(#ZBBD}<w4sQdtm|+u?M%2'
    'j|`-%t<oG%4t#|Th}Jmp`EBn0zy06-h0XC=i~d}Dy0l)jIliUViO03(%G5latF>3O^4SfIRh!G6+4ZH?Rl$xYClqR2Hqu&VI7(_g'
    '5`<<L>UiEa7JW4qt5KL~#8l<a)i{AyHb=UCd8E7>E|EyaA>2a?eylnc;tVds%@037D)Ye>26+;5OEweZd*+@1ets}(xFROA4lGU<'
    'x92-dc$cbC?#ir7m=FtC&$VN&>c_-+ig_xR$?*quCHz1'
)


def _normal_map(data: bytes) -> bytes:
    """Match the control LoRA's example's RGB distributions, then soften details at the map's own scale."""
    from PIL import Image, ImageFilter

    im = Image.open(io.BytesIO(data)).convert("RGB")
    source = im.histogram()
    target = struct.unpack("<768I", zlib.decompress(base64.b85decode(_NORMAL_TARGET)))
    pixels = im.width * im.height
    lut = []
    for channel in range(3):
        src = source[channel * 256:(channel + 1) * 256]
        dst = target[channel * 256:(channel + 1) * 256]
        total = sum(dst)
        cumulative = []
        n = 0
        for count in dst:
            n += count
            cumulative.append(n * pixels)
        n = 0
        for count in src:
            n += count
            lut.append(min(255, bisect.bisect_left(cumulative, n * total)))
    im = im.point(lut).filter(ImageFilter.GaussianBlur(2 * min(im.size) / 512))
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def preprocess(svc, gen, c: dict, cancel=None) -> dict:
    """The control image of the control `c` (parse_control's): the engine's CONTROL in a GPU turn of its own (its
    networks' weights image_aux's: AuxPending while they download) -> {"image": PNG bytes, "width", "height",
    "seconds", "engine_s"}; type "none": the picture as it is (a PNG)."""
    t0 = time.monotonic()
    if c.get("lineart_input"):
        if c["lineart_input"] == "drawing":
            png = _lineart_map(img.as_png(c["image"]), True)
            w, h = img.image_dims(png)
            return {"image": png, "width": w, "height": h, "seconds": round(time.monotonic() - t0, 3),
                    "engine_s": 0.0}
        made = preprocess(svc, gen, {**c, "type": "hed", "params": {"safe": False}, "lineart_input": None}, cancel)
        made["image"] = _lineart_map(made["image"], False)
        made["seconds"] = round(time.monotonic() - t0, 3)
        return made
    if c["type"] == "none":
        png = img.as_png(c["image"])
        w, h = img.image_dims(png)
        return {"image": png, "width": w, "height": h, "seconds": 0.0, "engine_s": 0.0}
    paths = weights(gen, list(image_aux._qimage().CNET_KINDS[c["type"]]))
    enc = [lora_api.encode_path(p) for p in paths]
    W, H = img.image_dims(c["image"])
    job = {"aux": {"kind": "control", "path": enc[0] if enc else "-",
                   "need_args": [c["type"], int(c.get("resolution") or RESOLUTION)] + enc[1:]},
           "n": 1, "steps": 1, "width": W, "height": H, "seed": 0}
    with img.gpu_turn(svc, gen, job, cancel):
        gen._live(stage="control")
        work = Path(tempfile.mkdtemp(prefix="control-", dir=gen.dir))
        try:
            pic, out = work / "picture", work / "map"
            pic.write_bytes(c["image"])
            line = f"CONTROL {c['type']} {pic} {out}{control_words(c)}"
            if enc:
                line += f" W:{enc[0]}" + (f" W2:{enc[1]}" if len(enc) > 1 else "")
            got = gen.aux_ask(line, "CONTROL")
            w, h = int(got["width"]), int(got["height"])
            png = _png(out.read_bytes(), w, h)
        finally:
            for f in work.iterdir():
                with contextlib.suppress(OSError):
                    f.unlink()
            with contextlib.suppress(OSError):
                work.rmdir()
    gen.end(None)
    if c["type"] == "normal" and c.get("mode") == "lora":
        png = _normal_map(png)
    return {"image": png, "width": w, "height": h, "seconds": round(time.monotonic() - t0, 3),
            "engine_s": float(got.get("control_s") or 0.0)}


def prepare(svc, gen, job: dict, cancel=None) -> None:
    """Before `job`'s drawing: its control map made (preprocess), then given to the model its way (capability) -
    ControlNet's branch and map (job["control"]["net"], ["map"]), or one more reference image and the prompt's ask of
    it.  The figures go to job["control"]["figures"]."""
    c = job["control"]
    if c.get("ready"):
        return
    mode = c["mode"]
    if mode == "moss":
        _prepare_moss(svc, gen, job, cancel)
        return
    # every weight's download started at once (the branch's 7 GB beside the network's), then the map
    prep_kind = "hed" if c.get("lineart_input") == "photo" else c["type"]
    if c.get("lineart_input") == "drawing":
        prep_kind = "canny"   # no preprocessor network; the ControlNet is still needed
    names = weight_names(prep_kind, mode) if prep_kind != "none" else weight_names("canny", mode)
    if c.get("as_map") and mode == "lora":
        names = names[-1:]
    paths = weights(gen, names)
    made = preprocess(svc, gen, {**c, "type": "none"} if c.get("as_map") else c, cancel)
    c["figures"] = {"type": c["type"], "size": f"{made['width']}x{made['height']}", "seconds": made["seconds"],
                    "engine_s": made["engine_s"], "mode": mode}
    c["map_png"] = made["image"]   # the chat's control links it, so another call takes it as it is (type none)
    if mode == "controlnet":
        c["net"] = weights(gen, [UNION])[0]
        c["map"] = made["image"]
    elif mode == "lora":
        job["refs"] = [img.moss_ref(made["image"], img.MOSS_AITK_REF_PIXELS, grow=False)]
        job["edit_lora"] = None
        job["loras"] = list(job.get("loras") or []) + [{"name": names[-1], "path": paths[-1], "scale": c["strength"]}]
    else:   # the edit's last reference (parse_job's place for it) becomes the map
        refs = list(job.get("refs") or [])
        if not refs:
            raise img.ImageRequestError("control: this image model takes the control image as a reference image")
        ref = made["image"]
        if img._qimage().family_capabilities(gen.cfg.get("family")).get("control_ref_resize"):
            ref = img.moss_ref(ref)
        refs[-1] = ref
        job["refs"] = refs
        follow = FOLLOW[c["type"]].format(i=len(refs))
        job["prompt"] = f"{job['prompt'].rstrip()} Follow {follow}."
    c["ready"] = True


def _prepare_moss(svc, gen, job: dict, cancel=None) -> None:
    """MOSS-Image's color and depth: the adapter (image_aux's MOSS_CONTROL) and each map's weights (Depth Anything V2
    Small's; SAM 3's and CLIP's tokenizer for the color, SAM 3 only once its license is agreed to), each map made (a
    picture sent as a map as it is), then job["mctl"] (the adapter and the maps, img_words') and the adapter's LoRA at
    the control's strength among the request's adapters."""
    c = job["control"]
    root = image_aux.aux_dir(gen.cfg)
    names = [MOSS_CONTROL]
    for m in c["maps"]:
        if m["as_map"]:
            continue
        if m["type"] == "depth":
            names.append("cnet-depth")
        else:
            if image_aux.needs_consent("sam3", root):
                from serve import image_regions
                raise img.ImageRequestError("control.type color: its map is SAM 3's, and " +
                                            image_regions.sam3_consent_words() + " (or send the color map itself, "
                                            "preprocess false)")
            names += ["sam3", "clip-tokenizer"]
    weights(gen, list(dict.fromkeys(names)))
    made: dict[str, bytes] = {}
    figures = []
    for m in c["maps"]:
        if m["as_map"]:
            made[m["type"]] = img.as_png(m["image"])
            continue
        got = (preprocess(svc, gen, {"type": "depth", "image": m["image"], "resolution": m["resolution"],
                                     "params": {}}, cancel)
               if m["type"] == "depth" else color_map(svc, gen, m["image"], cancel))
        made[m["type"]] = got["image"]
        figures.append({"type": m["type"], "size": f"{got['width']}x{got['height']}", "seconds": got["seconds"],
                        **({"objects": got["objects"]} if "objects" in got else {})})
    if c.get("regions"):
        addition = region_prompt(made["color"], c["regions"])
        if len(job["prompt"]) + len(addition) + 1 > 16000:
            raise img.ImageRequestError("control.regions: the combined prompt is longer than 16000 characters")
        job["prompt"] = job["prompt"].rstrip() + " " + addition
    path = weights(gen, [MOSS_CONTROL])[0]
    ends = {m["type"]: m["end"] for m in c["maps"]}
    job["mctl"] = {"path": path, "color": made.get("color"), "depth": made.get("depth"),
                   "color_end": ends.get("color", 1.0), "depth_end": ends.get("depth", 1.0)}
    job["loras"] = list(job.get("loras") or []) + [{"name": MOSS_CONTROL, "path": path, "scale": c["strength"]}]
    c["figures"] = {"type": c["type"], "mode": "moss", "maps": figures,
                    "size": figures[0]["size"] if figures else "as given"}
    c["map_png"] = made.get("color") or made.get("depth")
    c["ready"] = True


def color_map(svc, gen, picture: bytes, cancel=None) -> dict:
    """MOSS-Image's color map of `picture`: SAM 3's instances of COLOR_CONCEPTS, each a flat color of its own, black
    elsewhere (the engine's COLORMAP, in a GPU turn of its own) -> {"image": PNG bytes, "width", "height", "objects",
    "seconds", "engine_s"}."""
    from serve import image_regions
    t0 = time.monotonic()
    sam, tok = weights(gen, ["sam3", "clip-tokenizer"])
    clip = image_regions._clip(tok)
    words = ";".join(",".join(map(str, clip.encode(w, image_regions.SAM3_MAX_IDS))) for w in COLOR_CONCEPTS)
    W, H = img.image_dims(picture)
    job = image_regions.aux_job(gen, "sam3", lora_api.encode_path(sam), picture)
    with img.gpu_turn(svc, gen, job, cancel):
        gen._live(stage="control")
        work = Path(tempfile.mkdtemp(prefix="control-", dir=gen.dir))
        try:
            pic, out = work / "picture", work / "map"
            pic.write_bytes(picture)
            got = gen.aux_ask(f"COLORMAP {lora_api.encode_path(sam)} {pic} {out} {words}", "COLORMAP")
            png = _png(out.read_bytes(), int(got["width"]), int(got["height"]))
        finally:
            for f in work.iterdir():
                with contextlib.suppress(OSError):
                    f.unlink()
            with contextlib.suppress(OSError):
                work.rmdir()
    gen.end(None)
    return {"image": png, "width": W, "height": H, "objects": int(got.get("objects") or 0),
            "seconds": round(time.monotonic() - t0, 3), "engine_s": float(got.get("colormap_s") or 0.0)}


def img_words(job: dict, cfg: dict, width: int, height: int) -> str:
    """IMG's ControlNet words (xestrata-image's CNET:, CTRL:, CTRLS:, CTRLNOINPAINT) of a prepared job; "" without one.
    The start and end shares of the schedule as the model's sigmas (images.percent_sigma).  MOSS-Image's control:
    MCTL:, MCOLOR:, MDEPTH: and MCTLENDS: (its LoRA is among the request's LORA: words)."""
    m = job.get("mctl")
    if m:
        words = f" MCTL:{lora_api.encode_path(m['path'])}"
        if job.get("mctl_color_path"):
            words += f" MCOLOR:{lora_api.encode_path(job['mctl_color_path'])}"
        if job.get("mctl_depth_path"):
            words += f" MDEPTH:{lora_api.encode_path(job['mctl_depth_path'])}"
        return words + f" MCTLENDS:{m['color_end']!r},{m['depth_end']!r}"
    c = job.get("control") or {}
    if not c.get("net") or not job.get("control_path"):
        return ""
    hi = img.percent_sigma(cfg, width, height, c.get("start", 0.0))
    lo = img.percent_sigma(cfg, width, height, c.get("end", 1.0))
    words = (f" CNET:{lora_api.encode_path(c['net'])} CTRL:{lora_api.encode_path(job['control_path'])}"
             f" CTRLS:{c.get('strength', 1.0)!r},{hi!r},{lo!r}")
    if c.get("inpaint") is False:
        words += " CTRLNOINPAINT"
    return words


def need_words(job: dict) -> str:
    """NEED's ControlNet branch or MOSS-Image's control adapter and map count."""
    c = job.get("control") or {}
    m = job.get("mctl") or {}
    if m:
        return (f" MCTL:{lora_api.encode_path(m['path'])}" + (" MCOLOR:1" if m.get("color") else "")
                + (" MDEPTH:1" if m.get("depth") else ""))
    return f" CNET:{lora_api.encode_path(c['net'])}" if c.get("net") else ""


# ------------------------------------------------------------------------------------------- POST /v1/images/controls
def preview_request(fields: dict, picture: bytes | None) -> dict:
    """POST /v1/images/controls' fields (multipart: image, type and the kind's parameters; or JSON with image as
    base64) -> the control's fields (parse_control's, without the model's) and response_format (b64_json, url)."""
    rf = str(fields.get("response_format") or "b64_json")
    if rf not in ("b64_json", "url"):
        raise img.ImageRequestError(f"response_format: b64_json or url, not {rf!r}")
    return {"fields": {k: v for k, v in fields.items() if k not in ("response_format", "model", "image")},
            "picture": picture if picture else decode_image(fields.get("image")), "response_format": rf}


def preview_response(svc, gen, req: dict, base_url: str, cancel=None) -> dict:
    """The control image a request would give the model (the web page's look before sending) as OpenAI's images
    answer: created, data [{b64_json or url}], the type and size, and "xestrata" figures."""
    fields = dict(req["fields"])
    kind = str(fields.get("type") or fields.get("kind") or "").lower()
    kind = ALIASES.get(kind, kind)
    if kind in ("", "none"):
        raise img.ImageRequestError("type: one of " + ", ".join(KINDS + ("color",)))
    c: dict[str, Any] = {"type": kind, "image": _picture({}, req["picture"])}
    made: dict[str, Any]
    if kind == "color" and fields.get("preprocess") is False:   # a ready map: show it, do not paint it again
        w, h = img.image_dims(c["image"])
        made = {"image": img.as_png(c["image"]), "width": w, "height": h, "seconds": 0.0, "engine_s": 0.0}
    elif kind == "color":   # MOSS-Image's color map (SAM 3's)
        if image_aux.needs_consent("sam3", image_aux.aux_dir(gen.cfg)):
            from serve import image_regions
            raise img.ImageRequestError("type color: its map is SAM 3's, and " + image_regions.sam3_consent_words())
        made = color_map(svc, gen, c["image"], cancel)
    else:
        # the map alone: checked as a ControlNet's fields whatever the model (no strength applies here)
        c = parse_control({**{k: v for k, v in fields.items() if k not in ("strength", "start", "end")}},
                          gen.cfg, req["picture"])
        made = preprocess(svc, gen, {**c, "type": "none"} if c.get("as_map") else c, cancel)
    entry = ({"url": f"{base_url}/v1/images/files/{gen.store(made['image'], 'png')}"}
             if req["response_format"] == "url" else {"b64_json": base64.b64encode(made["image"]).decode()})
    return {"created": int(time.time()), "type": c["type"], "width": made["width"], "height": made["height"],
            "data": [entry], "xestrata": {"seconds": made["seconds"], "engine_s": made["engine_s"]}}
