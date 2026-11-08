# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
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
import contextlib
import io
import tempfile
import time
from pathlib import Path
from typing import Any

from serve import image_aux
from serve import images as img
from serve import loras as lora_api

KINDS = ("canny", "depth", "pose", "lineart", "lineart_coarse", "hed", "scribble_hed", "scribble", "scribble_xdog",
         "mlsd", "gray")
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
          "gray": "the shading and composition of image {i}", "none": "the structure of image {i}"}


def mode_of(cfg: dict, kind: str) -> str | None:
    """How the model of the config section `cfg` takes a control of `kind`: "controlnet", "moss", "reference" or
    None."""
    family = cfg.get("family") or "qwen-image"
    if family == "qwen-image":
        return "controlnet" if kind != "color" else None
    if family == "moss-image" and kind in MOSS_KINDS:
        return "moss"
    if family in ("mage-flow", "hunyuan-image-3", "moss-image") and img.can_edit(cfg) and kind != "color":
        return "reference"
    return None


def capability(cfg: dict) -> str | None:
    """How the model takes its control images, as /health says it: "controlnet", "moss" (its color and depth adapter,
    the other kinds as references where it edits), "reference" or None."""
    family = cfg.get("family") or "qwen-image"
    if family == "moss-image":
        return "moss"
    return mode_of(cfg, "canny")


def kinds_for(cfg: dict) -> list[str]:
    return [k for k in KINDS + ("color",) if mode_of(cfg, k)]


def weight_names(kind: str, mode: str | None) -> list[str]:
    """image_aux's weights a control of `kind` reaches the model of `mode` through: the kind's networks (W:, W2:),
    then the ControlNet."""
    q = image_aux._qimage()
    return list(q.CNET_KINDS.get(kind, ())) + ([UNION] if mode == "controlnet" else [])


def _num(v: dict, key: str, lo: float, hi: float, default: float) -> float:
    x = v.get(key, default)
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise img.ImageRequestError(f"control.{key}: a number from {lo:g} to {hi:g}")
    try:
        f = float(x)
    except ValueError:
        raise img.ImageRequestError(f"control.{key}: a number from {lo:g} to {hi:g}") from None
    if not lo <= f <= hi:
        raise img.ImageRequestError(f"control.{key}: a number from {lo:g} to {hi:g}")
    return f


def _flag(v: dict, key: str, default: bool) -> bool:
    x = v.get(key, default)
    if isinstance(x, bool):
        return x
    if str(x).lower() in ("1", "true", "on", "enable", "yes"):
        return True
    if str(x).lower() in ("0", "false", "off", "disable", "no"):
        return False
    raise img.ImageRequestError(f"control.{key}: true or false")


def decode_image(v) -> bytes:
    """A control image given inline: base64, or a data: URL of it."""
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if not isinstance(v, str) or not v:
        raise img.ImageRequestError("control.image: a picture (base64, or a data: URL)")
    if v.startswith("data:"):
        v = v.split(",", 1)[-1]
    try:
        return base64.b64decode(v, validate=True)
    except ValueError:
        raise img.ImageRequestError("control.image: not base64") from None


def _picture(v: dict, picture: bytes | None) -> bytes:
    if picture is None and v.get("image") is not None:
        picture = decode_image(v["image"])
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


def _moss(items: list, cfg: dict, picture: bytes | None) -> dict:
    """MOSS-Image's color and depth: one control or a list of the two (each its own picture; the multipart
    control_image the first's), the adapter's strength (the first's, or the only one given) -> {"type": "color",
    "depth" or "color+depth", "mode": "moss", "maps": [{"type", "image", "end", "resolution", "as_map"}],
    "strength"}."""
    maps: list[dict[str, Any]] = []
    strength: float | None = None
    for i, v in enumerate(items):
        if not isinstance(v, dict):
            raise img.ImageRequestError("control: an object {type, image, strength, end, ...}, or a list of them")
        kind = _kind(v)
        if kind not in MOSS_KINDS:
            raise img.ImageRequestError(f"control.type: MOSS-Image's control takes color and depth together (a list); "
                                        f"{kind!r} goes alone (a reference image)")
        if any(m["type"] == kind for m in maps):
            raise img.ImageRequestError(f"control: {kind} twice")
        if v.get("start") not in (None, 0, 0.0):
            raise img.ImageRequestError("control.start: MOSS-Image's control adapter is trained from the first step "
                                        "only (an end is taken)")
        if v.get("strength") is not None:
            s = _num(v, "strength", 0.0, 2.0, 1.0)
            if strength is not None and s != strength:
                raise img.ImageRequestError("control.strength: one for the adapter (color and depth share it)")
            strength = s
        maps.append({"type": kind, "image": _picture(v, picture if i == 0 else None),
                     "end": _num(v, "end", 0.0, 1.0, 1.0), "as_map": v.get("preprocess") is False,
                     "resolution": int(_num(v, "resolution", 64, MAX_RESOLUTION, RESOLUTION))})
    return {"type": "+".join(m["type"] for m in maps), "mode": "moss", "maps": maps,
            "strength": 1.0 if strength is None else strength, "image": maps[0]["image"]}


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
    mode = mode_of(cfg, kind if kind != "none" else "canny")
    if mode is None:
        raise img.ImageRequestError(f"control.type {kind}: not this image model's (it takes "
                                    f"{', '.join(kinds_for(cfg)) or 'none'})")
    if mode == "moss":
        return _moss([v], cfg, picture)
    if v.get("preprocess") is False:
        kind = "none"
    out: dict[str, Any] = {"type": kind, "image": _picture(v, picture), "mode": mode,
                           "resolution": int(_num(v, "resolution", 64, MAX_RESOLUTION, RESOLUTION)),
                           "params": _params(v, kind)}
    if mode == "controlnet":
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
    c = parse_control(req.get("control"), cfg, picture)
    r = {k: v for k, v in req.items() if k not in ("control", "control_image")}
    files = list(files or [])
    if c["mode"] == "moss" and r.get("strength") is not None and not files:
        raise img.ImageRequestError("strength: an edit's or img2img's; the control's strength is control.strength")
    if c["mode"] == "reference":
        job = img.parse_edit_request(r, files + [c["image"]], cfg, mask)
    elif edit or files:
        job = img.parse_edit_request(r, files, cfg, mask)
    else:
        if r.get("size") in (None, "", "auto"):
            r["size"] = "{}x{}".format(*img.ref_size(*img.image_dims(c["image"])))
        job = img.parse_request(r, cfg)
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


def preprocess(svc, gen, c: dict, cancel=None) -> dict:
    """The control image of the control `c` (parse_control's): the engine's CONTROL in a GPU turn of its own (its
    networks' weights image_aux's: AuxPending while they download) -> {"image": PNG bytes, "width", "height",
    "seconds", "engine_s"}; type "none": the picture as it is (a PNG)."""
    t0 = time.monotonic()
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
    names = weight_names(c["type"], mode) if c["type"] != "none" else weight_names("canny", mode)
    weights(gen, names)
    made = preprocess(svc, gen, c, cancel)
    c["figures"] = {"type": c["type"], "size": f"{made['width']}x{made['height']}", "seconds": made["seconds"],
                    "engine_s": made["engine_s"], "mode": mode}
    if mode == "controlnet":
        c["net"] = weights(gen, [UNION])[0]
        c["map"] = made["image"]
    else:   # the edit's last reference (parse_job's place for it) becomes the map
        refs = list(job.get("refs") or [])
        if not refs:
            raise img.ImageRequestError("control: this image model takes the control image as a reference image")
        ref = made["image"]
        if gen.cfg.get("family") == "moss-image":
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
    path = weights(gen, [MOSS_CONTROL])[0]
    ends = {m["type"]: m["end"] for m in c["maps"]}
    job["mctl"] = {"path": path, "color": made.get("color"), "depth": made.get("depth"),
                   "color_end": ends.get("color", 1.0), "depth_end": ends.get("depth", 1.0)}
    job["loras"] = list(job.get("loras") or []) + [{"name": MOSS_CONTROL, "path": path, "scale": c["strength"]}]
    c["figures"] = {"type": c["type"], "mode": "moss", "maps": figures,
                    "size": figures[0]["size"] if figures else "as given"}
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
    """NEED's ControlNet word: the branch's file (its blocks counted)."""
    c = job.get("control") or {}
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
    if kind == "color":   # MOSS-Image's color map (SAM 3's)
        if image_aux.needs_consent("sam3", image_aux.aux_dir(gen.cfg)):
            from serve import image_regions
            raise img.ImageRequestError("type color: its map is SAM 3's, and " + image_regions.sam3_consent_words())
        c: dict[str, Any] = {"type": "color", "image": _picture({}, req["picture"])}
        made = color_map(svc, gen, c["image"], cancel)
    else:
        # the map alone: checked as a ControlNet's fields whatever the model (no strength applies here)
        c = parse_control({**{k: v for k, v in fields.items() if k not in ("strength", "start", "end")}},
                          {"family": "qwen-image"}, req["picture"])
        made = preprocess(svc, gen, c, cancel)
    entry = ({"url": f"{base_url}/v1/images/files/{gen.store(made['image'], 'png')}"}
             if req["response_format"] == "url" else {"b64_json": base64.b64encode(made["image"]).decode()})
    return {"created": int(time.time()), "type": c["type"], "width": made["width"], "height": made["height"],
            "data": [entry], "xestrata": {"seconds": made["seconds"], "engine_s": made["engine_s"]}}
