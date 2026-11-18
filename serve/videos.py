# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/videos.py - the video tools' requests and jobs: a video upscaled by the image engine's upscalers (SeedVR2 in
time, RealESRGAN and Lanczos frame by frame: src/upscale's run_video, the engine's UPSV), and its objects found and
followed by SAM 3.1 (src/image/sam3_video.cpp, the engine's SEGV), read and written by ffmpeg.

The entry points are vLLM-Omni's: POST /v1/videos (a job, OpenAI's Videos API's shape: queued, in_progress, completed
or failed, its progress, GET /v1/videos/{id}, its file at /v1/videos/{id}/content, GET /v1/videos, DELETE) and POST
/v1/videos/sync (the file in the answer).  The source video comes as a multipart form's file (vLLM-Omni's
input_references, or video), written to a file as it arrives, never held in memory whole.  size is the output's
WIDTHxHEIGHT; without it, scale (2 by default); extra_params: vLLM-Omni's color_correction_method, and XeStrata's
chunk_frames and temporal_overlap (SeedVR2's chunks) and video_codec (h264, vp9,
ffv1).

The segmentation's entry point is vLLM-Omni's POST /v1/videos/segmentations: its form (video, prompt, prompts as a
JSON list, mask_video, model) and its answer (width, height, num_frames, fps, objects [{id, prompt, mask_video: a base64
mask video}], frames [{frame_index, instances [{id, score, box [x0, y0, x1, y1], mask: COCO's RLE}]}]).  XeStrata adds:
a prompts entry may be {"prompt", "frame"} (words looked for from that frame on); clicks, a JSON list of {"frame" (or
"time", seconds), "object" (an id to redo, or none: a new one), "points" [[x, y, 1 in or 0 out]...], "box" [x0, y0, x1,
y1]} on the clip's pixels (an object's prompt is then "" and its "click" the entry's index); outputs, a JSON list of
"mask" (one mask video of every object) and "overlay" (the clip with the objects over it, its sound kept), answered as
mask_union_video and overlay_video; and job true, which makes it a job (its record answered at once, the answer at
/v1/videos/{id}/content, the videos at its ?variant=overlay, mask or object_<id>).

Limits, in two tiers: the synchronous answer takes what vLLM-Omni's does (2048 frames, an upload of 512 MiB; past them
400); a job writes its result to the disk, so its frames are not counted against anything and its upload's limit is
the server's video.max_upload_mib (by default the disk's free space less a margin).  A job's record and files live in
the data folder's video-jobs/<id>/ until DELETE.  One job runs at a time, in the image jobs' GPU turn; a VFR source is
read at its average rate (the record says so), an HDR one refused (its reason).
"""
from __future__ import annotations

import base64
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from serve import image_aux
from serve import images as img
from serve import envvars

SYNC_MAX_FRAMES = 2048                 # vLLM-Omni's /v1/videos/segmentations and /v1/videos/sync limits
SYNC_MAX_BYTES = 512 << 20
JOB_DISK_MARGIN = 2 << 30              # what a job's upload leaves free on the disk (the result is written there too)
FILE_FIELDS = ("input_references", "input_reference", "video", "file")
CODECS = ("h264", "vp9", "ffv1")
SEG_MAX_PROMPTS = 16                   # vLLM-Omni's MAX_VIDEO_PROMPTS
SEG_OUTPUTS = ("mask", "overlay")
CHUNK = 1 << 20                        # an upload's read


class VideoRequestError(ValueError):
    """A request the video tools refuse (400), or past a limit (status 413)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------------------------------- the upload
def safe_suffix(name: str) -> str:
    """The file name's extension for a path the image engine reads from a space-separated line: only ASCII letters
    and digits after the dot.  A multipart header's filename can hold spaces and bare newlines; either would split
    the engine's UPSV/SEGV line into commands (a name like x.a\\nSHUTDOWN\\nX stops the image daemon)."""
    suffix = Path(name).suffix[:16]
    letters = "".join(c for c in suffix[1:] if c.isascii() and c.isalnum()) if suffix.startswith(".") else ""
    return "." + letters if letters else ""


def stream_multipart(rfile, ctype: str, length: int, dest: Path, max_file: int) -> tuple[dict, list[dict]]:
    """A multipart/form-data body of `length` bytes read from `rfile` -> (its text fields, its files: {"field",
    "filename", "path", "bytes"}, each written under `dest` as it arrives).  A file past `max_file` bytes stops it
    (VideoRequestError 413)."""
    bound = None
    for part in ctype.split(";"):
        k, _, v = part.strip().partition("=")
        if k.lower() == "boundary":
            bound = v.strip().strip('"')
    if not bound:
        raise VideoRequestError("a multipart/form-data body without its boundary")
    delim = b"\r\n--" + bound.encode("latin-1")
    left = length
    buf = b"\r\n"   # the first delimiter has no CR LF before it: as if it had
    fields: dict = {}
    files: list[dict] = []

    def fill() -> bool:
        nonlocal buf, left
        if left <= 0:
            return False
        got = rfile.read(min(CHUNK, left))
        if not got:
            raise VideoRequestError("the upload ended early")
        left -= len(got)
        buf += got
        return True

    # to the first delimiter
    while (i := buf.find(delim)) < 0:
        if not fill():
            raise VideoRequestError("the multipart body has no parts")
    buf = buf[i + len(delim):]
    n = 0
    while True:
        while len(buf) < 2 and fill():
            pass
        if buf.startswith(b"--"):   # the closing delimiter
            break
        while (h := buf.find(b"\r\n\r\n")) < 0:
            if not fill():
                raise VideoRequestError("a multipart part without its headers' end")
        head = buf[:h].decode("latin-1")
        buf = buf[h + 4:]
        name = filename = None
        for line in head.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for item in line.split(";")[1:]:
                    k, _, v = item.strip().partition("=")
                    v = v.strip().strip('"')
                    if k.lower() == "name":
                        name = v
                    elif k.lower() == "filename":
                        filename = v
        out = None
        size = 0
        data = bytearray()
        if filename is not None:
            n += 1
            path = dest / f"upload{n}{safe_suffix(filename)}"
            out = open(path, "wb")
        try:
            while True:
                i = buf.find(delim)
                take = i if i >= 0 else max(0, len(buf) - len(delim))   # what cannot be part of a delimiter
                if take:
                    size += take
                    if out is not None:
                        if size > max_file:
                            raise VideoRequestError(f"the file is past the limit of {max_file >> 20} MiB", 413)
                        out.write(buf[:take])
                    elif size <= 1 << 20:
                        data += buf[:take]
                    buf = buf[take:]
                if i >= 0:
                    buf = buf[len(delim):]
                    break
                if not fill():
                    raise VideoRequestError("a multipart part without its end")
        finally:
            if out is not None:
                out.close()
        if out is not None:
            files.append({"field": name, "filename": filename, "path": path, "bytes": size})
        elif name:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


# ------------------------------------------------------------------------------------------- the source
def command(name: str) -> str | None:
    """ffmpeg's or ffprobe's command, as the engine finds it (video_io.hpp): XESTRATA_FFMPEG / XESTRATA_FFPROBE, else
    the PATH's."""
    return envvars.get(name.upper()) or shutil.which(name)


def available() -> bool:
    return command("ffprobe") is not None and command("ffmpeg") is not None


def status(cfg: dict, root: Path) -> dict:
    """/health's "video": whether the video tools can run (ffmpeg) and their limits."""
    try:
        limit = upload_limit(cfg, root) if root.is_dir() else upload_limit(cfg, root.parent)
    except OSError:
        limit = 0
    return {"available": available(), "sync_max_frames": SYNC_MAX_FRAMES, "sync_max_mib": SYNC_MAX_BYTES >> 20,
            "max_upload_mib": limit >> 20}


def probe(path: Path) -> dict:
    """What ffprobe says of the video: width and height as shown (a rotation applied), fps (its average), vfr, frames
    (the stream's count, else its duration times the rate), duration, an audio stream.  VideoRequestError for a file
    that is not a video, or HDR (PQ, HLG: its mapping to SDR is the user's call)."""
    ffprobe = command("ffprobe")
    if ffprobe is None or command("ffmpeg") is None:
        raise VideoRequestError("the video tools need ffmpeg and ffprobe, which are not installed here "
                                "(xestrata --setup says how to get them)", 503)
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,color_transfer:"
                        "stream_side_data=rotation:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise VideoRequestError("not a video ffmpeg reads: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    j = json.loads(r.stdout or "{}")
    streams = j.get("streams") or []
    if not streams:
        raise VideoRequestError("no video stream in the file")
    s = streams[0]
    transfer = s.get("color_transfer") or ""
    if transfer in ("smpte2084", "arib-std-b67"):
        raise VideoRequestError(f"an HDR video ({transfer}) is not taken: its mapping to SDR changes how it looks; "
                                "convert it to SDR first")

    def rate(v: str) -> float:
        a, _, b = (v or "0/0").partition("/")
        try:
            return float(a) / float(b) if b else float(a)
        except (ValueError, ZeroDivisionError):
            return 0.0
    r_rate, avg = rate(s.get("r_frame_rate")), rate(s.get("avg_frame_rate"))
    fps = avg or r_rate
    if fps <= 0:
        raise VideoRequestError("the video's frame rate is unknown")
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    rot = 0
    for d in s.get("side_data_list") or []:
        if "rotation" in d:
            rot = int(round(float(d["rotation"])))
    if rot % 180:
        w, h = h, w
    dur = float(s.get("duration") or (j.get("format") or {}).get("duration") or 0)
    vfr = bool(r_rate and avg and abs(r_rate - avg) > 1e-3 * avg)
    frames = int(s.get("nb_frames") or 0) if not vfr else 0
    if frames <= 0:
        frames = max(1, round(dur * fps))
    a = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120)
    return {"width": w, "height": h, "fps": fps, "vfr": vfr, "frames": frames, "duration": dur,
            "audio": bool(a.stdout.strip())}


# ------------------------------------------------------------------------------------------- the request
def upscale_request(fields: dict, info: dict, cfg: dict, sync: bool) -> dict:
    """An upscale's form fields and its source's probe -> the job: images.upscale_fields' (extra_params, the upscaler:
    images.UPSCALE_MODELS' ids or a method's name, none: auto, SeedVR2 where it fits; color_correction_method), the
    output's size (size WIDTHxHEIGHT, else scale times the source's, even sides; images.UPSCALE_MAX_SIDE a side at
    most, as an image's upscale: the config's max_side is the drawing's, and a video is not drawn), the seed
    (images.parse_request's), and the video's own extra_params: chunk_frames, temporal_overlap and video_codec.  The
    synchronous answer's frames past SYNC_MAX_FRAMES: 400; every refusal a VideoRequestError."""
    if sync and info["frames"] > SYNC_MAX_FRAMES:
        raise VideoRequestError(f"the synchronous answer takes at most {SYNC_MAX_FRAMES} frames (this video has about "
                                f"{info['frames']}); POST /v1/videos makes a job of it")
    try:
        u = img.upscale_fields(fields)
        extra = u["extra"]
        size = str(fields.get("size") or "").strip()
        if size:
            try:
                w, h = (int(v) for v in size.lower().split("x"))
            except ValueError:
                raise VideoRequestError("size: WIDTHxHEIGHT") from None
        else:
            scale = img.parse_number(fields.get("scale") or extra.get("scale") or None, "scale", 0.25, 8, 2.0)
            w, h = round(info["width"] * scale / 2) * 2, round(info["height"] * scale / 2) * 2
        top = img.UPSCALE_MAX_SIDE
        if not (16 <= w <= top and 16 <= h <= top):
            raise VideoRequestError(f"size: each side from 16 to {top}")
        chunk = img.parse_number(extra.get("chunk_frames"), "chunk_frames", 0, math.inf, 0, whole=True)
        if chunk and (chunk - 1) % 4:
            raise VideoRequestError("chunk_frames: 4 n + 1 frames (1, 5, 9, 13, 17, 21 ...)")
        overlap = img.parse_number(extra.get("temporal_overlap"), "temporal_overlap", 0, math.inf, 0, whole=True)
        codec = extra.get("video_codec")
        if codec is not None and codec not in CODECS:
            raise VideoRequestError(f"video_codec: one of {', '.join(CODECS)}")
        up = img.upscaler_of("seedvr2" if u["method"] == "auto" else u["method"], u["model"], u["color"],
                             u["variant"])
        job = img.ups_only_job({"prompt": "upscale", "seed": fields.get("seed")}, cfg, w, h, up)   # its seed too
    except img.ImageRequestError as e:   # the video entries' 400 (explain: invalid_video)
        raise VideoRequestError(str(e)) from None
    job.update(video=True, chunk=chunk, overlap=overlap, codec=codec, source=info, gpu=gpu_of(fields))
    return job


def gpu_of(fields: dict) -> str | int:
    """The form's GPU for the job (images.gpu_field: gpu, else extra_params' gpu), else a VideoRequestError."""
    try:
        return img.gpu_field(fields)
    except img.ImageRequestError as e:
        raise VideoRequestError(str(e)) from None


def _json_field(fields: dict, name: str):
    v = fields.get(name)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            raise VideoRequestError(f"{name}: JSON") from None
    return v


def flag_field(fields: dict, name: str) -> bool:
    """A video form's true or false (images.parse_flag's words; absent or blank: false), else a VideoRequestError."""
    try:
        return img.parse_flag(fields.get(name), name, False)
    except img.ImageRequestError as e:
        raise VideoRequestError(str(e)) from None


def segment_request(fields: dict, info: dict, sync: bool) -> dict:
    """A segmentation's form fields and its source's probe -> the job: its words (prompt, then prompts: strings or
    {"prompt", "frame"}; repeats dropped, at most SEG_MAX_PROMPTS), its clicks, mask_video and outputs.  The words'
    tokens are the engine's call (segment's), the files' too."""
    words: list[dict] = []
    seen = set()
    extra = _json_field(fields, "prompts") or []
    if not isinstance(extra, list):
        raise VideoRequestError("prompts: a JSON list of strings (or of {\"prompt\", \"frame\"})")
    for e in [fields.get("prompt") or "", *extra]:
        text, frame = (e.get("prompt"), e.get("frame", 0)) if isinstance(e, dict) else (e, 0)
        if not isinstance(text, str):
            raise VideoRequestError("prompts: strings, or {\"prompt\": a string, \"frame\": a number}")
        text = " ".join(text.split())
        try:
            frame = int(frame or 0)
        except (TypeError, ValueError):
            raise VideoRequestError("prompts' frame: a frame's number") from None
        if text and (text, frame) not in seen:
            seen.add((text, frame))
            words.append({"prompt": text, "frame": max(0, frame)})
    if len(words) > SEG_MAX_PROMPTS:
        raise VideoRequestError(f"at most {SEG_MAX_PROMPTS} prompts")
    clicks = []
    for i, c in enumerate(_json_field(fields, "clicks") or []):
        if not isinstance(c, dict):
            raise VideoRequestError("clicks: a JSON list of {\"frame\", \"object\", \"points\", \"box\"}")
        try:
            pts = [[float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1] for p in c.get("points") or []]
            box = [float(v) for v in c.get("box") or []]
            # a page knows the time it shows, not the frame: the frame then that time's at the rate read
            frame = int(float(c["time"]) * info["fps"] + 1e-3) if c.get("frame") is None and c.get("time") is not None \
                else int(c.get("frame") or 0)
            given = c.get("object")
            obj = -1 if given in (None, "", -1) else int(str(given))
        except (TypeError, ValueError, IndexError):
            raise VideoRequestError(f"clicks[{i}]: frame, object, points [[x, y, label]...], box [x0, y0, x1, y1]") \
                from None
        if box and len(box) != 4:
            raise VideoRequestError(f"clicks[{i}].box: [x0, y0, x1, y1]")
        if any(p[2] not in (0, 1) for p in pts):
            raise VideoRequestError(f"clicks[{i}].points: labels 1 (in) or 0 (out)")
        if not pts and not box:
            raise VideoRequestError(f"clicks[{i}]: points or a box")
        if not 0 <= frame < max(info["frames"], 1) + 1:
            raise VideoRequestError(f"clicks[{i}].frame: from 0 to {info['frames'] - 1}")
        corners = [[min(box[0], box[2]), min(box[1], box[3]), 2], [max(box[0], box[2]), max(box[1], box[3]), 3]] \
            if box else []
        clicks.append({"frame": frame, "object": obj, "points": corners + pts})
    if not words and not clicks:
        raise VideoRequestError("prompt: the words to look for, or clicks on a frame (points or boxes)")
    outputs = _json_field(fields, "outputs") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    if not isinstance(outputs, list) or any(o not in SEG_OUTPUTS for o in outputs):
        raise VideoRequestError(f"outputs: a JSON list of {', '.join(SEG_OUTPUTS)}")
    if sync and info["frames"] > SYNC_MAX_FRAMES:
        raise VideoRequestError(f"the synchronous answer takes at most {SYNC_MAX_FRAMES} frames (this video has about "
                                f"{info['frames']}); job true makes a job of it")
    return {"segment": {"words": words, "clicks": clicks, "mask_video": flag_field(fields, "mask_video"),
                        "outputs": list(dict.fromkeys(outputs))},
            "width": info["width"], "height": info["height"], "n": 1, "steps": 1, "seed": 0, "source": info,
            "gpu": gpu_of(fields)}


def segment_answer(got: dict, job: dict, model: str, out: Path) -> dict:
    """The engine's SEGVIDEO fields and its files -> vLLM-Omni's answer written to `out` (the frames copied from the
    engine's JSON lines as they are, never parsed: a long clip's are tens of MB) -> {"path", "files": {variant: its
    file}, "objects"}."""
    seg = job["segment"]
    words = seg["words"]
    objects, files = [], {}
    masks = dict(e.split(":", 1) for e in str(got.get("masks") or "").split(",") if ":" in e)
    for e in str(got.get("objects") or "").split(","):
        if ":" not in e:
            continue
        oid, src = e.split(":", 1)
        o: dict = {"id": int(oid)}
        if src.startswith("w"):
            w = words[int(src[1:])]
            o["prompt"] = w["prompt"]
            if w["frame"]:
                o["frame"] = w["frame"]
        else:
            o["prompt"] = ""
            o["click"] = int(src[1:])
        if oid in masks:
            files[f"object_{oid}"] = Path(masks[oid])
        objects.append(o)
    if got.get("mask"):
        files["mask"] = Path(got["mask"])
    if got.get("overlay"):
        files["overlay"] = Path(got["overlay"])

    def b64(f: Path) -> str:
        return base64.b64encode(f.read_bytes()).decode("ascii")
    for o in objects:
        if seg["mask_video"] and f"object_{o['id']}" in files:
            o["mask_video"] = b64(files[f"object_{o['id']}"])
    head = {"created": int(time.time()), "model": model, "width": int(got["width"]), "height": int(got["height"]),
            "num_frames": int(got["frames"]), "fps": float(got["fps"]), "objects": objects}
    if int(got.get("vfr") or 0):
        head["vfr"] = "read at its average frame rate"
    if "mask" in seg["outputs"] and "mask" in files:
        head["mask_union_video"] = b64(files["mask"])
    if "overlay" in seg["outputs"] and "overlay" in files:
        head["overlay_video"] = b64(files["overlay"])
    lines = Path(str(got["json"]))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(head)[:-1] + ',"frames":[')
        with open(lines, encoding="utf-8") as jl:
            for i, line in enumerate(jl):
                if line.strip():
                    fh.write(("," if i else "") + line.strip())
        fh.write("]}")
    lines.unlink(missing_ok=True)
    return {"path": out, "files": files, "objects": objects}


def hold_long_job(svc, gen, state: dict, held: bool) -> None:
    """A video job's state while it holds `gen`'s engine (`held` false: no more): the engine's (images.EngineLock), its
    card's (images.ImageModels.long_job: refuse_during_long_job, place's "auto") and the server's (svc.long_job: the
    chat's Service.load, the Monitor).  Two can run at once, one on each card (the text model's GPU lent to a video's
    gpu "text"): the server's is the one that refuses the chat (the text engine made way for it, policy unload), else
    the first."""
    owner = getattr(gen, "owner", None)
    with LONG_LOCK:
        jobs = [s for s in getattr(svc, "long_jobs", None) or [] if s is not state] + ([state] if held else [])
        svc.long_jobs = jobs
        svc.long_job = next((s for s in jobs if s.get("policy") == "unload"), jobs[0] if jobs else None)
        gen.long_job = state if held else None
        if isinstance(owner, img.ImageModels):
            owner.long_job = state if held else None


LONG_LOCK = threading.Lock()


def segment(svc, gen, job: dict, source: Path, stem: Path, model: str, progress: Callable[[int, int], None] | None,
            cancel: threading.Event | None) -> dict:
    """A segmentation's run: SAM 3.1's weights and its license (image_regions'), the words' CLIP tokens, then the
    engine's SEGV in the image jobs' GPU turn -> segment_answer's, with "figures"."""
    from serve import image_regions as regions
    img.refuse_during_long_job(svc, gen)
    if image_aux.needs_consent("sam3", regions.root_of(gen)):
        raise img.ImageRequestError(regions.sam3_consent_words())
    weights, tok = regions.weights(gen, regions.seg_files(regions.SAM3))
    seg = job["segment"]
    clip = regions._clip(tok)
    lines = []
    for w in seg["words"]:
        ids = clip.encode(w["prompt"], regions.SAM3_MAX_IDS)
        lines.append(f"W {w['frame']} {','.join(map(str, ids))}")
    for c in seg["clicks"]:
        lines.append(f"C {c['frame']} {c['object']} " + ",".join(f"{v!r}" for p in c["points"] for v in p))
    prompts = stem.parent / "prompts.txt"
    prompts.write_text("\n".join(lines) + "\n")
    job = {**job, "aux": {"kind": "sam3video", "path": str(weights), "need_args": [1]}}
    t0 = time.monotonic()
    with img.gpu_turn(svc, gen, job, cancel) as policy:
        state = {"started": time.monotonic(), "done": 0, "total": int(job["source"]["frames"]), "policy": policy,
                 "thread": threading.get_ident(), "client": img.CLIENT_JOB.get()}

        def told(done: int, total: int) -> None:
            state.update(done=done, total=total)
            if progress is not None:
                progress(done, total)
        hold_long_job(svc, gen, state, True)
        try:
            got = gen.segment_video(weights, source, prompts, stem, seg, cancel, told)
        finally:
            hold_long_job(svc, gen, state, False)
    answer = segment_answer(got, job, model, stem.parent / "segmentation.json")
    figures = {"seconds": round(time.monotonic() - t0, 2), "engine": "xestrata-image", "model": regions.SAM3,
               "frames": int(got.get("frames", 0)), "fps": float(got.get("fps", 0)),
               "vfr": bool(int(got.get("vfr") or 0)), "objects": len(answer["objects"]),
               "segment": {k: float(got[k]) for k in ("segment_s", "read_s", "trunk_s", "detect_s", "track_s",
                                                      "memory_s", "rules_s", "output_s", "write_s") if k in got}}
    gen.last = figures
    gen.end(None)
    return {**answer, "mime": "application/json", "figures": figures}


# ------------------------------------------------------------------------------------------- the jobs
class VideoJobs:
    """The jobs' records (record.json) and files under `root`/<id>/, one job run at a time by a thread of its own."""

    def __init__(self, root: Path,
                 run: Callable[[dict, Callable[[int, int], None], threading.Event, Path, Path, str], dict]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.lock = threading.Lock()
        self.queue: list[str] = []
        self.cancels: dict[str, threading.Event] = {}
        self.worker: threading.Thread | None = None
        for d in self.root.iterdir():   # a job a server stopped in the middle of: failed
            rec = self._read(d.name)
            if rec and rec.get("status") in ("queued", "in_progress"):
                rec.update(status="failed", error={"code": "server_restarted",
                                                   "message": "the server stopped before the job ended"})
                self._write(rec)

    def dir_of(self, vid: str) -> Path:
        if not vid.startswith("video_") or "/" in vid or ".." in vid:
            raise KeyError(vid)
        return self.root / vid

    def _read(self, vid: str) -> dict | None:
        with contextlib.suppress(OSError, ValueError, KeyError):
            return json.loads((self.dir_of(vid) / "record.json").read_text())
        return None

    def _write(self, rec: dict) -> None:
        d = self.dir_of(rec["id"])
        tmp = d / "record.json.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, d / "record.json")

    def new_dir(self) -> tuple[str, Path]:
        vid = "video_" + uuid.uuid4().hex
        d = self.root / vid
        d.mkdir()
        return vid, d

    def create(self, vid: str, model: str, job: dict, source: Path) -> dict:
        """A job of the request `job` on the uploaded `source` (in the job's folder): queued, its record answered."""
        rec = {"id": vid, "object": "video", "model": model, "status": "queued", "progress": 0,
               "kind": "segmentation" if job.get("segment") else "upscale",
               "created_at": int(time.time()), "completed_at": None, "size": f"{job['width']}x{job['height']}",
               "seconds": f"{job['source']['duration']:.3f}", "prompt": "", "quality": "default", "error": None,
               "media_type": "video/mp4", "file_name": None, "remixed_from_video_id": None, "expires_at": None,
               "source": {k: job["source"][k] for k in ("width", "height", "fps", "vfr", "frames")},
               "job": {k: v for k, v in job.items() if k != "source"}, "input": source.name}
        with self.lock:
            self._write(rec)
            self.queue.append(vid)
            self.cancels[vid] = threading.Event()
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._work, name="video-jobs", daemon=True)
                self.worker.start()
        return self.public(rec)

    @staticmethod
    def public(rec: dict) -> dict:
        return {k: v for k, v in rec.items() if k not in ("job", "input")}

    def _work(self) -> None:
        while True:
            with self.lock:
                if not self.queue:
                    self.worker = None
                    return
                vid = self.queue.pop(0)
                cancel = self.cancels.get(vid) or threading.Event()
            rec = self._read(vid)
            if rec is None or rec.get("status") != "queued":
                continue
            rec.update(status="in_progress", started_at=int(time.time()))
            self._write(rec)
            last = [0.0]

            def progress(done: int, total: int, rec=rec) -> None:
                now = time.monotonic()
                if now - last[0] < 1.0 and done < total:
                    return
                last[0] = now
                rec["progress"] = int(min(99, 100 * done / max(total, 1)))
                rec["frames_done"] = done
                with contextlib.suppress(OSError):
                    self._write(rec)
            try:
                job = dict(rec["job"])
                job["source"] = {**rec["source"], "duration": float(rec["seconds"])}
                d = self.dir_of(vid)
                out = self.run(job, progress, cancel, d / rec["input"], d / "out", rec["model"])
                rec.update(status="completed", progress=100, completed_at=int(time.time()),
                           file_name=Path(out["path"]).name, media_type=out["mime"], result=out.get("figures"),
                           files={k: Path(f).name for k, f in (out.get("files") or {}).items()},
                           objects=[{k: v for k, v in o.items() if k != "mask_video"}
                                    for o in out.get("objects") or []] or None)
            except Exception as e:   # the job's failure, said in its record
                rec.update(status="failed", completed_at=int(time.time()),
                           error={"code": getattr(e, "code", "video_failed"), "message": str(e)})
            finally:
                with contextlib.suppress(OSError):   # the source is not kept past the job
                    (self.dir_of(vid) / rec["input"]).unlink()
                with self.lock:
                    self.cancels.pop(vid, None)
            # delete() may have removed this job's folder while it ran: a missing record must not end the worker
            # before the rest of the queue starts.
            with contextlib.suppress(OSError):
                self._write(rec)

    def get(self, vid: str) -> dict | None:
        try:
            return self._read(vid)
        except KeyError:
            return None

    def list(self, limit: int = 20, after: str | None = None, order: str = "desc") -> dict:
        recs = []
        for d in self.root.iterdir():
            rec = self._read(d.name)
            if rec:
                recs.append(rec)
        recs.sort(key=lambda r: (r.get("created_at", 0), r["id"]), reverse=order != "asc")
        if after:
            ids = [r["id"] for r in recs]
            recs = recs[ids.index(after) + 1:] if after in ids else []
        page = recs[:max(1, min(limit, 100))]
        return {"object": "list", "data": [self.public(r) for r in page],
                "first_id": page[0]["id"] if page else None, "last_id": page[-1]["id"] if page else None,
                "has_more": len(recs) > len(page)}

    def content(self, vid: str, variant: str = "") -> tuple[Path, str] | None:
        """A completed job's file: its answer, or (variant) a segmentation's video (overlay, mask, object_<id>)."""
        rec = self.get(vid)
        if rec is None or rec.get("status") != "completed" or not rec.get("file_name"):
            return None
        if variant and variant != "video":
            name = (rec.get("files") or {}).get(variant)
            if not name:
                return None
            f = self.dir_of(vid) / name
            return f, file_mime(f)
        return self.dir_of(vid) / rec["file_name"], rec.get("media_type") or "video/mp4"

    def delete(self, vid: str) -> dict | None:
        rec = self.get(vid)
        if rec is None:
            return None
        with self.lock:
            if vid in self.cancels:
                self.cancels[vid].set()
            if vid in self.queue:
                self.queue.remove(vid)
        if rec.get("status") == "in_progress":   # its run ends at its next frame; the folder goes with it
            for _ in range(600):
                time.sleep(0.1)
                cur = self.get(vid)
                if cur is None or cur.get("status") != "in_progress":
                    break
        shutil.rmtree(self.dir_of(vid), ignore_errors=True)
        return {"id": vid, "object": "video.deleted", "deleted": True}


def jobs_of(svc) -> VideoJobs:
    """The server's video jobs (svc.video_jobs) under its default image model's folder (video-jobs/): made at their
    first use, run by the default edit model's engine as /v1/images/upscale is."""
    with svc.status_lock:
        if svc.video_jobs is None:
            def run(job, progress, cancel, source, stem, model):
                gen, _ = img.place(svc, svc.images.pick_edit(None, True), job.get("gpu", "auto"))
                if job.get("segment"):
                    return segment(svc, gen, job, source, stem, model, progress, cancel)
                return upscale(svc, gen, job, source, stem, progress, cancel)
            svc.video_jobs = VideoJobs(Path(svc.images.default.dir) / "video-jobs", run)
        return svc.video_jobs


# ------------------------------------------------------------------------------------------- the chat's files
# A video the chat's page uploads (POST /v1/videos/files): kept under the jobs' folder's files/ by a random name, which
# is the access to it (as an image file's link), for the chat's tools to take (upscale_video) and its page to play;
# kept FILE_KEEP_S
FILE_REF = re.compile(r"/v1/videos/files/([0-9a-f]{32}\.(?:mp4|webm|mkv|mov|m4v|avi))")
FILE_KEEP_S = 7 * 86400
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi")


def files_dir(jobs: VideoJobs) -> Path:
    d = jobs.root / "files"
    d.mkdir(exist_ok=True)
    return d


def store_file(jobs: VideoJobs, tmp: Path, name: str) -> dict:
    """An uploaded video (`tmp`, its client's file name `name`) kept as files/<32 hex digits><its extension> once
    ffprobe reads it (VideoRequestError else, the file removed) -> what the page shows and the chat's message says:
    its link and probe.  Files past FILE_KEEP_S go."""
    d = files_dir(jobs)
    now = time.time()
    for f in d.iterdir():
        with contextlib.suppress(OSError):
            if now - f.stat().st_mtime > FILE_KEEP_S:
                f.unlink()
    try:
        info = probe(tmp)
    except VideoRequestError:
        tmp.unlink(missing_ok=True)
        raise
    ext = Path(name).suffix.lower()
    dest = d / (uuid.uuid4().hex + (ext if ext in VIDEO_EXTS else ".mp4"))
    os.replace(tmp, dest)
    return {"object": "video.file", "id": dest.name, "url": f"/v1/videos/files/{dest.name}", "filename": name,
            "bytes": dest.stat().st_size,
            **{k: info[k] for k in ("width", "height", "fps", "vfr", "frames", "duration")}}


def file_path(jobs: VideoJobs, name: str) -> Path | None:
    """A kept file by its name (FILE_REF's), else None."""
    if not re.fullmatch(r"[0-9a-f]{32}\.\w{2,4}", name or ""):
        return None
    f = jobs.root / "files" / name
    return f if f.is_file() else None


def file_mime(f: Path) -> str:
    return {".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo"}.get(f.suffix.lower(), "video/mp4")


def start_from_file(svc, jobs: VideoJobs, f: Path, fields: dict, cfg: dict, kind: str = "upscale") -> dict:
    """A job (POST /v1/videos's, or a segmentation's) on a kept file `f`: the file linked (or copied) into the job's
    folder, the request's `fields` as the form's -> the job's record."""
    info = probe(f)
    job = segment_request(fields, info, False) if kind == "segment" else upscale_request(fields, info, cfg, False)
    img.card_of(svc, job["gpu"])   # a card that takes no image job: refused before the job
    vid, folder = jobs.new_dir()
    dest = folder / ("input" + f.suffix)
    try:
        try:
            os.link(f, dest)
        except OSError:
            shutil.copyfile(f, dest)
        return jobs.create(vid, str(fields.get("model") or ("sam3.1" if kind == "segment" else "auto")), job, dest)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def upload_limit(cfg: dict, root: Path) -> int:
    """A job's upload's limit: video.max_upload_mib, else the disk's free space under `root` less a margin."""
    video = cfg.get("video")
    mib = video.get("max_upload_mib") if isinstance(video, dict) else None
    if mib:
        return int(mib) << 20
    free = shutil.disk_usage(root).free
    return max(0, free - JOB_DISK_MARGIN)


def upscale(svc, gen, job: dict, source: Path, stem: Path, progress: Callable[[int, int], None] | None,
            cancel: threading.Event | None) -> dict:
    """An upscale job's run: its upscaler's weights (images.upscaler_weights: AuxPending while they download), SeedVR2
    where it fits (auto), then the engine's UPSV in the image jobs' GPU turn -> {"path", "mime", "figures"}."""
    img.refuse_during_long_job(svc, gen)
    img.upscaler_weights(gen, job)
    img.seedvr2_or_realesrgan(gen, job)
    t0 = time.monotonic()
    with img.gpu_turn(svc, gen, job, cancel) as policy:
        # the long job (images.long_job_left): other image requests, and the chat's when its engine made way, are
        # refused at once with the seconds left; under the other policies the chat's run at the chunks' ends
        state = {"started": time.monotonic(), "done": 0, "total": int(job["source"]["frames"]), "policy": policy,
                 "thread": threading.get_ident(), "client": img.CLIENT_JOB.get()}

        def told(done: int, total: int) -> None:
            state.update(done=done, total=total)
            if progress is not None:
                progress(done, total)
        hold_long_job(svc, gen, state, True)
        try:
            got = gen.upscale_video(job, source, stem, cancel, told)
        finally:
            hold_long_job(svc, gen, state, False)
    u = job["upscaler"]
    figures = {"seconds": round(time.monotonic() - t0, 2), "engine": "xestrata-image",
               "size": f"{job['width']}x{job['height']}", "seed": job["seed"],
               "frames": int(got.get("frames", 0)), "fps": float(got.get("fps", 0)), "vfr": bool(got.get("vfr")),
               "upscale": {"method": u["method"], "asked": u["asked"], "color": got.get("color", "none"),
                           **{k: u[k] for k in ("variant", "file", "fallback") if u.get(k)},
                           **{k: float(got[k]) for k in ("upscale_s", "read_s", "model_s", "resize_s", "color_s",
                                                         "write_s", "encode_s", "dit_s", "decode_s") if k in got},
                           **{k: int(got[k]) for k in ("chunk", "vae_tile", "vae_host", "vae_latents", "vae_slice")
                              if k in got}}}
    gen.last = figures
    gen.end(None)
    return {"path": got["path"], "mime": got.get("mime", "video/mp4"), "figures": figures}


def model_ids() -> list[str]:
    """/v1/models' entries of the video tools."""
    return [*img.UPSCALE_MODELS, "sam3.1"]


def explain(e: Exception) -> tuple[int, dict]:
    """A failed video request's status and error body."""
    if isinstance(e, VideoRequestError):
        return e.status, {"error": {"type": "invalid_request_error", "code": "invalid_video", "message": str(e)}}
    if isinstance(e, img.ImageRequestError):
        return 400, {"error": {"type": "invalid_request_error", "message": str(e)}}
    return 500, {"error": {"type": "server_error", "code": "video_failed", "message": str(e)}}
