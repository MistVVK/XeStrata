# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/yolo_pt.py - a YOLOv8 checkpoint (a PyTorch .pt: a zip of a pickle and its tensors' bytes) to a
safetensors file the image engine reads (xestrata-image --detect, src/image/yolo_detect.cpp), without PyTorch and
without the package the checkpoint was made with.

    python -I tools/image/yolo_pt.py IN.pt OUT.safetensors [--sha256 HEX]

The pickle is read by an unpickler that builds nothing but what this file defines: the module classes it names become
empty stand-ins that keep their attributes, the tensors' rebuild calls become records of (storage, offset, shape,
stride), a storage is the zip member it names (read only after its size is checked), and every other global the
pickle asks for (a function, a class of another package) stops the read.  No function of the checkpoint's is called
and no state it carries is restored by its own code.  Opcodes outside the protocol-2 set a torch.save pickle uses are
refused, and so are the extension registry, persistent ids of another form, and members, tensors or totals past the
limits below.

What it writes changes no weight: each tensor's bytes are copied as the checkpoint stores them (FP16 or FP32, made
contiguous where a stride asks), under the module path the pickle gives it (model.<i>.<child>...<name>), with the
checkpoint's model configuration (its `yaml` dict: the layer list), class names and strides as JSON in the header's
__metadata__, beside the source file's SHA-256.  The output is written to a temporary file and renamed into place only
when it is complete.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import pickle
import pickletools
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

MAX_MEMBERS = 4096              # zip members (a YOLOv8-m has about 480)
MAX_PICKLE = 64 << 20           # the pickle's bytes
MAX_TOTAL = 1 << 30             # every member's bytes, uncompressed
MAX_TENSORS = 8192
MAX_DIMS = 8
DTYPES = {"HalfStorage": ("F16", 2), "FloatStorage": ("F32", 4), "LongStorage": ("I64", 8),
          "IntStorage": ("I32", 4), "BFloat16Storage": ("BF16", 2), "DoubleStorage": ("F64", 8)}
OPCODES = {"PROTO", "STOP", "MARK", "POP", "POP_MARK", "DUP", "BINGET", "LONG_BINGET", "BINPUT", "LONG_BINPUT",
           "MEMOIZE", "EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_SET", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3",
           "LIST", "DICT", "APPEND", "APPENDS", "SETITEM", "SETITEMS", "ADDITEMS", "BUILD", "NEWOBJ", "REDUCE",
           "GLOBAL", "STACK_GLOBAL", "BINPERSID", "NONE", "NEWTRUE", "NEWFALSE", "BININT", "BININT1", "BININT2",
           "LONG1", "BINFLOAT", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "BINBYTES", "SHORT_BINBYTES",
           "FRAME"}


class Refused(ValueError):
    """The checkpoint holds something this reader does not take."""


class Stub:
    """A module of the checkpoint's (any class of torch.nn or of the package that saved it): its attributes only."""

    cls = ""

    def __init__(self, *args, **kw):
        pass

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):   # (dict, slots)
            state = {**(state[0] or {}), **state[1]}
        if not isinstance(state, dict):
            raise Refused(f"a state of type {type(state).__name__} for {self.cls}")
        self.__dict__.update(state)


def _stub(module: str, name: str) -> type:
    return type(name, (Stub,), {"cls": f"{module}.{name}"})


class StorageType:
    def __init__(self, name: str):
        self.name = name
        self.dtype, self.size = DTYPES[name]


class Storage:
    def __init__(self, member: str, kind: StorageType, numel: int):
        self.member, self.kind, self.numel = member, kind, numel


class Tensor:
    def __init__(self, storage: Storage, offset: int, shape: tuple, stride: tuple):
        self.storage, self.offset, self.shape, self.stride = storage, offset, shape, stride


def _ints(v, what: str) -> tuple:
    if not isinstance(v, tuple) or len(v) > MAX_DIMS or not all(isinstance(i, int) and 0 <= i < 1 << 40 for i in v):
        raise Refused(f"a tensor's {what} {v!r}")
    return v


def _rebuild_tensor(storage, offset, shape, stride, *rest):
    if not isinstance(storage, Storage) or not isinstance(offset, int) or offset < 0:
        raise Refused("a tensor that is not a storage's view")
    shape, stride = _ints(tuple(shape), "shape"), _ints(tuple(stride), "stride")
    if len(shape) != len(stride):
        raise Refused("a tensor whose shape and stride differ in length")
    n = 1
    for d in shape:
        n *= d
    if n and offset + sum((d - 1) * s for d, s in zip(shape, stride)) >= storage.numel:
        raise Refused("a tensor past its storage's end")
    return Tensor(storage, offset, shape, stride)


def _rebuild_parameter(data, requires_grad=False, hooks=None):
    if not isinstance(data, Tensor):
        raise Refused("a parameter that is not a tensor")
    return data


def _getattr(obj, name):   # the pickle's getattr (a bound method saved as an attribute): never looked up
    return None


def _size(*args):
    return tuple(*args)


class _Device(str):
    pass


SAFE = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("__builtin__", "set"): set, ("builtins", "set"): set,
    ("__builtin__", "getattr"): _getattr, ("builtins", "getattr"): _getattr,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch", "Size"): _size,
    ("torch", "device"): _Device,
}
STUB_MODULES = ("torch.nn.modules.", "ultralytics.nn.")


class Reader(pickle.Unpickler):
    def __init__(self, data: bytes, zf: zipfile.ZipFile, prefix: str):
        super().__init__(io.BytesIO(data))
        self.zf, self.prefix = zf, prefix
        self.stubs: dict = {}
        self.storages: dict = {}

    def find_class(self, module, name):
        if (module, name) in SAFE:
            return SAFE[(module, name)]
        if module == "torch" and name in DTYPES:
            return StorageType(name)
        if module.startswith(STUB_MODULES) and name.isidentifier():
            return self.stubs.setdefault((module, name), _stub(module, name))
        raise Refused(f"the global {module}.{name}")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and isinstance(pid[1], StorageType)
                and isinstance(pid[2], str) and pid[2].isdigit() and isinstance(pid[4], int) and pid[4] >= 0):
            raise Refused(f"a persistent id {pid!r}")
        key = pid[2]
        if key not in self.storages:
            member = f"{self.prefix}data/{key}"
            try:
                info = self.zf.getinfo(member)
            except KeyError:
                raise Refused(f"a storage {key} the archive does not have") from None
            if info.file_size != pid[4] * pid[1].size:
                raise Refused(f"storage {key}: {info.file_size} bytes for {pid[4]} elements")
            self.storages[key] = Storage(member, pid[1], pid[4])
        return self.storages[key]


def _check_opcodes(data: bytes) -> None:
    for op, _, _ in pickletools.genops(data):
        if op.name not in OPCODES:
            raise Refused(f"the pickle opcode {op.name}")


def _modules(obj, path: str, out: dict) -> None:
    """The tensors of `obj` (a module stand-in) and its children, by their module path."""
    if len(out) > MAX_TENSORS:
        raise Refused("too many tensors")
    for group in ("_parameters", "_buffers"):
        for k, v in (getattr(obj, group, None) or {}).items():
            if isinstance(v, Tensor):
                out[f"{path}{k}"] = v
    for k, child in (getattr(obj, "_modules", None) or {}).items():
        if isinstance(child, Stub):
            _modules(child, f"{path}{k}.", out)


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(i) for k, i in v.items()}
    if isinstance(v, Tensor):
        return None
    return str(v)


def _bytes(zf: zipfile.ZipFile, t: Tensor, cache: dict) -> bytes:
    if t.storage.member not in cache:
        cache[t.storage.member] = zf.read(t.storage.member)
    raw, size = cache[t.storage.member], t.storage.kind.size
    n = 1
    for d in t.shape:
        n *= d
    contiguous: list[int] = []
    s = 1
    for d in reversed(t.shape):
        contiguous.insert(0, s)
        s *= d
    if n == 0:
        return b""
    if list(t.stride) == contiguous or n == 1:
        return raw[t.offset * size:(t.offset + n) * size]
    out = bytearray(n * size)   # strided: gathered element by element (a checkpoint's few transposed tensors)
    idx = [0] * len(t.shape)
    for i in range(n):
        src = t.offset + sum(a * b for a, b in zip(idx, t.stride))
        out[i * size:(i + 1) * size] = raw[src * size:(src + 1) * size]
        for d in range(len(idx) - 1, -1, -1):
            idx[d] += 1
            if idx[d] < t.shape[d]:
                break
            idx[d] = 0
    return bytes(out)


def read(path: Path) -> tuple[dict, dict]:
    """(tensors by name, metadata) of the checkpoint at `path`."""
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_TOTAL:
            raise Refused("an archive past the size limits")
        for i in infos:
            if i.filename.startswith("/") or ".." in i.filename.split("/") or "\\" in i.filename:
                raise Refused(f"the member name {i.filename!r}")
        pkl = [i for i in infos if i.filename.endswith("/data.pkl") and i.filename.count("/") == 1]
        if len(pkl) != 1 or pkl[0].file_size > MAX_PICKLE:
            raise Refused("no data.pkl, or more than one, or one past the size limit")
        prefix = pkl[0].filename[:-len("data.pkl")]
        data = zf.read(pkl[0])
        _check_opcodes(data)
        r = Reader(data, zf, prefix)
        top = r.load()
        if not isinstance(top, dict):
            raise Refused("a checkpoint that is not a dict")
        model = top.get("ema") if isinstance(top.get("ema"), Stub) else top.get("model")
        if not isinstance(model, Stub):
            raise Refused("no model in the checkpoint")
        tensors: dict = {}
        _modules(model, "", tensors)
        cache: dict = {}
        blobs = {k: (t.storage.kind.dtype, list(t.shape), _bytes(zf, t, cache)) for k, t in tensors.items()}
        head = model.__dict__
        stride = head.get("stride")
        meta = {"yaml": _jsonable(head.get("yaml")), "names": _jsonable(head.get("names")),
                "task": _jsonable(head.get("task")), "class": model.cls,
                "stride": _jsonable(_bytes_list(zf, stride, cache)) if isinstance(stride, Tensor) else None,
                "version": _jsonable(top.get("version")), "date": _jsonable(top.get("date")),
                "license": _jsonable(top.get("license"))}
        meta.update(_layout(model))
        return blobs, meta


def _layout(model: Stub) -> dict:
    """What the layer list leaves to the modules' attributes: the batch norms' epsilon and, for each block of
    bottlenecks, whether each adds its input back (its `add`)."""
    eps: set = set()
    residual: dict = {}

    def walk(o, path):
        if o.cls.endswith(".BatchNorm2d") and isinstance(o.__dict__.get("eps"), float):
            eps.add(o.__dict__["eps"])
        if o.cls.endswith(".Bottleneck"):
            residual.setdefault(path.rsplit(".", 2)[0], []).append(bool(o.__dict__.get("add")))
        for k, c in (o.__dict__.get("_modules") or {}).items():
            if isinstance(c, Stub):
                walk(c, f"{path}.{k}" if path else k)

    walk(model, "")
    if len(eps) > 1:
        raise Refused(f"batch norms of several epsilons {sorted(eps)}")
    return {"bn_eps": eps.pop() if eps else None, "residual": residual, "graph": _graph(model, residual)}


def _graph(model: Stub, residual: dict) -> str:
    """The layer list as the engine reads it, one layer a `;`-separated entry "<index> <module> <inputs> <stride>
    <shortcut>": the module's class name, its inputs' indices (-1 the layer before) joined by commas, a convolution's
    stride (1 otherwise) and whether a block's bottlenecks add their input back (1 or 0)."""
    yaml = model.__dict__.get("yaml")
    if not (isinstance(yaml, dict) and isinstance(yaml.get("backbone"), list) and isinstance(yaml.get("head"), list)):
        raise Refused("no layer list in the checkpoint's configuration")
    out = []
    for i, entry in enumerate(yaml["backbone"] + yaml["head"]):
        if not (isinstance(entry, list) and len(entry) == 4 and isinstance(entry[2], str)
                and isinstance(entry[3], list)):
            raise Refused(f"layer {i}: {entry!r}")
        f, _, mod, args = entry
        ins = f if isinstance(f, list) else [f]
        if not all(isinstance(v, int) for v in ins):
            raise Refused(f"layer {i}: inputs {f!r}")
        mod = mod.rsplit(".", 1)[-1]
        stride = args[2] if mod == "Conv" and len(args) > 2 and isinstance(args[2], int) else 1
        adds = residual.get(f"model.{i}", [])
        out.append(f"{i} {mod} {','.join(str(v) for v in ins)} {stride} {1 if adds and all(adds) else 0}")
    return ";".join(out)


def _bytes_list(zf, t: Tensor, cache) -> list:
    raw = _bytes(zf, t, cache)
    fmt = {"F16": "e", "F32": "f", "I64": "q", "I32": "i", "F64": "d"}.get(t.storage.kind.dtype)
    return list(struct.unpack(f"<{len(raw) // t.storage.kind.size}{fmt}", raw)) if fmt else []


def write(out: Path, blobs: dict, meta: dict) -> None:
    header: dict = {"__metadata__": {k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()}}
    off = 0
    for k in sorted(blobs):
        dt, shape, raw = blobs[k]
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out.name + ".", suffix=".part", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for k in sorted(blobs):
                f.write(blobs[k][2])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def convert(src: Path, out: Path, want_sha: str | None = None) -> dict:
    """Converts `src` into `out`; the metadata written.  Raises Refused (or zipfile's errors) on a file it does not
    take, and ValueError when `want_sha` is given and the file's SHA-256 is another."""
    got = sha256(src)
    if want_sha and got != want_sha.lower():
        raise ValueError(f"{src.name}: SHA-256 {got}, not the pinned {want_sha}")
    blobs, meta = read(src)
    meta["source"] = src.name
    meta["source_sha256"] = got
    write(out, blobs, meta)
    return meta


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--sha256")]
    want = None
    for i, a in enumerate(argv):
        if a == "--sha256" and i + 1 < len(argv):
            want = argv[i + 1]
            args.remove(want)
        elif a.startswith("--sha256="):
            want = a.split("=", 1)[1]
    if len(args) != 2:
        print(__doc__)
        return 2
    try:
        meta = convert(Path(args[0]), Path(args[1]), want)
    except (Refused, ValueError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError) as e:
        print(f"yolo_pt: {e}", file=sys.stderr)
        return 1
    print(f"yolo_pt: {args[1]} ({meta['class']}, {len(json.loads(json.dumps(meta['names'])) or [])} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
