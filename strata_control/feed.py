"""The signed update feed: what is published, how it is checked, which engine package fits a PC.

One small document, ``strata-feed.json``, with a detached Ed25519 signature ``strata-feed.json.sig`` (the signature covers the
file's exact bytes, so there is no canonicalisation to get wrong). It lists, for one channel:

* ``app``: the desktop app, in exactly the shape of Tauri's static updater JSON (``version``, ``notes``, ``pub_date``,
  ``platforms{<tauri target>: {url, signature}}``), plus ``sha256`` and ``size`` per platform. The Tauri updater plugin
  reads a copy of this member at ``app/latest.json`` and checks the minisign signature itself;
* ``control``: the runtime component (Control: bundled Python, ``strata_control``, the CLI and server), one archive per platform;
* ``engine``: the engine packages, one entry per platform / backend / variant, each with ``version``, ``archs``,
  ``min_driver``, ``min_runtime``, ``size``, ``sha256`` of the archive, ``manifest_sha256`` of the package's
  ``engine-component.json`` (which holds every file's hash) and a URL.

The feed's signature plus those hashes is the whole chain of trust: a package whose archive hash or manifest hash is not the
one in the signed feed is never installed. Three refusals are tested: a bad signature, a bad hash and a downgrade. The
downgrade guard has two parts: a feed whose ``sequence`` is lower than the last one this PC accepted is refused (a replayed
old feed), and a package older than the installed one is never installed by an update.

Nothing here touches the network itself; `strata_control.transfer` does.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field

from strata_control import netpolicy, shape_engine, signing, transfer
from strata_control.targets import BACKEND_MATRIX, host_platform

SCHEMA_VERSION = 1
FEED_NAME = "strata-feed.json"
#: The Linux ARM64 (DGX Spark) Terminal reads this file instead. A Control older than 0.1.53 refuses a feed that names a platform it does not know, so
#: the linux-aarch64 entries cannot sit in strata-feed.json: that file stays as every released client can read it. Same format, same key.
AARCH64_FEED_NAME = "strata-feed-linux-aarch64.json"
SIGNATURE_SUFFIX = ".sig"
FORMATS = ("tar.gz", "tar.xz", "zip")
PLATFORMS = ("windows-x86_64", "linux-x86_64", "linux-aarch64", "macos-arm64", "macos-x86_64")
#: Tauri's names for the same platforms (no desktop app is built for linux-aarch64: the entry only keeps the name parsable).
TAURI_TARGETS = {"windows-x86_64": "windows-x86_64", "linux-x86_64": "linux-x86_64", "linux-aarch64": "linux-aarch64",
                 "macos-arm64": "darwin-aarch64", "macos-x86_64": "darwin-x86_64"}
_VERSION = re.compile(r"\d+\.\d+\.\d+(?:\.\d+)?")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_log = logging.getLogger("strata.feed")


class FeedError(RuntimeError):
    """The feed or something it lists cannot be trusted or used. `exit_code` is the CLI's."""

    def __init__(self, message: str, exit_code: int = 5):
        super().__init__(message)
        self.exit_code = exit_code


class FeedUnreachable(FeedError):
    """The feed's host did not answer, or this computer is offline (strata_control/netpolicy.py's words): a check that runs by itself stays silent."""


class FeedDowngrade(FeedError):
    """A feed or a package older than what this PC already has."""
    # .sequence is the feed's, .accepted the last one this PC took (the updater adds the folder that holds it)
    sequence: int | None = None
    accepted: int | None = None


def version_key(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise FeedError(f"{value!r} is not a version number (expected like 0.1.46 or 0.1.46.1)")
    parts = [int(p) for p in value.split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


def app_version_key(value: str) -> tuple[int, ...]:
    """Compare the INSTALLED app's version: like `version_key`, but a pre-release (0.2.0-beta) is allowed and sorts below the same number without
    the suffix. The feed's own versions stay strict (`version_key`)."""
    base, dash, pre = str(value).partition("-")
    if dash and not re.fullmatch(r"[0-9A-Za-z.]+", pre):
        raise FeedError(f"{value!r} is not a version number (expected like 0.2.0 or 0.2.0-beta)")
    return version_key(base) + (0 if dash else 1,)


def _need(obj: dict, key: str, kind, where: str):
    value = obj.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise FeedError(f"the feed's {where} has no valid {key}")
    return value


def _hex(value, where: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value.lower()):
        raise FeedError(f"the feed's {where} has no valid sha256")
    return value.lower()


@dataclass(frozen=True)
class EnginePackage:
    version: str
    platform: str
    backend: str
    variant: str
    archs: tuple[str, ...]
    min_driver: dict
    min_runtime: str | None
    build_id: str
    url: str
    format: str
    size: int
    sha256: str
    manifest_sha256: str
    installed_bytes: int
    notes: str = ""
    shape: str = ""                 # the engine of one model shape (feed member ``shape_engines``, strata_control/shape_engine.py): its compiled geometry's name; "" for the others

    @property
    def label(self) -> str:
        return f"{self.version} {self.platform}-{self.backend}" + (f"-{self.variant}" if self.variant else "")

    def to_dict(self) -> dict:
        return {"version": self.version, "platform": self.platform, "backend": self.backend, "variant": self.variant,
                "archs": list(self.archs), "min_driver": dict(self.min_driver), "min_runtime": self.min_runtime,
                "build_id": self.build_id, "size": self.size, "installed_bytes": self.installed_bytes,
                "format": self.format, "url": self.url, "sha256": self.sha256, "manifest_sha256": self.manifest_sha256,
                **({"shape": self.shape} if self.shape else {})}


@dataclass(frozen=True)
class DrafterPackage:
    """A DFlash2 drafter file Strata hosts (feed member ``drafters``): one pinned file, not an archive. ``backends`` are the engine families it was measured on
    (``metal``); ``url`` is relative to the feed (flat release assets) or an https address; ``size`` and ``sha256`` are the file's. The code that fetches it
    pins the same size and SHA-256 (strata_control/drafting.py), so the feed can move the file but never swap its bytes."""
    name: str
    version: str
    url: str
    size: int
    sha256: str
    backends: tuple[str, ...]
    base: str = ""
    licence: str = ""
    notes: str = ""

    @property
    def label(self) -> str:
        return f"drafter {self.name} {self.version}"

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "url": self.url, "size": self.size, "sha256": self.sha256,
                "backends": list(self.backends), "base": self.base, "licence": self.licence, "notes": self.notes}


#: Families a ``models`` entry may name that this Control knows how to install (GGUF architectures the engine runs: ``strata --families``). A model of another family
#: is still listed, as "not available on this engine"; it is never offered to install.
MODEL_FAMILIES_INSTALLABLE = ("qwen4exp", "qwen35moe")
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9.-]{1,39}")
_MODEL_FAMILY = re.compile(r"[a-z0-9_]{2,32}")
_MODEL_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_MODEL_REVISION = re.compile(r"[0-9a-f]{40}")
_MODEL_QUANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")
MODEL_DRAFTERS = ("mtp", "none")


@dataclass(frozen=True)
class ModelFile:
    """One pinned file of a feed model: ``path`` is its place in the Hugging Face repository (``name`` when it sits at the top), ``size`` and ``sha256`` are its exact bytes."""
    name: str
    size: int
    sha256: str
    path: str = ""

    @property
    def repo_path(self) -> str:
        return self.path or self.name

    def to_dict(self) -> dict:
        out = {"name": self.name, "size": self.size, "sha256": self.sha256}
        if self.path:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class ModelQuant:
    """One size of a feed model: the files that make it, and the setup recommendation for it (``ram_gb`` the memory it wants; ``arena_gb`` the experts' share, 0: not given)."""
    name: str
    about: str
    ram_gb: float
    arena_gb: float
    files: tuple[ModelFile, ...]

    @property
    def download_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def to_dict(self) -> dict:
        return {"name": self.name, "about": self.about, "ram_gb": self.ram_gb, "arena_gb": self.arena_gb, "files": [f.to_dict() for f in self.files]}


@dataclass(frozen=True)
class FeedModel:
    """A model Strata lists after the built-in ones (feed member ``models``). Everything setup needs to install it is here and signed: the repository and the pinned revision,
    each file's exact size and SHA-256 (setup checks them after the download, and a wrong file is kept as ``.bad``, exactly as for a built-in model), the engine family,
    the licence and the first engine version that reads it."""
    id: str
    title: str
    by: str
    about: str
    licence: str
    licence_url: str
    family: str
    repo: str
    revision: str
    quants: tuple[ModelQuant, ...]
    mmproj: ModelFile | None = None
    pictures: bool = False
    drafter: str = "none"
    context: int = 0
    min_engine: str = ""
    kv_bytes_per_token: int = 0
    mtp_drafts: int = 0
    state_gb: float = 0.0

    @property
    def label(self) -> str:
        return f"model {self.id}"

    @property
    def shape(self) -> str | None:
        """The model shape whose own engine runs it (``q4_lite``), or None: it runs on the Strata engine."""
        return shape_engine.shape_of_family(self.family)

    def to_dict(self) -> dict:
        out = {"id": self.id, "title": self.title, "by": self.by, "about": self.about, "licence": self.licence, "licence_url": self.licence_url,
               "family": self.family, "repo": self.repo, "revision": self.revision, "quants": [q.to_dict() for q in self.quants],
               "pictures": self.pictures, "drafter": self.drafter, "context": self.context, "min_engine": self.min_engine}
        if self.mmproj:
            out["mmproj"] = self.mmproj.to_dict()
        if self.kv_bytes_per_token:
            out["kv_bytes_per_token"] = self.kv_bytes_per_token
        if self.mtp_drafts:
            out["mtp_drafts"] = self.mtp_drafts
        if self.state_gb:
            out["state_gb"] = self.state_gb
        return out


@dataclass(frozen=True)
class ControlPackage:
    version: str
    platform: str
    url: str
    format: str
    size: int
    sha256: str
    installed_bytes: int
    entry: str = "strata.py"

    @property
    def label(self) -> str:
        return f"control {self.version} {self.platform}"


@dataclass(frozen=True)
class AppRelease:
    version: str
    notes: str
    pub_date: str
    platforms: dict          # {platform: {"url", "signature", "sha256", "size", "format"?}}

    def for_platform(self, platform: str) -> dict | None:
        return self.platforms.get(platform)


@dataclass
class Feed:
    sequence: int
    channel: str
    generated_at: str
    expires_at: str | None
    app: AppRelease | None
    control: tuple[ControlPackage, ...]
    engine: tuple[EnginePackage, ...]
    source: str                    # where it was read from (the base for relative URLs)
    key_id: str                    # who signed it
    raw: bytes = field(default=b"", repr=False)
    #: the DeepSeek-V4.1 engine packages (feed member ``dsv41``, same entry shape as ``engine``): a separate download that setup fetches when
    #: DeepSeek is chosen. Kept out of ``engine`` so that `strata update` and select_engine never pick one as THE engine. Older Controls ignore the member.
    dsv41: tuple[EnginePackage, ...] = ()
    #: entries this Control does not know how to use (a platform, a backend or an archive format added after it was built): left out, one plain line each.
    #: A feed with such an entry is still a feed; only an entry of a kind this Control knows and that is malformed refuses the feed.
    skipped: tuple[str, ...] = ()
    #: the DFlash2 drafter files (feed member ``drafters``, DrafterPackage): fetched by setup, `strata drafting fetch` and Load for a set that has one. Older Controls ignore the member.
    drafters: tuple[DrafterPackage, ...] = ()
    #: the models Strata lists after the built-in ones (feed member ``models``, FeedModel). An entry that is malformed is left out of this tuple with one plain line in
    #: ``models_dropped`` (and the log); the rest of the feed, and the other models, work. Older Controls ignore the member.
    models: tuple[FeedModel, ...] = ()
    models_dropped: tuple[str, ...] = ()
    #: the detached signature that went with ``raw`` (fetch fills it for a feed that has models: the cache of the model list keeps both)
    signature: bytes = b""
    #: the engines of model shapes (feed member ``shape_engines``, EnginePackage with ``shape``): a separate component for each shape, fetched when a model of that shape is
    #: chosen (strata_control/shape_engine.py). Kept out of ``engine`` so that `strata update` never picks one as THE engine. Older Controls ignore the member.
    shape_engines: tuple[EnginePackage, ...] = ()

    def url(self, ref: str) -> str:
        return transfer.resolve_ref(transfer.base_of(self.source), ref)


# ---------------------------------------------------------------------------------------------- parsing

def _time(value, what: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise FeedError(f"the feed's {what} is not a timestamp") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _skipping(skipped: list | None, what: str, reason: str) -> None:
    if skipped is None:                                  # a caller that wants the strict reading
        raise FeedError(f"the update feed lists {what}, {reason}")
    skipped.append(f"The update feed lists {what}, {reason}: skipped (a newer Strata may use it).")


def _engine_package(item: dict, skipped: list | None = None) -> EnginePackage | None:
    if not isinstance(item, dict):
        raise FeedError("an engine entry in the feed is not an object")
    where = f"engine entry {item.get('version')!r}"
    version = _need(item, "version", str, where)
    version_key(version)
    platform = _need(item, "platform", str, where)
    backend = _need(item, "backend", str, where)
    label = f"{version} {platform}-{backend}" + (f"-{item['variant']}" if item.get("variant") else "")
    if platform not in PLATFORMS:
        _skipping(skipped, f"engine {label}", f"for a platform this Strata does not know ({platform})")
        return None
    if backend not in BACKEND_MATRIX:
        _skipping(skipped, f"engine {label}", f"for a backend this Strata does not know ({backend})")
        return None
    archs = item.get("archs")
    if not isinstance(archs, list) or not all(isinstance(a, str) and a for a in archs) or (backend != "cpu" and not archs):
        raise FeedError(f"the feed's {where} needs a list of archs")
    driver = item.get("min_driver") or {}
    if not isinstance(driver, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in driver.items()):
        raise FeedError(f"the feed's {where} has an invalid min_driver")
    fmt = _need(item, "format", str, where)
    if fmt not in FORMATS:
        _skipping(skipped, f"engine {label}", f"in an archive format this Strata does not know ({fmt})")
        return None
    size = _need(item, "size", int, where)
    installed = _need(item, "installed_bytes", int, where)
    if size <= 0 or installed <= 0:
        raise FeedError(f"the feed's {where} declares a size that cannot be a real file")
    runtime = item.get("min_runtime")
    if runtime is not None:
        version_key(runtime)
    return EnginePackage(version=version, platform=platform, backend=backend, variant=str(item.get("variant") or ""),
                         archs=tuple(archs), min_driver=dict(driver), min_runtime=runtime,
                         build_id=_need(item, "build_id", str, where), url=_need(item, "url", str, where), format=fmt,
                         size=size, sha256=_hex(item.get("sha256"), where),
                         manifest_sha256=_hex(item.get("manifest_sha256"), where), installed_bytes=installed,
                         notes=str(item.get("notes") or ""))


def _shape_engine_package(item: dict, skipped: list | None = None) -> EnginePackage | None:
    """An entry of ``shape_engines``: an engine entry that names its model shape. An entry whose shape is not a name this Control accepts is left out like any entry of a kind it
    does not know (a newer Strata may use it)."""
    if not isinstance(item, dict):
        raise FeedError("a shape engine entry in the feed is not an object")
    shape = item.get("shape")
    if not shape_engine.valid(shape):
        _skipping(skipped, f"an engine for the model shape {shape!r}", "a shape name this Strata does not accept")
        return None
    pkg = _engine_package(item, skipped)
    return dataclasses.replace(pkg, shape=shape) if pkg is not None else None


def _control_package(item: dict, version: str, skipped: list | None = None) -> ControlPackage | None:
    where = "control entry"
    platform = _need(item, "platform", str, where)
    if platform not in PLATFORMS:
        _skipping(skipped, f"Control {version} {platform}", "for a platform this Strata does not know")
        return None
    fmt = _need(item, "format", str, where)
    if fmt not in FORMATS:
        _skipping(skipped, f"Control {version} {platform}", f"in an archive format this Strata does not know ({fmt})")
        return None
    size, installed = _need(item, "size", int, where), _need(item, "installed_bytes", int, where)
    if size <= 0 or installed <= 0:
        raise FeedError("the feed's control entry declares a size that cannot be a real file")
    return ControlPackage(version=version, platform=platform, url=_need(item, "url", str, where), format=fmt,
                          size=size, sha256=_hex(item.get("sha256"), where), installed_bytes=installed,
                          entry=str(item.get("entry") or "strata.py"))


def _drafter_package(item: dict) -> DrafterPackage:
    if not isinstance(item, dict):
        raise FeedError("a drafter entry in the feed is not an object")
    where = f"drafter entry {item.get('name')!r}"
    name = _need(item, "name", str, where)
    if "/" in name or "\\" in name or name.startswith("."):
        raise FeedError(f"the feed's {where} has a name that is not a plain file name")
    size = _need(item, "size", int, where)
    if size <= 0:
        raise FeedError(f"the feed's {where} declares a size that cannot be a real file")
    backends = item.get("backends")
    if not isinstance(backends, list) or not backends or not all(isinstance(b, str) and b for b in backends):
        raise FeedError(f"the feed's {where} needs a list of backends")
    return DrafterPackage(name=name, version=str(item.get("version") or ""), url=_need(item, "url", str, where), size=size,
                          sha256=_hex(item.get("sha256"), where), backends=tuple(backends), base=str(item.get("base") or ""),
                          licence=str(item.get("licence") or ""), notes=str(item.get("notes") or ""))


def _model_text(item: dict, key: str, where: str, *, limit: int = 400, required: bool = True) -> str:
    value = item.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit or any(ord(c) < 32 for c in value):
        raise FeedError(f"{where} has no valid {key}")
    return value.strip()


def _model_file(item, where: str) -> ModelFile:
    if not isinstance(item, dict):
        raise FeedError(f"{where} lists a file that is not an object")
    name = _need(item, "name", str, where)
    if "/" in name or "\\" in name or name.startswith(".") or not name.lower().endswith(".gguf"):
        raise FeedError(f"{where} has a file name that is not a plain .gguf name")
    size = _need(item, "size", int, where)
    if size <= 0:
        raise FeedError(f"{where} declares a size that cannot be a real file")
    path = item.get("path")
    if path is not None:
        if not isinstance(path, str) or not path or path.startswith(("/", ".")) or "\\" in path or ".." in path.split("/") or path.rsplit("/", 1)[-1] != name:
            raise FeedError(f"{where} has a file path that does not end in the file's name")
    return ModelFile(name=name, size=size, sha256=_hex(item.get("sha256"), where), path=path or "")


def _number(value, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high


def _model_entry(item) -> FeedModel:
    """One ``models`` entry, strictly: any fault raises FeedError (the caller drops that entry and goes on with the next)."""
    if not isinstance(item, dict):
        raise FeedError("a model entry in the feed is not an object")
    ident = item.get("id")
    where = f"model entry {ident!r}"
    if not isinstance(ident, str) or not _MODEL_ID.fullmatch(ident) or ident.endswith(("-", ".")):
        raise FeedError(f"the feed's {where} has no valid id")
    family = _need(item, "family", str, where)
    if not _MODEL_FAMILY.fullmatch(family):
        raise FeedError(f"the feed's {where} has no valid family")
    if family.startswith(shape_engine.FAMILY_PREFIX) and shape_engine.shape_of_family(family) is None:
        raise FeedError(f"the feed's {where} names a model shape that is not a valid name ({family[len(shape_engine.FAMILY_PREFIX):]!r})")
    repo = _need(item, "repo", str, where)
    revision = _need(item, "revision", str, where)
    if not _MODEL_REPO.fullmatch(repo) or ".." in repo:
        raise FeedError(f"the feed's {where} has no valid repo")
    if not _MODEL_REVISION.fullmatch(revision):
        raise FeedError(f"the feed's {where} needs the repository's pinned revision (40 hex digits)")
    licence_url = _need(item, "licence_url", str, where)
    if not licence_url.startswith("https://") or any(c.isspace() for c in licence_url):
        raise FeedError(f"the feed's {where} needs an https licence_url")
    drafter = item.get("drafter", "none")
    if drafter not in MODEL_DRAFTERS:
        raise FeedError(f"the feed's {where} has an unknown drafter {drafter!r}")
    context = item.get("context", 0)
    if not isinstance(context, int) or isinstance(context, bool) or context < 0 or context > 4194304:
        raise FeedError(f"the feed's {where} has no valid context")
    min_engine = item.get("min_engine", "")
    if min_engine:
        version_key(min_engine)
    pictures = item.get("pictures", False)
    if not isinstance(pictures, bool):
        raise FeedError(f"the feed's {where} has no valid pictures flag")
    mmproj = _model_file(item["mmproj"], f"the feed's {where}") if item.get("mmproj") is not None else None
    if pictures and mmproj is None:
        raise FeedError(f"the feed's {where} offers pictures but lists no mmproj file")
    raw_quants = item.get("quants")
    if not isinstance(raw_quants, list) or not raw_quants:
        raise FeedError(f"the feed's {where} lists no quants")
    quants, seen_q, seen_f = [], set(), set()
    for q in raw_quants:
        if not isinstance(q, dict):
            raise FeedError(f"the feed's {where} has a quant that is not an object")
        qname = q.get("name")
        if not isinstance(qname, str) or not _MODEL_QUANT.fullmatch(qname) or qname in seen_q:
            raise FeedError(f"the feed's {where} has a quant without a valid, unique name")
        seen_q.add(qname)
        qwhere = f"the feed's {where} quant {qname!r}"
        files = q.get("files")
        if not isinstance(files, list) or not files:
            raise FeedError(f"{qwhere} lists no files")
        parsed = tuple(_model_file(f, qwhere) for f in files)
        if len({f.name for f in parsed}) != len(parsed):
            raise FeedError(f"{qwhere} lists a file twice")
        for f in parsed:
            if f.name in seen_f:
                raise FeedError(f"{qwhere} reuses the file name {f.name!r} of another quant")
            seen_f.add(f.name)
        if not _number(q.get("ram_gb"), 0.001, 4096) or not _number(q.get("arena_gb", 0), 0, 4096):
            raise FeedError(f"{qwhere} has no valid ram_gb or arena_gb")
        quants.append(ModelQuant(name=qname, about=_model_text(q, "about", qwhere, required=False), ram_gb=float(q["ram_gb"]),
                                 arena_gb=float(q.get("arena_gb", 0)), files=parsed))
    if mmproj is not None and mmproj.name in seen_f:
        raise FeedError(f"the feed's {where} uses the mmproj file name for a model file")
    kv = item.get("kv_bytes_per_token", 0)
    drafts = item.get("mtp_drafts", 2 if drafter == "mtp" else 0)
    state = item.get("state_gb", 0)
    if not isinstance(kv, int) or isinstance(kv, bool) or kv < 0 or not isinstance(drafts, int) or isinstance(drafts, bool) or not 0 <= drafts <= 8 \
            or not _number(state, 0, 64):
        raise FeedError(f"the feed's {where} has an invalid kv_bytes_per_token, mtp_drafts or state_gb")
    if family == "qwen35moe" and kv <= 0:
        raise FeedError(f"the feed's {where} (family qwen35moe) needs kv_bytes_per_token")
    here = f"the feed's {where}"
    return FeedModel(id=ident, title=_model_text(item, "title", here, limit=80), by=_model_text(item, "by", here, limit=160),
                     about=_model_text(item, "about", here, limit=400, required=False),
                     licence=_model_text(item, "licence", here, limit=120), licence_url=licence_url, family=family, repo=repo,
                     revision=revision, quants=tuple(quants), mmproj=mmproj, pictures=pictures, drafter=drafter, context=context, min_engine=min_engine,
                     kv_bytes_per_token=kv, mtp_drafts=drafts, state_gb=float(state))


def _models_member(raw) -> tuple[list[FeedModel], list[str]]:
    """The ``models`` member, leniently by entry: (the models that are well-formed, one plain line for each entry left out). A member that is not the
    expected shape is ignored whole, with a line. Nothing here refuses the feed."""
    dropped: list[str] = []
    if raw is None:
        return [], dropped
    if not isinstance(raw, dict) or not isinstance(raw.get("packages"), list):
        dropped.append("The update feed's models member is not a list of models: ignored.")
        _log.warning(dropped[-1])
        return [], dropped
    out: list[FeedModel] = []
    for item in raw["packages"]:
        try:
            model = _model_entry(item)
        except FeedError as exc:
            label = item.get("id") if isinstance(item, dict) else None
            dropped.append(f"The update feed lists a model ({label!r}) that is not well-formed, so it is skipped: {exc}")
            _log.warning(dropped[-1])
            continue
        if any(m.id == model.id for m in out):
            dropped.append(f"The update feed lists the model {model.id!r} twice: the second one is skipped.")
            _log.warning(dropped[-1])
            continue
        out.append(model)
    return out, dropped


def parse(payload: bytes, *, source: str, key_id: str) -> Feed:
    """The feed document, strictly. Unknown members are ignored; missing or mistyped ones are errors."""
    try:
        data = json.loads(payload.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        raise FeedError("the update feed is not valid JSON") from None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise FeedError("the update feed has an unsupported schema_version")
    sequence = data.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise FeedError("the update feed has no sequence number")
    channel = _need(data, "channel", str, "header")
    generated = _need(data, "generated_at", str, "header")
    _time(generated, "generated_at")
    expires = data.get("expires_at")
    if expires is not None:
        _time(expires, "expires_at")
    skipped: list = []
    app = None
    raw_app = data.get("app")
    if raw_app is not None:
        if not isinstance(raw_app, dict):
            raise FeedError("the feed's app member is not an object")
        app_version_key(_need(raw_app, "version", str, "app entry"))
        platforms = raw_app.get("platforms")
        if not isinstance(platforms, dict) or not platforms:
            raise FeedError("the feed's app entry lists no platforms")
        parsed_platforms = {}
        by_tauri = {v: k for k, v in TAURI_TARGETS.items()}
        for target, entry in platforms.items():
            plat = by_tauri.get(str(target).split("-nsis")[0].split("-msi")[0])
            if plat is None:
                _skipping(skipped, f"the app for {target}", "a platform this Strata does not know")
                continue
            if not isinstance(entry, dict):
                raise FeedError(f"the feed's app entry for {target!r} is not an object")
            parsed_platforms[plat] = {"url": _need(entry, "url", str, "app entry"),
                                      "signature": _need(entry, "signature", str, "app entry"),
                                      "sha256": _hex(entry.get("sha256"), "app entry"),
                                      "size": _need(entry, "size", int, "app entry"), "target": target}
        app = AppRelease(version=raw_app["version"], notes=str(raw_app.get("notes") or ""),
                         pub_date=str(raw_app.get("pub_date") or ""), platforms=parsed_platforms)
    control = []
    raw_control = data.get("control")
    if raw_control is not None:
        if not isinstance(raw_control, dict):
            raise FeedError("the feed's control member is not an object")
        cver = _need(raw_control, "version", str, "control entry")
        version_key(cver)
        control = [c for c in (_control_package(item, cver, skipped) for item in (raw_control.get("packages") or [])) if c is not None]
    engine = []
    raw_engine = data.get("engine")
    if raw_engine is not None:
        if not isinstance(raw_engine, dict) or not isinstance(raw_engine.get("packages"), list):
            raise FeedError("the feed's engine member lists no packages")
        engine = [e for e in (_engine_package(item, skipped) for item in raw_engine["packages"]) if e is not None]
        seen = set()
        for pkg in engine:
            ident = (pkg.version, pkg.platform, pkg.backend, pkg.variant)
            if ident in seen:
                raise FeedError(f"the feed lists engine {pkg.label} twice")
            seen.add(ident)
    dsv41 = []
    raw_dsv41 = data.get("dsv41")
    if raw_dsv41 is not None:
        if not isinstance(raw_dsv41, dict) or not isinstance(raw_dsv41.get("packages"), list):
            raise FeedError("the feed's dsv41 member lists no packages")
        dsv41 = [e for e in (_engine_package(item, skipped) for item in raw_dsv41["packages"]) if e is not None]
        seen = set()
        for pkg in dsv41:
            ident = (pkg.version, pkg.platform, pkg.backend, pkg.variant)
            if ident in seen:
                raise FeedError(f"the feed lists DeepSeek engine {pkg.label} twice")
            seen.add(ident)
    shapes = []
    raw_shapes = data.get("shape_engines")
    if raw_shapes is not None:
        if not isinstance(raw_shapes, dict) or not isinstance(raw_shapes.get("packages"), list):
            raise FeedError("the feed's shape_engines member lists no packages")
        shapes = [e for e in (_shape_engine_package(item, skipped) for item in raw_shapes["packages"]) if e is not None]
        seen = set()
        for pkg in shapes:
            ident = (pkg.shape, pkg.version, pkg.platform, pkg.backend, pkg.variant)
            if ident in seen:
                raise FeedError(f"the feed lists the engine for {pkg.shape} {pkg.label} twice")
            seen.add(ident)
    drafters = []
    raw_drafters = data.get("drafters")
    if raw_drafters is not None:
        if not isinstance(raw_drafters, dict) or not isinstance(raw_drafters.get("packages"), list):
            raise FeedError("the feed's drafters member lists no packages")
        drafters = [_drafter_package(item) for item in raw_drafters["packages"]]
        if len({d.name for d in drafters}) != len(drafters):
            raise FeedError("the feed lists a drafter twice")
    models, models_dropped = _models_member(data.get("models"))
    return Feed(sequence=sequence, channel=channel, generated_at=generated, expires_at=expires, app=app,
                control=tuple(control), engine=tuple(engine), source=source, key_id=key_id, raw=payload, dsv41=tuple(dsv41),
                skipped=tuple(skipped), drafters=tuple(drafters), models=tuple(models), models_dropped=tuple(models_dropped), shape_engines=tuple(shapes))


def _expired_words(expires_at: str, now: dt.datetime, server_now: dt.datetime | None) -> str:
    """Expired by this PC's clock. If the update server's own clock says otherwise (or says the PC is far off), the clock is what is wrong."""
    fix = "Set the date and time right (Windows: Settings, Time & language, Date & time, Set time automatically) and try again."
    if server_now is not None:
        off = abs((server_now - now).total_seconds())
        if off > 15 * 60 and _time(expires_at, "expires_at") >= server_now:
            return (f"your PC clock is wrong: it says {now:%Y-%m-%d %H:%M} (UTC) but the update server says {server_now:%Y-%m-%d %H:%M}, "
                    f"so the update feed looks expired. {fix}")
    return (f"the update feed expired on {expires_at} by this PC's clock ({now:%Y-%m-%d}). If that is not today's date, your PC clock is wrong. "
            f"{fix} Otherwise the update source needs to publish a new feed; nothing was downloaded.")


def verify_and_parse(payload: bytes, signature_bytes: bytes, *, source: str, trusted: dict[str, bytes],
                     last_sequence: int | None = None, channel: str | None = None,
                     now: dt.datetime | None = None, server_now: dt.datetime | None = None) -> Feed:
    """Check the signature over `payload`, then parse; then the three guards: the channel is the one asked for, the feed has
    not expired, and its sequence is not lower than the last one accepted (a replayed older feed is a downgrade)."""
    if not trusted:
        raise FeedError("this build trusts no update key (update_keys.json is empty and STRATA_UPDATE_TRUSTED_KEYS is not set), "
                        "so no update can be verified; nothing was downloaded")
    try:
        signature = json.loads(signature_bytes.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        raise FeedError("the update feed's signature file is not valid") from None
    try:
        key_id = signing.verify_bytes(trusted, payload, signature)
    except signing.SignatureError as exc:
        raise FeedError(f"the update feed is not trusted: {exc}") from None
    feed = parse(payload, source=source, key_id=key_id)
    if channel and feed.channel != channel:
        raise FeedError(f"the feed is for the {feed.channel!r} channel, this PC follows {channel!r}")
    now = now or dt.datetime.now(dt.timezone.utc)
    if feed.expires_at and _time(feed.expires_at, "expires_at") < now:
        raise FeedError(_expired_words(feed.expires_at, now, server_now))
    if last_sequence is not None and feed.sequence < last_sequence:
        down = FeedDowngrade(f"the update feed is older than the one already accepted (sequence {feed.sequence} < {last_sequence}); "
                             "refusing a replayed feed")
        down.sequence, down.accepted = feed.sequence, last_sequence
        raise down
    return feed


def feed_name(platform: str | None = None) -> str:
    """The feed file this machine reads from a folder or a release address: ``strata-feed-linux-aarch64.json`` on a Linux ARM64 machine, else
    ``strata-feed.json``. A source that names its file (``.../x.json``) is used as given, whatever the machine."""
    if platform is None:
        try:
            platform = host_platform()
        except Exception:                                             # noqa: BLE001 - an unknown OS reads the common feed
            platform = None
    return AARCH64_FEED_NAME if platform == "linux-aarch64" else FEED_NAME


def fetch(source: str, trusted: dict[str, bytes], *, timeout: float = 30, say=None, **guards) -> Feed:
    """Read the feed and its signature from `source` (an https URL, a loopback http URL, a file URL or a folder/file path)."""
    if not trusted:
        raise FeedError("this build trusts no update key (update_keys.json is empty and STRATA_UPDATE_TRUSTED_KEYS is not set), "
                        "so no update can be verified; nothing was downloaded")
    location = source
    if transfer.kind_of(source) == "path":
        path = transfer._local_path(source)
        if path.is_dir():
            location = str(path / feed_name())
            if not (path / feed_name()).is_file() and (path / FEED_NAME).is_file():
                location = str(path / FEED_NAME)                 # a local folder made for the common feed (a test feed, an offline copy)
    elif transfer.kind_of(source) in ("http", "https") and not source.rsplit("/", 1)[-1].endswith(".json"):
        location = source.rstrip("/") + "/" + feed_name()
    try:
        payload = transfer.read_small(location, timeout=timeout, say=say)
        signature = transfer.read_small(location + SIGNATURE_SUFFIX, timeout=timeout, say=say)
    except transfer.Unreachable as exc:
        f = exc.failure                                  # the network rule (W7 and its two companions): the host, and that nothing installed is lost
        key = {"offline": "update_offline", "again": "update_again"}.get(f.kind, "update_no_answer")
        raise FeedUnreachable(netpolicy.WORDS[key].format(host=f.host, seconds=max(1, int(round(f.seconds)))), 6) from None
    except transfer.TransferError as exc:
        raise FeedError(f"the update feed could not be read: {exc}", 6) from None
    guards.setdefault("server_now", transfer.server_date())
    loaded = verify_and_parse(payload, signature, source=location, trusted=trusted, **guards)
    return dataclasses.replace(loaded, signature=signature) if loaded.models else loaded


# ---------------------------------------------------------------------------------------------- hardware and selection

def gfx_covers(archs, present: list[str]) -> bool:
    """HIP: the build holds machine code for at least one of the installed AMD GPUs (``gfx1201``). Unknown GPUs: not a refusal."""
    if not present:
        return True
    return any(a in present for a in archs)


def engine_fit_reasons(pkg: EnginePackage, hardware: dict) -> list[str]:
    """Why `pkg` cannot run on the machine `hardware` describes; empty when it fits. `hardware` is what
    `updater.detect_hardware()` returns: platform, backend, nvidia driver, cuda archs, gfx names, runtime version."""
    from strata_control import engine_component as ec
    reasons = []
    if pkg.platform != hardware["platform"]:
        reasons.append(f"{pkg.label} is for {pkg.platform}, this PC is {hardware['platform']}")
        return reasons
    wanted = hardware.get("backend") or "cpu"
    if pkg.backend != wanted:
        reasons.append(f"{pkg.label} is a {pkg.backend} engine, this PC needs {wanted}")
        return reasons
    runtime = hardware.get("runtime")
    if pkg.min_runtime and runtime and ec._version_key(runtime) < ec._version_key(pkg.min_runtime):
        reasons.append(f"{pkg.label} needs control {pkg.min_runtime} or newer, this is {runtime}")
    cuda_archs = tuple(a for a in pkg.archs if a.startswith(("sm_", "compute_")))
    if pkg.backend == "cuda" and cuda_archs and hardware.get("cuda_archs") and not ec.arch_covers(cuda_archs, hardware["cuda_archs"]):
        reasons.append(f"{pkg.label} has code for {', '.join(cuda_archs)}, this PC's GPU is {', '.join(hardware['cuda_archs'])}")
    if pkg.backend == "hip" and not gfx_covers(pkg.archs, hardware.get("gfx") or []):
        reasons.append(f"{pkg.label} has code for {', '.join(pkg.archs)}, this PC's GPU is {', '.join(hardware['gfx'])}")
    need, have = pkg.min_driver.get("nvidia"), hardware.get("nvidia_driver")
    if pkg.backend == "cuda" and need and have and ec._version_key(have) < ec._version_key(need):
        reasons.append(f"{pkg.label} needs NVIDIA driver {need} or newer, this PC has {have}")
    return reasons


def select_engine(packages, hardware: dict) -> tuple[EnginePackage | None, list[str]]:
    """The best package for this PC and, when none fits, every reason. Best = newest version; with equal versions the build
    that needs the newer driver (CUDA 13 beside CUDA 12), as `engine_component.resolve` chooses between installed ones."""
    from strata_control import engine_component as ec
    fitting, reasons = [], []
    for pkg in packages:
        why = engine_fit_reasons(pkg, hardware)
        if why:
            reasons.extend(why)
        else:
            fitting.append(pkg)
    if not fitting:
        return None, reasons
    fitting = _settle_hip_variants(fitting, hardware, reasons)
    if not fitting:
        return None, reasons
    best = max(fitting, key=lambda p: (version_key(p.version), ec._version_key(p.min_driver.get("nvidia", "0")),
                                       ec.hip_preference(p.archs) if p.backend == "hip" else 1,
                                       ec.arch_native(p.archs, hardware.get("cuda_archs") or []) if p.backend == "cuda" else False))
    kept = str(hardware.get("installed_variant") or "")
    if best.backend == "cuda" and kept and best.variant != kept:
        # An update never switches the CUDA build (#3835): a model on the CUDA 12 engine (its card is one CUDA 13 has no code for) must not
        # be given CUDA 13 because ANOTHER card in the PC fits it. The same version of the build that runs here is kept when the feed has it.
        same = [p for p in fitting if p.backend == "cuda" and p.variant == kept and version_key(p.version) == version_key(best.version)]
        if same:
            reasons.append(f"the installed {kept} build is kept; {best.variant} would also fit another card in this PC")
            best = same[0]
    return best, reasons


def _hip_variants(fitting) -> set[str]:
    return {p.variant for p in fitting if p.backend == "hip"}


def _settle_hip_variants(fitting: list, hardware: dict, reasons: list[str]) -> list:
    """Several HIP builds "fit" only when this PC's gfx is unknown (an empty list fits everything). Version must never decide
    between them: the newest build of the wrong card family would be refused at its test start. Order of evidence: the variant asked
    for, the GPU's name, the variant already installed (an update never switches card family), then give up with a plain reason."""
    if hardware.get("backend") != "hip" or hardware.get("gfx"):
        return fitting
    hip = [p for p in fitting if p.backend == "hip"]
    if len({p.variant for p in hip}) < 2:
        return fitting
    from strata_control import engine_component as ec
    chosen, why = [], ""
    wanted = str(hardware.get("variant_wanted") or "").lower()
    if wanted:
        chosen = [p for p in hip if p.variant.lower() == wanted]
        why = f"the {wanted} build was asked for"
    if not chosen:
        guesses = {g for g in (ec.gfx_from_name(n) for n in hardware.get("gpu_names") or []) if g}
        chosen = [p for p in hip if any(a in guesses for a in p.archs)]
        why = f"the GPU name ({', '.join(hardware.get('gpu_names') or [])}) is a {', '.join(sorted(guesses))}" if chosen else ""
    if not chosen and hardware.get("installed_variant"):
        chosen = [p for p in hip if p.variant == hardware["installed_variant"]]
        why = f"the installed {hardware['installed_variant']} build is kept, this PC's AMD card could not be identified"
    if not chosen:
        reasons.append("Strata cannot tell which AMD build this PC needs (" + ", ".join(sorted(_hip_variants(hip))) +
                       "); the card was not recognised, so nothing was chosen. Name the build with STRATA_ENGINE_BACKEND=hip-<build>")
        return [p for p in fitting if p.backend != "hip"]
    reasons.append(f"several AMD builds fit; chose {chosen[0].variant}: {why}")
    return [p for p in fitting if p.backend != "hip" or p in chosen]
