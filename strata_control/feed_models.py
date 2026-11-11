"""The model list in the signed update feed (feed member ``models``): the cache of the last verified list, and how a feed model becomes a catalog entry.

A new model can ship as a signed feed update, with no new Control package. The feed carries, for each model, the Hugging Face repository at a pinned revision,
every file's exact size and SHA-256, the engine family, the licence and the first engine version that reads it (strata_control/feed.py ``FeedModel``). Setup, the
Terminal's `strata launch` and the app read the SAME tables (setup.py's MODELS / FAMILIES / OWN_PATH_FAMILIES), so a feed model is added to those tables here and
every screen shows it after the built-in ones. It then goes through the code a built-in model goes through: the same pins (REPO_PINS / REV_PINS, HF_REVISIONS), the
same SHA-256 check after the download and the same ``.bad`` handling (setup.py ``download`` / ``verify_sha256`` / ``keep_bad``).

Rules:

* With no ``models`` member in the feed (and no cached one) nothing here changes anything: no table is touched, nothing is written.
* A model whose family the installed engine does not run (``strata --families``) is listed as not available on this engine and is never offered to install.
* Offline, the models of the last feed that verified are shown (the signature is checked again when the cache is read).
* A model that would clash with a built-in one (an id, a tag, a file name pinned to other bytes) is left out with a log line; a built-in model is never replaced.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from strata_control import feed as feedmod, shape_engine

_log = logging.getLogger("strata.feed")

CACHE_NAME = "feed-models.json"
ENV_OFF = "STRATA_FEED_MODELS"             # "0": setup and the screens do not read or fetch the feed's model list (tests, a locked-down PC)
FETCH_TIMEOUT = 5.0

# ---- the words a person can see (agreed with the coordinator before they ship)
UNAVAILABLE_ENGINE = "needs a newer Strata engine: update Strata to use it"
UNAVAILABLE_UNKNOWN = "this Strata version doesn't know this model yet: update Strata to use it"


#: What happens to a model whose family the installed engine does not run (the owner decides; a one-line switch):
#:   "not_offered": it is listed as not available and never offered to install;
#: (owner, 11-06: "warn_allow", his standing rule: warn plainly, still allow the download, never refuse)
#:   "warn_allow":  it is offered like any other model with the same sentence as a warning, and the install goes on (a family this Control does not know is still not offered).
UNAVAILABLE_POLICY = "warn_allow"


def unavailable_reason(family: str) -> str:
    """The sentence a person sees. The family name is not part of it (support finds it in the log line)."""
    return UNAVAILABLE_ENGINE if family in feedmod.MODEL_FAMILIES_INSTALLABLE else UNAVAILABLE_UNKNOWN


# ---------------------------------------------------------------------------------------------------- the cache of the last verified list

def cache_file(root: str | os.PathLike | None = None) -> Path:
    """``<components>/.updates/feed-models.json``, beside the updater's state (STRATA_COMPONENTS_DIR moves both)."""
    from strata_control import engine_component, updater
    return updater.updates_dir(Path(root) if root else engine_component.components_dir()) / CACHE_NAME


def remember(feed: feedmod.Feed, root: str | os.PathLike | None = None) -> None:
    """Keep the signed feed document that carried a model list, so the list can be shown offline. A feed without models removes an older cache (the list was withdrawn);
    when there is no cache and no models, nothing is written. Never raises: a read-only folder just means no offline list."""
    try:
        path = cache_file(root)
        if not feed.models:
            if path.is_file():
                path.unlink()
            return
        if not feed.signature:
            return
        doc = {"schema_version": 1, "sequence": feed.sequence, "channel": feed.channel, "source": feed.source,
               "payload": base64.b64encode(feed.raw).decode("ascii"), "signature": base64.b64encode(feed.signature).decode("ascii")}
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(prefix=".feed-models-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(doc, stream)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    except OSError as exc:
        _log.warning("the model list of the update feed could not be kept for offline use: %s", exc)


def _read_cache(root, trusted: dict) -> feedmod.Feed | None:
    """The cached feed, its signature checked again against the trusted keys (a cache file edited by hand is worth nothing). Expiry is not applied: the list of an old
    feed is what an offline PC has. None when there is no cache or it does not verify."""
    from strata_control import signing
    try:
        doc = json.loads(cache_file(root).read_text(encoding="utf-8"))
        payload = base64.b64decode(doc["payload"], validate=True)
        signature = base64.b64decode(doc["signature"], validate=True)
        key_id = signing.verify_bytes(trusted, payload, json.loads(signature.decode("utf-8-sig")))
        return feedmod.parse(payload, source="cache", key_id=key_id)
    except (OSError, ValueError, KeyError, TypeError, signing.SignatureError, feedmod.FeedError):
        return None


@dataclass
class Loaded:
    models: tuple = ()
    dropped: tuple = ()
    via: str = "none"                   # "feed" (read now), "cache" (the last verified list) or "none"
    sequence: int | None = None
    notes: list = field(default_factory=list)


def load(source: str | None = None, *, fetch: bool = True, root=None, trusted: dict | None = None, timeout: float = FETCH_TIMEOUT,
         say=None) -> Loaded:
    """The feed's models: the feed itself when it can be read now (and verifies, and is not older than the cached one), else the cache. Silent either way:
    nothing is printed and no error is raised. ``fetch=False`` reads the cache only (the Terminal and app lists)."""
    if os.environ.get(ENV_OFF, "").strip() == "0":
        return Loaded()
    from strata_control import signing
    try:
        trusted = trusted if trusted is not None else signing.load_trusted_keys()
    except Exception:                                    # noqa: BLE001 - a build without keys has no verified list
        return Loaded()
    if not trusted:
        return Loaded()
    cached = _read_cache(root, trusted)
    if fetch and source:
        try:
            got = feedmod.fetch(source, trusted, timeout=timeout, channel="stable", last_sequence=cached.sequence if cached else None,
                                say=say)
        except feedmod.FeedError as exc:                 # offline, expired, a downgrade, a bad signature: the last verified list stays
            _log.info("the update feed's model list could not be refreshed: %s", exc)
        except Exception as exc:                         # noqa: BLE001
            _log.info("the update feed's model list could not be refreshed: %s", exc)
        else:
            remember(got, root)
            return Loaded(models=got.models, dropped=got.models_dropped, via="feed", sequence=got.sequence)
    if cached is not None and cached.models:
        return Loaded(models=cached.models, dropped=cached.models_dropped, via="cache", sequence=cached.sequence)
    return Loaded()


# ---------------------------------------------------------------------------------------------------- feed model -> catalog entry

class _ByQuant(str):
    """A catalog string that depends on the size: ``fam["file"].format(q=quant)`` and ``fam["hf"].format(q=quant)`` are how the built-in code asks for a
    file name or a download folder; the feed names each size's files explicitly, so the answer comes from a table."""

    def __new__(cls, table: dict):
        obj = super().__new__(cls, next(iter(table.values()), ""))
        obj.table = dict(table)
        return obj

    def format(self, *args, **kw):                       # noqa: A003 - the override is the point
        return self.table[kw["q"]]


def _template(names: list[str]) -> str | None:
    """The ``file`` template (``{i}`` the shard number, 1 to 9) that gives exactly these names in order, or None."""
    if len(names) == 1:
        return names[0] if not re.search(r"[{}]", names[0]) else None
    if len(names) > 9 or any(re.search(r"[{}]", n) for n in names):
        return None
    first = names[0]
    if "-00001-of-" not in first:
        return None
    tpl = first.replace("-00001-of-", "-0000{i}-of-")
    return tpl if [tpl.format(i=i) for i in range(1, len(names) + 1)] == names else None


def _engine_tuple(version: str) -> tuple | None:
    """setup.py's engine versions are 3-tuples (0, 1, 57); a fourth number only when the feed's version has one."""
    if not version:
        return None
    key = feedmod.version_key(version)
    return key if key[3] else key[:3]


def _endpoint(setup) -> str:
    """The model host setup downloads from (a mirror when the person set one), read off setup's own url builder so this module names no host."""
    repo = next(iter(setup.HF_REVISIONS))
    return setup.hf(repo).split(f"/{repo}/resolve/")[0]


def _urlbase(repo: str, revision: str, endpoint: str) -> str:
    return f"{endpoint}/{repo}/resolve/{revision}/"


@dataclass
class Applied:
    registered: list = field(default_factory=list)       # ids now in setup's tables
    unavailable: dict = field(default_factory=dict)      # id -> {"title", "by", "family", "reason"}
    skipped: list = field(default_factory=list)          # plain lines, one for each model left out (also logged)


def _skip(result: Applied, model, why: str) -> None:
    line = f"The update feed's model {model.id!r} is left out: {why}"
    result.skipped.append(line)
    _log.warning(line)


def _taken_tags(setup) -> list[str]:
    tags = [d.get("tag", "") for d in (*setup.FAMILIES.values(), *setup.OWN_PATH_FAMILIES.values(), *setup.MULTI_FILE_FAMILIES.values())]
    return [t for t in tags if t]


def apply(setup, models, engine_families=None) -> Applied:
    """Add the feed's models to setup's tables (``setup`` is the setup.py module). ``engine_families``: the set ``strata --families`` printed for the installed
    engine, or None when no engine is installed to ask (then every installable family is offered and setup checks the engine when it installs, as for any model).
    Safe to call again with the same list: a model already added is skipped quietly. Built-in entries are never replaced."""
    result = Applied()
    if not models:
        return result
    done_before = getattr(setup, "FEED_MODELS", None)
    if done_before is None:
        done_before = setup.FEED_MODELS = {}
        setup.FEED_UNAVAILABLE = {}
    for model in models:
        if model.id in done_before:
            if model.id not in setup.FEED_UNAVAILABLE:
                result.registered.append(model.id)
            continue
        if model.id in setup.FAMILIES or model.id in setup.OWN_PATH_FAMILIES or model.id in setup.MULTI_FILE_FAMILIES:
            _skip(result, model, "its id is the name of a built-in model")
            continue
        tag = model.id + "-"
        if any(t.startswith(tag) or tag.startswith(t) for t in _taken_tags(setup)):
            _skip(result, model, "its name would be mistaken for a built-in model's files")
            continue
        shape = model.shape
        known = model.family in feedmod.MODEL_FAMILIES_INSTALLABLE or shape is not None
        # a model of its own shape runs on that shape's engine, which setup fetches when the model is chosen (strata_control/shape_engine.py): the Strata engine's families do not decide
        runs = known and (shape is not None or engine_families is None or model.family in engine_families)
        if not runs:
            allowed = known and UNAVAILABLE_POLICY == "warn_allow"
            reason = unavailable_reason(model.family)
            _log.info("feed model %s: family %s is not run by the installed engine (%s); policy %s", model.id, model.family,
                      "unknown to this Control" if not known else "engine families " + ",".join(sorted(engine_families or ())), UNAVAILABLE_POLICY)
            setup.FEED_UNAVAILABLE[model.id] = {"title": model.title, "by": model.by, "family": model.family, "reason": reason, "allowed": allowed}
            result.unavailable[model.id] = setup.FEED_UNAVAILABLE[model.id]
            if not allowed:
                done_before[model.id] = model
                continue
        try:
            (_own_path if model.family == "qwen35moe" else _flash_next)(setup, model)
        except _Unsupported as exc:
            _skip(result, model, str(exc))
            continue
        done_before[model.id] = model
        result.registered.append(model.id)
    return result


class _Unsupported(Exception):
    pass


def _check_pins(setup, model) -> dict:
    """The model's files as setup pins them: name -> (size, sha256). A file name that setup already pins to other bytes (the picture reader is one file shared by
    several models) refuses the model rather than replace the pin."""
    pins: dict = {}
    for quant in model.quants:
        for f in quant.files:
            pins[f.name] = (f.size, f.sha256)
    if model.mmproj is not None:
        pins[model.mmproj.name] = (model.mmproj.size, model.mmproj.sha256)
        for repo_pins in setup.REPO_PINS.values():
            known = repo_pins.get(model.mmproj.name)
            if known and tuple(known) != pins[model.mmproj.name]:
                raise _Unsupported("its picture reader file has the name of another model's, with other bytes")
    return pins


def _pin_repo(setup, model, pins: dict) -> str:
    """Register the revision and the file pins where setup's download looks them up (file_pin: REV_PINS first), and return the download folder."""
    setup.REV_PINS.setdefault((model.repo, model.revision), {}).update(pins)
    setup.HF_REVISIONS.setdefault(model.repo, model.revision)
    return _urlbase(model.repo, model.revision, _endpoint(setup))


def _licence(model) -> str:
    return f"{model.licence}: {model.licence_url}"


def _dirs(model) -> set[str]:
    return {f.repo_path.rpartition("/")[0] for q in model.quants for f in q.files}


def _flash_next(setup, model) -> None:
    """A qwen4exp model: the family and size tables the Unsloth sets use (GGUF shards in the repository, packed on this PC), one MODELS entry for each size."""
    template = setup.FAMILIES.get("unsloth") or {}
    names_by_quant = {q.name: [f.name for f in q.files] for q in model.quants}
    files = {}
    for qname, names in names_by_quant.items():
        tpl = _template(names)
        if tpl is None:
            raise _Unsupported(f"the files of {qname} are not a single file or numbered parts 00001-of-0000N")
        files[qname] = tpl
    pins = _check_pins(setup, model)
    base = _pin_repo(setup, model, pins)
    folders = {q.name: (q.files[0].repo_path.rpartition("/")[0] + "/" if q.files[0].repo_path.rpartition("/")[0] else "") for q in model.quants}
    keys = {}
    for q in model.quants:
        key = q.name if q.name not in setup.MODELS and q.name not in keys.values() else f"{q.name}-{model.id}"
        if key in setup.MODELS or key in keys.values():
            raise _Unsupported(f"the size name {q.name} clashes with a built-in size")
        keys[q.name] = key
    fam = {"title": model.title, "by": model.by, "about": model.about, "hf": _ByQuant({q: base + folders[q] for q in files}),
           "file": _ByQuant(files), "shards": 2, "tag": model.id + "-", "name": model.id, "vision": bool(model.pictures),
           "pack_args": list(template.get("pack_args", ["--compat-bf16"])), "sha256": dict(pins), "license": _licence(model), "feed": True,
           "drafter": model.drafter}
    if model.mmproj is not None:
        fam.update(mmproj_hf=base + (model.mmproj.repo_path.rpartition("/")[0] + "/" if "/" in model.mmproj.repo_path else ""),
                   mmproj=model.mmproj.name, mmproj_sha256=(model.mmproj.size, model.mmproj.sha256))
    else:
        fam.update(vision=False)
    engine = _engine_tuple(model.min_engine)
    # nothing is added to the shared tables until every check above has passed
    for q in model.quants:
        gb = round(q.download_bytes / 1e9, 1)
        entry = {"about": q.about or f"{q.name} quantization", "download_gb": gb, "ram_gb": q.ram_gb,
                 "arena_gb": q.arena_gb or round(gb * 0.93, 1), "families": (model.id,), "shards": len(q.files),
                 "file": files[q.name], "quant": q.name, "vision": bool(model.pictures)}
        entry["budget"] = True                           # the Unsloth-set install: the GGUF read in place with a RAM budget of its experts (what the built-in UD sizes do). Without it
                                                         # setup picks the low-RAM mode on a PC with little RAM, which copies all the experts into a second file (77 GB for UD-Q4_K_XL)
        if engine:
            entry["engine"] = engine
        setup.MODELS[keys[q.name]] = entry
    if model.shape:                                      # its own engine: the run config names it, setup fetches it (strata_control/shape_engine.py)
        fam["shape"] = model.shape
        for key in keys.values():                        # its min_engine is about that engine, which the feed carries: the app must not compare it with the Strata engine
            setup.MODELS[key].pop("engine", None)
    setup.FAMILIES[model.id] = fam
    _mac_entries(model, keys)


def _mac_entries(model, keys: dict) -> None:
    """The Mac (Metal) tables: a size of one or two files is offered like the Qwen GSQ-RCO sizes (no repack), "not measured on a Mac". More files: not offered on a Mac."""
    try:
        from strata_control import mac_fit
    except Exception:                                    # noqa: BLE001
        return
    if any(len(q.files) > 2 for q in model.quants):
        return
    for q in model.quants:
        key = keys[q.name]
        mac_fit.MAC_MODELS[key] = {"shard1_gb": round(q.download_bytes / 1e9, 1), "tested": False, "family": model.id, "shards": len(q.files), "no_mtp": model.drafter == "none"}
        mac_fit.MAC_ABOUT[key] = q.about or f"{q.name} quantization"
    if model.id not in mac_fit.FEED_FAMILIES:
        mac_fit.FEED_FAMILIES.append(model.id)


def _own_path(setup, model) -> None:
    """A qwen35moe model: the Qwen3.6-35B-A3B install path (one GGUF, read in place by the engine). One file for each size, all in one folder."""
    if any(len(q.files) != 1 for q in model.quants):
        raise _Unsupported("this install path takes one file for each size")
    if len(_dirs(model)) != 1:
        raise _Unsupported("this install path takes all sizes from one folder of the repository")
    if model.mmproj is not None and model.mmproj.repo_path.rpartition("/")[0] != next(iter(_dirs(model))):
        raise _Unsupported("the picture reader is not in the folder of the model files")
    names = {q.name: q.files[0].name for q in model.quants}
    pins = _check_pins(setup, model)
    base = _pin_repo(setup, model, pins)
    folder = next(iter(_dirs(model)))
    base = base + (folder + "/" if folder else "")
    fam = {"title": model.title, "by": model.by, "arch": "qwen35moe", "about": model.about, "hf": base, "file": _ByQuant(names), "tag": model.id + "-",
           "mmproj_hf": base, "mmproj": model.mmproj.name if model.mmproj else "", "name": model.id, "license": _licence(model),
           "models": {q.name: {"about": q.about or f"{q.name} quantization", "ram_gb": q.ram_gb} for q in model.quants},
           "sha256": dict(pins), "kv_bytes_per_token": model.kv_bytes_per_token, "mtp": model.mtp_drafts,
           "vision": bool(model.pictures), "state_gb": model.state_gb, "feed": True}
    setup.OWN_PATH_FAMILIES[model.id] = fam
