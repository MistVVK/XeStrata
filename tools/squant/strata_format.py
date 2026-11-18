"""Native .strata v1.0 binary container reader/writer for Hy4-Flash.

Implements docs/FORMAT-SPEC.md §2 exactly:
  - Section 0 (0x0000, 4 KiB aligned):
      Fixed 4,096 B header frame:
        0x0000..0x0007 (8 B):   Magic b"STRATA\\x00\\x01" (major=1, minor=0)
        0x0008..0x000B (4 B):   uint32 flags
        0x000C..0x000D (2 B):   uint16 profile (0=tiered, 1=unified, 2=engram_shard, 3=vision_shard)
        0x000E..0x000F (2 B):   uint16 num_sections (6)
        0x0010..0x018F (384 B): 6 x 64 B StrataSectionEntry:
                                  <QQ32sIIQ (file_offset, byte_length, sha256[32], section_id, flags, reserved)
        0x0190..0x01AF (32 B):  default engram_shard_sha256[32]
        0x01B0..0x0FFF:         zero padding
      0x1000.. (4 KiB aligned): ZSTD-compressed JSON manifest (graph params, engram maps, tokenizer, metadata)
  - Section 1 (4 KiB aligned): Contiguous GPU Trunk Arena (ordered tensor blobs + index in manifest)
  - Section 2 (4 KiB aligned): Embedded Hot-Engram Seed Cache (64 MiB: 524,288 rows x 100 B + sorted 12 B index)
  - Section 3 (4 KiB aligned): Rank Map & Expert Directory (20,992 x 32 B StrataExpertEntry)
  - Section 4 (4 KiB aligned): Rank-Ordered Expert Body Arena (4 KiB-aligned per expert)
  - Section 5 (4 KiB aligned): Golden Self-Check Vectors (sq.selfcheck)
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zstandard as zstd

MAGIC = b"STRATA\x00\x01"
PAGE_SIZE = 4096
NUM_SECTIONS = 6

PROFILE_TIERED = 0
PROFILE_UNIFIED = 1
PROFILE_ENGRAM_SHARD = 2
PROFILE_VISION_SHARD = 3

SECTION_ENTRY_FMT = "<QQ32sIIQ"
SECTION_ENTRY_SIZE = struct.calcsize(SECTION_ENTRY_FMT)
assert SECTION_ENTRY_SIZE == 64

EXPERT_ENTRY_FMT = "<HBBHHQIIII"
EXPERT_ENTRY_SIZE = struct.calcsize(EXPERT_ENTRY_FMT)
assert EXPERT_ENTRY_SIZE == 32


def align4k(n: int) -> int:
    return ((n + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE


@dataclass
class StrataSectionEntry:
    file_offset: int
    byte_length: int
    sha256: bytes
    section_id: int
    flags: int = 0
    reserved: int = 0

    def pack(self) -> bytes:
        assert len(self.sha256) == 32
        return struct.pack(
            SECTION_ENTRY_FMT,
            self.file_offset,
            self.byte_length,
            self.sha256,
            self.section_id,
            self.flags,
            self.reserved,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "StrataSectionEntry":
        fo, bl, h, sid, fl, res = struct.unpack(SECTION_ENTRY_FMT, buf)
        return cls(fo, bl, h, sid, fl, res)


@dataclass
class StrataExpertEntry:
    expert_id: int
    gu_code_id: int
    dn_code_id: int
    cluster_id: int
    flags: int
    body_offset: int
    body_bytes: int
    dn_offset: int
    aux_offset_0: int = 0
    aux_offset_1: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            EXPERT_ENTRY_FMT,
            self.expert_id,
            self.gu_code_id,
            self.dn_code_id,
            self.cluster_id,
            self.flags,
            self.body_offset,
            self.body_bytes,
            self.dn_offset,
            self.aux_offset_0,
            self.aux_offset_1,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "StrataExpertEntry":
        vals = struct.unpack(EXPERT_ENTRY_FMT, buf)
        return cls(*vals)


def build_header_frame(
    profile: int,
    sections: List[StrataSectionEntry],
    engram_shard_sha256: bytes = b"\x00" * 32,
    flags: int = 0,
) -> bytes:
    assert len(sections) == NUM_SECTIONS
    assert len(engram_shard_sha256) == 32
    frame = bytearray(PAGE_SIZE)
    frame[0:8] = MAGIC
    struct.pack_into("<IHH", frame, 8, flags, profile, NUM_SECTIONS)
    pos = 0x10
    for sec in sections:
        frame[pos : pos + SECTION_ENTRY_SIZE] = sec.pack()
        pos += SECTION_ENTRY_SIZE
    assert pos == 0x190
    frame[0x190:0x1B0] = engram_shard_sha256
    return bytes(frame)


def parse_header_frame(frame: bytes) -> Tuple[int, int, List[StrataSectionEntry], bytes]:
    assert len(frame) >= PAGE_SIZE
    if frame[0:8] != MAGIC:
        raise ValueError(f"Bad .strata magic: {frame[0:8]!r}")
    flags, profile, num_sec = struct.unpack_from("<IHH", frame, 8)
    assert num_sec == NUM_SECTIONS, f"Unexpected num_sections={num_sec}"
    sections = []
    pos = 0x10
    for _ in range(NUM_SECTIONS):
        sections.append(StrataSectionEntry.unpack(frame[pos : pos + SECTION_ENTRY_SIZE]))
        pos += SECTION_ENTRY_SIZE
    engram_sha = bytes(frame[0x190:0x1B0])
    return flags, profile, sections, engram_sha


def pack_selfcheck_section(code_pairs: List[Tuple[str, str]], seed: int = 20261118) -> bytes:
    """Build Section 5 golden selfcheck payload verifying decode & Q8_1 dot products across C++/Python."""
    from .codecs import CODE_IDS, SUPERBLOCK_BYTES, decode_sk, dot_sk_q8_1, encode_sk, fwht64_np

    rng = np.random.default_rng(seed)
    records = []
    blob = bytearray()
    for gu_name, dn_name in code_pairs:
        for code_name in sorted(set([gu_name, dn_name])):
            if any(r["code"] == code_name for r in records):
                continue
            w = (rng.standard_normal(1536) * 0.05).astype(np.float32)
            x = rng.standard_normal(1536).astype(np.float32)
            x_h = fwht64_np(x)
            enc = encode_sk(w, code_name)
            dec = decode_sk(enc, code_name)
            dot_q = dot_sk_q8_1(enc, code_name, x)
            dot_qh = dot_sk_q8_1(enc, code_name, x_h)
            enc_off = len(blob)
            blob.extend(enc)
            x_off = len(blob)
            blob.extend(x.tobytes())
            dec_off = len(blob)
            blob.extend(dec.astype(np.float16).tobytes())
            records.append(
                {
                    "code": code_name,
                    "code_id": CODE_IDS[code_name],
                    "sb_bytes": SUPERBLOCK_BYTES[code_name],
                    "n_elems": 1536,
                    "enc_offset": enc_off,
                    "enc_bytes": len(enc),
                    "x_offset": x_off,
                    "dec_f16_offset": dec_off,
                    "dot_q8_1": dot_q,
                    "dot_q8_1_hadamard": dot_qh,
                }
            )
    header_json = json.dumps({"version": 1, "seed": seed, "records": records}).encode("utf-8")
    out = struct.pack("<I", len(header_json)) + header_json + bytes(blob)
    return out


class StrataWriter:
    """Streaming writer for .strata v1.0 files (supports 100+ GB files with constant RAM)."""

    def __init__(self, path: str, profile: int = PROFILE_TIERED, engram_shard_sha256: bytes = b"\x00" * 32):
        self.path = path
        self.profile = profile
        self.engram_shard_sha256 = engram_shard_sha256
        self.f = open(path, "wb")
        # Reserve 4 KiB header frame at offset 0
        self.f.write(b"\x00" * PAGE_SIZE)
        self.sections: Dict[int, StrataSectionEntry] = {}

    def _pad_4k(self) -> int:
        pos = self.f.tell()
        rem = pos % PAGE_SIZE
        if rem:
            self.f.write(b"\x00" * (PAGE_SIZE - rem))
            pos = self.f.tell()
        return pos

    def write_section_bytes(self, section_id: int, data: bytes, flags: int = 0) -> StrataSectionEntry:
        offset = self._pad_4k()
        h = hashlib.sha256(data).digest()
        self.f.write(data)
        entry = StrataSectionEntry(offset, len(data), h, section_id, flags)
        self.sections[section_id] = entry
        return entry

    def begin_streaming_section(self, section_id: int, flags: int = 0) -> "_StreamingSection":
        offset = self._pad_4k()
        return _StreamingSection(self, section_id, offset, flags)

    def write_manifest(self, manifest: Dict[str, Any]) -> StrataSectionEntry:
        raw = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        cctx = zstd.ZstdCompressor(level=6)
        comp = cctx.compress(raw)
        return self.write_section_bytes(0, comp)

    def finalize(self) -> None:
        self._pad_4k()
        sec_list = []
        for sid in range(NUM_SECTIONS):
            if sid in self.sections:
                sec_list.append(self.sections[sid])
            else:
                sec_list.append(StrataSectionEntry(0, 0, hashlib.sha256(b"").digest(), sid, 0))
        frame = build_header_frame(self.profile, sec_list, self.engram_shard_sha256)
        self.f.seek(0)
        self.f.write(frame)
        self.f.flush()
        self.f.close()


class _StreamingSection:
    def __init__(self, writer: StrataWriter, section_id: int, offset: int, flags: int):
        self.writer = writer
        self.section_id = section_id
        self.offset = offset
        self.flags = flags
        self.hasher = hashlib.sha256()
        self.length = 0

    def write(self, data: bytes | memoryview) -> int:
        rel_off = self.length
        self.writer.f.write(data)
        self.hasher.update(data)
        self.length += len(data)
        return rel_off

    def pad_4k(self) -> int:
        rem = self.length % PAGE_SIZE
        if rem:
            pad = b"\x00" * (PAGE_SIZE - rem)
            self.write(pad)
        return self.length

    def finish(self) -> StrataSectionEntry:
        entry = StrataSectionEntry(self.offset, self.length, self.hasher.digest(), self.section_id, self.flags)
        self.writer.sections[self.section_id] = entry
        return entry


def verify_strata_file(path: str, verify_hashes: bool = True) -> Dict[str, Any]:
    """Verify a .strata v1.0 file's header, 4 KiB section alignment, SHA-256 digests, and manifest."""
    with open(path, "rb") as f:
        frame = f.read(PAGE_SIZE)
        flags, profile, sections, engram_sha = parse_header_frame(frame)
        for sec in sections:
            if sec.byte_length == 0:
                continue
            assert sec.file_offset % PAGE_SIZE == 0, f"Section {sec.section_id} not 4K aligned: {sec.file_offset}"
            if verify_hashes:
                f.seek(sec.file_offset)
                h = hashlib.sha256()
                rem = sec.byte_length
                while rem > 0:
                    chunk = f.read(min(rem, 16 * 1024 * 1024))
                    if not chunk:
                        raise EOFError(f"Truncated section {sec.section_id}")
                    h.update(chunk)
                    rem -= len(chunk)
                if h.digest() != sec.sha256:
                    raise ValueError(f"SHA-256 mismatch in section {sec.section_id}")
        s0 = sections[0]
        f.seek(s0.file_offset)
        manifest_comp = f.read(s0.byte_length)
        dctx = zstd.ZstdDecompressor()
        manifest = json.loads(dctx.decompress(manifest_comp).decode("utf-8"))
    return {
        "flags": flags,
        "profile": profile,
        "engram_shard_sha256": engram_sha.hex(),
        "sections": [
            {"id": s.section_id, "offset": s.file_offset, "bytes": s.byte_length, "sha256": s.sha256.hex()[:16]}
            for s in sections
        ],
        "manifest": manifest,
    }
