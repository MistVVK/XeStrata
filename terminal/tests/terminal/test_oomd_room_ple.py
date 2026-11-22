"""0.2.17 retest (GB10 in a Vast container, IQ3_S): the container rule of make_room (5281c94b7) dropped the file cache of EVERY file in the engine's arguments before
each start, the PLE shard included. The engine then read the 26.8 GiB table cold from the disk (PLE table loaded in 48-52 s, against 0.0-1.8 s in 0.2.16, which
did not drop): +50 s per start. The shard named by --ple-gguf is left in the cache (unless --ple-io direct); the other files are dropped as before.

    python -m unittest tests.terminal.test_oomd_room_ple
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve import oomd_pacer as op  # noqa: E402

GIB = 1 << 30


def write_gguf(path: Path, tensors: tuple[str, ...] = (), metadata: dict | None = None, version: int = 3) -> Path:
    """Write a minimal valid GGUF v2/v3 header with optional metadata (scalars, strings, arrays) and named tensors."""
    buf = bytearray()
    buf += b"GGUF"
    kv = metadata or {}
    buf += struct.pack("<IQQ", version, len(tensors), len(kv))

    def put_str(s: str) -> None:
        raw = s.encode("utf-8")
        buf.extend(struct.pack("<Q", len(raw)))
        buf.extend(raw)

    for key, val in kv.items():
        put_str(key)
        if isinstance(val, str):
            buf += struct.pack("<I", 8)
            put_str(val)
        elif isinstance(val, int):
            buf += struct.pack("<II", 4, val)
        elif isinstance(val, list) and val and isinstance(val[0], str):
            buf += struct.pack("<IIQ", 9, 8, len(val))
            for item in val:
                put_str(item)
        elif isinstance(val, list):
            buf += struct.pack("<IIQ", 9, 4, len(val))
            for item in val:
                buf += struct.pack("<I", item)

    for i, name in enumerate(tensors):
        put_str(name)
        buf += struct.pack("<IQQIQ", 2, 16, 16, 0, i * 1024)
    path.write_bytes(bytes(buf))
    return path


class PleShardStaysCached(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.shard1 = d / "model-00001-of-00002.gguf"
        self.shard2 = d / "model-00002-of-00002.gguf"
        self.link = d / "elsewhere.gguf"
        for p in (self.shard1, self.shard2):
            p.write_bytes(b"x" * 1000)
        (d / "pack").mkdir()
        (d / "pack" / "t1.bin").write_bytes(b"y" * 500)
        self.dropped, self.said, self.cache = [], [], [40 * GIB]
        patch = mock.patch.object(op, "_DONTNEED", 4)
        patch.start()
        self.addCleanup(patch.stop)

    def run_room(self, args, free=78 * GIB):
        def drop(path):
            self.dropped.append(path.name)
            self.cache[0] = 0
            return True
        with mock.patch.object(op, "applicable", return_value=None), mock.patch.object(op, "in_container", return_value=True):
            return op.make_room(args, self.said.append, {}, drop=drop, cache_of=lambda: self.cache[0], free_of=lambda: free,
                                sizes=lambda p: 84 * GIB // 3)

    def args(self, *extra):
        return ["--pack", str(self.shard1.parent / "pack"), "--native", str(self.shard1), "--ple-gguf", str(self.shard2), *extra]

    def test_the_ple_shard_is_not_dropped_with_ple_io_ram(self):
        # the GB10 figures: 78 GB free is below 1.1 x the model's files, so the rule fires, as in 0.2.17
        freed = self.run_room(self.args("--ple-io", "ram"))
        self.assertGreater(freed, 0)
        self.assertEqual(sorted(self.dropped), ["model-00001-of-00002.gguf", "t1.bin"])
        self.assertNotIn("model-00002-of-00002.gguf", self.dropped)

    def test_the_ple_shard_is_not_dropped_with_ple_io_mmap_or_when_ple_io_is_not_named(self):
        for extra in (("--ple-io", "mmap"), ()):
            self.dropped.clear()
            self.cache[0] = 40 * GIB
            self.run_room(self.args(*extra))
            self.assertNotIn("model-00002-of-00002.gguf", self.dropped, extra)

    def test_ple_io_direct_reads_at_random_and_gains_nothing_from_the_cache_so_it_is_dropped_as_before(self):
        self.run_room(self.args("--ple-io", "direct"))
        self.assertIn("model-00002-of-00002.gguf", self.dropped)

    def test_a_link_to_the_shard_is_the_same_file(self):
        try:
            self.link.symlink_to(self.shard2)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks here")
        args = ["--native", str(self.shard1), "--native", str(self.link), "--ple-gguf", str(self.shard2), "--ple-io", "ram"]
        self.run_room(args)
        self.assertEqual(self.dropped, ["model-00001-of-00002.gguf"])

    def test_a_model_without_a_ple_table_drops_every_file_as_before(self):
        self.run_room(["--pack", str(self.shard1.parent / "pack"), "--native", str(self.shard1), "--native", str(self.shard2)])
        self.assertEqual(sorted(self.dropped), ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf", "t1.bin"])

    def test_ple_table_file(self):
        self.assertEqual(op.ple_table_file(["--ple-gguf", str(self.shard2), "--ple-io", "ram"]), self.shard2)
        self.assertIsNone(op.ple_table_file(["--ple-gguf", str(self.shard2), "--ple-io", "direct"]))
        self.assertIsNone(op.ple_table_file(["--ple-gguf", str(self.shard2), "--no-ple"]))
        self.assertIsNone(op.ple_table_file(["--native", str(self.shard1)]))
        self.assertIsNone(op.ple_table_file(["--ple-gguf", "relative.gguf"]))


class MultiShardNativeAndAutoPle(unittest.TestCase):
    """3+ shard Unsloth GGUFs (setup.py omits --ple-gguf and passes only shard 1 after --native), Swift (PLE in shard 1),
    and single-file GGUFs: make_room and trim_files discover sibling shards and keep the PLE shard cached."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        (self.d / "pack").mkdir()
        (self.d / "pack" / "t1.bin").write_bytes(b"y" * 500)
        # 4-shard layout like Unsloth UD-Q4_K_XL: metadata-only shard 1 (0 tensors, vocab array), experts in 2-3, PLE in 4
        self.s1 = write_gguf(self.d / "q4-00001-of-00004.gguf", tensors=(),
                             metadata={"general.architecture": "qwen4exp", "split.count": 4,
                                       "tokenizer.ggml.tokens": ["a", "b", "c"], "tokenizer.ggml.scores": [1, 2, 3]})
        self.s2 = write_gguf(self.d / "q4-00002-of-00004.gguf", tensors=("blk.0.ffn_up_exps.weight",), metadata={"split.no": 1})
        self.s3 = write_gguf(self.d / "q4-00003-of-00004.gguf", tensors=("blk.24.ffn_up_exps.weight",), metadata={"split.no": 2})
        self.s4 = write_gguf(self.d / "q4-00004-of-00004.gguf", tensors=("per_layer_token_embd.weight",), metadata={"split.no": 3})
        self.dropped, self.said, self.cache = [], [], [40 * GIB]
        patch = mock.patch.object(op, "_DONTNEED", 4)
        patch.start()
        self.addCleanup(patch.stop)

    def run_room(self, args, *, free=78 * GIB, sizes=None):
        self.dropped.clear()
        self.said.clear()
        self.cache[0] = 40 * GIB

        def drop(path):
            self.dropped.append(path.name)
            self.cache[0] = 0
            return True

        with mock.patch.object(op, "applicable", return_value=None), mock.patch.object(op, "in_container", return_value=True):
            return op.make_room(args, self.said.append, {}, drop=drop, cache_of=lambda: self.cache[0], free_of=lambda: free,
                                sizes=sizes or (lambda p: 26 * GIB if p.suffix == ".gguf" else 1 * GIB))

    def test_four_shard_native_without_ple_gguf_drops_expert_shards_and_keeps_ple_shard(self):
        # Shard 1 is tiny metadata-only (1 MB), shards 2-4 are ~35 GB each (~105 GB total): total MUST include shards 2-4
        # so free=78 GB triggers make_room, expert shards 1-3 are dropped, and shard 4 (PLE) stays cached.
        def sizes(p):
            return 1_000_000 if p == self.s1 else (35 * GIB if p.suffix == ".gguf" else 1 * GIB)
        for extra in (("--ple-io", "ram"), ("--ple-io", "mmap"), ()):
            freed = self.run_room(["--pack", str(self.d / "pack"), "--native", str(self.s1), *extra], free=78 * GIB, sizes=sizes)
            self.assertGreater(freed, 0, extra)
            self.assertEqual(sorted(self.dropped),
                             ["q4-00001-of-00004.gguf", "q4-00002-of-00004.gguf", "q4-00003-of-00004.gguf", "t1.bin"], extra)
            self.assertNotIn("q4-00004-of-00004.gguf", self.dropped, extra)

    def test_four_shard_native_drops_ple_shard_with_ple_io_direct_or_no_ple(self):
        for extra in (("--ple-io", "direct"), ("--no-ple",)):
            self.run_room(["--pack", str(self.d / "pack"), "--native", str(self.s1), *extra])
            self.assertIn("q4-00004-of-00004.gguf", self.dropped, extra)

    def test_trim_files_expands_split_shards_and_excludes_the_ple_shard(self):
        args = ["--pack", str(self.d / "pack"), "--native", str(self.s1)]
        self.assertEqual(op.trim_files(args), [self.s1, self.s2, self.s3])
        self.assertEqual(op.trim_files(args + ["--ple-io", "direct"]), [self.s1, self.s2, self.s3, self.s4])

    def test_swift_layout_with_ple_in_shard_1_spares_shard_1_and_trims_shard_2(self):
        sw1 = write_gguf(self.d / "swift-00001-of-00002.gguf", tensors=("per_layer_token_embd.weight", "blk.0.ffn_up_exps.weight"))
        sw2 = write_gguf(self.d / "swift-00002-of-00002.gguf", tensors=("blk.13.ffn_up_exps.weight",))
        args = ["--native", str(sw1), "--ple-io", "mmap"]
        self.assertEqual(op.ple_table_file(args), sw1)
        self.assertEqual(op.trim_files(args), [sw2])
        self.run_room(args, sizes=lambda p: 50 * GIB)
        self.assertEqual(self.dropped, ["swift-00002-of-00002.gguf"])

    def test_explicit_ple_gguf_wins_over_embedded_ple_tensor(self):
        side = write_gguf(self.d / "standalone-ple.gguf", tensors=("per_layer_token_embd.weight",))
        self.assertEqual(op.ple_table_file(["--native", str(self.s1), "--ple-gguf", str(side)]), side)

    def test_single_file_gguf_with_ple_is_kept_unless_direct(self):
        single = write_gguf(self.d / "one.gguf", tensors=("blk.0.ffn_up_exps.weight", "per_layer_token_embd.weight"))
        self.assertEqual(op.ple_table_file(["--native", str(single)]), single)
        self.assertEqual(op.trim_files(["--native", str(single)]), [])
        self.assertEqual(op.trim_files(["--native", str(single), "--ple-io", "direct"]), [single])

    def test_corrupt_or_truncated_gguf_headers_never_raise(self):
        bad_cases = [
            b"",
            b"GGUF" + b"\x00" * 10,                                              # truncated header
            b"GGUF" + struct.pack("<IQQ", 3, 999999999, 0),                      # huge n_tensors
            b"GGUF" + struct.pack("<IQQ", 3, 1, 999999999),                      # huge n_kv
            b"GGUF" + struct.pack("<IQQ", 99, 1, 0),                             # unsupported version
            b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 1) + b"k" + struct.pack("<I", 99),  # unknown type
        ]
        for idx, raw in enumerate(bad_cases):
            p = self.d / f"bad_{idx}.gguf"
            p.write_bytes(raw)
            self.assertFalse(op._gguf_has_tensor(p), idx)
            self.assertIsNone(op.ple_table_file(["--native", str(p)]), idx)

    def test_missing_native_file_does_not_expand_sibling_shards(self):
        missing = self.d / "gone-00001-of-00002.gguf"
        write_gguf(self.d / "gone-00002-of-00002.gguf", tensors=("per_layer_token_embd.weight",))
        self.assertEqual(op._native_shards(["--native", str(missing)]), [])
        self.assertIsNone(op.ple_table_file(["--native", str(missing)]))
        self.assertEqual(op.trim_files(["--native", str(missing)]), [])

    def test_missing_middle_sibling_is_skipped_and_ple_shard_is_still_found(self):
        self.s3.unlink()
        args = ["--native", str(self.s1)]
        self.assertEqual(op.ple_table_file(args), self.s4)
        self.assertEqual(op.trim_files(args), [self.s1, self.s2])


if __name__ == "__main__":
    unittest.main()
