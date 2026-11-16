"""The automatic --draft-conf counts a card's expert slots at the pack's own slot size where it is at least IQ3_S's: a native-experts pack states its expert
bytes in native_experts.txt (the ISTA-DASLab i-quant packs too); a smaller slot (IQ3_XXS) or a pack without the file keeps IQ3_S's SLOT_MIB."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strata_control import drafting

UD_Q4_K_XL_HEAD = ("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down] "
                   "(n_expert 512, total 77017907200; absolute offsets in Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf, "
                   "or in the named shard beside it - per role where the column is gate,up,down)")


def ud_q4_k_xl_pack(root: Path) -> Path:
    """A pack folder whose native_experts.txt has the UD-Q4_K_XL header and one line per layer (48)."""
    pack = root / "unsloth-ud-q4_k_xl"
    pack.mkdir()
    rows = "".join(f"{layer} 12 7 0 3072000 0 0 0 Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf\n" for layer in range(48))
    (pack / "native_experts.txt").write_text(UD_Q4_K_XL_HEAD + "\n" + rows, encoding="utf-8")
    return pack


IQ3_S_TOTAL, IQ3_XXS_TOTAL = 50292326400, 42912972800   # the first lines of the ISTA-DASLab packs' native_experts.txt (n_expert 512)


def ista_pack(root: Path, total: int) -> Path:
    """A pack folder whose native_experts.txt has an ISTA-DASLab header (v3, 48 layers)."""
    pack = root / "ista"
    pack.mkdir()
    rows = "".join(f"{layer} 18 20 0 1 0 0 0\n" for layer in range(48))
    (pack / "native_experts.txt").write_text(
        f"# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert 512, total {total}; "
        "absolute offsets in x.gguf, or in the named shard beside it)\n" + rows, encoding="utf-8")
    return pack


class PackSlotSize(unittest.TestCase):
    def test_a_native_experts_pack_gives_its_own_slot_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = ud_q4_k_xl_pack(Path(tmp))
            self.assertAlmostEqual(drafting.pack_slot_mib(["--pack", str(pack)]), 77017907200 / (512 * 48) / 2 ** 20, places=6)   # 2.99 MiB

    def test_the_ista_i_quant_packs_have_the_file_too(self):
        # the headers tools/iq_pack.py writes for the ISTA-DASLab sets (n_expert 512, 48 layers): IQ3_S at its own 1.952 MiB; IQ3_XXS (1.665 MiB, smaller
        # than SLOT_MIB) keeps SLOT_MIB until a measurement backs the lower steps its own size would give
        with tempfile.TemporaryDirectory() as tmp:
            self.assertAlmostEqual(drafting.pack_slot_mib(["--pack", str(ista_pack(Path(tmp), IQ3_S_TOTAL))]), 1.9516, places=4)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(drafting.pack_slot_mib(["--pack", str(ista_pack(Path(tmp), IQ3_XXS_TOTAL))]), drafting.SLOT_MIB)
        self.assertAlmostEqual(drafting.SLOT_MIB, 1.95, places=2)

    def test_iq3_xxs_keeps_todays_steps(self):
        # every card 8 .. 96 GB at 128K and 262K (int8, --kv-resident 32768): the same slots and the same --draft-conf as with no pack named
        with tempfile.TemporaryDirectory() as tmp:
            pack = str(ista_pack(Path(tmp), IQ3_XXS_TOTAL))
            for ctx in ("131072", "262144"):
                base = ["--max-context", ctx, "--kv", "int8", "--kv-resident", "32768", "--prefill", "auto"]
                for gb in (8, 10, 12, 16, 20, 24, 32, 48, 72, 96):
                    old, new = drafting.expected_slots(gb * 1024, base), drafting.expected_slots(gb * 1024, base + ["--pack", pack])
                    self.assertEqual(new, old, (ctx, gb))
                    self.assertEqual(drafting.conf_for_slots(new), drafting.conf_for_slots(old), (ctx, gb))

    def test_a_pack_without_the_file_keeps_the_iq3_s_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(drafting.pack_slot_mib(["--pack", tmp]), drafting.SLOT_MIB, "a pack folder with no native_experts.txt")
        self.assertEqual(drafting.pack_slot_mib([]), drafting.SLOT_MIB, "no --pack")

    def test_a_16_gb_card_with_ud_q4_k_xl_gets_the_step_its_slots_prescribe(self):
        """Measured start: RTX 4080 SUPER (16,376 MiB), UD-Q4_K_XL, --max-context 262144 --kv int8 --kv-resident 32768 --prefill auto. The engine logs
        "expert cache 2710 slots, 7.90 GiB of VRAM", which CONF_STEPS puts at 0.8; priced at IQ3_S's 1.95 MiB the estimate was 3208 slots, the 0.7 step."""
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--pack", str(ud_q4_k_xl_pack(Path(tmp))), "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768", "--prefill", "auto"]
            slots = drafting.expected_slots(16376, args)
            self.assertLess(slots, 3000)
            self.assertEqual(drafting.conf_for_slots(slots), 0.8)


if __name__ == "__main__":
    unittest.main()
