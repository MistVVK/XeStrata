"""The automatic --draft-conf counts a card's expert slots at the pack's own slot size: a native-experts pack states its expert bytes in
native_experts.txt; any other pack keeps IQ3_S's SLOT_MIB."""

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


class PackSlotSize(unittest.TestCase):
    def test_a_native_experts_pack_gives_its_own_slot_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = ud_q4_k_xl_pack(Path(tmp))
            self.assertAlmostEqual(drafting.pack_slot_mib(["--pack", str(pack)]), 77017907200 / (512 * 48) / 2 ** 20, places=6)   # 2.99 MiB

    def test_any_other_pack_keeps_the_iq3_s_size(self):
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
