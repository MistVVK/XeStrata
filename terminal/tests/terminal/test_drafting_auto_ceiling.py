"""#5252, #5129, #5807: `--prefill auto:N` is a ceiling, not a buffer beside the expert slots.

expected_slots() priced `--prefill auto:32768` at 32,768 tokens of prompt buffers (21.6 GiB), so a 32 GB card was estimated at 442 slots and got
--draft-conf 0.8 where its engine holds 8,158 slots (the 0.5 step); a 24 GB card with 5,662 engine slots was estimated at 0.  The engine lends the
buffers from the expert slots and picks its chunk by the card, so an `auto:N` is priced like a plain `auto`; only a number is a chunk the person
asked for.  Measured starts the numbers below come from:
  - RTX 5090 D 32 GB (32,607 MiB), NVFP4 (2.637 MiB a slot), --max-context 262144 --kv int8 --kv-resident 32768 --vram-reserve-mib 2048
    --prefill auto:32768: the engine loads 8,158 experts (21.01 GiB);
  - RTX 3090 24 GB, UD-Q4_K_XL, the same flags without the reserve: "expert cache 5662 slots, 16.52 GiB" (11-19, rented box, engine 0.1.62).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strata_control import drafting


def pack_of(root: Path, total: int, name: str = "pack") -> Path:
    pack = root / name
    pack.mkdir()
    rows = "".join(f"{layer} 12 7 0 3072000 0 0 0\n" for layer in range(48))
    (pack / "native_experts.txt").write_text(f"# strata native experts v4 (n_expert 512, total {total}; x)\n" + rows, encoding="utf-8")
    return pack


UD_TOTAL = 77017907200                       # UD-Q4_K_XL: 2.99 MiB a slot
NVFP4_TOTAL = int(2.637 * 2 ** 20 * 512 * 48)   # 21.01 GiB / 8,158 slots


class AutoCeiling(unittest.TestCase):
    def args(self, pack: Path, prefill: str, reserve: str | None = None) -> list[str]:
        out = ["--pack", str(pack), "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768", "--prefill", prefill]
        return out + (["--vram-reserve-mib", reserve] if reserve else [])

    def test_auto_n_is_priced_like_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = pack_of(Path(tmp), UD_TOTAL)
            for n in ("8192", "16384", "32768", "65536"):
                self.assertEqual(drafting.expected_slots(24576, self.args(pack, f"auto:{n}")), drafting.expected_slots(24576, self.args(pack, "auto")), n)

    def test_a_number_is_still_a_chunk_beside_the_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = pack_of(Path(tmp), UD_TOTAL)
            auto = drafting.expected_slots(24576, self.args(pack, "auto"))
            self.assertLess(drafting.expected_slots(24576, self.args(pack, "8192")), auto)
            self.assertLess(drafting.expected_slots(24576, self.args(pack, "32768")), drafting.expected_slots(24576, self.args(pack, "8192")))

    def test_the_5090_d_nvfp4_start_gets_the_0_5_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(pack_of(Path(tmp), NVFP4_TOTAL), "auto:32768", "2048")
            slots = drafting.expected_slots(32607, args)
            self.assertGreater(slots, 8158 * 0.9)      # the estimate is good to about 10%
            self.assertLess(slots, 8158 * 1.1)
            self.assertEqual(drafting.conf_for_slots(slots), 0.5)

    def test_the_3090_start_is_priced_within_the_steps_reach(self):
        with tempfile.TemporaryDirectory() as tmp:
            slots = drafting.expected_slots(24576, self.args(pack_of(Path(tmp), UD_TOTAL), "auto:32768"))
            self.assertGreater(slots, 3000)            # not 0 (the 0.8 step): its engine holds 5,662
            self.assertEqual(drafting.conf_for_slots(slots), 0.7)


if __name__ == "__main__":
    unittest.main()
