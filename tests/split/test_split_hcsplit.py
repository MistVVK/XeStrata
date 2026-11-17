"""The NVIDIA split's prompt-path defaults (src/split), as pictures of the sources (no GPU).

0.2.10.1 (owner 11-03, "do all 5"):
  * FNX_PF_ROWS: 8192 prompt rows a chunk on NVIDIA (the expert kernels cost ~30 percent less a token); FNX_PF_ROWS=4096 is the old chunk.
  * FNX_PF_HCSPLIT: the hyper-connection read of a chunk runs on one card's rows each. Mode 2 is the default, used only when every card's start-up calibration
    found the cublasLt algorithm (else the read stays on both cards); mode 1 is bit-identical to the unsplit read (whole 4096-row chunks only); 0 turns it off.
  * FNX_PFX_ATTN_I8 on, FNX_PFX_GDN_TC on except for the IQ3_S-type files (routed experts in IQ2_S/IQ3_XXS/IQ3_S/Q2_0), decided in the engine from the file;
    the variables override both ways.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "split" / "src"


def text(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


class HcSplit(unittest.TestCase):
    def test_default_is_on_and_needs_the_calibrated_algorithm(self):
        s = text("prefill.cu")
        self.assertRegex(s, r'p->hcs_mode = \(int\) env_int\("FNX_PF_HCSPLIT", 2\)')
        # the split runs on every card or on none, and only with the calibrated algorithm on every card (either mode)
        self.assertRegex(s, r"all = all && pf_\[c\]->hcs && pf_\[c\]->down_lt;")
        # mode 1 (exact): a pass is split only when every chunk is a whole 4096 rows, the sizes the algorithm was calibrated at
        self.assertRegex(s, r"hcs_mode == 1 && \(!pf_\[0\]->down_lt \|\| \(int\) ck\.tokens\.size\(\) != 4096\)")
        self.assertIn("if (n != 4096 && n != 2048) return false;", s)
        self.assertRegex(s, r'env_int\("FNX_PF_HCSPLIT_MIN", 3\)')   # fewer than 3 chunks in a pass: not split

    def test_nvidia_only(self):
        s = text("prefill.cu")
        i = s.index('p->hcs_mode = (int) env_int("FNX_PF_HCSPLIT"')
        self.assertIn("#if !FNX_HIP", s[i - 200:i])   # the AMD build never splits the read
        self.assertIn("NCARD == 2", s[i:i + 400])
        self.assertIn("CUDA::cublasLt", (ROOT / "src" / "split" / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_chunk_of_8192_rows_on_nvidia_only_and_sized_to_the_free_memory(self):
        e = text("engine.cu")
        i = e.index("        rows = 8192;")
        self.assertIn("#if !FNX_HIP", e[i - 80:i])                       # the AMD build keeps 4096
        self.assertIn('env_int("FNX_PF_ROWS", rows)', e)                  # set by hand: used as given
        # the sizes tried, largest first, with the bytes they take per card (measured): UD-Q4_K_XL on 2 x 48 GB keeps only 6.5 GB and must not take 8192 x 4
        # (the table is kPfPicks[] in pf_pick.hpp since #5496; engine.cu asks it through pf_pick())
        self.assertIn("{{8192, 4, 4148}, {8192, 2, 3028}, {4096, 4, 1673}, {4096, 2, 1112}}", text("pf_pick.hpp"))
        self.assertIn("pf_pick(free_min, margin)", e)
        self.assertIn("pf_pass_pick_ > 0 ? pf_pass_pick_", text("prefill.cu"))

    def test_the_sum_before_a_split_read_is_a_reduce_scatter_only(self):
        s = text("pf_common.cuh")
        self.assertIn("bool own_out = false", s)
        self.assertRegex(s, r"o\.k == PfRingOp::PASS \|\| o\.k == PfRingOp::SIGNALA \|\| o\.k == PfRingOp::WAITA")

    def test_passes_stay_short_while_other_requests_decode(self):
        sf = (ROOT / "src" / "split" / "tools" / "sfnx.cpp").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"e\.set_pass_rows\(any_decoding\(\) \? 8192 : 0\)", sf)), 2)   # solo and batch admissions
        self.assertIn("void set_pass_rows(int rows)", text("engine.hpp"))


class PerFileSwitches(unittest.TestCase):
    def test_attention_int8_is_decided_by_the_files_expert_types(self):
        k, e = text("kernels.cu"), text("engine.cu")
        # IQ3_S-type files: on (the suite shows no loss); the others (UD-Q4_K_XL lost agent-A8 and A10 with it): off; the variable overrides
        self.assertRegex(k, r'i8_env = \(int\) env_int\("FNX_PFX_ATTN_I8", -1\)')
        self.assertRegex(k, r"i8_env >= 0 \? i8_env != 0 : attn_i8_default\(\)")
        self.assertIn("attn_i8_set_default(m_.g.iq_experts());", e)

    def test_gdn_scan_is_decided_by_the_files_expert_types(self):
        k, g, m = text("kernels.cu"), text("gguf.cpp"), text("engine.cu")
        # the rule: IQ-type routed experts (the set convert_small_tensors converts) -> off; the variable overrides
        self.assertRegex(g, r"mixed \|= kv\.second\.type == T_IQ2_S \|\| kv\.second\.type == T_IQ3_XXS \|\| kv\.second\.type == T_IQ3_S \|\| kv\.second\.type == T_Q2_0;")
        self.assertIn("iq_experts_ = mixed;", g)
        self.assertIn("gdn_tc_set_default(!m_.g.iq_experts());", m)
        self.assertRegex(k, r'gdn_tc_env = \(int\) fnx::env_int\("FNX_PFX_GDN_TC", -1\)')
        self.assertRegex(k, r"gdn_tc_env >= 0 \? gdn_tc_env != 0 : g_gdn_tc_default\.load\(\)")
        self.assertIn("g_gdn_tc_default{true}", k)

    def test_the_rule_in_python_terms(self):
        """The same rule as a function, over the type names a GGUF's routed experts carry (what strata_control/split_engine.py reads)."""
        iq = {"IQ2_S", "IQ3_XXS", "IQ3_S", "Q2_0"}

        def gdn_tc(expert_types, env=None):
            if env is not None:
                return env != "0"
            return not (set(expert_types) & iq)

        self.assertFalse(gdn_tc({"IQ4_NL", "IQ2_S", "IQ3_XXS", "IQ3_S", "Q4_0"}))   # the ISTA IQ3_S set
        self.assertTrue(gdn_tc({"Q4_K", "Q5_K", "Q5_1", "Q8_0"}))                    # UD-Q4_K_XL
        self.assertTrue(gdn_tc({"Q8_0"}))
        self.assertTrue(gdn_tc({"IQ3_S"}, env="1"))
        self.assertFalse(gdn_tc({"Q4_K"}, env="0"))

    def test_one_hop_stays_a_screen_switch(self):
        self.assertRegex(text("engine.cu"), r'env_int\("FNX_ONEHOP", 1\) == 2')


if __name__ == "__main__":
    unittest.main()
