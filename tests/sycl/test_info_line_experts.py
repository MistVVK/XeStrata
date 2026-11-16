"""#5012 (SYCL): the Intel engine's INFO line names experts= beside expert_slots=, so the Monitor shows "cached / all" experts.

experts= is every routed layer-expert pair (n_layers x n_expert, the MTP block not counted: 48 x 512 = 24576 on Qwen3.8-Flash-Next).
The line is printed by sycl/src/program/generate.cpp; this pins its format and the argument that fills the key (the A750 start that
shows it live is in the lane's results). No GPU."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sycl" / "src" / "program" / "generate.cpp"


def info_call() -> tuple[str, list[str]]:
    """(the format string, the argument expressions) of the std::printf("INFO context=..." call."""
    text = SRC.read_text(encoding="utf-8")
    start = text.index('std::printf("INFO context=')
    depth, i = 0, text.index("(", start)
    for j in range(i, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            break
    body = text[i + 1:j]
    fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body.split('\\n"', 1)[0] + '\\n"'))
    args, depth, cur = [], 0, ""
    for ch in body[body.index('\\n"') + 3:]:
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
            continue
        depth += {"(": 1, ")": -1, "[": 1, "]": -1}.get(ch, 0)
        cur += ch
    args.append(cur.strip())
    return fmt, [a for a in args if a]


class InfoLineExperts(unittest.TestCase):
    def test_experts_sits_beside_expert_slots(self):
        fmt, _ = info_call()
        self.assertIn("expert_slots=%lld experts=%lld expert_cache_mib=%lld", fmt)

    def test_experts_is_layers_times_experts(self):
        fmt, args = info_call()
        conv = r"%[-+ #0]*\d*(?:\.\d+)?(?:ll|l|z)?[dsfu]"
        self.assertEqual(len(re.findall(conv, fmt)), len(args), "one argument per conversion")
        at = lambda key: len(re.findall(conv, fmt[:fmt.index(f" {key}=%")]))      # noqa: E731
        self.assertRegex(args[at("experts")], r"g\.n_layers\s*\*\s*\(long long\)\s*g\.n_expert")
        self.assertEqual(args[at("expert_slots")], "(long long) slots_all")


if __name__ == "__main__":
    unittest.main()
