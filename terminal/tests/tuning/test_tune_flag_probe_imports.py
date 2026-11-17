"""#4652: the tune's call sites ask engine_flag_inventory(wanted=tuning.dimension_flags() | tune_knobs.arm_flags()). Every function of
strata_cli/tune_commands.py that names tune_knobs imports it (the PR's base had the import; this tree's `auto` and `_recheck` did not, a NameError at
the first tune)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "strata_cli" / "tune_commands.py"


class Imports(unittest.TestCase):
    def test_every_function_that_uses_tune_knobs_imports_it(self):
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        module_level = {a.asname or a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        missing = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            uses = any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "tune_knobs" for x in ast.walk(fn))
            local = {a.asname or a.name for x in ast.walk(fn) if isinstance(x, (ast.Import, ast.ImportFrom)) for a in x.names}
            if uses and "tune_knobs" not in local and "tune_knobs" not in module_level:
                missing.append(fn.name)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
