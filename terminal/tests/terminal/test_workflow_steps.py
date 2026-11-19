"""A workflow's steps are separate list items (#6206): the Metal workflow had its Build step glued to the end of the
Configure step's last line, so the file had one step too few and two `run:` keys in one step.

Plain-text checks (no YAML library is needed): a step opens with `- name:` / `- uses:` at the start of a line, and a
step has at most one `run:` and one `uses:` key.
"""
import re
import unittest
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def files():
    return sorted(WORKFLOWS.glob("*.yml")) if WORKFLOWS.is_dir() else []


class WorkflowSteps(unittest.TestCase):
    def test_no_step_starts_in_the_middle_of_a_line(self):
        for path in files():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                with self.subTest(file=path.name, line=n):
                    self.assertIsNone(re.search(r"\S\s{2,}- (name|uses|run):", line), line)

    def test_a_step_has_one_run_and_one_uses(self):
        for path in files():
            step, keys, indent = 0, {}, None
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                m = re.match(r"^(\s*)- (\w+):", line)
                if m and m.group(2) in ("name", "uses", "run", "id", "if", "env", "with", "shell"):
                    step, keys, indent = n, {}, len(m.group(1)) + 2
                    key = m.group(2)
                else:
                    m = re.match(r"^(\s*)(\w+):", line)
                    if not m or indent is None or len(m.group(1)) != indent:
                        continue
                    key = m.group(2)
                if key in ("run", "uses"):
                    keys[key] = keys.get(key, 0) + 1
                    with self.subTest(file=path.name, step_starts_at=step, key=key):
                        self.assertEqual(keys[key], 1, f"{path.name}:{n}: a second `{key}:` in the step of line {step}")


if __name__ == "__main__":
    unittest.main()
