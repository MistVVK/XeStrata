"""P2 (0.2.16): install.ps1 adds its folder to the user PATH without rewriting the rest of it.

Before, it read the PATH expanded and wrote it back as a plain string: an entry like %USERPROFILE%\\AppData\\Local\\Microsoft\\WindowsApps lost its
variable form, the value's type (REG_EXPAND_SZ) was lost and a trailing ";" was dropped. The step is run here against a throw-away key under
HKCU\\Software (never the real Environment key) on Windows PowerShell.
"""
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

PS1 = Path(__file__).resolve().parents[3] / "tools" / "packaging" / "install.ps1"
POWERSHELL = shutil.which("powershell") if sys.platform == "win32" else None
BIN = r"C:\Users\Someone\AppData\Local\Strata\terminal\current\bin"


def step_text() -> str:
    text = PS1.read_text(encoding="utf-8")
    m = re.search(r"# <path-step>\n(.*?)# </path-step>", text, re.S)
    assert m, "the PATH step markers are missing from install.ps1"
    return m.group(1)


class Text(unittest.TestCase):
    def test_install_uses_the_step_and_not_the_flattening_calls(self):
        text = PS1.read_text(encoding="utf-8")
        self.assertIn("Add-ToUserPath $bin", text)
        self.assertNotIn("SetEnvironmentVariable('Path'", text)
        self.assertNotIn("GetEnvironmentVariable('Path'", text)
        step = step_text()
        self.assertIn("DoNotExpandEnvironmentNames", step)
        self.assertIn("GetValueKind", step)
        self.assertIn("SetValue('Path', $new, $kind)", step)

    def test_the_step_is_skipped_with_strata_no_path(self):
        text = PS1.read_text(encoding="utf-8")
        self.assertRegex(text, r"if \(\$env:STRATA_NO_PATH -ne '1'\) \{\s*try \{ \[void\]\(Add-ToUserPath")


@unittest.skipUnless(POWERSHELL, "needs Windows PowerShell")
class Registry(unittest.TestCase):
    def run_case(self, kind, value, cases="$r = Add-ToUserPath $bin $sub"):
        """Set the key's Path (kind None = no value), run the step, return (changed, raw value, kind or None, second-run changed)."""
        sub = r"Software\StrataInstallTest-" + uuid.uuid4().hex[:10]
        quoted = "'" + (value or "").replace("'", "''") + "'"
        seed = "" if kind is None else f"$k.SetValue('Path', {quoted}, [Microsoft.Win32.RegistryValueKind]::{kind})"
        script = f"""
$ErrorActionPreference = 'Stop'
$sub = '{sub}'
$bin = '{BIN}'
{step_text()}
$k = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($sub)
try {{
    {seed}
    $k.Close()
    $first = Add-ToUserPath $bin $sub
    $second = Add-ToUserPath $bin $sub
    $k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($sub)
    $raw = $k.GetValue('Path', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $kind = $k.GetValueKind('Path')
    $k.Close()
    Write-Output ("RESULT|$first|$second|$kind|$raw")
}}
finally {{ [Microsoft.Win32.Registry]::CurrentUser.DeleteSubKeyTree($sub, $false) }}
"""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "case.ps1"
            f.write_text(script, encoding="utf-8")
            done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(f)],
                                  capture_output=True, text=True, timeout=120)
        line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT|")), None)
        self.assertIsNotNone(line, done.stdout + done.stderr)
        _, first, second, got_kind, raw = line.split("|", 4)
        return first, second, got_kind, raw

    def test_an_expand_string_value_keeps_its_variables_its_type_and_its_trailing_semicolon(self):
        old = r"%USERPROFILE%\AppData\Local\Microsoft\WindowsApps;"
        first, second, kind, raw = self.run_case("ExpandString", old)
        self.assertEqual((first, second, kind), ("True", "False", "ExpandString"))
        self.assertEqual(raw, old + BIN)

    def test_other_entries_stay_byte_for_byte(self):
        old = r"C:\Tools;%SystemRoot%\system32;%LOCALAPPDATA%\Programs\Python\Python312\;D:\x y\bin"
        first, second, kind, raw = self.run_case("ExpandString", old)
        self.assertEqual(raw, old + ";" + BIN)
        self.assertEqual(kind, "ExpandString")

    def test_a_plain_string_value_stays_a_plain_string(self):
        old = r"C:\Tools;D:\More"
        first, second, kind, raw = self.run_case("String", old)
        self.assertEqual((first, kind, raw), ("True", "String", old + ";" + BIN))

    def test_no_path_value_is_created_with_only_our_folder(self):
        first, second, kind, raw = self.run_case(None, None)
        self.assertEqual((first, second, raw), ("True", "False", BIN))

    def test_our_folder_already_there_changes_nothing_even_through_a_variable(self):
        old = r"C:\Tools;" + BIN + ";"
        first, second, kind, raw = self.run_case("ExpandString", old)
        self.assertEqual((first, raw), ("False", old))

    def test_an_empty_value_does_not_start_with_a_semicolon(self):
        first, second, kind, raw = self.run_case("ExpandString", "")
        self.assertEqual(raw, BIN)


if __name__ == "__main__":
    unittest.main()
