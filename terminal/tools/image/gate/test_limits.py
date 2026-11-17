"""tools/image/gate/test_limits.py - TEST-ONLY: the gate's limits (limits.py) accept the floors they were set from, refuse a
result just past the limit, and check_limits.py reads compare.py's and stage_cmp.py's files.

    python -m pytest tools/image/gate/test_limits.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import limits as L  # noqa: E402


class Limits(unittest.TestCase):
    def test_every_floor_passes_its_limit(self):
        for name, e in L.LIMITS.items():
            self.assertEqual(L.passes(name, e["floor"]), [], name)

    def test_just_past_the_limit_fails(self):
        for name, e in L.LIMITS.items():
            for key, lim in e["limit"].items():
                if isinstance(lim, bool):
                    self.assertTrue(L.passes(name, {**e["floor"], key: not lim}), (name, key))
                    continue
                past = lim * 1.001 if key.startswith("rel_l2") else lim * 0.999
                self.assertTrue(L.passes(name, {**e["floor"], key: past}), (name, key))

    def test_the_limit_is_not_far_from_the_floor(self):
        # about 1.2x the worst floor: a limit far looser than that would hide a real bug
        for name, e in L.LIMITS.items():
            for key, lim in e["limit"].items():
                fl = e["floor"].get(key)
                if isinstance(lim, bool) or fl is None or name == "vae_across_tiling":
                    continue
                if key.startswith("rel_l2"):
                    self.assertLessEqual(lim / fl, 1.4, (name, key))
                elif key.startswith("psnr"):
                    self.assertLessEqual(fl - lim, 12.0, (name, key))
                else:
                    self.assertLessEqual(abs(fl - lim), 0.02, (name, key))


class CheckLimits(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", "-B", str(HERE / "check_limits.py"), *args],
                              capture_output=True, text=True)

    def test_pictures_and_stage_files(self):
        with tempfile.TemporaryDirectory() as d:
            pic = Path(d) / "cmp.json"
            pic.write_text(json.dumps({"summary": {"pairs": 5, "bytes_equal": 0, "psnr_min": 56.6, "psnr_mean": 57.4,
                                                   "ssim_min": 0.9988, "ssim_mean": 0.999}}))
            self.assertEqual(self.run_cli("vae_same_tiling", str(pic)).returncode, 0)
            pic.write_text(json.dumps({"summary": {"pairs": 5, "bytes_equal": 0, "psnr_min": 44.0, "psnr_mean": 57.4,
                                                   "ssim_min": 0.9988, "ssim_mean": 0.999}}))
            r = self.run_cli("vae_same_tiling", str(pic))
            self.assertEqual(r.returncode, 1, r.stdout)
            st = Path(d) / "stage.json"
            st.write_text(json.dumps({"cond": {"summary": {"n": 8, "identical": 0, "rel_l2_max": 0.0603,
                                                           "cos_min": 0.9982, "token_cos_min": 0.9909}}}))
            self.assertEqual(self.run_cli("text_encoder_same_backend", str(st), "cond").returncode, 0)
            self.assertEqual(self.run_cli("text_encoder_same_backend", str(st), "vel_0").returncode, 2)
            st.write_text(json.dumps({"vel_0": {"summary": {"n": 8, "identical": 0, "rel_l2_max": 0.02, "cos_min": 0.9999}}}))
            self.assertEqual(self.run_cli("velocity", str(st), "vel_0").returncode, 1)


if __name__ == "__main__":
    unittest.main()
