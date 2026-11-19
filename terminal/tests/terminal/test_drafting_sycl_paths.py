"""#5395: an Intel (SYCL) run config names the engine's container paths (/work/..., sycl/setup_intel.py), and the host-side check that
the MTP layer is installed looked for /work on the host: every such start dropped --mtp (decode ~30% slower on a B70). The check now
looks through the mount (STRATA_SYCL_ROOT, else the folder above the checkout); the engine's arguments keep the container's paths.
No GPU, no container."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import drafting  # noqa: E402


class SyclContainerPaths(unittest.TestCase):
    def test_the_mtp_layer_under_the_mount_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / "Strata-data" / "mtp" / "rt"
            rt.mkdir(parents=True)
            (rt / "experts.bin").write_bytes(b"\0")
            with mock.patch.dict(os.environ, {"STRATA_SYCL_ROOT": d}):
                self.assertTrue(drafting.mtp_ready(["--spec", "4", "--mtp", "/work/Strata-data/mtp/rt"]),
                                "the container's /work path checked through STRATA_SYCL_ROOT")

    def test_a_missing_layer_under_the_mount_is_still_missing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"STRATA_SYCL_ROOT": d}):
            self.assertFalse(drafting.mtp_ready(["--mtp", "/work/Strata-data/mtp/rt"]))

    def test_the_default_mount_is_the_folder_above_the_checkout(self):
        here = Path(drafting.__file__).resolve().parents[1].parent
        env = {k: v for k, v in os.environ.items() if k != "STRATA_SYCL_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(drafting._sycl_host_path("/work/Strata-data/mtp/rt"), here / "Strata-data" / "mtp" / "rt")

    def test_a_config_written_before_the_layout_move_is_found_above_the_checkout(self):
        # 0.2.17 (#5395, again on 0.2.16): the mount was the folder above the checkout; since the layout move the default is the checkout
        # itself. A data folder beside the checkout (<above>/Strata-data) is where an older Intel config's /work/Strata-data points.
        with tempfile.TemporaryDirectory() as d:
            above = Path(d).resolve()
            checkout = above / "Strata"
            fake_file = checkout / "terminal" / "strata_control" / "drafting.py"
            fake_file.parent.mkdir(parents=True)
            rt = above / "Strata-data" / "mtp" / "rt"
            rt.mkdir(parents=True)
            (rt / "experts.bin").write_bytes(b"\0")
            env = {k: v for k, v in os.environ.items() if k != "STRATA_SYCL_ROOT"}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(drafting, "__file__", str(fake_file)):
                self.assertEqual(drafting._sycl_host_path("/work/Strata-data/mtp/rt"), rt)
                self.assertTrue(drafting.mtp_ready(["--mtp", "/work/Strata-data/mtp/rt"]))
                # a data folder inside the checkout (a fresh 0.2.16 install) is found there first
                inside = checkout / "Strata-data" / "mtp" / "rt"
                inside.mkdir(parents=True)
                self.assertEqual(drafting._sycl_host_path("/work/Strata-data/mtp/rt"), inside)
                # nowhere: the checkout's path, as before
                self.assertEqual(drafting._sycl_host_path("/work/nothing/here"), checkout / "nothing" / "here")

    def test_host_paths_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / "mtp" / "rt"
            rt.mkdir(parents=True)
            (rt / "experts.bin").write_bytes(b"\0")
            self.assertTrue(drafting.mtp_ready(["--mtp", str(rt)]))
            self.assertEqual(drafting._sycl_host_path(str(rt)), rt)


if __name__ == "__main__":
    unittest.main()
