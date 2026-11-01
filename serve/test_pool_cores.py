# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The run config's experimental "pool_cores" (the CPU expert workers' logical processors; --pool-cores in the
engine)."""
import contextlib
import io
import unittest

from serve.server import engine_args, pool_cores_args


class PoolCoresConfig(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(pool_cores_args({}, []), [])
        self.assertEqual(pool_cores_args({"pool_cores": None}, []), [])
        self.assertEqual(pool_cores_args({"pool_cores": ""}, []), [])
        self.assertEqual(pool_cores_args({"pool_cores": []}, []), [])
        self.assertEqual(pool_cores_args({"pool_cores": False}, []), [])
        # nothing is added to an ordinary config
        self.assertEqual(engine_args({"args": ["--pack", "p"], "session_save_reclaim": False}), ["--pack", "p"])

    def test_text_and_list_forms(self):
        self.assertEqual(pool_cores_args({"pool_cores": "2,4,6,8"}, []), ["--pool-cores", "2,4,6,8"])
        self.assertEqual(pool_cores_args({"pool_cores": " 2-14:2 "}, []), ["--pool-cores", "2-14:2"])
        self.assertEqual(pool_cores_args({"pool_cores": [2, 4, 6]}, []), ["--pool-cores", "2,4,6"])
        self.assertEqual(pool_cores_args({"pool_cores": "0,2-6,10"}, []), ["--pool-cores", "0,2-6,10"])

    def test_the_args_own_value_wins(self):
        self.assertEqual(pool_cores_args({"pool_cores": "2,4"}, ["--pool-cores", "6,8"]), [])

    def test_a_list_that_cannot_be_read_is_said_and_ignored(self):
        for bad in ("two,four", "1,,2", "6-2x", -1, True, 5, [1, "a"], [-1], {"a": 1}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = pool_cores_args({"pool_cores": bad}, [])
            self.assertEqual(r, [], bad)
            self.assertIn("pool_cores", buf.getvalue(), bad)

    def test_reaches_the_engine_arguments(self):
        args = engine_args({"args": ["--pack", "p"], "pool_cores": "2,4,6"})
        self.assertEqual(args[-2:], ["--pool-cores", "2,4,6"])


if __name__ == "__main__":
    unittest.main()
