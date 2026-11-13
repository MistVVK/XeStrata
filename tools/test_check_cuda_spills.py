"""Tests for tools/check_cuda_spills.py, without a CUDA toolkit: synthetic resource rows and stand-in demanglers.

    python -m unittest tools.test_check_cuda_spills
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_cuda_spills as CS  # noqa: E402

MANGLED = "_ZN6strata7kernels10test_spillEv"
DEMANGLED = "void strata::kernels::test_spill()"
ROWS = [("test.cu.o", "sm_86", MANGLED, {"REG": 32, "STACK": 64, "LOCAL": 0})]   # a spilling kernel, not in ALLOWED


def demangler(d, body):
    """An executable stand-in for cu++filt / c++filt: reads names on stdin, runs `body`."""
    p = Path(d, "filt")
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = fn(*args)
    return rc, out.getvalue()


def run_main(d, filt, lib_rows=ROWS, argv=()):
    """main() with tool discovery and cuobjdump's rows stood in: the demangler is the only real process."""
    lib = Path(d, "libstrata_kernels.a")
    lib.write_bytes(b"")
    with mock.patch.object(CS, "find_tools", return_value=("cuobjdump", filt)), \
         mock.patch.object(CS, "resource_usage", return_value=lib_rows), \
         mock.patch.object(sys, "argv", ["check_cuda_spills.py", *argv, str(lib)]):
        return quiet(CS.main)


class Classification(unittest.TestCase):
    def test_a_spilling_strata_kernel_is_a_violation(self):
        rc, out = quiet(CS.check, "x.a", ROWS, {MANGLED: DEMANGLED})
        self.assertEqual(rc, 1)
        self.assertIn("not in ALLOWED", out)

    def test_the_shipped_table_allows_its_own_measurements_and_a_lowered_allowance_fails(self):
        mangled = "_ZN6strata7kernels12_GLOBAL__N_113rope_table_kernelEPKiifPf"
        demangled = "strata::kernels::(anonymous namespace)::rope_table_kernel(int const*, int, float, float*)"
        rows = [("dflash2.cu.o", "sm_86", mangled, {"REG": 16, "STACK": 32, "LOCAL": 0})]
        rc, out = quiet(CS.check, "x.a", rows, {mangled: demangled})
        self.assertEqual(rc, 0, out)                       # the measured frame is the allowance
        key = "strata::kernels::{anonymous}::rope_table_kernel"
        self.assertEqual(CS.ALLOWED[key]["sm_86"], (32, 0))
        with mock.patch.dict(CS.ALLOWED, {key: {"sm_86": (16, 0)}}):
            rc, out = quiet(CS.check, "x.a", rows, {mangled: demangled})
        self.assertEqual(rc, 1)                            # one allowance lowered: the same object now fails
        self.assertIn("over its allowance STACK 16 LOCAL 0", out)

    def test_sm_89_has_the_allowance_of_the_same_kernels_as_sm_86(self):
        # Ada runs the same device code as Ampere: every kernel the table lists for sm_86 that sm_89 also builds has the same stack there
        key = "strata::kernels::{anonymous}::native_mmvq_il_pf_kernel"
        self.assertEqual(CS.ALLOWED[key]["sm_89"], CS.ALLOWED[key]["sm_86"])
        mangled = "_ZN6strata7kernels12_GLOBAL__N_113rope_table_kernelEPKiifPf"
        demangled = "strata::kernels::(anonymous namespace)::rope_table_kernel(int const*, int, float, float*)"
        rows = [("dflash2.cu.o", "sm_89", mangled, {"REG": 16, "STACK": 32, "LOCAL": 0})]
        rc, out = quiet(CS.check, "x.a", rows, {mangled: demangled})
        self.assertEqual(rc, 0, out)
        rows = [("dflash2.cu.o", "sm_89", mangled, {"REG": 16, "STACK": 40, "LOCAL": 0})]
        rc, out = quiet(CS.check, "x.a", rows, {mangled: demangled})
        self.assertEqual(rc, 1)                            # growth on sm_89 still fails

    def test_a_vendored_ggml_kernel_is_skipped_and_counted(self):
        rows = [("mmq.cu.o", "sm_86", "_Z9mul_mat_qILi2ELi8EEvPKvS1_PfRiiiiii", {"REG": 128, "STACK": 64})]
        rc, out = quiet(CS.check, "x.a", rows, {rows[0][2]: "void mul_mat_q<2, 8>(void const*, void const*, float*, int&, int, int, int, int, int)"})
        self.assertEqual(rc, 0)
        self.assertIn("1 vendored ggml entries skipped", out)


KNOWN_NAME = "void strata::prefill::(anonymous namespace)::iq4_w4a8_kernel<12, 4, true>(int const*, int)"
KNOWN_MANGLED = "_ZN6strata7prefill12_GLOBAL__N_116iq4_w4a8_kernelILi12ELi4ELb1EEEvPKii"
KNOWN_KEY = "strata::prefill::{anonymous}::iq4_w4a8_kernel<12, 4, true>"
OTHER_NAME = "void strata::prefill::(anonymous namespace)::iq4_w4a8_kernel<16, 4, true>(int const*, int)"
OTHER_MANGLED = "_ZN6strata7prefill12_GLOBAL__N_116iq4_w4a8_kernelILi16ELi4ELb1EEEvPKii"
NAMES = {KNOWN_MANGLED: KNOWN_NAME, OTHER_MANGLED: OTHER_NAME}
BASE = {"sm_86": {KNOWN_KEY: (48, 0, CS.KNOWN_REASON)}, "sm_120": {KNOWN_KEY: (32, 0, CS.KNOWN_REASON)}}


def row(sm="sm_86", mangled=KNOWN_MANGLED, stack=48, local=0):
    return ("moe_iq4_w4a8.cu.o", sm, mangled, {"REG": 64, "STACK": stack, "LOCAL": local})


class InstanceKey(unittest.TestCase):
    def test_the_exact_instance_without_return_type_and_parameters(self):
        self.assertEqual(CS.instance_key(KNOWN_NAME), KNOWN_KEY)
        self.assertEqual(CS.instance_key(OTHER_NAME), "strata::prefill::{anonymous}::iq4_w4a8_kernel<16, 4, true>")

    def test_both_demanglers_spell_the_anonymous_namespace_alike(self):
        self.assertEqual(CS.instance_key(KNOWN_NAME.replace("(anonymous namespace)", "<unnamed>")), KNOWN_KEY)

    def test_cu_filt_and_c_filt_spell_one_instance_alike(self):
        cu = ("void strata::prefill::mmq::<unnamed>::iq4_w4a8_kernel<(bool)1, strata::prefill::mmq::<unnamed>::Cfg<(int)96, (int)2, "
              "(unsigned int)7, (long)-1>>(strata::prefill::mmq::Args)")
        gnu = ("void strata::prefill::mmq::(anonymous namespace)::iq4_w4a8_kernel<true, strata::prefill::mmq::(anonymous namespace)::Cfg<96, 2, "
               "7u, -1l> >(strata::prefill::mmq::Args)")
        want = "strata::prefill::mmq::{anonymous}::iq4_w4a8_kernel<true, strata::prefill::mmq::{anonymous}::Cfg<96, 2, 7, -1>>"
        self.assertEqual(CS.instance_key(cu), want)
        self.assertEqual(CS.instance_key(gnu), want)
        self.assertEqual(CS.instance_key(cu.replace("(bool)1", "(bool)0")), want.replace("<true,", "<false,"))   # a bool stays a bool

    def test_template_arguments_with_spaces_and_parentheses_stay_whole(self):
        self.assertEqual(CS.instance_key("void ns::k<(ns::E)3, ns::T<1, 2>, true>(float*)"), "ns::k<(ns::E)3, ns::T<1, 2>, true>")   # an enum cast is kept

    def test_a_plain_kernel_and_a_name_without_parameters(self):
        self.assertEqual(CS.instance_key("void strata::kernels::tp_band_kernel(int)"), "strata::kernels::tp_band_kernel")
        self.assertEqual(CS.instance_key("strata::k"), "strata::k")

    def test_instances_of_one_template_have_different_keys_but_one_allowance_key(self):
        self.assertNotEqual(CS.instance_key(KNOWN_NAME), CS.instance_key(OTHER_NAME))
        self.assertEqual(CS.kernel_key(KNOWN_NAME), CS.kernel_key(OTHER_NAME))


class Baseline(unittest.TestCase):
    """A known violation passes at or below its listed bytes; a new instance and a known instance that grew fail; an architecture
    with no baseline says so."""

    def judge(self, rows, baseline=BASE):
        return quiet(CS.check, "x.a", rows, NAMES, baseline)

    def test_the_kernel_of_the_baseline_is_not_in_allowed(self):
        # the instances below are violations because the kernel has no allowance: a baseline entry is not an allowance
        self.assertNotIn(CS.kernel_key(KNOWN_NAME), CS.ALLOWED)

    def test_a_known_instance_at_its_listed_bytes_passes_and_is_counted(self):
        rc, out = self.judge([row()])
        self.assertEqual(rc, 0, out)
        self.assertIn("1 use stack / local memory, 1 of them known from the baseline, 0 not allowed", out)
        self.assertNotIn("VIOLATION", out)

    def test_a_known_instance_below_its_listed_bytes_passes(self):
        rc, out = self.judge([row(stack=32)])
        self.assertEqual(rc, 0, out)

    def test_a_known_instance_that_grew_fails_in_stack_and_in_local(self):
        for stack, local in ((56, 0), (48, 8), (0, 8)):
            rc, out = self.judge([row(stack=stack, local=local)])
            self.assertEqual(rc, 1, (stack, local, out))
            self.assertIn("known instance grew: over its baseline STACK 48 LOCAL 0", out)
            self.assertIn("0 of them known from the baseline, 1 not allowed", out)

    def test_a_new_instance_of_a_known_kernel_fails_beside_the_known_one(self):
        rc, out = self.judge([row(), row(mangled=OTHER_MANGLED, stack=8)])
        self.assertEqual(rc, 1, out)
        self.assertIn("2 use stack / local memory, 1 of them known from the baseline, 1 not allowed", out)
        self.assertIn("VIOLATION %s sm_86: STACK 8 LOCAL 0 (not in ALLOWED)" % OTHER_NAME, out)
        self.assertNotIn("VIOLATION %s" % KNOWN_NAME, out)

    def test_the_baseline_is_per_architecture(self):
        rc, out = self.judge([row(sm="sm_120", stack=32)])
        self.assertEqual(rc, 0, out)
        rc, out = self.judge([row(sm="sm_120", stack=40)])        # sm_120 lists 32, not sm_86's 48
        self.assertEqual(rc, 1, out)
        only_86 = {"sm_86": BASE["sm_86"], "sm_120": {}}
        rc, out = self.judge([row(sm="sm_120", stack=32)], only_86)  # an architecture with an empty baseline knows nothing
        self.assertEqual(rc, 1, out)
        self.assertNotIn("no baseline", out)

    def test_an_architecture_without_a_baseline_says_so_and_still_fails(self):
        rc, out = self.judge([row(sm="sm_60", stack=48)])
        self.assertEqual(rc, 1, out)
        self.assertIn("no baseline for sm_60: its 1 violations cannot be told from new ones (baselines: sm_120, sm_86)", out)
        self.assertIn("VIOLATION", out)

    def test_an_architecture_without_a_baseline_and_without_violations_is_quiet(self):
        rc, out = self.judge([row(sm="sm_60", stack=0)])
        self.assertEqual(rc, 0, out)
        self.assertNotIn("no baseline", out)

    def test_no_baseline_at_all_is_not_a_pass(self):
        rc, out = self.judge([row()], {})
        self.assertEqual(rc, 1, out)
        self.assertIn("no baseline for sm_86", out)
        self.assertIn("baselines: none", out)

    def test_a_violation_that_is_allowed_is_not_counted_as_known(self):
        names = {"_ZN6strata7kernels12_GLOBAL__N_113rope_table_kernelEv": "void strata::kernels::(anonymous namespace)::rope_table_kernel()"}
        rows = [("d.cu.o", "sm_86", next(iter(names)), {"STACK": 32})]
        rc, out = quiet(CS.check, "x.a", rows, names, BASE)
        self.assertEqual(rc, 0, out)
        self.assertIn("1 use stack / local memory, 0 of them known from the baseline, 0 not allowed", out)

    def test_seen_collects_the_known_instances_and_verbose_prints_them(self):
        seen = {}
        rc, out = quiet(CS.check, "x.a", [row()], NAMES, BASE, seen, True)
        self.assertEqual(rc, 0)
        self.assertEqual(seen, {"sm_86": {KNOWN_KEY}})
        self.assertIn("known %s sm_86: STACK 48 LOCAL 0" % KNOWN_NAME, out)


class BaselineFile(unittest.TestCase):
    def write(self, d, obj):
        p = Path(d, "baseline.json")
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return str(p)

    def entry(self, **kw):
        e = {"name": KNOWN_KEY, "stack": 48, "local": 0, "reason": CS.KNOWN_REASON}
        e.update(kw)
        return e

    def test_a_good_file_loads_and_null_and_underscore_keys_are_not_baselines(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, {"_about": "x", "sm_86": {"measured_with": "t", "instances": [self.entry()]}, "sm_60": None})
            self.assertEqual(CS.load_baseline(p), {"sm_86": {KNOWN_KEY: (48, 0, CS.KNOWN_REASON)}})

    def test_a_file_that_cannot_be_read_or_is_incomplete_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = [self.write(d, "{not json"), self.write(d, "[]"), os.path.join(d, "missing.json"),
                   self.write(d, {"sm_86": {}}),
                   self.write(d, {"sm_86": {"instances": [self.entry(reason="")]}}),
                   self.write(d, {"sm_86": {"instances": [self.entry(stack="48")]}}),
                   self.write(d, {"sm_86": {"instances": [self.entry(stack=-1)]}}),
                   self.write(d, {"sm_86": {"instances": [{"name": KNOWN_KEY, "stack": 1, "local": 0}]}}),
                   self.write(d, {"sm_86": {"instances": [self.entry(), self.entry(stack=8)]}})]
            for p in bad:
                with self.assertRaises(ValueError, msg=p):
                    CS.load_baseline(p)

    def test_the_default_baseline_is_the_file_beside_the_script_and_a_missing_one_is_empty(self):
        self.assertEqual(Path(CS.BASELINE_FILE), ROOT / "tools" / "cuda_spill_baseline.json")
        with mock.patch.object(CS, "BASELINE_FILE", str(ROOT / "tools" / "no_such_baseline.json")):
            self.assertEqual(CS.default_baseline(), {})

    def test_the_shipped_baseline(self):
        base = CS.load_baseline(CS.BASELINE_FILE)
        self.assertIn("sm_120", base)
        self.assertIn("sm_86", base)
        self.assertNotIn("sm_60", base)                       # about 400 violations, nothing measured per instance: no baseline
        for sm, per in base.items():
            for name, (stack, local, reason) in per.items():
                self.assertTrue(reason.strip(), (sm, name))
                mangled = "_Z%d" % len(name)                  # one stand-in kernel per entry, at its own bytes
                names = {mangled: "void %s()" % name}
                gated, held, viols = CS.find_violations([("o.cu.o", sm, mangled, {"STACK": stack, "LOCAL": local})], names)
                self.assertEqual(len(viols), 1, "%s %s is not a violation at its own bytes: a dead entry" % (sm, name))
                rc, out = quiet(CS.check, "x.a", [("o.cu.o", sm, mangled, {"STACK": stack, "LOCAL": local})], names, base)
                self.assertEqual(rc, 0, out)
                for grown in ((stack + 8, local), (stack, local + 8)):
                    rc, out = quiet(CS.check, "x.a", [("o.cu.o", sm, mangled, {"STACK": grown[0], "LOCAL": grown[1]})], names, base)
                    self.assertEqual(rc, 1, (sm, name, grown))


@unittest.skipIf(os.name == "nt", "the stand-in demangler is a #!/bin/sh script; Windows cannot run it as a program (needs a POSIX shell and exec of a shebang file)")
class BaselineMain(unittest.TestCase):
    """main(): the --baseline file, the known count over the libraries, a baseline that cannot be read, --print-baseline."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.filt = demangler(self.d.name, "while read -r n; do echo '%s'; done" % KNOWN_NAME)
        self.path = Path(self.d.name, "baseline.json")
        self.path.write_text(json.dumps({"sm_86": {"instances": [
            {"name": KNOWN_KEY, "stack": 48, "local": 0, "reason": CS.KNOWN_REASON}]}}), encoding="utf-8")

    def test_known_passes_prints_the_known_count_and_a_grown_one_fails(self):
        rc, out = run_main(self.d.name, self.filt, [row()], ["--baseline", str(self.path)])
        self.assertEqual(rc, 0, out)
        self.assertIn("check_cuda_spills: baseline sm_86: 1 known violations seen, 1 listed", out)
        rc, out = run_main(self.d.name, self.filt, [row(stack=56)], ["--baseline", str(self.path)])
        self.assertEqual(rc, 1, out)
        self.assertIn("known instance grew", out)

    def test_an_unreadable_baseline_is_exit_2_and_judges_nothing(self):
        for text in ("{not json", '{"sm_86": {"instances": [{"name": "x"}]}}'):
            self.path.write_text(text, encoding="utf-8")
            rc, out = run_main(self.d.name, self.filt, [row()], ["--baseline", str(self.path)])
            self.assertEqual(rc, 2, out)
            self.assertIn("baseline", out)
            self.assertNotIn("VIOLATION", out)
        rc, out = run_main(self.d.name, self.filt, [row()], ["--baseline", str(Path(self.d.name, "none.json"))])
        self.assertEqual(rc, 2, out)

    def test_print_baseline_makes_a_file_the_check_accepts(self):
        with mock.patch.object(CS, "toolkit_line", return_value="Cuda compilation tools, release 99.9"):
            rc, out = run_main(self.d.name, self.filt, [row(), row(sm="sm_120", stack=32)], ["--print-baseline"])
        self.assertEqual(rc, 0)
        made = json.loads(out)
        self.assertEqual(sorted(made), ["sm_120", "sm_86"])
        self.assertEqual(made["sm_86"]["measured_with"], "Cuda compilation tools, release 99.9")
        self.assertEqual(made["sm_86"]["instances"], [{"name": KNOWN_KEY, "stack": 48, "local": 0, "reason": CS.KNOWN_REASON}])
        self.path.write_text(out, encoding="utf-8")
        rc, out = run_main(self.d.name, self.filt, [row(), row(sm="sm_120", stack=32)], ["--baseline", str(self.path)])
        self.assertEqual(rc, 0, out)
        self.assertIn("2 use stack / local memory, 2 of them known", out)


@unittest.skipIf(os.name == "nt", "the stand-in demangler is a #!/bin/sh script; Windows cannot run it as a program (needs a POSIX shell and exec of a shebang file)")
class Demangling(unittest.TestCase):
    """A mangled name has no "::", so a kernel the demangler did not translate would read as vendored and be skipped:
    every way the demangler can be absent or wrong must end in 77, never in a pass."""

    def test_no_demangler_is_not_a_pass(self):
        rc, out = quiet(CS.demangle, None, [MANGLED])
        self.assertIsNone(rc)
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d, "libstrata_kernels.a")
            lib.write_bytes(b"")
            with mock.patch.object(CS, "find_tools", return_value=("cuobjdump", None)), \
                 mock.patch.object(CS, "resource_usage", side_effect=AssertionError("read a library with no demangler")), \
                 mock.patch.object(sys, "argv", ["check_cuda_spills.py", str(lib)]):
                rc, out = quiet(CS.main)
        self.assertEqual(rc, 77)
        self.assertIn("no demangler", out)

    def test_a_demangler_that_exits_nonzero_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            filt = demangler(d, "cat >/dev/null; echo 'no such tool' >&2; exit 3")
            self.assertIsNone(quiet(CS.demangle, filt, [MANGLED])[0])
            rc, out = run_main(d, filt)
        self.assertEqual(rc, 77)
        self.assertIn("exited 3", out)
        self.assertNotIn("not allowed", out)

    def test_a_demangler_that_prints_the_wrong_number_of_lines_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            filt = demangler(d, "cat >/dev/null; exit 0")
            self.assertIsNone(quiet(CS.demangle, filt, [MANGLED])[0])
            rc, out = run_main(d, filt)
        self.assertEqual(rc, 77)
        self.assertIn("printed 0 lines for 1 names", out)

    def test_a_demangler_that_hands_a_mangled_name_back_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            filt = demangler(d, "cat")                                   # echoes every name as it came
            self.assertIsNone(quiet(CS.demangle, filt, [MANGLED])[0])
            rc, out = run_main(d, filt)
            rc_list, _ = run_main(d, filt, argv=["--list"])
        self.assertEqual(rc, 77)
        self.assertEqual(rc_list, 77)
        self.assertIn("left 1 of 1 names mangled", out)

    def test_a_demangler_that_prints_a_blank_name_is_not_a_pass(self):
        # exit 0 and one line per name, but the line is empty: the key would be "" and vendored("") is true, so a
        # spilling Strata kernel would be skipped as ggml's
        with tempfile.TemporaryDirectory() as d:
            filt = demangler(d, "while read -r n; do echo; done")
            self.assertIsNone(quiet(CS.demangle, filt, [MANGLED])[0])
            rc, out = run_main(d, filt)
        self.assertEqual(rc, 77)
        self.assertIn("printed a blank name for 1 of 1 names", out)
        self.assertNotIn("VIOLATION", out)

    def test_an_unmangled_name_may_come_back_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            filt = demangler(d, "cat")
            names, _ = quiet(CS.demangle, filt, ["plain_c_kernel"])     # extern "C": nothing to demangle
        self.assertEqual(names, {"plain_c_kernel": "plain_c_kernel"})

    def test_a_working_demangler_reaches_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            filt = demangler(d, "while read -r n; do echo '%s'; done" % DEMANGLED)
            rc, out = run_main(d, filt)
        self.assertEqual(rc, 1)
        self.assertIn("VIOLATION %s sm_86: STACK 64 LOCAL 0 (not in ALLOWED)" % DEMANGLED, out)


if __name__ == "__main__":
    unittest.main()
