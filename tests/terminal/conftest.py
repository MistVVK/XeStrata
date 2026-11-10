"""Terminal tests do not depend on the graphics card of the box they run on.

The Drafting step of `strata launch` (and the Drafting line of the summary) exists only where the engine can run DFlash2 (CUDA or HIP), so the number
of steps and the layout of the summary used to differ between an Nvidia box and an Intel one. Every test starts on a computer with no card the engine
knows; a test about drafting names the card it means (test_drafting.py and test_drafting_hardware.py patch `detect_hardware` themselves)."""
import os
from unittest import mock

import pytest

# The tests draw the full look (Windows Terminal, Linux, macOS). A Windows run with no modern-terminal marker (a CI runner, a classic console)
# would get the classic console's ASCII / solid look and its "use Windows Terminal" tip line instead; the tests of that look patch
# consoleinfo or set STRATA_GLYPHS themselves.
if os.name == "nt" and not any(os.environ.get(k) for k in ("STRATA_GLYPHS", "WT_SESSION", "TERM_PROGRAM", "ConEmuANSI", "ANSICON")) \
        and not os.environ.get("TERM", "").startswith(("xterm", "screen", "tmux", "vt")):
    os.environ["STRATA_GLYPHS"] = "unicode"


@pytest.fixture(autouse=True)
def _no_card_unless_a_test_names_one(request):
    from strata_control import drafting
    if request.module.__name__.rsplit(".", 1)[-1].startswith("test_drafting"):
        yield                                    # these name their card themselves (setUpModule / per test)
        return
    with mock.patch.object(drafting, "detect_hardware", return_value=("cpu", [])):
        yield
