"""Is anybody there to answer a question? (0.2.17: `strata launch < /dev/null` showed "Start it now? [Y/n]" and then ended with "There is nobody to answer".)

The Terminal's conventions for a run with no keyboard: no screens, no question that nobody can answer, a plain sentence that says what to name,
and exit code 2 (``strata launch`` with no choices and no terminal; ``--no-tui`` never asks; the data-folder move keeps the library). This module
holds the one test they share. A pipe or a file may carry answers (Git Bash and mintty give a person's typing as a pipe), so only the null device
counts as "no keyboard".
"""

from __future__ import annotations

import os
import stat
import sys


def stdin_is_no_keyboard(stream=None) -> bool:
    """True when the input is the null device (`< /dev/null`, NUL on Windows): a character device that is not a terminal. False for a terminal, a
    pipe, a file, and for anything whose kind cannot be read (a replaced stdin without a file number, a closed one)."""
    stream = sys.stdin if stream is None else stream
    try:
        fd = stream.fileno()
        return stat.S_ISCHR(os.fstat(fd).st_mode) and not os.isatty(fd)
    except (AttributeError, OSError, ValueError):
        return False
