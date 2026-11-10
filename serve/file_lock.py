# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The platform's exclusive advisory file lock; callers choose how to wait and handle failure."""
import os


def acquire(f, blocking: bool = False, rewind: bool = False) -> None:
    if os.name == "nt":
        import msvcrt
        if rewind:
            f.seek(0)
        getattr(msvcrt, "locking")(f.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def release(f) -> None:
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        getattr(msvcrt, "locking")(f.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
