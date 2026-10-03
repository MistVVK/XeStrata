"""evict.py FILE...: drop the files' pages from the OS cache (POSIX_FADV_DONTNEED; clean pages only)."""
import os
import sys

for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
