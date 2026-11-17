// strata/platform/io_priority.hpp - #4733.  On Windows a process starts with the disk (I/O) priority of the program that
// started it.  An engine started from a low-priority launcher (a Task Scheduler task, a service, a terminal that an IDE or a
// coding agent runs at low priority) therefore reads its model at Low I/O priority, even when its CPU priority is Normal:
// the reporter measured PLE direct reads about 15x slower, cold prompt reads 3-5x slower and decode 14% slower on the same
// drive, with the drive blamed ("--ple-io direct measures slow here").  restore_normal_io_priority() puts the process back
// to Normal when it started lower, and says so in one plain line.  Raising from Low to Normal needs no privilege.
// STRATA_KEEP_IO_PRIORITY=1 leaves it as inherited (for a person who wants the engine in the background on purpose).
// Other systems: nothing to do.
#pragma once

#include <cstdio>

namespace strata::platform {

/// Windows: puts the process's I/O priority (ProcessIoPriority: 0 very low, 1 low, 2 normal) back to Normal when it started
/// lower, and prints one plain line to `log` (none when it is null).  Returns the value the process had before (2 when nothing
/// was changed; -1 when it could not be read, STRATA_KEEP_IO_PRIORITY=1 is set, or on another system).
/// Defined in src/platform/io_priority.cpp (windows.h stays out of every file that includes this).
int restore_normal_io_priority(std::FILE* log = stderr);

}  // namespace strata::platform
