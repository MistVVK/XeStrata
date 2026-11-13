// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/platform/process.hpp - a child process with pipes: the program found on the PATH (as a shell would),
// its standard input and output pipes the caller's (either may be left alone: then the child's is the null device), its
// standard error read by a thread of its own into a bounded tail (so a child that writes much to it never blocks on a
// full pipe), the exit code waited for.  The engine runs ffmpeg and ffprobe this way (src/video/).
//
// Linux: posix_spawnp with the pipes' ends dup'ed onto 0, 1 and 2.  Windows: CreateProcessW with inheritable pipe
// handles, the command line quoted as the C runtime parses it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::platform {

class Process {
public:
    /// Starts argv[0] (looked up on the PATH) with `argv`; `in`: the caller writes its standard input, `out`: the
    /// caller reads its standard output.  std::runtime_error when it cannot start (not found among them).
    Process(const std::vector<std::string>& argv, bool in, bool out);
    ~Process();   // kills a child still running, then waits for it
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    /// Writes all of `n` bytes to the child's standard input; false when the child has closed it (it ended).
    bool write(const void* p, size_t n);
    /// Closes the child's standard input (its end of file).
    void close_in();
    /// Reads up to `n` bytes of the child's standard output; 0 at its end.
    size_t read(void* p, size_t n);
    /// Reads exactly `n` bytes unless the output ends first; the bytes read.
    size_t read_all(void* p, size_t n);
    /// Reads the whole of the standard output, up to its end.
    std::string read_rest();
    /// Waits for the child; its exit code (128 + the signal where a signal ended it).
    int wait();
    /// Ends the child now (SIGKILL, TerminateProcess).
    void kill();
    /// The last bytes (up to 16 KiB) the child wrote to its standard error.
    std::string error_tail() const;

    /// Whether `program` is found on the PATH (an executable file).
    static bool found(const std::string& program);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// This process's standard output sent to the null device until restore (the serve engines' start: the requests of
/// its measurement of the drafters print their lines as a request does, and nobody reads them).  Flush the C stream
/// before both.  mute returns the saved descriptor, or -1 when it could not (the output then stays as it is).
int stdout_mute();
void stdout_restore(int saved);

}  // namespace strata::platform
