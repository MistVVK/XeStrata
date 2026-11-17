// io_priority_test (#4733): on Windows a process that starts at Low I/O priority is put back to Normal; elsewhere the call does nothing.
#include "strata/platform/io_priority.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
static unsigned long io_prio() {
    using Query = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    auto q = reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationProcess")));
    ULONG v = 99, r = 0;
    return q != nullptr && q(GetCurrentProcess(), 33, &v, sizeof v, &r) == 0 ? v : 99;
}
static bool set_io_prio(ULONG v) {
    using Set = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG);
    auto s = reinterpret_cast<Set>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSetInformationProcess")));
    return s != nullptr && s(GetCurrentProcess(), 33, &v, sizeof v) == 0;
}
#endif

int main() {
#if defined(_WIN32)
    int bad = 0;
    // 1. started at Low: back to Normal, and the call reports what it found
    if (!set_io_prio(1)) { std::printf("io_priority_test: SKIP (cannot lower the I/O priority here)\n"); return 77; }
    int was = strata::platform::restore_normal_io_priority(stderr);
    if (was != 1 || io_prio() != 2) { std::printf("FAIL lowered: was=%d now=%lu\n", was, io_prio()); ++bad; }
    // 2. already Normal: untouched, reported as 2
    was = strata::platform::restore_normal_io_priority(nullptr);
    if (was != 2 || io_prio() != 2) { std::printf("FAIL normal: was=%d now=%lu\n", was, io_prio()); ++bad; }
    // 3. STRATA_KEEP_IO_PRIORITY=1 leaves a low value alone
    _putenv("STRATA_KEEP_IO_PRIORITY=1");
    set_io_prio(1);
    was = strata::platform::restore_normal_io_priority(nullptr);
    if (was != -1 || io_prio() != 1) { std::printf("FAIL keep: was=%d now=%lu\n", was, io_prio()); ++bad; }
    std::printf(bad == 0 ? "io_priority_test: OK\n" : "io_priority_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
#else
    std::printf("io_priority_test: SKIP (Windows only)\n");
    return 77;
#endif
}
