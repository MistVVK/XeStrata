// #4733: see include/strata/platform/io_priority.hpp
#include "strata/platform/io_priority.hpp"

#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::platform {

int restore_normal_io_priority(std::FILE* log) {
#if defined(_WIN32)
    const char* keep = std::getenv("STRATA_KEEP_IO_PRIORITY");
    if (keep != nullptr && keep[0] == '1') return -1;
    using Query = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    using Set = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    if (nt == nullptr) return -1;
    auto query = reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(nt, "NtQueryInformationProcess")));
    auto set = reinterpret_cast<Set>(reinterpret_cast<void*>(GetProcAddress(nt, "NtSetInformationProcess")));
    if (query == nullptr || set == nullptr) return -1;
    constexpr ULONG kProcessIoPriority = 33;
    ULONG now = 2, ret = 0;
    if (query(GetCurrentProcess(), kProcessIoPriority, &now, sizeof now, &ret) != 0 || ret < sizeof now) return -1;
    if (now >= 2) return (int) now;
    ULONG want = 2;
    if (set(GetCurrentProcess(), kProcessIoPriority, &want, sizeof want) != 0) {
        if (log != nullptr)
            std::fprintf(log, "strata: this program runs at a low disk priority (inherited from the program that started it) "
                              "and Windows did not let it go back to normal; reading the model can be several times slower (#4733)\n");
        return (int) now;
    }
    if (log != nullptr)
        std::fprintf(log, "strata: the program that started the engine runs at a low disk priority, which the engine inherits and "
                          "which makes model reads several times slower; the engine's disk priority is back to normal (#4733)\n");
    return (int) now;
#else
    (void) log;
    return -1;
#endif
}

}  // namespace strata::platform
