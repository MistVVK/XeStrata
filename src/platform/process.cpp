// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/platform/process.cpp - see include/strata/platform/process.hpp.

#include "strata/platform/process.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <spawn.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace strata::platform {

namespace {
constexpr size_t kTail = 16 << 10;   // the standard error's bytes kept
}

#if defined(_WIN32)

namespace {

std::wstring wide(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

// an argument quoted as the C runtime's command line parser reads it back
void quote(std::wstring& line, const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        line += a;
        return;
    }
    line += L'"';
    for (size_t i = 0;; ++i) {
        size_t slashes = 0;
        while (i < a.size() && a[i] == L'\\') ++i, ++slashes;
        if (i == a.size()) {
            line.append(slashes * 2, L'\\');
            break;
        }
        if (a[i] == L'"') line.append(slashes * 2 + 1, L'\\');
        else line.append(slashes, L'\\');
        line += a[i];
    }
    line += L'"';
}

std::string why(const char* what) { return std::string(what) + ": error " + std::to_string(GetLastError()); }

}  // namespace

struct Process::Impl {
    HANDLE proc = nullptr, in = nullptr, out = nullptr, err = nullptr;
    std::thread err_thread;
    mutable std::mutex mu;
    std::string tail;
    int code = -1;
    bool waited = false;
};

Process::Process(const std::vector<std::string>& argv, bool in, bool out) : impl_(std::make_unique<Impl>()) {
    if (argv.empty()) throw std::invalid_argument("Process: no program");
    Impl& m = *impl_;
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE in_child = nullptr, out_child = nullptr, err_child = nullptr, nul = nullptr;
    const auto pipe = [&](HANDLE& parent, HANDLE& child, bool parent_writes) {
        HANDLE r = nullptr, w = nullptr;
        if (!CreatePipe(&r, &w, &sa, 1 << 20)) throw std::runtime_error(why("CreatePipe"));
        parent = parent_writes ? w : r;
        child = parent_writes ? r : w;
        SetHandleInformation(parent, HANDLE_FLAG_INHERIT, 0);
    };
    if (!in || !out) {
        nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING,
                          0, nullptr);
        if (nul == INVALID_HANDLE_VALUE) throw std::runtime_error(why("NUL"));
    }
    if (in) pipe(m.in, in_child, true);
    if (out) pipe(m.out, out_child, false);
    pipe(m.err, err_child, false);
    std::wstring line;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i) line += L' ';
        quote(line, wide(argv[i]));
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in ? in_child : nul;
    si.hStdOutput = out ? out_child : nul;
    si.hStdError = err_child;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr,
                                   &si, &pi);
    const DWORD e = GetLastError();
    for (HANDLE h : {in_child, out_child, err_child, nul})
        if (h != nullptr) CloseHandle(h);
    if (!ok) {
        for (HANDLE h : {m.in, m.out, m.err})
            if (h != nullptr) CloseHandle(h);
        m.in = m.out = m.err = nullptr;
        throw std::runtime_error(argv[0] + ": could not start it (error " + std::to_string(e) + ")");
    }
    CloseHandle(pi.hThread);
    m.proc = pi.hProcess;
    m.err_thread = std::thread([&m] {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(m.err, buf, sizeof buf, &n, nullptr) && n > 0) {
            std::lock_guard<std::mutex> lock(m.mu);
            m.tail.append(buf, n);
            if (m.tail.size() > kTail) m.tail.erase(0, m.tail.size() - kTail);
        }
    });
}

Process::~Process() {
    Impl& m = *impl_;
    if (m.proc != nullptr && !m.waited) {
        TerminateProcess(m.proc, 1);
        wait();
    }
    close_in();
    if (m.err_thread.joinable()) m.err_thread.join();
    for (HANDLE h : {m.out, m.err, m.proc})
        if (h != nullptr) CloseHandle(h);
}

bool Process::write(const void* p, size_t n) {
    Impl& m = *impl_;
    const char* c = static_cast<const char*>(p);
    while (n > 0) {
        DWORD w = 0;
        if (m.in == nullptr || !WriteFile(m.in, c, (DWORD) std::min<size_t>(n, 1 << 30), &w, nullptr)) return false;
        c += w;
        n -= w;
    }
    return true;
}

void Process::close_in() {
    if (impl_->in != nullptr) CloseHandle(impl_->in);
    impl_->in = nullptr;
}

size_t Process::read(void* p, size_t n) {
    DWORD r = 0;
    if (impl_->out == nullptr || !ReadFile(impl_->out, p, (DWORD) std::min<size_t>(n, 1 << 30), &r, nullptr)) return 0;
    return r;
}

int Process::wait() {
    Impl& m = *impl_;
    if (m.waited) return m.code;
    WaitForSingleObject(m.proc, INFINITE);
    DWORD c = 1;
    GetExitCodeProcess(m.proc, &c);
    m.code = (int) c;
    m.waited = true;
    if (m.err_thread.joinable()) m.err_thread.join();
    return m.code;
}

void Process::kill() {
    if (impl_->proc != nullptr && !impl_->waited) TerminateProcess(impl_->proc, 1);
}

bool Process::found(const std::string& program) {
    const std::wstring w = wide(program);
    wchar_t buf[MAX_PATH];
    return SearchPathW(nullptr, w.c_str(), L".exe", MAX_PATH, buf, nullptr) > 0;
}

#else

namespace {

std::string why(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

// a write to a pipe whose reader has gone is EPIPE here, not a signal that ends the engine
void ignore_sigpipe() {
    static std::once_flag once;
    std::call_once(once, [] {
        struct sigaction old {};
        if (sigaction(SIGPIPE, nullptr, &old) == 0 && old.sa_handler == SIG_DFL) std::signal(SIGPIPE, SIG_IGN);
    });
}

}  // namespace

struct Process::Impl {
    pid_t pid = -1;
    int in = -1, out = -1, err = -1;
    std::thread err_thread;
    mutable std::mutex mu;
    std::string tail;
    int code = -1;
    bool waited = false;
};

Process::Process(const std::vector<std::string>& argv, bool in, bool out) : impl_(std::make_unique<Impl>()) {
    if (argv.empty()) throw std::invalid_argument("Process: no program");
    ignore_sigpipe();
    Impl& m = *impl_;
    int pin[2] = {-1, -1}, pout[2] = {-1, -1}, perr[2] = {-1, -1};
    const auto close_all = [&] {
        for (int fd : {pin[0], pin[1], pout[0], pout[1], perr[0], perr[1]})
            if (fd >= 0) ::close(fd);
    };
    if ((in && ::pipe2(pin, O_CLOEXEC) != 0) || (out && ::pipe2(pout, O_CLOEXEC) != 0) ||
        ::pipe2(perr, O_CLOEXEC) != 0) {
        const std::string e = why("pipe");
        close_all();
        throw std::runtime_error(e);
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    if (in) posix_spawn_file_actions_adddup2(&fa, pin[0], 0);
    else posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    if (out) posix_spawn_file_actions_adddup2(&fa, pout[1], 1);
    else posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, perr[1], 2);
    std::vector<char*> args;
    args.reserve(argv.size() + 1);
    for (const std::string& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    posix_spawnattr_t attr;
    int rc = posix_spawnattr_init(&attr);
    if (rc == 0) {
        sigset_t mask;
        rc = pthread_sigmask(SIG_SETMASK, nullptr, &mask);
        if (rc == 0) {
            sigdelset(&mask, SIGTERM);
            sigdelset(&mask, SIGINT);
            rc = posix_spawnattr_setsigmask(&attr, &mask);
        }
        if (rc == 0) rc = posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK);
        if (rc == 0) rc = posix_spawnp(&m.pid, args[0], &fa, &attr, args.data(), environ);
        posix_spawnattr_destroy(&attr);
    }
    posix_spawn_file_actions_destroy(&fa);
    for (int* fd : {&pin[0], &pout[1], &perr[1]})
        if (*fd >= 0) ::close(*fd), *fd = -1;
    if (rc != 0) {
        close_all();
        m.pid = -1;
        throw std::runtime_error(argv[0] + ": could not start it (" + std::strerror(rc) + ")");
    }
    m.in = pin[1];
    m.out = pout[0];
    m.err = perr[0];
    m.err_thread = std::thread([&m] {
        char buf[4096];
        for (;;) {
            const ssize_t n = ::read(m.err, buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            std::lock_guard<std::mutex> lock(m.mu);
            m.tail.append(buf, (size_t) n);
            if (m.tail.size() > kTail) m.tail.erase(0, m.tail.size() - kTail);
        }
    });
}

Process::~Process() {
    Impl& m = *impl_;
    if (m.pid > 0 && !m.waited) {
        ::kill(m.pid, SIGKILL);
        wait();
    }
    close_in();
    if (m.err_thread.joinable()) m.err_thread.join();
    for (int fd : {m.out, m.err})
        if (fd >= 0) ::close(fd);
}

bool Process::write(const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    while (n > 0) {
        if (impl_->in < 0) return false;
        const ssize_t w = ::write(impl_->in, c, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        c += w;
        n -= (size_t) w;
    }
    return true;
}

void Process::close_in() {
    if (impl_->in >= 0) ::close(impl_->in);
    impl_->in = -1;
}

size_t Process::read(void* p, size_t n) {
    for (;;) {
        if (impl_->out < 0) return 0;
        const ssize_t r = ::read(impl_->out, p, n);
        if (r < 0 && errno == EINTR) continue;
        return r > 0 ? (size_t) r : 0;
    }
}

int Process::wait() {
    Impl& m = *impl_;
    if (m.waited || m.pid <= 0) return m.code;
    int st = 0;
    while (::waitpid(m.pid, &st, 0) < 0 && errno == EINTR) {
    }
    m.code = WIFEXITED(st) ? WEXITSTATUS(st) : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : 1;
    m.waited = true;
    if (m.err_thread.joinable()) m.err_thread.join();
    return m.code;
}

void Process::kill() {
    if (impl_->pid > 0 && !impl_->waited) ::kill(impl_->pid, SIGKILL);
}

bool Process::found(const std::string& program) {
    if (program.find('/') != std::string::npos) return ::access(program.c_str(), X_OK) == 0;
    const char* path = std::getenv("PATH");
    std::string p = path != nullptr ? path : "/usr/bin:/bin";
    size_t a = 0;
    for (;;) {
        const size_t b = p.find(':', a);
        const std::string dir = p.substr(a, b == std::string::npos ? std::string::npos : b - a);
        const std::string f = (dir.empty() ? std::string(".") : dir) + "/" + program;
        struct stat st {};
        if (::stat(f.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::access(f.c_str(), X_OK) == 0) return true;
        if (b == std::string::npos) return false;
        a = b + 1;
    }
}

#endif

size_t Process::read_all(void* p, size_t n) {
    char* c = static_cast<char*>(p);
    size_t got = 0;
    while (got < n) {
        const size_t r = read(c + got, n - got);
        if (r == 0) break;
        got += r;
    }
    return got;
}

std::string Process::read_rest() {
    std::string s;
    char buf[1 << 16];
    for (size_t r; (r = read(buf, sizeof buf)) > 0;) s.append(buf, r);
    return s;
}

std::string Process::error_tail() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->tail;
}

int stdout_mute() {
#if defined(_WIN32)
    const int saved = _dup(1);
    const int null_fd = _open("NUL", _O_WRONLY);
    if (saved < 0 || null_fd < 0 || _dup2(null_fd, 1) != 0) {
        if (null_fd >= 0) _close(null_fd);
        if (saved >= 0) _close(saved);
        return -1;
    }
    _close(null_fd);
#else
    const int saved = dup(1);
    const int null_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (saved < 0 || null_fd < 0 || dup2(null_fd, 1) < 0) {
        if (null_fd >= 0) close(null_fd);
        if (saved >= 0) close(saved);
        return -1;
    }
    close(null_fd);
#endif
    return saved;
}

void stdout_restore(int saved) {
    if (saved < 0) return;
#if defined(_WIN32)
    _dup2(saved, 1);
    _close(saved);
#else
    dup2(saved, 1);
    close(saved);
#endif
}

}  // namespace strata::platform
