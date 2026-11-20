// hy4_io.h - sequential, unbuffered 4 KiB-aligned file reads for the 62 GB .strata (Windows ReadFile NO_BUFFERING, POSIX pread fallback)
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hy4 {

class AlignedFile {
public:
    ~AlignedFile() { close(); }
    bool open(const std::string& path, bool unbuffered = true) {
#ifdef _WIN32
        DWORD fl = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN | (unbuffered ? FILE_FLAG_NO_BUFFERING : 0);
        h_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, fl, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz;
        GetFileSizeEx(h_, &sz);
        size_ = (uint64_t)sz.QuadPart;
        return true;
#else
        direct_ = false;
        fadv_dontneed_ = false;
        if (unbuffered) {
#ifdef O_DIRECT
            fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd_ >= 0) direct_ = true;
#endif
        }
        if (fd_ < 0) {
            fd_ = ::open(path.c_str(), O_RDONLY);
            if (fd_ < 0) return false;
            if (unbuffered) fadv_dontneed_ = true;
        }
        size_ = (uint64_t)lseek(fd_, 0, SEEK_END);
        return true;
#endif
    }
    void close() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE && h_) CloseHandle(h_);
        h_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }
    uint64_t size() const { return size_; }
    // off and n must be multiples of 4096 when unbuffered (n is clipped at EOF by the OS); dst must be 4096-aligned. Returns bytes read.
    size_t read_at(uint64_t off, void* dst, size_t n) const {
#ifdef _WIN32
        size_t got = 0;
        while (got < n) {
            OVERLAPPED ov{};
            uint64_t o = off + got;
            ov.Offset = (DWORD)(o & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD)(o >> 32);
            DWORD chunk = (DWORD)((n - got) > (1u << 30) ? (1u << 30) : (n - got));
            DWORD r = 0;
            if (!ReadFile(h_, (char*)dst + got, chunk, &r, &ov)) {
                if (GetLastError() == ERROR_HANDLE_EOF) break;
                if (r == 0) break;
            }
            if (r == 0) break;
            got += r;
        }
        return got;
#else
        size_t got = 0;
        while (got < n) {
            ssize_t r = pread(fd_, (char*)dst + got, n - got, (off_t)(off + got));
            if (r < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (r == 0) break;
            got += (size_t)r;
        }
#if defined(POSIX_FADV_DONTNEED)
        if (fadv_dontneed_ && got > 0) posix_fadvise(fd_, (off_t)off, (off_t)got, POSIX_FADV_DONTNEED);
#endif
        return got;
#endif
    }

private:
    uint64_t size_ = 0;
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
    bool direct_ = false;
    bool fadv_dontneed_ = false;
#endif
};

}  // namespace hy4
