// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/platform/mapped_file.hpp - a file mapped read-only (a safetensors file's weights read in place, as the
// GGUF reader maps its files), and a way to drop a range of it from the page cache once it is no longer read.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::platform {

class MappedFile {
public:
    explicit MappedFile(const std::string& path) : path_(path) {
#ifdef _WIN32
        HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER li{};
        GetFileSizeEx(h, &li);
        size_ = (uint64_t) li.QuadPart;
        HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m) {
            CloseHandle(h);
            throw std::runtime_error("CreateFileMapping failed: " + path);
        }
        base_ = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base_) {
            CloseHandle(m);
            CloseHandle(h);
            throw std::runtime_error("MapViewOfFile failed: " + path);
        }
        map_ = m;
        file_ = h;
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("fstat failed: " + path);
        }
        size_ = (uint64_t) st.st_size;
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("mmap failed: " + path);
        }
        base_ = (const uint8_t*) p;
#endif
    }
    ~MappedFile() {
#ifdef _WIN32
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle((HANDLE) map_);
        if (file_) CloseHandle((HANDLE) file_);
#else
        if (base_) munmap((void*) base_, size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return base_; }
    uint64_t size() const { return size_; }

    /// Drops bytes off .. off + n of the file from the page cache, the mapping's pages first (a mapped page stays in
    /// the cache): Linux, madvise and posix_fadvise's DONTNEED; Windows nothing, its cache keeps the pages until the
    /// memory is wanted.  The range is read again from the file if touched after.
    void drop_cache(uint64_t off, uint64_t n) const {
#ifndef _WIN32
        const auto page = (uint64_t) sysconf(_SC_PAGESIZE);
        const uint64_t a = off / page * page, e = std::min(size_, off + n);
        if (e > a) madvise((void*) (base_ + a), e - a, MADV_DONTNEED);
        posix_fadvise(fd_, (off_t) off, (off_t) n, POSIX_FADV_DONTNEED);
#else
        (void) off;
        (void) n;
#endif
    }

private:
    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
#ifdef _WIN32
    void* map_ = nullptr;
    void* file_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace strata::platform
