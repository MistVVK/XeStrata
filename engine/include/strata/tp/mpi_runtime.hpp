#pragma once

#include <mpi.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace strata::tp {
// One runtime per process. GPU objects must be destroyed before finish().
class MpiRuntime {
public:
    MpiRuntime(int& argc, char**& argv, std::chrono::milliseconds timeout);
    ~MpiRuntime();
    MpiRuntime(const MpiRuntime&) = delete;
    MpiRuntime& operator=(const MpiRuntime&) = delete;
    int rank() const { return rank_; }
    int size() const { return size_; }
    int local_rank() const { return local_rank_; }
    void agree(const std::string& identity);
    void broadcast(std::span<std::byte> bytes);
    void broadcast_control(std::span<uint32_t> words);
    std::string command(const std::string& rank_zero_text);
    uint64_t minimum(uint64_t local);
    void gather(std::span<const std::byte> local, std::span<std::byte> result, std::span<const int> sizes);
    void finish();
    [[noreturn]] void fail(const std::string& message);
private:
    int rank_ = -1, size_ = 0, local_rank_ = -1;
    bool finished_ = false;
    uint64_t sequence_ = 0;
    std::chrono::milliseconds timeout_;
    void check(int result, const char* operation);
    std::string message(const std::string& rank_zero_text, bool idle);
    void wait(MPI_Request&, const char* operation, bool idle = false, bool short_poll = false);
};
} // namespace strata::tp
