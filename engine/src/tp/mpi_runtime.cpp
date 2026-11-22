#include "strata/tp/mpi_runtime.hpp"
#include "strata/tp/limits.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::tp {
MpiRuntime::MpiRuntime(int& argc, char**& argv, std::chrono::milliseconds timeout) : timeout_(timeout) {
    if (timeout.count() <= 0) throw std::runtime_error("MPI timeout must be positive");
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        throw std::runtime_error("MPI initialization failed");
    check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI error handler");
    check(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI rank");
    check(MPI_Comm_size(MPI_COMM_WORLD, &size_), "MPI size");
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide funneled thread support");
    MPI_Comm local;
    check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &local), "MPI node group");
    check(MPI_Comm_rank(local, &local_rank_), "MPI local rank");
    check(MPI_Comm_free(&local), "MPI node group release");
}

MpiRuntime::~MpiRuntime() {
    // An unfinished rank cannot leave other ranks waiting on its collectives.
    if (!finished_) MPI_Abort(MPI_COMM_WORLD, 1);
}

[[noreturn]] void MpiRuntime::fail(const std::string& message) {
    std::fprintf(stderr, "distributed rank %d: %s\n", rank_, message.c_str());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void MpiRuntime::check(int result, const char* operation) {
    if (result == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]; int length = 0;
    MPI_Error_string(result, message, &length);
    fail(std::string(operation) + ": " + std::string(message, length));
}

void MpiRuntime::wait(MPI_Request& request, const char* operation, bool idle, bool short_poll) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + timeout_;
    for (;;) {
        int completed = 0;
        check(MPI_Test(&request, &completed, MPI_STATUS_IGNORE), operation);
        if (completed) return;
        const auto now = std::chrono::steady_clock::now();
        if (!idle && now >= deadline) fail(std::string(operation) + ": timeout");
        if (short_poll && now - start < std::chrono::milliseconds(2)) std::this_thread::yield();
        else std::this_thread::sleep_for(idle ? std::chrono::microseconds(1000) : std::chrono::microseconds(50));
    }
}

void MpiRuntime::broadcast(std::span<std::byte> bytes) {
    if (bytes.empty() || bytes.size() > max_prompt_bytes) fail("broadcast payload must contain 1 byte..8 MiB");
    MPI_Request request;
    check(MPI_Ibcast(bytes.data(), int(bytes.size()), MPI_BYTE, 0, MPI_COMM_WORLD, &request), "MPI bootstrap broadcast");
    wait(request, "MPI bootstrap broadcast");
}

void MpiRuntime::broadcast_control(std::span<uint32_t> words) {
    if (words.empty() || words.size() > 16) fail("control message must contain 1..16 words");
    MPI_Request request;
    check(MPI_Ibcast(words.data(), int(words.size()), MPI_UINT32_T, 0, MPI_COMM_WORLD, &request), "MPI control");
    wait(request, "MPI control", rank_ != 0, true);
}

std::string MpiRuntime::command(const std::string& rank_zero_text) {
    return message(rank_zero_text, true);
}

std::string MpiRuntime::message(const std::string& rank_zero_text, bool idle) {
    std::array<uint64_t, 2> header{++sequence_, rank_zero_text.size()};
    if (rank_ == 0 && (rank_zero_text.empty() || rank_zero_text.size() > max_prompt_bytes)) fail("command must contain 1 byte..8 MiB");
    MPI_Request request;
    check(MPI_Ibcast(header.data(), int(header.size()), MPI_UINT64_T, 0, MPI_COMM_WORLD, &request), "MPI command header");
    wait(request, "MPI command header", idle && rank_ != 0);
    if (header[0] != sequence_ || !header[1] || header[1] > max_prompt_bytes) fail("command sequence or length mismatch");
    std::string text = rank_ == 0 ? rank_zero_text : std::string(size_t(header[1]), '\0');
    broadcast(std::as_writable_bytes(std::span(text)));
    return text;
}

void MpiRuntime::agree(const std::string& identity) {
    if (identity.empty() || identity.size() > 4096) fail("workload identity must contain 1..4096 bytes");
    const auto root = message(rank_ == 0 ? identity : "", false);
    int different = root != identity, any = 0;
    MPI_Request request;
    check(MPI_Iallreduce(&different, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD, &request), "MPI workload agreement");
    wait(request, "MPI workload agreement");
    if (any) fail("workload identity mismatch");
}

uint64_t MpiRuntime::minimum(uint64_t local) {
    uint64_t result = 0;
    MPI_Request request;
    check(MPI_Iallreduce(&local, &result, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD, &request), "MPI minimum");
    wait(request, "MPI minimum");
    return result;
}

void MpiRuntime::gather(std::span<const std::byte> local, std::span<std::byte> result, std::span<const int> sizes) {
    if (sizes.size() != size_t(size_)) fail("gather needs one byte count per rank");
    std::vector<int> offsets(size_);
    size_t total = 0;
    for (int rank = 0; rank < size_; ++rank) {
        if (sizes[rank] < 0 || size_t(sizes[rank]) > 128 * 1024 * 1024 - total) fail("gather exceeds 128 MiB");
        offsets[rank] = int(total); total += sizes[rank];
    }
    if (local.size() != size_t(sizes[rank_]) || result.size() != total) fail("gather byte count mismatch");
    const void* send = result.data() && local.data() == result.data() + offsets[rank_] ? MPI_IN_PLACE : local.data();
    MPI_Request request;
    check(MPI_Iallgatherv(send, int(local.size()), MPI_BYTE, result.data(), sizes.data(), offsets.data(), MPI_BYTE,
                         MPI_COMM_WORLD, &request), "MPI gather");
    wait(request, "MPI gather");
}

void MpiRuntime::finish() {
    MPI_Request request;
    check(MPI_Ibarrier(MPI_COMM_WORLD, &request), "MPI shutdown barrier");
    wait(request, "MPI shutdown barrier");
    const int result = MPI_Finalize();
    finished_ = true;
    if (result != MPI_SUCCESS) std::abort();
}
} // namespace strata::tp
