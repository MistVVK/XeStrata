#include "strata/tp/nccl_session.hpp"
#include <stdexcept>
#include <thread>

namespace strata::tp {
namespace {
void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
}

NcclSession::NcclSession(MpiRuntime& runtime, std::chrono::milliseconds timeout)
    : rank_(runtime.rank()), size_(runtime.size()), timeout_(timeout) {
    if (timeout.count() <= 0) throw std::runtime_error("NCCL timeout must be positive");
    int version = 0;
    ready(ncclGetVersion(&version), "NCCL version");
    if (version != NCCL_VERSION_CODE) throw std::runtime_error("NCCL runtime and headers differ");
    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices), "local GPU count");
    if (runtime.local_rank() >= devices) throw std::runtime_error("MPI requires one distinct GPU per local rank");
    cuda_check(cudaSetDevice(runtime.local_rank()), "select local GPU");
    ncclUniqueId id{};
    if (rank_ == 0) ready(ncclGetUniqueId(&id), "NCCL unique ID");
    runtime.broadcast(std::as_writable_bytes(std::span(&id, 1)));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    try {
        ready(ncclCommInitRankConfig(&comm_, size_, id, rank_, &config), "NCCL initialize rank");
    } catch (...) { abort(); throw; }
}

NcclSession::~NcclSession() { abort(); }

void NcclSession::abort() noexcept {
    if (comm_) {
        auto comm = comm_;
        comm_ = nullptr;
        ncclCommAbort(comm);
    }
}

void NcclSession::ready(ncclResult_t result, const char* operation) {
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (result == ncclInProgress) {
        if (std::chrono::steady_clock::now() >= deadline) {
            abort();
            throw std::runtime_error(std::string(operation) + ": timeout");
        }
        const auto status = ncclCommGetAsyncError(comm_, &result);
        if (status != ncclSuccess) result = status;
        if (result == ncclInProgress) std::this_thread::yield();
    }
    if (result != ncclSuccess) {
        const auto error = std::string(operation) + ": " + ncclGetErrorString(result);
        abort();
        throw std::runtime_error(error);
    }
}

void NcclSession::sum(float* data, size_t count, cudaStream_t stream) {
    if (!data || !count) throw std::runtime_error("TP sum requires a nonempty device buffer");
    if (size_ > 1) ready(ncclAllReduce(data, data, count, ncclFloat, ncclSum, comm_, stream), "NCCL sum");
}

void NcclSession::broadcast_bytes(void* data, size_t bytes, cudaStream_t stream) {
    if (size_ > 1) ready(ncclBroadcast(data, data, bytes, ncclUint8, 0, comm_, stream), "NCCL broadcast");
}

void NcclSession::send(const float* data, size_t count, int peer, cudaStream_t stream) {
    ready(ncclSend(data, count, ncclFloat, peer, comm_, stream), "NCCL send");
}

void NcclSession::receive(float* data, size_t count, int peer, cudaStream_t stream) {
    ready(ncclRecv(data, count, ncclFloat, peer, comm_, stream), "NCCL receive");
}

void NcclSession::gather(const float* input, float* output, size_t count, cudaStream_t stream) {
    if (!input || !output || !count) throw std::runtime_error("TP gather requires nonempty device buffers");
    if (size_ == 1) {
        cuda_check(cudaMemcpyAsync(output, input, count * sizeof(float), cudaMemcpyDeviceToDevice, stream), "reference gather");
        return;
    }
    ready(ncclAllGather(input, output, count, ncclFloat, comm_, stream), "NCCL gather");
}

void NcclSession::exchange(const float* input, float* output, size_t stride, size_t offset, size_t count, cudaStream_t stream) {
    if (!input || !count || count > stride || offset > stride - count || (size_ > 1 && !output))
        throw std::runtime_error("TP exchange has invalid peer blocks");
    if (size_ == 1) return;
    ready(ncclGroupStart(), "NCCL exchange group");
    ncclResult_t submission = ncclSuccess;
    for (int step = 1; step < size_; ++step) {
        const int peer = (rank_ + step) % size_;
        for (auto result : {ncclSend(input + size_t(peer) * stride + offset, count, ncclFloat, peer, comm_, stream),
                            ncclRecv(output + size_t(step - 1) * stride + offset, count, ncclFloat, peer, comm_, stream)})
            if (result != ncclSuccess && result != ncclInProgress) submission = result;
    }
    const auto result = ncclGroupEnd();
    ready(submission, "NCCL exchange submission");
    ready(result, "NCCL exchange group end");
}

void NcclSession::synchronize(cudaStream_t stream) {
    cudaEvent_t event;
    cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "TP event create");
    try {
        cuda_check(cudaEventRecord(event, stream), "TP event record");
        const auto deadline = std::chrono::steady_clock::now() + timeout_;
        for (;;) {
            const auto query = cudaEventQuery(event);
            if (query == cudaSuccess) break;
            if (query != cudaErrorNotReady) cuda_check(query, "TP stream");
            ncclResult_t state;
            ready(ncclCommGetAsyncError(comm_, &state), "NCCL stream status");
            ready(state, "NCCL stream");
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("TP stream: timeout");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    } catch (...) {
        // Abort before callers release buffers still owned by an unfinished collective.
        abort();
        cudaEventDestroy(event);
        throw;
    }
    cuda_check(cudaEventDestroy(event), "TP event destroy");
}
} // namespace strata::tp
