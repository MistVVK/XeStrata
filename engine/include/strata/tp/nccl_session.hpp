#pragma once

#include "strata/tp/mpi_runtime.hpp"
#include <cuda_runtime_api.h>
#include <nccl.h>

namespace strata::tp {
class NcclSession {
public:
    NcclSession(MpiRuntime&, std::chrono::milliseconds timeout);
    ~NcclSession();
    NcclSession(const NcclSession&) = delete;
    NcclSession& operator=(const NcclSession&) = delete;
    void sum(float* data, size_t count, cudaStream_t stream);
    void broadcast_bytes(void* data, size_t bytes, cudaStream_t stream);
    void send(const float* data, size_t count, int peer, cudaStream_t stream);
    void receive(float* data, size_t count, int peer, cudaStream_t stream);
    void gather(const float* input, float* output, size_t count, cudaStream_t stream);
    // Peer blocks land in cyclic source order, excluding this rank's own block.
    void exchange(const float* input, float* output, size_t stride, size_t offset, size_t count, cudaStream_t stream);
    void synchronize(cudaStream_t stream);
    void abort() noexcept;
    int rank() const { return rank_; }
    int size() const { return size_; }
private:
    ncclComm_t comm_ = nullptr;
    int rank_, size_;
    std::chrono::milliseconds timeout_;
    void ready(ncclResult_t, const char* operation);
};
} // namespace strata::tp
