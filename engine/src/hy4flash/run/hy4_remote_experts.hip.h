#pragma once

#include <array>
#include <chrono>
#include <stdexcept>
#include "hy4_gpu_model.hip.h"
#include "skhy4.hip.h"
#include "strata/tp/nccl_session.hpp"

namespace hy4 {

// The baseline sums the weighted contributions in router slot order. Keep that
// order even when adjacent slots were evaluated on different ranks.
__global__ void k_moe_reduce_remote(const float* local, const float* remote, float* output,
                                    const ExpertDev* expert, const int32_t* idx) {
    const size_t first = size_t(blockIdx.x) * MOE_TOPK;
    __shared__ uint8_t worker_slot[MOE_TOPK];
    if (threadIdx.x < MOE_TOPK) worker_slot[threadIdx.x] = expert[idx[first + threadIdx.x]].pad != 0;
    __syncthreads();
    for (int col = threadIdx.x; col < MOE_D; col += blockDim.x) {
        float sum = 0.f;
#pragma unroll
        for (int slot = 0; slot < MOE_TOPK; ++slot) {
            const auto* source = worker_slot[slot] ? remote : local;
            sum += source[(first + slot) * MOE_D + col];
        }
        output[size_t(blockIdx.x) * MOE_D + col] = sum;
    }
}

// The coordinator owns the request and all its state. The peer consumes the
// same routed expert inputs as a local peer GPU, and returns its weighted slots.
class RemoteExperts {
public:
    explicit RemoteExperts(strata::tp::MpiRuntime& mpi)
        : mpi_(mpi), channel_(mpi, std::chrono::seconds(60)) {
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&inputs_, cudaEventDisableTiming));
    }
    ~RemoteExperts() {
        channel_.abort();
        cudaFree(result_);
        cudaEventDestroy(inputs_);
        cudaStreamDestroy(stream_);
    }

    void start(int rows) {
        check(cudaMalloc(&result_, slot_values(rows) * sizeof(float)));
        // DFly2 switches the coordinator to the decode kernels for verify
        // windows. Both ranks must execute the same expert math for those rows.
        control(Op::start, skd::moe_dec_max_m(), rows);
        mpi_.minimum(1);  // Both ranks have allocated their workspaces before READY.
    }

    void submit(int layer, int rows, __half* z, int32_t* idx, float* weight, cudaStream_t compute) try {
        check(cudaEventRecord(inputs_, compute));
        check(cudaStreamWaitEvent(stream_, inputs_, 0));
        control(Op::run, layer, rows);
        transfer(z, idx, weight, rows);
        channel_.receive(result_, slot_values(rows), 1, stream_);
    } catch (const std::exception& failure) {
        mpi_.fail(failure.what());
    }

    void wait() try {
        channel_.synchronize(stream_);
    } catch (const std::exception& failure) {
        mpi_.fail(failure.what());
    }

    // Stable destination for the coordinator's captured reduction kernels.
    const float* result() const { return result_; }

    void finish() { control(Op::stop); }

    void work(const std::string& model) {
        GpuModel g;
        g.expert_shard = 1;
        std::string err;
        if (!g.rd.open(model, err)) throw std::runtime_error(err);
        size_t free_bytes = 0, total = 0;
        gpu_mem_avail(free_bytes, total);
        constexpr size_t reserve = 6ull << 30;
        if (!load_expert_shard(g, model, free_bytes > reserve ? free_bytes - reserve : 0, err))
            throw std::runtime_error(err);
        const auto start = control();
        const uint32_t max_rows = start[2];
        const uint32_t decode_rows = start[1];
        if (start[0] != uint32_t(Op::start) || decode_rows == 0 || decode_rows > 16 ||
            max_rows == 0 || max_rows > INT32_MAX / MOE_TOPK)
            throw std::runtime_error("invalid Hy4 expert worker start");
        skd::moe_dec_max_m() = int(decode_rows);

        std::vector<ExpertDev> host(g.experts.size());
        for (size_t i = 0; i < host.size(); ++i) {
            const auto& e = g.experts[i];
            host[i].body_off = e.body_off;
            host[i].dn_off = e.dn_off;
            host[i].gu_code = e.gu_code;
            host[i].dn_code = e.dn_code;
            host[i].pad = e.tier != 0;
        }
        ExpertDev* experts = nullptr;
        __half* z = nullptr;
        int32_t* idx = nullptr;
        float* weight = nullptr;
        void* workspace = nullptr;
        check(cudaMalloc(&experts, host.size() * sizeof(ExpertDev)));
        check(cudaMemcpy(experts, host.data(), host.size() * sizeof(ExpertDev), cudaMemcpyHostToDevice));
        check(cudaMalloc(&z, size_t(max_rows) * 1536 * sizeof(__half)));
        check(cudaMalloc(&idx, size_t(max_rows) * 16 * sizeof(int32_t)));
        check(cudaMalloc(&weight, size_t(max_rows) * 16 * sizeof(float)));
        check(cudaMalloc(&workspace, moe_ws_bytes(max_rows)));
        skd::moe_has_cpu() = 1;  // Each rank evaluates only the slots it owns.
        mpi_.minimum(1);
        for (;;) {
            const auto cmd = control();
            if (cmd[0] == uint32_t(Op::stop)) break;
            const uint32_t layer = cmd[1], rows = cmd[2];
            if (cmd[0] != uint32_t(Op::run) ||
                layer < 1 || layer > 41 || rows < 1 || rows > max_rows)
                throw std::runtime_error("invalid Hy4 expert worker request");
            transfer(z, idx, weight, rows);
            auto* contributions = skd::moe_ws_split(workspace, rows * MOE_TOPK).y;
            check(cudaMemsetAsync(contributions, 0, slot_values(rows) * sizeof(float), stream_));
            moe_experts(stream_, g.d_sec4, experts + size_t(layer - 1) * 512, z, idx, weight, nullptr, rows, workspace);
            channel_.send(contributions, slot_values(rows), 0, stream_);
            channel_.synchronize(stream_);
        }
        for (void* buffer : {static_cast<void*>(experts), static_cast<void*>(z), static_cast<void*>(idx),
                             static_cast<void*>(weight), workspace, static_cast<void*>(g.d_sec4)}) cudaFree(buffer);
    }

private:
    enum class Op : uint32_t { start = 1, run, stop };

    static size_t slot_values(size_t rows) { return rows * MOE_TOPK * MOE_D; }

    std::array<uint32_t, 3> control(Op op = Op::stop, uint32_t layer = 0, uint32_t rows = 0) {
        std::array<uint32_t, 3> message{uint32_t(op), layer, rows};
        mpi_.broadcast_control(message);
        return message;
    }

    static void check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }

    void transfer(__half* z, int32_t* idx, float* weight, size_t rows) {
        channel_.broadcast_bytes(z, rows * 1536 * sizeof(__half), stream_);
        channel_.broadcast_bytes(idx, rows * 16 * sizeof(int32_t), stream_);
        channel_.broadcast_bytes(weight, rows * 16 * sizeof(float), stream_);
    }

    strata::tp::MpiRuntime& mpi_;
    strata::tp::NcclSession channel_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t inputs_ = nullptr;
    float* result_ = nullptr;
};

inline std::string expert_model_identity(const std::string& model) {
    strata::artifact::StrataFileReader reader;
    std::string err;
    if (!reader.open(model, err)) throw std::runtime_error(err);
    std::string id = "strata-hy4-experts-v3:";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto& section : reader.header().sections)
        for (uint8_t b : section.sha256) { id += hex[b >> 4]; id += hex[b & 15]; }
    return id;
}

} // namespace hy4
