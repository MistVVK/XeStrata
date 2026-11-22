#include "strata/tp/nccl_session.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Device {
    float* data = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(&data, count * sizeof(float))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
void equal(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual != expected) throw std::runtime_error("NCCL collective changed tensor elements");
}

void collectives(strata::tp::NcclSession& session, cudaStream_t stream) {
    for (const size_t count : {size_t(1), size_t(2560), size_t(40960), size_t(1024 * 1024)}) {
        Device input(count), gathered(count * session.size());
        std::vector<float> host(count), expected(count), result(count * session.size());
        for (size_t i = 0; i < count; ++i) {
            host[i] = float(int(i % 97) - 48 + session.rank());
            expected[i] = float(session.size() * (int(i % 97) - 48) + session.size() * (session.size() - 1) / 2);
        }
        check(cudaMemcpyAsync(input.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
        session.sum(input.data, count, stream);
        check(cudaMemcpyAsync(host.data(), input.data, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        session.synchronize(stream);
        equal(host, expected);
        std::fill(host.begin(), host.end(), float(session.rank() + 1));
        check(cudaMemcpyAsync(input.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
        session.gather(input.data, gathered.data, count, stream);
        check(cudaMemcpyAsync(result.data(), gathered.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        session.synchronize(stream);
        for (int rank = 0; rank < session.size(); ++rank)
            for (size_t i = 0; i < count; ++i)
                if (result[size_t(rank) * count + i] != float(rank + 1)) throw std::runtime_error("NCCL gather changed rank order");
    }
}

void expert_exchange(strata::tp::NcclSession& session, cudaStream_t stream) {
    if (session.size() != 2) return;
    for (const size_t rows : {size_t(1), size_t(8), size_t(512)}) {
        const size_t count = rows * 1536;
        Device input(count), partial(count);
        std::vector<float> host(count), expected(count);
        for (size_t i = 0; i < count; ++i) {
            expected[i] = float(i % 135);
            host[i] = session.rank() == 0 ? expected[i] : -1.f;
        }
        check(cudaMemcpyAsync(input.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
        session.broadcast_bytes(input.data, count * sizeof(float), stream);
        if (session.rank() == 1) {
            session.send(input.data, count, 0, stream);
        } else {
            session.receive(partial.data, count, 1, stream);
        }
        auto* source = session.rank() == 0 ? partial.data : input.data;
        check(cudaMemcpyAsync(host.data(), source, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        session.synchronize(stream);
        equal(host, expected);
    }
}

void exchange(strata::tp::NcclSession& session, cudaStream_t stream, bool captured) {
    constexpr size_t count = 2560, offset = 2, stride = count + 5;
    Device input(stride * session.size()), output(stride * session.size());
    std::vector<float> host(stride * session.size()), result(host.size()), expected(host.size(), -9.f);
    for (int peer = 0; peer < session.size(); ++peer)
        for (size_t i = 0; i < stride; ++i) host[size_t(peer) * stride + i] = float(session.rank() * 1000 + peer * 10 + int(i % 97));
    for (int step = 1; step < session.size(); ++step) {
        const int peer = (session.rank() + step) % session.size();
        for (size_t i = offset; i < offset + count; ++i)
            expected[size_t(step - 1) * stride + i] = float(peer * 1000 + session.rank() * 10 + int(i % 97));
    }
    Graph graph;
    if (captured) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        session.exchange(input.data, output.data, stride, offset, count, stream);
        check(cudaStreamEndCapture(stream, &graph.graph));
        check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    }
    for (int replay = 0; replay < (captured ? 3 : 1); ++replay) {
        check(cudaMemcpyAsync(input.data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        std::fill(result.begin(), result.end(), -9.f);
        check(cudaMemcpyAsync(output.data, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        if (captured) check(cudaGraphLaunch(graph.executable, stream));
        else session.exchange(input.data, output.data, stride, offset, count, stream);
        check(cudaMemcpyAsync(result.data(), output.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        session.synchronize(stream);
        equal(result, expected);
    }
}

void graph_replay(strata::tp::NcclSession& session, cudaStream_t stream) {
    constexpr size_t count = 2560;
    constexpr int sums = 4;
    Device input(count), gathered(count * session.size());
    std::vector<float> host(count), output(count * session.size()), expected(output.size());
    Graph graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        for (int i = 0; i < sums; ++i) session.sum(input.data, count, stream);
        session.gather(input.data, gathered.data, count, stream);
    } catch (...) {
        cudaStreamEndCapture(stream, &graph.graph);
        throw;
    }
    check(cudaStreamEndCapture(stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
        std::fill(host.begin(), host.end(), float(session.rank() + 1 + replay));
        float sum = float(session.size() * (session.size() + 1) / 2 + replay * session.size());
        for (int i = 1; i < sums; ++i) sum *= session.size();
        std::fill(expected.begin(), expected.end(), sum);
        check(cudaMemcpyAsync(input.data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(graph.executable, stream));
        check(cudaMemcpyAsync(output.data(), gathered.data, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        session.synchronize(stream);
        equal(output, expected);
    }
}
} // namespace

int main(int argc, char** argv) try {
    strata::tp::MpiRuntime runtime(argc, argv, std::chrono::seconds(30));
    try {
        runtime.agree("nccl-runtime-test-v2;four-sizes;three-graph-replays;strided-peer-exchange");
        {
            strata::tp::NcclSession session(runtime, std::chrono::seconds(30));
            Stream stream;
            collectives(session, stream.value);
            expert_exchange(session, stream.value);
            graph_replay(session, stream.value);
            exchange(session, stream.value, false);
            exchange(session, stream.value, true);
            runtime.agree("nccl-test-completed");
            std::printf("NCCL runtime rank=%d world=%d sum=1 gather=1 exchange=1 graph_replay=3 correct=1\n", runtime.rank(), runtime.size());
        }
        runtime.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NCCL rank %d: %s\n", runtime.rank(), error.what());
        return 1;
    }
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
