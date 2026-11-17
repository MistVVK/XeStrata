// What an engine made on the cards and in page-locked memory (buffers, streams, events, handles), each with how to
// release it; ~Engine releases them, newest first, so a lab process can make and drop engines around one loaded model.
// Every model's engine registers through own(): save_state and load_state make page-locked stages on the cards'
// threads at once, which unguarded pushes onto the one list would race on (engine_common.cu holds the list).
#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace fnx {

void own(std::function<void()> release);
// the releases, for ~Engine to run once no other thread can register (the cards are idle)
std::vector<std::function<void()>> take_owned();
// the current card, for a release that has to run on it; and the releases of what is made most often
inline int current_card() {
    int d = -1;
    cudaGetDevice(&d);
    return d;
}
inline void own_dev(void* p) {
    const int d = current_card();
    own([d, p] { cudaSetDevice(d); cudaFree(p); });
}
inline void own_pinned(void* p) { own([p] { cudaFreeHost(p); }); }
inline void own_stream(cudaStream_t s) {
    const int d = current_card();
    own([d, s] { cudaSetDevice(d); cudaStreamDestroy(s); });
}
inline void own_event(cudaEvent_t e) {
    const int d = current_card();
    own([d, e] { cudaSetDevice(d); cudaEventDestroy(e); });
}

}  // namespace fnx
