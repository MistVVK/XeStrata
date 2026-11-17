// The routed experts' matrices and the prompt path's jobs, shared by kernels.cu (moe.cuh) and moe_mma.cu.
struct MoeW {
    const char* wa;      // the experts' matrices, expert stride sa
    size_t sa;
    size_t row_a;        // row bytes
    int nrows, ncols;
    const char* const* ep;   // or each expert's matrix (VRAM or mapped host memory): DW::ep
};
__device__ __forceinline__ const char* moe_expert(const MoeW& w, int e) { return w.ep ? w.ep[e] : w.wa + (size_t) e * w.sa; }

// a BigJob in shared memory (thread 0 can't copy 520 bytes alone without stalling the block)
__device__ __forceinline__ void load_bigjob(BigJob& jb, const BigJob* src) {
    for (int k = threadIdx.x; k < (int) (sizeof(BigJob) / 4); k += blockDim.x)
        reinterpret_cast<int*>(&jb)[k] = reinterpret_cast<const int*>(src)[k];
    __syncthreads();
}
