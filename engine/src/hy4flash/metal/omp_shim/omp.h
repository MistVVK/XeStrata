// omp.h shim for builds without OpenMP (Apple clang has no libomp): the oracle's "#pragma omp" lines are ignored (single thread), these calls become stubs.
#pragma once
inline void omp_set_num_threads(int) {}
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
