#pragma once

// Common device/host helpers used by kernels

// Portable atomic add for signed long long (not all CUDA versions provide atomicAdd(long long*, long long))
static inline __device__ long long atomicAdd_sll(long long* address, long long val) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    (unsigned long long)((long long)assumed + val));
  } while (assumed != old);
  return (long long)old;
}

static inline __host__ __device__ int nextMultipleOf32(int num) {
    return ((num + 32) / 32) * 32;
}

static inline __host__ __device__ int nextMultipleOf4(int num) {
    if (num == 0) return 0;
    return ((num + 4) / 4) * 4;
}

static inline __device__ int ceil_log2(int x) {
    int log = 0;
    while ((1 << log) < x) ++log;
    return log;
}

static inline __device__ int floor_log2(int x) {
    int log = 0;
    while (x >>= 1) ++log;
    return log;
}


