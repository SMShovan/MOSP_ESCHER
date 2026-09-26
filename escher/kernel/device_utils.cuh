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

// In-order rank (0-based index into the sorted keys) of the node stored at
// heap position `pos` of a complete binary search tree with n nodes (the
// CBST layout). (2*(pos+1-2^k)+1) * 2^log2_n / 2^k is evaluated as a 64-bit
// shift: the int32 product of the original formula overflowed for
// n > 65,535, which broke the BST order (keys unreachable) and produced
// negative ranks that indexed the key / offset arrays out of bounds.
static inline __device__ int cbstRankOfPosition(int pos, int n) {
    int k = floor_log2(pos + 1);
    int log2_n = floor_log2(n);
    long long index = (2LL * (pos + 1 - (1LL << k)) + 1) << (log2_n - k);
    long long index2 =
        min(index, index - (index / 2) + (n + 1 - (1LL << log2_n)));
    return static_cast<int>(index2 - 1);
}
