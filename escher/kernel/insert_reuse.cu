#include "device_utils.cuh"
#include "kernels.cuh"
#include <climits>

// Phase 1: Read-only order-statistic tree walk.
// Each thread finds the k-th deleted node and writes its array position.
// No modification to avail[] or nodes[], so concurrent reads are safe.
__global__ void locateReusableSlots(int *subtreeAvail, int *avail,
                                    int numRecords, int *outPositions, int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= K)
    return;
  int k = tid + 1; // 1-based order statistic
  int idx = 0;
  while (idx < numRecords) {
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    int leftCount = (left < numRecords) ? subtreeAvail[left] : 0;
    int self = avail[idx];
    if (k <= leftCount) {
      idx = left;
      continue;
    }
    if (self == 1 && k == leftCount + 1) {
      break;
    }
    k -= leftCount + self;
    idx = right;
  }
  outPositions[tid] = (idx < numRecords) ? idx : -1;
}

// ── Best-fit metadata extraction ────────────────────────────────────────

// Extract usable capacity for each located deleted slot.
// capacity = node->length - 1 (last position reserved for INT_MIN sentinel).
__global__ void extractSlotCapacities(CBSTNode *nodes, int *positions,
                                      int *outCapacities, int D) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= D)
    return;
  int pos = positions[tid];
  if (pos >= 0) {
    outCapacities[tid] = nodes[pos].length - 1;
  } else {
    outCapacities[tid] = 0;
  }
}

// Compute per-item payload size from prefix-sum array.
__global__ void computeItemSizes(int *prefixSizes, int *outSizes, int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= K)
    return;
  outSizes[tid] =
      (tid == 0) ? prefixSizes[0] : prefixSizes[tid] - prefixSizes[tid - 1];
}

// ── GPU-parallel best-fit matching kernels ──────────────────────────────

// Binary search: for each sorted item, find first slot (in sorted capacity
// order) with capacity >= item size.  Equivalent to std::lower_bound.
__global__ void lowerBoundKernel(int *sortedCaps, int D, int *sortedSizes,
                                 int M, int *outLo) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= M)
    return;
  int target = sortedSizes[tid];
  int lo = 0, hi = D;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (sortedCaps[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  outLo[tid] = lo;
}

// In-place transform: b[i] = lo[i] - i
__global__ void computeBInPlace(int *lo, int M) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < M)
    lo[tid] -= tid;
}

// assigned[i] = i + prefixMax[i]
__global__ void computeAssigned(int *prefixMax, int *assigned, int M) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < M)
    assigned[tid] = tid + prefixMax[tid];
}

// Recover the original key for each located deleted-slot position.
// Uses the same CBST layout formula as storeItemsIntoNodes (inverse mapping:
// array position → in-order rank → d_keys[rank]).
__global__ void extractKeysFromPositions(int *d_keys, int *positions,
                                         int *outKeys, int numRecords, int D) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= D)
    return;
  int pos = positions[tid];
  if (pos < 0) {
    outKeys[tid] = 0;
    return;
  }
  outKeys[tid] = d_keys[cbstRankOfPosition(pos, numRecords)];
}

// Pairs each matched item with its slot. Items are sorted by size and slots
// by capacity; assigned[i] (the slot rank of sorted item i) is strictly
// increasing and >= the first slot that fits, so the matched items are
// exactly sorted items 0..matchCount-1 and every one fits its slot.
__global__ void pairMatches(const int *itemOrder, const int *assigned,
                            const int *slotOrder, int matchCount,
                            int *matchedItemIndices, int *matchedSlotIndices) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= matchCount)
    return;
  matchedItemIndices[tid] = itemOrder[tid];
  matchedSlotIndices[tid] = slotOrder[assigned[tid]];
}

// ── Phase 2: Apply reuse ────────────────────────────────────────────────
// Best-fit matching guarantees len <= capacity, so a matched item always
// fits its slot (the original assigned items to slots in BST order and
// truncated rows that did not fit).

// Value of slot position i (0 <= i <= capacity) after reusing it for a row
// of len values: the row, its INT_MIN terminator, then zeros (the rest of the
// deleted row's data is cleared so the free part of the segment is empty).
static __device__ int reusedSlotValue(const int *newPayload, int start,
                                      int len, int i) {
  return i < len ? newPayload[start + i] : (i == len ? INT_MIN : 0);
}

// Metadata of a reused slot.
static __device__ void finishReuse(CBSTNode *node, int *avail, int pos,
                                   int base, int len, int capacity, int key) {
  node->occupancy = len;
  // Restore the slot's own key so the BST order is preserved
  node->index = key;
  node->tailBase = base;
  node->tailCapacity = capacity;
  avail[pos] = 0;
}

// Thread-level applyReuse (small payloads, len < 32): one thread per matched
// item. binIndices[t] -> index into matchedItemIndices/matchedSlotIndices.
__global__ void applyReuse(CBSTNode *nodes, int *flatValues, int *avail,
                           int *positions, int *newPayload, int *newPrefixSizes,
                           int *matchedItemIndices, int *matchedSlotIndices,
                           int *deletedKeys, int *binIndices, int binCount) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= binCount)
    return;
  int matchIdx = binIndices[tid];
  int itemIdx = matchedItemIndices[matchIdx];
  int slotIdx = matchedSlotIndices[matchIdx];
  int pos = positions[slotIdx];
  if (pos < 0)
    return;
  CBSTNode *node = &nodes[pos];
  int start = (itemIdx == 0) ? 0 : newPrefixSizes[itemIdx - 1];
  int len = newPrefixSizes[itemIdx] - start;
  int base = node->value;
  int capacity = node->length - 1;
  for (int i = 0; i <= capacity; ++i) {
    flatValues[base + i] = reusedSlotValue(newPayload, start, len, i);
  }
  finishReuse(node, avail, pos, base, len, capacity, deletedKeys[slotIdx]);
}

// ── Warp-level applyReuse (medium payloads, 32 <= len < 1024) ───────────
// One warp (32 lanes) cooperatively copies payload for one matched item.
// binIndices[warpIdx] → index into matchedItemIndices/matchedSlotIndices.
__global__ void applyReuse_warp(CBSTNode *nodes, int *flatValues, int *avail,
                                int *positions, int *newPayload,
                                int *newPrefixSizes, int *matchedItemIndices,
                                int *matchedSlotIndices, int *deletedKeys,
                                int *binIndices, int binCount) {
  int globalTid = threadIdx.x + blockIdx.x * blockDim.x;
  int warpIdx = globalTid >> 5;
  int lane = threadIdx.x & 31;
  if (warpIdx >= binCount)
    return;

  int matchIdx = binIndices[warpIdx];
  int itemIdx = matchedItemIndices[matchIdx];
  int slotIdx = matchedSlotIndices[matchIdx];
  int pos = positions[slotIdx];
  if (pos < 0)
    return;

  CBSTNode *node = &nodes[pos];
  int start = (itemIdx == 0) ? 0 : newPrefixSizes[itemIdx - 1];
  int len = newPrefixSizes[itemIdx] - start;
  int base = node->value;
  int capacity = node->length - 1;

  // Warp-strided cooperative copy
  for (int i = lane; i <= capacity; i += 32) {
    flatValues[base + i] = reusedSlotValue(newPayload, start, len, i);
  }

  // Lane 0 handles the metadata
  if (lane == 0)
    finishReuse(node, avail, pos, base, len, capacity, deletedKeys[slotIdx]);
}

// ── Block-level applyReuse (large payloads, len >= 1024) ────────────────
// One CTA cooperatively copies payload for one matched item.
// binIndices[blockIdx.x] → index into matchedItemIndices/matchedSlotIndices.
__global__ void applyReuse_block(CBSTNode *nodes, int *flatValues, int *avail,
                                 int *positions, int *newPayload,
                                 int *newPrefixSizes, int *matchedItemIndices,
                                 int *matchedSlotIndices, int *deletedKeys,
                                 int *binIndices, int binCount) {
  int idx = blockIdx.x;
  if (idx >= binCount)
    return;

  int matchIdx = binIndices[idx];
  int itemIdx = matchedItemIndices[matchIdx];
  int slotIdx = matchedSlotIndices[matchIdx];
  int pos = positions[slotIdx];
  if (pos < 0)
    return;

  CBSTNode *node = &nodes[pos];
  int start = (itemIdx == 0) ? 0 : newPrefixSizes[itemIdx - 1];
  int len = newPrefixSizes[itemIdx] - start;
  int base = node->value;
  int capacity = node->length - 1;

  // Block-strided cooperative copy
  for (int i = threadIdx.x; i <= capacity; i += blockDim.x) {
    flatValues[base + i] = reusedSlotValue(newPayload, start, len, i);
  }

  // Thread 0 handles the metadata
  if (threadIdx.x == 0)
    finishReuse(node, avail, pos, base, len, capacity, deletedKeys[slotIdx]);
}
