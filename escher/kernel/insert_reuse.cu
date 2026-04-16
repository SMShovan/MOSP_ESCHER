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
  int log2_pos = floor_log2(pos + 1);
  int log2_n = floor_log2(numRecords);
  int index =
      ((2 * (pos + 1 - (1 << log2_pos))) + 1) * (1 << log2_n) / (1 << log2_pos);
  int index2 =
      min(index, index - (index / 2) + (numRecords + 1 - (1 << log2_n)));
  index2--;
  outKeys[tid] = d_keys[index2];
}

// ── Phase 2: Apply reuse ────────────────────────────────────────────────

// Thread-level applyReuse (small payloads, len < 32): one thread per matched
// item. Also used as the "small bin" kernel in degree-binned dispatch.
__global__ void applyReuse(CBSTNode *nodes, int *flatValues, int *avail,
                           int *positions, int *newKeys, int *newPayload,
                           int *newPrefixSizes, int *relocationPlan,
                           int *matchedItemIndices, int *matchedSlotIndices,
                           int *deletedKeys, int matchCount) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= matchCount)
    return;
  int itemIdx = matchedItemIndices[tid];
  int slotIdx = matchedSlotIndices[tid];
  int pos = positions[slotIdx];
  if (pos < 0)
    return;
  CBSTNode *node = &nodes[pos];
  int start = (itemIdx == 0) ? 0 : newPrefixSizes[itemIdx - 1];
  int end = newPrefixSizes[itemIdx];
  int len = end - start;
  int base = node->value;
  int capacity = node->length - 1;
  if (len <= capacity) {
    for (int i = 0; i < len; ++i) {
      flatValues[base + i] = newPayload[start + i];
    }
    flatValues[base + len] = INT_MIN;
    // Update metadata
    node->occupancy = len;
  } else {
    for (int i = 0; i < capacity; ++i) {
      flatValues[base + i] = newPayload[start + i];
    }
    int idx3 = tid * 3;
    relocationPlan[idx3] = base + capacity;
    relocationPlan[idx3 + 1] = capacity;
    relocationPlan[idx3 + 2] = len - capacity;
    node->occupancy = capacity;
  }
  // Write the SLOT's original key (not the item's key) to preserve BST order
  node->index = deletedKeys[slotIdx];
  node->tailBase = base;
  node->tailCapacity = capacity;
  avail[pos] = 0;
}

// ── Warp-level applyReuse (medium payloads, 32 <= len < 1024) ───────────
// One warp (32 lanes) cooperatively copies payload for one matched item.
// binIndices[warpIdx] → index into matchedItemIndices/matchedSlotIndices.
__global__ void applyReuse_warp(CBSTNode *nodes, int *flatValues, int *avail,
                                int *positions, int *newPayload,
                                int *newPrefixSizes, int *relocationPlan,
                                int *matchedItemIndices,
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
  int end = newPrefixSizes[itemIdx];
  int len = end - start;
  int base = node->value;
  int capacity = node->length - 1;

  int copyLen = (len <= capacity) ? len : capacity;

  // Warp-strided cooperative copy
  for (int i = lane; i < copyLen; i += 32) {
    flatValues[base + i] = newPayload[start + i];
  }

  // Lane 0 handles sentinel, metadata, overflow plan
  if (lane == 0) {
    if (len <= capacity) {
      flatValues[base + len] = INT_MIN;
      node->occupancy = len;
    } else {
      int idx3 = matchIdx * 3;
      relocationPlan[idx3] = base + capacity;
      relocationPlan[idx3 + 1] = capacity;
      relocationPlan[idx3 + 2] = len - capacity;
      node->occupancy = capacity;
    }
    node->index = deletedKeys[slotIdx];
    node->tailBase = base;
    node->tailCapacity = capacity;
    avail[pos] = 0;
  }
}

// ── Block-level applyReuse (large payloads, len >= 1024) ────────────────
// One CTA cooperatively copies payload for one matched item.
// binIndices[blockIdx.x] → index into matchedItemIndices/matchedSlotIndices.
__global__ void applyReuse_block(CBSTNode *nodes, int *flatValues, int *avail,
                                 int *positions, int *newPayload,
                                 int *newPrefixSizes, int *relocationPlan,
                                 int *matchedItemIndices,
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
  int end = newPrefixSizes[itemIdx];
  int len = end - start;
  int base = node->value;
  int capacity = node->length - 1;

  int copyLen = (len <= capacity) ? len : capacity;

  // Block-strided cooperative copy
  for (int i = threadIdx.x; i < copyLen; i += blockDim.x) {
    flatValues[base + i] = newPayload[start + i];
  }

  // Thread 0 handles sentinel, metadata, overflow plan
  if (threadIdx.x == 0) {
    if (len <= capacity) {
      flatValues[base + len] = INT_MIN;
      node->occupancy = len;
    } else {
      int idx3 = matchIdx * 3;
      relocationPlan[idx3] = base + capacity;
      relocationPlan[idx3 + 1] = capacity;
      relocationPlan[idx3 + 2] = len - capacity;
      node->occupancy = capacity;
    }
    node->index = deletedKeys[slotIdx];
    node->tailBase = base;
    node->tailCapacity = capacity;
    avail[pos] = 0;
  }
}
