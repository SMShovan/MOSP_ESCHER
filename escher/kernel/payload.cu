#include "device_utils.cuh"
#include "kernels.cuh"
#include <climits>

__global__ void computeNextMultipleOf4(int *partialSolution, int *tmp, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K) {
    int val = partialSolution[3 * idx + 2];
    tmp[idx] = nextMultipleOf4(val);
  }
}

__global__ void updatePartialSolution(int *partialSolution, int *tmp, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K) {
    partialSolution[3 * idx + 2] = tmp[idx];
  }
}

__global__ void insertNode(CBSTNode *nodes, int *flatValues, int *insertIndices,
                           int *insertValues, int *insertSizes, int insertSize,
                           int *partialSolution) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < insertSize) {
    int insertIndex = insertIndices[tid];
    int *values;
    int numValues;
    if (tid == 0) {
      values = insertValues;
      numValues = insertSizes[tid];
    } else {
      values = insertValues + insertSizes[tid - 1];
      numValues = insertSizes[tid] - insertSizes[tid - 1];
    }
    CBSTNode *current = nodes;
    while (current != nullptr && current->index != insertIndex) {
      if (current->index > insertIndex) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    if (current != nullptr) {
      int valueIndex = current->value;
      for (int i = 0; i < numValues; ++i) {
        bool isOverflow = false;
        while (true) {
          int val = flatValues[valueIndex];
          if (val == 0 || val == INT_MIN)
            break;
          if (val < 0) {
            valueIndex = -val;
            continue;
          }
          if (flatValues[valueIndex + 1] == INT_MIN) {
            partialSolution[tid * 3] = valueIndex + 1;
            partialSolution[tid * 3 + 1] = i;
            partialSolution[tid * 3 + 2] = numValues - i;
            isOverflow = true;
            break;
          }
          valueIndex++;
        }
        if (isOverflow)
          break;
        if (flatValues[valueIndex] != INT_MIN)
          flatValues[valueIndex] = values[i];
      }
    }
  }
}

__global__ void allocateSpace(int *partialSolution, int *flatValues,
                              int spaceAvailableFrom, int *insertIndices,
                              int *insertValues, int *insertSizes,
                              int insertSize) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < insertSize) {
    int *values;
    int numValues;
    if (tid == 0) {
      values = insertValues;
      numValues = insertSizes[tid];
    } else {
      values = insertValues + insertSizes[tid - 1];
      numValues = insertSizes[tid] - insertSizes[tid - 1];
    }

    int idxPartialSolution = tid * 3;
    int startPartialSolution = idxPartialSolution + 1;
    int lenPartialSolution = idxPartialSolution + 2;

    if (tid == 0) {
      if (partialSolution[lenPartialSolution] == 0)
        return;
    } else {
      if (partialSolution[lenPartialSolution] ==
          partialSolution[lenPartialSolution - 3])
        return;
    }

    int startIdx;
    int storeStartIdx;
    if (tid == 0) {
      startIdx = spaceAvailableFrom;

    } else {
      startIdx = spaceAvailableFrom + partialSolution[idxPartialSolution - 1];
    }

    storeStartIdx = startIdx;

    for (int i = partialSolution[startPartialSolution]; i < numValues;
         i++, startIdx++) {
      flatValues[startIdx] = values[i];
    }

    // Compute per-thread padded size (cumulative minus previous cumulative)
    int perThreadPadded = (tid == 0)
                              ? partialSolution[lenPartialSolution]
                              : partialSolution[lenPartialSolution] -
                                    partialSolution[lenPartialSolution - 3];

    // Zero gap between written data and sentinel
    int numWritten = numValues - partialSolution[startPartialSolution];
    for (int z = numWritten; z < perThreadPadded - 1; ++z)
      flatValues[storeStartIdx + z] = 0;

    // Sentinel at last position within padded space
    flatValues[storeStartIdx + perThreadPadded - 1] = INT_MIN;

    // Back-pointer from original chunk to this overflow location
    flatValues[partialSolution[idxPartialSolution]] = storeStartIdx * (-1);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Degree-binned insertNode kernels (using occupancy/tailBase/tailCapacity)
// ═══════════════════════════════════════════════════════════════════════════

// ── Helper: BST traversal to find node by key ───────────────────────────
// Returns pointer to the found node, or nullptr.
static __device__ CBSTNode *bstFind(CBSTNode *nodes, int key) {
  CBSTNode *current = nodes;
  while (current != nullptr && current->index != key) {
    current = (current->index > key) ? current->left : current->right;
  }
  return current;
}

// ── Thread-level insertNode (small payloads, numValues < 32) ────────────
// Uses occupancy to skip serial scan. Direct write at tailBase + occupancy.
__global__ void insertNode_thread(CBSTNode *nodes, int *flatValues,
                                  int *insertIndices, int *insertValues,
                                  int *insertSizes, int *partialSolution,
                                  int *binIndices, int binCount) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= binCount)
    return;

  int origIdx = binIndices[tid];
  int insertIndex = insertIndices[origIdx];
  int *values =
      (origIdx == 0) ? insertValues : insertValues + insertSizes[origIdx - 1];
  int numValues = (origIdx == 0)
                      ? insertSizes[origIdx]
                      : insertSizes[origIdx] - insertSizes[origIdx - 1];

  CBSTNode *current = bstFind(nodes, insertIndex);
  if (current == nullptr)
    return;

  int writePos = current->tailBase + current->occupancy;
  int remaining = current->tailCapacity - current->occupancy;

  if (numValues <= remaining) {
    // All values fit in the current segment
    for (int i = 0; i < numValues; ++i) {
      flatValues[writePos + i] = values[i];
    }
    flatValues[writePos + numValues] = INT_MIN;
    current->occupancy += numValues;
  } else {
    // Partial fit: write what we can, record overflow
    for (int i = 0; i < remaining; ++i) {
      flatValues[writePos + i] = values[i];
    }
    // Record overflow in partialSolution (using origIdx for correct
    // positioning)
    partialSolution[origIdx * 3] = writePos + remaining;
    partialSolution[origIdx * 3 + 1] = remaining;
    partialSolution[origIdx * 3 + 2] = numValues - remaining;
    current->occupancy = current->tailCapacity;
  }
}

// ── Warp-level insertNode (medium payloads, 32 <= numValues < 1024) ─────
// One warp (32 lanes) cooperatively writes payload values.
// binIndices[warpIdx] → original insert index.
__global__ void insertNode_warp(CBSTNode *nodes, int *flatValues,
                                int *insertIndices, int *insertValues,
                                int *insertSizes, int *partialSolution,
                                int *binIndices, int binCount) {
  int globalTid = threadIdx.x + blockIdx.x * blockDim.x;
  int warpIdx = globalTid >> 5;
  int lane = threadIdx.x & 31;
  if (warpIdx >= binCount)
    return;

  int origIdx = binIndices[warpIdx];

  // Lane 0 does BST traversal and reads metadata
  int insertIndex = insertIndices[origIdx];
  int valuesStart = (origIdx == 0) ? 0 : insertSizes[origIdx - 1];
  int numValues = (origIdx == 0)
                      ? insertSizes[origIdx]
                      : insertSizes[origIdx] - insertSizes[origIdx - 1];

  // We need shared data across the warp. Use __shfl_sync for broadcast.
  // Lane 0 finds the node and broadcasts metadata.
  int writePos = 0, remaining = 0, tailBase = 0, tailCap = 0;
  int nodeFound = 0;
  // All lanes participate in the bstFind independently (same key, same result)
  // This avoids the need for shfl broadcast of the pointer.
  CBSTNode *current = bstFind(nodes, insertIndex);
  if (current != nullptr) {
    nodeFound = 1;
    writePos = current->tailBase + current->occupancy;
    remaining = current->tailCapacity - current->occupancy;
    tailBase = current->tailBase;
    tailCap = current->tailCapacity;
  }

  // Broadcast from lane 0 (consistent across lanes since same key)
  nodeFound = __shfl_sync(0xFFFFFFFF, nodeFound, 0);
  if (!nodeFound)
    return;
  writePos = __shfl_sync(0xFFFFFFFF, writePos, 0);
  remaining = __shfl_sync(0xFFFFFFFF, remaining, 0);

  int copyLen = (numValues <= remaining) ? numValues : remaining;

  // Warp-strided cooperative copy
  for (int i = lane; i < copyLen; i += 32) {
    flatValues[writePos + i] = insertValues[valuesStart + i];
  }

  // Lane 0 handles sentinel, metadata, overflow plan
  if (lane == 0) {
    if (numValues <= remaining) {
      flatValues[writePos + numValues] = INT_MIN;
      current->occupancy += numValues;
    } else {
      partialSolution[origIdx * 3] = writePos + remaining;
      partialSolution[origIdx * 3 + 1] = remaining;
      partialSolution[origIdx * 3 + 2] = numValues - remaining;
      current->occupancy = current->tailCapacity;
    }
  }
}

// ── Block-level insertNode (large payloads, numValues >= 1024) ──────────
// One CTA cooperatively writes payload values.
// binIndices[blockIdx.x] → original insert index.
__global__ void insertNode_block(CBSTNode *nodes, int *flatValues,
                                 int *insertIndices, int *insertValues,
                                 int *insertSizes, int *partialSolution,
                                 int *binIndices, int binCount) {
  int idx = blockIdx.x;
  if (idx >= binCount)
    return;

  // Shared metadata for the block
  __shared__ int sh_writePos;
  __shared__ int sh_remaining;
  __shared__ int sh_numValues;
  __shared__ int sh_valuesStart;
  __shared__ int sh_origIdx;
  __shared__ CBSTNode *sh_node;

  if (threadIdx.x == 0) {
    sh_origIdx = binIndices[idx];
    int insertIndex = insertIndices[sh_origIdx];
    sh_valuesStart = (sh_origIdx == 0) ? 0 : insertSizes[sh_origIdx - 1];
    sh_numValues = (sh_origIdx == 0)
                       ? insertSizes[sh_origIdx]
                       : insertSizes[sh_origIdx] - insertSizes[sh_origIdx - 1];

    sh_node = bstFind(nodes, insertIndex);
    if (sh_node != nullptr) {
      sh_writePos = sh_node->tailBase + sh_node->occupancy;
      sh_remaining = sh_node->tailCapacity - sh_node->occupancy;
    }
  }
  __syncthreads();

  if (sh_node == nullptr)
    return;

  int copyLen = (sh_numValues <= sh_remaining) ? sh_numValues : sh_remaining;

  // Block-strided cooperative copy
  for (int i = threadIdx.x; i < copyLen; i += blockDim.x) {
    flatValues[sh_writePos + i] = insertValues[sh_valuesStart + i];
  }
  __syncthreads();

  // Thread 0 handles sentinel, metadata, overflow plan
  if (threadIdx.x == 0) {
    if (sh_numValues <= sh_remaining) {
      flatValues[sh_writePos + sh_numValues] = INT_MIN;
      sh_node->occupancy += sh_numValues;
    } else {
      partialSolution[sh_origIdx * 3] = sh_writePos + sh_remaining;
      partialSolution[sh_origIdx * 3 + 1] = sh_remaining;
      partialSolution[sh_origIdx * 3 + 2] = sh_numValues - sh_remaining;
      sh_node->occupancy = sh_node->tailCapacity;
    }
  }
}

// ── Metadata fixup after allocateSpace ──────────────────────────────────
// Updates tailBase, tailCapacity, occupancy for nodes that overflowed.
// Called after allocateSpace to keep metadata consistent.
__global__ void fixupOverflowMetadata(CBSTNode *nodes, int *insertIndices,
                                      int *insertSizes, int *partialSolution,
                                      int spaceAvailableFrom, int insertSize) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= insertSize)
    return;

  int idxPS = tid * 3;
  int lenPS = idxPS + 2;

  // Check if this item had overflow (same condition as allocateSpace)
  bool hadOverflow;
  if (tid == 0) {
    hadOverflow = (partialSolution[lenPS] != 0);
  } else {
    hadOverflow = (partialSolution[lenPS] != partialSolution[lenPS - 3]);
  }
  if (!hadOverflow)
    return;

  // Compute the overflow segment base (same as allocateSpace)
  int storeStartIdx;
  if (tid == 0) {
    storeStartIdx = spaceAvailableFrom;
  } else {
    storeStartIdx = spaceAvailableFrom + partialSolution[idxPS - 1];
  }

  // Compute overflow segment capacity
  int perThreadPadded =
      (tid == 0) ? partialSolution[lenPS]
                 : partialSolution[lenPS] - partialSolution[lenPS - 3];

  // Find the node and update its metadata
  int insertIndex = insertIndices[tid];
  CBSTNode *current = bstFind(nodes, insertIndex);
  if (current != nullptr) {
    int numValues =
        (tid == 0) ? insertSizes[tid] : insertSizes[tid] - insertSizes[tid - 1];
    int overflowStart =
        partialSolution[idxPS + 1]; // how many values fit in original segment
    int overflowCount = numValues - overflowStart;
    current->tailBase = storeStartIdx;
    current->tailCapacity = perThreadPadded - 1; // -1 for sentinel
    current->occupancy = overflowCount;
  }
}
