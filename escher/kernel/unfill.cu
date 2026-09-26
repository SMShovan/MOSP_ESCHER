#include "kernels.cuh"
#include <climits>

// ═══════════════════════════════════════════════════════════════════════════
// Degree-binned unfill kernels (using occupancy/tailBase/tailCapacity)
// ═══════════════════════════════════════════════════════════════════════════

// ── Helper: BST traversal ───────────────────────────────────────────────
static __device__ CBSTNode *unfillBstFind(CBSTNode *nodes, int key) {
  CBSTNode *current = nodes;
  while (current != nullptr && current->index != key) {
    current = (current->index > key) ? current->left : current->right;
  }
  return current;
}

// ── Serial unfill of one row (all segments) ─────────────────────────────
// Removes valuesToRemove[start, end) from the row of `node`, compacting each
// segment in place and keeping the chain pointers. Returns the number of
// live entries left in the tail segment, which is what `occupancy` counts:
// fill appends at tailBase + occupancy. (The original subtracted removals
// from every segment from the tail occupancy, so a later fill overwrote
// live entries of the tail segment.)
static __device__ int unfillRowSerial(const CBSTNode *node, int *flatValues,
                                      const int *valuesToRemove, int start,
                                      int end) {
  int segBase = node->value;
  for (;;) {
    int w = 0;
    int i = 0;
    for (;;) {
      int val = flatValues[segBase + i];
      if (val == INT_MIN || val == 0 || val < 0)
        break;
      bool removeIt = false;
      for (int r = start; r < end; ++r) {
        if (valuesToRemove[r] == val) {
          removeIt = true;
          break;
        }
      }
      if (!removeIt) {
        flatValues[segBase + w] = val;
        ++w;
      }
      ++i;
    }
    int endVal = flatValues[segBase + i];
    bool chained = endVal < 0 && endVal != INT_MIN;
    flatValues[segBase + w] = chained ? endVal : INT_MIN;
    for (int z = w + 1; z <= i; ++z)
      flatValues[segBase + z] = 0;
    if (!chained)
      return w;
    segBase = -endVal;
  }
}

// ── Thread-level unfill with metadata (small bins) ──────────────────────
__global__ void unfill_thread(CBSTNode *nodes, int *flatValues, int *keys,
                              int *valuesToRemove, int *removePrefixSizes,
                              int *binIndices, int binCount) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= binCount)
    return;

  int origIdx = binIndices[tid];
  int key = keys[origIdx];
  int start = (origIdx == 0) ? 0 : removePrefixSizes[origIdx - 1];
  int end = removePrefixSizes[origIdx];

  CBSTNode *cur = unfillBstFind(nodes, key);
  if (cur == nullptr)
    return;
  cur->occupancy = unfillRowSerial(cur, flatValues, valuesToRemove, start, end);
}

// ── Warp-level unfill (medium bins, 32 <= occupancy < 1024) ─────────────
// One warp cooperatively processes a single-segment node.
// Uses ballot + popc for warp-level stream compaction.
// Falls back to thread-level logic for multi-segment (overflow chain) nodes.
__global__ void unfill_warp(CBSTNode *nodes, int *flatValues, int *keys,
                            int *valuesToRemove, int *removePrefixSizes,
                            int *binIndices, int binCount) {
  int globalTid = threadIdx.x + blockIdx.x * blockDim.x;
  int warpIdx = globalTid >> 5;
  int lane = threadIdx.x & 31;
  if (warpIdx >= binCount)
    return;

  int origIdx = binIndices[warpIdx];
  int key = keys[origIdx];
  int remStart = (origIdx == 0) ? 0 : removePrefixSizes[origIdx - 1];
  int remEnd = removePrefixSizes[origIdx];

  // All lanes do BST traversal (same key, same result)
  CBSTNode *cur = unfillBstFind(nodes, key);
  if (cur == nullptr)
    return;

  int segBase = cur->tailBase;
  int occ = cur->occupancy;

  // For single-segment nodes (tailBase == value), do cooperative compaction.
  // For multi-segment (overflow chain), fall back to lane 0 doing serial work.
  if (cur->tailBase != cur->value) {
    if (lane == 0)
      cur->occupancy =
          unfillRowSerial(cur, flatValues, valuesToRemove, remStart, remEnd);
    return;
  }

  // Single-segment cooperative compaction using ballot
  // Process elements in warp-sized chunks
  int writeOffset = 0; // Accumulated across chunks (identical in all lanes)

  for (int chunk = 0; chunk < occ; chunk += 32) {
    int elemIdx = chunk + lane;
    int val = 0;
    bool keep = false;

    if (elemIdx < occ) {
      val = flatValues[segBase + elemIdx];
      if (val != 0 && val != INT_MIN && val > 0) {
        keep = true;
        // Check if this value should be removed
        for (int r = remStart; r < remEnd; ++r) {
          if (valuesToRemove[r] == val) {
            keep = false;
            break;
          }
        }
      }
    }

    // Ballot: which lanes have a "keep" element?
    unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
    int keptInChunk = __popc(mask);
    // My position within the kept elements in this chunk
    unsigned lowerMask = (1u << lane) - 1u;
    int myPos = __popc(mask & lowerMask);

    // Write kept elements compactly
    if (keep) {
      flatValues[segBase + writeOffset + myPos] = val;
    }

    writeOffset += keptInChunk;
  }

  // Lane 0 writes sentinel and zeros, updates metadata
  if (lane == 0) {
    flatValues[segBase + writeOffset] = INT_MIN;
    // Zero remaining positions (between writeOffset+1 and occ)
    for (int z = writeOffset + 1; z <= occ; ++z) {
      flatValues[segBase + z] = 0;
    }
    cur->occupancy = writeOffset;
  }
}

// ── Block-level unfill (large bins, occupancy >= 1024) ──────────────────
// One CTA cooperatively processes a single-segment node using shared memory
// for the removal set and block-wide scan for compaction.
__global__ void unfill_block(CBSTNode *nodes, int *flatValues, int *keys,
                             int *valuesToRemove, int *removePrefixSizes,
                             int *binIndices, int binCount) {
  int idx = blockIdx.x;
  if (idx >= binCount)
    return;

  // Shared metadata
  __shared__ int sh_segBase;
  __shared__ int sh_occ;
  __shared__ int sh_remStart;
  __shared__ int sh_remEnd;
  __shared__ CBSTNode *sh_node;
  __shared__ int sh_writeOffset;

  if (threadIdx.x == 0) {
    int origIdx = binIndices[idx];
    int key = keys[origIdx];
    sh_remStart = (origIdx == 0) ? 0 : removePrefixSizes[origIdx - 1];
    sh_remEnd = removePrefixSizes[origIdx];

    sh_node = unfillBstFind(nodes, key);
    if (sh_node != nullptr) {
      sh_segBase = sh_node->tailBase;
      sh_occ = sh_node->occupancy;
    }
    sh_writeOffset = 0;
  }
  __syncthreads();

  if (sh_node == nullptr)
    return;

  // Multi-segment fallback: thread 0 does serial
  if (sh_node->tailBase != sh_node->value) {
    if (threadIdx.x == 0)
      sh_node->occupancy = unfillRowSerial(sh_node, flatValues, valuesToRemove,
                                           sh_remStart, sh_remEnd);
    return;
  }

  // Single-segment: block-cooperative compaction
  // Process in block-sized chunks using warp ballots + shared prefix sums
  extern __shared__ int sh_compact[]; // blockDim.x ints for compaction
  int warpLane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  int numWarps = blockDim.x >> 5;

  for (int chunk = 0; chunk < sh_occ; chunk += blockDim.x) {
    int elemIdx = chunk + threadIdx.x;
    int val = 0;
    bool keep = false;

    if (elemIdx < sh_occ) {
      val = flatValues[sh_segBase + elemIdx];
      if (val != 0 && val != INT_MIN && val > 0) {
        keep = true;
        for (int r = sh_remStart; r < sh_remEnd; ++r) {
          if (valuesToRemove[r] == val) {
            keep = false;
            break;
          }
        }
      }
    }

    // Warp-level ballot
    unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
    int warpKept = __popc(mask);
    unsigned lowerMask = (1u << warpLane) - 1u;
    int laneOffset = __popc(mask & lowerMask);

    // Per-warp kept counts into shared memory for block prefix sum
    if (warpLane == 0) {
      sh_compact[warpId] = warpKept;
    }
    __syncthreads();

    // Simple serial prefix sum (numWarps is small, max ~8 for 256 threads)
    if (threadIdx.x == 0) {
      int sum = 0;
      for (int w = 0; w < numWarps; ++w) {
        int tmp = sh_compact[w];
        sh_compact[w] = sum;
        sum += tmp;
      }
      // Store total kept in this chunk at sh_compact[numWarps]
      sh_compact[numWarps] = sum;
    }
    __syncthreads();

    int warpBase = sh_compact[warpId];
    int chunkKept = sh_compact[numWarps];

    // Write kept elements
    if (keep) {
      flatValues[sh_segBase + sh_writeOffset + warpBase + laneOffset] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      sh_writeOffset += chunkKept;
    }
    __syncthreads();
  }

  // Thread 0 writes sentinel, zeros remaining, updates metadata
  if (threadIdx.x == 0) {
    flatValues[sh_segBase + sh_writeOffset] = INT_MIN;
    for (int z = sh_writeOffset + 1; z <= sh_occ; ++z) {
      flatValues[sh_segBase + z] = 0;
    }
    sh_node->occupancy = sh_writeOffset;
  }
}
