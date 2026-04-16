#include "kernels.cuh"

// Phase 1: Read-only BST traversal to locate array positions of delete targets.
// Safe for concurrent threads since no node is modified.
__global__ void locateDeleteTargets(CBSTNode *nodes, int *deleteIndices,
                                    int deleteSize, int *outPositions) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < deleteSize) {
    int deleteIndex = deleteIndices[tid];
    CBSTNode *current = nodes;
    while (current != nullptr && current->index != deleteIndex) {
      if (current->index > deleteIndex) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    if (current != nullptr) {
      outPositions[tid] = static_cast<int>(current - nodes);
    } else {
      outPositions[tid] = -1;
    }
  }
}

// Phase 2: Apply deletions using precomputed positions. No traversal needed.
__global__ void applyDeletes(CBSTNode *nodes, int *positions, int deleteSize,
                             int *avail) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < deleteSize) {
    int pos = positions[tid];
    if (pos >= 0) {
      nodes[pos].index = -1;
      avail[pos] = 1;
      nodes[pos].occupancy = 0;
      nodes[pos].tailBase = nodes[pos].value;
      nodes[pos].tailCapacity =
          (nodes[pos].length > 0) ? nodes[pos].length - 1 : 0;
    }
  }
}

__global__ void reduceAvailLevel(int levelStart, int levelEnd, int numRecords,
                                 int *avail, int *subtreeAvail) {
  int g = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = levelStart + g;
  if (idx < 0 || idx > levelEnd || idx >= numRecords)
    return;
  int left = 2 * idx + 1;
  int right = 2 * idx + 2;
  int leftSum = (left < numRecords) ? subtreeAvail[left] : 0;
  int rightSum = (right < numRecords) ? subtreeAvail[right] : 0;
  subtreeAvail[idx] = avail[idx] + leftSum + rightSum;
}
