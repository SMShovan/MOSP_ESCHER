#include "device_utils.cuh"
#include "kernels.cuh"
#include <cstdio>

__global__ void buildEmptyBinaryTree(CBSTNode *nodes, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    nodes[tid].index = tid;
    nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
    nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
    nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
    nodes[tid].occupancy = 0;
    nodes[tid].tailBase = 0;
    nodes[tid].tailCapacity = 0;
  }
}

__global__ void storeItemsIntoNodes(CBSTNode *nodes, int *indices, int *values,
                                    int n, int totalSize) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    int index2 = cbstRankOfPosition(tid, n);
    nodes[tid].size = totalSize;
    if (index2 < n) {
      nodes[tid].index = indices[index2];
      nodes[tid].value = values[index2];
      int segLen;
      if (index2 < n - 1) {
        segLen = values[index2 + 1] - values[index2];
      } else {
        segLen = totalSize - values[index2];
      }
      nodes[tid].length = segLen;
      nodes[tid].tailBase = values[index2];
      nodes[tid].tailCapacity = (segLen > 0) ? segLen - 1 : 0;
      // Count actual data elements: scan until sentinel or zero
      // For initial construction, occupancy = number of non-zero,
      // non-sentinel values at the start of the segment.
      // We defer this to a separate pass or set to 0 (segments may
      // not be populated yet at kernel launch time).
      // After storeItemsIntoNodes + data copy, a fixup sets occupancy.
      nodes[tid].occupancy = 0;
    }
  }
}

__global__ void printEachNode(CBSTNode *nodes, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid <= n) {
    CBSTNode *current = nodes;
    while (current != nullptr && current->index != tid) {
      if (current->index > tid) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    if (current != nullptr) {
      printf("Node %d: Index = %d, Value = %d, Length = %d, Size = %d\n", tid,
             current->index, current->value, current->length, current->size);
    }
  }
}

// Brings the live nodes into key order for a rebuild: the node at heap
// position t has in-order rank cbstRankOfPosition(t, n), and keys are sorted
// by rank. rankedLive[r] = 1 for a live (not deleted) node.
__global__ void rankOrderNodes(const CBSTNode *nodes, const int *avail, int n,
                               CBSTNode *ranked, int *rankedLive) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  int rank = cbstRankOfPosition(tid, n);
  ranked[rank] = nodes[tid];
  rankedLive[rank] = avail[tid] == 0;
}

// Sorted key and row offset arrays of a record list.
__global__ void recordKeysAndStarts(const CBSTNode *records, int n, int *keys,
                                    int *starts) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  keys[tid] = records[tid].index;
  starts[tid] = records[tid].value;
}

// Rebuild after surplus inserts: places full node records (sorted by key) at
// their heap positions, so surviving rows keep their offset, length and
// tail metadata (occupancy, tailBase, tailCapacity). Child / parent pointers
// come from buildEmptyBinaryTree.
__global__ void placeNodeRecords(CBSTNode *nodes, const CBSTNode *sortedRecords,
                                 int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  const CBSTNode &r = sortedRecords[cbstRankOfPosition(tid, n)];
  CBSTNode &node = nodes[tid];
  node.index = r.index;
  node.value = r.value;
  node.length = r.length;
  node.size = r.size;
  node.occupancy = r.occupancy;
  node.tailBase = r.tailBase;
  node.tailCapacity = r.tailCapacity;
}
