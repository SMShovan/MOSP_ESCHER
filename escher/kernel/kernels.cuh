#pragma once

#include "../include/structure.hpp"

// Build tree
__global__ void buildEmptyBinaryTree(CBSTNode *nodes, int n);
__global__ void storeItemsIntoNodes(CBSTNode *nodes, int *indices, int *values,
                                    int n, int totalSize);
__global__ void printEachNode(CBSTNode *nodes, int n);
__global__ void rankOrderNodes(const CBSTNode *nodes, const int *avail, int n,
                               CBSTNode *ranked, int *rankedLive);
__global__ void recordKeysAndStarts(const CBSTNode *records, int n, int *keys,
                                    int *starts);
__global__ void placeNodeRecords(CBSTNode *nodes, const CBSTNode *sortedRecords,
                                 int n);

// Payload ops
__global__ void allocateSpace(int *partialSolution, int *flatValues,
                              int spaceAvailableFrom, int *insertIndices,
                              int *insertValues, int *insertSizes,
                              int insertSize);
__global__ void computeNextMultipleOf4(int *partialSolution, int *tmp, int K);
__global__ void updatePartialSolution(int *partialSolution, int *tmp, int K);

// Delete / availability (two-phase: locate then apply)
__global__ void locateDeleteTargets(CBSTNode *nodes, int *deleteIndices,
                                    int deleteSize, int *outPositions);
__global__ void applyDeletes(CBSTNode *nodes, int *positions, int deleteSize,
                             int *avail);
__global__ void reduceAvailLevel(int levelStart, int levelEnd, int numRecords,
                                 int *avail, int *subtreeAvail);

// Lookup / find
__global__ void findNode(CBSTNode *nodes, int *searchIndices, int searchSize);
__global__ void findContents(CBSTNode *nodes, int *searchIndices,
                             int searchSize, int *flatValues);

// Insert reuse (two-phase: locate then apply)
__global__ void locateReusableSlots(int *subtreeAvail, int *avail,
                                    int numRecords, int *outPositions, int K);

// Best-fit metadata extraction
__global__ void extractSlotCapacities(CBSTNode *nodes, int *positions,
                                      int *outCapacities, int D);
__global__ void computeItemSizes(int *prefixSizes, int *outSizes, int K);

// GPU-parallel best-fit matching
__global__ void lowerBoundKernel(int *sortedCaps, int D, int *sortedSizes,
                                 int M, int *outLo);
__global__ void computeBInPlace(int *lo, int M);
__global__ void computeAssigned(int *prefixMax, int *assigned, int M);

// Recover original keys for deleted-slot positions via CBST layout formula
__global__ void extractKeysFromPositions(int *d_keys, int *positions,
                                         int *outKeys, int numRecords, int D);

// Pair matched items with their best-fit slots
__global__ void pairMatches(const int *itemOrder, const int *assigned,
                            const int *slotOrder, int matchCount,
                            int *matchedItemIndices, int *matchedSlotIndices);

// Degree-binned apply reuse (thread / warp / block per matched item); keys
// come from deletedKeys to preserve the BST order
__global__ void applyReuse(CBSTNode *nodes, int *flatValues, int *avail,
                           int *positions, int *newPayload, int *newPrefixSizes,
                           int *matchedItemIndices, int *matchedSlotIndices,
                           int *deletedKeys, int *binIndices, int binCount);
__global__ void applyReuse_warp(CBSTNode *nodes, int *flatValues, int *avail,
                                int *positions, int *newPayload,
                                int *newPrefixSizes, int *matchedItemIndices,
                                int *matchedSlotIndices, int *deletedKeys,
                                int *binIndices, int binCount);
__global__ void applyReuse_block(CBSTNode *nodes, int *flatValues, int *avail,
                                 int *positions, int *newPayload,
                                 int *newPrefixSizes, int *matchedItemIndices,
                                 int *matchedSlotIndices, int *deletedKeys,
                                 int *binIndices, int binCount);

// Degree-binned insertNode (fill)
__global__ void insertNode_thread(CBSTNode *nodes, int *flatValues,
                                  int *insertIndices, int *insertValues,
                                  int *insertSizes, int *partialSolution,
                                  int *binIndices, int binCount);
__global__ void insertNode_warp(CBSTNode *nodes, int *flatValues,
                                int *insertIndices, int *insertValues,
                                int *insertSizes, int *partialSolution,
                                int *binIndices, int binCount);
__global__ void insertNode_block(CBSTNode *nodes, int *flatValues,
                                 int *insertIndices, int *insertValues,
                                 int *insertSizes, int *partialSolution,
                                 int *binIndices, int binCount);
__global__ void fixupOverflowMetadata(CBSTNode *nodes, int *insertIndices,
                                      int *insertSizes, int *partialSolution,
                                      int spaceAvailableFrom, int insertSize);

// Unfill (degree-binned)
__global__ void unfill_thread(CBSTNode *nodes, int *flatValues, int *keys,
                              int *valuesToRemove, int *removePrefixSizes,
                              int *binIndices, int binCount);
__global__ void unfill_warp(CBSTNode *nodes, int *flatValues, int *keys,
                            int *valuesToRemove, int *removePrefixSizes,
                            int *binIndices, int binCount);
__global__ void unfill_block(CBSTNode *nodes, int *flatValues, int *keys,
                             int *valuesToRemove, int *removePrefixSizes,
                             int *binIndices, int binCount);

// Payload compaction (two passes, one warp per node)
__global__ void compactSizesKernel(const CBSTNode *nodes, const int *avail,
                                   int n, const int *flat, int limit,
                                   int *liveCount, long long *slotsByRank);
__global__ void compactCopyKernel(CBSTNode *nodes, const int *avail, int n,
                                  const int *flat, int limit,
                                  const int *liveCount,
                                  const long long *offsetByRank, int *dst,
                                  int *startOffsets);
