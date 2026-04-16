#include "../include/printUtils.hpp"
#include "../include/structure.hpp"
#include "../include/flatten.hpp"
#include "../include/escher_errors.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

// Functor for thrust predicate: returns true when value < threshold
struct LessThan {
  int threshold;
  __host__ __device__ bool operator()(int x) const { return x < threshold; }
};

// Kernel prototypes moved to kernel/kernels.cuh
#include "../kernel/kernels.cuh"

// Utility: dump node index and value to plain arrays
__global__ void dumpNodeIndexValue(CBSTNode *nodes, int n, int *outIndex,
                                   int *outValue) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    outIndex[tid] = nodes[tid].index;
    outValue[tid] = nodes[tid].value;
  }
}

// Local CUDA error checker for this TU. Delegates to the public
// escher::checkCudaImpl helper so failures surface as EscherError
// exceptions rather than abrupt process exits.
static inline void checkCuda(cudaError_t result) {
  ::escher::checkCudaImpl(result, __FILE__, __LINE__, "checkCuda");
}

void constructCBST(int *keys, int *startOffsets, int numRecords,
                   int *flatPayload, int flatPayloadSize, int payloadCapacity,
                   const char *datasetName, CBSTContext &ctx) {
  // Argument validation. The upstream ESCHER code silently exited or
  // returned on bad input; we now surface the error as an EscherError
  // so DynamicGraph and MOSP callers can handle failures gracefully.
  if (numRecords < 0) {
    throw ::escher::EscherError("constructCBST: numRecords must be non-negative");
  }
  if (flatPayloadSize < 0 || payloadCapacity < 0) {
    throw ::escher::EscherError("constructCBST: flat payload sizes must be non-negative");
  }
  if (payloadCapacity < flatPayloadSize) {
    throw ::escher::EscherError(
        "constructCBST: payloadCapacity < flatPayloadSize (flat payload does not fit)");
  }
  if (numRecords > 0 && (keys == nullptr || startOffsets == nullptr)) {
    throw ::escher::EscherError("constructCBST: keys/startOffsets must not be null");
  }
  if (flatPayloadSize > 0 && flatPayload == nullptr) {
    throw ::escher::EscherError("constructCBST: flatPayload must not be null when size > 0");
  }

  ctx.fixedSize = payloadCapacity;
  ctx.numRecords = numRecords;
  ctx.initialPayloadSize = flatPayloadSize;
  ctx.datasetName = datasetName;
  // ctx.alignment should be set by the owner (CBSTOperations) before calling

  if (numRecords == 0) {
    return;
  }

  checkCuda(cudaMalloc(&ctx.d_nodes, numRecords * sizeof(CBSTNode)));
  checkCuda(cudaMalloc(&ctx.d_keys, numRecords * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_startOffsets, numRecords * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_flatPayload, ctx.fixedSize * sizeof(int)));

  checkCuda(cudaMemcpy(ctx.d_keys, keys, numRecords * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_startOffsets, startOffsets,
                       numRecords * sizeof(int), cudaMemcpyHostToDevice));

  checkCuda(cudaMemcpy(ctx.d_flatPayload, flatPayload,
                       flatPayloadSize * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemset(ctx.d_flatPayload + flatPayloadSize, 0,
                       (ctx.fixedSize - flatPayloadSize) * sizeof(int)));

  checkCuda(cudaMalloc(&ctx.d_insertKeys, numRecords * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_insertPayload, numRecords * 3 * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, numRecords * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_relocationPlan, 3 * numRecords * sizeof(int)));
  // Availability arrays (0/1 per node) and subtree sums (per node)
  checkCuda(cudaMalloc(&ctx.d_avail, numRecords * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_subtreeAvail, numRecords * sizeof(int)));
  checkCuda(cudaMemset(ctx.d_avail, 0, numRecords * sizeof(int)));
  checkCuda(cudaMemset(ctx.d_subtreeAvail, 0, numRecords * sizeof(int)));

  int blockSize = 256;
  int numBlocks = (numRecords + blockSize - 1) / blockSize;

  buildEmptyBinaryTree<<<numBlocks, blockSize>>>(ctx.d_nodes, numRecords);
  checkCuda(cudaDeviceSynchronize());

  storeItemsIntoNodes<<<numBlocks, blockSize>>>(
      ctx.d_nodes, ctx.d_keys, ctx.d_startOffsets, numRecords, flatPayloadSize);
  checkCuda(cudaDeviceSynchronize());

  // The upstream ESCHER code launched @c printEachNode here for debugging,
  // which spams stdout with one printf per record. MOSP's stress tests run
  // hundreds of constructs per invocation, so the debug launch has been
  // disabled. Re-enable under @c ESCHER_DEBUG_CONSTRUCT if needed.
#ifdef ESCHER_DEBUG_CONSTRUCT
  std::cout << "Printing the tree from the device (" << datasetName
            << "):" << std::endl;
  printEachNode<<<numBlocks, blockSize>>>(ctx.d_nodes, numRecords);
  checkCuda(cudaDeviceSynchronize());
#endif
}

void fillCBST(const std::vector<int> &insertKeys,
              const std::vector<int> &insertPayload,
              const std::vector<int> &insertPrefixSizes, CBSTContext &ctx) {
  if (insertKeys.empty())
    return;
  int K = static_cast<int>(insertKeys.size());
  std::vector<int> relocationPlanHost(K * 3, 0);

  checkCuda(cudaMemcpy(ctx.d_insertKeys, insertKeys.data(), K * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_insertPayload, insertPayload.data(),
                       insertPayload.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, insertPrefixSizes.data(),
                       K * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_relocationPlan, relocationPlanHost.data(),
                       relocationPlanHost.size() * sizeof(int),
                       cudaMemcpyHostToDevice));

  int blockSize = 256;

  // ── Degree-binned insertNode dispatch ────────────────────────────────
  // Compute per-item payload sizes and bin them
  std::vector<int> smallBin, medBin, largeBin;
  for (int i = 0; i < K; ++i) {
    int numValues = (i == 0) ? insertPrefixSizes[0]
                             : insertPrefixSizes[i] - insertPrefixSizes[i - 1];
    if (numValues < 32)
      smallBin.push_back(i);
    else if (numValues < 1024)
      medBin.push_back(i);
    else
      largeBin.push_back(i);
  }

  // Upload bin index arrays and launch specialized kernels
  if (!smallBin.empty()) {
    int *d_binIdx;
    int n = static_cast<int>(smallBin.size());
    checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
    checkCuda(cudaMemcpy(d_binIdx, smallBin.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    int blocks = (n + blockSize - 1) / blockSize;
    insertNode_thread<<<blocks, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, ctx.d_relocationPlan, d_binIdx, n);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_binIdx));
  }
  if (!medBin.empty()) {
    int *d_binIdx;
    int n = static_cast<int>(medBin.size());
    checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
    checkCuda(cudaMemcpy(d_binIdx, medBin.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    int totalThreads = n * 32;
    int blocks = (totalThreads + blockSize - 1) / blockSize;
    insertNode_warp<<<blocks, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, ctx.d_relocationPlan, d_binIdx, n);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_binIdx));
  }
  if (!largeBin.empty()) {
    int *d_binIdx;
    int n = static_cast<int>(largeBin.size());
    checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
    checkCuda(cudaMemcpy(d_binIdx, largeBin.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    insertNode_block<<<n, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, ctx.d_relocationPlan, d_binIdx, n);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_binIdx));
  }

  // ── Overflow handling (same as before) ───────────────────────────────
  int *d_tmp;
  checkCuda(cudaMalloc(&d_tmp, K * sizeof(int)));
  computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(
      ctx.d_relocationPlan, d_tmp, K);
  checkCuda(cudaDeviceSynchronize());
  thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(d_tmp);
  thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
  checkCuda(cudaDeviceSynchronize());
  updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(
      ctx.d_relocationPlan, d_tmp, K);
  checkCuda(cudaDeviceSynchronize());

  std::vector<int> relocationPlanHostOut(K * 3);
  checkCuda(cudaMemcpy(relocationPlanHostOut.data(), ctx.d_relocationPlan,
                       K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
#ifdef ESCHER_DEBUG_FILL
  printVector(relocationPlanHostOut, "Cumulative Relocation Plan");
#endif

  int totalAppended = (K > 0) ? relocationPlanHostOut[3 * (K - 1) + 2] : 0;
  if (ctx.initialPayloadSize + totalAppended > ctx.fixedSize) {
    checkCuda(cudaFree(d_tmp));
    throw ::escher::EscherError(
        std::string("fillCBST [") + (ctx.datasetName ? ctx.datasetName : "?") +
        "]: payload overflow (" +
        std::to_string(ctx.initialPayloadSize + totalAppended) + " > " +
        std::to_string(ctx.fixedSize) +
        "). Increase payloadCapacity.");
  }

#ifdef ESCHER_DEBUG_FILL
  printf("[%s] Space available from: %d \n", ctx.datasetName,
         ctx.initialPayloadSize);
#endif
  int numBlocks = (K + blockSize - 1) / blockSize;
  allocateSpace<<<numBlocks, blockSize>>>(
      ctx.d_relocationPlan, ctx.d_flatPayload, ctx.initialPayloadSize,
      ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes, K);
  checkCuda(cudaDeviceSynchronize());

  // Fixup metadata for overflowed nodes
  fixupOverflowMetadata<<<numBlocks, blockSize>>>(
      ctx.d_nodes, ctx.d_insertKeys, ctx.d_insertPrefixSizes,
      ctx.d_relocationPlan, ctx.initialPayloadSize, K);
  checkCuda(cudaDeviceSynchronize());

  if (K > 0) {
    ctx.initialPayloadSize += totalAppended;
  }

#ifdef ESCHER_DEBUG_FILL
  std::vector<int> updatedFlat(ctx.fixedSize);
  checkCuda(cudaMemcpy(updatedFlat.data(), ctx.d_flatPayload,
                       ctx.fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
  printVector(updatedFlat, "Updated Flattened Values (vec1d)");
#endif

  checkCuda(cudaFree(d_tmp));
}

void deleteCBST(const std::vector<int> &deleteKeys, CBSTContext &ctx) {
  if (deleteKeys.empty())
    return;
  int deleteSize = static_cast<int>(deleteKeys.size());
  int *d_deleteKeys;
  checkCuda(cudaMalloc(&d_deleteKeys, deleteSize * sizeof(int)));
  checkCuda(cudaMemcpy(d_deleteKeys, deleteKeys.data(),
                       deleteSize * sizeof(int), cudaMemcpyHostToDevice));

  // Temporary buffer for located node positions
  int *d_deletePositions;
  checkCuda(cudaMalloc(&d_deletePositions, deleteSize * sizeof(int)));

  int blockSize = 256;
  int numBlocks = (deleteSize + blockSize - 1) / blockSize;

  // Phase 1: Read-only traversal to locate targets (no races)
  locateDeleteTargets<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deleteKeys,
                                                deleteSize, d_deletePositions);
  checkCuda(cudaDeviceSynchronize());

  // Phase 2: Apply deletions + mark avail using precomputed positions (no
  // traversal)
  applyDeletes<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deletePositions,
                                         deleteSize, ctx.d_avail);
  checkCuda(cudaDeviceSynchronize());

  checkCuda(cudaFree(d_deletePositions));
  checkCuda(cudaFree(d_deleteKeys));

  // Bottom-up level-wise reduction to recompute subtreeAvail
  int lastLevelStart = 1;
  while (lastLevelStart * 2 <= ctx.numRecords)
    lastLevelStart <<= 1;
  int levelStart = lastLevelStart - 1;
  if (levelStart >= ctx.numRecords)
    levelStart = (lastLevelStart >> 1) - 1;

  for (int levelEnd = ctx.numRecords - 1; levelStart >= 0;) {
    int count = levelEnd - levelStart + 1;
    int blocks = (count + blockSize - 1) / blockSize;
    reduceAvailLevel<<<blocks, blockSize>>>(
        levelStart, levelEnd, ctx.numRecords, ctx.d_avail, ctx.d_subtreeAvail);
    checkCuda(cudaDeviceSynchronize());
    if (levelStart == 0)
      break;
    levelEnd = levelStart - 1;
    levelStart = (levelStart - 1) / 2;
  }
}

InsertMapping insertCBST(const std::vector<int> &newKeys,
                         const std::vector<int> &newPayload,
                         const std::vector<int> &newPrefixSizes,
                         CBSTContext &ctx) {
  InsertMapping mapping;
  int K = static_cast<int>(newKeys.size());
  mapping.itemToKey.resize(K, 0);
  if (K == 0)
    return mapping;

  int blockSize = 256;

  // Copy inputs to device
  checkCuda(cudaMemcpy(ctx.d_insertKeys, newKeys.data(), K * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_insertPayload, newPayload.data(),
                       newPayload.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, newPrefixSizes.data(),
                       K * sizeof(int), cudaMemcpyHostToDevice));

  // Determine number of deleted slots (root's subtreeAvail)
  int D = 0;
  checkCuda(
      cudaMemcpy(&D, ctx.d_subtreeAvail, sizeof(int), cudaMemcpyDeviceToHost));

  int reuseK = std::min(K, D);

  // ── GPU Best-Fit Matching Pipeline ──────────────────────────────────
  int matchCount = 0;
  std::vector<int> h_matchedItemIdx; // host copy for surplus computation
  std::vector<int> h_deletedKeys;    // host copy of original keys for mapping
  int *d_allPositions = nullptr;
  int *d_matchedItemIdx = nullptr;
  int *d_matchedSlotIdx = nullptr;
  int *d_deletedKeys = nullptr;

  if (D > 0 && reuseK > 0) {
    // Locate ALL D deleted slots via order-statistic tree
    checkCuda(cudaMalloc(&d_allPositions, D * sizeof(int)));
    int numBlocksD = (D + blockSize - 1) / blockSize;
    locateReusableSlots<<<numBlocksD, blockSize>>>(
        ctx.d_subtreeAvail, ctx.d_avail, ctx.numRecords, d_allPositions, D);
    checkCuda(cudaDeviceSynchronize());

    // Recover original keys of deleted slots via CBST layout formula
    checkCuda(cudaMalloc(&d_deletedKeys, D * sizeof(int)));
    extractKeysFromPositions<<<numBlocksD, blockSize>>>(
        ctx.d_keys, d_allPositions, d_deletedKeys, ctx.numRecords, D);
    checkCuda(cudaDeviceSynchronize());

    // D2H: transfer deleted keys for mapping construction
    h_deletedKeys.resize(D);
    checkCuda(cudaMemcpy(h_deletedKeys.data(), d_deletedKeys, D * sizeof(int),
                         cudaMemcpyDeviceToHost));

    // Step 1: Extract slot capacities (GPU parallel)
    int *d_slotCaps;
    checkCuda(cudaMalloc(&d_slotCaps, D * sizeof(int)));
    extractSlotCapacities<<<numBlocksD, blockSize>>>(
        ctx.d_nodes, d_allPositions, d_slotCaps, D);
    checkCuda(cudaDeviceSynchronize());

    // Step 2: Compute item sizes for eligible items (GPU parallel)
    int *d_itemSizes;
    checkCuda(cudaMalloc(&d_itemSizes, reuseK * sizeof(int)));
    int numBlocksR = (reuseK + blockSize - 1) / blockSize;
    computeItemSizes<<<numBlocksR, blockSize>>>(ctx.d_insertPrefixSizes,
                                                d_itemSizes, reuseK);
    checkCuda(cudaDeviceSynchronize());

    // Step 3: Sort slot capacities in-place (no need to track original indices)
    thrust::device_ptr<int> caps_ptr = thrust::device_pointer_cast(d_slotCaps);
    thrust::sort(caps_ptr, caps_ptr + D);

    // Step 4: Sort item sizes with original-index tracking
    int *d_itemSortIdx;
    checkCuda(cudaMalloc(&d_itemSortIdx, reuseK * sizeof(int)));
    thrust::device_ptr<int> sortIdx_ptr =
        thrust::device_pointer_cast(d_itemSortIdx);
    thrust::sequence(sortIdx_ptr, sortIdx_ptr + reuseK);
    thrust::device_ptr<int> sizes_ptr =
        thrust::device_pointer_cast(d_itemSizes);
    thrust::sort_by_key(sizes_ptr, sizes_ptr + reuseK, sortIdx_ptr);

    // Step 5: Binary search — lo[i] = first slot with capacity >=
    // sorted_size[i]
    int *d_lo;
    checkCuda(cudaMalloc(&d_lo, reuseK * sizeof(int)));
    lowerBoundKernel<<<numBlocksR, blockSize>>>(d_slotCaps, D, d_itemSizes,
                                                reuseK, d_lo);
    checkCuda(cudaDeviceSynchronize());

    // Step 6: b[i] = lo[i] - i  (in-place, d_lo becomes d_b)
    computeBInPlace<<<numBlocksR, blockSize>>>(d_lo, reuseK);
    checkCuda(cudaDeviceSynchronize());

    // Step 7: prefix_max = inclusive_scan(b, max)
    int *d_prefixMax;
    checkCuda(cudaMalloc(&d_prefixMax, reuseK * sizeof(int)));
    thrust::device_ptr<int> b_ptr = thrust::device_pointer_cast(d_lo);
    thrust::device_ptr<int> pmax_ptr = thrust::device_pointer_cast(d_prefixMax);
    thrust::inclusive_scan(b_ptr, b_ptr + reuseK, pmax_ptr,
                           thrust::maximum<int>());

    // Step 8: assigned[i] = i + prefix_max[i]
    int *d_assigned;
    checkCuda(cudaMalloc(&d_assigned, reuseK * sizeof(int)));
    computeAssigned<<<numBlocksR, blockSize>>>(d_prefixMax, d_assigned, reuseK);
    checkCuda(cudaDeviceSynchronize());

    // Step 9: Count matched items (assigned[i] < D)
    thrust::device_ptr<int> assigned_ptr =
        thrust::device_pointer_cast(d_assigned);
    matchCount = static_cast<int>(
        thrust::count_if(assigned_ptr, assigned_ptr + reuseK, LessThan{D}));

#ifdef ESCHER_DEBUG_INSERT
    printf("[%s] GPU best-fit: %d/%d eligible items matched to deleted slots "
           "(D=%d)\n",
           ctx.datasetName, matchCount, reuseK, D);
#endif

    if (matchCount > 0) {
      // Step 10: Extract matched original item indices (GPU parallel)
      checkCuda(cudaMalloc(&d_matchedItemIdx, matchCount * sizeof(int)));
      thrust::device_ptr<int> matchOut_ptr =
          thrust::device_pointer_cast(d_matchedItemIdx);
      thrust::copy_if(sortIdx_ptr, sortIdx_ptr + reuseK, assigned_ptr,
                      matchOut_ptr, LessThan{D});

      // Step 11: Sort matched indices to restore key order (BST constraint)
      thrust::sort(matchOut_ptr, matchOut_ptr + matchCount);

      // Step 12: Matched slot indices = [0, 1, ..., matchCount-1] (BST order)
      checkCuda(cudaMalloc(&d_matchedSlotIdx, matchCount * sizeof(int)));
      thrust::device_ptr<int> slotOut_ptr =
          thrust::device_pointer_cast(d_matchedSlotIdx);
      thrust::sequence(slotOut_ptr, slotOut_ptr + matchCount);

      // D2H: transfer matched item indices for surplus computation + mapping
      h_matchedItemIdx.resize(matchCount);
      checkCuda(cudaMemcpy(h_matchedItemIdx.data(), d_matchedItemIdx,
                           matchCount * sizeof(int), cudaMemcpyDeviceToHost));

      // Build mapping for matched items:
      // matchedItemIndices[k] → slot k → deletedKeys[k]
      for (int k = 0; k < matchCount; ++k) {
        mapping.itemToKey[h_matchedItemIdx[k]] = h_deletedKeys[k];
      }
    }

    // Free intermediate GPU buffers
    checkCuda(cudaFree(d_slotCaps));
    checkCuda(cudaFree(d_itemSizes));
    checkCuda(cudaFree(d_itemSortIdx));
    checkCuda(cudaFree(d_lo));
    checkCuda(cudaFree(d_prefixMax));
    checkCuda(cudaFree(d_assigned));

    // ── Degree-binned applyReuse dispatch ─────────────────────────────
    if (matchCount > 0) {
      checkCuda(
          cudaMemset(ctx.d_relocationPlan, 0, matchCount * 3 * sizeof(int)));

      // Compute per-item payload sizes for binning
      std::vector<int> h_prefixSizes(K);
      checkCuda(cudaMemcpy(h_prefixSizes.data(), ctx.d_insertPrefixSizes,
                           K * sizeof(int), cudaMemcpyDeviceToHost));

      std::vector<int> reuseSmall, reuseMed, reuseLarge;
      for (int k = 0; k < matchCount; ++k) {
        int itemIdx = h_matchedItemIdx[k];
        int len = (itemIdx == 0)
                      ? h_prefixSizes[0]
                      : h_prefixSizes[itemIdx] - h_prefixSizes[itemIdx - 1];
        if (len < 32)
          reuseSmall.push_back(k);
        else if (len < 1024)
          reuseMed.push_back(k);
        else
          reuseLarge.push_back(k);
      }

      // Small bin: thread-per-item (original applyReuse)
      if (!reuseSmall.empty()) {
        // For the small bin, we use the original applyReuse which processes
        // all matchCount items. We need to create a subset. However, the
        // original kernel expects contiguous matchedItemIndices. Instead,
        // since the small bin indices map directly into the matched arrays,
        // we can launch the original kernel for all items when all are small,
        // or use a binned approach.
        // For simplicity and consistency, we always use the binned approach:
        // the thread-level kernel IS the original applyReuse, so we just
        // launch it for all matchCount. We only use warp/block for upgrades.
      }

      // If all items are small (common case), just launch the original kernel
      if (reuseMed.empty() && reuseLarge.empty()) {
        int numBlocksM = (matchCount + blockSize - 1) / blockSize;
        applyReuse<<<numBlocksM, blockSize>>>(
            ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
            ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes,
            ctx.d_relocationPlan, d_matchedItemIdx, d_matchedSlotIdx,
            d_deletedKeys, matchCount);
        checkCuda(cudaDeviceSynchronize());
      } else {
        // Mixed bins: launch separate kernels
        // Small bin
        if (!reuseSmall.empty()) {
          int n = static_cast<int>(reuseSmall.size());
          // For small bin, we need a subset. Build temp arrays.
          // Actually, the original applyReuse uses tid as index into matched
          // arrays. We need to create device arrays of just small-bin matched
          // indices.
          int *d_smallMatchItem, *d_smallMatchSlot;
          checkCuda(cudaMalloc(&d_smallMatchItem, n * sizeof(int)));
          checkCuda(cudaMalloc(&d_smallMatchSlot, n * sizeof(int)));
          // Extract small-bin items from matched arrays
          std::vector<int> h_smallItems(n), h_smallSlots(n);
          std::vector<int> h_allMatchItems(matchCount),
              h_allMatchSlots(matchCount);
          checkCuda(cudaMemcpy(h_allMatchItems.data(), d_matchedItemIdx,
                               matchCount * sizeof(int),
                               cudaMemcpyDeviceToHost));
          checkCuda(cudaMemcpy(h_allMatchSlots.data(), d_matchedSlotIdx,
                               matchCount * sizeof(int),
                               cudaMemcpyDeviceToHost));
          for (int i = 0; i < n; ++i) {
            h_smallItems[i] = h_allMatchItems[reuseSmall[i]];
            h_smallSlots[i] = h_allMatchSlots[reuseSmall[i]];
          }
          checkCuda(cudaMemcpy(d_smallMatchItem, h_smallItems.data(),
                               n * sizeof(int), cudaMemcpyHostToDevice));
          checkCuda(cudaMemcpy(d_smallMatchSlot, h_smallSlots.data(),
                               n * sizeof(int), cudaMemcpyHostToDevice));
          int blocks = (n + blockSize - 1) / blockSize;
          applyReuse<<<blocks, blockSize>>>(
              ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
              ctx.d_insertKeys, ctx.d_insertPayload, ctx.d_insertPrefixSizes,
              ctx.d_relocationPlan, d_smallMatchItem, d_smallMatchSlot,
              d_deletedKeys, n);
          checkCuda(cudaDeviceSynchronize());
          checkCuda(cudaFree(d_smallMatchItem));
          checkCuda(cudaFree(d_smallMatchSlot));
        }
        // Medium bin: warp-per-item
        if (!reuseMed.empty()) {
          int n = static_cast<int>(reuseMed.size());
          int *d_binIdx;
          checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
          checkCuda(cudaMemcpy(d_binIdx, reuseMed.data(), n * sizeof(int),
                               cudaMemcpyHostToDevice));
          int totalThreads = n * 32;
          int blocks = (totalThreads + blockSize - 1) / blockSize;
          applyReuse_warp<<<blocks, blockSize>>>(
              ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
              ctx.d_insertPayload, ctx.d_insertPrefixSizes,
              ctx.d_relocationPlan, d_matchedItemIdx, d_matchedSlotIdx,
              d_deletedKeys, d_binIdx, n);
          checkCuda(cudaDeviceSynchronize());
          checkCuda(cudaFree(d_binIdx));
        }
        // Large bin: block-per-item
        if (!reuseLarge.empty()) {
          int n = static_cast<int>(reuseLarge.size());
          int *d_binIdx;
          checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
          checkCuda(cudaMemcpy(d_binIdx, reuseLarge.data(), n * sizeof(int),
                               cudaMemcpyHostToDevice));
          applyReuse_block<<<n, blockSize>>>(
              ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
              ctx.d_insertPayload, ctx.d_insertPrefixSizes,
              ctx.d_relocationPlan, d_matchedItemIdx, d_matchedSlotIdx,
              d_deletedKeys, d_binIdx, n);
          checkCuda(cudaDeviceSynchronize());
          checkCuda(cudaFree(d_binIdx));
        }
      }

      checkCuda(cudaFree(d_matchedItemIdx));
      checkCuda(cudaFree(d_matchedSlotIdx));
      d_matchedItemIdx = nullptr;
      d_matchedSlotIdx = nullptr;
    }
  }

  if (d_allPositions)
    checkCuda(cudaFree(d_allPositions));
  if (d_deletedKeys)
    checkCuda(cudaFree(d_deletedKeys));

  // ── Build surplus list ──────────────────────────────────────────────
  // Unmatched items from the first reuseK plus all items beyond reuseK.
  std::vector<int> surplusIndices;
  {
    int matchPtr = 0;
    for (int i = 0; i < K; ++i) {
      if (matchPtr < matchCount && h_matchedItemIdx[matchPtr] == i) {
        matchPtr++;
      } else {
        surplusIndices.push_back(i);
      }
    }
  }
  int surplus = static_cast<int>(surplusIndices.size());

  // ── Surplus inserts: append at tail, then reconstruct ───────────────
  if (surplus > 0) {
    auto nextMultiple = [](int num, int a) {
      if (num <= 0)
        return 0;
      int q = (num + a - 1) / a;
      return q * a;
    };
    std::vector<int> appendedOffsets;
    appendedOffsets.reserve(surplus);
    int cursor = ctx.initialPayloadSize;
    for (int s = 0; s < surplus; ++s) {
      int globalIdx = surplusIndices[s];
      int start = (globalIdx == 0) ? 0 : newPrefixSizes[globalIdx - 1];
      int end = newPrefixSizes[globalIdx];
      int len = end - start;
      int aligned = nextMultiple(len, ctx.alignment);
      int base = cursor;
      int neededEnd = base + aligned + 1;
      if (neededEnd > ctx.fixedSize) {
        throw ::escher::EscherError(
            std::string("insertCBST [") +
            (ctx.datasetName ? ctx.datasetName : "?") +
            "]: surplus insert exceeds payload capacity (" +
            std::to_string(neededEnd) + " > " +
            std::to_string(ctx.fixedSize) +
            "). Increase payloadCapacity.");
      }
      appendedOffsets.push_back(base);
      if (len > 0) {
        checkCuda(cudaMemcpy(ctx.d_flatPayload + base,
                             newPayload.data() + start, len * sizeof(int),
                             cudaMemcpyHostToDevice));
      }
      if (aligned > len) {
        checkCuda(cudaMemset(ctx.d_flatPayload + base + len, 0,
                             (aligned - len) * sizeof(int)));
      }
      int sentinel = INT_MIN;
      checkCuda(cudaMemcpy(ctx.d_flatPayload + base + aligned, &sentinel,
                           sizeof(int), cudaMemcpyHostToDevice));
      cursor += aligned + 1;
    }
    ctx.initialPayloadSize = cursor;

    // Reconstruct CBST from valid (non-deleted) nodes + surplus
    int oldN = ctx.numRecords;

    int *d_idx, *d_val;
    checkCuda(cudaMalloc(&d_idx, oldN * sizeof(int)));
    checkCuda(cudaMalloc(&d_val, oldN * sizeof(int)));
    int blocksDump = (oldN + blockSize - 1) / blockSize;
    dumpNodeIndexValue<<<blocksDump, blockSize>>>(ctx.d_nodes, oldN, d_idx,
                                                  d_val);
    checkCuda(cudaDeviceSynchronize());
    std::vector<int> h_idx(oldN), h_val(oldN);
    checkCuda(cudaMemcpy(h_idx.data(), d_idx, oldN * sizeof(int),
                         cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_val.data(), d_val, oldN * sizeof(int),
                         cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(d_idx));
    checkCuda(cudaFree(d_val));

    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(oldN);
    for (int i = 0; i < oldN; ++i) {
      if (h_idx[i] > 0)
        pairs.emplace_back(h_idx[i], h_val[i]);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.first < b.first;
              });

    int validOldCount = static_cast<int>(pairs.size());
    int newN = validOldCount + surplus;

    // ── Option A: Preserve original keys (no compaction) ────────────
    // Surviving nodes keep their original keys.  Surplus items get keys
    // beyond the current maximum so no external references are invalidated.
    std::vector<int> h_newKeys(newN);
    std::vector<int> h_newStarts(newN);
    for (int i = 0; i < validOldCount; ++i) {
      h_newKeys[i] = pairs[i].first; // original key preserved
      h_newStarts[i] = pairs[i].second;
    }
    int nextKey = (validOldCount > 0) ? h_newKeys[validOldCount - 1] + 1 : 1;
    for (int i = 0; i < surplus; ++i) {
      h_newKeys[validOldCount + i] = nextKey;
      h_newStarts[validOldCount + i] = appendedOffsets[i];
      // Build mapping for surplus items
      mapping.itemToKey[surplusIndices[i]] = nextKey;
      nextKey++;
    }

    // Free old device arrays
    if (ctx.d_keys)
      checkCuda(cudaFree(ctx.d_keys));
    if (ctx.d_startOffsets)
      checkCuda(cudaFree(ctx.d_startOffsets));
    if (ctx.d_nodes)
      checkCuda(cudaFree(ctx.d_nodes));
    if (ctx.d_avail)
      checkCuda(cudaFree(ctx.d_avail));
    if (ctx.d_subtreeAvail)
      checkCuda(cudaFree(ctx.d_subtreeAvail));
    if (ctx.d_insertKeys)
      checkCuda(cudaFree(ctx.d_insertKeys));
    if (ctx.d_insertPayload)
      checkCuda(cudaFree(ctx.d_insertPayload));
    if (ctx.d_insertPrefixSizes)
      checkCuda(cudaFree(ctx.d_insertPrefixSizes));
    if (ctx.d_relocationPlan)
      checkCuda(cudaFree(ctx.d_relocationPlan));

    ctx.numRecords = newN;
    checkCuda(cudaMalloc(&ctx.d_nodes, newN * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&ctx.d_keys, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_startOffsets, newN * sizeof(int)));
    checkCuda(cudaMemcpy(ctx.d_keys, h_newKeys.data(), newN * sizeof(int),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ctx.d_startOffsets, h_newStarts.data(),
                         newN * sizeof(int), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&ctx.d_avail, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_subtreeAvail, newN * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_avail, 0, newN * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_subtreeAvail, 0, newN * sizeof(int)));

    checkCuda(cudaMalloc(&ctx.d_insertKeys, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_insertPayload, newN * 3 * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_relocationPlan, newN * 3 * sizeof(int)));

    int blocksBuild = (newN + blockSize - 1) / blockSize;
    buildEmptyBinaryTree<<<blocksBuild, blockSize>>>(ctx.d_nodes, newN);
    checkCuda(cudaDeviceSynchronize());
    storeItemsIntoNodes<<<blocksBuild, blockSize>>>(ctx.d_nodes, ctx.d_keys,
                                                    ctx.d_startOffsets, newN,
                                                    ctx.initialPayloadSize);
    checkCuda(cudaDeviceSynchronize());
  } else {
    // No surplus: mapping for matched items was already populated above.
    // Any items that were NOT matched and NOT surplus don't exist (K == 0
    // or all items matched), so the mapping is complete.
  }

  // Recompute subtreeAvail bottom-up
  int blockNodes = (ctx.numRecords + blockSize - 1) / blockSize;
  reduceAvailLevel<<<blockNodes, blockSize>>>(
      ctx.numRecords - 1, ctx.numRecords - 1, ctx.numRecords, ctx.d_avail,
      ctx.d_subtreeAvail);
  checkCuda(cudaDeviceSynchronize());
  int lastLevelStart = 1;
  while (lastLevelStart * 2 <= ctx.numRecords)
    lastLevelStart <<= 1;
  int levelStart = lastLevelStart - 1;
  if (levelStart >= ctx.numRecords)
    levelStart = (lastLevelStart >> 1) - 1;
  for (int levelEnd = ctx.numRecords - 1; levelStart >= 0;
       levelStart = (levelStart - 1) / 2) {
    int start = levelStart;
    int end = levelEnd;
    int count = end - start + 1;
    int blocks = (count + blockSize - 1) / blockSize;
    reduceAvailLevel<<<blocks, blockSize>>>(start, end, ctx.numRecords,
                                            ctx.d_avail, ctx.d_subtreeAvail);
    checkCuda(cudaDeviceSynchronize());
    if (levelStart == 0)
      break;
    levelEnd = levelStart - 1;
    levelStart = (levelStart - 1) / 2;
  }

  return mapping;
}

// CBSTOperations implementation
CBSTOperations::CBSTOperations(const char *datasetName, int payloadCapacity,
                               int alignment) {
  ctx_.datasetName = datasetName;
  ctx_.fixedSize = payloadCapacity;
  ctx_.alignment = alignment;
}

CBSTOperations::~CBSTOperations() {
  if (ctx_.d_insertKeys)
    checkCuda(cudaFree(ctx_.d_insertKeys));
  if (ctx_.d_insertPayload)
    checkCuda(cudaFree(ctx_.d_insertPayload));
  if (ctx_.d_insertPrefixSizes)
    checkCuda(cudaFree(ctx_.d_insertPrefixSizes));
  if (ctx_.d_relocationPlan)
    checkCuda(cudaFree(ctx_.d_relocationPlan));
  if (ctx_.d_keys)
    checkCuda(cudaFree(ctx_.d_keys));
  if (ctx_.d_startOffsets)
    checkCuda(cudaFree(ctx_.d_startOffsets));
  if (ctx_.d_nodes)
    checkCuda(cudaFree(ctx_.d_nodes));
  if (ctx_.d_flatPayload)
    checkCuda(cudaFree(ctx_.d_flatPayload));
  if (ctx_.d_avail)
    checkCuda(cudaFree(ctx_.d_avail));
  if (ctx_.d_subtreeAvail)
    checkCuda(cudaFree(ctx_.d_subtreeAvail));
}

CBSTOperations::CBSTOperations(CBSTOperations &&other) noexcept {
  ctx_ = other.ctx_;
  constructed_ = other.constructed_;
  // Null out other's pointers to avoid double free
  other.ctx_.d_nodes = nullptr;
  other.ctx_.d_keys = nullptr;
  other.ctx_.d_startOffsets = nullptr;
  other.ctx_.d_flatPayload = nullptr;
  other.ctx_.d_insertKeys = nullptr;
  other.ctx_.d_insertPayload = nullptr;
  other.ctx_.d_insertPrefixSizes = nullptr;
  other.ctx_.d_relocationPlan = nullptr;
  other.ctx_.d_avail = nullptr;
  other.ctx_.d_subtreeAvail = nullptr;
  other.constructed_ = false;
}

CBSTOperations &CBSTOperations::operator=(CBSTOperations &&other) noexcept {
  if (this != &other) {
    // Free current resources
    this->~CBSTOperations();
    // Steal other's resources
    ctx_ = other.ctx_;
    constructed_ = other.constructed_;
    // Null out other's pointers
    other.ctx_.d_nodes = nullptr;
    other.ctx_.d_keys = nullptr;
    other.ctx_.d_startOffsets = nullptr;
    other.ctx_.d_flatPayload = nullptr;
    other.ctx_.d_insertKeys = nullptr;
    other.ctx_.d_insertPayload = nullptr;
    other.ctx_.d_insertPrefixSizes = nullptr;
    other.ctx_.d_relocationPlan = nullptr;
    other.ctx_.d_avail = nullptr;
    other.ctx_.d_subtreeAvail = nullptr;
    other.constructed_ = false;
  }
  return *this;
}

void CBSTOperations::construct(int *keys, int *startOffsets, int numRecords,
                               int *flatPayload, int flatPayloadSize) {
  constructCBST(keys, startOffsets, numRecords, flatPayload, flatPayloadSize,
                ctx_.fixedSize, ctx_.datasetName, ctx_);
  constructed_ = true;
}

InsertMapping
CBSTOperations::insert(const std::vector<int> &insertKeys,
                       const std::vector<int> &insertPayload,
                       const std::vector<int> &insertPrefixSizes) {
  return insertCBST(insertKeys, insertPayload, insertPrefixSizes, ctx_);
}

void CBSTOperations::fill(const std::vector<int> &insertKeys,
                          const std::vector<int> &insertPayload,
                          const std::vector<int> &insertPrefixSizes) {
  fillCBST(insertKeys, insertPayload, insertPrefixSizes, ctx_);
}

void CBSTOperations::erase(const std::vector<int> &deleteKeys) {
  deleteCBST(deleteKeys, ctx_);
}

void CBSTOperations::findAndPrint(const std::vector<int> &ids) const {
  if (ids.empty())
    return;
  int *d_search;
  checkCuda(cudaMalloc(&d_search, ids.size() * sizeof(int)));
  checkCuda(cudaMemcpy(d_search, ids.data(), ids.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  findContents<<<(ids.size() + 256 - 1) / 256, 256>>>(
      ctx_.d_nodes, d_search, ids.size(), ctx_.d_flatPayload);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(d_search));
}

const CBSTContext &CBSTOperations::context() const { return ctx_; }

void unfillCBST(const std::vector<int> &keysToUnfill,
                const std::vector<int> &valuesToRemove,
                const std::vector<int> &removePrefixSizes, CBSTContext &ctx) {
  if (keysToUnfill.empty())
    return;
  // Reuse insert buffers for passing inputs
  checkCuda(cudaMemcpy(ctx.d_insertKeys, keysToUnfill.data(),
                       keysToUnfill.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_insertPayload, valuesToRemove.data(),
                       valuesToRemove.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(ctx.d_insertPrefixSizes, removePrefixSizes.data(),
                       removePrefixSizes.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  int K = static_cast<int>(keysToUnfill.size());
  int blockSize = 256;

  // ── Degree-binned unfill dispatch ────────────────────────────────────
  // We need per-node occupancy to bin by work size. Read it from device.
  // For simplicity, we bin by removal count (end - start per item) since
  // occupancy requires a device read per node. The removal count is a
  // good proxy: more removals = more work per element.
  std::vector<int> smallBin, medBin, largeBin;
  for (int i = 0; i < K; ++i) {
    int numRemovals = (i == 0)
                          ? removePrefixSizes[0]
                          : removePrefixSizes[i] - removePrefixSizes[i - 1];
    // Use removal count as a proxy for work. The actual segment scan
    // work is O(segment_length), but without reading node metadata to
    // host, we approximate using a fixed threshold.
    // For small removal sets, the inner loop is short -> thread is fine.
    // For larger sets, cooperative processing helps.
    if (numRemovals < 32)
      smallBin.push_back(i);
    else if (numRemovals < 1024)
      medBin.push_back(i);
    else
      largeBin.push_back(i);
  }

  if (!smallBin.empty()) {
    int n = static_cast<int>(smallBin.size());
    int *d_binIdx;
    checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
    checkCuda(cudaMemcpy(d_binIdx, smallBin.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    int blocks = (n + blockSize - 1) / blockSize;
    unfill_thread<<<blocks, blockSize>>>(ctx.d_nodes, ctx.d_flatPayload,
                                         ctx.d_insertKeys, ctx.d_insertPayload,
                                         ctx.d_insertPrefixSizes, d_binIdx, n);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_binIdx));
  }
  if (!medBin.empty()) {
    int n = static_cast<int>(medBin.size());
    int *d_binIdx;
    checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
    checkCuda(cudaMemcpy(d_binIdx, medBin.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    int totalThreads = n * 32;
    int blocks = (totalThreads + blockSize - 1) / blockSize;
    unfill_warp<<<blocks, blockSize>>>(ctx.d_nodes, ctx.d_flatPayload,
                                       ctx.d_insertKeys, ctx.d_insertPayload,
                                       ctx.d_insertPrefixSizes, d_binIdx, n);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_binIdx));
  }
  if (!largeBin.empty()) {
    int n = static_cast<int>(largeBin.size());
    int *d_binIdx;
    checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
    checkCuda(cudaMemcpy(d_binIdx, largeBin.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    size_t shmem = sizeof(int) * static_cast<size_t>(blockSize + 1);
    unfill_block<<<n, blockSize, shmem>>>(ctx.d_nodes, ctx.d_flatPayload,
                                          ctx.d_insertKeys, ctx.d_insertPayload,
                                          ctx.d_insertPrefixSizes, d_binIdx, n);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_binIdx));
  }
}
