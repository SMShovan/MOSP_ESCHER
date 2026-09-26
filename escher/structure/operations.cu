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
#include "../kernel/device_utils.cuh"

// Functor for thrust::copy_if on a 0/1 stencil
struct IsSet {
  __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// Local CUDA error checker for this TU. Delegates to the public
// escher::checkCudaImpl helper so failures surface as EscherError
// exceptions rather than abrupt process exits.
static inline void checkCuda(cudaError_t result) {
  ::escher::checkCudaImpl(result, __FILE__, __LINE__, "checkCuda");
}

// Set each node's occupancy to the true number of data values stored in its
// segment at construction time. Mirrors the tid -> index2 rank mapping of
// storeItemsIntoNodes so occupancy[rank] lands on the right tree node.
__global__ void setInitialOccupancy(CBSTNode *nodes, const int *rowOccupancy,
                                    int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    int index2 = cbstRankOfPosition(tid, n);
    if (index2 < n) {
      nodes[tid].occupancy = rowOccupancy[index2];
    }
  }
}

// Grow the reusable scratch buffers (d_insertKeys / d_insertPayload /
// d_insertPrefixSizes / d_relocationPlan) so a batch of @p K items with
// @p payloadInts total payload values fits. The upstream code allocated
// these once in constructCBST sized to the *initial record count*, so any
// fill/insert/unfill batch larger than that corrupted device memory.
static void ensureScratchCapacity(CBSTContext &ctx, int K,
                                  long long payloadInts) {
  if (K <= ctx.scratchKeysCap &&
      payloadInts <= ctx.scratchPayloadCap) {
    return;
  }
  int newKeysCap = ctx.scratchKeysCap > 0 ? ctx.scratchKeysCap : 1;
  while (newKeysCap < K) newKeysCap += newKeysCap / 2 + 1;
  long long newPayloadCap =
      ctx.scratchPayloadCap > 0 ? ctx.scratchPayloadCap : 4;
  while (newPayloadCap < payloadInts) newPayloadCap += newPayloadCap / 2 + 1;

  if (ctx.d_insertKeys) checkCuda(cudaFree(ctx.d_insertKeys));
  if (ctx.d_insertPayload) checkCuda(cudaFree(ctx.d_insertPayload));
  if (ctx.d_insertPrefixSizes) checkCuda(cudaFree(ctx.d_insertPrefixSizes));
  if (ctx.d_relocationPlan) checkCuda(cudaFree(ctx.d_relocationPlan));

  checkCuda(cudaMalloc(&ctx.d_insertKeys, newKeysCap * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_insertPayload,
                       static_cast<size_t>(newPayloadCap) * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, newKeysCap * sizeof(int)));
  checkCuda(
      cudaMalloc(&ctx.d_relocationPlan, 3LL * newKeysCap * sizeof(int)));

  ctx.scratchKeysCap = newKeysCap;
  ctx.scratchPayloadCap = newPayloadCap;
}

// Recomputes subtreeAvail (number of deleted slots per subtree) bottom-up,
// one launch per tree level, so a level's parents read children that the
// previous launch finished.
static void recomputeSubtreeAvail(CBSTContext &ctx) {
  const int blockSize = 256;
  int lastLevelStart = 1;
  while (lastLevelStart * 2 <= ctx.numRecords)
    lastLevelStart <<= 1;
  int levelStart = lastLevelStart - 1;
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

void constructCBST(int *keys, int *startOffsets, int numRecords,
                   int *flatPayload, int flatPayloadSize, int payloadCapacity,
                   const char *datasetName, CBSTContext &ctx,
                   const int *rowOccupancy) {
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
  checkCuda(cudaMalloc(&ctx.d_insertPayload,
                       static_cast<size_t>(numRecords) * 3 * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_insertPrefixSizes, numRecords * sizeof(int)));
  checkCuda(cudaMalloc(&ctx.d_relocationPlan,
                       3LL * numRecords * sizeof(int)));
  ctx.scratchKeysCap = numRecords;
  ctx.scratchPayloadCap = static_cast<long long>(numRecords) * 3;
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

  // Initialize per-node occupancy from the caller-provided true row counts
  // so subsequent fillCBST appends land after the construct-time data
  // instead of overwriting it (see structure.hpp for background).
  if (rowOccupancy != nullptr) {
    int *d_rowOcc = nullptr;
    checkCuda(cudaMalloc(&d_rowOcc, numRecords * sizeof(int)));
    checkCuda(cudaMemcpy(d_rowOcc, rowOccupancy, numRecords * sizeof(int),
                         cudaMemcpyHostToDevice));
    setInitialOccupancy<<<numBlocks, blockSize>>>(ctx.d_nodes, d_rowOcc,
                                                  numRecords);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_rowOcc));
  }

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

  ensureScratchCapacity(ctx, K,
                        static_cast<long long>(insertPayload.size()));
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
  recomputeSubtreeAvail(ctx);
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
  ensureScratchCapacity(ctx, K, static_cast<long long>(newPayload.size()));
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
  std::vector<char> matched(K, 0);

  // ── GPU Best-Fit Matching Pipeline ──────────────────────────────────
  if (D > 0 && reuseK > 0) {
    // Locate ALL D deleted slots via order-statistic tree
    int *d_allPositions = nullptr, *d_deletedKeys = nullptr,
        *d_slotCaps = nullptr, *d_slotOrder = nullptr;
    checkCuda(cudaMalloc(&d_allPositions, D * sizeof(int)));
    checkCuda(cudaMalloc(&d_deletedKeys, D * sizeof(int)));
    checkCuda(cudaMalloc(&d_slotCaps, D * sizeof(int)));
    checkCuda(cudaMalloc(&d_slotOrder, D * sizeof(int)));
    int numBlocksD = (D + blockSize - 1) / blockSize;
    locateReusableSlots<<<numBlocksD, blockSize>>>(
        ctx.d_subtreeAvail, ctx.d_avail, ctx.numRecords, d_allPositions, D);

    // Recover original keys of deleted slots via CBST layout formula
    extractKeysFromPositions<<<numBlocksD, blockSize>>>(
        ctx.d_keys, d_allPositions, d_deletedKeys, ctx.numRecords, D);

    // Step 1: Extract slot capacities (GPU parallel)
    extractSlotCapacities<<<numBlocksD, blockSize>>>(
        ctx.d_nodes, d_allPositions, d_slotCaps, D);

    // Step 2: Compute item sizes for eligible items (GPU parallel)
    int *d_itemSizes = nullptr, *d_itemOrder = nullptr, *d_lo = nullptr,
        *d_prefixMax = nullptr, *d_assigned = nullptr;
    checkCuda(cudaMalloc(&d_itemSizes, reuseK * sizeof(int)));
    checkCuda(cudaMalloc(&d_itemOrder, reuseK * sizeof(int)));
    checkCuda(cudaMalloc(&d_lo, reuseK * sizeof(int)));
    checkCuda(cudaMalloc(&d_prefixMax, reuseK * sizeof(int)));
    checkCuda(cudaMalloc(&d_assigned, reuseK * sizeof(int)));
    int numBlocksR = (reuseK + blockSize - 1) / blockSize;
    computeItemSizes<<<numBlocksR, blockSize>>>(ctx.d_insertPrefixSizes,
                                                d_itemSizes, reuseK);
    checkCuda(cudaGetLastError());

    // Step 3: Sort slot capacities, remembering which slot each one is.
    // (The original sorted the capacities alone, lost the slot identity,
    // and then gave the k-th matched item the k-th slot in BST order,
    // truncating items larger than that slot.)
    thrust::device_ptr<int> caps_ptr = thrust::device_pointer_cast(d_slotCaps);
    thrust::device_ptr<int> slotOrder_ptr =
        thrust::device_pointer_cast(d_slotOrder);
    thrust::sequence(slotOrder_ptr, slotOrder_ptr + D);
    thrust::sort_by_key(caps_ptr, caps_ptr + D, slotOrder_ptr);

    // Step 4: Sort item sizes with original-index tracking
    thrust::device_ptr<int> itemOrder_ptr =
        thrust::device_pointer_cast(d_itemOrder);
    thrust::sequence(itemOrder_ptr, itemOrder_ptr + reuseK);
    thrust::device_ptr<int> sizes_ptr = thrust::device_pointer_cast(d_itemSizes);
    thrust::sort_by_key(sizes_ptr, sizes_ptr + reuseK, itemOrder_ptr);

    // Step 5: Binary search — lo[i] = first slot with capacity >=
    // sorted_size[i]
    lowerBoundKernel<<<numBlocksR, blockSize>>>(d_slotCaps, D, d_itemSizes,
                                                reuseK, d_lo);

    // Step 6: b[i] = lo[i] - i  (in-place, d_lo becomes d_b)
    computeBInPlace<<<numBlocksR, blockSize>>>(d_lo, reuseK);
    checkCuda(cudaGetLastError());

    // Step 7: prefix_max = inclusive_scan(b, max)
    thrust::device_ptr<int> b_ptr = thrust::device_pointer_cast(d_lo);
    thrust::device_ptr<int> pmax_ptr = thrust::device_pointer_cast(d_prefixMax);
    thrust::inclusive_scan(b_ptr, b_ptr + reuseK, pmax_ptr,
                           thrust::maximum<int>());

    // Step 8: assigned[i] = i + prefix_max[i] (rank of the slot for sorted
    // item i; strictly increasing, and >= lo[i] so the slot fits the item)
    computeAssigned<<<numBlocksR, blockSize>>>(d_prefixMax, d_assigned, reuseK);

    // Step 9: Count matched items (assigned[i] < D)
    thrust::device_ptr<int> assigned_ptr =
        thrust::device_pointer_cast(d_assigned);
    int matchCount = static_cast<int>(
        thrust::count_if(assigned_ptr, assigned_ptr + reuseK, LessThan{D}));

#ifdef ESCHER_DEBUG_INSERT
    printf("[%s] GPU best-fit: %d/%d eligible items matched to deleted slots "
           "(D=%d)\n",
           ctx.datasetName, matchCount, reuseK, D);
#endif

    if (matchCount > 0) {
      // Step 10: matched (item, slot) pairs: sorted item i -> slot of rank
      // assigned[i]
      int *d_matchedItemIdx = nullptr, *d_matchedSlotIdx = nullptr;
      checkCuda(cudaMalloc(&d_matchedItemIdx, matchCount * sizeof(int)));
      checkCuda(cudaMalloc(&d_matchedSlotIdx, matchCount * sizeof(int)));
      pairMatches<<<(matchCount + blockSize - 1) / blockSize, blockSize>>>(
          d_itemOrder, d_assigned, d_slotOrder, matchCount, d_matchedItemIdx,
          d_matchedSlotIdx);

      // D2H: matched pairs and slot keys for the mapping and the binning
      std::vector<int> h_items(matchCount), h_slots(matchCount), h_keys(D);
      checkCuda(cudaMemcpy(h_items.data(), d_matchedItemIdx,
                           matchCount * sizeof(int), cudaMemcpyDeviceToHost));
      checkCuda(cudaMemcpy(h_slots.data(), d_matchedSlotIdx,
                           matchCount * sizeof(int), cudaMemcpyDeviceToHost));
      checkCuda(cudaMemcpy(h_keys.data(), d_deletedKeys, D * sizeof(int),
                           cudaMemcpyDeviceToHost));
      std::vector<int> bins[3];
      for (int k = 0; k < matchCount; ++k) {
        int itemIdx = h_items[k];
        mapping.itemToKey[itemIdx] = h_keys[h_slots[k]];
        matched[itemIdx] = 1;
        int len = newPrefixSizes[itemIdx] -
                  (itemIdx == 0 ? 0 : newPrefixSizes[itemIdx - 1]);
        bins[len < 32 ? 0 : (len < 1024 ? 1 : 2)].push_back(k);
      }

      // ── Degree-binned applyReuse dispatch (thread / warp / block) ──────
      for (int b = 0; b < 3; ++b) {
        int n = static_cast<int>(bins[b].size());
        if (n == 0)
          continue;
        int *d_binIdx = nullptr;
        checkCuda(cudaMalloc(&d_binIdx, n * sizeof(int)));
        checkCuda(cudaMemcpy(d_binIdx, bins[b].data(), n * sizeof(int),
                             cudaMemcpyHostToDevice));
        if (b == 0)
          applyReuse<<<(n + blockSize - 1) / blockSize, blockSize>>>(
              ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
              ctx.d_insertPayload, ctx.d_insertPrefixSizes, d_matchedItemIdx,
              d_matchedSlotIdx, d_deletedKeys, d_binIdx, n);
        else if (b == 1)
          applyReuse_warp<<<(n * 32 + blockSize - 1) / blockSize, blockSize>>>(
              ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
              ctx.d_insertPayload, ctx.d_insertPrefixSizes, d_matchedItemIdx,
              d_matchedSlotIdx, d_deletedKeys, d_binIdx, n);
        else
          applyReuse_block<<<n, blockSize>>>(
              ctx.d_nodes, ctx.d_flatPayload, ctx.d_avail, d_allPositions,
              ctx.d_insertPayload, ctx.d_insertPrefixSizes, d_matchedItemIdx,
              d_matchedSlotIdx, d_deletedKeys, d_binIdx, n);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaFree(d_binIdx));
      }
      checkCuda(cudaFree(d_matchedItemIdx));
      checkCuda(cudaFree(d_matchedSlotIdx));
    }

    checkCuda(cudaFree(d_itemSizes));
    checkCuda(cudaFree(d_itemOrder));
    checkCuda(cudaFree(d_lo));
    checkCuda(cudaFree(d_prefixMax));
    checkCuda(cudaFree(d_assigned));
    checkCuda(cudaFree(d_allPositions));
    checkCuda(cudaFree(d_deletedKeys));
    checkCuda(cudaFree(d_slotCaps));
    checkCuda(cudaFree(d_slotOrder));
  }

  // ── Build surplus list ──────────────────────────────────────────────
  // Unmatched items from the first reuseK plus all items beyond reuseK.
  std::vector<int> surplusIndices;
  for (int i = 0; i < K; ++i)
    if (!matched[i])
      surplusIndices.push_back(i);
  int surplus = static_cast<int>(surplusIndices.size());

  // ── Surplus inserts: append at tail, then reconstruct ───────────────
  if (surplus > 0) {
    auto nextMultiple = [](int num, int a) {
      if (num <= 0)
        return 0;
      int q = (num + a - 1) / a;
      return q * a;
    };
    // Tail metadata of each surplus row: one segment holding len values,
    // zero padding up to the alignment, then the INT_MIN terminator.
    std::vector<CBSTNode> surplusRecords(surplus);
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
      CBSTNode &r = surplusRecords[s];
      r = CBSTNode{};
      r.value = base;
      r.length = aligned + 1;
      r.occupancy = len;
      r.tailBase = base;
      r.tailCapacity = aligned;
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

    // Reconstruct the CBST from the surviving (non-deleted) nodes plus the
    // surplus rows. Surviving nodes keep their full records (offset,
    // length, occupancy and tail segment); the original rebuilt them from
    // (key, offset) pairs with storeItemsIntoNodes, which reset occupancy
    // to 0 and the tail to the first segment, so the next fill overwrote
    // the row from its base and overflow chains were lost. Live nodes are
    // brought into key order through their in-order rank, then compacted.
    int oldN = ctx.numRecords;
    CBSTNode *d_ranked = nullptr, *d_records = nullptr;
    int *d_rankedLive = nullptr;
    checkCuda(cudaMalloc(&d_ranked, oldN * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&d_records,
                         static_cast<size_t>(oldN + surplus) * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&d_rankedLive, oldN * sizeof(int)));
    int blocksOld = (oldN + blockSize - 1) / blockSize;
    rankOrderNodes<<<blocksOld, blockSize>>>(ctx.d_nodes, ctx.d_avail, oldN,
                                             d_ranked, d_rankedLive);
    checkCuda(cudaGetLastError());
    thrust::device_ptr<CBSTNode> ranked_ptr(d_ranked), records_ptr(d_records);
    thrust::device_ptr<int> live_ptr(d_rankedLive);
    int validOldCount = static_cast<int>(
        thrust::copy_if(ranked_ptr, ranked_ptr + oldN, live_ptr, records_ptr,
                        IsSet()) -
        records_ptr);

    // ── Option A: Preserve original keys (no compaction) ────────────
    // Surviving nodes keep their original keys.  Surplus items get keys
    // beyond the current maximum so no external references are invalidated.
    int nextKey = 1;
    if (validOldCount > 0) {
      int lastKey = 0;
      checkCuda(cudaMemcpy(&lastKey, &d_records[validOldCount - 1].index,
                           sizeof(int), cudaMemcpyDeviceToHost));
      nextKey = lastKey + 1;
    }
    for (int i = 0; i < surplus; ++i) {
      surplusRecords[i].index = nextKey;
      surplusRecords[i].size = ctx.initialPayloadSize;
      mapping.itemToKey[surplusIndices[i]] = nextKey;
      nextKey++;
    }
    checkCuda(cudaMemcpy(d_records + validOldCount, surplusRecords.data(),
                         surplus * sizeof(CBSTNode), cudaMemcpyHostToDevice));
    int newN = validOldCount + surplus;

    // Free old device arrays
    checkCuda(cudaFree(ctx.d_keys));
    checkCuda(cudaFree(ctx.d_startOffsets));
    checkCuda(cudaFree(ctx.d_nodes));
    checkCuda(cudaFree(ctx.d_avail));
    checkCuda(cudaFree(ctx.d_subtreeAvail));

    ctx.numRecords = newN;
    checkCuda(cudaMalloc(&ctx.d_nodes, newN * sizeof(CBSTNode)));
    checkCuda(cudaMalloc(&ctx.d_keys, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_startOffsets, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_avail, newN * sizeof(int)));
    checkCuda(cudaMalloc(&ctx.d_subtreeAvail, newN * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_avail, 0, newN * sizeof(int)));
    checkCuda(cudaMemset(ctx.d_subtreeAvail, 0, newN * sizeof(int)));

    int blocksBuild = (newN + blockSize - 1) / blockSize;
    recordKeysAndStarts<<<blocksBuild, blockSize>>>(d_records, newN,
                                                    ctx.d_keys,
                                                    ctx.d_startOffsets);
    buildEmptyBinaryTree<<<blocksBuild, blockSize>>>(ctx.d_nodes, newN);
    placeNodeRecords<<<blocksBuild, blockSize>>>(ctx.d_nodes, d_records, newN);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_ranked));
    checkCuda(cudaFree(d_records));
    checkCuda(cudaFree(d_rankedLive));
  } else {
    // No surplus: mapping for matched items was already populated above.
    // Any items that were NOT matched and NOT surplus don't exist (K == 0
    // or all items matched), so the mapping is complete.
  }

  // Recompute subtreeAvail bottom-up (reused slots are no longer
  // available). The original loop advanced levelStart both in the for
  // header and in the body, so each launch covered two tree levels and
  // parents read children being written in the same launch: the counts
  // went stale and the next insert located invalid slots (key 0,
  // duplicate keys).
  recomputeSubtreeAvail(ctx);

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
                               int *flatPayload, int flatPayloadSize,
                               const int *rowOccupancy) {
  constructCBST(keys, startOffsets, numRecords, flatPayload, flatPayloadSize,
                ctx_.fixedSize, ctx_.datasetName, ctx_, rowOccupancy);
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
  ensureScratchCapacity(ctx, static_cast<int>(keysToUnfill.size()),
                        static_cast<long long>(valuesToRemove.size()));
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
