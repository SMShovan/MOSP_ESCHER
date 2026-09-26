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

// Stream-ordered temporary buffer (cudaMallocAsync / cudaFreeAsync on the
// default stream). The operations used cudaMalloc / cudaFree for every
// temporary, and cudaFree synchronizes the device; with the pool, an
// operation queues its kernels without host round trips except where it
// reads a result back.
template <class T> class TempBuffer {
public:
  explicit TempBuffer(size_t n) {
    static const bool poolReady = [] {
      // Keep freed blocks in the pool instead of returning them to the
      // driver at every synchronization.
      int device = 0;
      cudaMemPool_t pool;
      if (cudaGetDevice(&device) == cudaSuccess &&
          cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
        unsigned long long threshold = 1ull << 30;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                &threshold);
      }
      return true;
    }();
    (void)poolReady;
    checkCuda(cudaMallocAsync(reinterpret_cast<void **>(&ptr_),
                              std::max<size_t>(n, 1) * sizeof(T), 0));
  }
  ~TempBuffer() { cudaFreeAsync(ptr_, 0); }
  TempBuffer(const TempBuffer &) = delete;
  TempBuffer &operator=(const TempBuffer &) = delete;
  T *get() const { return ptr_; }

private:
  T *ptr_ = nullptr;
};

// Degree binning of per-item work: items with fewer than 32 values run one
// per thread, fewer than 1024 one per warp, larger ones one per block. The
// three index lists are uploaded with one copy (the original allocated,
// uploaded, synchronized and freed each list separately).
struct DegreeBins {
  explicit DegreeBins(const std::vector<int> &sizes) : index(sizes.size()) {
    std::vector<int> lists[3];
    for (size_t i = 0; i < sizes.size(); ++i)
      lists[sizes[i] < 32 ? 0 : (sizes[i] < 1024 ? 1 : 2)].push_back(
          static_cast<int>(i));
    std::vector<int> all;
    all.reserve(sizes.size());
    for (int b = 0; b < 3; ++b) {
      count[b] = static_cast<int>(lists[b].size());
      all.insert(all.end(), lists[b].begin(), lists[b].end());
    }
    if (!all.empty())
      checkCuda(cudaMemcpyAsync(index.get(), all.data(),
                                all.size() * sizeof(int),
                                cudaMemcpyHostToDevice, 0));
  }
  int *bin(int b) const {
    return index.get() +
           (b == 0 ? 0 : (b == 1 ? count[0] : count[0] + count[1]));
  }
  TempBuffer<int> index;
  int count[3] = {0, 0, 0};
};

// Sizes of the items of an inclusive prefix-sum layout.
static std::vector<int> itemSizes(const std::vector<int> &prefixSizes) {
  std::vector<int> sizes(prefixSizes.size());
  for (size_t i = 0; i < prefixSizes.size(); ++i)
    sizes[i] = prefixSizes[i] - (i == 0 ? 0 : prefixSizes[i - 1]);
  return sizes;
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
// previous launch finished (stream order; no host synchronization).
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
    checkCuda(cudaGetLastError());
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
  // Zeroed so that host copies of the nodes (checks) never read bytes the
  // device did not write (struct padding).
  checkCuda(cudaMemset(ctx.d_nodes, 0, numRecords * sizeof(CBSTNode)));
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
    TempBuffer<int> d_rowOcc(numRecords);
    checkCuda(cudaMemcpy(d_rowOcc.get(), rowOccupancy,
                         numRecords * sizeof(int), cudaMemcpyHostToDevice));
    setInitialOccupancy<<<numBlocks, blockSize>>>(ctx.d_nodes, d_rowOcc.get(),
                                                  numRecords);
    checkCuda(cudaDeviceSynchronize());
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

  ensureScratchCapacity(ctx, K,
                        static_cast<long long>(insertPayload.size()));
  checkCuda(cudaMemcpyAsync(ctx.d_insertKeys, insertKeys.data(),
                            K * sizeof(int), cudaMemcpyHostToDevice, 0));
  checkCuda(cudaMemcpyAsync(ctx.d_insertPayload, insertPayload.data(),
                            insertPayload.size() * sizeof(int),
                            cudaMemcpyHostToDevice, 0));
  checkCuda(cudaMemcpyAsync(ctx.d_insertPrefixSizes, insertPrefixSizes.data(),
                            K * sizeof(int), cudaMemcpyHostToDevice, 0));
  checkCuda(
      cudaMemsetAsync(ctx.d_relocationPlan, 0, 3LL * K * sizeof(int), 0));

  int blockSize = 256;

  // ── Degree-binned insertNode dispatch ────────────────────────────────
  DegreeBins bins(itemSizes(insertPrefixSizes));
  if (int n = bins.count[0]) {
    insertNode_thread<<<(n + blockSize - 1) / blockSize, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, ctx.d_relocationPlan, bins.bin(0), n);
    checkCuda(cudaGetLastError());
  }
  if (int n = bins.count[1]) {
    insertNode_warp<<<(n * 32 + blockSize - 1) / blockSize, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, ctx.d_relocationPlan, bins.bin(1), n);
    checkCuda(cudaGetLastError());
  }
  if (int n = bins.count[2]) {
    insertNode_block<<<n, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, ctx.d_relocationPlan, bins.bin(2), n);
    checkCuda(cudaGetLastError());
  }

  // ── Overflow handling ────────────────────────────────────────────────
  TempBuffer<int> d_tmp(K);
  computeNextMultipleOf4<<<(K + blockSize - 1) / blockSize, blockSize>>>(
      ctx.d_relocationPlan, d_tmp.get(), K);
  checkCuda(cudaGetLastError());
  thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(d_tmp.get());
  thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
  updatePartialSolution<<<(K + blockSize - 1) / blockSize, blockSize>>>(
      ctx.d_relocationPlan, d_tmp.get(), K);
  checkCuda(cudaGetLastError());

  // Only the total appended size is needed on the host.
  int totalAppended = 0;
  checkCuda(cudaMemcpy(&totalAppended, ctx.d_relocationPlan + 3 * (K - 1) + 2,
                       sizeof(int), cudaMemcpyDeviceToHost));
#ifdef ESCHER_DEBUG_FILL
  std::vector<int> relocationPlanHostOut(K * 3);
  checkCuda(cudaMemcpy(relocationPlanHostOut.data(), ctx.d_relocationPlan,
                       K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
  printVector(relocationPlanHostOut, "Cumulative Relocation Plan");
#endif

  if (ctx.initialPayloadSize + totalAppended > ctx.fixedSize) {
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
  checkCuda(cudaGetLastError());

  // Fixup metadata for overflowed nodes
  fixupOverflowMetadata<<<numBlocks, blockSize>>>(
      ctx.d_nodes, ctx.d_insertKeys, ctx.d_insertPrefixSizes,
      ctx.d_relocationPlan, ctx.initialPayloadSize, K);
  checkCuda(cudaGetLastError());

  ctx.initialPayloadSize += totalAppended;

#ifdef ESCHER_DEBUG_FILL
  std::vector<int> updatedFlat(ctx.fixedSize);
  checkCuda(cudaMemcpy(updatedFlat.data(), ctx.d_flatPayload,
                       ctx.fixedSize * sizeof(int), cudaMemcpyDeviceToHost));
  printVector(updatedFlat, "Updated Flattened Values (vec1d)");
#endif
}

void deleteCBST(const std::vector<int> &deleteKeys, CBSTContext &ctx) {
  if (deleteKeys.empty())
    return;
  int deleteSize = static_cast<int>(deleteKeys.size());
  TempBuffer<int> d_deleteKeys(deleteSize);
  checkCuda(cudaMemcpyAsync(d_deleteKeys.get(), deleteKeys.data(),
                            deleteSize * sizeof(int), cudaMemcpyHostToDevice,
                            0));

  // Temporary buffer for located node positions
  TempBuffer<int> d_deletePositions(deleteSize);

  int blockSize = 256;
  int numBlocks = (deleteSize + blockSize - 1) / blockSize;

  // Phase 1: Read-only traversal to locate targets (no races)
  locateDeleteTargets<<<numBlocks, blockSize>>>(
      ctx.d_nodes, d_deleteKeys.get(), deleteSize, d_deletePositions.get());
  checkCuda(cudaGetLastError());

  // Phase 2: Apply deletions + mark avail using precomputed positions (no
  // traversal)
  applyDeletes<<<numBlocks, blockSize>>>(ctx.d_nodes, d_deletePositions.get(),
                                         deleteSize, ctx.d_avail);
  checkCuda(cudaGetLastError());

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
  checkCuda(cudaMemcpyAsync(ctx.d_insertKeys, newKeys.data(), K * sizeof(int),
                            cudaMemcpyHostToDevice, 0));
  checkCuda(cudaMemcpyAsync(ctx.d_insertPayload, newPayload.data(),
                            newPayload.size() * sizeof(int),
                            cudaMemcpyHostToDevice, 0));
  checkCuda(cudaMemcpyAsync(ctx.d_insertPrefixSizes, newPrefixSizes.data(),
                            K * sizeof(int), cudaMemcpyHostToDevice, 0));

  // Determine number of deleted slots (root's subtreeAvail)
  int D = 0;
  checkCuda(
      cudaMemcpy(&D, ctx.d_subtreeAvail, sizeof(int), cudaMemcpyDeviceToHost));

  int reuseK = std::min(K, D);
  std::vector<char> matched(K, 0);

  // ── GPU Best-Fit Matching Pipeline ──────────────────────────────────
  if (D > 0 && reuseK > 0) {
    // Locate ALL D deleted slots via order-statistic tree
    TempBuffer<int> allPositions(D), deletedKeys(D), slotCaps(D),
        slotOrder(D);
    int *d_allPositions = allPositions.get(),
        *d_deletedKeys = deletedKeys.get(), *d_slotCaps = slotCaps.get(),
        *d_slotOrder = slotOrder.get();
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
    TempBuffer<int> itemSizesBuf(reuseK), itemOrder(reuseK), lo(reuseK),
        prefixMax(reuseK), assigned(reuseK);
    int *d_itemSizes = itemSizesBuf.get(), *d_itemOrder = itemOrder.get(),
        *d_lo = lo.get(), *d_prefixMax = prefixMax.get(),
        *d_assigned = assigned.get();
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
      TempBuffer<int> matchedItemIdx(matchCount), matchedSlotIdx(matchCount);
      int *d_matchedItemIdx = matchedItemIdx.get(),
          *d_matchedSlotIdx = matchedSlotIdx.get();
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
      std::vector<int> matchSizes(matchCount);
      for (int k = 0; k < matchCount; ++k) {
        int itemIdx = h_items[k];
        mapping.itemToKey[itemIdx] = h_keys[h_slots[k]];
        matched[itemIdx] = 1;
        matchSizes[k] = newPrefixSizes[itemIdx] -
                        (itemIdx == 0 ? 0 : newPrefixSizes[itemIdx - 1]);
      }

      // ── Degree-binned applyReuse dispatch (thread / warp / block) ──────
      DegreeBins bins(matchSizes);
      for (int b = 0; b < 3; ++b) {
        int n = bins.count[b];
        if (n == 0)
          continue;
        int *d_binIdx = bins.bin(b);
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
        checkCuda(cudaGetLastError());
      }
    }
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
    // zero padding up to the alignment, then the INT_MIN terminator. The
    // rows are packed on the host and appended with one copy (the original
    // issued two or three copies per row: 220 ms for the 25K new rows of a
    // DBLP batch).
    std::vector<CBSTNode> surplusRecords(surplus);
    std::vector<int> packed;
    int cursor = ctx.initialPayloadSize;
    for (int s = 0; s < surplus; ++s) {
      int globalIdx = surplusIndices[s];
      int start = (globalIdx == 0) ? 0 : newPrefixSizes[globalIdx - 1];
      int end = newPrefixSizes[globalIdx];
      int len = end - start;
      int aligned = nextMultiple(len, ctx.alignment);
      int base = cursor;
      long long neededEnd = static_cast<long long>(base) + aligned + 1;
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
      packed.insert(packed.end(), newPayload.begin() + start,
                    newPayload.begin() + end);
      packed.insert(packed.end(), aligned - len, 0);
      packed.push_back(INT_MIN);
      cursor += aligned + 1;
    }
    checkCuda(cudaMemcpyAsync(ctx.d_flatPayload + ctx.initialPayloadSize,
                              packed.data(), packed.size() * sizeof(int),
                              cudaMemcpyHostToDevice, 0));
    ctx.initialPayloadSize = cursor;

    // Reconstruct the CBST from the surviving (non-deleted) nodes plus the
    // surplus rows. Surviving nodes keep their full records (offset,
    // length, occupancy and tail segment); the original rebuilt them from
    // (key, offset) pairs with storeItemsIntoNodes, which reset occupancy
    // to 0 and the tail to the first segment, so the next fill overwrote
    // the row from its base and overflow chains were lost. Live nodes are
    // brought into key order through their in-order rank, then compacted.
    int oldN = ctx.numRecords;
    TempBuffer<CBSTNode> ranked(oldN), records(oldN + surplus);
    TempBuffer<int> rankedLive(oldN);
    CBSTNode *d_ranked = ranked.get(), *d_records = records.get();
    int *d_rankedLive = rankedLive.get();
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
    checkCuda(cudaMemset(ctx.d_nodes, 0, newN * sizeof(CBSTNode)));
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
    checkCuda(cudaGetLastError());
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
  checkCuda(cudaMemcpyAsync(ctx.d_insertKeys, keysToUnfill.data(),
                            keysToUnfill.size() * sizeof(int),
                            cudaMemcpyHostToDevice, 0));
  checkCuda(cudaMemcpyAsync(ctx.d_insertPayload, valuesToRemove.data(),
                            valuesToRemove.size() * sizeof(int),
                            cudaMemcpyHostToDevice, 0));
  checkCuda(cudaMemcpyAsync(ctx.d_insertPrefixSizes, removePrefixSizes.data(),
                            removePrefixSizes.size() * sizeof(int),
                            cudaMemcpyHostToDevice, 0));
  int blockSize = 256;

  // ── Degree-binned unfill dispatch ────────────────────────────────────
  // Binned by the number of values to remove from the row (a proxy for
  // the work; the row length would need a read of the node metadata).
  DegreeBins bins(itemSizes(removePrefixSizes));
  if (int n = bins.count[0]) {
    unfill_thread<<<(n + blockSize - 1) / blockSize, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, bins.bin(0), n);
    checkCuda(cudaGetLastError());
  }
  if (int n = bins.count[1]) {
    unfill_warp<<<(n * 32 + blockSize - 1) / blockSize, blockSize>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, bins.bin(1), n);
    checkCuda(cudaGetLastError());
  }
  if (int n = bins.count[2]) {
    size_t shmem = sizeof(int) * static_cast<size_t>(blockSize + 1);
    unfill_block<<<n, blockSize, shmem>>>(
        ctx.d_nodes, ctx.d_flatPayload, ctx.d_insertKeys, ctx.d_insertPayload,
        ctx.d_insertPrefixSizes, bins.bin(2), n);
    checkCuda(cudaGetLastError());
  }
}
