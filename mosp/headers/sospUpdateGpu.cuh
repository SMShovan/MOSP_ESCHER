#ifndef SOSP_UPDATE_GPU_CUH
#define SOSP_UPDATE_GPU_CUH

/**
 * @file sospUpdateGpu.cuh
 * @brief Device-resident SOSP update engine (Steps 1 and 2 of the SOSP
 *        update, and SSSP from scratch for Step 3 of the MOSP update).
 *
 * All arrays are device pointers. Distances are 64-bit, parents 32-bit,
 * weights 32-bit. See sospUpdateGpu.cu for the algorithm.
 */

/// Distance of an unreachable vertex (the value the MOSP code writes).
constexpr long long DISTANCE_INF = 0x7fffffffffffffffLL / 4;

/// A CSR graph with one weight per edge, in device memory.
struct DeviceCsr {
  int numberOfNodes = 0;
  int numberOfEdges = 0;
  const int *rowPtr = nullptr;
  const int *colInd = nullptr;
  const int *weights = nullptr;
};

/// The change batch as seen by one objective, in device memory.
struct DeviceChanges {
  /// Edges (from[i], to[i]) that were deleted or whose weight increased;
  /// the head of such an edge is a root if the edge is its tree edge.
  const int *changedFrom = nullptr;
  const int *changedTo = nullptr;
  int numberOfChanged = 0;
  /// Heads of inserted edges (their distance may decrease).
  const int *insertHeads = nullptr;
  int numberOfInsertHeads = 0;
};

/// Counters reported by one update.
struct SospStats {
  int invalidated = 0;  ///< vertices in invalidated subtrees
  int iterations = 0;   ///< near-far push iterations
  int epochs = 0;       ///< far-pile threshold increases
  long long pushes = 0; ///< vertex expansions
};

/**
 * @brief Device scratch space for updates on graphs with up to
 *        capacity vertices; reserve once and reuse for every objective
 *        and for the combined graph (no allocation inside the updates).
 */
struct SospWorkspace {
  SospWorkspace() = default;
  ~SospWorkspace();
  SospWorkspace(const SospWorkspace &) = delete;
  SospWorkspace &operator=(const SospWorkspace &) = delete;

  /// Allocate for @p capacity vertices (no-op if already large enough).
  bool reserve(int capacity);
  /// Fresh stamp generation (stamps deduplicate list insertions).
  int nextGeneration();
  void release();

  int capacity = 0;
  int generation = 0;
  unsigned long long *packed = nullptr; ///< (distance << b | parent) words
  int *stamp = nullptr;                 ///< last generation a vertex was listed
  int *inFar = nullptr;                 ///< vertex is in the far pile
  int *flag = nullptr;                  ///< invalidation marks
  int *ancestor = nullptr;              ///< pointer-jumping ancestors
  int *listA = nullptr, *listB = nullptr, *farA = nullptr, *farB = nullptr;
  int *candidates = nullptr;
  int *counters = nullptr;              ///< device counters
  unsigned long long *minimum = nullptr;
  int *hostCounters = nullptr;          ///< pinned mirror of the counters
  unsigned long long *hostMinimum = nullptr;
};

/**
 * @brief Default near-far bucket width: 32 * average weight / average
 *        out-degree (at least 1).
 */
long long defaultDelta(long long numberOfEdges, int numberOfNodes,
                       long long weightSum);

/**
 * @brief Incremental SOSP update.
 *
 * On entry @p d_distances / @p d_parent hold the tree of the old graph
 * (canonical: ties broken by the lowest parent id); on return they hold
 * the canonical tree of the new graph, whose out- and in-edges are
 * @p out and @p in. Unreachable vertices get DISTANCE_INF and parent -1.
 *
 * @param maxWeight Largest edge weight (bounds the distances for packing).
 * @param delta     Near-far bucket width (> 0).
 */
bool sospUpdateGpu(const DeviceCsr &out, const DeviceCsr &in,
                   const DeviceChanges &changes, int source, long long delta,
                   long long maxWeight, SospWorkspace &workspace,
                   long long *d_distances, int *d_parent,
                   SospStats *stats = nullptr);

/**
 * @brief Single-source shortest paths from scratch (canonical tree).
 */
bool sospFromScratchGpu(const DeviceCsr &out, int source, long long delta,
                        long long maxWeight, SospWorkspace &workspace,
                        long long *d_distances, int *d_parent,
                        SospStats *stats = nullptr);

#endif // SOSP_UPDATE_GPU_CUH
