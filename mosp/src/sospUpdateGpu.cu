/**
 * @file sospUpdateGpu.cu
 * @brief Work-efficient, disconnection-safe SOSP update on the GPU.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Step 1 (from the change list, grouped by destination):
 *   - Roots: the head v of every deleted or weight-increased edge (u,v)
 *     with Parent[v] == u.
 *   - Subtree invalidation: pointer jumping over the parent array marks
 *     every descendant of a root (ceil(log2 n) rounds, no host sync). The
 *     marked vertices lose their distance (INF) and parent.
 *   - Pull pass: every invalidated vertex and every head of an inserted
 *     edge takes the best (distance, parent id) pair over its in-neighbours
 *     (a destination-grouped Step 1: one thread per destination).
 *
 * Step 2 (propagation) is a push-based near-far worklist (a Delta-stepping
 * variant, Davidson et al., IPDPS'14): vertices whose distance improved
 * relax their out-edges with a 64-bit atomicMin on the packed pair
 * (distance << b | parent). A vertex whose new distance is below the
 * current threshold joins the next near frontier, otherwise the far pile;
 * when the near frontier is empty the threshold moves to the smallest far
 * distance plus Delta.
 *
 * Properties:
 *   - Monotone: a distance only decreases, and every distance is an upper
 *     bound (valid vertices keep an intact tree path, invalidated ones
 *     start at INF). The loop terminates without an iteration cap, cannot
 *     count to infinity through a stale cycle, and vertices cut off from
 *     the source keep INF (no reachability post-pass).
 *   - Canonical: the packed atomicMin keeps, among equal distances, the
 *     lowest parent id. Every in-neighbour whose final distance differs
 *     from its old one pushes that final value, the pull pass covers the
 *     unchanged in-neighbours of invalidated vertices and insert heads,
 *     so a canonical input tree gives the canonical output tree.
 *   - Work-efficient: only improved vertices are expanded (no candidate
 *     re-scan of all in-edges, no O(n) reset per iteration).
 *
 * Packing: b = number of bits needed for the vertex ids plus a "no parent"
 * value; the remaining 64 - b bits hold the distance, and the all-ones word
 * is INF. The distance bound (n - 1) * maxWeight must fit (checked).
 * ============================================================================
 */

#include "sospUpdateGpu.cuh"


#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <iostream>

using namespace std;

namespace {

using u64 = unsigned long long;
constexpr u64 PACKED_INF = ~0ULL;
constexpr int BLOCK_SIZE = 256;

// Device counter slots.
constexpr int NEAR_A = 0;     // near-frontier output (even iterations)
constexpr int FAR_COUNT = 1;  // far pile size
constexpr int NEAR_B = 2;     // near-frontier output (odd iterations)
constexpr int LIST_COUNT = 3; // candidate / frontier list size
constexpr int OVERFLOW = 4;   // a distance does not fit the packing
constexpr int NUM_COUNTERS = 8;

#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "            \
           << __FILE__ << ":" << __LINE__ << "\n";                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

int blocks(long long work) {
  return static_cast<int>((max(work, 1LL) + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

/// Packed (distance, parent) words.
struct Packing {
  int parentBits;
  u64 noParent; // all-ones parent field

  __host__ __device__ u64 pack(u64 distance, int parent) const {
    return (distance << parentBits) |
           (parent < 0 ? noParent : static_cast<u64>(parent));
  }
  __device__ u64 distance(u64 word) const { return word >> parentBits; }
  __device__ int parent(u64 word) const {
    u64 p = word & noParent;
    return p == noParent ? -1 : static_cast<int>(p);
  }
  /// Largest distance that can be stored (the all-ones field is INF).
  u64 maxDistance() const { return (PACKED_INF >> parentBits) - 1; }
};

Packing makePacking(int numberOfNodes) {
  int bits = 1;
  while ((1ULL << bits) - 1 < static_cast<u64>(numberOfNodes)) {
    ++bits;
  }
  return {bits, (1ULL << bits) - 1};
}

__device__ __forceinline__ bool claim(int *stamp, int v, int generation) {
  return atomicExch(&stamp[v], generation) != generation;
}

// ---------------------------------------------------------------------------
// Step 1 kernels
// ---------------------------------------------------------------------------

__global__ void packKernel(int n, const long long *distances,
                           const int *parent, u64 *packed, Packing packing,
                           u64 maxDistance, int *counters) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) {
    return;
  }
  long long d = distances[v];
  if (d >= DISTANCE_INF / 2) {
    packed[v] = PACKED_INF;
    return;
  }
  if (d < 0 || static_cast<u64>(d) > maxDistance) {
    counters[OVERFLOW] = 1;
    packed[v] = PACKED_INF;
    return;
  }
  packed[v] = packing.pack(static_cast<u64>(d), parent[v]);
}

__global__ void unpackKernel(int n, const u64 *packed, long long *distances,
                             int *parent, Packing packing) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) {
    return;
  }
  u64 word = packed[v];
  if (word == PACKED_INF) {
    distances[v] = DISTANCE_INF;
    parent[v] = -1;
    return;
  }
  distances[v] = static_cast<long long>(packing.distance(word));
  parent[v] = packing.parent(word);
}

/// Roots: heads of deleted or weight-increased tree edges.
__global__ void markRootsKernel(const int *from, const int *to, int count,
                                const int *parent, int *flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && parent[to[i]] == from[i]) {
    flag[to[i]] = 1;
  }
}

__global__ void initAncestorsKernel(int n, const int *parent, int *ancestor) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) {
    ancestor[v] = parent[v];
  }
}

/**
 * One pointer-jumping round. Invariant for every vertex x: flag[x] == 1
 * implies a root among x and its ancestors; flag[x] == 0 implies no root
 * on the tree path from x up to (excluding) ancestor[x]. A round either
 * inherits the ancestor's flag or jumps to the ancestor's ancestor, which
 * at least doubles the covered distance, so ceil(log2 n) rounds suffice.
 * Rounds update in place: every (ancestor, flag) value a thread can
 * observe satisfies the invariant, so the races are benign.
 */
__global__ void pointerJumpKernel(int n, int *ancestor, int *flag) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) {
    return;
  }
  int a = ancestor[v];
  if (a < 0 || flag[v]) {
    return;
  }
  if (flag[a]) {
    flag[v] = 1;
    return;
  }
  ancestor[v] = ancestor[a];
}

/// Invalidate flagged vertices and list them as candidates.
__global__ void invalidateKernel(int n, int *flag, u64 *packed, int *stamp,
                                 int generation, int *list, int *counters) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n || !flag[v]) {
    return;
  }
  flag[v] = 0; // leave the flag array clean for the next update
  packed[v] = PACKED_INF;
  stamp[v] = generation;
  list[atomicAdd(&counters[LIST_COUNT], 1)] = v;
}

/// Add the heads of inserted edges to the candidate list (deduplicated).
__global__ void addCandidatesKernel(const int *vertices, int count, int source,
                                    int *stamp, int generation, int *list,
                                    int *counters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  int v = vertices[i];
  if (v != source && claim(stamp, v, generation)) {
    list[atomicAdd(&counters[LIST_COUNT], 1)] = v;
  }
}

/// Pull pass: candidate v takes the best (distance, id) over its
/// in-neighbours; improved vertices form the first frontier.
__global__ void pullKernel(const int *candidates, int count, DeviceCsr in,
                           u64 *packed, Packing packing, int *stamp,
                           int generation, int *frontier, int *counters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  int v = candidates[i];
  u64 current = __ldcg(&packed[v]);
  u64 best = current;
  for (int e = in.rowPtr[v]; e < in.rowPtr[v + 1]; ++e) {
    int u = in.colInd[e];
    u64 word = __ldcg(&packed[u]);
    if (word == PACKED_INF) {
      continue;
    }
    u64 candidate = packing.pack(packing.distance(word) + in.weights[e], u);
    best = min(best, candidate);
  }
  if (best < current) {
    u64 old = atomicMin(&packed[v], best);
    if (packing.distance(best) < packing.distance(old) &&
        claim(stamp, v, generation)) {
      frontier[atomicAdd(&counters[LIST_COUNT], 1)] = v;
    }
  }
}

// ---------------------------------------------------------------------------
// Step 2 kernels (near-far push)
// ---------------------------------------------------------------------------

/**
 * Relax the out-edges of the near frontier. An improved head joins the next
 * near frontier (distance below the threshold, deduplicated by stamp) or
 * the far pile (deduplicated by the inFar flag). Thread 0 also zeroes the
 * counter the next iteration will write to.
 */
__global__ void pushKernel(const int *nearList, int nearCount, DeviceCsr out,
                           u64 *packed, Packing packing, int *stamp,
                           int generation, int *nextNear, int nearSlot,
                           int *inFar, int *far, u64 threshold, int source,
                           int resetSlot, int *counters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    counters[resetSlot] = 0;
  }
  if (i >= nearCount) {
    return;
  }
  int u = nearList[i];
  u64 word = __ldcg(&packed[u]);
  if (word == PACKED_INF) {
    return;
  }
  u64 du = packing.distance(word);
  for (int e = out.rowPtr[u]; e < out.rowPtr[u + 1]; ++e) {
    int w = out.colInd[e];
    if (w == source) {
      continue;
    }
    u64 nd = du + out.weights[e];
    u64 candidate = packing.pack(nd, u);
    if (candidate >= __ldcg(&packed[w])) {
      continue;
    }
    u64 old = atomicMin(&packed[w], candidate);
    if (nd < packing.distance(old)) {
      if (nd < threshold) {
        if (claim(stamp, w, generation)) {
          nextNear[atomicAdd(&counters[nearSlot], 1)] = w;
        }
      } else if (atomicExch(&inFar[w], 1) == 0) {
        far[atomicAdd(&counters[FAR_COUNT], 1)] = w;
      }
    }
  }
}

/// Smallest distance over a vertex list (warp-reduced atomicMin).
__global__ void minDistanceKernel(const int *list, int count,
                                  const u64 *packed, Packing packing,
                                  u64 *minimum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  u64 d = PACKED_INF;
  if (i < count) {
    u64 word = __ldcg(&packed[list[i]]);
    if (word != PACKED_INF) {
      d = packing.distance(word);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    d = min(d, __shfl_down_sync(0xffffffffu, d, offset));
  }
  if ((threadIdx.x & 31) == 0 && d != PACKED_INF) {
    atomicMin(minimum, d);
  }
}

/**
 * Split a list by the threshold: below goes to the near frontier, the rest
 * to the far pile. For the first split (fromFar = false) the far pile is
 * deduplicated with the inFar flag; when re-splitting the far pile the
 * flags are already set and are cleared for vertices leaving it.
 */
__global__ void splitKernel(const int *list, int count, const u64 *packed,
                            Packing packing, u64 threshold, int *stamp,
                            int generation, int *nearList, int nearSlot,
                            int *inFar, int *far, bool fromFar,
                            int *counters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  int v = list[i];
  u64 word = __ldcg(&packed[v]);
  u64 d = word == PACKED_INF ? PACKED_INF : packing.distance(word);
  if (d < threshold) {
    if (fromFar) {
      inFar[v] = 0;
    }
    if (claim(stamp, v, generation)) {
      nearList[atomicAdd(&counters[nearSlot], 1)] = v;
    }
  } else if (fromFar || atomicExch(&inFar[v], 1) == 0) {
    far[atomicAdd(&counters[FAR_COUNT], 1)] = v;
  }
}

__global__ void initFromScratchKernel(int n, u64 *packed, int source,
                                      Packing packing) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) {
    packed[v] = v == source ? packing.pack(0, -1) : PACKED_INF;
  }
}

} // namespace

// ============================================================================
// Workspace
// ============================================================================

SospWorkspace::~SospWorkspace() { release(); }

void SospWorkspace::release() {
  cudaFree(packed);
  cudaFree(stamp);
  cudaFree(inFar);
  cudaFree(flag);
  cudaFree(ancestor);
  cudaFree(listA);
  cudaFree(listB);
  cudaFree(farA);
  cudaFree(farB);
  cudaFree(candidates);
  cudaFree(counters);
  cudaFree(minimum);
  cudaFreeHost(hostCounters);
  cudaFreeHost(hostMinimum);
  packed = nullptr;
  stamp = inFar = flag = ancestor = nullptr;
  listA = listB = farA = farB = candidates = counters = nullptr;
  minimum = nullptr;
  hostCounters = nullptr;
  hostMinimum = nullptr;
  capacity = 0;
  generation = 0;
}

bool SospWorkspace::reserve(int requested) {
  if (requested <= capacity) {
    return true;
  }
  release();
  const size_t n = static_cast<size_t>(max(requested, 1));
  GPU_CHECK(cudaMalloc(&packed, n * sizeof(u64)));
  GPU_CHECK(cudaMalloc(&stamp, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&inFar, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&flag, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&ancestor, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&listA, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&listB, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&farA, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&farB, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&candidates, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&counters, NUM_COUNTERS * sizeof(int)));
  GPU_CHECK(cudaMalloc(&minimum, sizeof(u64)));
  GPU_CHECK(cudaMallocHost(&hostCounters, NUM_COUNTERS * sizeof(int)));
  GPU_CHECK(cudaMallocHost(&hostMinimum, sizeof(u64)));
  GPU_CHECK(cudaMemset(stamp, 0, n * sizeof(int)));
  GPU_CHECK(cudaMemset(inFar, 0, n * sizeof(int)));
  GPU_CHECK(cudaMemset(flag, 0, n * sizeof(int)));
  capacity = requested;
  generation = 0;
  return true;
}

int SospWorkspace::nextGeneration() {
  if (generation == INT_MAX) {
    cudaMemset(stamp, 0, static_cast<size_t>(capacity) * sizeof(int));
    generation = 0;
  }
  return ++generation;
}

long long defaultDelta(long long numberOfEdges, int numberOfNodes,
                       long long weightSum) {
  if (numberOfEdges <= 0 || numberOfNodes <= 0) {
    return 1;
  }
  // 32 * (average weight) / (average out-degree)
  double averageWeight = static_cast<double>(weightSum) / numberOfEdges;
  double averageDegree = static_cast<double>(numberOfEdges) / numberOfNodes;
  return max(1LL, static_cast<long long>(32.0 * averageWeight / averageDegree));
}

// ============================================================================
// Near-far loop (shared by the update and the from-scratch search)
// ============================================================================

namespace {

/**
 * Run the near-far propagation from the vertices in @p frontier (a list in
 * ws.listB). One blocking host synchronization per push iteration and two
 * per threshold increase.
 */
bool nearFar(const DeviceCsr &out, int source, u64 delta, Packing packing,
             SospWorkspace &ws, int frontierCount, SospStats &stats) {
  if (frontierCount == 0) {
    return true;
  }
  int *frontier = ws.listB;
  // Threshold = smallest frontier distance + delta.
  *ws.hostMinimum = PACKED_INF;
  GPU_CHECK(cudaMemcpy(ws.minimum, ws.hostMinimum, sizeof(u64),
                       cudaMemcpyHostToDevice));
  minDistanceKernel<<<blocks(frontierCount), BLOCK_SIZE>>>(
      frontier, frontierCount, ws.packed, packing, ws.minimum);
  GPU_CHECK(cudaMemcpy(ws.hostMinimum, ws.minimum, sizeof(u64),
                       cudaMemcpyDeviceToHost));
  u64 threshold =
      (*ws.hostMinimum == PACKED_INF ? 0 : *ws.hostMinimum) + delta;

  int *current = ws.candidates, *next = ws.listA;
  int *far = ws.farA, *far2 = ws.farB;
  GPU_CHECK(cudaMemset(ws.counters, 0, NUM_COUNTERS * sizeof(int)));
  splitKernel<<<blocks(frontierCount), BLOCK_SIZE>>>(
      frontier, frontierCount, ws.packed, packing, threshold, ws.stamp,
      ws.nextGeneration(), current, NEAR_A, ws.inFar, far, false,
      ws.counters);
  GPU_CHECK(cudaMemcpy(ws.hostCounters, ws.counters, 3 * sizeof(int),
                       cudaMemcpyDeviceToHost));
  int nearCount = ws.hostCounters[NEAR_A];
  int farCount = ws.hostCounters[FAR_COUNT];
  int nearSlot = NEAR_A;

  while (nearCount > 0 || farCount > 0) {
    while (nearCount > 0) {
      ++stats.iterations;
      stats.pushes += nearCount;
      // The kernel appends to counters[outSlot] (zeroed by the previous
      // iteration) and zeroes counters[nearSlot] for the next one, so each
      // iteration needs a single blocking copy of the counters.
      const int outSlot = nearSlot == NEAR_A ? NEAR_B : NEAR_A;
      pushKernel<<<blocks(nearCount), BLOCK_SIZE>>>(
          current, nearCount, out, ws.packed, packing, ws.stamp,
          ws.nextGeneration(), next, outSlot, ws.inFar, far, threshold,
          source, nearSlot, ws.counters);
      GPU_CHECK(cudaGetLastError());
      GPU_CHECK(cudaMemcpy(ws.hostCounters, ws.counters, 3 * sizeof(int),
                           cudaMemcpyDeviceToHost));
      nearCount = ws.hostCounters[outSlot];
      farCount = ws.hostCounters[FAR_COUNT];
      nearSlot = outSlot;
      swap(current, next);
    }
    if (farCount == 0) {
      break;
    }
    // Move the threshold past the smallest far distance and re-split.
    ++stats.epochs;
    *ws.hostMinimum = PACKED_INF;
    GPU_CHECK(cudaMemcpy(ws.minimum, ws.hostMinimum, sizeof(u64),
                         cudaMemcpyHostToDevice));
    minDistanceKernel<<<blocks(farCount), BLOCK_SIZE>>>(
        far, farCount, ws.packed, packing, ws.minimum);
    GPU_CHECK(cudaMemcpy(ws.hostMinimum, ws.minimum, sizeof(u64),
                         cudaMemcpyDeviceToHost));
    threshold = max(threshold, *ws.hostMinimum) + delta;
    GPU_CHECK(cudaMemset(ws.counters, 0, 3 * sizeof(int)));
    splitKernel<<<blocks(farCount), BLOCK_SIZE>>>(
        far, farCount, ws.packed, packing, threshold, ws.stamp,
        ws.nextGeneration(), current, NEAR_A, ws.inFar, far2, true,
        ws.counters);
    GPU_CHECK(cudaMemcpy(ws.hostCounters, ws.counters, 3 * sizeof(int),
                         cudaMemcpyDeviceToHost));
    nearCount = ws.hostCounters[NEAR_A];
    farCount = ws.hostCounters[FAR_COUNT];
    nearSlot = NEAR_A;
    swap(far, far2);
  }
  return true;
}

} // namespace

// ============================================================================
// Public entry points
// ============================================================================

bool sospUpdateGpu(const DeviceCsr &out, const DeviceCsr &in,
                   const DeviceChanges &changes, int source, long long delta,
                   long long maxWeight, SospWorkspace &ws,
                   long long *d_distances, int *d_parent, SospStats *stats) {
  const int n = out.numberOfNodes;
  SospStats local;
  SospStats &s = stats != nullptr ? *stats : local;
  s = SospStats();
  if (n == 0) {
    return true;
  }
  if (!ws.reserve(n) || delta <= 0) {
    return false;
  }
  const Packing packing = makePacking(n);
  const u64 bound = static_cast<u64>(max(maxWeight, 1LL)) * (n - 1);
  if (bound > packing.maxDistance()) {
    cerr << "Error: distances up to " << bound << " do not fit the packed "
         << (64 - packing.parentBits) << "-bit distance field.\n";
    return false;
  }

  GPU_CHECK(cudaMemset(ws.counters, 0, NUM_COUNTERS * sizeof(int)));
  packKernel<<<blocks(n), BLOCK_SIZE>>>(n, d_distances, d_parent, ws.packed,
                                        packing, bound, ws.counters);

  // --- Step 1: roots and subtree invalidation -------------------------------
  if (changes.numberOfChanged > 0) {
    markRootsKernel<<<blocks(changes.numberOfChanged), BLOCK_SIZE>>>(
        changes.changedFrom, changes.changedTo, changes.numberOfChanged,
        d_parent, ws.flag);
    initAncestorsKernel<<<blocks(n), BLOCK_SIZE>>>(n, d_parent, ws.ancestor);
    int rounds = 0;
    while ((1LL << rounds) < n) {
      ++rounds;
    }
    for (int r = 0; r < rounds; ++r) {
      pointerJumpKernel<<<blocks(n), BLOCK_SIZE>>>(n, ws.ancestor, ws.flag);
    }
    const int generation = ws.nextGeneration();
    invalidateKernel<<<blocks(n), BLOCK_SIZE>>>(n, ws.flag, ws.packed,
                                                ws.stamp, generation,
                                                ws.candidates, ws.counters);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaMemcpy(ws.hostCounters, ws.counters,
                         NUM_COUNTERS * sizeof(int), cudaMemcpyDeviceToHost));
    s.invalidated = ws.hostCounters[LIST_COUNT];
    if (changes.numberOfInsertHeads > 0) {
      addCandidatesKernel<<<blocks(changes.numberOfInsertHeads), BLOCK_SIZE>>>(
          changes.insertHeads, changes.numberOfInsertHeads, source, ws.stamp,
          generation, ws.candidates, ws.counters);
    }
  } else if (changes.numberOfInsertHeads > 0) {
    addCandidatesKernel<<<blocks(changes.numberOfInsertHeads), BLOCK_SIZE>>>(
        changes.insertHeads, changes.numberOfInsertHeads, source, ws.stamp,
        ws.nextGeneration(), ws.candidates, ws.counters);
  }
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpy(ws.hostCounters, ws.counters,
                       NUM_COUNTERS * sizeof(int), cudaMemcpyDeviceToHost));
  if (ws.hostCounters[OVERFLOW] != 0) {
    cerr << "Error: an initial distance does not fit the packed format.\n";
    return false;
  }
  const int numberOfCandidates = ws.hostCounters[LIST_COUNT];

  // --- Step 1: pull pass over invalidated vertices and insert heads --------
  int frontierCount = 0;
  if (numberOfCandidates > 0) {
    GPU_CHECK(cudaMemset(ws.counters + LIST_COUNT, 0, sizeof(int)));
    pullKernel<<<blocks(numberOfCandidates), BLOCK_SIZE>>>(
        ws.candidates, numberOfCandidates, in, ws.packed, packing,
        ws.stamp, ws.nextGeneration(), ws.listB, ws.counters);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaMemcpy(ws.hostCounters, ws.counters,
                         NUM_COUNTERS * sizeof(int), cudaMemcpyDeviceToHost));
    frontierCount = ws.hostCounters[LIST_COUNT];
  }

  // --- Step 2: near-far propagation ----------------------------------------
  {
    if (!nearFar(out, source, static_cast<u64>(delta), packing, ws,
                 frontierCount, s)) {
      return false;
    }
  }
  unpackKernel<<<blocks(n), BLOCK_SIZE>>>(n, ws.packed, d_distances,
                                          d_parent, packing);
  GPU_CHECK(cudaGetLastError());
  return true;
}

bool sospFromScratchGpu(const DeviceCsr &out, int source, long long delta,
                        long long maxWeight, SospWorkspace &ws,
                        long long *d_distances, int *d_parent,
                        SospStats *stats) {
  const int n = out.numberOfNodes;
  SospStats local;
  SospStats &s = stats != nullptr ? *stats : local;
  s = SospStats();
  if (n == 0) {
    return true;
  }
  if (!ws.reserve(n) || delta <= 0 || source < 0 || source >= n) {
    return false;
  }
  const Packing packing = makePacking(n);
  const u64 bound = static_cast<u64>(max(maxWeight, 1LL)) * (n - 1);
  if (bound > packing.maxDistance()) {
    cerr << "Error: distances up to " << bound << " do not fit the packed "
         << (64 - packing.parentBits) << "-bit distance field.\n";
    return false;
  }
  initFromScratchKernel<<<blocks(n), BLOCK_SIZE>>>(n, ws.packed, source,
                                                   packing);
  GPU_CHECK(cudaMemcpy(ws.listB, &source, sizeof(int),
                       cudaMemcpyHostToDevice));
  if (!nearFar(out, source, static_cast<u64>(delta), packing, ws, 1, s)) {
    return false;
  }
  unpackKernel<<<blocks(n), BLOCK_SIZE>>>(n, ws.packed, d_distances,
                                          d_parent, packing);
  GPU_CHECK(cudaGetLastError());
  return true;
}
