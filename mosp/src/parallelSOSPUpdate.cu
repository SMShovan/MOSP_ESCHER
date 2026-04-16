/**
 * @file parallelSOSPUpdate.cu
 * @brief Parallel (CUDA) Single-Objective Shortest Path (SOSP) Update.
 *
 * This is the CUDA-parallel version of sequentialSOSPUpdate.cu.
 * It produces identical final results (distances + parent arrays) to the
 * sequential version — and therefore matches Dijkstra recalculation on the
 * updated graph.
 *
 * ============================================================================
 * PARALLELIZATION STRATEGY
 * ============================================================================
 *
 * Phases 0 and 1 (preparation + initial edge processing) remain sequential
 * on the host because they are I/O-dominated and operate on tiny batches.
 *
 * Phase 2 (iterative propagation) is parallelized with CUDA kernels:
 *   - collectCandidatesKernel: Each thread processes one affected vertex,
 *     iterates its out-neighbors, uses atomicCAS on flag arrays for
 *     deduplication, and atomicAdd on a global counter for worklist
 *     compaction.
 *   - updateDistancesKernel: Each thread processes one candidate vertex,
 *     scans all in-neighbors (findBestParent logic), updates distances
 *     and parent arrays, and marks newly affected vertices.
 *
 * Post-processing BFS (reachability check) uses level-synchronous parallel
 * BFS with atomicCAS on visited flags.
 *
 * Race Condition Analysis:
 *   - d_isCandidate[]/d_isAffected[]: atomicCAS deduplication
 *   - d_candidateList/d_affectedList: atomicAdd worklist compaction
 *   - d_distances[v]/d_parent[v] writes in updateDistancesKernel: no race
 *     because each candidate v appears exactly once (deduplicated)
 *   - d_distances[u] reads in findBestParent: benign race under Chaotic
 *     Bellman-Ford semantics — convergence guaranteed, same final result
 *   - d_reachable[] in BFS: atomicCAS deduplication
 *
 * ============================================================================
 */

#include "parallelSOSPUpdate.cuh"

#include "read.cuh"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      cerr << "CUDA error: " << cudaGetErrorString(err) << " at "             \
           << __FILE__ << ":" << __LINE__ << "\n";                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

// ============================================================================
// CUDA KERNELS
// ============================================================================

/**
 * @brief Collect candidate vertices from affected vertices' out-neighbors.
 *
 * Each thread processes one affected vertex, iterates its out-neighbors
 * in the CSR, and uses atomicCAS for deduplication and atomicAdd for
 * worklist compaction.
 *
 * @param d_affectedList    Array of affected vertex indices.
 * @param numAffected       Number of affected vertices.
 * @param d_outRowPtr       CSR row pointer for forward graph.
 * @param d_outColInd       CSR column indices for forward graph.
 * @param d_isCandidate     Flag array for deduplication (0/1).
 * @param d_candidateList   Output worklist of candidate vertices.
 * @param d_candidateCount  Atomic counter for candidateList size.
 * @param d_isAffected      Flag array for affected vertices (cleared here).
 * @param source            Source vertex (never becomes a candidate).
 */
__global__ void collectCandidatesKernel(
    const int *d_affectedList, int numAffected, const int *d_outRowPtr,
    const int *d_outColInd, int *d_isCandidate, int *d_candidateList,
    int *d_candidateCount, int *d_isAffected, int source) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numAffected)
    return;

  int affectedVertex = d_affectedList[tid];
  d_isAffected[affectedVertex] = 0; // Clear affected flag

  int rowStart = d_outRowPtr[affectedVertex];
  int rowEnd = d_outRowPtr[affectedVertex + 1];

  for (int e = rowStart; e < rowEnd; ++e) {
    int neighborVertex = d_outColInd[e];

    // CRITICAL: Never update the source vertex
    if (neighborVertex == source)
      continue;

    // Atomic test-and-set for deduplication
    int old = atomicCAS(&d_isCandidate[neighborVertex], 0, 1);
    if (old == 0) {
      int pos = atomicAdd(d_candidateCount, 1);
      d_candidateList[pos] = neighborVertex;
    }
  }
}

/**
 * @brief Update distances and parents for candidate vertices.
 *
 * Each thread processes one candidate vertex, scans all its in-neighbors
 * to find the best parent (minimum distance), updates the distance and
 * parent arrays, and marks the vertex as affected if the distance changed.
 *
 * @param d_candidateList   Array of candidate vertex indices.
 * @param numCandidates     Number of candidate vertices.
 * @param d_inRowPtr        CSR row pointer for reverse graph.
 * @param d_inColInd        CSR column indices for reverse graph.
 * @param d_inWeights       CSR edge weights for reverse graph.
 * @param d_distances       Distance array (read/write).
 * @param d_parent          Parent array (read/write).
 * @param d_isAffected      Flag array for newly affected vertices.
 * @param d_affectedList    Output worklist of newly affected vertices.
 * @param d_affectedCount   Atomic counter for affectedList size.
 * @param INF_VALUE         Sentinel value for unreachable vertices.
 */
__global__ void updateDistancesKernel(
    const int *d_candidateList, int numCandidates, const int *d_inRowPtr,
    const int *d_inColInd, const long long *d_inWeights,
    long long *d_distances, int *d_parent, int *d_isAffected,
    int *d_affectedList, int *d_affectedCount, long long INF_VALUE) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numCandidates)
    return;

  int candidateVertex = d_candidateList[tid];

  // Find best parent among in-neighbors
  int bestParent = -1;
  long long bestDistance = INF_VALUE;

  int rowStart = d_inRowPtr[candidateVertex];
  int rowEnd = d_inRowPtr[candidateVertex + 1];

  for (int e = rowStart; e < rowEnd; ++e) {
    int candidateParent = d_inColInd[e];
    long long candidateWeight = d_inWeights[e];

    // Skip unreachable in-neighbors to avoid overflow
    long long parentDist = d_distances[candidateParent];
    if (parentDist >= INF_VALUE / 2)
      continue;

    long long candidateDistance = parentDist + candidateWeight;
    if (candidateDistance < bestDistance) {
      bestDistance = candidateDistance;
      bestParent = candidateParent;
    }
  }

  // ALWAYS update parent to current best (parent consistency fix).
  // Only propagate further if the DISTANCE actually changed.
  bool distanceChanged = (bestDistance != d_distances[candidateVertex]);

  // No race: each candidateVertex is unique in the list
  d_parent[candidateVertex] = bestParent;
  d_distances[candidateVertex] = bestDistance;

  if (distanceChanged) {
    // Atomic test-and-set for deduplication in affected list
    int old = atomicCAS(&d_isAffected[candidateVertex], 0, 1);
    if (old == 0) {
      int pos = atomicAdd(d_affectedCount, 1);
      d_affectedList[pos] = candidateVertex;
    }
  }
}

/**
 * @brief BFS frontier expansion kernel.
 *
 * Each thread processes one frontier vertex, marks unvisited out-neighbors
 * as reachable, and adds them to the next frontier.
 *
 * @param d_frontier         Current BFS frontier.
 * @param frontierSize       Size of current frontier.
 * @param d_outRowPtr        CSR row pointer for forward graph.
 * @param d_outColInd        CSR column indices for forward graph.
 * @param d_reachable        Visited/reachable flag array (0/1).
 * @param d_nextFrontier     Output next frontier.
 * @param d_nextFrontierCount Atomic counter for next frontier size.
 */
__global__ void bfsKernel(const int *d_frontier, int frontierSize,
                          const int *d_outRowPtr, const int *d_outColInd,
                          int *d_reachable, int *d_nextFrontier,
                          int *d_nextFrontierCount) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontierSize)
    return;

  int current = d_frontier[tid];
  int rowStart = d_outRowPtr[current];
  int rowEnd = d_outRowPtr[current + 1];

  for (int e = rowStart; e < rowEnd; ++e) {
    int neighbor = d_outColInd[e];
    int old = atomicCAS(&d_reachable[neighbor], 0, 1);
    if (old == 0) {
      int pos = atomicAdd(d_nextFrontierCount, 1);
      d_nextFrontier[pos] = neighbor;
    }
  }
}

/**
 * @brief Mark unreachable vertices with INF distance and parent = -1.
 *
 * @param numberOfNodes  Total number of vertices.
 * @param d_reachable    Reachable flag array.
 * @param d_distances    Distance array to update.
 * @param d_parent       Parent array to update.
 * @param INF_VALUE      Sentinel value for unreachable vertices.
 */
__global__ void markUnreachableKernel(int numberOfNodes,
                                      const int *d_reachable,
                                      long long *d_distances, int *d_parent,
                                      long long INF_VALUE) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfNodes)
    return;

  if (!d_reachable[tid]) {
    d_distances[tid] = INF_VALUE;
    d_parent[tid] = -1;
  }
}

// ============================================================================
// HOST HELPER FUNCTIONS
// ============================================================================

namespace {

/// A lightweight edge structure for the internal adjacency lists.
struct WeightedNeighbor {
  int vertex;
  long long weight;
};

/**
 * @brief Parse a line of space-separated integers from a string.
 */
vector<int> parseIntTokens(const string &line) {
  vector<int> tokens;
  istringstream stream(line);
  int value;
  while (stream >> value) {
    tokens.push_back(value);
  }
  return tokens;
}

/**
 * @brief Build forward and reverse adjacency lists from a Graph.
 */
void buildAdjacencyLists(const Graph &graph, int objectiveIndex,
                         vector<vector<WeightedNeighbor>> &outAdjacency,
                         vector<vector<WeightedNeighbor>> &inAdjacency) {
  int numberOfNodes = static_cast<int>(graph.size());
  outAdjacency.assign(numberOfNodes, {});
  inAdjacency.assign(numberOfNodes, {});

  for (int u = 0; u < numberOfNodes; ++u) {
    for (const auto &edge : graph[u]) {
      int v = edge.to;
      long long w = edge.weights[objectiveIndex];
      outAdjacency[u].push_back({v, w});
      inAdjacency[v].push_back({u, w});
    }
  }
}

/**
 * @brief Remove a specific directed edge from an adjacency list entry.
 */
void removeEdgeFromList(vector<WeightedNeighbor> &neighbors, int targetVertex) {
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    if (it->vertex == targetVertex) {
      neighbors.erase(it);
      return;
    }
  }
}

/**
 * @brief Read distances from a Dijkstra output file.
 */
bool readDistancesFromFile(const string &path, vector<long long> &distances,
                           int numberOfNodes, long long INF_VALUE) {
  ifstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open distances file: " << path << "\n";
    return false;
  }

  distances.assign(numberOfNodes, INF_VALUE);

  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    istringstream stream(line);
    int vertexId;
    string distanceStr;
    stream >> vertexId >> distanceStr;

    if (vertexId < 0 || vertexId >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in distances file.\n";
      return false;
    }

    if (distanceStr == "INF") {
      distances[vertexId] = INF_VALUE;
    } else {
      distances[vertexId] = stoll(distanceStr);
    }
  }

  return true;
}

/**
 * @brief Read SSSP tree (parent array) from a Dijkstra output file.
 */
bool readParentFromFile(const string &path, vector<int> &parent,
                        int numberOfNodes) {
  ifstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open SSSP tree file: " << path << "\n";
    return false;
  }

  parent.assign(numberOfNodes, -1);

  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    istringstream stream(line);
    int vertexId, parentId;
    stream >> vertexId >> parentId;

    if (vertexId < 0 || vertexId >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in SSSP tree file.\n";
      return false;
    }

    parent[vertexId] = parentId;
  }

  return true;
}

/**
 * @brief Find the in-neighbor that gives the minimum distance to a vertex.
 *
 * Host-side version used in Phase 1 (sequential edge processing).
 */
void findBestParent(int vertex,
                    const vector<vector<WeightedNeighbor>> &inAdjacency,
                    const vector<long long> &distances, long long INF_VALUE,
                    int &bestParent, long long &bestDistance) {
  bestParent = -1;
  bestDistance = INF_VALUE;

  for (const auto &inNeighbor : inAdjacency[vertex]) {
    int candidateParent = inNeighbor.vertex;
    long long candidateWeight = inNeighbor.weight;

    // Skip unreachable in-neighbors to avoid overflow
    if (distances[candidateParent] >= INF_VALUE / 2) {
      continue;
    }

    long long candidateDistance = distances[candidateParent] + candidateWeight;
    if (candidateDistance < bestDistance) {
      bestDistance = candidateDistance;
      bestParent = candidateParent;
    }
  }
}

/**
 * @brief Flatten adjacency list to CSR format for device transfer.
 *
 * @param adjacency  Adjacency list (vector of vectors).
 * @param rowPtr     Output CSR row pointer.
 * @param colInd     Output CSR column indices.
 * @param weights    Output CSR edge weights.
 */
void flattenToCSR(const vector<vector<WeightedNeighbor>> &adjacency,
                  vector<int> &rowPtr, vector<int> &colInd,
                  vector<long long> &weights) {
  int n = static_cast<int>(adjacency.size());
  rowPtr.resize(n + 1);
  rowPtr[0] = 0;
  for (int i = 0; i < n; ++i) {
    rowPtr[i + 1] = rowPtr[i] + static_cast<int>(adjacency[i].size());
  }

  int nnz = rowPtr[n];
  colInd.resize(nnz);
  weights.resize(nnz);

  for (int i = 0; i < n; ++i) {
    int offset = rowPtr[i];
    for (int j = 0; j < static_cast<int>(adjacency[i].size()); ++j) {
      colInd[offset + j] = adjacency[i][j].vertex;
      weights[offset + j] = adjacency[i][j].weight;
    }
  }
}

} // namespace

// ============================================================================
// MAIN FUNCTION
// ============================================================================

/**
 * @brief Run the parallel (CUDA) SOSP Update algorithm.
 *
 * @see parallelSOSPUpdate.cuh for full parameter documentation.
 */
bool parallelSOSPUpdate(const string &originalCsrPrefix,
                        const string &distancesInputPath,
                        const string &treeInputPath, const string &insertPath,
                        const string &deletePath, int objectiveIndex,
                        int source, const string &distancesOutputPath,
                        const string &treeOutputPath) {
  const long long INF_VALUE = numeric_limits<long long>::max() / 4;

  // ========================================================================
  // PHASE 0: PREPARATION (Host — I/O dominated)
  // ========================================================================

  // --- 0a. Read original graph from CSR and determine dimensions ---
  Graph originalGraph;
  int numberOfObjectives = 0;
  if (!readCSR(originalCsrPrefix, originalGraph, numberOfObjectives)) {
    cout << "Error: Could not read original CSR graph.\n";
    return false;
  }

  int numberOfNodes = static_cast<int>(originalGraph.size());
  if (numberOfNodes == 0) {
    cout << "Error: Graph has no vertices.\n";
    return false;
  }

  if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
    cout << "Error: objectiveIndex out of range.\n";
    return false;
  }

  if (source < 0 || source >= numberOfNodes) {
    cout << "Error: source vertex out of range.\n";
    return false;
  }

  // --- 0b. Build forward and reverse adjacency lists ---
  vector<vector<WeightedNeighbor>> outAdjacency;
  vector<vector<WeightedNeighbor>> inAdjacency;
  buildAdjacencyLists(originalGraph, objectiveIndex, outAdjacency, inAdjacency);

  originalGraph.clear();

  // --- 0c. Read original distances and parent arrays ---
  vector<long long> distances;
  if (!readDistancesFromFile(distancesInputPath, distances, numberOfNodes,
                             INF_VALUE)) {
    return false;
  }

  vector<int> parent;
  if (!readParentFromFile(treeInputPath, parent, numberOfNodes)) {
    return false;
  }

  // --- 0d. Read inserted and deleted edges ---
  struct InsertedEdge {
    int from;
    int to;
    long long weight;
  };

  struct DeletedEdge {
    int from;
    int to;
  };

  vector<InsertedEdge> insertedEdges;
  {
    ifstream insertFile(insertPath);
    if (!insertFile.is_open()) {
      cout << "Error: Could not open insert file: " << insertPath << "\n";
      return false;
    }
    string line;
    while (getline(insertFile, line)) {
      if (line.empty())
        continue;
      vector<int> tokens = parseIntTokens(line);
      if (static_cast<int>(tokens.size()) < 2 + numberOfObjectives) {
        cout << "Error: Invalid insert line.\n";
        return false;
      }
      int u = tokens[0];
      int v = tokens[1];
      long long w = tokens[2 + objectiveIndex];
      insertedEdges.push_back({u, v, w});
    }
  }

  vector<DeletedEdge> deletedEdges;
  {
    ifstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
      cout << "Error: Could not open delete file: " << deletePath << "\n";
      return false;
    }
    string line;
    while (getline(deleteFile, line)) {
      if (line.empty())
        continue;
      vector<int> tokens = parseIntTokens(line);
      if (tokens.size() < 2)
        continue;
      int u = tokens[0];
      int v = tokens[1];
      deletedEdges.push_back({u, v});
    }
  }

  // --- 0e. Apply topological changes to adjacency lists (Host) ---
  struct WeightIncrease {
    int from;
    int to;
  };
  vector<WeightIncrease> weightIncreases;

  // Deletions first
  for (const auto &edge : deletedEdges) {
    removeEdgeFromList(outAdjacency[edge.from], edge.to);
    removeEdgeFromList(inAdjacency[edge.to], edge.from);
  }

  // Then insertions: REPLACE if edge already exists, otherwise add.
  for (const auto &edge : insertedEdges) {
    bool replacedOut = false;
    for (auto &neighbor : outAdjacency[edge.from]) {
      if (neighbor.vertex == edge.to) {
        if (edge.weight > neighbor.weight) {
          weightIncreases.push_back({edge.from, edge.to});
        }
        neighbor.weight = edge.weight;
        replacedOut = true;
        break;
      }
    }
    if (!replacedOut) {
      outAdjacency[edge.from].push_back({edge.to, edge.weight});
    }

    bool replacedIn = false;
    for (auto &neighbor : inAdjacency[edge.to]) {
      if (neighbor.vertex == edge.from) {
        neighbor.weight = edge.weight;
        replacedIn = true;
        break;
      }
    }
    if (!replacedIn) {
      inAdjacency[edge.to].push_back({edge.from, edge.weight});
    }
  }

  // ========================================================================
  // PHASE 1: PROCESS CHANGED EDGES (Host — tiny batch, order-dependent)
  // ========================================================================

  vector<int> isAffected(numberOfNodes, 0);
  vector<int> affectedVertices;

  // --- 1a. Process insertions ---
  for (const auto &edge : insertedEdges) {
    int u = edge.from;
    int v = edge.to;

    if (distances[u] >= INF_VALUE / 2) {
      continue;
    }

    long long actualWeight = -1;
    for (const auto &neighbor : outAdjacency[u]) {
      if (neighbor.vertex == v) {
        actualWeight = neighbor.weight;
        break;
      }
    }
    if (actualWeight < 0) {
      continue;
    }

    long long newDistance = distances[u] + actualWeight;
    if (newDistance < distances[v]) {
      distances[v] = newDistance;
      parent[v] = u;
      if (!isAffected[v]) {
        isAffected[v] = 1;
        affectedVertices.push_back(v);
      }
    }
  }

  // --- 1b. Process deletions ---
  for (const auto &edge : deletedEdges) {
    int u = edge.from;
    int v = edge.to;

    if (parent[v] != u) {
      continue;
    }

    int bestAlternativeParent = -1;
    long long bestAlternativeDistance = INF_VALUE;
    findBestParent(v, inAdjacency, distances, INF_VALUE, bestAlternativeParent,
                   bestAlternativeDistance);

    parent[v] = bestAlternativeParent;
    distances[v] = bestAlternativeDistance;

    if (!isAffected[v]) {
      isAffected[v] = 1;
      affectedVertices.push_back(v);
    }
  }

  // --- 1c. Process weight increases on existing edges ---
  for (const auto &wi : weightIncreases) {
    int u = wi.from;
    int v = wi.to;

    if (parent[v] != u) {
      continue;
    }

    int bestNewParent = -1;
    long long bestNewDistance = INF_VALUE;
    findBestParent(v, inAdjacency, distances, INF_VALUE, bestNewParent,
                   bestNewDistance);

    parent[v] = bestNewParent;
    distances[v] = bestNewDistance;

    if (!isAffected[v]) {
      isAffected[v] = 1;
      affectedVertices.push_back(v);
    }
  }

  // ========================================================================
  // PHASE 2: PROPAGATE THE UPDATE (CUDA Kernels)
  // ========================================================================

  // --- Flatten adjacency lists to CSR for device transfer ---
  vector<int> h_outRowPtr, h_outColInd;
  vector<long long> h_outWeights;
  flattenToCSR(outAdjacency, h_outRowPtr, h_outColInd, h_outWeights);

  vector<int> h_inRowPtr, h_inColInd;
  vector<long long> h_inWeights;
  flattenToCSR(inAdjacency, h_inRowPtr, h_inColInd, h_inWeights);

  // Free host adjacency lists (no longer needed)
  outAdjacency.clear();
  inAdjacency.clear();

  int outNnz = static_cast<int>(h_outColInd.size());
  int inNnz = static_cast<int>(h_inColInd.size());

  // --- Allocate device memory ---
  int *d_outRowPtr = nullptr, *d_outColInd = nullptr;
  int *d_inRowPtr = nullptr, *d_inColInd = nullptr;
  long long *d_inWeights = nullptr;
  long long *d_distances = nullptr;
  int *d_parent = nullptr;
  int *d_isAffected = nullptr, *d_isCandidate = nullptr;
  int *d_affectedList = nullptr, *d_candidateList = nullptr;
  int *d_affectedCount = nullptr, *d_candidateCount = nullptr;

  CUDA_CHECK(cudaMalloc(&d_outRowPtr, (numberOfNodes + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_outColInd, max(outNnz, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inRowPtr, (numberOfNodes + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inColInd, max(inNnz, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inWeights, max(inNnz, 1) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_distances, numberOfNodes * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_parent, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isAffected, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isCandidate, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_affectedList, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_candidateList, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_affectedCount, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_candidateCount, sizeof(int)));

  // --- Copy data to device ---
  CUDA_CHECK(cudaMemcpy(d_outRowPtr, h_outRowPtr.data(),
                        (numberOfNodes + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (outNnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_outColInd, h_outColInd.data(),
                          outNnz * sizeof(int), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_inRowPtr, h_inRowPtr.data(),
                        (numberOfNodes + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (inNnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_inColInd, h_inColInd.data(),
                          inNnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inWeights, h_inWeights.data(),
                          inNnz * sizeof(long long), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_distances, distances.data(),
                        numberOfNodes * sizeof(long long),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), numberOfNodes * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Copy affected flags and list to device
  CUDA_CHECK(cudaMemcpy(d_isAffected, isAffected.data(),
                        numberOfNodes * sizeof(int), cudaMemcpyHostToDevice));

  int h_affectedCount = static_cast<int>(affectedVertices.size());
  if (h_affectedCount > 0) {
    CUDA_CHECK(cudaMemcpy(d_affectedList, affectedVertices.data(),
                          h_affectedCount * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  // --- Iterative propagation loop ---
  const int BLOCK_SIZE = 256;
  int iterationCount = 0;
  const int maxIterations = numberOfNodes;

  while (h_affectedCount > 0 && iterationCount < maxIterations) {
    ++iterationCount;

    // Reset candidate structures
    CUDA_CHECK(
        cudaMemset(d_isCandidate, 0, numberOfNodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_candidateCount, 0, sizeof(int)));

    // --- 2a. Collect candidates ---
    int gridSize = (h_affectedCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    collectCandidatesKernel<<<gridSize, BLOCK_SIZE>>>(
        d_affectedList, h_affectedCount, d_outRowPtr, d_outColInd,
        d_isCandidate, d_candidateList, d_candidateCount, d_isAffected,
        source);
    CUDA_CHECK(cudaGetLastError());

    // Get candidate count
    int h_candidateCount = 0;
    CUDA_CHECK(cudaMemcpy(&h_candidateCount, d_candidateCount, sizeof(int),
                          cudaMemcpyDeviceToHost));

    if (h_candidateCount == 0)
      break;

    // Reset affected structures for next iteration
    CUDA_CHECK(cudaMemset(d_affectedCount, 0, sizeof(int)));

    // --- 2b. Update distances ---
    gridSize = (h_candidateCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateDistancesKernel<<<gridSize, BLOCK_SIZE>>>(
        d_candidateList, h_candidateCount, d_inRowPtr, d_inColInd, d_inWeights,
        d_distances, d_parent, d_isAffected, d_affectedList, d_affectedCount,
        INF_VALUE);
    CUDA_CHECK(cudaGetLastError());

    // Get affected count for next iteration
    CUDA_CHECK(cudaMemcpy(&h_affectedCount, d_affectedCount, sizeof(int),
                          cudaMemcpyDeviceToHost));
  }

  if (iterationCount >= maxIterations && h_affectedCount > 0) {
    cout << "Warning: SOSP update reached maximum iteration limit ("
         << maxIterations << "). Running reachability check.\n";
  }

  // ========================================================================
  // POST-PROCESSING: REACHABILITY CHECK (CUDA BFS)
  // ========================================================================

  {
    int *d_reachable = nullptr;
    int *d_frontier = nullptr, *d_nextFrontier = nullptr;
    int *d_nextFrontierCount = nullptr;

    CUDA_CHECK(cudaMalloc(&d_reachable, numberOfNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, numberOfNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextFrontier, numberOfNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextFrontierCount, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_reachable, 0, numberOfNodes * sizeof(int)));

    // Mark source as reachable and set as initial frontier
    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_reachable + source, &one, sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));

    int h_frontierSize = 1;

    while (h_frontierSize > 0) {
      CUDA_CHECK(cudaMemset(d_nextFrontierCount, 0, sizeof(int)));

      int gridSize = (h_frontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
      bfsKernel<<<gridSize, BLOCK_SIZE>>>(d_frontier, h_frontierSize,
                                          d_outRowPtr, d_outColInd,
                                          d_reachable, d_nextFrontier,
                                          d_nextFrontierCount);
      CUDA_CHECK(cudaGetLastError());

      CUDA_CHECK(cudaMemcpy(&h_frontierSize, d_nextFrontierCount, sizeof(int),
                            cudaMemcpyDeviceToHost));

      // Swap frontier pointers
      int *temp = d_frontier;
      d_frontier = d_nextFrontier;
      d_nextFrontier = temp;
    }

    // Mark unreachable vertices
    int gridSize = (numberOfNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    markUnreachableKernel<<<gridSize, BLOCK_SIZE>>>(
        numberOfNodes, d_reachable, d_distances, d_parent, INF_VALUE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_reachable));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_nextFrontier));
    CUDA_CHECK(cudaFree(d_nextFrontierCount));
  }

  // ========================================================================
  // COPY RESULTS BACK TO HOST
  // ========================================================================

  CUDA_CHECK(cudaMemcpy(distances.data(), d_distances,
                        numberOfNodes * sizeof(long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, numberOfNodes * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // --- Free device memory ---
  CUDA_CHECK(cudaFree(d_outRowPtr));
  CUDA_CHECK(cudaFree(d_outColInd));
  CUDA_CHECK(cudaFree(d_inRowPtr));
  CUDA_CHECK(cudaFree(d_inColInd));
  CUDA_CHECK(cudaFree(d_inWeights));
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_parent));
  CUDA_CHECK(cudaFree(d_isAffected));
  CUDA_CHECK(cudaFree(d_isCandidate));
  CUDA_CHECK(cudaFree(d_affectedList));
  CUDA_CHECK(cudaFree(d_candidateList));
  CUDA_CHECK(cudaFree(d_affectedCount));
  CUDA_CHECK(cudaFree(d_candidateCount));

  // ========================================================================
  // WRITE OUTPUT (Host — I/O)
  // ========================================================================

  filesystem::path distOutPath(distancesOutputPath);
  if (!distOutPath.parent_path().empty()) {
    filesystem::create_directories(distOutPath.parent_path());
  }

  filesystem::path treeOutPath(treeOutputPath);
  if (!treeOutPath.parent_path().empty()) {
    filesystem::create_directories(treeOutPath.parent_path());
  }

  ofstream distancesOut(distancesOutputPath);
  if (!distancesOut.is_open()) {
    cout << "Error: Could not write updated distances file.\n";
    return false;
  }

  for (int i = 0; i < numberOfNodes; ++i) {
    distancesOut << i << " ";
    if (distances[i] >= INF_VALUE / 2) {
      distancesOut << "INF";
    } else {
      distancesOut << distances[i];
    }
    distancesOut << "\n";
  }

  ofstream treeOut(treeOutputPath);
  if (!treeOut.is_open()) {
    cout << "Error: Could not write updated SSSP tree file.\n";
    return false;
  }

  for (int i = 0; i < numberOfNodes; ++i) {
    treeOut << i << " " << parent[i] << "\n";
  }

  return true;
}
