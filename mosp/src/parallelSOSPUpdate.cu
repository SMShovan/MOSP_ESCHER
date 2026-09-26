/**
 * @file parallelSOSPUpdate.cu
 * @brief Parallel (CUDA) Single-Objective Shortest Path (SOSP) Update.
 *
 * Produces the same distances as Dijkstra on the updated graph and, with
 * the lowest-id tie-break, the same SSSP tree.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Phase 0 (host): read the graph, the initial tree and the change batch and
 * apply the batch to forward/reverse adjacency lists.
 *
 * Steps 1 and 2 run on the GPU in sospUpdateGpu() (sospUpdateGpu.cu):
 *   - Step 1, straight from the change list: the head v of every deleted
 *     or weight-increased edge (u,v) with Parent[v] == u is a root; the
 *     SOSP subtrees of the roots are invalidated by pointer jumping; the
 *     invalidated vertices and the heads of inserted edges pull the best
 *     (distance, id) over their in-neighbours (grouped by destination).
 *   - Step 2: a push-based near-far worklist with a packed 64-bit
 *     atomicMin on (distance << b | parent) propagates the improvements.
 *     Distances only decrease, so there is no iteration cap and no
 *     reachability post-pass; only improved vertices are expanded.
 *
 * ============================================================================
 */

#include "parallelSOSPUpdate.cuh"

#include "deviceArray.cuh"
#include "read.cuh"
#include "sospUpdateGpu.cuh"

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
 * @brief Flatten adjacency list to CSR format for device transfer.
 *
 * @param adjacency  Adjacency list (vector of vectors).
 * @param rowPtr     Output CSR row pointer.
 * @param colInd     Output CSR column indices.
 * @param weights    Output CSR edge weights.
 */
void flattenToCSR(const vector<vector<WeightedNeighbor>> &adjacency,
                  vector<int> &rowPtr, vector<int> &colInd,
                  vector<int> &weights) {
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
      weights[offset + j] = static_cast<int>(adjacency[i][j].weight);
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
  long long maxWeight = 1;
  for (const auto &row : outAdjacency) {
    for (const auto &neighbor : row) {
      maxWeight = max(maxWeight, neighbor.weight);
    }
  }

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


  // Heads of inserted edges, and edges that may invalidate a subtree:
  // deletions and weight increases (as (from, to) pairs).
  vector<int> insertHeads, changedFrom, changedTo;
  for (const auto &edge : insertedEdges) {
    insertHeads.push_back(edge.to);
  }
  for (const auto &edge : deletedEdges) {
    changedFrom.push_back(edge.from);
    changedTo.push_back(edge.to);
  }
  for (const auto &wi : weightIncreases) {
    changedFrom.push_back(wi.from);
    changedTo.push_back(wi.to);
  }

  // --- Flatten adjacency lists to CSR for device transfer ---
  vector<int> h_outRowPtr, h_outColInd, h_outWeights;
  flattenToCSR(outAdjacency, h_outRowPtr, h_outColInd, h_outWeights);

  vector<int> h_inRowPtr, h_inColInd, h_inWeights;
  flattenToCSR(inAdjacency, h_inRowPtr, h_inColInd, h_inWeights);

  // Free host adjacency lists (no longer needed)
  outAdjacency.clear();
  inAdjacency.clear();

  // Near-far bucket width and the bound on distances (packed format).
  long long weightSum = 0;
  for (int w : h_outWeights) {
    weightSum += w;
    maxWeight = max(maxWeight, static_cast<long long>(w));
  }
  const long long delta = defaultDelta(
      static_cast<long long>(h_outColInd.size()), numberOfNodes, weightSum);

  // --- Allocate device memory and copy the data ---
  DeviceArray<int> d_outRowPtr, d_outColInd, d_outWeights;
  DeviceArray<int> d_inRowPtr, d_inColInd, d_inWeights;
  DeviceArray<int> d_changedFrom, d_changedTo, d_insertHeads, d_parent;
  DeviceArray<long long> d_distances;
  if (!d_outRowPtr.upload(h_outRowPtr) || !d_outColInd.upload(h_outColInd) ||
      !d_outWeights.upload(h_outWeights) || !d_inRowPtr.upload(h_inRowPtr) ||
      !d_inColInd.upload(h_inColInd) || !d_inWeights.upload(h_inWeights) ||
      !d_changedFrom.upload(changedFrom) || !d_changedTo.upload(changedTo) ||
      !d_insertHeads.upload(insertHeads) || !d_distances.upload(distances) ||
      !d_parent.upload(parent)) {
    cout << "Error: could not copy the graph to the GPU.\n";
    return false;
  }
  SospWorkspace workspace;
  if (!workspace.reserve(numberOfNodes)) {
    cout << "Error: could not allocate the GPU workspace.\n";
    return false;
  }
  const int numberOfEdges = static_cast<int>(h_outColInd.size());
  DeviceCsr outCsr;
  outCsr.numberOfNodes = numberOfNodes;
  outCsr.numberOfEdges = numberOfEdges;
  outCsr.rowPtr = d_outRowPtr.data();
  outCsr.colInd = d_outColInd.data();
  outCsr.weights = d_outWeights.data();
  DeviceCsr inCsr = outCsr;
  inCsr.rowPtr = d_inRowPtr.data();
  inCsr.colInd = d_inColInd.data();
  inCsr.weights = d_inWeights.data();
  DeviceChanges changes;
  changes.changedFrom = d_changedFrom.data();
  changes.changedTo = d_changedTo.data();
  changes.numberOfChanged = static_cast<int>(changedFrom.size());
  changes.insertHeads = d_insertHeads.data();
  changes.numberOfInsertHeads = static_cast<int>(insertHeads.size());

  // ========================================================================
  // STEPS 1 AND 2 ON THE GPU (see sospUpdateGpu.cu)
  // ========================================================================
  SospStats stats;
  {
    if (!sospUpdateGpu(outCsr, inCsr, changes, source, delta, maxWeight,
                       workspace, d_distances.data(), d_parent.data(),
                       &stats)) {
      cout << "Error: SOSP update failed on the GPU.\n";
      return false;
    }
  }

  // ========================================================================
  // COPY RESULTS BACK TO HOST
  // ========================================================================
  if (!d_distances.download(distances) || !d_parent.download(parent)) {
    cout << "Error: could not copy the result from the GPU.\n";
    return false;
  }

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
