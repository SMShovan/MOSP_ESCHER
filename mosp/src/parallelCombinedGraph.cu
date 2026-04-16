/**
 * @file parallelCombinedGraph.cu
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Given K SSSP trees (one per objective), produced by K independent runs of
 * parallelSOSPUpdate, this module:
 *
 *   Phase 0 — Read & extract tree edges
 *     For each of the K parent arrays, derive the set of directed tree edges.
 *     An edge (u → v) is in tree k if parent_k[v] == u.
 *
 *   Phase 1 — Count membership & assign weights
 *     For every unique edge across all K trees, count m = number of trees
 *     it belongs to.  Assign weight  w = K + 1 - m:
 *       m == K  →  w = 1  (appears in ALL trees — highest preference)
 *       m == K-1 →  w = 2
 *       ...
 *       m == 1  →  w = K  (appears in ONE tree — lowest preference)
 *
 *   Phase 2 — Write temporary input files for parallelSOSPUpdate
 *     Write an EMPTY base CSR (same vertex count, no edges).
 *     Write blank initial distances (dist[source]=0, rest=INF) and
 *       a blank SSSP tree (all parents = -1).
 *     Write all combined-graph edges as an insertion file.
 *     Write an empty deletion file.
 *
 *   Phase 3 — Run parallelSOSPUpdate on the combined graph
 *     With no base edges and all combined edges treated as insertions,
 *     Phase 1 of parallelSOSPUpdate seeds affectedVertices from the source,
 *     and Phase 2 propagates exactly like Bellman-Ford — correct & parallel.
 *
 * ============================================================================
 * PARALLELISM (CUDA version)
 * ============================================================================
 *
 * Edge membership counting (Phase 1) is performed on the host using
 * std::map (K is small, typically 3). For large-scale graphs this can be
 * upgraded to a thrust-based sort+reduce_by_key approach on the GPU.
 * File I/O (Phase 2) is sequential (I/O dominated, tiny data).
 * Phase 3 is fully parallel (delegated to parallelSOSPUpdate CUDA kernels).
 *
 * ============================================================================
 */

#include "parallelCombinedGraph.cuh"

#include "parallelSOSPUpdate.cuh"
#include "read.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

/**
 * @brief Read a parent array from a Dijkstra / parallelSOSPUpdate tree file.
 *
 * Expected format per line: "vertexId parentId"  (parentId == -1 for source).
 *
 * @param path          Path to the SSP tree file.
 * @param parent        Output parent array.
 * @param numberOfNodes Expected vertex count.
 * @return True on success; false otherwise.
 */
bool readParent(const string &path, vector<int> &parent, int numberOfNodes) {
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
    istringstream ss(line);
    int v, p;
    if (!(ss >> v >> p))
      continue;
    if (v < 0 || v >= numberOfNodes) {
      cout << "Error: Vertex ID " << v << " out of range in: " << path << "\n";
      return false;
    }
    parent[v] = p;
  }
  return true;
}

} // namespace

/**
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * @see parallelCombinedGraph.cuh for full parameter documentation.
 */
bool parallelCombinedGraph(const string &originalCsrPrefix,
                           const vector<string> &treeInputPaths, int K,
                           int source, const string &workDir,
                           const string &distancesOutputPath,
                           const string &treeOutputPath) {

  if (K <= 0 || static_cast<int>(treeInputPaths.size()) < K) {
    cout << "Error: K=" << K << " but only " << treeInputPaths.size()
         << " tree paths supplied.\n";
    return false;
  }

  // ========================================================================
  // PHASE 0: READ BASE GRAPH (for vertex count) AND ALL K PARENT ARRAYS
  // ========================================================================

  // --- 0a. Determine numberOfNodes from the CSR ---
  Graph baseGraph;
  int numberOfObjectives = 0;
  if (!readCSR(originalCsrPrefix, baseGraph, numberOfObjectives)) {
    cout << "Error: Could not read original CSR graph.\n";
    return false;
  }
  int numberOfNodes = static_cast<int>(baseGraph.size());
  baseGraph.clear(); // free immediately — we only needed the vertex count

  if (numberOfNodes == 0) {
    cout << "Error: Graph has no vertices.\n";
    return false;
  }
  if (source < 0 || source >= numberOfNodes) {
    cout << "Error: source vertex " << source << " out of range.\n";
    return false;
  }

  // --- 0b. Read K parent arrays ---
  vector<vector<int>> parents(K);
  for (int k = 0; k < K; ++k) {
    if (!readParent(treeInputPaths[k], parents[k], numberOfNodes)) {
      return false;
    }
  }

  // ========================================================================
  // PHASE 1: COUNT EDGE MEMBERSHIP & ASSIGN WEIGHTS (Host — K is small)
  // ========================================================================
  //
  // For each tree k and each vertex v (v != source, parent_k[v] != -1),
  // the tree edge is (parent_k[v] → v).
  // We count how many trees each directed edge belongs to, then assign weight.

  map<pair<int, int>, int>
      edgeMembership; // edge → count of trees it appears in

  for (int k = 0; k < K; ++k) {
    const vector<int> &par = parents[k];
    for (int v = 0; v < numberOfNodes; ++v) {
      if (v == source)
        continue;
      int u = par[v];
      if (u < 0 || u >= numberOfNodes)
        continue; // unreachable vertex or invalid parent
      edgeMembership[{u, v}] += 1;
    }
  }

  // ========================================================================
  // PHASE 2: WRITE TEMPORARY FILES FOR parallelSOSPUpdate
  // ========================================================================

  // Create the work directory
  filesystem::create_directories(workDir);

  // --- 2a. Write EMPTY base CSR (same vertex count, zero edges) ---
  //
  // CSR format expected by readCSR:
  //   prefixRowPtr.txt   — n+1 zeros (row offsets, all 0 → no edges)
  //   prefixColInd.txt   — empty (no column indices)
  //   prefixValues.txt   — empty (no values)
  //
  // parallelSOSPUpdate passes objectiveIndex=0 and reads weights[0]; the
  // combined graph has exactly 1 objective weight per edge, so
  // numberOfObjectives=1.

  const string tempCsrPrefix = workDir + "/tempCsr";

  // Write a minimal temp CSR with a single dummy self-loop on the source
  // vertex (source → source, weight 0, 1 objective).  This edge is inert:
  // parallelSOSPUpdate never updates the source vertex (it is guarded by
  // "if (neighborVertex == source) continue"), so the self-loop can never
  // propagate or affect any distance. Its only purpose is to make readCSR
  // infer numberOfObjectives=1 (otherwise an empty Values file gives 0,
  // which causes parallelSOSPUpdate to reject objectiveIndex=0).
  {
    // RowPtr: source row has 1 outgoing edge; all others have 0.
    // row_ptr[i+1] - row_ptr[i] = number of edges from vertex i.
    // source row: ptr goes 0 → 1; all subsequent rows stay at 1.
    ofstream rowFile(tempCsrPrefix + "RowPtr.txt");
    if (!rowFile.is_open()) {
      cout << "Error: Could not write temp CSR RowPtr file.\n";
      return false;
    }
    for (int i = 0; i <= numberOfNodes; ++i) {
      // ptr[0..source] = 0; ptr[source+1..n] = 1
      rowFile << (i <= source ? 0 : 1) << "\n";
    }
  }
  {
    // ColInd: the single edge goes to 'source' itself.
    ofstream colFile(tempCsrPrefix + "ColInd.txt");
    if (!colFile.is_open()) {
      cout << "Error: Could not write temp CSR ColInd file.\n";
      return false;
    }
    colFile << source << "\n";
  }
  {
    // Values: weight 0 for the dummy self-loop (1 objective).
    ofstream valFile(tempCsrPrefix + "Values.txt");
    if (!valFile.is_open()) {
      cout << "Error: Could not write temp CSR Values file.\n";
      return false;
    }
    valFile << "0\n"; // one edge, one objective, weight 0
  }

  // --- 2b. Write blank initial distances (source=0, rest=INF) ---
  const string blankDistPath = workDir + "/blankDistances.txt";
  {
    ofstream distFile(blankDistPath);
    if (!distFile.is_open()) {
      cout << "Error: Could not write blank distances file.\n";
      return false;
    }
    for (int v = 0; v < numberOfNodes; ++v) {
      distFile << v << " ";
      if (v == source) {
        distFile << "0";
      } else {
        distFile << "INF";
      }
      distFile << "\n";
    }
  }

  // --- 2c. Write blank initial SSSP tree (all parents = -1) ---
  const string blankTreePath = workDir + "/blankTree.txt";
  {
    ofstream treeFile(blankTreePath);
    if (!treeFile.is_open()) {
      cout << "Error: Could not write blank tree file.\n";
      return false;
    }
    for (int v = 0; v < numberOfNodes; ++v) {
      treeFile << v << " -1\n";
    }
  }

  // --- 2d. Write combined-graph edges as insertion file ---
  //
  // Format expected by parallelSOSPUpdate: "u v w1" (1 objective weight).
  // Weight = K + 1 - membershipCount  (so all-3 → 1, two → 2, one → 3).

  const string insertPath = workDir + "/insert.txt";
  {
    ofstream insertFile(insertPath);
    if (!insertFile.is_open()) {
      cout << "Error: Could not write combined graph insert file.\n";
      return false;
    }
    for (const auto &entry : edgeMembership) {
      int u = entry.first.first;
      int v = entry.first.second;
      int m = entry.second;
      // Clamp m to [1, K] defensively (shouldn't be needed, just in case)
      if (m < 1)
        m = 1;
      if (m > K)
        m = K;
      int w = K + 1 - m; // weight: 1 (best) to K (worst)
      insertFile << u << " " << v << " " << w << "\n";
    }
  }

  // --- 2e. Write empty deletion file ---
  const string deletePath = workDir + "/delete.txt";
  {
    ofstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
      cout << "Error: Could not write combined graph delete file.\n";
      return false;
    }
    // intentionally empty — no deletions
  }

  // ========================================================================
  // PHASE 3: RUN parallelSOSPUpdate ON THE COMBINED GRAPH
  // ========================================================================
  //
  // objectiveIndex = 0  (the combined graph has exactly 1 weight per edge)
  // The empty base CSR + "all edges as insertions" pattern causes Phase 1 of
  // parallelSOSPUpdate to relax every combined edge from the source outward,
  // and Phase 2 propagates like Chaotic Bellman-Ford until convergence.

  cout << "parallelCombinedGraph: running parallelSOSPUpdate on combined graph"
       << " (" << edgeMembership.size() << " edges, " << numberOfNodes
       << " vertices)...\n";

  if (!parallelSOSPUpdate(tempCsrPrefix, blankDistPath, blankTreePath,
                          insertPath, deletePath,
                          /*objectiveIndex=*/0, source, distancesOutputPath,
                          treeOutputPath)) {
    cout << "Error: parallelSOSPUpdate failed on combined graph.\n";
    return false;
  }

  cout << "parallelCombinedGraph: done. Output written to:\n"
       << "  Distances: " << distancesOutputPath << "\n"
       << "  SSSP Tree: " << treeOutputPath << "\n";

  return true;
}
