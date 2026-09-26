/**
 * @file main.cu
 * @brief Entry point for the MOSPCUDA project.
 *
 * Demonstrates the complete MOSP pipeline using CUDA:
 *   1. Generate a random graph (Matrix Market + CSR formats).
 *   2. Generate edge changes (insertions + deletions).
 *   3. Apply changes to produce an updated graph.
 *   4. Run Dijkstra on original and updated graphs (ground truth).
 *   5. Run sequential SOSP Update (host baseline).
 *   6. Run parallel SOSP Update (CUDA) per objective.
 *   7. Construct the combined MOSP graph and find its SSSP.
 *   8. Generate and verify deterministic test cases.
 */

#include "dijkstra.cuh"
#include "generateChangedEdges.cuh"
#include "generateGraph.cuh"
#include "generateGraphCSR.cuh"
#include "generateTestCases.cuh"
#include "parallelCombinedGraph.cuh"
#include "parallelSOSPUpdate.cuh"
#include "sequentialSOSPUpdate.cuh"
#include "updateGraphCSR.cuh"
#include "updateGraphWithESCHER.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Every step's result is checked: main exits with 1 when a step fails or a
// generated test case does not match Dijkstra (it ignored all of them).
int main() {

  int numberOfNodes = 100;
  int numberOfEdges = 300;
  int numberOfObjectives = 3;
  int objectiveStartRange = 1;
  int objectiveEndRange = 100;
  int source = 0;
  int numberOfChangedEdges = 20;
  double insertionPercentage = 50;
  double deletionPercentage = 50;

  // 1a. Generate graph in MTX format (for Dijkstra on MTX input)
  if (!generateGraph(numberOfNodes, numberOfEdges, true, "data/graph.mtx",
                    numberOfObjectives, objectiveStartRange, objectiveEndRange)) {
    return 1;
  }

  // 1b. Generate graph in CSR format
  if (!generateGraphCSR(numberOfNodes, numberOfEdges, true,
                       "data/originalGraph/graphCsr", numberOfObjectives,
                       objectiveStartRange, objectiveEndRange)) {
    return 1;
  }

  // 2. Generate edge changes
  if (!generateChangedEdges(objectiveStartRange, objectiveEndRange,
                           numberOfObjectives, numberOfNodes,
                           numberOfChangedEdges, insertionPercentage,
                           deletionPercentage, true, true, true, false,
                           "data/originalGraph/graphCsr")) {
    return 1;
  }

  // 3. Update graph with changes — routed through the ESCHER CBST adapter.
  //    Every edge insert/delete below flows through CBSTOperations::insert,
  //    ::erase, ::fill and unfillCBST; the resulting CSR files are
  //    byte-identical to the legacy updateGraphCSR output for directed mode
  //    (verified by tests/unit/test_snapshot_matches_updateCSR).
  if (!escher_mosp::updateGraphWithESCHER(
          "data/originalGraph/graphCsr", "data/updatedGraph/updatedGraphCsr",
          "output/changedEdges/insert.txt", "output/changedEdges/delete.txt",
          /*payloadCapacity=*/65536, /*directed=*/true)) {
    return 1;
  }

  // 4a. Dijkstra on original MTX graph (objective 0)
  if (!runDijkstra("data/graph.mtx", 0, source,
                  "output/distancesTrees/distances.txt",
                  "output/distancesTrees/SSSPTree.txt")) {
    return 1;
  }

  // 4b. Dijkstra on original CSR graph (objective 0)
  if (!runDijkstraCSR("data/originalGraph/graphCsr", 0, source,
                     "output/distancesTrees/distancesCsr.txt",
                     "output/distancesTrees/SSSPTreeCsr.txt")) {
    return 1;
  }

  // 4c. Dijkstra on updated CSR graph (objective 0)
  if (!runDijkstraCSR("data/updatedGraph/updatedGraphCsr", 0, source,
                     "output/updatedDistancesTrees/updatedDistancesCsr.txt",
                     "output/updatedDistancesTrees/updatedSSSPTreeCsr.txt")) {
    return 1;
  }

  // 5. Sequential SOSP Update (host baseline, objective 0)
  if (!sequentialSOSPUpdate(
          "data/originalGraph/graphCsr",
          "output/distancesTrees/distancesCsr.txt",
          "output/distancesTrees/SSSPTreeCsr.txt",
          "output/changedEdges/insert.txt", "output/changedEdges/delete.txt",
          0, source, "output/sospUpdateDistancesTrees/distancesCsr.txt",
          "output/sospUpdateDistancesTrees/SSSPTreeCsr.txt")) {
    return 1;
  }

  // 6. Parallel SOSP Update (CUDA) — one per objective
  vector<string> treeOutputPaths;
  for (int obj = 0; obj < numberOfObjectives; ++obj) {
    string objDir = "output/parallelSospObj" + to_string(obj);

    // Run Dijkstra for this objective on the original graph
    string dijkstraDistPath = objDir + "/distancesOriginal.txt";
    string dijkstraTreePath = objDir + "/SSSPTreeOriginal.txt";
    if (!runDijkstraCSR("data/originalGraph/graphCsr", obj, source, dijkstraDistPath,
                       dijkstraTreePath)) {
      return 1;
    }

    // Run CUDA parallel SOSP Update for this objective
    string parallelDistPath = objDir + "/distancesParallelUpdate.txt";
    string parallelTreePath = objDir + "/SSSPTreeParallelUpdate.txt";
    if (!parallelSOSPUpdate("data/originalGraph/graphCsr", dijkstraDistPath,
                           dijkstraTreePath, "output/changedEdges/insert.txt",
                           "output/changedEdges/delete.txt", obj, source,
                           parallelDistPath, parallelTreePath)) {
      return 1;
    }

    treeOutputPaths.push_back(parallelTreePath);
  }

  // 7. Combined graph MOSP
  if (!parallelCombinedGraph("data/originalGraph/graphCsr", treeOutputPaths,
                            numberOfObjectives, source, "output/combinedGraph",
                            "output/combinedGraph/distancesCsr.txt",
                            "output/combinedGraph/SSSPTreeCsr.txt")) {
    return 1;
  }

  // 8. Generate and verify 10 test cases
  if (!generateTestCases("tests")) {
    return 1;
  }

  cout << "\n=== MOSPCUDA pipeline complete ===\n";

  return 0;
}
