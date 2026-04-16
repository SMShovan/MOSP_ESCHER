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
  generateGraph(numberOfNodes, numberOfEdges, true, "data/graph.mtx",
                numberOfObjectives, objectiveStartRange, objectiveEndRange);

  // 1b. Generate graph in CSR format
  generateGraphCSR(numberOfNodes, numberOfEdges, true,
                   "data/originalGraph/graphCsr", numberOfObjectives,
                   objectiveStartRange, objectiveEndRange);

  // 2. Generate edge changes
  generateChangedEdges(objectiveStartRange, objectiveEndRange,
                       numberOfObjectives, numberOfNodes,
                       numberOfChangedEdges, insertionPercentage,
                       deletionPercentage, true, true, true, false,
                       "data/originalGraph/graphCsr");

  // 3. Update graph with changes — routed through the ESCHER CBST adapter.
  //    Every edge insert/delete below flows through CBSTOperations::insert,
  //    ::erase, ::fill and unfillCBST; the resulting CSR files are
  //    byte-identical to the legacy updateGraphCSR output for directed mode
  //    (verified by tests/unit/test_snapshot_matches_updateCSR).
  escher_mosp::updateGraphWithESCHER(
      "data/originalGraph/graphCsr", "data/updatedGraph/updatedGraphCsr",
      "output/changedEdges/insert.txt", "output/changedEdges/delete.txt",
      /*payloadCapacity=*/65536, /*directed=*/true);

  // 4a. Dijkstra on original MTX graph (objective 0)
  runDijkstra("data/graph.mtx", 0, source,
              "output/distancesTrees/distances.txt",
              "output/distancesTrees/SSSPTree.txt");

  // 4b. Dijkstra on original CSR graph (objective 0)
  runDijkstraCSR("data/originalGraph/graphCsr", 0, source,
                 "output/distancesTrees/distancesCsr.txt",
                 "output/distancesTrees/SSSPTreeCsr.txt");

  // 4c. Dijkstra on updated CSR graph (objective 0)
  runDijkstraCSR("data/updatedGraph/updatedGraphCsr", 0, source,
                 "output/updatedDistancesTrees/updatedDistancesCsr.txt",
                 "output/updatedDistancesTrees/updatedSSSPTreeCsr.txt");

  // 5. Sequential SOSP Update (host baseline, objective 0)
  sequentialSOSPUpdate(
      "data/originalGraph/graphCsr",
      "output/distancesTrees/distancesCsr.txt",
      "output/distancesTrees/SSSPTreeCsr.txt",
      "output/changedEdges/insert.txt", "output/changedEdges/delete.txt",
      0, source, "output/sospUpdateDistancesTrees/distancesCsr.txt",
      "output/sospUpdateDistancesTrees/SSSPTreeCsr.txt");

  // 6. Parallel SOSP Update (CUDA) — one per objective
  vector<string> treeOutputPaths;
  for (int obj = 0; obj < numberOfObjectives; ++obj) {
    string objDir = "output/parallelSospObj" + to_string(obj);

    // Run Dijkstra for this objective on the original graph
    string dijkstraDistPath = objDir + "/distancesOriginal.txt";
    string dijkstraTreePath = objDir + "/SSSPTreeOriginal.txt";
    runDijkstraCSR("data/originalGraph/graphCsr", obj, source, dijkstraDistPath,
                   dijkstraTreePath);

    // Run CUDA parallel SOSP Update for this objective
    string parallelDistPath = objDir + "/distancesParallelUpdate.txt";
    string parallelTreePath = objDir + "/SSSPTreeParallelUpdate.txt";
    parallelSOSPUpdate("data/originalGraph/graphCsr", dijkstraDistPath,
                       dijkstraTreePath, "output/changedEdges/insert.txt",
                       "output/changedEdges/delete.txt", obj, source,
                       parallelDistPath, parallelTreePath);

    treeOutputPaths.push_back(parallelTreePath);
  }

  // 7. Combined graph MOSP
  parallelCombinedGraph("data/originalGraph/graphCsr", treeOutputPaths,
                        numberOfObjectives, source, "output/combinedGraph",
                        "output/combinedGraph/distancesCsr.txt",
                        "output/combinedGraph/SSSPTreeCsr.txt");

  // 8. Generate and verify 10 test cases
  generateTestCases("tests");

  cout << "\n=== MOSPCUDA pipeline complete ===\n";

  return 0;
}
