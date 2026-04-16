#ifndef SEQUENTIAL_SOSP_UPDATE_CUH
#define SEQUENTIAL_SOSP_UPDATE_CUH

#include <string>

/**
 * @brief Update single-objective shortest path distances and SSSP tree
 *        without recomputing from scratch.
 *
 * @details
 * Implements the sequential version of the SOSP Update algorithm described
 * in "Parallel Multi Objective Shortest Path Update Algorithm in Large
 * Dynamic Networks" (Shovan, Khanda, Das — IEEE TPDS 2025).
 *
 * Instead of rebuilding the updated graph and running Dijkstra from scratch,
 * this function takes the original graph (CSR), the pre-computed distances
 * and SSSP tree, and a set of edge changes (insertions and deletions), then
 * incrementally updates the distances and SSSP tree.
 *
 * Internally builds a reverse (in-edge) adjacency list from the forward CSR
 * so no separate reverse CSR files are needed.
 *
 * @param originalCsrPrefix  Prefix for original CSR files
 *                           (e.g. "data/originalGraph/graphCsr").
 * @param distancesInputPath Path to original distances file from Dijkstra
 *                           (format: "vertex distance" per line).
 * @param treeInputPath      Path to original SSSP tree file from Dijkstra
 *                           (format: "vertex parent" per line).
 * @param insertPath         Path to insert.txt (format: "u v w1 w2 ... wK").
 * @param deletePath         Path to delete.txt (format: "u v").
 * @param objectiveIndex     Which objective (0-indexed) to use as edge weight.
 * @param source             Source vertex (0-indexed, default 0).
 * @param distancesOutputPath Output path for updated distances.
 * @param treeOutputPath      Output path for updated SSSP tree (parent array).
 * @return True on success; false otherwise.
 */
bool sequentialSOSPUpdate(
    const std::string &originalCsrPrefix,
    const std::string &distancesInputPath,
    const std::string &treeInputPath,
    const std::string &insertPath,
    const std::string &deletePath,
    int objectiveIndex,
    int source = 0,
    const std::string &distancesOutputPath = "output/sospUpdateDistancesTrees/distancesCsr.txt",
    const std::string &treeOutputPath = "output/sospUpdateDistancesTrees/SSSPTreeCsr.txt"
);

#endif
