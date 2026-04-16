#ifndef PARALLEL_COMBINED_GRAPH_CUH
#define PARALLEL_COMBINED_GRAPH_CUH

#include <string>
#include <vector>

/**
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * @details
 * Given K pre-computed SSSP tree parent arrays (one per objective),
 * this function:
 *
 * 1. Extracts all directed tree edges from each parent array.
 * 2. Counts how many trees each edge belongs to (membership count m ∈ {1..K}).
 * 3. Assigns edge weights by preference:
 *       m == K  →  weight 1  (highest preference — in all trees)
 *       m == K-1 → weight 2  (intermediate)
 *       ...
 *       m == 1  →  weight K  (lowest preference — in only one tree)
 *    Generalized formula: weight = K + 1 - m
 * 4. Writes the combined-graph edges as a fresh insertion batch.
 * 5. Calls parallelSOSPUpdate with a blank base graph (all distances = INF,
 *    source distance = 0, empty SSSP tree) so the algorithm runs like
 *    Bellman-Ford seeded from the source on the combined graph.
 *
 * All intermediate files are written under @p workDir.
 *
 * @param originalCsrPrefix   Prefix of the original CSR files (used only to
 *                             determine the number of vertices; no edge data
 *                             from this graph is used in the combined graph).
 * @param treeInputPaths       Paths to the K SSSP tree parent files produced
 *                             by parallelSOSPUpdate (one per objective).
 * @param K                    Number of objectives / trees (must equal
 *                             treeInputPaths.size()).
 * @param source               Source vertex (0-indexed, default 0).
 * @param workDir              Directory for intermediate temp files
 *                             (default "output/combinedGraph").
 * @param distancesOutputPath  Output path for SSSP distances on combined graph.
 * @param treeOutputPath       Output path for SSSP parent array on combined
 * graph.
 * @return True on success; false on any I/O or algorithm error.
 */
bool parallelCombinedGraph(
    const std::string &originalCsrPrefix,
    const std::vector<std::string> &treeInputPaths, int K, int source = 0,
    const std::string &workDir = "output/combinedGraph",
    const std::string &distancesOutputPath =
        "output/combinedGraph/distancesCsr.txt",
    const std::string &treeOutputPath = "output/combinedGraph/SSSPTreeCsr.txt");

#endif // PARALLEL_COMBINED_GRAPH_CUH
