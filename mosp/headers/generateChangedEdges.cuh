#ifndef GENERATE_CHANGED_EDGES_CUH
#define GENERATE_CHANGED_EDGES_CUH

#include <string>

/**
 * @brief Generate change-edge files for graph updates.
 *
 * output/changedEdges/insert.txt lines: u v w1 w2 ... wK
 * output/changedEdges/delete.txt lines: u v
 *
 * @param objectiveStartRange Minimum objective weight value (inclusive).
 * @param objectiveEndRange Maximum objective weight value (inclusive).
 * @param numberOfObjectives Number of objectives (weights per inserted edge).
 * @param numberOfNodes Number of nodes in the original graph.
 * @param numberOfChangedEdges Total number of changed edges to generate.
 * @param insertionPercentage Percentage of insert operations.
 * @param deletionPercentage Percentage of delete operations.
 * @param directed Whether edges are directed. Defaults to true.
 * @param exist If true, deletions are sampled from existing CSR edges. Defaults to false.
 * @param duplicate If true, duplicates are allowed. Defaults to true.
 * @param selfLoop If true, self-loops are allowed. Defaults to false.
 * @param csrPrefix CSR prefix used when exist=true. Defaults to "data/originalGraph/graphCsr".
 * @param insertOutputPath Path to write insert.txt. Defaults to "output/changedEdges/insert.txt".
 * @param deleteOutputPath Path to write delete.txt. Defaults to "output/changedEdges/delete.txt".
 * @param seed RNG seed. 0 means non-deterministic (default).
 * @return True on success; false otherwise.
 */
bool generateChangedEdges(
    int objectiveStartRange,
    int objectiveEndRange,
    int numberOfObjectives,
    int numberOfNodes,
    int numberOfChangedEdges,
    double insertionPercentage,
    double deletionPercentage,
    bool directed = true,
    bool exist = false,
    bool duplicate = true,
    bool selfLoop = false,
    const std::string &csrPrefix = "data/originalGraph/graphCsr",
    const std::string &insertOutputPath = "output/changedEdges/insert.txt",
    const std::string &deleteOutputPath = "output/changedEdges/delete.txt",
    unsigned int seed = 0
);

#endif
