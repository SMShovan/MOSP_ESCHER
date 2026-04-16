#ifndef DIJKSTRA_CUH
#define DIJKSTRA_CUH

#include <string>

/**
 * @brief Run single-objective Dijkstra on an MTX file.
 *
 * @param inputFile Path to input Matrix Market file.
 * @param objectiveNumber Objective index (0-based).
 * @param source Source vertex (0-based).
 * @param distanceOutputPath Output file for distances.
 * @param treeOutputPath Output file for parent tree.
 * @return True on success; false otherwise.
 */
bool runDijkstra(
    const std::string &inputFile,
    int objectiveNumber,
    int source,
    const std::string &distanceOutputPath = "output/distances.txt",
    const std::string &treeOutputPath = "output/SSSPTree.txt"
);

/**
 * @brief Run single-objective Dijkstra on CSR files.
 *
 * @param inputPrefix CSR input prefix.
 * @param objectiveNumber Objective index (0-based).
 * @param source Source vertex (0-based).
 * @param distanceOutputPath Output file for distances.
 * @param treeOutputPath Output file for parent tree.
 * @return True on success; false otherwise.
 */
bool runDijkstraCSR(
    const std::string &inputPrefix,
    int objectiveNumber,
    int source,
    const std::string &distanceOutputPath = "output/distancesCsr.txt",
    const std::string &treeOutputPath = "output/SSSPTreeCsr.txt"
);

#endif
