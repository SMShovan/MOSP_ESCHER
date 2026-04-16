#ifndef GENERATE_GRAPH_CSR_CUH
#define GENERATE_GRAPH_CSR_CUH

#include <string>

/**
 * @brief Generate a connected graph and write it in CSR format.
 *
 * Writes outputPrefixRowPtr.txt, outputPrefixColInd.txt, outputPrefixValues.txt
 *
 * @param numberOfNodes Number of vertices.
 * @param numberOfEdges Number of edges.
 * @param directed Whether the graph is directed.
 * @param outputPrefix Base path for the three output files.
 * @param numberOfObjectives Number of weights per edge.
 * @param objectiveStartRange Minimum weight (inclusive).
 * @param objectiveEndRange Maximum weight (inclusive).
 * @param seed RNG seed. 0 means non-deterministic (default).
 * @return True on success; false otherwise.
 */
bool generateGraphCSR(
    int numberOfNodes,
    int numberOfEdges,
    bool directed,
    const std::string &outputPrefix,
    int numberOfObjectives,
    int objectiveStartRange,
    int objectiveEndRange,
    unsigned int seed = 0
);

#endif
