#ifndef PARALLEL_SOSP_UPDATE_CUH
#define PARALLEL_SOSP_UPDATE_CUH

#include <string>

/**
 * @brief Update single-objective shortest path distances and SSSP tree
 *        using CUDA parallelism, without recomputing from scratch.
 *
 * @details
 * Implements the parallel (CUDA) version of the SOSP Update algorithm.
 * Produces identical results to sequentialSOSPUpdate() (and thus matches
 * Dijkstra recalculation on the updated graph).
 *
 * Phase 0 (reading the inputs and applying the batch) runs on the host.
 * Step 1 (roots from the change list, subtree invalidation by pointer
 * jumping, first pull pass) and Step 2 (monotone propagation) run on the
 * GPU; see parallelSOSPUpdate.cu. Vertices cut off from the source end
 * with distance INF and parent -1.
 *
 * @param originalCsrPrefix  Prefix for original CSR files.
 * @param distancesInputPath Path to original distances file from Dijkstra.
 * @param treeInputPath      Path to original SSSP tree file from Dijkstra.
 * @param insertPath         Path to insert.txt.
 * @param deletePath         Path to delete.txt.
 * @param objectiveIndex     Which objective (0-indexed) to use as edge weight.
 * @param source             Source vertex (0-indexed, default 0).
 * @param distancesOutputPath Output path for updated distances.
 * @param treeOutputPath      Output path for updated SSSP tree.
 * @return True on success; false otherwise.
 */
bool parallelSOSPUpdate(
    const std::string &originalCsrPrefix, const std::string &distancesInputPath,
    const std::string &treeInputPath, const std::string &insertPath,
    const std::string &deletePath, int objectiveIndex, int source = 0,
    const std::string &distancesOutputPath =
        "output/parallelSospUpdateDistancesTrees/distancesCsr.txt",
    const std::string &treeOutputPath =
        "output/parallelSospUpdateDistancesTrees/SSSPTreeCsr.txt");

#endif
