#ifndef ESCHER_MOSP_HYPERGRAPH_ORACLE_HPP
#define ESCHER_MOSP_HYPERGRAPH_ORACLE_HPP

/**
 * @file HypergraphOracle.hpp
 * @brief Test oracle for the H-SOSP pipeline, independent of every
 *        incrementally maintained structure.
 *
 * The line graph is rebuilt from the hyperedges' vertex lists
 * (HostHypergraph::heVerts and alive flags) alone, so it does not share
 * code or state with the incrementally maintained structures (the device
 * CSR, the ESCHER CBSTs, the incidence lists). Distances come from a
 * textbook binary-heap Dijkstra on that rebuilt graph.
 */

#include <vector>

#include "HostHypergraph.hpp"

namespace escher_mosp {

/** Rebuilds the line graph from heVerts: every pair of alive hyperedges
 *  sharing a vertex is emitted per vertex, then sorted and deduplicated
 *  (a different algorithm from HostHypergraph::lineGraph, which unions
 *  incidence lists per hyperedge). */
LineGraphCSR rebuildLineGraph(const HostHypergraph& hg);

/** Net line-graph change between two line graphs (pairs a < b adjacent
 *  only in @p post: insEdges; only in @p pre: delEdges), by definition;
 *  the reference for the GPU-derived delta. newHe / deadHe are left
 *  empty. */
H2HDelta lineGraphDelta(const LineGraphCSR& pre, const LineGraphCSR& post);

/** Node-weighted Dijkstra from @p sourceId on @p lg (stepping into h costs
 *  hg.heW[h-1]); indexed by id - 1, HostHypergraph::INF if unreachable. */
std::vector<long long> referenceDistances(const HostHypergraph& hg,
                                          const LineGraphCSR& lg,
                                          int sourceId);

/** Outcome of comparing an SOSP result with the oracle. */
struct SospCheck {
    long long distMismatches = 0;
    long long parentErrors = 0;
    int firstBadId = 0;          ///< first id with a wrong distance (or 0)
    long long firstGot = 0, firstWant = 0;
    bool ok() const { return distMismatches == 0 && parentErrors == 0; }
};

/**
 * Compares @p dist (indexed by id - 1) with @p reference and checks the
 * shortest-path tree: the parent (0-based node index, the device
 * convention; -1 = none) of every reachable alive node other than the
 * source must be its canonical parent, the lowest-id neighbour p in @p lg
 * with dist[v] = dist[p] + w[v]; unreachable nodes have no parent. Pass an
 * empty @p parent0 to skip the tree check.
 */
SospCheck checkSosp(const HostHypergraph& hg, const LineGraphCSR& lg,
                    const std::vector<long long>& reference,
                    const std::vector<long long>& dist,
                    const std::vector<int>& parent0);

} // namespace escher_mosp

#endif // ESCHER_MOSP_HYPERGRAPH_ORACLE_HPP
