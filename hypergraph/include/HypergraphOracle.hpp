#ifndef ESCHER_MOSP_HYPERGRAPH_ORACLE_HPP
#define ESCHER_MOSP_HYPERGRAPH_ORACLE_HPP

/**
 * @file HypergraphOracle.hpp
 * @brief Test oracle for the H-SOSP pipeline, independent of every
 *        incrementally maintained structure.
 *
 * The line graph is rebuilt from the incidence lists (HostHypergraph::heVerts
 * and alive flags) alone, so it does not share code or state with the
 * maintained h2h lists, the ESCHER CBSTs or the device CSR. Distances come
 * from a textbook binary-heap Dijkstra on that rebuilt graph.
 */

#include <vector>

#include "HostHypergraph.hpp"

namespace escher_mosp {

/** Line graph in CSR form; row (id - 1) lists the ids of the alive
 *  hyperedges sharing a vertex with alive hyperedge id, sorted ascending. */
struct LineGraphCSR {
    int numIds = 0;                  ///< == hg.maxId() when built
    std::vector<long long> offset;   ///< numIds + 1 entries
    std::vector<int> nbr;            ///< 1-based hyperedge ids

    long long degree(int id) const { return offset[id] - offset[id - 1]; }
    const int* row(int id) const { return nbr.data() + offset[id - 1]; }
    bool adjacent(int a, int b) const;
};

/** Rebuilds the line graph from heVerts (OpenMP-parallel). */
LineGraphCSR rebuildLineGraph(const HostHypergraph& hg);

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
 * shortest-path tree: every reachable alive node other than the source has
 * a parent (0-based node index, the device convention; -1 = none) that is
 * alive, adjacent in @p lg and satisfies dist[v] = dist[parent] + w[v];
 * unreachable nodes have no parent. Pass an empty @p parent0 to skip the
 * tree check.
 */
SospCheck checkSosp(const HostHypergraph& hg, const LineGraphCSR& lg,
                    const std::vector<long long>& reference,
                    const std::vector<long long>& dist,
                    const std::vector<int>& parent0);

} // namespace escher_mosp

#endif // ESCHER_MOSP_HYPERGRAPH_ORACLE_HPP
