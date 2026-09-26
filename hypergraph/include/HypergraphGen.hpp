#ifndef ESCHER_MOSP_HYPERGRAPH_GEN_HPP
#define ESCHER_MOSP_HYPERGRAPH_GEN_HPP

/**
 * @file HypergraphGen.hpp
 * @brief Seeded synthetic hypergraph + change-batch generators (pure C++).
 *
 * Clustered-pool model: vertices are partitioned into pools of size
 * @c poolSize; each hyperedge samples its vertices inside one home pool and,
 * with probability @c bridgeFrac, swaps one vertex for a member of the next
 * pool. Pool size controls the expected h2h degree
 * ( ~ (m * poolSize / n) * (1 - exp(-c^2 / poolSize)) ), which uniform
 * sampling cannot keep bounded at scale.
 *
 * Row 1 is the virtual source hyperedge {s} and the last row the virtual
 * target {t}, both with weight 0, per the meeting notes (Step 1).
 */

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "HostHypergraph.hpp"

namespace escher_mosp {

struct GenParams {
    long long numHyperedges = 100000;  ///< real hyperedges (excl. virtuals)
    int numVertices = 33334;
    int cMin = 2;
    int cMax = 8;
    int wMin = 1;
    int wMax = 100;
    int poolSize = 1024;
    double bridgeFrac = 0.05;
    std::uint64_t seed = 1;
};

/** Raw generated hypergraph: rows[i] is the vertex list of hyperedge i+1. */
struct GeneratedHypergraph {
    int numVertices = 0;
    std::vector<std::vector<int>> rows;
    std::vector<long long> weights;
    int sourceHe = 0;   ///< always 1
    int targetHe = 0;   ///< always rows.size()
    int sourceVertex = 0;
    int targetVertex = 0;
};

GeneratedHypergraph generateHypergraph(const GenParams& p);

/** Write the generated hypergraph as text: first line "n m", then one line
 *  per hyperedge: "w c v1 ... vc". For inspection / external tooling. */
bool writeHypergraphText(const GeneratedHypergraph& g, const std::string& path);

enum class BatchKind { Hyperedge, Vertex };
enum class Placement { Random, Targeted, Near, Far };

struct BatchParams {
    int size = 50000;
    double delPct = 50.0;
    BatchKind kind = BatchKind::Hyperedge;
    Placement placement = Placement::Random;
    std::uint64_t seed = 1;
};

/**
 * Generate a change batch against the CURRENT hypergraph state. Without a
 * live real hyperedge (only the virtual ones) a vertex batch is empty and
 * a hyperedge batch has insertions only.
 *
 * @param hg      current host hypergraph (for alive ids / pools membership).
 * @param gen     generator parameters (pool model for inserted hyperedges).
 * @param bp      batch parameters.
 * @param dist    current SOSP distances indexed by (heId-1); required for
 *                Targeted / Near / Far placements (pass empty for Random).
 * @param parent  current SOSP parents as 1-based hyperedge ids (-1 or 0
 *                for none), indexed by (heId-1); required for Targeted.
 *                Device parents are 0-based node indices: download them
 *                with HsospState::downloadParentIds.
 */
HgBatch generateBatch(const HostHypergraph& hg, const GenParams& gen,
                      const BatchParams& bp,
                      const std::vector<long long>& dist,
                      const std::vector<int>& parent);

/**
 * @brief Loads a real hypergraph with the paper's preprocessing.
 *
 * Input: one hyperedge per line, vertex ids separated by spaces, tabs or
 * commas (e.g. coauth-DBLP). Ids are signed 64-bit integers. Duplicate
 * vertices of a line are merged, hyperedges with more than
 * @p maxCardinality vertices are dropped, and the vertex ids are
 * renumbered 0..n-1 in order of first appearance. Repeated lines are not
 * merged (each becomes its own hyperedge). Real
 * hyperedges get weights U[1,100] (seeded; the paper does not say how it
 * weights real data). The virtual source hyperedge {s} is row 1 (s = a
 * vertex of maximum degree) and the virtual target {t} the last row (t =
 * a uniformly random vertex), both with weight 0.
 *
 * @throws std::runtime_error if the file cannot be read, has a token
 *         that is not an integer or does not fit 64 bits, has more than
 *         2^31 - 1 distinct vertices, or has no hyperedge within the
 *         cardinality bound.
 */
GeneratedHypergraph loadHypergraphFile(const std::string& path,
                                       int maxCardinality,
                                       std::uint64_t seed);

/**
 * @brief Change batch following the paper's model (Section VI).
 *
 * Hyperedge batch: @p delPct % of @p size are deletions of distinct random
 * hyperedges; each insertion clones a random hyperedge, replaces about
 * @p replaceFrac of its vertices by vertices of a neighbouring hyperedge
 * (one that shares a vertex with it) and draws a weight U[1,100]. Vertex
 * batch: each change removes a random member of a random hyperedge
 * (probability @p delPct %) or adds a vertex of a neighbouring hyperedge.
 * The virtual source and target hyperedges are never changed. Neighbours
 * are found through the incidence lists (HostHypergraph::v2h).
 */
HgBatch generatePaperBatch(const HostHypergraph& hg, BatchKind kind,
                           int size, double delPct, double replaceFrac,
                           std::mt19937_64& rng);

const char* toString(BatchKind k);
const char* toString(Placement p);

} // namespace escher_mosp

#endif // ESCHER_MOSP_HYPERGRAPH_GEN_HPP
