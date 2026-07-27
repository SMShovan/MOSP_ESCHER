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
 * Generate a change batch against the CURRENT hypergraph state.
 *
 * @param hg      current host hypergraph (for alive ids / pools membership).
 * @param gen     generator parameters (pool model for inserted hyperedges).
 * @param bp      batch parameters.
 * @param dist    current SOSP distances indexed by (heId-1); required for
 *                Targeted / Near / Far placements (pass empty for Random).
 * @param parent  current SOSP parents; required for Targeted.
 */
HgBatch generateBatch(const HostHypergraph& hg, const GenParams& gen,
                      const BatchParams& bp,
                      const std::vector<long long>& dist,
                      const std::vector<int>& parent);

const char* toString(BatchKind k);
const char* toString(Placement p);

} // namespace escher_mosp

#endif // ESCHER_MOSP_HYPERGRAPH_GEN_HPP
