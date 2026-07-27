#ifndef ESCHER_MOSP_HOST_HYPERGRAPH_HPP
#define ESCHER_MOSP_HOST_HYPERGRAPH_HPP

/**
 * @file HostHypergraph.hpp
 * @brief Host-side model of a dynamic weighted hypergraph and its h2h
 *        (hyperedge-to-hyperedge / line graph) view.
 *
 * This is the authoritative *shadow* that mirrors the ESCHER CBSTs, in the
 * same spirit as DynamicGraph's host shadow for ordinary graphs: every
 * dynamic update flows through the ESCHER structures (see
 * DynamicHypergraph), while reads (CSR materialization, delta extraction,
 * ground truth) are served from this linear-time host model.
 *
 * Everything in this translation unit is plain C++ (no CUDA), so the
 * correctness-critical delta rules are unit-testable off-GPU.
 *
 * Model (from the project meeting):
 *  - Hyperedge h_i carries one non-negative weight w_i.
 *  - h2h edge (h_i, h_j) exists iff the two hyperedges share >= 1 vertex.
 *  - Stepping into h_j costs w_j, so ALL in-edges of node j in the line
 *    graph have weight w_j; the h2h adjacency is therefore kept as a single
 *    symmetric neighbor list plus a per-node weight array.
 *  - Hypergraph is undirected (meeting decision).
 *
 * Conventions:
 *  - Hyperedge ids are 1-based (0 is invalid); vertex ids are 0-based.
 *  - heVerts rows are kept sorted so overlap tests are O(c).
 */

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace escher_mosp {

/** One batch of hypergraph changes, applied in this fixed op order:
 *  hyperedge deletions -> incident vertex deletions -> incident vertex
 *  insertions -> hyperedge insertions (deletions first, matching the MOSP
 *  pipeline convention). */
struct HgBatch {
    struct HeIns {
        std::vector<int> vertices;   ///< 0-based, need not be sorted
        long long weight = 1;
    };
    struct VtxChange {
        int heId = 0;                ///< 1-based hyperedge id
        int vertex = 0;              ///< 0-based vertex id
    };
    std::vector<int>       heDelete;  ///< 1-based hyperedge ids
    std::vector<VtxChange> vtxDelete;
    std::vector<VtxChange> vtxInsert;
    std::vector<HeIns>     heInsert;

    std::size_t totalOps() const {
        return heDelete.size() + vtxDelete.size() + vtxInsert.size() +
               heInsert.size();
    }
};

/** Net structural effect of a batch on the h2h line graph. */
struct H2HDelta {
    /// Undirected pairs (a,b), a<b, that exist after the batch but not before.
    std::vector<std::pair<int, int>> insEdges;
    /// Undirected pairs (a,b), a<b, that existed before but not after.
    std::vector<std::pair<int, int>> delEdges;
    /// Final ids of hyperedges inserted by the batch (includes recycled ids).
    std::vector<int> newHe;
    /// Ids deleted by the batch and still dead at the end of it.
    std::vector<int> deadHe;
    /// Ids deleted at some point during the batch (superset of deadHe).
    std::vector<int> deletedAtAnyPoint;
    /// Seed candidates for the SOSP update: every alive node whose distance
    /// may have changed (endpoints of touched edges, new nodes, recreated
    /// nodes). Excludes the source node and dead nodes.
    std::vector<int> seeds;
    /// Number of batch ops skipped as no-ops (dead target, missing vertex...).
    int skippedOps = 0;
};

/** Batched instructions for the ESCHER routing layer, grouped by the phase
 *  in which they must execute. Values follow the CBST payload conventions:
 *  hyperedge ids stored as-is (1-based), vertex ids stored +1, so payload
 *  values are always strictly positive (ESCHER treats 0 / INT_MIN as
 *  empty / sentinel). */
struct EscherHorizOps {
    // Phase A+B (deletion-driven), execute before fills:
    std::vector<std::pair<int, int>> v2hUnfill;  ///< (vertex+1, heId)
    std::vector<std::pair<int, int>> h2vUnfill;  ///< (heId, vertex+1)
    std::vector<std::pair<int, int>> h2hUnfill;  ///< (rowHeId, valueHeId)
    // Phase C+D (insertion-driven), execute after unfills:
    std::vector<std::pair<int, int>> v2hFill;    ///< (vertex+1, heId)
    std::vector<std::pair<int, int>> h2vFill;    ///< (heId, vertex+1)
    std::vector<std::pair<int, int>> h2hFill;    ///< (rowHeId, valueHeId)
};

class HostHypergraph {
public:
    static const long long INF;

    int numVertices = 0;

    // Indexed by (heId - 1):
    std::vector<std::vector<int>> heVerts;  ///< sorted vertex lists; empty if dead
    std::vector<long long>        heW;
    std::vector<std::uint8_t>     alive;

    std::vector<int> freeIds;               ///< dead ids available for reuse (LIFO)

    std::vector<std::vector<int>> v2h;      ///< per vertex: alive he ids (unsorted)
    std::vector<std::vector<int>> h2h;      ///< per (heId-1): alive neighbor he ids (unsorted)

    long long h2hPairCount = 0;             ///< number of undirected h2h pairs
    int aliveCount = 0;

    int sourceHe = 0;                       ///< virtual source hyperedge id
    int targetHe = 0;                       ///< virtual target hyperedge id

    /** Bulk build from rows (vertex lists) + weights. Row i becomes
     *  hyperedge id i+1. Rows are sorted in place. Builds v2h and the full
     *  h2h line graph. */
    void buildFrom(int nVerts,
                   std::vector<std::vector<int>>&& rows,
                   std::vector<long long>&& weights);

    int maxId() const { return static_cast<int>(heVerts.size()); }

    /** Tentative ids for @p count insertions: recycled dead ids first, then
     *  fresh ids past maxId(). Does not commit anything. */
    std::vector<int> reserveIds(int count) const;

    /** Apply a batch. @p finalIds are the ids to use for b.heInsert (one per
     *  entry; from reserveIds or from the ESCHER insert mapping). Fills
     *  @p delta and @p ops. */
    void applyBatch(const HgBatch& b, const std::vector<int>& finalIds,
                    H2HDelta& delta, EscherHorizOps& ops);

    /** Ground-truth node-weighted SSSP on the current h2h graph.
     *  Returns dist indexed by (heId-1); INF for dead / unreachable. */
    std::vector<long long> dijkstra(int sourceId) const;

    /** Brute-force rebuild of the h2h adjacency from heVerts (test oracle).
     *  Returns per-(id-1) sorted neighbor lists. */
    std::vector<std::vector<int>> bruteForceH2H() const;

    /** Total h2h adjacency entries (2 * pair count). */
    long long h2hEntryCount() const { return 2 * h2hPairCount; }

private:
    bool overlaps_(int a, int b) const;          // heVerts intersection test
    static void removeValue_(std::vector<int>& v, int value);
};

/**
 * @brief Sequential emulation of the device SOSP update loop.
 *
 * Mirrors hsospUpdate()'s semantics exactly (recompute-best-parent per
 * candidate, propagate while distances change, iteration cap with
 * recompute-from-blank fallback) so the algorithm's correctness against
 * Dijkstra can be validated without a GPU. Returns the iteration count
 * actually used, or -1 if the fallback recompute was taken.
 */
int emulateSospUpdate(const HostHypergraph& hg,
                      std::vector<long long>& dist,
                      std::vector<int>& parent,
                      const std::vector<int>& seeds,
                      int maxIterations);

/** Sequential emulation of recompute-from-blank (static baseline). */
int emulateSospRecompute(const HostHypergraph& hg,
                         std::vector<long long>& dist,
                         std::vector<int>& parent,
                         int maxIterations);

} // namespace escher_mosp

#endif // ESCHER_MOSP_HOST_HYPERGRAPH_HPP
