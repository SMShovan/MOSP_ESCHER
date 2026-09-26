#ifndef ESCHER_MOSP_HOST_HYPERGRAPH_HPP
#define ESCHER_MOSP_HOST_HYPERGRAPH_HPP

/**
 * @file HostHypergraph.hpp
 * @brief Host-side incidence model of a dynamic weighted hypergraph.
 *
 * The host keeps the small, branchy part of the state: the sorted vertex
 * list of every hyperedge (heVerts), the vertex -> hyperedge incidence
 * lists (v2h), weights, liveness and the free-id list. It validates each
 * batch and turns it into incidence changes (IncidenceBatch). The line
 * graph (h2h) itself is not kept on the host: its net change for a batch
 * is derived on the GPU from the incidence changes (see hsosp.cuh), and the
 * host builds it only at load (lineGraph()).
 *
 * Everything in this translation unit is plain C++ (no CUDA), so it is
 * unit-testable off-GPU (tests/local).
 *
 * Model (from the project meeting and the paper):
 *  - Hyperedge h_i carries one positive weight w_i >= 1 (the virtual
 *    source and target hyperedges: 0). DynamicHypergraph rejects other
 *    weights; with a zero-weight pair cut off from the source the update
 *    would keep a stale finite distance.
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

/**
 * Incidence changes of one batch (what the host ships to the GPU, which
 * derives the net line-graph delta from them).
 */
struct IncidenceBatch {
    /// (vertex << 32 | heId) of every incidence added (+1) or removed (-1);
    /// a pair may appear more than once (e.g. removed, then re-added).
    std::vector<std::uint64_t> incKey;
    std::vector<int> incSign;
    /// Hyperedges whose vertex list changed (unique ids), with their vertex
    /// lists before and after the batch (CSR; dead = empty).
    std::vector<int> touched;
    std::vector<int> preOff, preVals;
    std::vector<int> postOff, postVals;
    /// Vertices whose incidence list changed (unique), with the length of
    /// their list after the batch.
    std::vector<int> touchedVertices;
    std::vector<int> touchedVertexLen;
    /// Final ids of hyperedges inserted by the batch (includes recycled
    /// ids) and their weights.
    std::vector<int> newHe;
    std::vector<long long> newW;
    /// Ids deleted by the batch and still dead at the end of it.
    std::vector<int> deadHe;
    /// Number of batch ops skipped as no-ops (dead target, missing vertex...).
    int skippedOps = 0;
};

/** Net structural effect of a batch on the h2h line graph (test oracles
 *  and the host emulation of the update). */
struct H2HDelta {
    /// Undirected pairs (a,b), a<b, that exist after the batch but not before.
    std::vector<std::pair<int, int>> insEdges;
    /// Undirected pairs (a,b), a<b, that existed before but not after.
    std::vector<std::pair<int, int>> delEdges;
    /// Final ids of hyperedges inserted by the batch (includes recycled ids).
    std::vector<int> newHe;
    /// Ids deleted by the batch and still dead at the end of it.
    std::vector<int> deadHe;
};

/** Batched instructions for the ESCHER routing layer. Values follow the
 *  CBST payload conventions: hyperedge ids stored as-is (1-based), vertex
 *  ids stored +1, so payload values are always strictly positive (ESCHER
 *  treats 0 / INT_MIN as empty / sentinel). The h2h CBST operations come
 *  from the line-graph delta (DynamicHypergraph::finishBatch). */
struct EscherHorizOps {
    // Deletion-driven, execute before fills:
    std::vector<std::pair<int, int>> v2hUnfill;  ///< (vertex+1, heId)
    std::vector<std::pair<int, int>> h2vUnfill;  ///< (heId, vertex+1)
    // Insertion-driven, execute after unfills:
    std::vector<std::pair<int, int>> v2hFill;    ///< (vertex+1, heId)
    std::vector<std::pair<int, int>> h2vFill;    ///< (heId, vertex+1)
};

/** Line graph in CSR form; row (id - 1) lists the ids of the alive
 *  hyperedges sharing a vertex with alive hyperedge id, sorted ascending. */
struct LineGraphCSR {
    int numIds = 0;                  ///< == hg.maxId() when built
    std::vector<long long> offset;   ///< numIds + 1 entries
    std::vector<int> nbr;            ///< 1-based hyperedge ids

    long long degree(int id) const { return offset[id] - offset[id - 1]; }
    const int* row(int id) const { return nbr.data() + offset[id - 1]; }
    long long numEntries() const { return offset.empty() ? 0 : offset.back(); }
    bool adjacent(int a, int b) const;
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
    std::vector<int> freePos;               ///< position of an id in freeIds (-1: none)

    std::vector<std::vector<int>> v2h;      ///< per vertex: alive he ids (unsorted)

    long long h2hPairCount = 0;             ///< undirected line-graph pairs
    int aliveCount = 0;

    int sourceHe = 0;                       ///< virtual source hyperedge id
    int targetHe = 0;                       ///< virtual target hyperedge id

    /** Bulk build from rows (vertex lists) + weights. Row i becomes
     *  hyperedge id i+1. Rows are sorted in place. Builds v2h and counts the
     *  line-graph pairs. */
    void buildFrom(int nVerts,
                   std::vector<std::vector<int>>&& rows,
                   std::vector<long long>&& weights);

    int maxId() const { return static_cast<int>(heVerts.size()); }

    /** Tentative ids for @p count insertions: recycled dead ids first, then
     *  fresh ids past maxId(). Does not commit anything. */
    std::vector<int> reserveIds(int count) const;

    /** Apply a batch to the incidence model. @p finalIds are the ids to use
     *  for b.heInsert (one per entry; from reserveIds or from the ESCHER
     *  insert mapping). Fills @p inc (for the GPU line-graph delta) and
     *  @p ops (ESCHER h2v / v2h operations). */
    void applyBatch(const HgBatch& b, const std::vector<int>& finalIds,
                    IncidenceBatch& inc, EscherHorizOps& ops);

    /** The current line graph (per-hyperedge union of the incidence lists,
     *  OpenMP-parallel). */
    LineGraphCSR lineGraph() const;

    /** Brute-force rebuild of the h2h adjacency from heVerts (test oracle).
     *  Returns per-(id-1) sorted neighbor lists. */
    std::vector<std::vector<int>> bruteForceH2H() const;

    /** Total h2h adjacency entries (2 * pair count). */
    long long h2hEntryCount() const { return 2 * h2hPairCount; }

private:
    void freeListRemove_(int id);
    void freeListPush_(int id);

    // applyBatch scratch, all -1 / 0 between batches: slot of a touched
    // hyperedge in IncidenceBatch::touched, touched-vertex flags.
    std::vector<int> heTouchSlot_;
    std::vector<std::uint8_t> vertexTouched_;
};

/**
 * @brief Sequential emulation of the device SOSP update (hsospUpdate).
 *
 * Same steps as the device, so the algorithm can be validated without a
 * GPU: invalidate the pre-batch subtree of every deleted tree edge and of
 * every new, recreated or dead node (distance INF), pull the best
 * (distance, id) for the invalidated nodes, relax both directions of the
 * inserted pairs, then propagate the decreases (in Dijkstra order; the
 * device pushes in frontiers, which gives the same result). @p lg is the
 * post-batch line graph. Parents are 1-based ids (-1 = none) and ties go
 * to the lowest id. Returns the number of invalidated nodes.
 */
int emulateSospUpdate(const HostHypergraph& hg, const LineGraphCSR& lg,
                      std::vector<long long>& dist,
                      std::vector<int>& parent,
                      const H2HDelta& delta);

/** Sequential emulation of recompute-from-blank (static baseline). */
int emulateSospRecompute(const HostHypergraph& hg, const LineGraphCSR& lg,
                         std::vector<long long>& dist,
                         std::vector<int>& parent,
                         int maxIterations);

} // namespace escher_mosp

#endif // ESCHER_MOSP_HOST_HYPERGRAPH_HPP
