#ifndef ESCHER_MOSP_HSOSP_CUH
#define ESCHER_MOSP_HSOSP_CUH

/**
 * @file hsosp.cuh
 * @brief Device-side hypergraph SOSP: resident h2h CSR with slack rows,
 *        delta application, and the node-weighted SOSP update / recompute.
 *
 * The h2h line graph of the hypergraph has a special property under the
 * meeting's cost model (stepping into hyperedge h_j costs w_j): every
 * in-edge of node j carries the same weight w_j. The device graph therefore
 * stores ONE symmetric adjacency (colInd) plus a per-node weight array,
 * halving memory versus the in/out CSR pair the MOSP kernels use.
 *
 * The kernels are node-weighted adaptations of the MOSP project's
 * parallelSOSPUpdate kernels (collectCandidates / updateDistances with
 * atomicCAS dedup + atomicAdd worklist compaction); the originals in
 * mosp/src are untouched.
 *
 * Rows keep slack capacity (aligned to 32, ESCHER-style). A batch delta is
 * applied by one thread per touched row (race-free by construction, the
 * same grouping idea as the MOSP paper's Step 0); rows that outgrow their
 * capacity relocate to the tail bump region. If the tail region is
 * exhausted an overflow flag is raised and the caller rebuilds the device
 * graph from the host shadow (correctness fallback, counted).
 */

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HostHypergraph.hpp"

namespace escher_mosp {
namespace hsosp {

/**
 * The last batch delta in device memory (0-based node indices), derived by
 * applyBatch and read by hsospUpdate: deleted pairs first, then inserted
 * pairs; ids = new (inserted or recreated) nodes, then dead ones.
 */
struct DeviceDelta {
    int2* d_pairs = nullptr;
    int* d_ids = nullptr;
    long long* d_newW = nullptr;     ///< weights of the new nodes
    long long pairCapacity = 0;
    long long idCapacity = 0;
    int numDel = 0;
    int numIns = 0;
    int numNew = 0;
    int numDead = 0;

    // Scratch of the device-side grouping of the CSR apply: directed
    // row keys (double buffered for the radix sort), row runs and offsets,
    // and the CUB temporary storage. Grown on demand, reused across batches.
    unsigned long long* d_keys = nullptr;
    unsigned long long* d_keysAlt = nullptr;
    int* d_rows = nullptr;
    int* d_rowLen = nullptr;
    int* d_rowOff = nullptr;
    int* d_numRows = nullptr;
    void* d_temp = nullptr;
    std::size_t tempBytes = 0;
    long long keyCapacity = 0;
    /// After the CSR apply: the sorted directed keys (row << 33 |
    /// isInsert << 32 | col), 2 per pair (one of d_keys / d_keysAlt).
    const unsigned long long* d_sortedKeys = nullptr;
    long long numSortedKeys = 0;
    /// Pinned host copy of the sorted keys (for the CBST maintenance).
    unsigned long long* h_sortedKeys = nullptr;
    long long hostKeyCapacity = 0;

    void reserve(long long pairs, long long ids);
    void reserveKeys(long long keys);
    void reserveTemp(std::size_t bytes);
    void free();
};

/**
 * Device mirror of the incidence model, input of the line-graph delta
 * derivation: the vertex list of every hyperedge and the hyperedge list of
 * every vertex, as slack rows. The host decides the layout (it knows every
 * row length) and keeps a copy of the offsets and capacities; rows that
 * outgrow their capacity move to the tail, and a full tail triggers a
 * re-upload with more capacity.
 */
struct DeviceIncidence {
    int maxNodes = 0;
    int numVertices = 0;
    // Hyperedge -> vertices (node-indexed).
    long long* d_heOff = nullptr;
    int* d_heLen = nullptr;
    int* d_heVal = nullptr;
    long long heCapacity = 0;
    long long heTail = 0;
    std::vector<long long> heOff;
    std::vector<int> heCap;
    // Vertex -> hyperedges (0-based node indices).
    long long* d_vOff = nullptr;
    int* d_vLen = nullptr;
    int* d_vVal = nullptr;
    long long vCapacity = 0;
    long long vTail = 0;
    std::vector<long long> vOff;
    std::vector<int> vCap;
    // Per-batch scratch, grown on demand.
    int* d_touchedIdx = nullptr;          ///< node -> touched slot or -1
    int* d_postLen = nullptr;             ///< vertex -> post-batch length
    struct Scratch;
    Scratch* scratch = nullptr;

    DeviceIncidence() = default;
    ~DeviceIncidence();
    DeviceIncidence(const DeviceIncidence&) = delete;
    DeviceIncidence& operator=(const DeviceIncidence&) = delete;
    DeviceIncidence(DeviceIncidence&&) noexcept;
    DeviceIncidence& operator=(DeviceIncidence&&) noexcept;

    /** Uploads the incidence of @p hg; the value arrays get free tail
     *  space of (headroom - 1) x their initial size (+64 entries). */
    void build(const HostHypergraph& hg, int maxNodes, double headroom);
    long long deviceBytes() const;
    void free();
};

/** Resident device h2h graph. Node index = heId - 1. */
struct DeviceH2H {
    int maxNodes = 0;              ///< capacity of the node-indexed arrays
    int numNodes = 0;              ///< nodes in use ( == host maxId() )
    long long capEntries = 0;      ///< capacity of colInd
    long long usedEntries = 0;     ///< host mirror of the tail cursor

    long long* d_rowStart = nullptr;
    int* d_deg = nullptr;
    int* d_cap = nullptr;
    int* d_colInd = nullptr;
    long long* d_nodeW = nullptr;
    unsigned long long* d_tailCursor = nullptr;
    int* d_overflowFlag = nullptr;
    long long numEntries = 0;      ///< live adjacency entries (2 x pairs)

    DeviceDelta delta;             ///< last applied batch delta
    DeviceIncidence inc;           ///< incidence mirror for the delta

    DeviceH2H() = default;
    ~DeviceH2H();
    DeviceH2H(const DeviceH2H&) = delete;
    DeviceH2H& operator=(const DeviceH2H&) = delete;
    DeviceH2H(DeviceH2H&&) noexcept;
    DeviceH2H& operator=(DeviceH2H&&) noexcept;

    long long deviceBytes() const;
    void free();
};

/**
 * Build (or rebuild) the resident device graph: the CSR from the line
 * graph @p lg of @p hg, and the incidence mirror from @p hg.
 *
 * @param hg            host hypergraph (incidence + weights + liveness).
 * @param lg            its line graph (HostHypergraph::lineGraph).
 * @param maxNodes      node capacity; must cover every id the run will see.
 * @param entryHeadroom colInd capacity = initial slack entries *
 *                      entryHeadroom (+4096); growth room for relocations.
 *                      Must be >= 1 (std::invalid_argument otherwise).
 */
void buildDeviceH2H(DeviceH2H& dev, const HostHypergraph& hg,
                    const LineGraphCSR& lg, int maxNodes,
                    double entryHeadroom);

/** Timings (ms) and counts of one batch (applyBatch). */
struct BatchTimes {
    double escherMs = 0.0;   ///< ESCHER maintenance (all three CBSTs)
    double deltaMs = 0.0;    ///< unification: host incidence + GPU delta
    double csrMs = 0.0;      ///< device CSR apply (and a rebuild if needed)
    long long candidates = 0;   ///< candidate pairs examined
    long long insPairs = 0;     ///< net line-graph pairs inserted
    long long delPairs = 0;     ///< net line-graph pairs deleted
    int newHe = 0;
    int deadHe = 0;
    int skippedOps = 0;
    bool csrRebuilt = false;    ///< the CSR tail overflowed
    int mirrorReuploads = 0;    ///< incidence mirror tails that overflowed
};

/**
 * Applies one batch to the whole pipeline except the SOSP update:
 *  1. DynamicHypergraph::beginBatch: h2v insert (ids) + host incidence;
 *  2. unification on the GPU: for every changed incidence (v, h), every o
 *     in v's hyperedge list before or after the batch gives a candidate
 *     pair (h, o); each distinct candidate is classified by whether the
 *     two vertex lists overlap before and after the batch (inserted if
 *     only after, deleted if only before). This is order-free and exact:
 *     a pair can only change when a shared vertex was added or removed.
 *     The pairs go to dev.delta;
 *  3. CSR apply (device sort + warp per row), rebuilding the CSR from the
 *     host incidence if its tail overflows (timed in csrMs);
 *  4. DynamicHypergraph::finishBatch: the remaining CBST maintenance from
 *     the sorted pairs.
 * Stages are separated by device synchronization so each time is complete.
 */
BatchTimes applyBatch(DynamicHypergraph& dh, DeviceH2H& dev,
                      const HgBatch& batch, double entryHeadroom = 1.6);

/** Number of rows of the incidence mirror (hyperedge -> vertices and
 *  vertex -> hyperedges, compared as sorted lists) that differ from @p hg;
 *  a test check, it copies the whole mirror to the host. */
long long incidenceMirrorMismatches(const DeviceIncidence& inc,
                                    const HostHypergraph& hg);

/** Copies the rows of nodes 0..m-1 to the host as sorted 1-based
 *  hyperedge id lists (duplicates kept), for checks against an oracle. */
std::vector<std::vector<int>> downloadRows(const DeviceH2H& dev, int m);

/** Persistent SOSP state (distances over hyperedge nodes). */
struct HsospState {
    int maxNodes = 0;
    long long* d_dist = nullptr;
    int* d_parent = nullptr;
    int* d_isAffected = nullptr;
    int* d_isCandidate = nullptr;
    int* d_candList = nullptr;
    int* d_affList = nullptr;
    int* d_counters = nullptr;   // [0] = affected, [1] = candidates

    // Incremental update (hsospUpdate): packed (dist << 32 | parent) words,
    // invalidation marks and pointer-jumping ancestors (double buffered),
    // frontier lists with epoch stamps, device counters and their pinned
    // host mirror.
    unsigned long long* d_packed = nullptr;
    int* d_invA = nullptr;
    int* d_invB = nullptr;
    int* d_jumpA = nullptr;
    int* d_jumpB = nullptr;
    int* d_front = nullptr;
    int* d_next = nullptr;
    int* d_stamp = nullptr;
    unsigned long long* d_updCounters = nullptr;
    unsigned long long* h_updCounters = nullptr;
    int epoch = 0;

    HsospState() = default;
    ~HsospState();
    HsospState(const HsospState&) = delete;
    HsospState& operator=(const HsospState&) = delete;
    HsospState(HsospState&&) noexcept;
    HsospState& operator=(HsospState&&) noexcept;

    void allocate(int maxNodes);
    long long deviceBytes() const;
    void free();

    void downloadDistances(std::vector<long long>& dist, int n) const;
    /** Parents as 0-based node indices (the device convention), -1 for
     *  none. */
    void downloadParents(std::vector<int>& parent, int n) const;
    /** Parents as 1-based hyperedge ids (the host convention, e.g. for
     *  generateBatch), -1 for none. */
    void downloadParentIds(std::vector<int>& parentIds, int n) const;
};

struct UpdateConfig {
    /// Update budget: push iterations before falling back to a recompute.
    int maxIterations = 4096;
    /// Update budget: edge relaxations (pull + push) before falling back,
    /// as a multiple of the live adjacency entries.
    double workBudget = 1.0;
    /// Threads per block of the recompute kernels (hsospRecompute and the
    /// update's fallback), which run one warp per node: a multiple of 32
    /// in [32, 1024], else both entry points throw std::invalid_argument.
    /// The update kernels always use 256.
    int blockSize = 256;
};

struct UpdateStats {
    int iterations = 0;          ///< update: push iterations; recompute: rounds
    bool fallbackRecompute = false;
    int fallbackIterations = 0;  ///< rounds of the fallback recompute
    int maxFrontier = 0;
    long long seedCount = 0;     ///< update: invalidated nodes
    int jumpRounds = 0;          ///< pointer-jumping rounds
    long long work = 0;          ///< edge relaxations of the update
};

/**
 * Exact dynamic SOSP update for the delta last applied by
 * applyBatch (dev.delta):
 *  1. roots: the endpoint b of every deleted pair (a, b) whose tree parent
 *     was a, and every new, recreated or dead node;
 *  2. every descendant of a root in the pre-batch shortest-path tree is
 *     invalidated (pointer jumping over the parent array): its distance
 *     becomes INF, so every finite distance left is realised by a path of
 *     the new graph (no stale value can count to infinity);
 *  3. every invalidated node pulls the best (distance, id) over its
 *     neighbours, and both directions of every inserted pair are relaxed;
 *  4. improved nodes push to their neighbours with a packed 64-bit
 *     atomicMin on (distance << 32 | parent) until no distance decreases.
 * Distances only decrease after step 2, so the loop needs no cap; ties go
 * to the lowest parent id, so a canonical input tree gives the canonical
 * tree of the new graph. If the update exceeds its budget (cfg) or a
 * distance does not fit 32 bits, it falls back to hsospRecompute; the
 * fallback is reported in the stats and included in the call's time.
 */
UpdateStats hsospUpdate(const DeviceH2H& dev, HsospState& st, int sourceId,
                        const UpdateConfig& cfg);

/** Static baseline: recompute from blank (GPU Bellman-Ford with frontier
 *  dedup, seeded at the source), on the current device graph. Parent ties
 *  go to the lowest node id (canonical tree). */
UpdateStats hsospRecompute(const DeviceH2H& dev, HsospState& st, int sourceId,
                           const UpdateConfig& cfg);

/** Compare two device distance arrays; returns number of mismatches. */
long long compareDistances(const HsospState& a, const HsospState& b, int n);

} // namespace hsosp
} // namespace escher_mosp

#endif // ESCHER_MOSP_HSOSP_CUH
