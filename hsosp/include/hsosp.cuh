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

#include "HostHypergraph.hpp"

namespace escher_mosp {
namespace hsosp {

/**
 * The last batch delta in device memory (0-based node indices), uploaded
 * by applyDeltaToDevice and read by hsospUpdate: deleted pairs first, then
 * inserted pairs; ids = new (inserted or recreated) nodes, then dead ones.
 */
struct DeviceDelta {
    int2* d_pairs = nullptr;
    int* d_ids = nullptr;
    long long pairCapacity = 0;
    long long idCapacity = 0;
    int numDel = 0;
    int numIns = 0;
    int numNew = 0;
    int numDead = 0;

    void reserve(long long pairs, long long ids);
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
 * Build (or rebuild) the resident device graph from the host shadow.
 *
 * @param hg            host hypergraph (adjacency + weights + liveness).
 * @param maxNodes      node capacity; must cover every id the run will see.
 * @param entryHeadroom colInd capacity = initial slack entries *
 *                      entryHeadroom (+4096); growth room for relocations.
 *                      Must be >= 1 (std::invalid_argument otherwise).
 */
void buildDeviceH2H(DeviceH2H& dev, const HostHypergraph& hg, int maxNodes,
                    double entryHeadroom);

/**
 * Apply a net H2HDelta to the resident device graph and keep the delta on
 * the device (dev.delta) for the following hsospUpdate.
 *
 * @return false if the tail region overflowed (caller must rebuild via
 *         buildDeviceH2H, which keeps dev.delta; the graph contents are
 *         unspecified until then).
 */
bool applyDeltaToDevice(DeviceH2H& dev, const HostHypergraph& hg,
                        const H2HDelta& delta);

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
 * applyDeltaToDevice (dev.delta):
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
