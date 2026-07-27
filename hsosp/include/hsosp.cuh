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
 * @param entryHeadroom colInd capacity = ceil(initial slack entries *
 *                      entryHeadroom); growth room for relocations.
 */
void buildDeviceH2H(DeviceH2H& dev, const HostHypergraph& hg, int maxNodes,
                    double entryHeadroom);

/**
 * Apply a net H2HDelta to the resident device graph.
 *
 * @return false if the tail region overflowed (caller must rebuild via
 *         buildDeviceH2H; the graph contents are unspecified until then).
 */
bool applyDeltaToDevice(DeviceH2H& dev, const HostHypergraph& hg,
                        const H2HDelta& delta);

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
    void downloadParents(std::vector<int>& parent, int n) const;
};

struct UpdateConfig {
    int maxIterations = 512;   ///< convergence cap before the fallback
    int blockSize = 256;
};

struct UpdateStats {
    int iterations = 0;
    bool fallbackRecompute = false;
    int maxFrontier = 0;
    long long seedCount = 0;
};

/**
 * Dynamic SOSP update after a batch: seeds are the delta's touched nodes
 * (1-based he ids); dead nodes get dist=INF first. If the propagation does
 * not converge within cfg.maxIterations (stale loop in a disconnected
 * region), falls back to a full recompute and reports it in the stats.
 */
UpdateStats hsospUpdate(const DeviceH2H& dev, HsospState& st,
                        const std::vector<int>& seedIds,
                        const std::vector<int>& deadIds, int sourceId,
                        const UpdateConfig& cfg);

/** Static baseline: recompute from blank (GPU Bellman-Ford with frontier
 *  dedup, seeded at the source), on the current device graph. */
UpdateStats hsospRecompute(const DeviceH2H& dev, HsospState& st, int sourceId,
                           const UpdateConfig& cfg);

/** Compare two device distance arrays; returns number of mismatches. */
long long compareDistances(const HsospState& a, const HsospState& b, int n);

} // namespace hsosp
} // namespace escher_mosp

#endif // ESCHER_MOSP_HSOSP_CUH
