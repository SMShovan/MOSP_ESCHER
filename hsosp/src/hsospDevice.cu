/**
 * @file hsospDevice.cu
 * @brief Device h2h graph maintenance + node-weighted SOSP kernels.
 *
 * See hsosp.cuh for the design rationale. Kernel structure follows
 * mosp/src/parallelSOSPUpdate.cu (atomicCAS dedup + atomicAdd worklist
 * compaction); adapted to the single symmetric adjacency + node weights.
 */

#include "hsosp.cuh"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace escher_mosp {
namespace hsosp {

#define HSOSP_CUDA_CHECK(call)                                                \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err__) + " at " +     \
                                     __FILE__ + ":" +                         \
                                     std::to_string(__LINE__));               \
        }                                                                     \
    } while (0)

namespace {

// Same value as HostHypergraph::INF (which is not constexpr).
constexpr long long INF_VALUE = std::numeric_limits<long long>::max() / 4;
static_assert(INF_VALUE > 0, "INF_VALUE must be positive");

inline int alignUp32(int x) { return (x + 31) & ~31; }

inline int rowCapacityFor(int deg) {
    // Slack: at least 4 spare slots, at least 25% headroom, 32-aligned
    // (matching ESCHER's warp-aligned block philosophy).
    return alignUp32(deg + std::max(4, deg / 4));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void fillLLKernel(long long* arr, int n, long long value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = value;
}

__global__ void fillIntKernel(int* arr, int n, int value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = value;
}

__global__ void scatterLLKernel(const int* idx, const long long* vals, int n,
                                long long* arr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[idx[tid]] = vals[tid];
}

__global__ void markDeadKernel(const int* idx, int n, int* deg,
                               long long* dist, int* parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = idx[tid];
        deg[v] = 0;
        dist[v] = INF_VALUE;
        parent[v] = -1;
    }
}

/**
 * One thread per touched row: apply that row's deletions (swap-remove) then
 * insertions (append; relocate to the tail region on overflow). Race-free
 * because every touched row appears exactly once (grouped on the host).
 */
__global__ void applyRowChangesKernel(
    int nRows, const int* rows, const long long* delOff, const int* delLen,
    const int* delVals, const long long* insOff, const int* insLen,
    const int* insVals, long long* rowStart, int* deg, int* cap, int* colInd,
    unsigned long long* tailCursor, long long capEntries, int* overflowFlag) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nRows) return;

    const int r = rows[tid];
    long long base = rowStart[r];
    int d = deg[r];

    // Deletions: swap with last.
    const long long doff = delOff[tid];
    const int dn = delLen[tid];
    for (int k = 0; k < dn; ++k) {
        const int v = delVals[doff + k];
        for (int e = 0; e < d; ++e) {
            if (colInd[base + e] == v) {
                colInd[base + e] = colInd[base + d - 1];
                --d;
                break;
            }
        }
    }

    // Insertions: append, relocating the row if capacity is exceeded.
    const int ni = insLen[tid];
    if (ni > 0) {
        if (d + ni > cap[r]) {
            const int need = d + ni;
            int newCap = (need + max(4, need / 4) + 31) & ~31;
            const unsigned long long pos =
                atomicAdd(tailCursor, static_cast<unsigned long long>(newCap));
            if (static_cast<long long>(pos) + newCap > capEntries) {
                *overflowFlag = 1;
                deg[r] = d;
                return;
            }
            for (int e = 0; e < d; ++e) {
                colInd[pos + e] = colInd[base + e];
            }
            rowStart[r] = static_cast<long long>(pos);
            cap[r] = newCap;
            base = static_cast<long long>(pos);
        }
        const long long ioff = insOff[tid];
        for (int k = 0; k < ni; ++k) {
            colInd[base + d] = insVals[ioff + k];
            ++d;
        }
    }
    deg[r] = d;
}

/**
 * Node-weighted variant of updateDistancesKernel: recompute the best parent
 * of each candidate from its (symmetric) neighbor list; every in-edge of v
 * costs nodeW[v]. Candidates are unique (dedup at enqueue), so the
 * dist/parent writes are race-free; dist reads of neighbors are the benign
 * chaotic-relaxation race of the original kernel.
 */
__global__ void updateDistancesNW(
    const int* candList, int numCand, const long long* rowStart,
    const int* deg, const int* colInd, const long long* nodeW,
    long long* dist, int* parent, int* isCandidate, int* isAffected,
    int* affList, int* affCount, int source) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numCand) return;

    const int v = candList[tid];
    isCandidate[v] = 0;
    if (v == source) return;

    long long best = INF_VALUE;
    int bestP = -1;
    const long long base = rowStart[v];
    const int d = deg[v];
    for (int e = 0; e < d; ++e) {
        const int p = colInd[base + e];
        const long long dp = dist[p];
        if (dp >= INF_VALUE / 2) continue;
        const long long cd = dp + nodeW[v];
        if (cd < best) {
            best = cd;
            bestP = p;
        }
    }

    const bool changed = (best != dist[v]);
    parent[v] = bestP;
    dist[v] = best;

    if (changed) {
        if (atomicCAS(&isAffected[v], 0, 1) == 0) {
            const int pos = atomicAdd(affCount, 1);
            affList[pos] = v;
        }
    }
}

/** Node-weighted variant of collectCandidatesKernel. */
__global__ void collectCandidatesNW(const int* affList, int numAff,
                                    const long long* rowStart, const int* deg,
                                    const int* colInd, int* isCandidate,
                                    int* candList, int* candCount,
                                    int* isAffected, int source) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numAff) return;

    const int u = affList[tid];
    isAffected[u] = 0;

    const long long base = rowStart[u];
    const int d = deg[u];
    for (int e = 0; e < d; ++e) {
        const int nb = colInd[base + e];
        if (nb == source) continue;
        if (atomicCAS(&isCandidate[nb], 0, 1) == 0) {
            const int pos = atomicAdd(candCount, 1);
            candList[pos] = nb;
        }
    }
}

__global__ void countMismatchesKernel(const long long* a, const long long* b,
                                      int n, unsigned long long* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && a[tid] != b[tid]) {
        atomicAdd(out, 1ull);
    }
}

int gridFor(long long n, int block) {
    return static_cast<int>((n + block - 1) / block);
}

} // namespace

// ---------------------------------------------------------------------------
// DeviceH2H
// ---------------------------------------------------------------------------

void DeviceH2H::free() {
    if (d_rowStart) cudaFree(d_rowStart);
    if (d_deg) cudaFree(d_deg);
    if (d_cap) cudaFree(d_cap);
    if (d_colInd) cudaFree(d_colInd);
    if (d_nodeW) cudaFree(d_nodeW);
    if (d_tailCursor) cudaFree(d_tailCursor);
    if (d_overflowFlag) cudaFree(d_overflowFlag);
    d_rowStart = nullptr;
    d_deg = nullptr;
    d_cap = nullptr;
    d_colInd = nullptr;
    d_nodeW = nullptr;
    d_tailCursor = nullptr;
    d_overflowFlag = nullptr;
}

DeviceH2H::~DeviceH2H() { free(); }

DeviceH2H::DeviceH2H(DeviceH2H&& o) noexcept { *this = std::move(o); }

DeviceH2H& DeviceH2H::operator=(DeviceH2H&& o) noexcept {
    if (this != &o) {
        free();
        maxNodes = o.maxNodes;
        numNodes = o.numNodes;
        capEntries = o.capEntries;
        usedEntries = o.usedEntries;
        d_rowStart = o.d_rowStart;
        d_deg = o.d_deg;
        d_cap = o.d_cap;
        d_colInd = o.d_colInd;
        d_nodeW = o.d_nodeW;
        d_tailCursor = o.d_tailCursor;
        d_overflowFlag = o.d_overflowFlag;
        o.d_rowStart = nullptr;
        o.d_deg = nullptr;
        o.d_cap = nullptr;
        o.d_colInd = nullptr;
        o.d_nodeW = nullptr;
        o.d_tailCursor = nullptr;
        o.d_overflowFlag = nullptr;
    }
    return *this;
}

long long DeviceH2H::deviceBytes() const {
    return static_cast<long long>(maxNodes) *
               (sizeof(long long) + 2 * sizeof(int) + sizeof(long long)) +
           capEntries * sizeof(int) + sizeof(unsigned long long) + sizeof(int);
}

void buildDeviceH2H(DeviceH2H& dev, const HostHypergraph& hg, int maxNodes,
                    double entryHeadroom) {
    const int m = hg.maxId();
    if (m > maxNodes) {
        throw std::runtime_error(
            "buildDeviceH2H: maxNodes too small for current hypergraph");
    }
    dev.free();
    dev.maxNodes = maxNodes;
    dev.numNodes = m;

    std::vector<long long> rowStart(maxNodes, 0);
    std::vector<int> deg(maxNodes, 0);
    std::vector<int> cap(maxNodes, 0);
    std::vector<long long> nodeW(maxNodes, 0);

    long long cursor = 0;
    for (int id = 1; id <= m; ++id) {
        const int d =
            hg.alive[id - 1] ? static_cast<int>(hg.h2h[id - 1].size()) : 0;
        const int c = rowCapacityFor(d);
        rowStart[id - 1] = cursor;
        deg[id - 1] = d;
        cap[id - 1] = c;
        nodeW[id - 1] = hg.heW[id - 1];
        cursor += c;
    }
    dev.usedEntries = cursor;
    dev.capEntries =
        static_cast<long long>(static_cast<double>(cursor) * entryHeadroom) +
        4096;

    std::vector<int> colInd(static_cast<std::size_t>(dev.capEntries), 0);
    for (int id = 1; id <= m; ++id) {
        if (!hg.alive[id - 1]) continue;
        long long base = rowStart[id - 1];
        const auto& nbs = hg.h2h[id - 1];
        for (std::size_t e = 0; e < nbs.size(); ++e) {
            colInd[base + e] = nbs[e] - 1;   // device nodes are 0-based
        }
    }

    HSOSP_CUDA_CHECK(cudaMalloc(&dev.d_rowStart,
                                sizeof(long long) * maxNodes));
    HSOSP_CUDA_CHECK(cudaMalloc(&dev.d_deg, sizeof(int) * maxNodes));
    HSOSP_CUDA_CHECK(cudaMalloc(&dev.d_cap, sizeof(int) * maxNodes));
    HSOSP_CUDA_CHECK(
        cudaMalloc(&dev.d_colInd, sizeof(int) * dev.capEntries));
    HSOSP_CUDA_CHECK(cudaMalloc(&dev.d_nodeW, sizeof(long long) * maxNodes));
    HSOSP_CUDA_CHECK(
        cudaMalloc(&dev.d_tailCursor, sizeof(unsigned long long)));
    HSOSP_CUDA_CHECK(cudaMalloc(&dev.d_overflowFlag, sizeof(int)));

    HSOSP_CUDA_CHECK(cudaMemcpy(dev.d_rowStart, rowStart.data(),
                                sizeof(long long) * maxNodes,
                                cudaMemcpyHostToDevice));
    HSOSP_CUDA_CHECK(cudaMemcpy(dev.d_deg, deg.data(), sizeof(int) * maxNodes,
                                cudaMemcpyHostToDevice));
    HSOSP_CUDA_CHECK(cudaMemcpy(dev.d_cap, cap.data(), sizeof(int) * maxNodes,
                                cudaMemcpyHostToDevice));
    HSOSP_CUDA_CHECK(cudaMemcpy(dev.d_colInd, colInd.data(),
                                sizeof(int) * dev.capEntries,
                                cudaMemcpyHostToDevice));
    HSOSP_CUDA_CHECK(cudaMemcpy(dev.d_nodeW, nodeW.data(),
                                sizeof(long long) * maxNodes,
                                cudaMemcpyHostToDevice));
    unsigned long long tc = static_cast<unsigned long long>(cursor);
    HSOSP_CUDA_CHECK(cudaMemcpy(dev.d_tailCursor, &tc,
                                sizeof(unsigned long long),
                                cudaMemcpyHostToDevice));
    HSOSP_CUDA_CHECK(cudaMemset(dev.d_overflowFlag, 0, sizeof(int)));
}

bool applyDeltaToDevice(DeviceH2H& dev, const HostHypergraph& hg,
                        const H2HDelta& delta) {
    // Grow the node range for fresh ids introduced by the batch.
    const int m = hg.maxId();
    if (m > dev.maxNodes) {
        throw std::runtime_error(
            "applyDeltaToDevice: node capacity exceeded; raise maxHyperedges");
    }

    // Group per-row insert / delete lists (0-based node indices).
    std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>>
        rowOps;   // row -> (inserts, deletes)
    rowOps.reserve(delta.insEdges.size() * 2 + delta.delEdges.size() * 2 + 8);
    for (auto [a, b] : delta.delEdges) {
        rowOps[a - 1].second.push_back(b - 1);
        rowOps[b - 1].second.push_back(a - 1);
    }
    for (auto [a, b] : delta.insEdges) {
        rowOps[a - 1].first.push_back(b - 1);
        rowOps[b - 1].first.push_back(a - 1);
    }

    const int nRows = static_cast<int>(rowOps.size());
    dev.numNodes = m;

    if (nRows > 0) {
        std::vector<int> rows;
        std::vector<long long> insOff, delOff;
        std::vector<int> insLen, delLen, insVals, delVals;
        rows.reserve(nRows);
        insOff.reserve(nRows);
        delOff.reserve(nRows);
        insLen.reserve(nRows);
        delLen.reserve(nRows);
        for (auto& kv : rowOps) {
            rows.push_back(kv.first);
            insOff.push_back(static_cast<long long>(insVals.size()));
            insLen.push_back(static_cast<int>(kv.second.first.size()));
            for (int v : kv.second.first) insVals.push_back(v);
            delOff.push_back(static_cast<long long>(delVals.size()));
            delLen.push_back(static_cast<int>(kv.second.second.size()));
            for (int v : kv.second.second) delVals.push_back(v);
        }

        auto upload = [](const void* src, std::size_t bytes) {
            void* p = nullptr;
            HSOSP_CUDA_CHECK(cudaMalloc(&p, bytes == 0 ? 4 : bytes));
            if (bytes > 0) {
                HSOSP_CUDA_CHECK(
                    cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
            }
            return p;
        };

        int* d_rows = static_cast<int*>(
            upload(rows.data(), rows.size() * sizeof(int)));
        long long* d_insOff = static_cast<long long*>(
            upload(insOff.data(), insOff.size() * sizeof(long long)));
        int* d_insLen = static_cast<int*>(
            upload(insLen.data(), insLen.size() * sizeof(int)));
        int* d_insVals = static_cast<int*>(
            upload(insVals.data(), insVals.size() * sizeof(int)));
        long long* d_delOff = static_cast<long long*>(
            upload(delOff.data(), delOff.size() * sizeof(long long)));
        int* d_delLen = static_cast<int*>(
            upload(delLen.data(), delLen.size() * sizeof(int)));
        int* d_delVals = static_cast<int*>(
            upload(delVals.data(), delVals.size() * sizeof(int)));

        const int block = 256;
        applyRowChangesKernel<<<gridFor(nRows, block), block>>>(
            nRows, d_rows, d_delOff, d_delLen, d_delVals, d_insOff, d_insLen,
            d_insVals, dev.d_rowStart, dev.d_deg, dev.d_cap, dev.d_colInd,
            dev.d_tailCursor, dev.capEntries, dev.d_overflowFlag);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        HSOSP_CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_rows);
        cudaFree(d_insOff);
        cudaFree(d_insLen);
        cudaFree(d_insVals);
        cudaFree(d_delOff);
        cudaFree(d_delLen);
        cudaFree(d_delVals);
    }

    // Node weights of inserted (or recreated) hyperedges.
    if (!delta.newHe.empty()) {
        std::vector<int> idx;
        std::vector<long long> w;
        idx.reserve(delta.newHe.size());
        for (int id : delta.newHe) {
            idx.push_back(id - 1);
            w.push_back(hg.heW[id - 1]);
        }
        int* d_idx = nullptr;
        long long* d_w = nullptr;
        HSOSP_CUDA_CHECK(cudaMalloc(&d_idx, idx.size() * sizeof(int)));
        HSOSP_CUDA_CHECK(cudaMalloc(&d_w, w.size() * sizeof(long long)));
        HSOSP_CUDA_CHECK(cudaMemcpy(d_idx, idx.data(),
                                    idx.size() * sizeof(int),
                                    cudaMemcpyHostToDevice));
        HSOSP_CUDA_CHECK(cudaMemcpy(d_w, w.data(),
                                    w.size() * sizeof(long long),
                                    cudaMemcpyHostToDevice));
        const int block = 256;
        scatterLLKernel<<<gridFor(idx.size(), block), block>>>(
            d_idx, d_w, static_cast<int>(idx.size()), dev.d_nodeW);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_idx);
        cudaFree(d_w);
    }

    int overflow = 0;
    HSOSP_CUDA_CHECK(cudaMemcpy(&overflow, dev.d_overflowFlag, sizeof(int),
                                cudaMemcpyDeviceToHost));
    unsigned long long tc = 0;
    HSOSP_CUDA_CHECK(cudaMemcpy(&tc, dev.d_tailCursor,
                                sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
    dev.usedEntries = static_cast<long long>(tc);
    return overflow == 0;
}

// ---------------------------------------------------------------------------
// HsospState
// ---------------------------------------------------------------------------

void HsospState::free() {
    if (d_dist) cudaFree(d_dist);
    if (d_parent) cudaFree(d_parent);
    if (d_isAffected) cudaFree(d_isAffected);
    if (d_isCandidate) cudaFree(d_isCandidate);
    if (d_candList) cudaFree(d_candList);
    if (d_affList) cudaFree(d_affList);
    if (d_counters) cudaFree(d_counters);
    d_dist = nullptr;
    d_parent = nullptr;
    d_isAffected = nullptr;
    d_isCandidate = nullptr;
    d_candList = nullptr;
    d_affList = nullptr;
    d_counters = nullptr;
}

HsospState::~HsospState() { free(); }

HsospState::HsospState(HsospState&& o) noexcept { *this = std::move(o); }

HsospState& HsospState::operator=(HsospState&& o) noexcept {
    if (this != &o) {
        free();
        maxNodes = o.maxNodes;
        d_dist = o.d_dist;
        d_parent = o.d_parent;
        d_isAffected = o.d_isAffected;
        d_isCandidate = o.d_isCandidate;
        d_candList = o.d_candList;
        d_affList = o.d_affList;
        d_counters = o.d_counters;
        o.d_dist = nullptr;
        o.d_parent = nullptr;
        o.d_isAffected = nullptr;
        o.d_isCandidate = nullptr;
        o.d_candList = nullptr;
        o.d_affList = nullptr;
        o.d_counters = nullptr;
    }
    return *this;
}

void HsospState::allocate(int n) {
    free();
    maxNodes = n;
    HSOSP_CUDA_CHECK(cudaMalloc(&d_dist, sizeof(long long) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_parent, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_isAffected, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_isCandidate, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_candList, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_affList, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_counters, sizeof(int) * 2));

    const int block = 256;
    fillLLKernel<<<gridFor(n, block), block>>>(d_dist, n, INF_VALUE);
    fillIntKernel<<<gridFor(n, block), block>>>(d_parent, n, -1);
    HSOSP_CUDA_CHECK(cudaMemset(d_isAffected, 0, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMemset(d_isCandidate, 0, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
}

long long HsospState::deviceBytes() const {
    return static_cast<long long>(maxNodes) *
               (sizeof(long long) + 5 * sizeof(int)) +
           2 * sizeof(int);
}

void HsospState::downloadDistances(std::vector<long long>& dist,
                                   int n) const {
    dist.resize(n);
    HSOSP_CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, sizeof(long long) * n,
                                cudaMemcpyDeviceToHost));
}

void HsospState::downloadParents(std::vector<int>& parent, int n) const {
    parent.resize(n);
    HSOSP_CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, sizeof(int) * n,
                                cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Propagation loop
// ---------------------------------------------------------------------------

namespace {

/**
 * Core loop shared by update and recompute.
 *
 * @param startWithCandidates  true: initialList holds candidate nodes to
 *        evaluate first (dynamic update seeds). false: initialList holds
 *        affected nodes whose neighbors are collected first (recompute
 *        seeded at the source).
 * @return iterations used, or -1 if the cap was hit before convergence.
 */
int propagate(const DeviceH2H& dev, HsospState& st,
              const std::vector<int>& initialList, bool startWithCandidates,
              int source0, const UpdateConfig& cfg, int* maxFrontierOut) {
    const int block = cfg.blockSize;
    int* d_affCount = st.d_counters;
    int* d_candCount = st.d_counters + 1;

    int listCount = static_cast<int>(initialList.size());
    if (listCount == 0) return 0;

    int* d_initial = startWithCandidates ? st.d_candList : st.d_affList;
    HSOSP_CUDA_CHECK(cudaMemcpy(d_initial, initialList.data(),
                                sizeof(int) * listCount,
                                cudaMemcpyHostToDevice));

    int iterations = 0;
    int maxFrontier = listCount;
    int candCount = 0;

    if (!startWithCandidates) {
        // Prime: collect candidates from the initial affected list.
        HSOSP_CUDA_CHECK(cudaMemset(d_candCount, 0, sizeof(int)));
        collectCandidatesNW<<<gridFor(listCount, block), block>>>(
            st.d_affList, listCount, dev.d_rowStart, dev.d_deg, dev.d_colInd,
            st.d_isCandidate, st.d_candList, d_candCount, st.d_isAffected,
            source0);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        HSOSP_CUDA_CHECK(cudaMemcpy(&candCount, d_candCount, sizeof(int),
                                    cudaMemcpyDeviceToHost));
    } else {
        candCount = listCount;
    }

    while (candCount > 0) {
        if (iterations >= cfg.maxIterations) return -1;
        ++iterations;
        maxFrontier = std::max(maxFrontier, candCount);

        HSOSP_CUDA_CHECK(cudaMemset(d_affCount, 0, sizeof(int)));
        updateDistancesNW<<<gridFor(candCount, block), block>>>(
            st.d_candList, candCount, dev.d_rowStart, dev.d_deg, dev.d_colInd,
            dev.d_nodeW, st.d_dist, st.d_parent, st.d_isCandidate,
            st.d_isAffected, st.d_affList, d_affCount, source0);
        HSOSP_CUDA_CHECK(cudaGetLastError());

        int affCount = 0;
        HSOSP_CUDA_CHECK(cudaMemcpy(&affCount, d_affCount, sizeof(int),
                                    cudaMemcpyDeviceToHost));
        if (affCount == 0) {
            candCount = 0;
            break;
        }

        HSOSP_CUDA_CHECK(cudaMemset(d_candCount, 0, sizeof(int)));
        collectCandidatesNW<<<gridFor(affCount, block), block>>>(
            st.d_affList, affCount, dev.d_rowStart, dev.d_deg, dev.d_colInd,
            st.d_isCandidate, st.d_candList, d_candCount, st.d_isAffected,
            source0);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        HSOSP_CUDA_CHECK(cudaMemcpy(&candCount, d_candCount, sizeof(int),
                                    cudaMemcpyDeviceToHost));
    }

    if (maxFrontierOut) *maxFrontierOut = maxFrontier;
    return iterations;
}

void resetForRecompute(const DeviceH2H& dev, HsospState& st, int source0) {
    const int n = dev.numNodes;
    const int block = 256;
    fillLLKernel<<<gridFor(n, block), block>>>(st.d_dist, n, INF_VALUE);
    fillIntKernel<<<gridFor(n, block), block>>>(st.d_parent, n, -1);
    HSOSP_CUDA_CHECK(cudaMemset(st.d_isAffected, 0, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMemset(st.d_isCandidate, 0, sizeof(int) * n));
    const long long zero = 0;
    HSOSP_CUDA_CHECK(cudaMemcpy(st.d_dist + source0, &zero,
                                sizeof(long long), cudaMemcpyHostToDevice));
    HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

UpdateStats hsospRecompute(const DeviceH2H& dev, HsospState& st, int sourceId,
                           const UpdateConfig& cfg) {
    UpdateStats stats;
    const int source0 = sourceId - 1;
    resetForRecompute(dev, st, source0);

    UpdateConfig rcfg = cfg;
    // Recompute-from-blank converges in <= eccentricity(source) rounds; the
    // stale-loop hazard of the dynamic path cannot occur. Cap generously.
    rcfg.maxIterations = std::max(cfg.maxIterations, dev.numNodes + 1);

    std::vector<int> initial = {source0};
    int it = propagate(dev, st, initial, /*startWithCandidates=*/false,
                       source0, rcfg, &stats.maxFrontier);
    stats.iterations = (it < 0) ? rcfg.maxIterations : it;
    return stats;
}

UpdateStats hsospUpdate(const DeviceH2H& dev, HsospState& st,
                        const std::vector<int>& seedIds,
                        const std::vector<int>& deadIds, int sourceId,
                        const UpdateConfig& cfg) {
    UpdateStats stats;
    const int source0 = sourceId - 1;
    stats.seedCount = static_cast<long long>(seedIds.size());

    // Dead nodes: distance INF, no parent, degree already zeroed by the
    // delta (their edges are all in delEdges); force deg=0 anyway.
    if (!deadIds.empty()) {
        std::vector<int> idx;
        idx.reserve(deadIds.size());
        for (int id : deadIds) idx.push_back(id - 1);
        int* d_idx = nullptr;
        HSOSP_CUDA_CHECK(cudaMalloc(&d_idx, idx.size() * sizeof(int)));
        HSOSP_CUDA_CHECK(cudaMemcpy(d_idx, idx.data(),
                                    idx.size() * sizeof(int),
                                    cudaMemcpyHostToDevice));
        const int block = 256;
        markDeadKernel<<<gridFor(idx.size(), block), block>>>(
            d_idx, static_cast<int>(idx.size()), dev.d_deg, st.d_dist,
            st.d_parent);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_idx);
    }

    if (seedIds.empty()) return stats;

    std::vector<int> seeds0;
    seeds0.reserve(seedIds.size());
    for (int id : seedIds) seeds0.push_back(id - 1);

    int it = propagate(dev, st, seeds0, /*startWithCandidates=*/true, source0,
                       cfg, &stats.maxFrontier);
    if (it < 0) {
        // Stale-loop in a disconnected region: fall back to the (always
        // convergent) recompute, exactly like the host emulation.
        stats.fallbackRecompute = true;
        UpdateStats rs = hsospRecompute(dev, st, sourceId, cfg);
        stats.iterations = rs.iterations;
        stats.maxFrontier = std::max(stats.maxFrontier, rs.maxFrontier);
    } else {
        stats.iterations = it;
    }
    return stats;
}

long long compareDistances(const HsospState& a, const HsospState& b, int n) {
    unsigned long long* d_out = nullptr;
    HSOSP_CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    HSOSP_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    const int block = 256;
    countMismatchesKernel<<<gridFor(n, block), block>>>(a.d_dist, b.d_dist, n,
                                                        d_out);
    HSOSP_CUDA_CHECK(cudaGetLastError());
    unsigned long long out = 0;
    HSOSP_CUDA_CHECK(cudaMemcpy(&out, d_out, sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return static_cast<long long>(out);
}

} // namespace hsosp
} // namespace escher_mosp
