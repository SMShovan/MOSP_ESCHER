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
 * Pull step of the recompute, one warp per candidate v: the best parent
 * over v's (symmetric) neighbour list, read with coalesced strided loads
 * and a shuffle min-reduction; every in-edge of v costs nodeW[v], and ties
 * go to the lowest parent id (canonical tree). Candidates are unique
 * (dedup at enqueue), so the dist / parent writes are race-free; reading
 * the neighbours' distances while other warps lower them is the benign
 * chaotic-relaxation race (a changed node re-enqueues its neighbours).
 * (The original ran one thread per candidate over the whole row with
 * dependent loads; rows of the DBLP line graph have up to 3,000 entries.)
 */
__global__ void pullDistancesWarp(
    const int* candList, int numCand, const long long* rowStart,
    const int* deg, const int* colInd, const long long* nodeW,
    long long* dist, int* parent, int* isCandidate, int* isAffected,
    int* affList, int* affCount, int source) {

    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= numCand) return;

    const int v = candList[warp];
    if (lane == 0) isCandidate[v] = 0;
    if (v == source) return;

    long long best = INF_VALUE;
    int bestP = -1;
    const long long base = rowStart[v];
    const int d = deg[v];
    const long long wv = nodeW[v];
    for (int e = lane; e < d; e += 32) {
        const int p = colInd[base + e];
        const long long dp = dist[p];
        if (dp >= INF_VALUE / 2) continue;
        const long long cd = dp + wv;
        if (cd < best || (cd == best && p < bestP)) {
            best = cd;
            bestP = p;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const long long ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int op = __shfl_xor_sync(0xffffffffu, bestP, o);
        if (ob < best || (ob == best && op < bestP)) {
            best = ob;
            bestP = op;
        }
    }
    if (lane == 0) {
        const bool changed = (best != dist[v]);
        parent[v] = bestP;
        dist[v] = best;
        if (changed && atomicCAS(&isAffected[v], 0, 1) == 0)
            affList[atomicAdd(affCount, 1)] = v;
    }
}

/** Collect step, one warp per changed node: its neighbours become the next
 *  candidates. A plain read of isCandidate skips the atomicCAS for nodes
 *  that are already listed (the flag only goes 0 -> 1 in this kernel). */
__global__ void collectCandidatesWarp(const int* affList, int numAff,
                                      const long long* rowStart,
                                      const int* deg, const int* colInd,
                                      int* isCandidate, int* candList,
                                      int* candCount, int* isAffected,
                                      int source) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= numAff) return;

    const int u = affList[warp];
    if (lane == 0) isAffected[u] = 0;

    const long long base = rowStart[u];
    const int d = deg[u];
    for (int e = lane; e < d; e += 32) {
        const int nb = colInd[base + e];
        if (nb == source) continue;
        if (isCandidate[nb] == 0 && atomicCAS(&isCandidate[nb], 0, 1) == 0)
            candList[atomicAdd(candCount, 1)] = nb;
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

// ---------------------------------------------------------------------------
// Incremental update kernels (hsospUpdate)
// ---------------------------------------------------------------------------
//
// The update works on packed words (dist << 32 | parent): a 64-bit
// atomicMin then keeps the smaller distance and, among equal distances,
// the lower parent id. PACKED_INF marks an unreachable node; the source is
// (0 << 32 | 0xffffffff). Distances must stay below PACKED_DIST_LIMIT; a
// larger candidate raises the overflow counter and the update falls back
// to the 64-bit recompute.

using u64 = unsigned long long;
constexpr u64 PACKED_INF = ~0ull;
constexpr long long PACKED_DIST_LIMIT = 0xffffffffLL;

// Update counters (device, mirrored to pinned host memory).
enum UpdCounter {
    kCntList = 0,     // invalidated nodes
    kCntNext = 1,     // size of the next frontier
    kCntActive = 2,   // pointer jumping still active
    kCntWork = 3,     // edge relaxations
    kCntOverflow = 4, // a distance did not fit 32 bits
};

__device__ __forceinline__ u64 packDist(long long d, int parent) {
    return (static_cast<u64>(d) << 32) | static_cast<unsigned>(parent);
}
__device__ __forceinline__ long long packedDist(u64 x) {
    return static_cast<long long>(x >> 32);
}
__device__ __forceinline__ int packedParent(u64 x) {
    return static_cast<int>(static_cast<unsigned>(x & 0xffffffffu));
}

__global__ void packStateKernel(const long long* dist, const int* parent,
                                u64* packed, int n, u64* counters) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const long long d = dist[v];
    if (d >= INF_VALUE / 2) {
        packed[v] = PACKED_INF;
    } else {
        if (d >= PACKED_DIST_LIMIT) atomicAdd(&counters[kCntOverflow], 1ull);
        packed[v] = packDist(d, parent[v]);
    }
}

__global__ void unpackStateKernel(const u64* packed, long long* dist,
                                  int* parent, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const u64 x = packed[v];
    if (x == PACKED_INF) {
        dist[v] = INF_VALUE;
        parent[v] = -1;
    } else {
        dist[v] = packedDist(x);
        parent[v] = packedParent(x);
    }
}

// Roots: b of a deleted pair (a, b) whose tree parent was a (and vice
// versa), from the pre-batch packed state.
__global__ void markDeletedTreeEdgesKernel(const int2* pairs, int nDel,
                                           const u64* packed, int* inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nDel) return;
    const int a = pairs[i].x, b = pairs[i].y;
    const u64 pa = packed[a], pb = packed[b];
    if (pb != PACKED_INF && packedParent(pb) == a) inv[b] = 1;
    if (pa != PACKED_INF && packedParent(pa) == b) inv[a] = 1;
}

__global__ void markIdsKernel(const int* ids, int n, int* inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inv[ids[i]] = 1;
}

__global__ void zeroDegreeKernel(const int* ids, int n, int* deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) deg[ids[i]] = 0;
}

__global__ void initJumpKernel(const u64* packed, int* jump, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const u64 x = packed[v];
    const int p = (x == PACKED_INF) ? -1 : packedParent(x);
    jump[v] = (p == -1) ? -1 : p;
}

// One pointer-jumping round (double buffered): a node is invalidated if it
// or its current ancestor is; otherwise its ancestor pointer doubles. After
// ceil(log2(depth)) + 1 rounds every descendant of a root is marked.
__global__ void jumpKernel(const int* invIn, int* invOut, const int* jumpIn,
                           int* jumpOut, int n, u64* counters) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const int iv = invIn[v];
    const int j = jumpIn[v];
    if (iv || j < 0) {
        invOut[v] = iv;
        jumpOut[v] = -1;
        return;
    }
    const int ij = invIn[j];
    const int jj = jumpIn[j];
    invOut[v] = ij;
    jumpOut[v] = ij ? -1 : jj;
    if (!ij && jj >= 0) counters[kCntActive] = 1;
}

__global__ void invalidateKernel(const int* inv, u64* packed, int* list,
                                 u64* counters, int n, int source) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (inv[v] && v != source) {
        packed[v] = PACKED_INF;
        list[atomicAdd(&counters[kCntList], 1ull)] = v;
    }
}

// Enqueues v for the frontier of epoch ep unless it already is (epoch
// stamps instead of a reset flag, which raced with concurrent pushes).
__device__ __forceinline__ void enqueue(int v, int* stamp, int ep, int* next,
                                        u64* counters) {
    if (atomicMax(&stamp[v], ep) < ep)
        next[atomicAdd(&counters[kCntNext], 1ull)] = v;
}

// Relaxes v with the candidate word c; enqueues v when it improved.
__device__ __forceinline__ void relax(u64* packed, int v, u64 c, int* stamp,
                                      int ep, int* next, u64* counters) {
    if (c < packed[v]) {
        const u64 old = atomicMin(&packed[v], c);
        if (c < old) enqueue(v, stamp, ep, next, counters);
    }
}

// One warp per invalidated node: best (distance, id) over its neighbours.
__global__ void pullInvalidatedKernel(const int* list, int nList,
                                      const long long* rowStart,
                                      const int* deg, const int* colInd,
                                      const long long* nodeW, u64* packed,
                                      int* stamp, int ep, int* next,
                                      u64* counters) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nList) return;
    const int v = list[warp];
    const long long base = rowStart[v];
    const int d = deg[v];
    const long long wv = nodeW[v];
    u64 best = PACKED_INF;
    bool overflow = false;
    for (int e = lane; e < d; e += 32) {
        const int p = colInd[base + e];
        const u64 x = packed[p];
        if (x == PACKED_INF) continue;
        const long long cd = packedDist(x) + wv;
        if (cd >= PACKED_DIST_LIMIT) {
            overflow = true;
            continue;
        }
        const u64 c = packDist(cd, p);
        if (c < best) best = c;
    }
    for (int o = 16; o > 0; o >>= 1) {
        const u64 y = __shfl_xor_sync(0xffffffffu, best, o);
        if (y < best) best = y;
    }
    overflow = __any_sync(0xffffffffu, overflow);
    if (lane == 0) {
        atomicAdd(&counters[kCntWork], static_cast<u64>(d));
        if (overflow) atomicAdd(&counters[kCntOverflow], 1ull);
        if (best != PACKED_INF) relax(packed, v, best, stamp, ep, next, counters);
    }
}

// Both directions of every inserted pair.
__global__ void relaxInsertedKernel(const int2* pairs, int nIns,
                                    const long long* nodeW, u64* packed,
                                    int* stamp, int ep, int* next,
                                    u64* counters, int source) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nIns) return;
    const int ab[2] = {pairs[i].x, pairs[i].y};
    for (int k = 0; k < 2; ++k) {
        const int u = ab[k], v = ab[1 - k];
        if (v == source) continue;
        const u64 x = packed[u];
        if (x == PACKED_INF) continue;
        const long long cd = packedDist(x) + nodeW[v];
        if (cd >= PACKED_DIST_LIMIT) {
            atomicAdd(&counters[kCntOverflow], 1ull);
            continue;
        }
        relax(packed, v, packDist(cd, u), stamp, ep, next, counters);
    }
}

// One warp per frontier node: push its current word to the neighbours
// (decrease-only).
__global__ void pushKernel(const int* front, int nFront,
                           const long long* rowStart, const int* deg,
                           const int* colInd, const long long* nodeW,
                           u64* packed, int* stamp, int ep, int* next,
                           u64* counters, int source) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nFront) return;
    const int u = front[warp];
    const u64 x = packed[u];
    if (x == PACKED_INF) return;
    const long long du = packedDist(x);
    const long long base = rowStart[u];
    const int d = deg[u];
    bool overflow = false;
    for (int e = lane; e < d; e += 32) {
        const int v = colInd[base + e];
        if (v == source) continue;
        const long long cd = du + nodeW[v];
        if (cd >= PACKED_DIST_LIMIT) {
            overflow = true;
            continue;
        }
        relax(packed, v, packDist(cd, u), stamp, ep, next, counters);
    }
    overflow = __any_sync(0xffffffffu, overflow);
    if (lane == 0) {
        atomicAdd(&counters[kCntWork], static_cast<u64>(d));
        if (overflow) atomicAdd(&counters[kCntOverflow], 1ull);
    }
}

} // namespace

constexpr int kUpdCounters = 8;

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

void DeviceDelta::reserve(long long pairs, long long ids) {
    if (pairs > pairCapacity) {
        if (d_pairs) cudaFree(d_pairs);
        pairCapacity = pairs + pairs / 2 + 1024;
        HSOSP_CUDA_CHECK(cudaMalloc(&d_pairs, sizeof(int2) * pairCapacity));
    }
    if (ids > idCapacity) {
        if (d_ids) cudaFree(d_ids);
        idCapacity = ids + ids / 2 + 1024;
        HSOSP_CUDA_CHECK(cudaMalloc(&d_ids, sizeof(int) * idCapacity));
    }
}

void DeviceDelta::free() {
    if (d_pairs) cudaFree(d_pairs);
    if (d_ids) cudaFree(d_ids);
    d_pairs = nullptr;
    d_ids = nullptr;
    pairCapacity = idCapacity = 0;
    numDel = numIns = numNew = numDead = 0;
}

// free() releases the graph arrays only: buildDeviceH2H (also used to
// rebuild after a tail overflow) keeps the last delta for hsospUpdate.
DeviceH2H::~DeviceH2H() {
    free();
    delta.free();
}

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
        numEntries = o.numEntries;
        delta.free();
        delta = o.delta;
        o.delta = DeviceDelta{};
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
    // !(x >= 1) also rejects NaN. Below 1 the colInd capacity is smaller
    // than the row layout and the host fill below overran its vector.
    if (!(entryHeadroom >= 1.0)) {
        throw std::invalid_argument(
            "buildDeviceH2H: entryHeadroom must be >= 1");
    }
    dev.free();
    dev.maxNodes = maxNodes;
    dev.numNodes = m;

    std::vector<long long> rowStart(maxNodes, 0);
    std::vector<int> deg(maxNodes, 0);
    std::vector<int> cap(maxNodes, 0);
    std::vector<long long> nodeW(maxNodes, 0);

    long long cursor = 0;
    dev.numEntries = 0;
    for (int id = 1; id <= m; ++id) {
        const int d =
            hg.alive[id - 1] ? static_cast<int>(hg.h2h[id - 1].size()) : 0;
        dev.numEntries += d;
        const int c = rowCapacityFor(d);
        rowStart[id - 1] = cursor;
        deg[id - 1] = d;
        cap[id - 1] = c;
        nodeW[id - 1] = hg.heW[id - 1];
        cursor += c;
    }
    dev.usedEntries = cursor;
    dev.capEntries =
        std::max(cursor, static_cast<long long>(static_cast<double>(cursor) *
                                                entryHeadroom)) +
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
    dev.numEntries += 2LL * (static_cast<long long>(delta.insEdges.size()) -
                             static_cast<long long>(delta.delEdges.size()));

    // Keep the delta on the device for hsospUpdate (0-based indices).
    {
        DeviceDelta& dd = dev.delta;
        std::vector<int2> pairs;
        pairs.reserve(delta.delEdges.size() + delta.insEdges.size());
        for (auto [a, b] : delta.delEdges) pairs.push_back(make_int2(a - 1, b - 1));
        for (auto [a, b] : delta.insEdges) pairs.push_back(make_int2(a - 1, b - 1));
        std::vector<int> ids;
        ids.reserve(delta.newHe.size() + delta.deadHe.size());
        for (int id : delta.newHe) ids.push_back(id - 1);
        for (int id : delta.deadHe) ids.push_back(id - 1);
        dd.reserve(static_cast<long long>(pairs.size()),
                   static_cast<long long>(ids.size()));
        if (!pairs.empty())
            HSOSP_CUDA_CHECK(cudaMemcpy(dd.d_pairs, pairs.data(),
                                        sizeof(int2) * pairs.size(),
                                        cudaMemcpyHostToDevice));
        if (!ids.empty())
            HSOSP_CUDA_CHECK(cudaMemcpy(dd.d_ids, ids.data(),
                                        sizeof(int) * ids.size(),
                                        cudaMemcpyHostToDevice));
        dd.numDel = static_cast<int>(delta.delEdges.size());
        dd.numIns = static_cast<int>(delta.insEdges.size());
        dd.numNew = static_cast<int>(delta.newHe.size());
        dd.numDead = static_cast<int>(delta.deadHe.size());
    }

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

std::vector<std::vector<int>> downloadRows(const DeviceH2H& dev, int m) {
    std::vector<long long> rowStart(m);
    std::vector<int> deg(m);
    std::vector<int> colInd(static_cast<std::size_t>(dev.capEntries));
    if (m > 0) {
        HSOSP_CUDA_CHECK(cudaMemcpy(rowStart.data(), dev.d_rowStart,
                                    sizeof(long long) * m,
                                    cudaMemcpyDeviceToHost));
        HSOSP_CUDA_CHECK(cudaMemcpy(deg.data(), dev.d_deg, sizeof(int) * m,
                                    cudaMemcpyDeviceToHost));
        HSOSP_CUDA_CHECK(cudaMemcpy(colInd.data(), dev.d_colInd,
                                    sizeof(int) * dev.capEntries,
                                    cudaMemcpyDeviceToHost));
    }
    std::vector<std::vector<int>> rows(m);
    for (int i = 0; i < m; ++i) {
        rows[i].reserve(deg[i]);
        for (int e = 0; e < deg[i]; ++e)
            rows[i].push_back(colInd[rowStart[i] + e] + 1);
        std::sort(rows[i].begin(), rows[i].end());
    }
    return rows;
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
    if (d_packed) cudaFree(d_packed);
    if (d_invA) cudaFree(d_invA);
    if (d_invB) cudaFree(d_invB);
    if (d_jumpA) cudaFree(d_jumpA);
    if (d_jumpB) cudaFree(d_jumpB);
    if (d_front) cudaFree(d_front);
    if (d_next) cudaFree(d_next);
    if (d_stamp) cudaFree(d_stamp);
    if (d_updCounters) cudaFree(d_updCounters);
    if (h_updCounters) cudaFreeHost(h_updCounters);
    d_packed = nullptr;
    d_invA = d_invB = d_jumpA = d_jumpB = nullptr;
    d_front = d_next = d_stamp = nullptr;
    d_updCounters = nullptr;
    h_updCounters = nullptr;
    epoch = 0;
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
        d_packed = o.d_packed;
        d_invA = o.d_invA;
        d_invB = o.d_invB;
        d_jumpA = o.d_jumpA;
        d_jumpB = o.d_jumpB;
        d_front = o.d_front;
        d_next = o.d_next;
        d_stamp = o.d_stamp;
        d_updCounters = o.d_updCounters;
        h_updCounters = o.h_updCounters;
        epoch = o.epoch;
        o.d_packed = nullptr;
        o.d_invA = o.d_invB = o.d_jumpA = o.d_jumpB = nullptr;
        o.d_front = o.d_next = o.d_stamp = nullptr;
        o.d_updCounters = nullptr;
        o.h_updCounters = nullptr;
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
    HSOSP_CUDA_CHECK(cudaMalloc(&d_packed, sizeof(unsigned long long) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_invA, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_invB, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_jumpA, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_jumpB, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_front, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_next, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_stamp, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_updCounters,
                                sizeof(unsigned long long) * kUpdCounters));
    HSOSP_CUDA_CHECK(cudaMallocHost(&h_updCounters,
                                    sizeof(unsigned long long) * kUpdCounters));
    HSOSP_CUDA_CHECK(cudaMemset(d_stamp, 0, sizeof(int) * n));
    epoch = 0;

    const int block = 256;
    fillLLKernel<<<gridFor(n, block), block>>>(d_dist, n, INF_VALUE);
    fillIntKernel<<<gridFor(n, block), block>>>(d_parent, n, -1);
    HSOSP_CUDA_CHECK(cudaMemset(d_isAffected, 0, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaMemset(d_isCandidate, 0, sizeof(int) * n));
    HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
}

long long HsospState::deviceBytes() const {
    return static_cast<long long>(maxNodes) *
               (sizeof(long long) + 5 * sizeof(int) +
                sizeof(unsigned long long) + 8 * sizeof(int)) +
           2 * sizeof(int) + kUpdCounters * sizeof(unsigned long long);
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

void HsospState::downloadParentIds(std::vector<int>& parentIds,
                                   int n) const {
    downloadParents(parentIds, n);
    for (int& p : parentIds)
        if (p >= 0) p += 1;
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
        collectCandidatesWarp<<<gridFor(32LL * listCount, block), block>>>(
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
        pullDistancesWarp<<<gridFor(32LL * candCount, block), block>>>(
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
        collectCandidatesWarp<<<gridFor(32LL * affCount, block), block>>>(
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

UpdateStats hsospUpdate(const DeviceH2H& dev, HsospState& st, int sourceId,
                        const UpdateConfig& cfg) {
    UpdateStats stats;
    const int n = dev.numNodes;
    const int source0 = sourceId - 1;
    const int block = 256;
    const DeviceDelta& dd = dev.delta;
    const int2* delPairs = dd.d_pairs;
    const int2* insPairs = dd.d_pairs + dd.numDel;
    const int* newIds = dd.d_ids;
    const int* deadIds = dd.d_ids + dd.numNew;
    u64* cnt = st.d_updCounters;
    u64* hcnt = st.h_updCounters;
    auto readCounters = [&]() {
        HSOSP_CUDA_CHECK(cudaMemcpy(hcnt, cnt, sizeof(u64) * kUpdCounters,
                                    cudaMemcpyDeviceToHost));
    };

    // Dead nodes have no edges left (all their pairs are in the delta);
    // zero their degree anyway so they can never be pushed from.
    if (dd.numDead > 0)
        zeroDegreeKernel<<<gridFor(dd.numDead, block), block>>>(
            deadIds, dd.numDead, dev.d_deg);

    // 1. Pack the pre-batch state; mark the roots.
    HSOSP_CUDA_CHECK(cudaMemset(cnt, 0, sizeof(u64) * kUpdCounters));
    packStateKernel<<<gridFor(n, block), block>>>(st.d_dist, st.d_parent,
                                                  st.d_packed, n, cnt);
    HSOSP_CUDA_CHECK(cudaMemset(st.d_invA, 0, sizeof(int) * n));
    if (dd.numDel > 0)
        markDeletedTreeEdgesKernel<<<gridFor(dd.numDel, block), block>>>(
            delPairs, dd.numDel, st.d_packed, st.d_invA);
    if (dd.numNew + dd.numDead > 0)
        markIdsKernel<<<gridFor(dd.numNew + dd.numDead, block), block>>>(
            newIds, dd.numNew + dd.numDead, st.d_invA);
    HSOSP_CUDA_CHECK(cudaGetLastError());

    // 2. Invalidate the subtrees of the roots by pointer jumping. A parent
    //    cycle cannot occur with positive weights; the round cap only
    //    guards against a corrupt tree (then: fall back).
    initJumpKernel<<<gridFor(n, block), block>>>(st.d_packed, st.d_jumpA, n);
    int* invIn = st.d_invA;
    int* invOut = st.d_invB;
    int* jumpIn = st.d_jumpA;
    int* jumpOut = st.d_jumpB;
    bool treeOk = false;
    for (int round = 0; round < 64; ++round) {
        HSOSP_CUDA_CHECK(
            cudaMemset(cnt + kCntActive, 0, sizeof(u64)));
        jumpKernel<<<gridFor(n, block), block>>>(invIn, invOut, jumpIn,
                                                 jumpOut, n, cnt);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        std::swap(invIn, invOut);
        std::swap(jumpIn, jumpOut);
        ++stats.jumpRounds;
        readCounters();
        if (hcnt[kCntActive] == 0) {
            treeOk = true;
            break;
        }
    }
    const double workLimit =
        cfg.workBudget * static_cast<double>(std::max(1LL, dev.numEntries));
    bool budgetExceeded = !treeOk;

    if (!budgetExceeded) {
        invalidateKernel<<<gridFor(n, block), block>>>(
            invIn, st.d_packed, st.d_front, cnt, n, source0);
        readCounters();
        const int nInv = static_cast<int>(hcnt[kCntList]);
        stats.seedCount = nInv;

        // 3. Pull for the invalidated nodes, relax the inserted pairs; the
        //    improved nodes form the first frontier (epoch stamps).
        const int ep0 = ++st.epoch;
        HSOSP_CUDA_CHECK(cudaMemset(cnt + kCntNext, 0, sizeof(u64)));
        if (nInv > 0)
            pullInvalidatedKernel<<<gridFor(32LL * nInv, block), block>>>(
                st.d_front, nInv, dev.d_rowStart, dev.d_deg, dev.d_colInd,
                dev.d_nodeW, st.d_packed, st.d_stamp, ep0, st.d_next, cnt);
        if (dd.numIns > 0)
            relaxInsertedKernel<<<gridFor(dd.numIns, block), block>>>(
                insPairs, dd.numIns, dev.d_nodeW, st.d_packed, st.d_stamp,
                ep0, st.d_next, cnt, source0);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        readCounters();

        // 4. Push until no distance decreases (or the budget is spent).
        int frontier = static_cast<int>(hcnt[kCntNext]);
        stats.maxFrontier = frontier;
        while (frontier > 0) {
            if (stats.iterations >= cfg.maxIterations ||
                static_cast<double>(hcnt[kCntWork]) > workLimit ||
                hcnt[kCntOverflow] != 0) {
                budgetExceeded = true;
                break;
            }
            ++stats.iterations;
            std::swap(st.d_front, st.d_next);
            const int ep = ++st.epoch;
            HSOSP_CUDA_CHECK(cudaMemset(cnt + kCntNext, 0, sizeof(u64)));
            pushKernel<<<gridFor(32LL * frontier, block), block>>>(
                st.d_front, frontier, dev.d_rowStart, dev.d_deg,
                dev.d_colInd, dev.d_nodeW, st.d_packed, st.d_stamp, ep,
                st.d_next, cnt, source0);
            HSOSP_CUDA_CHECK(cudaGetLastError());
            readCounters();
            frontier = static_cast<int>(hcnt[kCntNext]);
            stats.maxFrontier = std::max(stats.maxFrontier, frontier);
        }
        stats.work = static_cast<long long>(hcnt[kCntWork]);
        if (hcnt[kCntOverflow] != 0) budgetExceeded = true;
    }

    if (budgetExceeded) {
        // Exact fallback: recompute from scratch (64-bit distances).
        stats.fallbackRecompute = true;
        UpdateStats rs = hsospRecompute(dev, st, sourceId, cfg);
        stats.fallbackIterations = rs.iterations;
        stats.maxFrontier = std::max(stats.maxFrontier, rs.maxFrontier);
    } else {
        unpackStateKernel<<<gridFor(n, block), block>>>(
            st.d_packed, st.d_dist, st.d_parent, n);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
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
