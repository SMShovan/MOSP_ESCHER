/**
 * @file hsospDelta.cu
 * @brief Unification on the GPU: the net line-graph delta of a batch,
 *        derived from the host's incidence changes, and the batch pipeline
 *        (applyBatch).
 *
 * The host ships the changed incidences (vertex, hyperedge, +/-1) and the
 * vertex lists of the touched hyperedges before and after the batch. On the
 * device:
 *   1. the incidence changes are sorted and reduced to a net sign per
 *      (vertex, hyperedge);
 *   2. candidates: for every changed incidence (v, h) and every o in v's
 *      hyperedge list before the batch (pass 0) and after it (pass 1), the
 *      pair (min(h,o), max(h,o)); the vertex -> hyperedge mirror is updated
 *      between the passes;
 *   3. the candidates are sorted and deduplicated, and each pair (a, b) is
 *      classified by whether the vertex lists of a and b overlap before and
 *      after the batch: inserted if only after, deleted if only before.
 * The classification looks only at the state before and after the batch,
 * so it does not depend on the order of the batch's ops, and it is exact:
 * a pair can only change if a vertex was added to or removed from one of
 * the two hyperedges while it belonged to the other, and that incidence
 * change lists the pair as a candidate. (The original ran this step
 * single-threaded on a host copy of the whole line graph, emulating every
 * op with +/-1 counts in a hash map.)
 */

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include "hsosp.cuh"
#include "hsospInternal.cuh"

namespace escher_mosp {
namespace hsosp {

using detail::gridFor;

namespace {

using u64 = unsigned long long;
constexpr u64 SENTINEL = ~0ull;
using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}

/** Grows a device buffer to hold at least @p n elements (contents lost). */
template <class T>
void grow(T*& ptr, long long& cap, long long n) {
    if (n <= cap) return;
    if (ptr) cudaFree(ptr);
    cap = n + n / 2 + 1024;
    HSOSP_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * cap));
}

template <class T>
void upload(T* dst, const std::vector<T>& src) {
    if (!src.empty())
        HSOSP_CUDA_CHECK(cudaMemcpy(dst, src.data(), sizeof(T) * src.size(),
                                    cudaMemcpyHostToDevice));
}

int bitsFor(long long n) {
    int b = 1;
    while ((1LL << b) < n) ++b;
    return b;
}

struct HighWord {
    __host__ __device__ int operator()(u64 k) const {
        return static_cast<int>(k >> 32);
    }
};
struct IsDeleted {
    __host__ __device__ bool operator()(signed char c) const { return c < 0; }
};
struct IsInserted {
    __host__ __device__ bool operator()(signed char c) const { return c > 0; }
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void setIndexKernel(const int* nodes, int n, int* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) idx[nodes[i]] = i;
}

__global__ void resetIndexKernel(const int* nodes, int n, int* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) idx[nodes[i]] = -1;
}

__global__ void scatterIntKernel(const int* idx, const int* vals, int n,
                                 int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[idx[i]] = vals[i];
}

// Candidates of each changed incidence: pass 0 counts the vertex's list
// before the batch (the current mirror), pass 1 after it (postLen).
__global__ void countCandidatesKernel(const u64* inc, const int* net, int n,
                                      const int* vLen, const int* postLen,
                                      long long* cnt0, long long* cnt1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = static_cast<int>(inc[i] >> 32);
    const bool changed = net[i] != 0;
    cnt0[i] = changed ? vLen[v] : 0;
    cnt1[i] = changed ? postLen[v] : 0;
}

__global__ void totalsKernel(const long long* off0, const long long* cnt0,
                             const long long* off1, const long long* cnt1,
                             int n, long long* totals) {
    if (n == 0) {
        totals[0] = totals[1] = 0;
        return;
    }
    totals[0] = off0[n - 1] + cnt0[n - 1];
    totals[1] = off1[n - 1] + cnt1[n - 1];
}

// One warp per changed incidence (v, h): the pairs (h, o) for o in v's
// current hyperedge list.
__global__ void emitCandidatesKernel(const u64* inc, const int* net, int n,
                                     const long long* vOff, const int* vLen,
                                     const int* vVal, const long long* off,
                                     long long base, u64* cand) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n || net[warp] == 0) return;
    const int v = static_cast<int>(inc[warp] >> 32);
    const unsigned h = static_cast<unsigned>(inc[warp] & 0xffffffffu);
    const long long rowStart = vOff[v];
    const int len = vLen[v];
    u64* out = cand + base + off[warp];
    for (int k = lane; k < len; k += 32) {
        const unsigned o = static_cast<unsigned>(vVal[rowStart + k]);
        const unsigned a = min(h, o), b = max(h, o);
        out[k] = (o == h) ? SENTINEL : ((static_cast<u64>(a) << 32) | b);
    }
}

// Moves a vertex row to a new (larger) slot before it grows.
__global__ void relocateRowsKernel(const int* verts, const long long* newOff,
                                   int n, long long* vOff, const int* vLen,
                                   int* vVal) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n) return;
    const int v = verts[warp];
    const long long from = vOff[v], to = newOff[warp];
    const int len = vLen[v];
    for (int k = lane; k < len; k += 32) vVal[to + k] = vVal[from + k];
    __syncwarp();
    if (lane == 0) vOff[v] = to;
}

// One warp per vertex segment of the net incidence changes (sorted by
// vertex, then hyperedge): removals by ballot search + swap-remove, then
// appends. Capacities were checked on the host.
__global__ void applyVertexRowsKernel(const int* segOff, const int* segLen,
                                      int nSeg, const u64* inc,
                                      const int* net, const long long* vOff,
                                      int* vLen, int* vVal) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nSeg) return;
    const int s = segOff[warp];
    const int len = segLen[warp];
    const int v = static_cast<int>(inc[s] >> 32);
    const long long base = vOff[v];
    int d = vLen[v];
    for (int k = 0; k < len; ++k) {
        if (net[s + k] >= 0) continue;
        const int h = static_cast<int>(inc[s + k] & 0xffffffffu);
        int pos = -1;
        for (int e0 = 0; e0 < d && pos < 0; e0 += 32) {
            const int e = e0 + lane;
            const unsigned m =
                __ballot_sync(0xffffffffu, e < d && vVal[base + e] == h);
            if (m) pos = e0 + __ffs(m) - 1;
        }
        if (pos >= 0) {
            if (lane == 0) vVal[base + pos] = vVal[base + d - 1];
            --d;
        }
        __syncwarp();
    }
    int appended = 0;
    for (int k = 0; k < len; ++k) {
        if (net[s + k] <= 0) continue;
        if (lane == 0)
            vVal[base + d + appended] =
                static_cast<int>(inc[s + k] & 0xffffffffu);
        ++appended;
    }
    __syncwarp();
    if (lane == 0) vLen[v] = d + appended;
}

__device__ bool sortedOverlap(const int* a, int la, const int* b, int lb) {
    int i = 0, j = 0;
    while (i < la && j < lb) {
        if (a[i] == b[j]) return true;
        if (a[i] < b[j]) ++i;
        else ++j;
    }
    return false;
}

// Vertex list of node x before (pre) or after the batch: from the shipped
// rows if x was touched, otherwise from the mirror (unchanged).
__device__ void rowOf(int x, bool pre, const int* touchedIdx,
                      const int* preOff, const int* preVals,
                      const int* postOff, const int* postVals,
                      const long long* heOff, const int* heLen,
                      const int* heVal, const int*& row, int& len) {
    const int t = touchedIdx[x];
    if (t >= 0) {
        const int* off = pre ? preOff : postOff;
        row = (pre ? preVals : postVals) + off[t];
        len = off[t + 1] - off[t];
    } else {
        row = heVal + heOff[x];
        len = heLen[x];
    }
}

__global__ void classifyPairsKernel(const u64* pairs, long long n,
                                    const int* touchedIdx, const int* preOff,
                                    const int* preVals, const int* postOff,
                                    const int* postVals,
                                    const long long* heOff, const int* heLen,
                                    const int* heVal, signed char* cls) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= n) return;
    const u64 k = pairs[i];
    if (k == SENTINEL) {
        cls[i] = 0;
        return;
    }
    const int a = static_cast<int>(k >> 32);
    const int b = static_cast<int>(k & 0xffffffffu);
    const int *ra, *rb;
    int la, lb;
    rowOf(a, true, touchedIdx, preOff, preVals, postOff, postVals, heOff,
          heLen, heVal, ra, la);
    rowOf(b, true, touchedIdx, preOff, preVals, postOff, postVals, heOff,
          heLen, heVal, rb, lb);
    const bool pre = sortedOverlap(ra, la, rb, lb);
    rowOf(a, false, touchedIdx, preOff, preVals, postOff, postVals, heOff,
          heLen, heVal, ra, la);
    rowOf(b, false, touchedIdx, preOff, preVals, postOff, postVals, heOff,
          heLen, heVal, rb, lb);
    const bool post = sortedOverlap(ra, la, rb, lb);
    cls[i] = static_cast<signed char>(post ? (pre ? 0 : 1) : (pre ? -1 : 0));
}

__global__ void keysToPairsKernel(const u64* keys, long long n, int2* out) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= n) return;
    out[i] = make_int2(static_cast<int>(keys[i] >> 32),
                       static_cast<int>(keys[i] & 0xffffffffu));
}

// Writes the post-batch vertex lists of the touched hyperedges into the
// mirror (slots chosen by the host).
__global__ void writeHeRowsKernel(const int* touched, const long long* off,
                                  int nT, const int* postOff,
                                  const int* postVals, long long* heOff,
                                  int* heLen, int* heVal) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nT) return;
    const int node = touched[warp];
    const int b = postOff[warp];
    const int len = postOff[warp + 1] - b;
    const long long o = off[warp];
    for (int k = lane; k < len; k += 32) heVal[o + k] = postVals[b + k];
    if (lane == 0) {
        heOff[node] = o;
        heLen[node] = len;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// DeviceIncidence
// ---------------------------------------------------------------------------

struct DeviceIncidence::Scratch {
    u64* incKeys = nullptr;
    u64* incKeysAlt = nullptr;
    int* signs = nullptr;
    int* signsAlt = nullptr;
    u64* uniq = nullptr;
    int* net = nullptr;
    long long* cnt0 = nullptr;
    long long* cnt1 = nullptr;
    long long* off0 = nullptr;
    long long* off1 = nullptr;
    int* segLen = nullptr;
    int* segOff = nullptr;
    int* segVert = nullptr;
    long long incCap = 0, incCap2 = 0, incCap3 = 0, incCap4 = 0, incCap5 = 0,
              incCap6 = 0, incCap7 = 0, incCap8 = 0, incCap9 = 0,
              incCap10 = 0, incCap11 = 0, incCap12 = 0, incCap13 = 0;
    u64* cand = nullptr;
    u64* candAlt = nullptr;
    signed char* cls = nullptr;
    u64* selected = nullptr;
    long long candCap = 0, candCap2 = 0, clsCap = 0, selCap = 0;
    int* touched = nullptr;
    int* preOff = nullptr;
    int* postOff = nullptr;
    long long* heWriteOff = nullptr;
    long long tCap = 0, tCap2 = 0, tCap3 = 0, tCap4 = 0;
    int* preVals = nullptr;
    int* postVals = nullptr;
    long long valCap = 0, valCap2 = 0;
    int* tVerts = nullptr;
    int* tVertLen = nullptr;
    int* relocVerts = nullptr;
    long long* relocOff = nullptr;
    long long tvCap = 0, tvCap2 = 0, rCap = 0, rCap2 = 0;
    long long* totals = nullptr;
    int* counts = nullptr;             // [0] uniq, [1] segs, [2] cand, [3] sel
    long long* hostTotals = nullptr;   // pinned
    int* hostCounts = nullptr;         // pinned
    void* temp = nullptr;
    std::size_t tempCap = 0;

    Scratch() {
        HSOSP_CUDA_CHECK(cudaMalloc(&totals, sizeof(long long) * 2));
        HSOSP_CUDA_CHECK(cudaMalloc(&counts, sizeof(int) * 4));
        HSOSP_CUDA_CHECK(cudaMallocHost(&hostTotals, sizeof(long long) * 2));
        HSOSP_CUDA_CHECK(cudaMallocHost(&hostCounts, sizeof(int) * 4));
    }
    ~Scratch() {
        for (void* p :
             {static_cast<void*>(incKeys), static_cast<void*>(incKeysAlt),
              static_cast<void*>(signs), static_cast<void*>(signsAlt),
              static_cast<void*>(uniq), static_cast<void*>(net),
              static_cast<void*>(cnt0), static_cast<void*>(cnt1),
              static_cast<void*>(off0), static_cast<void*>(off1),
              static_cast<void*>(segLen), static_cast<void*>(segOff),
              static_cast<void*>(segVert), static_cast<void*>(cand),
              static_cast<void*>(candAlt), static_cast<void*>(cls),
              static_cast<void*>(selected), static_cast<void*>(touched),
              static_cast<void*>(preOff), static_cast<void*>(postOff),
              static_cast<void*>(heWriteOff), static_cast<void*>(preVals),
              static_cast<void*>(postVals), static_cast<void*>(tVerts),
              static_cast<void*>(tVertLen), static_cast<void*>(relocVerts),
              static_cast<void*>(relocOff), static_cast<void*>(totals),
              static_cast<void*>(counts), temp})
            if (p) cudaFree(p);
        if (hostTotals) cudaFreeHost(hostTotals);
        if (hostCounts) cudaFreeHost(hostCounts);
    }

    void reserveTemp(std::size_t bytes) {
        if (bytes <= tempCap) return;
        if (temp) cudaFree(temp);
        tempCap = bytes + bytes / 2 + 4096;
        HSOSP_CUDA_CHECK(cudaMalloc(&temp, tempCap));
    }
};

DeviceIncidence::~DeviceIncidence() { free(); }

DeviceIncidence::DeviceIncidence(DeviceIncidence&& o) noexcept {
    *this = std::move(o);
}

DeviceIncidence& DeviceIncidence::operator=(DeviceIncidence&& o) noexcept {
    if (this != &o) {
        free();
        maxNodes = o.maxNodes;
        numVertices = o.numVertices;
        d_heOff = o.d_heOff;
        d_heLen = o.d_heLen;
        d_heVal = o.d_heVal;
        heCapacity = o.heCapacity;
        heTail = o.heTail;
        heOff = std::move(o.heOff);
        heCap = std::move(o.heCap);
        d_vOff = o.d_vOff;
        d_vLen = o.d_vLen;
        d_vVal = o.d_vVal;
        vCapacity = o.vCapacity;
        vTail = o.vTail;
        vOff = std::move(o.vOff);
        vCap = std::move(o.vCap);
        d_touchedIdx = o.d_touchedIdx;
        d_postLen = o.d_postLen;
        scratch = o.scratch;
        o.d_heOff = nullptr;
        o.d_heLen = nullptr;
        o.d_heVal = nullptr;
        o.d_vOff = nullptr;
        o.d_vLen = nullptr;
        o.d_vVal = nullptr;
        o.d_touchedIdx = nullptr;
        o.d_postLen = nullptr;
        o.scratch = nullptr;
    }
    return *this;
}

void DeviceIncidence::free() {
    for (void* p : {static_cast<void*>(d_heOff), static_cast<void*>(d_heLen),
                    static_cast<void*>(d_heVal), static_cast<void*>(d_vOff),
                    static_cast<void*>(d_vLen), static_cast<void*>(d_vVal),
                    static_cast<void*>(d_touchedIdx),
                    static_cast<void*>(d_postLen)})
        if (p) cudaFree(p);
    delete scratch;
    d_heOff = nullptr;
    d_heLen = nullptr;
    d_heVal = nullptr;
    d_vOff = nullptr;
    d_vLen = nullptr;
    d_vVal = nullptr;
    d_touchedIdx = nullptr;
    d_postLen = nullptr;
    scratch = nullptr;
    heCapacity = vCapacity = heTail = vTail = 0;
    heOff.clear();
    heCap.clear();
    vOff.clear();
    vCap.clear();
}

long long DeviceIncidence::deviceBytes() const {
    return static_cast<long long>(maxNodes) *
               (sizeof(long long) + 2 * sizeof(int)) +
           heCapacity * sizeof(int) +
           static_cast<long long>(numVertices) *
               (sizeof(long long) + 2 * sizeof(int)) +
           vCapacity * sizeof(int);
}

namespace {

// Slack of a row: hyperedges gain vertices in vertex batches, vertices
// gain hyperedges with every insertion.
int heSlotFor(int len) { return len + std::max(2, len / 2); }
int vSlotFor(int len) { return len + std::max(4, len / 2); }

// (Re)uploads the hyperedge rows from the host, with free tail space.
void uploadHeRows(DeviceIncidence& di, const HostHypergraph& hg,
                  double growth) {
    const int m = hg.maxId();
    di.heOff.assign(di.maxNodes, 0);
    di.heCap.assign(di.maxNodes, 0);
    std::vector<int> len(di.maxNodes, 0);
    long long cursor = 0;
    for (int id = 1; id <= m; ++id) {
        const int l = static_cast<int>(hg.heVerts[id - 1].size());
        len[id - 1] = l;
        di.heOff[id - 1] = cursor;
        di.heCap[id - 1] = heSlotFor(l);
        cursor += di.heCap[id - 1];
    }
    di.heTail = cursor;
    const long long capacity =
        std::max(cursor, static_cast<long long>(cursor * growth)) + 64;
    std::vector<int> val(static_cast<std::size_t>(cursor), 0);
    for (int id = 1; id <= m; ++id)
        std::copy(hg.heVerts[id - 1].begin(), hg.heVerts[id - 1].end(),
                  val.begin() + di.heOff[id - 1]);
    if (di.d_heVal) cudaFree(di.d_heVal);
    HSOSP_CUDA_CHECK(cudaMalloc(&di.d_heVal, sizeof(int) * capacity));
    di.heCapacity = capacity;
    upload(di.d_heVal, val);
    // The free tail is zeroed too (rows move there with slack).
    HSOSP_CUDA_CHECK(cudaMemset(di.d_heVal + cursor, 0,
                                sizeof(int) * (capacity - cursor)));
    upload(di.d_heOff, di.heOff);
    upload(di.d_heLen, len);
}

// (Re)uploads the vertex rows (0-based node indices) from the host.
void uploadVertexRows(DeviceIncidence& di, const HostHypergraph& hg,
                      double growth) {
    const int n = di.numVertices;
    di.vOff.assign(n, 0);
    di.vCap.assign(n, 0);
    std::vector<int> len(n, 0);
    long long cursor = 0;
    for (int v = 0; v < n; ++v) {
        const int l = static_cast<int>(hg.v2h[v].size());
        len[v] = l;
        di.vOff[v] = cursor;
        di.vCap[v] = vSlotFor(l);
        cursor += di.vCap[v];
    }
    di.vTail = cursor;
    const long long capacity =
        std::max(cursor, static_cast<long long>(cursor * growth)) + 64;
    std::vector<int> val(static_cast<std::size_t>(cursor), 0);
    for (int v = 0; v < n; ++v)
        for (std::size_t k = 0; k < hg.v2h[v].size(); ++k)
            val[di.vOff[v] + k] = hg.v2h[v][k] - 1;
    if (di.d_vVal) cudaFree(di.d_vVal);
    HSOSP_CUDA_CHECK(cudaMalloc(&di.d_vVal, sizeof(int) * capacity));
    di.vCapacity = capacity;
    upload(di.d_vVal, val);
    // The free tail is zeroed too (rows move there with slack).
    HSOSP_CUDA_CHECK(cudaMemset(di.d_vVal + cursor, 0,
                                sizeof(int) * (capacity - cursor)));
    upload(di.d_vOff, di.vOff);
    upload(di.d_vLen, len);
}

} // namespace

void DeviceIncidence::build(const HostHypergraph& hg, int nodes,
                            double headroom) {
    free();
    maxNodes = nodes;
    numVertices = hg.numVertices;
    HSOSP_CUDA_CHECK(cudaMalloc(&d_heOff, sizeof(long long) * maxNodes));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_heLen, sizeof(int) * maxNodes));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_touchedIdx, sizeof(int) * maxNodes));
    HSOSP_CUDA_CHECK(cudaMemset(d_touchedIdx, 0xff, sizeof(int) * maxNodes));
    HSOSP_CUDA_CHECK(
        cudaMalloc(&d_vOff, sizeof(long long) * std::max(1, numVertices)));
    HSOSP_CUDA_CHECK(cudaMalloc(&d_vLen, sizeof(int) * std::max(1, numVertices)));
    HSOSP_CUDA_CHECK(
        cudaMalloc(&d_postLen, sizeof(int) * std::max(1, numVertices)));
    uploadHeRows(*this, hg, headroom);
    uploadVertexRows(*this, hg, headroom);
    scratch = new Scratch();
}

long long incidenceMirrorMismatches(const DeviceIncidence& di,
                                    const HostHypergraph& hg) {
    auto download = [](auto* src, long long n, auto& dst) {
        dst.resize(static_cast<std::size_t>(n));
        if (n > 0)
            HSOSP_CUDA_CHECK(cudaMemcpy(dst.data(), src,
                                        sizeof(dst[0]) * n,
                                        cudaMemcpyDeviceToHost));
    };
    std::vector<long long> off;
    std::vector<int> len, val;
    long long bad = 0;
    // Hyperedge rows (dead hyperedges: empty).
    const int m = hg.maxId();
    download(di.d_heOff, m, off);
    download(di.d_heLen, m, len);
    download(di.d_heVal, di.heTail, val);   // rows lie below the tail
    for (int id = 1; id <= m; ++id) {
        std::vector<int> got(val.begin() + off[id - 1],
                             val.begin() + off[id - 1] + len[id - 1]);
        std::vector<int> want =
            hg.alive[id - 1] ? hg.heVerts[id - 1] : std::vector<int>{};
        std::sort(got.begin(), got.end());
        if (got != want) ++bad;
    }
    // Vertex rows (0-based node indices).
    download(di.d_vOff, di.numVertices, off);
    download(di.d_vLen, di.numVertices, len);
    download(di.d_vVal, di.vTail, val);
    for (int v = 0; v < di.numVertices; ++v) {
        std::vector<int> got(val.begin() + off[v],
                             val.begin() + off[v] + len[v]);
        std::vector<int> want;
        for (int id : hg.v2h[v]) want.push_back(id - 1);
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        if (got != want) ++bad;
    }
    return bad;
}

// ---------------------------------------------------------------------------
// Delta derivation
// ---------------------------------------------------------------------------

namespace {

/** Derives the batch's net line-graph delta into dev.delta; returns the
 *  number of candidate pairs examined and counts mirror re-uploads. */
long long deriveDelta(DeviceH2H& dev, const HostHypergraph& hg,
                      const IncidenceBatch& inc, int& reuploads) {
    DeviceIncidence& di = dev.inc;
    DeviceIncidence::Scratch& sc = *di.scratch;
    DeviceDelta& dd = dev.delta;
    const int block = 256;
    const int nInc = static_cast<int>(inc.incKey.size());
    const int nT = static_cast<int>(inc.touched.size());
    const int nTV = static_cast<int>(inc.touchedVertices.size());

    // ---- upload the batch (0-based node indices) ------------------------
    std::vector<u64> keys(nInc);
    for (int i = 0; i < nInc; ++i) keys[i] = inc.incKey[i] - 1;   // h -> h-1
    std::vector<int> touched0(nT);
    for (int t = 0; t < nT; ++t) touched0[t] = inc.touched[t] - 1;
    grow(sc.incKeys, sc.incCap, nInc);
    grow(sc.incKeysAlt, sc.incCap2, nInc);
    grow(sc.signs, sc.incCap3, nInc);
    grow(sc.signsAlt, sc.incCap4, nInc);
    grow(sc.uniq, sc.incCap5, nInc);
    grow(sc.net, sc.incCap6, nInc);
    grow(sc.cnt0, sc.incCap7, nInc);
    grow(sc.cnt1, sc.incCap8, nInc);
    grow(sc.off0, sc.incCap9, nInc);
    grow(sc.off1, sc.incCap10, nInc);
    grow(sc.segLen, sc.incCap11, nInc);
    grow(sc.segOff, sc.incCap12, nInc);
    grow(sc.segVert, sc.incCap13, nInc);
    grow(sc.touched, sc.tCap, nT);
    grow(sc.preOff, sc.tCap2, nT + 1);
    grow(sc.postOff, sc.tCap3, nT + 1);
    grow(sc.heWriteOff, sc.tCap4, nT);
    grow(sc.preVals, sc.valCap, static_cast<long long>(inc.preVals.size()));
    grow(sc.postVals, sc.valCap2, static_cast<long long>(inc.postVals.size()));
    grow(sc.tVerts, sc.tvCap, nTV);
    grow(sc.tVertLen, sc.tvCap2, nTV);
    upload(sc.incKeys, keys);
    upload(sc.signs, inc.incSign);
    upload(sc.touched, touched0);
    upload(sc.preOff, inc.preOff);
    upload(sc.postOff, inc.postOff);
    upload(sc.preVals, inc.preVals);
    upload(sc.postVals, inc.postVals);
    upload(sc.tVerts, inc.touchedVertices);
    upload(sc.tVertLen, inc.touchedVertexLen);

    // ---- 1. net sign per (vertex, hyperedge) -------------------------------
    int nUniq = 0;
    if (nInc > 0) {
        const int endBit = 32 + bitsFor(std::max(2, di.numVertices));
        cub::DoubleBuffer<u64> k(sc.incKeys, sc.incKeysAlt);
        cub::DoubleBuffer<int> s(sc.signs, sc.signsAlt);
        std::size_t b1 = 0, b2 = 0;
        HSOSP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, b1, k, s,
                                                         nInc, 0, endBit));
        HSOSP_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
            nullptr, b2, sc.incKeys, sc.uniq, sc.signs, sc.net, sc.counts,
            ::cuda::std::plus<int>(), nInc));
        sc.reserveTemp(std::max(b1, b2));
        HSOSP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            sc.temp, sc.tempCap, k, s, nInc, 0, endBit));
        HSOSP_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
            sc.temp, sc.tempCap, k.Current(), sc.uniq, s.Current(), sc.net,
            sc.counts, ::cuda::std::plus<int>(), nInc));
        HSOSP_CUDA_CHECK(cudaMemcpy(sc.hostCounts, sc.counts, sizeof(int),
                                    cudaMemcpyDeviceToHost));
        nUniq = sc.hostCounts[0];
    }

    // ---- 2. candidate counts (lists before / after the batch) --------------
    long long total0 = 0, total1 = 0;
    if (nUniq > 0) {
        if (nTV > 0)
            scatterIntKernel<<<gridFor(nTV, block), block>>>(
                sc.tVerts, sc.tVertLen, nTV, di.d_postLen);
        countCandidatesKernel<<<gridFor(nUniq, block), block>>>(
            sc.uniq, sc.net, nUniq, di.d_vLen, di.d_postLen, sc.cnt0,
            sc.cnt1);
        std::size_t b = 0;
        HSOSP_CUDA_CHECK(
            cub::DeviceScan::ExclusiveSum(nullptr, b, sc.cnt0, sc.off0, nUniq));
        sc.reserveTemp(b);
        HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(sc.temp, sc.tempCap,
                                                       sc.cnt0, sc.off0, nUniq));
        HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(sc.temp, sc.tempCap,
                                                       sc.cnt1, sc.off1, nUniq));
        totalsKernel<<<1, 1>>>(sc.off0, sc.cnt0, sc.off1, sc.cnt1, nUniq,
                               sc.totals);
        HSOSP_CUDA_CHECK(cudaMemcpy(sc.hostTotals, sc.totals,
                                    sizeof(long long) * 2,
                                    cudaMemcpyDeviceToHost));
        total0 = sc.hostTotals[0];
        total1 = sc.hostTotals[1];
    }
    const long long nCand = total0 + total1;
    grow(sc.cand, sc.candCap, nCand);
    grow(sc.candAlt, sc.candCap2, nCand);

    // ---- 3. pass 0, update the vertex -> hyperedge mirror, pass 1 ----------
    if (nUniq > 0) {
        emitCandidatesKernel<<<gridFor(32LL * nUniq, block), block>>>(
            sc.uniq, sc.net, nUniq, di.d_vOff, di.d_vLen, di.d_vVal, sc.off0,
            0, sc.cand);
        HSOSP_CUDA_CHECK(cudaGetLastError());

        // Rows that outgrow their slot move to the tail (host layout); a
        // full tail re-uploads the mirror from the host (post-batch state).
        std::vector<int> relocV;
        std::vector<long long> relocOff;
        bool reupload = false;
        for (int i = 0; i < nTV; ++i) {
            const int v = inc.touchedVertices[i];
            const int l = inc.touchedVertexLen[i];
            if (l <= di.vCap[v]) continue;
            const int c = vSlotFor(l);
            if (di.vTail + c > di.vCapacity) {
                reupload = true;
                break;
            }
            di.vOff[v] = di.vTail;
            di.vCap[v] = c;
            di.vTail += c;
            relocV.push_back(v);
            relocOff.push_back(di.vOff[v]);
        }
        if (reupload) {
            uploadVertexRows(di, hg, 1.5);
            ++reuploads;
        } else {
            const int nR = static_cast<int>(relocV.size());
            if (nR > 0) {
                grow(sc.relocVerts, sc.rCap, nR);
                grow(sc.relocOff, sc.rCap2, nR);
                upload(sc.relocVerts, relocV);
                upload(sc.relocOff, relocOff);
                relocateRowsKernel<<<gridFor(32LL * nR, block), block>>>(
                    sc.relocVerts, sc.relocOff, nR, di.d_vOff, di.d_vLen,
                    di.d_vVal);
            }
            // Vertex segments of the sorted net changes.
            auto verts =
                thrust::make_transform_iterator(sc.uniq, HighWord());
            std::size_t b = 0;
            HSOSP_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
                nullptr, b, verts, sc.segVert, sc.segLen, sc.counts + 1,
                nUniq));
            sc.reserveTemp(b);
            HSOSP_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
                sc.temp, sc.tempCap, verts, sc.segVert, sc.segLen,
                sc.counts + 1, nUniq));
            HSOSP_CUDA_CHECK(cudaMemcpy(sc.hostCounts + 1, sc.counts + 1,
                                        sizeof(int), cudaMemcpyDeviceToHost));
            const int nSeg = sc.hostCounts[1];
            b = 0;
            HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                nullptr, b, sc.segLen, sc.segOff, nSeg));
            sc.reserveTemp(b);
            HSOSP_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                sc.temp, sc.tempCap, sc.segLen, sc.segOff, nSeg));
            applyVertexRowsKernel<<<gridFor(32LL * nSeg, block), block>>>(
                sc.segOff, sc.segLen, nSeg, sc.uniq, sc.net, di.d_vOff,
                di.d_vLen, di.d_vVal);
        }
        emitCandidatesKernel<<<gridFor(32LL * nUniq, block), block>>>(
            sc.uniq, sc.net, nUniq, di.d_vOff, di.d_vLen, di.d_vVal, sc.off1,
            total0, sc.cand);
        HSOSP_CUDA_CHECK(cudaGetLastError());
    }

    // ---- 4. distinct candidates, classified by pre / post overlap ----------
    int numDel = 0, numIns = 0;
    if (nCand > 0) {
        if (nCand > std::numeric_limits<int>::max())
            throw std::runtime_error("applyBatch: too many candidate pairs");
        const int n = static_cast<int>(nCand);
        cub::DoubleBuffer<u64> k(sc.cand, sc.candAlt);
        std::size_t b1 = 0, b2 = 0, b3 = 0;
        HSOSP_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, b1, k, n));
        HSOSP_CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, b2, sc.cand,
                                                   sc.candAlt, sc.counts + 2,
                                                   n));
        grow(sc.cls, sc.clsCap, nCand);
        grow(sc.selected, sc.selCap, nCand);
        HSOSP_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr, b3, sc.cand,
            thrust::make_transform_iterator(sc.cls, IsDeleted()),
            sc.selected, sc.counts + 3, n));
        sc.reserveTemp(std::max({b1, b2, b3}));
        HSOSP_CUDA_CHECK(
            cub::DeviceRadixSort::SortKeys(sc.temp, sc.tempCap, k, n));
        u64* sorted = k.Current();
        u64* uniq = k.Alternate();
        HSOSP_CUDA_CHECK(cub::DeviceSelect::Unique(
            sc.temp, sc.tempCap, sorted, uniq, sc.counts + 2, n));
        HSOSP_CUDA_CHECK(cudaMemcpy(sc.hostCounts + 2, sc.counts + 2,
                                    sizeof(int), cudaMemcpyDeviceToHost));
        const int nPairs = sc.hostCounts[2];

        if (nT > 0)
            setIndexKernel<<<gridFor(nT, block), block>>>(sc.touched, nT,
                                                          di.d_touchedIdx);
        classifyPairsKernel<<<gridFor(nPairs, block), block>>>(
            uniq, nPairs, di.d_touchedIdx, sc.preOff, sc.preVals, sc.postOff,
            sc.postVals, di.d_heOff, di.d_heLen, di.d_heVal, sc.cls);
        HSOSP_CUDA_CHECK(cudaGetLastError());
        if (nT > 0)
            resetIndexKernel<<<gridFor(nT, block), block>>>(sc.touched, nT,
                                                            di.d_touchedIdx);

        // Deleted pairs first, then inserted ones.
        dd.reserve(nPairs, 0);
        HSOSP_CUDA_CHECK(cub::DeviceSelect::Flagged(
            sc.temp, sc.tempCap, uniq,
            thrust::make_transform_iterator(sc.cls, IsDeleted()), sc.selected,
            sc.counts + 3, nPairs));
        HSOSP_CUDA_CHECK(cudaMemcpy(sc.hostCounts + 3, sc.counts + 3,
                                    sizeof(int), cudaMemcpyDeviceToHost));
        numDel = sc.hostCounts[3];
        if (numDel > 0)
            keysToPairsKernel<<<gridFor(numDel, block), block>>>(
                sc.selected, numDel, dd.d_pairs);
        HSOSP_CUDA_CHECK(cub::DeviceSelect::Flagged(
            sc.temp, sc.tempCap, uniq,
            thrust::make_transform_iterator(sc.cls, IsInserted()),
            sc.selected, sc.counts + 3, nPairs));
        HSOSP_CUDA_CHECK(cudaMemcpy(sc.hostCounts + 3, sc.counts + 3,
                                    sizeof(int), cudaMemcpyDeviceToHost));
        numIns = sc.hostCounts[3];
        if (numIns > 0)
            keysToPairsKernel<<<gridFor(numIns, block), block>>>(
                sc.selected, numIns, dd.d_pairs + numDel);
        HSOSP_CUDA_CHECK(cudaGetLastError());
    }

    // ---- 5. post-batch vertex lists of the touched hyperedges --------------
    if (nT > 0) {
        std::vector<long long> writeOff(nT);
        bool reupload = false;
        for (int t = 0; t < nT && !reupload; ++t) {
            const int node = inc.touched[t] - 1;
            const int l = inc.postOff[t + 1] - inc.postOff[t];
            if (l > di.heCap[node]) {
                const int c = heSlotFor(l);
                if (di.heTail + c > di.heCapacity) {
                    reupload = true;
                    break;
                }
                di.heOff[node] = di.heTail;
                di.heCap[node] = c;
                di.heTail += c;
            }
            writeOff[t] = di.heOff[node];
        }
        if (reupload) {
            uploadHeRows(di, hg, 1.5);
            ++reuploads;
        } else {
            upload(sc.heWriteOff, writeOff);
            writeHeRowsKernel<<<gridFor(32LL * nT, block), block>>>(
                sc.touched, sc.heWriteOff, nT, sc.postOff, sc.postVals,
                di.d_heOff, di.d_heLen, di.d_heVal);
            HSOSP_CUDA_CHECK(cudaGetLastError());
        }
    }

    // ---- 6. ids of new and dead nodes, weights -----------------------------
    std::vector<int> ids;
    ids.reserve(inc.newHe.size() + inc.deadHe.size());
    for (int id : inc.newHe) ids.push_back(id - 1);
    for (int id : inc.deadHe) ids.push_back(id - 1);
    dd.reserve(0, static_cast<long long>(ids.size()));
    upload(dd.d_ids, ids);
    upload(dd.d_newW, inc.newW);
    dd.numDel = numDel;
    dd.numIns = numIns;
    dd.numNew = static_cast<int>(inc.newHe.size());
    dd.numDead = static_cast<int>(inc.deadHe.size());
    return nCand;
}

} // namespace

BatchTimes applyBatch(DynamicHypergraph& dh, DeviceH2H& dev,
                      const HgBatch& batch, double entryHeadroom) {
    BatchTimes t;
    HostHypergraph& hg = dh.host();

    // 1. h2v insert (ids) + host incidence.
    DynamicHypergraph::BatchResult br = dh.beginBatch(batch);
    t.escherMs += br.escherMs;
    t.deltaMs += br.deltaMs;
    t.newHe = static_cast<int>(br.inc.newHe.size());
    t.deadHe = static_cast<int>(br.inc.deadHe.size());
    t.skippedOps = br.inc.skippedOps;
    if (hg.maxId() > dev.maxNodes)
        throw std::runtime_error(
            "applyBatch: node capacity exceeded; raise maxHyperedges");

    // 2. Unification on the GPU.
    auto t0 = Clock::now();
    t.candidates = deriveDelta(dev, hg, br.inc, t.mirrorReuploads);
    HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
    t.deltaMs += msSince(t0);
    t.insPairs = dev.delta.numIns;
    t.delPairs = dev.delta.numDel;

    // 3. Device CSR.
    t0 = Clock::now();
    dev.numNodes = hg.maxId();
    dev.numEntries += 2LL * (t.insPairs - t.delPairs);
    if (!detail::applyDeltaPairs(dev)) {
        // Tail exhausted: rebuild the CSR from the host incidence (the
        // mirror is already up to date); the delta stays on the device for
        // the update and for step 4.
        t.csrRebuilt = true;
        LineGraphCSR lg = hg.lineGraph();
        detail::buildCsr(dev, hg, lg, dev.maxNodes, entryHeadroom);
    }
    HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
    t.csrMs = msSince(t0);

    // 4. Remaining ESCHER maintenance from the sorted directed pairs.
    t0 = Clock::now();
    DeviceDelta& dd = dev.delta;
    const long long nKeys = dd.numSortedKeys;
    if (nKeys > dd.hostKeyCapacity) {
        if (dd.h_sortedKeys) cudaFreeHost(dd.h_sortedKeys);
        dd.h_sortedKeys = nullptr;
        dd.hostKeyCapacity = nKeys + nKeys / 2 + 4096;
        HSOSP_CUDA_CHECK(cudaMallocHost(&dd.h_sortedKeys,
                                        sizeof(u64) * dd.hostKeyCapacity));
    }
    if (nKeys > 0)
        HSOSP_CUDA_CHECK(cudaMemcpy(dd.h_sortedKeys, dd.d_sortedKeys,
                                    sizeof(u64) * nKeys,
                                    cudaMemcpyDeviceToHost));
    static_assert(sizeof(u64) == sizeof(std::uint64_t), "key width");
    dh.finishBatch(br, reinterpret_cast<const std::uint64_t*>(dd.h_sortedKeys),
                   static_cast<std::size_t>(nKeys));
    HSOSP_CUDA_CHECK(cudaDeviceSynchronize());
    t.escherMs += msSince(t0);
    return t;
}

} // namespace hsosp
} // namespace escher_mosp
