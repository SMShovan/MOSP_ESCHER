/**
 * @file snapshot.cu
 * @brief Device-resident CSR materialization for @ref escher_mosp::DynamicGraph.
 *
 * This translation unit owns the CUDA-runtime API calls used to move the
 * host-side CSR layout onto the device. @ref GraphSnapshot's destructor also
 * lives here so the @c cudaFree calls stay inside a @c .cu TU.
 *
 * Splitting @c DynamicGraph into @c .cpp (host orchestration) and @c .cu
 * (device transfers) keeps the host compiler (g++) from having to see any
 * CUDA runtime symbols outside of the ESCHER core it already links against.
 */

#include "DynamicGraph.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

#include "escher_errors.hpp"

namespace escher_mosp {

// ---------------------------------------------------------------------------
// GraphSnapshot lifetime
// ---------------------------------------------------------------------------

void GraphSnapshot::freeDevice_() {
    if (d_outRowPtr) { cudaFree(d_outRowPtr); d_outRowPtr = nullptr; }
    if (d_outColInd) { cudaFree(d_outColInd); d_outColInd = nullptr; }
    if (d_outWeight) { cudaFree(d_outWeight); d_outWeight = nullptr; }
    if (d_inRowPtr)  { cudaFree(d_inRowPtr);  d_inRowPtr  = nullptr; }
    if (d_inColInd)  { cudaFree(d_inColInd);  d_inColInd  = nullptr; }
    if (d_inWeight)  { cudaFree(d_inWeight);  d_inWeight  = nullptr; }
}

GraphSnapshot::~GraphSnapshot() { freeDevice_(); }

GraphSnapshot::GraphSnapshot(GraphSnapshot&& other) noexcept
    : numVertices(other.numVertices),
      numEdges(other.numEdges),
      objective(other.objective),
      d_outRowPtr(other.d_outRowPtr),
      d_outColInd(other.d_outColInd),
      d_outWeight(other.d_outWeight),
      d_inRowPtr(other.d_inRowPtr),
      d_inColInd(other.d_inColInd),
      d_inWeight(other.d_inWeight) {
    other.d_outRowPtr = nullptr;
    other.d_outColInd = nullptr;
    other.d_outWeight = nullptr;
    other.d_inRowPtr  = nullptr;
    other.d_inColInd  = nullptr;
    other.d_inWeight  = nullptr;
    other.numVertices = 0;
    other.numEdges    = 0;
}

GraphSnapshot& GraphSnapshot::operator=(GraphSnapshot&& other) noexcept {
    if (this != &other) {
        freeDevice_();
        numVertices = other.numVertices;
        numEdges    = other.numEdges;
        objective   = other.objective;
        d_outRowPtr = other.d_outRowPtr;
        d_outColInd = other.d_outColInd;
        d_outWeight = other.d_outWeight;
        d_inRowPtr  = other.d_inRowPtr;
        d_inColInd  = other.d_inColInd;
        d_inWeight  = other.d_inWeight;
        other.d_outRowPtr = nullptr;
        other.d_outColInd = nullptr;
        other.d_outWeight = nullptr;
        other.d_inRowPtr  = nullptr;
        other.d_inColInd  = nullptr;
        other.d_inWeight  = nullptr;
        other.numVertices = 0;
        other.numEdges    = 0;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// DynamicGraph::snapshot
// ---------------------------------------------------------------------------

namespace {

/**
 * @brief Copy a host @c int vector to a freshly-allocated device buffer.
 *
 * @throws escher::EscherError on any CUDA failure (allocation or memcpy).
 */
int* uploadIntVec(const std::vector<int>& host) {
    int* d_ptr = nullptr;
    const std::size_t bytes = host.size() * sizeof(int);
    ESCHER_CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
    ESCHER_CHECK_CUDA(cudaMemcpy(d_ptr, host.data(), bytes, cudaMemcpyHostToDevice));
    return d_ptr;
}

/**
 * @brief Copy a host @c long long vector to a freshly-allocated device buffer.
 *
 * @throws escher::EscherError on any CUDA failure.
 */
long long* uploadLongLongVec(const std::vector<long long>& host) {
    long long* d_ptr = nullptr;
    const std::size_t bytes = host.size() * sizeof(long long);
    ESCHER_CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
    ESCHER_CHECK_CUDA(cudaMemcpy(d_ptr, host.data(), bytes, cudaMemcpyHostToDevice));
    return d_ptr;
}

} // namespace

GraphSnapshot DynamicGraph::snapshot(int objective) const {
    if (objective < 0 || objective >= numObjectives()) {
        throw escher::EscherError("snapshot: objective index out of range");
    }

    // Round-trip through dumpToCSR so the snapshot sees the canonical
    // deterministic edge ordering (by destination within each source row).
    std::vector<int> rowPtr;
    std::vector<int> colInd;
    std::vector<std::vector<int>> values;
    dumpToCSR(rowPtr, colInd, values);

    const int V = numVertices();
    const int E = static_cast<int>(colInd.size());

    // Out-adjacency weight column for this objective.
    std::vector<long long> outWeights(E);
    for (int i = 0; i < E; ++i) {
        outWeights[i] = static_cast<long long>(values[i][objective]);
    }

    // Reverse CSR: in-adjacency layout. Counting sort by destination.
    std::vector<int> inRowPtr(V + 1, 0);
    for (int u = 0; u < V; ++u) {
        for (int k = rowPtr[u]; k < rowPtr[u + 1]; ++k) {
            inRowPtr[colInd[k] + 1]++;
        }
    }
    for (int v = 1; v <= V; ++v) inRowPtr[v] += inRowPtr[v - 1];

    std::vector<int> inColInd(E);
    std::vector<long long> inWeights(E);
    std::vector<int> cursor(inRowPtr.begin(), inRowPtr.end() - 1);
    for (int u = 0; u < V; ++u) {
        for (int k = rowPtr[u]; k < rowPtr[u + 1]; ++k) {
            const int v = colInd[k];
            const int slot = cursor[v]++;
            inColInd[slot]  = u;
            inWeights[slot] = static_cast<long long>(values[k][objective]);
        }
    }

    GraphSnapshot snap;
    snap.numVertices = V;
    snap.numEdges    = E;
    snap.objective   = objective;
    snap.d_outRowPtr = uploadIntVec(rowPtr);
    snap.d_outColInd = uploadIntVec(colInd);
    snap.d_outWeight = uploadLongLongVec(outWeights);
    snap.d_inRowPtr  = uploadIntVec(inRowPtr);
    snap.d_inColInd  = uploadIntVec(inColInd);
    snap.d_inWeight  = uploadLongLongVec(inWeights);
    return snap;
}

} // namespace escher_mosp
