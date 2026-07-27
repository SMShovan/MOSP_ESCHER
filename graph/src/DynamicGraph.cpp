/**
 * @file DynamicGraph.cpp
 * @brief Host-side implementation of @ref escher_mosp::DynamicGraph.
 *
 * Drives three @c CBSTOperations instances (edges, out-adjacency, in-adjacency)
 * from @c libescher_core and mirrors the topology in host shadow vectors for
 * fast CSR materialization. Every dynamic edge update goes through the ESCHER
 * CBST operations so the data structure becomes the authoritative store for
 * MOSP.
 */

#include "DynamicGraph.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "structure.hpp"
#include "flatten.hpp"
#include "escher_errors.hpp"

namespace escher_mosp {

namespace {

/**
 * @brief Pack a (src, dst) pair into a single 64-bit key for the edge lookup map.
 *
 * @param src 0-indexed source vertex.
 * @param dst 0-indexed destination vertex.
 * @return Packed key.
 */
inline std::int64_t packSrcDst(int src, int dst) noexcept {
    return (static_cast<std::int64_t>(src) << 32) | static_cast<std::uint32_t>(dst);
}

/**
 * @brief Build the 1..N keys and startOffsets arrays that @c constructCBST expects.
 *
 * @param startOffsets  Input per-record start offsets from @c flatten2DVector.
 * @return Pair @c (keys, offsets) where @c keys[i] = i+1 and @c offsets[i] = startOffsets[i].
 */
std::pair<std::vector<int>, std::vector<int>>
buildKeysAndOffsets(const std::vector<int>& startOffsets) {
    std::vector<int> keys(startOffsets.size());
    std::iota(keys.begin(), keys.end(), 1);
    return {std::move(keys), startOffsets};
}

} // namespace

/**
 * @brief Private state for @ref DynamicGraph, hidden via pImpl to keep the
 *        public header free of ESCHER core types.
 */
struct DynamicGraph::Impl {
    int numVertices_    = 0;
    int numObjectives_  = 0;
    int payloadCapacity_ = 0;
    int numEdges_       = 0;

    // ESCHER-backed authoritative storage.
    std::unique_ptr<CBSTOperations> edgesCBST;
    std::unique_ptr<CBSTOperations> outAdjCBST;
    std::unique_ptr<CBSTOperations> inAdjCBST;

    // Host shadow of the adjacency topology for fast snapshot. Mirrors the
    // state of outAdjCBST / inAdjCBST after every update.
    std::vector<std::vector<int>> outAdjShadow; // outAdjShadow[v] = list of edge-ids outgoing from v
    std::vector<std::vector<int>> inAdjShadow;  // inAdjShadow[v]  = list of edge-ids incoming to v

    // Per-edge metadata, indexed by (edge-id - 1). A free-list on edge-ids
    // lets us recycle slots that ESCHER has marked available via @c erase,
    // keeping the edge-id space dense.
    std::vector<int>              edgeSrc;      // edgeSrc[eid-1] = source vertex (or -1 if slot free)
    std::vector<int>              edgeDst;      // edgeDst[eid-1] = destination vertex (or -1 if slot free)
    std::vector<std::vector<int>> edgeWeights;  // edgeWeights[eid-1] = K weights

    std::vector<int>              freeEdgeIds;  // LIFO free-list of deleted edge-ids

    // (src,dst) -> edge-id lookup for @c deleteEdges.
    std::unordered_map<std::int64_t, int> edgeIdBySrcDst;

    /**
     * @brief Allocate the next edge-id, reusing a free slot if possible.
     *
     * Preferring a free-list entry over a fresh id matches ESCHER's best-fit
     * slot-reuse philosophy in @c insertCBST and keeps the dense edge-id
     * space tidy.
     *
     * @return A valid 1-based edge-id.
     */
    int allocateEdgeId_() {
        if (!freeEdgeIds.empty()) {
            int id = freeEdgeIds.back();
            freeEdgeIds.pop_back();
            return id;
        }
        int id = static_cast<int>(edgeSrc.size()) + 1;
        edgeSrc.push_back(-1);
        edgeDst.push_back(-1);
        edgeWeights.emplace_back();
        return id;
    }

    /**
     * @brief Release an edge-id back to the free-list and clear its metadata.
     *
     * @param edgeId 1-based edge-id previously returned from @c allocateEdgeId_.
     */
    void releaseEdgeId_(int edgeId) {
        int idx = edgeId - 1;
        edgeSrc[idx] = -1;
        edgeDst[idx] = -1;
        edgeWeights[idx].clear();
        freeEdgeIds.push_back(edgeId);
    }

    /**
     * @brief Bootstrap all three CBSTs with an initial set of edges.
     *
     * @param perVertexOut  outAdjShadow, already populated.
     * @param perVertexIn   inAdjShadow, already populated.
     */
    void constructCbsts_(std::vector<std::vector<int>>& perEdgeRecords,
                         std::vector<std::vector<int>>& perVertexOut,
                         std::vector<std::vector<int>>& perVertexIn) {
        // True per-row value counts, so constructCBST can initialize each
        // node's occupancy and later fills append instead of overwriting.
        auto rowCounts = [](const std::vector<std::vector<int>>& rows) {
            std::vector<int> counts(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i)
                counts[i] = static_cast<int>(rows[i].size());
            return counts;
        };

        // --- edgesCBST ---
        edgesCBST = std::make_unique<CBSTOperations>("edges", payloadCapacity_, 4);
        auto [edgesFlat, edgesOffsets] = flatten2DVector(perEdgeRecords);
        auto [edgesKeys, edgesStarts]  = buildKeysAndOffsets(edgesOffsets);
        std::vector<int> edgesCounts = rowCounts(perEdgeRecords);
        edgesCBST->construct(edgesKeys.data(), edgesStarts.data(),
                             static_cast<int>(perEdgeRecords.size()),
                             edgesFlat.data(), static_cast<int>(edgesFlat.size()),
                             edgesCounts.data());

        // --- outAdjCBST ---
        outAdjCBST = std::make_unique<CBSTOperations>("outAdj", payloadCapacity_, 4);
        auto [outFlat, outOffsets] = flatten2DVector(perVertexOut);
        auto [outKeys, outStarts]  = buildKeysAndOffsets(outOffsets);
        std::vector<int> outCounts = rowCounts(perVertexOut);
        outAdjCBST->construct(outKeys.data(), outStarts.data(),
                              static_cast<int>(perVertexOut.size()),
                              outFlat.data(), static_cast<int>(outFlat.size()),
                              outCounts.data());

        // --- inAdjCBST ---
        inAdjCBST = std::make_unique<CBSTOperations>("inAdj", payloadCapacity_, 4);
        auto [inFlat, inOffsets] = flatten2DVector(perVertexIn);
        auto [inKeys, inStarts]  = buildKeysAndOffsets(inOffsets);
        std::vector<int> inCounts = rowCounts(perVertexIn);
        inAdjCBST->construct(inKeys.data(), inStarts.data(),
                             static_cast<int>(perVertexIn.size()),
                             inFlat.data(), static_cast<int>(inFlat.size()),
                             inCounts.data());
    }
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

DynamicGraph::DynamicGraph(int numVertices, int numObjectives, int payloadCapacity)
    : pImpl(std::make_unique<Impl>()) {
    if (numVertices <= 0 || numObjectives <= 0 || payloadCapacity <= 0) {
        throw escher::EscherError(
            "DynamicGraph: numVertices, numObjectives, payloadCapacity must all be positive");
    }
    pImpl->numVertices_     = numVertices;
    pImpl->numObjectives_   = numObjectives;
    pImpl->payloadCapacity_ = payloadCapacity;
    pImpl->outAdjShadow.assign(numVertices, {});
    pImpl->inAdjShadow.assign(numVertices, {});
}

DynamicGraph::~DynamicGraph() = default;
DynamicGraph::DynamicGraph(DynamicGraph&&) noexcept = default;
DynamicGraph& DynamicGraph::operator=(DynamicGraph&&) noexcept = default;

int DynamicGraph::numVertices()   const noexcept { return pImpl->numVertices_; }
int DynamicGraph::numEdges()      const noexcept { return pImpl->numEdges_; }
int DynamicGraph::numObjectives() const noexcept { return pImpl->numObjectives_; }

// ---------------------------------------------------------------------------
// loadFromCSR
// ---------------------------------------------------------------------------

void DynamicGraph::loadFromCSR(const std::vector<int>& rowPtr,
                               const std::vector<int>& colInd,
                               const std::vector<std::vector<int>>& values) {
    const int V = pImpl->numVertices_;
    const int K = pImpl->numObjectives_;

    if (static_cast<int>(rowPtr.size()) != V + 1) {
        throw escher::EscherError("loadFromCSR: rowPtr.size() must equal numVertices+1");
    }
    const int E = rowPtr.back();
    if (static_cast<int>(colInd.size()) != E || static_cast<int>(values.size()) != E) {
        throw escher::EscherError("loadFromCSR: colInd/values size mismatch with rowPtr.back()");
    }
    for (const auto& w : values) {
        if (static_cast<int>(w.size()) != K) {
            throw escher::EscherError("loadFromCSR: per-edge weights must have numObjectives entries");
        }
    }

    pImpl->numEdges_ = E;
    pImpl->edgeSrc.assign(E, -1);
    pImpl->edgeDst.assign(E, -1);
    pImpl->edgeWeights.assign(E, std::vector<int>{});
    pImpl->freeEdgeIds.clear();
    pImpl->edgeIdBySrcDst.clear();
    pImpl->edgeIdBySrcDst.reserve(static_cast<std::size_t>(E) * 2);

    // Build per-edge records and per-vertex adjacency shadows.
    std::vector<std::vector<int>> perEdgeRecords(E);
    pImpl->outAdjShadow.assign(V, {});
    pImpl->inAdjShadow.assign(V, {});

    int edgeId = 1;
    for (int u = 0; u < V; ++u) {
        for (int k = rowPtr[u]; k < rowPtr[u + 1]; ++k, ++edgeId) {
            const int v = colInd[k];
            const int idx = edgeId - 1;

            pImpl->edgeSrc[idx]     = u;
            pImpl->edgeDst[idx]     = v;
            pImpl->edgeWeights[idx] = values[k];

            // Record layout: [src, dst, w_0, ..., w_{K-1}]
            std::vector<int>& rec = perEdgeRecords[idx];
            rec.reserve(2 + K);
            rec.push_back(u);
            rec.push_back(v);
            for (int w : values[k]) rec.push_back(w);

            pImpl->outAdjShadow[u].push_back(edgeId);
            pImpl->inAdjShadow[v].push_back(edgeId);
            pImpl->edgeIdBySrcDst[packSrcDst(u, v)] = edgeId;
        }
    }

    // Adjacency CBSTs need one record per vertex (including isolated ones);
    // flatten2DVector pads empty rows to 4 ints so the CBST is well-formed.
    pImpl->constructCbsts_(perEdgeRecords,
                           pImpl->outAdjShadow,
                           pImpl->inAdjShadow);
}

// ---------------------------------------------------------------------------
// insertEdges / deleteEdges
// ---------------------------------------------------------------------------

void DynamicGraph::insertEdges(const std::vector<EdgeInsert>& edges) {
    if (edges.empty()) return;
    const int K = pImpl->numObjectives_;
    const int V = pImpl->numVertices_;

    // Validate input up front so we fail before mutating any CBST.
    for (const auto& e : edges) {
        if (e.src < 0 || e.src >= V || e.dst < 0 || e.dst >= V) {
            throw escher::EscherError("insertEdges: vertex index out of range");
        }
        if (static_cast<int>(e.weights.size()) != K) {
            throw escher::EscherError("insertEdges: weights must have numObjectives entries");
        }
        // Guard against parallel-edge leaks: re-inserting an existing (src,dst)
        // previously allocated a second edge record and orphaned the first one
        // in the (src,dst)->id map. Upsert semantics are handled one level up
        // (updateGraphWithESCHER deletes before inserting); reaching this point
        // with a duplicate is a caller bug, so fail loudly instead of leaking.
        if (pImpl->edgeIdBySrcDst.count(packSrcDst(e.src, e.dst)) != 0) {
            throw escher::EscherError(
                "insertEdges: edge already exists (delete it first for upsert)");
        }
    }

    // Allocate edge-ids (reusing freed slots first) and build flat payload
    // vectors in the format @c insertCBST expects:
    //   - newKeys[i]          = edge-id for item i
    //   - newPayload          = concatenation of per-item payloads
    //   - newPrefixSizes[i]   = cumulative payload size up to and including item i
    const int M = static_cast<int>(edges.size());
    std::vector<int> newKeys;
    std::vector<int> newPayload;
    std::vector<int> newPrefixSizes;
    newKeys.reserve(M);
    newPrefixSizes.reserve(M);
    newPayload.reserve(static_cast<std::size_t>(M) * (2 + K));

    std::vector<int> assignedIds(M);
    int running = 0;
    for (int i = 0; i < M; ++i) {
        const auto& e = edges[i];
        const int eid = pImpl->allocateEdgeId_();
        assignedIds[i] = eid;

        pImpl->edgeSrc[eid - 1]     = e.src;
        pImpl->edgeDst[eid - 1]     = e.dst;
        pImpl->edgeWeights[eid - 1] = e.weights;
        pImpl->edgeIdBySrcDst[packSrcDst(e.src, e.dst)] = eid;

        newKeys.push_back(eid);
        newPayload.push_back(e.src);
        newPayload.push_back(e.dst);
        for (int w : e.weights) newPayload.push_back(w);
        running += (2 + K);
        newPrefixSizes.push_back(running);
    }

    // Route through ESCHER: insertCBST applies best-fit slot reuse for
    // edges whose slots were freed by a prior @c erase.
    pImpl->edgesCBST->insert(newKeys, newPayload, newPrefixSizes);

    // Push the new edge-ids into outAdj / inAdj per-vertex payload lists.
    // Group by source vertex for outAdj and by destination for inAdj, then
    // call @c fillCBST once per side.
    std::unordered_map<int, std::vector<int>> outAdds;
    std::unordered_map<int, std::vector<int>> inAdds;
    outAdds.reserve(edges.size());
    inAdds.reserve(edges.size());
    for (int i = 0; i < M; ++i) {
        outAdds[edges[i].src].push_back(assignedIds[i]);
        inAdds [edges[i].dst].push_back(assignedIds[i]);
        pImpl->outAdjShadow[edges[i].src].push_back(assignedIds[i]);
        pImpl->inAdjShadow [edges[i].dst].push_back(assignedIds[i]);
    }

    auto flushFill = [&](std::unordered_map<int, std::vector<int>>& adds,
                         CBSTOperations& cbst) {
        if (adds.empty()) return;
        std::vector<int> keys;
        std::vector<int> payload;
        std::vector<int> prefixSizes;
        keys.reserve(adds.size());
        prefixSizes.reserve(adds.size());
        int run = 0;
        for (auto& kv : adds) {
            keys.push_back(kv.first + 1); // 1-based CBST keys
            for (int id : kv.second) payload.push_back(id);
            run += static_cast<int>(kv.second.size());
            prefixSizes.push_back(run);
        }
        cbst.fill(keys, payload, prefixSizes);
    };

    flushFill(outAdds, *pImpl->outAdjCBST);
    flushFill(inAdds,  *pImpl->inAdjCBST);

    pImpl->numEdges_ += M;
}

void DynamicGraph::deleteEdges(const std::vector<EdgeDelete>& edges) {
    if (edges.empty()) return;
    const int V = pImpl->numVertices_;

    // Resolve (src,dst) -> edge-id and partition removals per source / dest.
    std::vector<int> edgeKeysToErase;
    edgeKeysToErase.reserve(edges.size());

    std::unordered_map<int, std::vector<int>> outRem;
    std::unordered_map<int, std::vector<int>> inRem;

    for (const auto& d : edges) {
        if (d.src < 0 || d.src >= V || d.dst < 0 || d.dst >= V) {
            throw escher::EscherError("deleteEdges: vertex index out of range");
        }
        auto it = pImpl->edgeIdBySrcDst.find(packSrcDst(d.src, d.dst));
        if (it == pImpl->edgeIdBySrcDst.end()) {
            // Silently skip: matches updateGraphCSR.cu's "delete-nonexistent is a no-op".
            continue;
        }
        const int eid = it->second;
        edgeKeysToErase.push_back(eid);
        outRem[d.src].push_back(eid);
        inRem [d.dst].push_back(eid);

        // Remove from host shadow.
        auto& outList = pImpl->outAdjShadow[d.src];
        outList.erase(std::remove(outList.begin(), outList.end(), eid), outList.end());
        auto& inList = pImpl->inAdjShadow[d.dst];
        inList.erase(std::remove(inList.begin(), inList.end(), eid), inList.end());

        pImpl->edgeIdBySrcDst.erase(it);
        pImpl->releaseEdgeId_(eid);
    }

    if (edgeKeysToErase.empty()) return;

    // Remove edge-ids from the per-vertex adjacency payloads via unfillCBST.
    auto flushUnfill = [](std::unordered_map<int, std::vector<int>>& rem,
                          CBSTContext& ctx) {
        if (rem.empty()) return;
        std::vector<int> keys;
        std::vector<int> valuesToRemove;
        std::vector<int> removePrefixSizes;
        keys.reserve(rem.size());
        removePrefixSizes.reserve(rem.size());
        int run = 0;
        for (auto& kv : rem) {
            keys.push_back(kv.first + 1);
            for (int id : kv.second) valuesToRemove.push_back(id);
            run += static_cast<int>(kv.second.size());
            removePrefixSizes.push_back(run);
        }
        unfillCBST(keys, valuesToRemove, removePrefixSizes, ctx);
    };

    // @c unfillCBST is a free function taking a mutable @c CBSTContext&. The
    // @c context() accessor on @c CBSTOperations is const; cast away const
    // locally because the underlying device buffers are being mutated.
    flushUnfill(outRem, const_cast<CBSTContext&>(pImpl->outAdjCBST->context()));
    flushUnfill(inRem,  const_cast<CBSTContext&>(pImpl->inAdjCBST->context()));

    // Finally erase the edge records themselves so their slots become
    // available for future @c insert best-fit reuse.
    pImpl->edgesCBST->erase(edgeKeysToErase);
    pImpl->numEdges_ -= static_cast<int>(edgeKeysToErase.size());
}

// ---------------------------------------------------------------------------
// dumpToCSR (snapshot is implemented in snapshot.cu so it can call cudaMemcpy)
// ---------------------------------------------------------------------------

void DynamicGraph::dumpToCSR(std::vector<int>& rowPtr,
                             std::vector<int>& colInd,
                             std::vector<std::vector<int>>& values) const {
    const int V = pImpl->numVertices_;
    const int K = pImpl->numObjectives_;

    rowPtr.assign(V + 1, 0);
    for (int u = 0; u < V; ++u) {
        rowPtr[u + 1] = rowPtr[u] + static_cast<int>(pImpl->outAdjShadow[u].size());
    }
    const int E = rowPtr.back();

    colInd.assign(E, 0);
    values.assign(E, std::vector<int>(K, 0));

    // Sort each vertex's neighbor list by destination so dumps are
    // deterministic and match MOSP's CSR conventions.
    for (int u = 0; u < V; ++u) {
        std::vector<int> sorted = pImpl->outAdjShadow[u];
        std::sort(sorted.begin(), sorted.end(),
                  [this](int a, int b) {
                      return pImpl->edgeDst[a - 1] < pImpl->edgeDst[b - 1];
                  });
        int base = rowPtr[u];
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            const int eid = sorted[i];
            colInd[base + static_cast<int>(i)] = pImpl->edgeDst[eid - 1];
            values[base + static_cast<int>(i)] = pImpl->edgeWeights[eid - 1];
        }
    }
}

} // namespace escher_mosp
