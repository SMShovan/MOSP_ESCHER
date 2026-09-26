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
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "structure.hpp"
#include "flatten.hpp"
#include "escher_errors.hpp"
#include "integrity.hpp"

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
 * @brief ESCHER record of one edge: [src+1, dst+1, w_0+1, ..., w_{K-1}+1].
 *
 * Every field is shifted by one because a 0 value ends a CBST row: the
 * original stored [src, dst, w...], so edges leaving vertex 0 (or with a
 * zero weight) read back empty or truncated. Weights must be non-negative.
 */
std::vector<int> edgeRecord(int src, int dst, const std::vector<int>& w) {
    std::vector<int> rec;
    rec.reserve(2 + w.size());
    rec.push_back(src + 1);
    rec.push_back(dst + 1);
    for (int x : w) rec.push_back(x + 1);
    return rec;
}

void checkWeights(const std::vector<int>& w, const char* where) {
    for (int x : w) {
        if (x < 0) {
            throw escher::EscherError(std::string(where) +
                                      ": edge weights must be non-negative");
        }
    }
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

    // Per-edge metadata, indexed by (edge-id - 1). Edge-ids are the keys of
    // the edge records in edgesCBST; inserts adopt the key ESCHER assigns
    // (a reused deleted slot or a fresh key).
    std::vector<int>              edgeSrc;      // edgeSrc[eid-1] = source vertex (or -1 if slot free)
    std::vector<int>              edgeDst;      // edgeDst[eid-1] = destination vertex (or -1 if slot free)
    std::vector<std::vector<int>> edgeWeights;  // edgeWeights[eid-1] = K weights

    // (src,dst) -> edge-id lookup for @c deleteEdges.
    std::unordered_map<std::int64_t, int> edgeIdBySrcDst;

    /**
     * @brief Clear the metadata of a deleted edge-id. Ids are chosen by
     *        ESCHER (the key its insert assigns), so no host free list is
     *        kept.
     *
     * @param edgeId 1-based edge-id.
     */
    void releaseEdgeId_(int edgeId) {
        int idx = edgeId - 1;
        edgeSrc[idx] = -1;
        edgeDst[idx] = -1;
        edgeWeights[idx].clear();
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
        checkWeights(w, "loadFromCSR");
    }

    pImpl->numEdges_ = E;
    pImpl->edgeSrc.assign(E, -1);
    pImpl->edgeDst.assign(E, -1);
    pImpl->edgeWeights.assign(E, std::vector<int>{});
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

            perEdgeRecords[idx] = edgeRecord(u, v, values[k]);

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
        checkWeights(e.weights, "insertEdges");
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

    // Build the flat payload vectors in the format @c insertCBST expects
    // and route the insert through ESCHER first: its best-fit slot reuse
    // decides the key of every record, and that key becomes the edge-id.
    // (The original allocated ids from a host free list and discarded the
    // returned mapping, so host ids and ESCHER keys diverged and a later
    // erase by host id removed the wrong record.)
    const int M = static_cast<int>(edges.size());
    std::vector<int> tentativeKeys;
    std::vector<int> newPayload;
    std::vector<int> newPrefixSizes;
    tentativeKeys.reserve(M);
    newPrefixSizes.reserve(M);
    newPayload.reserve(static_cast<std::size_t>(M) * (2 + K));
    int running = 0;
    for (int i = 0; i < M; ++i) {
        const auto& e = edges[i];
        tentativeKeys.push_back(static_cast<int>(pImpl->edgeSrc.size()) + i + 1);
        std::vector<int> rec = edgeRecord(e.src, e.dst, e.weights);
        newPayload.insert(newPayload.end(), rec.begin(), rec.end());
        running += static_cast<int>(rec.size());
        newPrefixSizes.push_back(running);
    }
    InsertMapping mapping =
        pImpl->edgesCBST->insert(tentativeKeys, newPayload, newPrefixSizes);

    std::vector<int> assignedIds(M);
    for (int i = 0; i < M; ++i) {
        const auto& e = edges[i];
        const int eid = mapping.itemToKey[i];
        if (eid < 1) {
            throw escher::EscherError("insertEdges: ESCHER returned an invalid edge key");
        }
        if (eid > static_cast<int>(pImpl->edgeSrc.size())) {
            pImpl->edgeSrc.resize(eid, -1);
            pImpl->edgeDst.resize(eid, -1);
            pImpl->edgeWeights.resize(eid);
        }
        if (pImpl->edgeSrc[eid - 1] != -1) {
            throw escher::EscherError("insertEdges: ESCHER reused the key of a live edge");
        }
        assignedIds[i] = eid;
        pImpl->edgeSrc[eid - 1]     = e.src;
        pImpl->edgeDst[eid - 1]     = e.dst;
        pImpl->edgeWeights[eid - 1] = e.weights;
        pImpl->edgeIdBySrcDst[packSrcDst(e.src, e.dst)] = eid;
    }

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

long long DynamicGraph::checkEscher(std::ostream& log) const {
    const int V = pImpl->numVertices_;
    std::vector<std::vector<int>> edgeRows(pImpl->edgeSrc.size());
    for (std::size_t i = 0; i < edgeRows.size(); ++i) {
        if (pImpl->edgeSrc[i] < 0) continue;
        edgeRows[i] = edgeRecord(pImpl->edgeSrc[i], pImpl->edgeDst[i],
                                 pImpl->edgeWeights[i]);
    }
    std::vector<std::vector<int>> outRows(V), inRows(V);
    for (std::size_t i = 0; i < edgeRows.size(); ++i) {
        if (pImpl->edgeSrc[i] < 0) continue;
        outRows[pImpl->edgeSrc[i]].push_back(static_cast<int>(i) + 1);
        inRows[pImpl->edgeDst[i]].push_back(static_cast<int>(i) + 1);
    }
    long long errors = 0;
    if (pImpl->edgesCBST) {
        errors += checkTreeRows(pImpl->edgesCBST->context(), edgeRows,
                                /*orderInsensitive=*/false, "edges", log);
        errors += checkSubtreeAvail(pImpl->edgesCBST->context(), "edges", log);
    }
    if (pImpl->outAdjCBST)
        errors += checkTreeRows(pImpl->outAdjCBST->context(), outRows, true,
                                "outAdj", log);
    if (pImpl->inAdjCBST)
        errors += checkTreeRows(pImpl->inAdjCBST->context(), inRows, true,
                                "inAdj", log);
    return errors;
}

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
