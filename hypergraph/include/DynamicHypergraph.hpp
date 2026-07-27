#ifndef ESCHER_MOSP_DYNAMIC_HYPERGRAPH_HPP
#define ESCHER_MOSP_DYNAMIC_HYPERGRAPH_HPP

/**
 * @file DynamicHypergraph.hpp
 * @brief ESCHER-backed dynamic hypergraph: the authoritative store for the
 *        H-SOSP pipeline.
 *
 * Three CBSTs from libescher_core hold the three mappings of the ESCHER
 * paper (Table "Supported operations in different mappings"):
 *
 *   - h2vCBST: key = hyperedge id, payload = incident vertices (+1).
 *              Vertical ops (hyperedge insert / delete) use the real
 *              CBSTOperations::insert best-fit reuse and ::erase avail
 *              propagation; horizontal ops (incident vertex insert /
 *              delete) use fillCBST / unfillCBST.
 *   - v2hCBST: key = vertex id + 1, payload = incident hyperedge ids.
 *              Horizontal ops only (the vertex universe is fixed).
 *   - h2hCBST: key = its own id space (see below), payload = neighboring
 *              hyperedge ids. Vertical + horizontal ops.
 *
 * Hyperedge ids adopt the keys returned by the h2v insert mapping (recycled
 * dead slots or fresh keys), exactly the reassignment scheme described in
 * the ESCHER paper's insertion Case 1. Because best-fit matching is
 * size-driven, the h2h CBST's insert may map the same hyperedge to a
 * *different* recycled key; the host keeps the (hyperedge id -> h2h key)
 * translation so all three structures stay consistent.
 *
 * A HostHypergraph shadow mirrors every operation (the DynamicGraph
 * pattern): reads (device CSR materialization, delta extraction, ground
 * truth) are served from the shadow in linear time, while every write goes
 * through the ESCHER structures and is timed as data structure maintenance.
 */

#include <memory>
#include <vector>

#include "HostHypergraph.hpp"

namespace escher_mosp {

class DynamicHypergraph {
public:
    struct Caps {
        int maxHyperedges = 0;   ///< id-space bound (initial + all inserts)
        /// CBST payload capacity = initial flat size * headroomFactor +
        /// extraPayloadInts (applied to each of the three structures).
        double headroomFactor = 1.3;
        long long extraPayloadInts = 1 << 20;
    };

    struct BatchResult {
        double escherMs = 0.0;   ///< CBST maintenance (all three mappings)
        double deltaMs = 0.0;    ///< host shadow update + delta extraction
        H2HDelta delta;
    };

    DynamicHypergraph(int numVertices, const Caps& caps);
    ~DynamicHypergraph();
    DynamicHypergraph(const DynamicHypergraph&) = delete;
    DynamicHypergraph& operator=(const DynamicHypergraph&) = delete;

    /** Initial build: host shadow + all three CBSTs (with the occupancy fix
     *  so later fills append correctly). Rows become ids 1..rows.size(). */
    void bulkLoad(std::vector<std::vector<int>>&& rows,
                  std::vector<long long>&& weights, int sourceHe,
                  int targetHe);

    /** Apply one batch through ESCHER + shadow; returns timings + delta. */
    BatchResult applyBatch(const HgBatch& batch);

    HostHypergraph& host();
    const HostHypergraph& host() const;

    /** Approximate device memory held by the three CBSTs. */
    long long escherDeviceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace escher_mosp

#endif // ESCHER_MOSP_DYNAMIC_HYPERGRAPH_HPP
