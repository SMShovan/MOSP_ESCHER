#ifndef ESCHER_MOSP_DYNAMIC_GRAPH_HPP
#define ESCHER_MOSP_DYNAMIC_GRAPH_HPP

#include <memory>
#include <vector>

#include "GraphSnapshot.hpp"

namespace escher_mosp {

/**
 * @brief A single directed edge to insert with its multi-objective weights.
 *
 * @c weights must have exactly @c numObjectives entries, matching the value
 * passed to the owning @c DynamicGraph constructor.
 */
struct EdgeInsert {
    int src;                    /**< 0-indexed source vertex. */
    int dst;                    /**< 0-indexed destination vertex. */
    std::vector<int> weights;   /**< One weight per objective. */
};

/**
 * @brief A single directed edge to delete, identified by its endpoints.
 *
 * If multiple parallel edges exist between @c src and @c dst only the
 * first one registered is removed. MOSP's CSR format does not produce
 * parallel edges, so this is not a concern in practice.
 */
struct EdgeDelete {
    int src;   /**< 0-indexed source vertex. */
    int dst;   /**< 0-indexed destination vertex. */
};

/**
 * @brief A dynamic multi-objective directed graph backed by the ESCHER CBST.
 *
 * Externally this exposes a flat CSR-style API: load from CSR, batch insert
 * and delete edges, and materialize a @c GraphSnapshot for a given objective
 * that MOSP's CUDA kernels can consume directly.
 *
 * Internally it maintains three Complete Binary Search Trees (CBSTs) from
 * @c libescher_core:
 *   - @b edgesCBST: one record per directed edge keyed by edge-id,
 *     payload = @c [src, dst, w_0, ..., w_{K-1}]. Inserts/deletes on the
 *     edge set flow through @c CBSTOperations::insert / @c ::erase so every
 *     dynamic update goes through the ESCHER structure.
 *   - @b outAdjCBST: one record per source vertex keyed by vertex-id,
 *     payload = variable-length list of outgoing edge-ids. Grows via
 *     @c fillCBST and shrinks via @c unfillCBST.
 *   - @b inAdjCBST: one record per destination vertex keyed by vertex-id,
 *     payload = variable-length list of incoming edge-ids. Same semantics
 *     as @c outAdjCBST.
 *
 * A host shadow of the adjacency lists and per-edge weights is kept in
 * parallel so that @c snapshot / @c dumpToCSR can build CSR views in
 * linear time without reading the CBST back. The shadow is updated in
 * lock-step with every ESCHER operation so it never drifts.
 *
 * @note The number of objectives @c K is fixed for the lifetime of the
 *       graph. This matches MOSP's per-run objective model.
 */
class DynamicGraph {
public:
    /**
     * @param numVertices      Number of vertices (graph is 0-indexed).
     * @param numObjectives    Weights-per-edge @c K.
     * @param payloadCapacity  Upper-bound capacity for each CBST's flat payload buffer.
     *                         Conservatively size this above @c (2+K) * expectedEdges.
     */
    DynamicGraph(int numVertices, int numObjectives, int payloadCapacity);
    ~DynamicGraph();

    DynamicGraph(const DynamicGraph&)            = delete;
    DynamicGraph& operator=(const DynamicGraph&) = delete;
    DynamicGraph(DynamicGraph&&) noexcept;
    DynamicGraph& operator=(DynamicGraph&&) noexcept;

    /**
     * @brief Initial bulk load from MOSP's CSR in-memory representation.
     *
     * @param rowPtr   CSR row offsets, size @c numVertices+1.
     * @param colInd   CSR destination vertices, size @c rowPtr.back().
     * @param values   Per-edge weight vectors, size @c rowPtr.back(), each with @c numObjectives entries.
     */
    void loadFromCSR(const std::vector<int>& rowPtr,
                     const std::vector<int>& colInd,
                     const std::vector<std::vector<int>>& values);

    /** @brief Batch-insert edges, routed through @c CBSTOperations::insert and @c ::fill. */
    void insertEdges(const std::vector<EdgeInsert>& edges);

    /** @brief Batch-delete edges, routed through @c CBSTOperations::erase and @c unfillCBST. */
    void deleteEdges(const std::vector<EdgeDelete>& edges);

    /**
     * @brief Materialize a device-resident CSR view for the given objective.
     *
     * Allocates device buffers inside the returned @c GraphSnapshot; the
     * caller takes ownership and the snapshot's destructor frees them.
     */
    GraphSnapshot snapshot(int objective) const;

    /**
     * @brief Read the current graph state back into host CSR arrays.
     *
     * Used to (a) write the updated CSR files that feed MOSP's Dijkstra
     * ground-truth path and (b) rebuild the host-side adjacency lists that
     * @c sequentialSOSPUpdate consumes.
     */
    void dumpToCSR(std::vector<int>& rowPtr,
                   std::vector<int>& colInd,
                   std::vector<std::vector<int>>& values) const;

    int numVertices() const noexcept;
    int numEdges()    const noexcept;
    int numObjectives() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace escher_mosp

#endif // ESCHER_MOSP_DYNAMIC_GRAPH_HPP
