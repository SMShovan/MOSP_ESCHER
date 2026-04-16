#ifndef ESCHER_MOSP_GRAPH_SNAPSHOT_HPP
#define ESCHER_MOSP_GRAPH_SNAPSHOT_HPP

namespace escher_mosp {

/**
 * @brief A device-resident CSR view of a @c DynamicGraph for a single objective.
 *
 * The snapshot owns its device buffers and frees them in its destructor. Move
 * semantics are supported; copies are disabled to make ownership explicit.
 *
 * Layout matches the format MOSP's existing CUDA kernels expect:
 * forward adjacency (out-edges from each vertex) and reverse adjacency
 * (in-edges into each vertex) in parallel CSR arrays.
 */
struct GraphSnapshot {
    int numVertices = 0;   /**< Number of vertices. */
    int numEdges    = 0;   /**< Number of directed edges. */
    int objective   = 0;   /**< Which weight index this CSR materializes. */

    int*       d_outRowPtr = nullptr; /**< Device: out-adjacency row offsets (size numVertices+1). */
    int*       d_outColInd = nullptr; /**< Device: out-adjacency destination vertices (size numEdges). */
    long long* d_outWeight = nullptr; /**< Device: out-edge weights for the selected objective (size numEdges). */

    int*       d_inRowPtr  = nullptr; /**< Device: in-adjacency row offsets (size numVertices+1). */
    int*       d_inColInd  = nullptr; /**< Device: in-adjacency source vertices (size numEdges). */
    long long* d_inWeight  = nullptr; /**< Device: in-edge weights for the selected objective (size numEdges). */

    GraphSnapshot() = default;
    ~GraphSnapshot();

    GraphSnapshot(const GraphSnapshot&)            = delete;
    GraphSnapshot& operator=(const GraphSnapshot&) = delete;

    GraphSnapshot(GraphSnapshot&& other) noexcept;
    GraphSnapshot& operator=(GraphSnapshot&& other) noexcept;

private:
    void freeDevice_();
};

} // namespace escher_mosp

#endif // ESCHER_MOSP_GRAPH_SNAPSHOT_HPP
