#ifndef ESCHER_MOSP_UPDATE_GRAPH_WITH_ESCHER_HPP
#define ESCHER_MOSP_UPDATE_GRAPH_WITH_ESCHER_HPP

#include <string>

namespace escher_mosp {

/**
 * @brief Drop-in replacement for MOSP's @c updateGraphCSR, routed through ESCHER.
 *
 * Reads the original CSR graph, loads it into a @ref DynamicGraph, applies
 * deletions from @c deletePath and upsert insertions from @c insertPath via
 * @c DynamicGraph::deleteEdges and @c DynamicGraph::insertEdges (which
 * ultimately call @c CBSTOperations::erase / @c ::insert / @c ::fill /
 * @c unfillCBST), and finally writes the updated graph back in CSR format.
 *
 * The on-disk output is byte-for-byte equivalent to what @c updateGraphCSR
 * would produce for the same inputs when @c directed is @c true. This is the
 * correctness contract validated by @c test_snapshot_matches_updateCSR.
 *
 * Matching @c updateGraphCSR's permissive behavior:
 *   - Deletes of nonexistent edges are silently ignored.
 *   - Inserts with out-of-range vertices are silently skipped.
 *   - Inserts of existing edges overwrite the weights (upsert).
 *
 * @param originalPrefix   Prefix of original CSR files (e.g. @c data/originalGraph/graphCsr).
 * @param updatedPrefix    Prefix of updated CSR files to write.
 * @param insertPath       Path to @c insert.txt (lines @c "u v w1 w2 ... wK").
 * @param deletePath       Path to @c delete.txt (lines @c "u v").
 * @param payloadCapacity  Per-CBST flat payload capacity; size this above
 *                         @c (2+K) * (numEdges + expectedInserts).
 * @param directed         Only @c true is supported in the initial implementation.
 *                         @c false throws @c EscherError.
 * @return True on success; false if the original CSR could not be read.
 *
 * @throws escher::EscherError on invalid inputs or CUDA failure.
 */
bool updateGraphWithESCHER(const std::string& originalPrefix,
                           const std::string& updatedPrefix,
                           const std::string& insertPath,
                           const std::string& deletePath,
                           int  payloadCapacity = 65536,
                           bool directed        = true);

} // namespace escher_mosp

#endif // ESCHER_MOSP_UPDATE_GRAPH_WITH_ESCHER_HPP
