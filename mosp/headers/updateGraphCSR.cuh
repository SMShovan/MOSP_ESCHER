#ifndef UPDATE_GRAPH_CSR_CUH
#define UPDATE_GRAPH_CSR_CUH

#include <string>

/**
 * @brief Apply edge deletions and insertions to a CSR graph.
 *
 * The function reads an original CSR graph, applies deletions first, then
 * applies insertions (overwriting weights for existing edges), and writes
 * the updated graph back in CSR format.
 *
 * @param originalPrefix Prefix of original CSR files.
 * @param updatedPrefix Prefix of updated CSR files to write.
 * @param insertPath Path to insert.txt (u v w1 ... wK).
 * @param deletePath Path to delete.txt (u v).
 * @param directed Graph direction mode used for edge-key normalization.
 * @return True on success; false otherwise.
 */
bool updateGraphCSR(
    const std::string &originalPrefix,
    const std::string &updatedPrefix,
    const std::string &insertPath,
    const std::string &deletePath,
    bool directed = true
);

#endif
