/**
 * @file updateGraphWithESCHER.cpp
 * @brief Implementation of the ESCHER-backed replacement for @c updateGraphCSR.
 */

#include "updateGraphWithESCHER.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "DynamicGraph.hpp"
#include "escher_errors.hpp"
#include "read.cuh"

namespace escher_mosp {

namespace {

/**
 * @brief Parse a whitespace-separated line of non-negative integers.
 */
std::vector<int> parseInts(const std::string& line) {
    std::stringstream ss(line);
    std::vector<int> tokens;
    int value = 0;
    while (ss >> value) tokens.push_back(value);
    return tokens;
}

/**
 * @brief Write a @c DynamicGraph dump to CSR files under @c prefix.
 *
 * Matches the output file layout of @c updateGraphCSR.cu so downstream
 * consumers (Dijkstra, sequentialSOSPUpdate, parallelSOSPUpdate,
 * parallelCombinedGraph) can read the files without modification.
 */
bool writeCsrFiles(const std::string& prefix,
                   const std::vector<int>& rowPtr,
                   const std::vector<int>& colInd,
                   const std::vector<std::vector<int>>& values,
                   int numberOfObjectives) {
    std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
    std::ofstream rowFile(prefix + "RowPtr.txt");
    std::ofstream colFile(prefix + "ColInd.txt");
    std::ofstream valFile(prefix + "Values.txt");
    if (!rowFile.is_open() || !colFile.is_open() || !valFile.is_open()) {
        std::cout << "Error: Could not write updated CSR files (ESCHER path).\n";
        return false;
    }
    for (int x : rowPtr) rowFile << x << "\n";
    for (int x : colInd) colFile << x << "\n";
    for (const auto& w : values) {
        for (int i = 0; i < numberOfObjectives; ++i) {
            valFile << w[i] << (i + 1 < numberOfObjectives ? " " : "");
        }
        valFile << "\n";
    }
    return true;
}

/**
 * @brief Pack a (u,v) pair for use as an @c unordered_set key.
 */
inline std::int64_t packPair(int u, int v) noexcept {
    return (static_cast<std::int64_t>(u) << 32) | static_cast<std::uint32_t>(v);
}

} // namespace

bool updateGraphWithESCHER(const std::string& originalPrefix,
                           const std::string& updatedPrefix,
                           const std::string& insertPath,
                           const std::string& deletePath,
                           int  payloadCapacity,
                           bool directed) {
    if (!directed) {
        throw escher::EscherError(
            "updateGraphWithESCHER: undirected mode is not yet implemented");
    }

    // ------- 1. Read original CSR via MOSP's existing reader -------
    Graph graph;
    int numberOfObjectives = 0;
    if (!readCSR(originalPrefix, graph, numberOfObjectives)) {
        std::cout << "Error: Could not read original CSR graph (ESCHER path).\n";
        return false;
    }
    if (numberOfObjectives <= 0) {
        std::cout << "Error: Invalid objective count in original CSR graph.\n";
        return false;
    }

    const int V = static_cast<int>(graph.size());
    const int K = numberOfObjectives;

    // Convert MOSP's adjacency-list Graph into the flat CSR triple
    // that DynamicGraph::loadFromCSR expects.
    std::vector<int> rowPtr(V + 1, 0);
    for (int u = 0; u < V; ++u) rowPtr[u + 1] = rowPtr[u] + static_cast<int>(graph[u].size());
    const int E0 = rowPtr.back();
    std::vector<int> colInd(E0);
    std::vector<std::vector<int>> values(E0);
    for (int u = 0; u < V; ++u) {
        int base = rowPtr[u];
        for (std::size_t j = 0; j < graph[u].size(); ++j) {
            colInd[base + static_cast<int>(j)] = graph[u][j].to;
            values[base + static_cast<int>(j)] = graph[u][j].weights;
        }
    }

    // ------- 2. Build the DynamicGraph (every update now goes through ESCHER) -------
    DynamicGraph dg(V, K, payloadCapacity);
    dg.loadFromCSR(rowPtr, colInd, values);

    // ------- 3. Apply deletions first (matches updateGraphCSR.cu ordering) -------
    std::ifstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
        std::cout << "Error: Could not open delete file.\n";
        return false;
    }
    {
        std::vector<EdgeDelete> batch;
        std::string line;
        while (std::getline(deleteFile, line)) {
            if (line.empty()) continue;
            auto toks = parseInts(line);
            if (toks.size() < 2) continue;
            int u = toks[0], v = toks[1];
            if (u < 0 || u >= V || v < 0 || v >= V) continue;
            batch.push_back({u, v});
        }
        dg.deleteEdges(batch);
    }

    // ------- 4. Apply insertions with upsert semantics -------
    //
    // updateGraphCSR.cu overwrites weights when an inserted edge already
    // exists. We emulate that by first deleting any colliding (u,v) pairs,
    // then inserting every line as a fresh edge. The delete-then-insert
    // dance still exercises ESCHER's erase + best-fit-reuse insert path.
    {
        std::ifstream insertFile(insertPath);
        if (!insertFile.is_open()) {
            std::cout << "Error: Could not open insert file.\n";
            return false;
        }

        std::vector<EdgeInsert> inserts;
        std::vector<EdgeDelete> preDeletes;
        std::unordered_set<std::int64_t> seen;
        preDeletes.reserve(64);
        inserts.reserve(64);

        std::string line;
        while (std::getline(insertFile, line)) {
            if (line.empty()) continue;
            auto toks = parseInts(line);
            if (static_cast<int>(toks.size()) != K + 2) {
                std::cout << "Error: Invalid insert line objective count.\n";
                return false;
            }
            int u = toks[0], v = toks[1];
            if (u < 0 || u >= V || v < 0 || v >= V) continue;

            // Pre-delete so upsert doesn't produce a duplicate edge.
            preDeletes.push_back({u, v});

            EdgeInsert ins;
            ins.src = u;
            ins.dst = v;
            ins.weights.assign(toks.begin() + 2, toks.end());
            inserts.push_back(std::move(ins));

            // Guard against duplicate insert lines: only the last one wins,
            // matching the std::unordered_map overwrite in updateGraphCSR.
            seen.insert(packPair(u, v));
        }

        if (!preDeletes.empty()) dg.deleteEdges(preDeletes);

        // Deduplicate: walk inserts back-to-front, keep only the last entry
        // per (u,v) so the final weight matches @c edgeWeights[...] = weights
        // semantics in the legacy updater.
        std::vector<EdgeInsert> dedup;
        dedup.reserve(inserts.size());
        std::unordered_set<std::int64_t> kept;
        for (auto it = inserts.rbegin(); it != inserts.rend(); ++it) {
            auto key = packPair(it->src, it->dst);
            if (kept.insert(key).second) dedup.push_back(std::move(*it));
        }
        std::reverse(dedup.begin(), dedup.end());
        if (!dedup.empty()) dg.insertEdges(dedup);
    }

    // ------- 5. Dump to CSR and write the output files -------
    std::vector<int> outRow, outCol;
    std::vector<std::vector<int>> outVal;
    dg.dumpToCSR(outRow, outCol, outVal);
    return writeCsrFiles(updatedPrefix, outRow, outCol, outVal, K);
}

} // namespace escher_mosp
