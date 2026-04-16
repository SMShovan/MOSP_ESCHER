/**
 * @file updateGraphCSR.cu
 * @brief Apply insert/delete changes to CSR graph and write updated CSR.
 */

#include "updateGraphCSR.cuh"

#include "read.cuh"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {

uint64_t edgeKey(int u, int v) {
    return (static_cast<uint64_t>(u) << 32) ^ static_cast<uint32_t>(v);
}

pair<int, int> normalizeEdge(int u, int v, bool directed) {
    if (!directed && u > v) {
        swap(u, v);
    }
    return {u, v};
}

vector<int> parseInts(const string &line) {
    stringstream ss(line);
    vector<int> tokens;
    int value = 0;
    while (ss >> value) {
        tokens.push_back(value);
    }
    return tokens;
}

bool writeCSRFromMap(
    const unordered_map<uint64_t, vector<int>> &edgeWeights,
    int numberOfNodes,
    int numberOfObjectives,
    const string &updatedPrefix,
    bool directed
) {
    struct WeightedEdge {
        int u;
        int v;
        vector<int> weights;
    };

    vector<WeightedEdge> edges;
    edges.reserve(edgeWeights.size() * (directed ? 1U : 2U));

    for (const auto &entry : edgeWeights) {
        int u = static_cast<int>(entry.first >> 32);
        int v = static_cast<int>(entry.first & 0xffffffffU);
        edges.push_back({u, v, entry.second});
        if (!directed && u != v) {
            edges.push_back({v, u, entry.second});
        }
    }

    sort(edges.begin(), edges.end(), [](const WeightedEdge &a, const WeightedEdge &b) {
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    });

    vector<int> rowPtr(numberOfNodes + 1, 0);
    for (const auto &e : edges) {
        if (e.u < 0 || e.u >= numberOfNodes || e.v < 0 || e.v >= numberOfNodes) {
            cout << "Error: Edge endpoint out of range while writing CSR.\n";
            return false;
        }
        rowPtr[e.u + 1]++;
    }
    for (int i = 0; i < numberOfNodes; ++i) {
        rowPtr[i + 1] += rowPtr[i];
    }

    vector<int> colInd(edges.size(), 0);
    vector<vector<int>> values(edges.size(), vector<int>(numberOfObjectives, 0));
    vector<int> offset(numberOfNodes, 0);
    for (const auto &e : edges) {
        int idx = rowPtr[e.u] + offset[e.u]++;
        colInd[idx] = e.v;
        values[idx] = e.weights;
    }

    filesystem::create_directories(filesystem::path(updatedPrefix).parent_path());
    ofstream rowFile(updatedPrefix + "RowPtr.txt");
    ofstream colFile(updatedPrefix + "ColInd.txt");
    ofstream valFile(updatedPrefix + "Values.txt");
    if (!rowFile.is_open() || !colFile.is_open() || !valFile.is_open()) {
        cout << "Error: Could not write updated CSR files.\n";
        return false;
    }

    for (int x : rowPtr) {
        rowFile << x << "\n";
    }
    for (int x : colInd) {
        colFile << x << "\n";
    }
    for (const auto &w : values) {
        for (int i = 0; i < numberOfObjectives; ++i) {
            valFile << w[i] << (i + 1 < numberOfObjectives ? " " : "");
        }
        valFile << "\n";
    }

    return true;
}

} // namespace

/**
 * @brief Update a CSR graph by applying edge deletions and insertions.
 *
 * @details
 * Reads original CSR files identified by `originalPrefix`, then:
 * 1) applies deletions from `deletePath` (ignores non-existing edges),
 * 2) applies insertions from `insertPath` (overwrites weights if edge exists),
 * and finally writes the updated CSR graph to `updatedPrefix`.
 *
 * @param originalPrefix Prefix for original CSR files (e.g. "data/originalGraph/graphCsr").
 * @param updatedPrefix Prefix for updated CSR files (e.g. "data/updatedGraph/updatedGraphCsr").
 * @param insertPath Path to insert file lines: `u v w1 ... wK`.
 * @param deletePath Path to delete file lines: `u v`.
 * @param directed Direction mode used for edge identity.
 * @return True if update succeeds; false otherwise.
 */
bool updateGraphCSR(
    const string &originalPrefix,
    const string &updatedPrefix,
    const string &insertPath,
    const string &deletePath,
    bool directed
) {
    Graph graph;
    int numberOfObjectives = 0;
    if (!readCSR(originalPrefix, graph, numberOfObjectives)) {
        cout << "Error: Could not read original CSR graph.\n";
        return false;
    }
    if (numberOfObjectives <= 0) {
        cout << "Error: Invalid objective count in original CSR graph.\n";
        return false;
    }

    int numberOfNodes = static_cast<int>(graph.size());
    unordered_map<uint64_t, vector<int>> edgeWeights;
    edgeWeights.reserve(2048);

    for (int u = 0; u < numberOfNodes; ++u) {
        for (const auto &edge : graph[u]) {
            auto normalized = normalizeEdge(u, edge.to, directed);
            edgeWeights[edgeKey(normalized.first, normalized.second)] = edge.weights;
        }
    }

    ifstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
        cout << "Error: Could not open delete file.\n";
        return false;
    }
    string line;
    while (getline(deleteFile, line)) {
        if (line.empty()) {
            continue;
        }
        vector<int> tokens = parseInts(line);
        if (tokens.size() < 2) {
            continue;
        }
        int u = tokens[0];
        int v = tokens[1];
        if (u < 0 || u >= numberOfNodes || v < 0 || v >= numberOfNodes) {
            continue;
        }
        auto normalized = normalizeEdge(u, v, directed);
        edgeWeights.erase(edgeKey(normalized.first, normalized.second));
    }

    ifstream insertFile(insertPath);
    if (!insertFile.is_open()) {
        cout << "Error: Could not open insert file.\n";
        return false;
    }
    while (getline(insertFile, line)) {
        if (line.empty()) {
            continue;
        }
        vector<int> tokens = parseInts(line);
        if (static_cast<int>(tokens.size()) != numberOfObjectives + 2) {
            cout << "Error: Invalid insert line objective count.\n";
            return false;
        }
        int u = tokens[0];
        int v = tokens[1];
        if (u < 0 || u >= numberOfNodes || v < 0 || v >= numberOfNodes) {
            continue;
        }
        vector<int> weights(tokens.begin() + 2, tokens.end());
        auto normalized = normalizeEdge(u, v, directed);
        edgeWeights[edgeKey(normalized.first, normalized.second)] = weights;
    }

    return writeCSRFromMap(
        edgeWeights,
        numberOfNodes,
        numberOfObjectives,
        updatedPrefix,
        directed
    );
}
