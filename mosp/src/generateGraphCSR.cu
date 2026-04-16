/**
 * @file generateGraphCSR.cu
 * @brief Generate a random connected graph and write it in CSR format.
 */

#include "generateGraphCSR.cuh"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

/**
 * @brief Generate a connected graph and write it in CSR format.
 *
 * Creates three files: outputPrefixRowPtr.txt, outputPrefixColInd.txt,
 * outputPrefixValues.txt
 *
 * @param numberOfNodes Number of vertices (must be > 1).
 * @param numberOfEdges Number of edges (must be >= numberOfNodes - 1).
 * @param directed Whether the graph is directed.
 * @param outputPrefix Base path for output files (e.g. "data/originalGraph/graphCsr").
 * @param numberOfObjectives Number of weight values per edge.
 * @param objectiveStartRange Minimum weight (inclusive).
 * @param objectiveEndRange Maximum weight (inclusive).
 * @param seed RNG seed. 0 means non-deterministic (default).
 * @return True on success; false otherwise.
 */
bool generateGraphCSR(
    int numberOfNodes,
    int numberOfEdges,
    bool directed,
    const string &outputPrefix,
    int numberOfObjectives,
    int objectiveStartRange,
    int objectiveEndRange,
    unsigned int seed
) {
    if (numberOfNodes <= 1 || numberOfEdges < numberOfNodes - 1) {
        cout << "Invalid graph parameters\n";
        return false;
    }

    mt19937 rng(seed == 0 ? random_device{}() : seed);
    uniform_int_distribution<int> nodeDist(0, numberOfNodes - 1);
    uniform_int_distribution<int> weightDist(objectiveStartRange, objectiveEndRange);

    // ---------- 1. Generate connected edge list (no duplicates) ----------
    vector<pair<int, int>> edges;
    unordered_set<uint64_t> edgeSet; // tracks existing directed edges

    auto edgeKey = [](int u, int v) -> uint64_t {
        return (static_cast<uint64_t>(u) << 32) ^ static_cast<uint32_t>(v);
    };

    // Spanning chain (guarantees connectivity)
    for (int i = 0; i < numberOfNodes - 1; ++i) {
        edges.push_back({i, i + 1});
        edgeSet.insert(edgeKey(i, i + 1));
        if (!directed) {
            edges.push_back({i + 1, i});
            edgeSet.insert(edgeKey(i + 1, i));
        }
    }

    // Add random edges (skip duplicates)
    while ((int)edges.size() < numberOfEdges) {
        int u = nodeDist(rng);
        int v = nodeDist(rng);
        if (u != v && edgeSet.find(edgeKey(u, v)) == edgeSet.end()) {
            edges.push_back({u, v});
            edgeSet.insert(edgeKey(u, v));
            if (!directed) {
                edges.push_back({v, u});
                edgeSet.insert(edgeKey(v, u));
            }
        }
    }

    sort(edges.begin(), edges.end());

    // ---------- 2. Build CSR ----------
    vector<int> row_ptr(numberOfNodes + 1, 0);
    for (auto &e : edges) {
        row_ptr[e.first + 1]++;
    }
    for (int i = 0; i < numberOfNodes; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    vector<int> col_ind(edges.size());
    vector<vector<int>> values(edges.size(), vector<int>(numberOfObjectives));

    vector<int> offset(numberOfNodes, 0);
    for (size_t i = 0; i < edges.size(); ++i) {
        int u = edges[i].first;
        int v = edges[i].second;
        int idx = row_ptr[u] + offset[u]++;
        col_ind[idx] = v;
        for (int j = 0; j < numberOfObjectives; ++j) {
            values[idx][j] = weightDist(rng);
        }
    }

    // ---------- 3. Write files ----------
    filesystem::create_directories(filesystem::path(outputPrefix).parent_path());

    ofstream out_row(outputPrefix + "RowPtr.txt");
    ofstream out_col(outputPrefix + "ColInd.txt");
    ofstream out_val(outputPrefix + "Values.txt");

    if (!out_row || !out_col || !out_val) {
        cout << "File write error\n";
        return false;
    }

    for (int x : row_ptr) out_row << x << "\n";
    for (int x : col_ind) out_col << x << "\n";
    for (auto &row : values) {
        for (int j = 0; j < numberOfObjectives; ++j) {
            out_val << row[j] << (j + 1 < numberOfObjectives ? " " : "");
        }
        out_val << "\n";
    }

    return true;
}
