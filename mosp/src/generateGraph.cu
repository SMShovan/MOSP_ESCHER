/**
 * @file generateGraph.cu
 * @brief Generate a connected graph and write it to a Matrix Market file.
 */

#include "generateGraph.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace {
uint64_t edgeKey(int u, int v) {
    return (static_cast<uint64_t>(u) << 32) ^ static_cast<uint32_t>(v);
}

bool addEdgeIfNew(
    int u,
    int v,
    bool directed,
    unordered_set<uint64_t> &edgeSet,
    vector<pair<int, int>> &edges
) {
    if (!directed && u > v) {
        swap(u, v);
    }

    uint64_t key = edgeKey(u, v);
    if (edgeSet.find(key) != edgeSet.end()) {
        return false;
    }

    edgeSet.insert(key);
    edges.push_back({u, v});
    return true;
}
} // namespace

/**
 * @brief Generate a connected graph and write it to a Matrix Market file.
 *
 * The graph uses zero-indexed vertex IDs [0, numberOfNodes - 1] and includes
 * multiple objective weights per edge when numberOfObjectives > 1.
 *
 * @param numberOfNodes Number of vertices in the graph (must be > 0).
 * @param numberOfEdges Number of edges to generate (must be > numberOfNodes - 1).
 * @param directed Whether the graph is directed.
 * @param outputFile Output filename (e.g., "data/graph.mtx").
 * @param numberOfObjectives Number of objectives (edge weights per edge).
 * @param objectiveStartRange Minimum objective value (inclusive).
 * @param objectiveEndRange Maximum objective value (inclusive).
 * @return True if the graph was generated and written successfully; false otherwise.
 */
bool generateGraph(
    int numberOfNodes,
    int numberOfEdges,
    bool directed,
    const string &outputFile,
    int numberOfObjectives,
    int objectiveStartRange,
    int objectiveEndRange
) {
    if (numberOfNodes <= 0 || numberOfObjectives <= 0) {
        cout << "Unable to create a connected graph\n";
        return false;
    }

    if (numberOfEdges <= numberOfNodes - 1) {
        cout << "Unable to create a connected graph\n";
        return false;
    }

    int64_t maxEdges = directed
        ? static_cast<int64_t>(numberOfNodes) * (numberOfNodes - 1)
        : (static_cast<int64_t>(numberOfNodes) * (numberOfNodes - 1)) / 2;
    if (numberOfEdges > maxEdges) {
        cout << "Unable to create a connected graph\n";
        return false;
    }

    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> nodeDist(0, numberOfNodes - 1);
    uniform_int_distribution<int> weightDist(objectiveStartRange, objectiveEndRange);
    uniform_int_distribution<int> dirDist(0, 1);

    vector<int> nodes(numberOfNodes);
    for (int i = 0; i < numberOfNodes; ++i) {
        nodes[i] = i;
    }
    shuffle(nodes.begin(), nodes.end(), rng);

    unordered_set<uint64_t> edgeSet;
    vector<pair<int, int>> edges;
    edges.reserve(numberOfEdges);
    edgeSet.reserve(static_cast<size_t>(numberOfEdges * 2));

    // Ensure connectivity with a spanning chain.
    for (int i = 0; i < numberOfNodes - 1; ++i) {
        int u = nodes[i];
        int v = nodes[i + 1];
        if (directed && dirDist(rng) == 1) {
            swap(u, v);
        }
        addEdgeIfNew(u, v, directed, edgeSet, edges);
    }

    // Add remaining random edges.
    while (static_cast<int>(edges.size()) < numberOfEdges) {
        int u = nodeDist(rng);
        int v = nodeDist(rng);
        if (u == v) {
            continue;
        }
        if (directed && dirDist(rng) == 1) {
            swap(u, v);
        }
        addEdgeIfNew(u, v, directed, edgeSet, edges);
    }

    ofstream out(outputFile);
    if (!out.is_open()) {
        cout << "Error: Could not open output file.\n";
        return false;
    }

    out << "%%MatrixMarket matrix coordinate integer general\n";
    out << numberOfNodes << " " << numberOfNodes << " " << numberOfEdges << "\n";
    for (const auto &edge : edges) {
        out << edge.first << " " << edge.second;
        for (int i = 0; i < numberOfObjectives; ++i) {
            out << " " << weightDist(rng);
        }
        out << "\n";
    }

    return true;
}
