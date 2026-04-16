/**
 * @file generateChangedEdges.cu
 * @brief Generate insert/delete edge change sets for a graph.
 */

#include "generateChangedEdges.cuh"

#include "read.cuh"

#include <algorithm>
#include <cmath>
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

bool isEdgeAllowed(int u, int v, bool selfLoop) {
    if (!selfLoop && u == v) {
        return false;
    }
    return true;
}

int maxUniqueEdges(int numberOfNodes, bool directed, bool selfLoop) {
    if (directed) {
        int64_t total = static_cast<int64_t>(numberOfNodes) * numberOfNodes;
        if (!selfLoop) {
            total -= numberOfNodes;
        }
        return static_cast<int>(total);
    }

    int64_t total = static_cast<int64_t>(numberOfNodes) * (numberOfNodes - 1) / 2;
    if (selfLoop) {
        total += numberOfNodes;
    }
    return static_cast<int>(total);
}

bool generateRandomEdges(
    int count,
    int numberOfNodes,
    bool directed,
    bool duplicate,
    bool selfLoop,
    mt19937 &rng,
    vector<pair<int, int>> &edges
) {
    edges.clear();
    edges.reserve(count);

    if (!duplicate && count > maxUniqueEdges(numberOfNodes, directed, selfLoop)) {
        cout << "Error: Not enough unique edges available.\n";
        return false;
    }

    uniform_int_distribution<int> nodeDist(0, numberOfNodes - 1);
    unordered_set<uint64_t> used;
    if (!duplicate) {
        used.reserve(static_cast<size_t>(count * 2));
    }

    while (static_cast<int>(edges.size()) < count) {
        int u = nodeDist(rng);
        int v = nodeDist(rng);
        if (!isEdgeAllowed(u, v, selfLoop)) {
            continue;
        }

        auto e = normalizeEdge(u, v, directed);
        if (!duplicate) {
            uint64_t key = edgeKey(e.first, e.second);
            if (used.find(key) != used.end()) {
                continue;
            }
            used.insert(key);
        }
        edges.push_back(e);
    }

    return true;
}

} // namespace

/**
 * @brief Generate edge update files for dynamic graph changes.
 *
 * @details
 * Creates `output/changedEdges/insert.txt` and `output/changedEdges/delete.txt`.
 * The insert file contains edge endpoints plus objective weights:
 * `u v w1 ... wK`.
 * The delete file contains only edge endpoints: `u v`.
 * If `exist=true`, deletion edges are sampled from the existing CSR graph
 * identified by `csrPrefix`.
 *
 * @param objectiveStartRange Minimum objective value (inclusive).
 * @param objectiveEndRange Maximum objective value (inclusive).
 * @param numberOfObjectives Number of objective values per inserted edge.
 * @param numberOfNodes Number of vertices in the original graph.
 * @param numberOfChangedEdges Total number of edge changes to generate.
 * @param insertionPercentage Percentage of insert operations.
 * @param deletionPercentage Percentage of delete operations.
 * @param directed If false, edges are normalized as undirected pairs.
 * @param exist If true, deletion edges are taken from existing CSR edges.
 * @param duplicate If false, duplicate generated edges are disallowed.
 * @param selfLoop If false, self-loop edges are disallowed.
 * @param csrPrefix CSR input prefix used when `exist=true`.
 * @param insertOutputPath Path to write insert.txt.
 * @param deleteOutputPath Path to write delete.txt.
 * @param seed RNG seed. 0 means non-deterministic.
 * @return True when generation and file writing succeed; false otherwise.
 */
bool generateChangedEdges(
    int objectiveStartRange,
    int objectiveEndRange,
    int numberOfObjectives,
    int numberOfNodes,
    int numberOfChangedEdges,
    double insertionPercentage,
    double deletionPercentage,
    bool directed,
    bool exist,
    bool duplicate,
    bool selfLoop,
    const string &csrPrefix,
    const string &insertOutputPath,
    const string &deleteOutputPath,
    unsigned int seed
) {
    if (numberOfNodes <= 0 || numberOfObjectives <= 0 || numberOfChangedEdges < 0) {
        cout << "Error: Invalid parameters.\n";
        return false;
    }
    if (objectiveStartRange > objectiveEndRange) {
        cout << "Error: objectiveStartRange cannot be greater than objectiveEndRange.\n";
        return false;
    }

    double totalPercentage = insertionPercentage + deletionPercentage;
    if (totalPercentage <= 0.0) {
        cout << "Error: insertionPercentage + deletionPercentage must be > 0.\n";
        return false;
    }

    int insertCount = static_cast<int>(llround(
        static_cast<double>(numberOfChangedEdges) * insertionPercentage / totalPercentage
    ));
    if (insertCount < 0) {
        insertCount = 0;
    }
    if (insertCount > numberOfChangedEdges) {
        insertCount = numberOfChangedEdges;
    }
    int deleteCount = numberOfChangedEdges - insertCount;

    mt19937 rng(seed == 0 ? random_device{}() : seed);
    uniform_int_distribution<int> weightDist(objectiveStartRange, objectiveEndRange);

    vector<pair<int, int>> insertEdges;
    if (!generateRandomEdges(
            insertCount,
            numberOfNodes,
            directed,
            duplicate,
            selfLoop,
            rng,
            insertEdges)) {
        return false;
    }

    vector<pair<int, int>> deleteEdges;
    if (!exist) {
        if (!generateRandomEdges(
                deleteCount,
                numberOfNodes,
                directed,
                duplicate,
                selfLoop,
                rng,
                deleteEdges)) {
            return false;
        }
    } else {
        Graph graph;
        int objectiveCountFromCsr = 0;
        if (!readCSR(csrPrefix, graph, objectiveCountFromCsr)) {
            cout << "Error: Could not read existing CSR graph for deletions.\n";
            return false;
        }

        vector<pair<int, int>> existingEdges;
        existingEdges.reserve(1024);
        unordered_set<uint64_t> seen;
        seen.reserve(2048);

        for (int u = 0; u < static_cast<int>(graph.size()); ++u) {
            for (const auto &edge : graph[u]) {
                auto e = normalizeEdge(u, edge.to, directed);
                if (!isEdgeAllowed(e.first, e.second, selfLoop)) {
                    continue;
                }
                uint64_t key = edgeKey(e.first, e.second);
                if (seen.find(key) != seen.end()) {
                    continue;
                }
                seen.insert(key);
                existingEdges.push_back(e);
            }
        }

        if (existingEdges.empty()) {
            cout << "Error: No existing edges available for deletions.\n";
            return false;
        }

        if (!duplicate && deleteCount > static_cast<int>(existingEdges.size())) {
            cout << "Error: Not enough existing unique edges for deletions.\n";
            return false;
        }

        if (duplicate) {
            uniform_int_distribution<int> edgeDist(0, static_cast<int>(existingEdges.size()) - 1);
            deleteEdges.reserve(deleteCount);
            for (int i = 0; i < deleteCount; ++i) {
                deleteEdges.push_back(existingEdges[edgeDist(rng)]);
            }
        } else {
            shuffle(existingEdges.begin(), existingEdges.end(), rng);
            deleteEdges.assign(existingEdges.begin(), existingEdges.begin() + deleteCount);
        }
    }

    filesystem::path insParent = filesystem::path(insertOutputPath).parent_path();
    filesystem::path delParent = filesystem::path(deleteOutputPath).parent_path();
    if (!insParent.empty()) filesystem::create_directories(insParent);
    if (!delParent.empty()) filesystem::create_directories(delParent);

    ofstream insertFile(insertOutputPath);
    ofstream deleteFile(deleteOutputPath);
    if (!insertFile.is_open() || !deleteFile.is_open()) {
        cout << "Error: Could not open change-edge output files for writing.\n";
        return false;
    }

    for (const auto &e : insertEdges) {
        insertFile << e.first << " " << e.second;
        for (int i = 0; i < numberOfObjectives; ++i) {
            insertFile << " " << weightDist(rng);
        }
        insertFile << "\n";
    }

    for (const auto &e : deleteEdges) {
        deleteFile << e.first << " " << e.second << "\n";
    }

    return true;
}
