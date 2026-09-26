/**
 * @file sequentialSOSPUpdate.cu
 * @brief Sequential Single-Objective Shortest Path (SOSP) Update Algorithm.
 *
 * This file implements the sequential version of the SOSP Update algorithm
 * from "Parallel Multi Objective Shortest Path Update Algorithm in Large
 * Dynamic Networks" (Shovan, Khanda, Das — IEEE TPDS 2025).
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Given a graph G with a pre-computed SOSP tree (distances + parent array)
 * and a batch of edge changes (insertions + deletions), this algorithm
 * incrementally updates the distances and parent array WITHOUT recomputing
 * Dijkstra from scratch.
 *
 * The algorithm has three main phases:
 *
 * --- Phase 0: Preparation ---
 *   - Read original CSR, build forward (outAdjacency) and reverse
 *     (inAdjacency) adjacency lists.
 *   - Read original distances and parent arrays from Dijkstra output.
 *   - Read inserted and deleted edges.
 *   - Apply topological changes to both adjacency lists so they reflect
 *     the updated graph structure.
 *
 * --- Phase 1: Process Changed Edges ---
 *   The head v of every deleted or weight-increased tree edge (u,v)
 *   (parent[v] == u) is a root. The SOSP subtree of every root is
 *   invalidated (distance INF, parent -1): these vertices lost their
 *   shortest paths. The invalidated vertices and the heads of all inserted
 *   edges then take the best (distance, id) pair over their in-neighbours
 *   in the UPDATED graph; the ones whose distance decreased are affected.
 *   (The thesis' Step 1 instead picks the best current in-neighbour of a
 *   root, which may be one of its own descendants; the resulting stale
 *   cycle "counts to infinity" and the original code stopped it with an
 *   iteration cap that could leave reachable vertices with wrong
 *   distances.)
 *
 * --- Phase 2: Propagate the Update ---
 *   Iteratively propagate changes until no more vertices are affected:
 *     1. Collect all out-neighbors of affected vertices as candidates.
 *     2. For each candidate, recompute the best (distance, parent id) pair
 *        over ALL in-neighbors and keep it if it is better (monotone).
 *     3. If the distance decreased, mark the candidate as newly affected.
 *   Distances only decrease, so the loop terminates without an iteration
 *   cap, and vertices that became unreachable keep INF (no reachability
 *   post-pass is needed).
 *
 * ============================================================================
 * NOTES
 * ============================================================================
 *
 * 1. SOURCE VERTEX PROTECTION: The source vertex (distance = 0) is never
 *    updated, even if it appears as a candidate during propagation.
 *
 * 2. TIE-BREAK: among in-neighbours with equal distance the lowest vertex
 *    id becomes the parent, so the result equals the (canonical) Dijkstra
 *    tree of the updated graph.
 *
 * ============================================================================
 */

#include "sequentialSOSPUpdate.cuh"

#include "read.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

/// A lightweight edge structure for the internal adjacency lists,
/// storing only the neighbor vertex and the single objective weight.
struct WeightedNeighbor {
    int vertex;
    long long weight;
};

/**
 * @brief Parse a line of space-separated integers from a string.
 * @param line The input string.
 * @return Vector of parsed integer tokens.
 */
vector<int> parseIntTokens(const string &line) {
    vector<int> tokens;
    istringstream stream(line);
    int value;
    while (stream >> value) {
        tokens.push_back(value);
    }
    return tokens;
}

/**
 * @brief Build forward and reverse adjacency lists from a Graph (read via CSR).
 *
 * For each edge (u -> v) with weights[], extracts the objectiveIndex-th weight
 * and adds it to both the forward list (outAdjacency[u]) and the reverse list
 * (inAdjacency[v]).
 *
 * @param graph           The graph read from CSR files.
 * @param objectiveIndex  Which objective weight to extract.
 * @param outAdjacency    Output: forward adjacency (out-edges per vertex).
 * @param inAdjacency     Output: reverse adjacency (in-edges per vertex).
 */
void buildAdjacencyLists(
    const Graph &graph,
    int objectiveIndex,
    vector<vector<WeightedNeighbor>> &outAdjacency,
    vector<vector<WeightedNeighbor>> &inAdjacency
) {
    int numberOfNodes = static_cast<int>(graph.size());
    outAdjacency.assign(numberOfNodes, {});
    inAdjacency.assign(numberOfNodes, {});

    for (int u = 0; u < numberOfNodes; ++u) {
        for (const auto &edge : graph[u]) {
            int v = edge.to;
            long long w = edge.weights[objectiveIndex];
            outAdjacency[u].push_back({v, w});
            inAdjacency[v].push_back({u, w});
        }
    }
}

/**
 * @brief Remove a specific directed edge (fromVertex -> toVertex) from an
 *        adjacency list entry.
 *
 * Removes the FIRST occurrence of the target vertex from the neighbor list.
 *
 * @param neighbors   The adjacency list entry to modify.
 * @param targetVertex The vertex to remove from the list.
 */
void removeEdgeFromList(vector<WeightedNeighbor> &neighbors, int targetVertex) {
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
        if (it->vertex == targetVertex) {
            neighbors.erase(it);
            return;
        }
    }
}

/**
 * @brief Read distances from a Dijkstra output file.
 *
 * Expected format per line: "vertexId distance" where distance is either
 * a long long value or the string "INF".
 *
 * @param path           Path to the distances file.
 * @param distances      Output vector of distances (indexed by vertex).
 * @param numberOfNodes  Expected number of vertices.
 * @param INF_VALUE      The sentinel value used for unreachable vertices.
 * @return True on success; false otherwise.
 */
bool readDistancesFromFile(
    const string &path,
    vector<long long> &distances,
    int numberOfNodes,
    long long INF_VALUE
) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "Error: Could not open distances file: " << path << "\n";
        return false;
    }

    distances.assign(numberOfNodes, INF_VALUE);

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream stream(line);
        int vertexId;
        string distanceStr;
        stream >> vertexId >> distanceStr;

        if (vertexId < 0 || vertexId >= numberOfNodes) {
            cout << "Error: Vertex ID out of range in distances file.\n";
            return false;
        }

        if (distanceStr == "INF") {
            distances[vertexId] = INF_VALUE;
        } else {
            distances[vertexId] = stoll(distanceStr);
        }
    }

    return true;
}

/**
 * @brief Read SSSP tree (parent array) from a Dijkstra output file.
 *
 * Expected format per line: "vertexId parentId".
 *
 * @param path           Path to the SSSP tree file.
 * @param parent         Output vector of parent IDs (indexed by vertex).
 * @param numberOfNodes  Expected number of vertices.
 * @return True on success; false otherwise.
 */
bool readParentFromFile(
    const string &path,
    vector<int> &parent,
    int numberOfNodes
) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "Error: Could not open SSSP tree file: " << path << "\n";
        return false;
    }

    parent.assign(numberOfNodes, -1);

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream stream(line);
        int vertexId, parentId;
        stream >> vertexId >> parentId;

        if (vertexId < 0 || vertexId >= numberOfNodes) {
            cout << "Error: Vertex ID out of range in SSSP tree file.\n";
            return false;
        }

        parent[vertexId] = parentId;
    }

    return true;
}

/**
 * @brief Find the in-neighbor that gives the minimum distance to a vertex.
 *
 * Searches all in-neighbors of the given vertex and returns the one that
 * minimizes (dist[inNeighbor] + edgeWeight).
 *
 * @param vertex         The vertex whose best parent we seek.
 * @param inAdjacency    Reverse adjacency list.
 * @param distances      Current distance array.
 * @param INF_VALUE      Sentinel for unreachable vertices.
 * @param[out] bestParent   The in-neighbor giving shortest distance (-1 if none).
 * @param[out] bestDistance  The shortest achievable distance (INF if none).
 */
void findBestParent(
    int vertex,
    const vector<vector<WeightedNeighbor>> &inAdjacency,
    const vector<long long> &distances,
    long long INF_VALUE,
    int &bestParent,
    long long &bestDistance
) {
    bestParent = -1;
    bestDistance = INF_VALUE;

    for (const auto &inNeighbor : inAdjacency[vertex]) {
        int candidateParent = inNeighbor.vertex;
        long long candidateWeight = inNeighbor.weight;

        // Skip unreachable in-neighbors to avoid overflow
        if (distances[candidateParent] >= INF_VALUE / 2) {
            continue;
        }

        long long candidateDistance = distances[candidateParent] + candidateWeight;
        // Ties go to the lowest parent id (canonical SOSP tree).
        if (candidateDistance < bestDistance ||
            (candidateDistance == bestDistance && candidateParent < bestParent)) {
            bestDistance = candidateDistance;
            bestParent = candidateParent;
        }
    }
}

} // namespace

/**
 * @brief Run the sequential SOSP Update algorithm.
 *
 * @see sequentialSOSPUpdate.cuh for full parameter documentation.
 */
bool sequentialSOSPUpdate(
    const string &originalCsrPrefix,
    const string &distancesInputPath,
    const string &treeInputPath,
    const string &insertPath,
    const string &deletePath,
    int objectiveIndex,
    int source,
    const string &distancesOutputPath,
    const string &treeOutputPath
) {
    const long long INF_VALUE = numeric_limits<long long>::max() / 4;

    // ========================================================================
    // PHASE 0: PREPARATION
    // ========================================================================

    // --- 0a. Read original graph from CSR and determine dimensions ---
    Graph originalGraph;
    int numberOfObjectives = 0;
    if (!readCSR(originalCsrPrefix, originalGraph, numberOfObjectives)) {
        cout << "Error: Could not read original CSR graph.\n";
        return false;
    }

    int numberOfNodes = static_cast<int>(originalGraph.size());
    if (numberOfNodes == 0) {
        cout << "Error: Graph has no vertices.\n";
        return false;
    }

    if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
        cout << "Error: objectiveIndex out of range.\n";
        return false;
    }

    if (source < 0 || source >= numberOfNodes) {
        cout << "Error: source vertex out of range.\n";
        return false;
    }

    // --- 0b. Build forward and reverse adjacency lists ---
    // outAdjacency[u] = list of {v, weight} for edges u -> v
    // inAdjacency[v]  = list of {u, weight} for edges u -> v
    vector<vector<WeightedNeighbor>> outAdjacency;
    vector<vector<WeightedNeighbor>> inAdjacency;
    buildAdjacencyLists(originalGraph, objectiveIndex, outAdjacency, inAdjacency);

    // Free the original graph data since we now work with adjacency lists
    originalGraph.clear();

    // --- 0c. Read original distances and parent arrays ---
    vector<long long> distances;
    if (!readDistancesFromFile(distancesInputPath, distances, numberOfNodes, INF_VALUE)) {
        return false;
    }

    vector<int> parent;
    if (!readParentFromFile(treeInputPath, parent, numberOfNodes)) {
        return false;
    }

    // --- 0d. Read inserted and deleted edges ---
    struct InsertedEdge {
        int from;
        int to;
        long long weight; // only the objectiveIndex-th weight
    };

    struct DeletedEdge {
        int from;
        int to;
    };

    vector<InsertedEdge> insertedEdges;
    {
        ifstream insertFile(insertPath);
        if (!insertFile.is_open()) {
            cout << "Error: Could not open insert file: " << insertPath << "\n";
            return false;
        }
        string line;
        while (getline(insertFile, line)) {
            if (line.empty()) continue;
            vector<int> tokens = parseIntTokens(line);
            // tokens: u v w1 w2 ... wK
            if (static_cast<int>(tokens.size()) < 2 + numberOfObjectives) {
                cout << "Error: Invalid insert line.\n";
                return false;
            }
            int u = tokens[0];
            int v = tokens[1];
            long long w = tokens[2 + objectiveIndex];
            insertedEdges.push_back({u, v, w});
        }
    }

    vector<DeletedEdge> deletedEdges;
    {
        ifstream deleteFile(deletePath);
        if (!deleteFile.is_open()) {
            cout << "Error: Could not open delete file: " << deletePath << "\n";
            return false;
        }
        string line;
        while (getline(deleteFile, line)) {
            if (line.empty()) continue;
            vector<int> tokens = parseIntTokens(line);
            if (tokens.size() < 2) continue;
            int u = tokens[0];
            int v = tokens[1];
            deletedEdges.push_back({u, v});
        }
    }

    // --- 0e. Apply topological changes to adjacency lists ---
    // Track which insertions overwrote an existing edge with a HIGHER weight.
    // These "weight increases" on tree edges must be treated like deletions
    // because the previous shortest path through that edge is no longer valid.
    struct WeightIncrease {
        int from;
        int to;
    };
    vector<WeightIncrease> weightIncreases;

    // Deletions first (remove edges from both forward and reverse lists)
    for (const auto &edge : deletedEdges) {
        removeEdgeFromList(outAdjacency[edge.from], edge.to);
        removeEdgeFromList(inAdjacency[edge.to], edge.from);
    }

    // Then insertions: REPLACE if edge already exists, otherwise add.
    // This matches updateGraphCSR behavior where inserting an existing edge
    // overwrites its weight.
    for (const auto &edge : insertedEdges) {
        bool replacedOut = false;
        for (auto &neighbor : outAdjacency[edge.from]) {
            if (neighbor.vertex == edge.to) {
                if (edge.weight > neighbor.weight) {
                    weightIncreases.push_back({edge.from, edge.to});
                }
                neighbor.weight = edge.weight;
                replacedOut = true;
                break;
            }
        }
        if (!replacedOut) {
            outAdjacency[edge.from].push_back({edge.to, edge.weight});
        }

        bool replacedIn = false;
        for (auto &neighbor : inAdjacency[edge.to]) {
            if (neighbor.vertex == edge.from) {
                neighbor.weight = edge.weight;
                replacedIn = true;
                break;
            }
        }
        if (!replacedIn) {
            inAdjacency[edge.to].push_back({edge.from, edge.weight});
        }
    }

    // ========================================================================
    // PHASE 1: PROCESS CHANGED EDGES
    // ========================================================================
    // 1a. Roots: the head v of every deleted or weight-increased edge (u,v)
    //     that is the tree edge of v (parent[v] == u).
    // 1b. Invalidate the SOSP subtree of every root (distance INF, parent
    //     -1): each of these vertices lost its shortest path. Choosing a new
    //     parent for a root among its current in-neighbours, as the thesis'
    //     Step 1 does, can pick one of its own descendants and start a stale
    //     cycle that only "counts to infinity" by the cycle weight per round.
    // 1c. The invalidated vertices and the heads of all inserted edges are
    //     re-evaluated over their in-neighbours (monotone: a vertex only
    //     takes a strictly better (distance, parent id) pair).

    vector<bool> isInvalid(numberOfNodes, false);
    vector<int> roots;
    auto markRoot = [&](int u, int v) {
        if (parent[v] == u && !isInvalid[v]) {
            isInvalid[v] = true;
            roots.push_back(v);
        }
    };
    for (const auto &edge : deletedEdges) {
        markRoot(edge.from, edge.to);
    }
    for (const auto &wi : weightIncreases) {
        markRoot(wi.from, wi.to);
    }

    {
        // Children lists of the SOSP tree, then a traversal from the roots.
        vector<int> childStart(numberOfNodes + 1, 0), children;
        for (int v = 0; v < numberOfNodes; ++v) {
            if (v != source && parent[v] >= 0) {
                ++childStart[parent[v] + 1];
            }
        }
        for (int v = 0; v < numberOfNodes; ++v) {
            childStart[v + 1] += childStart[v];
        }
        children.resize(childStart[numberOfNodes]);
        vector<int> cursor(childStart.begin(), childStart.end() - 1);
        for (int v = 0; v < numberOfNodes; ++v) {
            if (v != source && parent[v] >= 0) {
                children[cursor[parent[v]]++] = v;
            }
        }
        vector<int> invalidated = roots;
        for (size_t i = 0; i < invalidated.size(); ++i) {
            int x = invalidated[i];
            for (int c = childStart[x]; c < childStart[x + 1]; ++c) {
                if (!isInvalid[children[c]]) {
                    isInvalid[children[c]] = true;
                    invalidated.push_back(children[c]);
                }
            }
        }
        for (int v : invalidated) {
            distances[v] = INF_VALUE;
            parent[v] = -1;
        }
        roots.swap(invalidated);
    }

    vector<bool> isAffected(numberOfNodes, false);
    vector<int> affectedVertices;

    // Re-evaluate one vertex; returns true if its distance decreased.
    auto relax = [&](int v) {
        int bestNewParent = -1;
        long long bestNewDistance = INF_VALUE;
        findBestParent(v, inAdjacency, distances, INF_VALUE, bestNewParent,
                       bestNewDistance);
        bool better = bestNewDistance < distances[v] ||
                      (bestNewDistance == distances[v] && bestNewParent >= 0 &&
                       bestNewParent < parent[v]);
        if (!better) {
            return false;
        }
        bool decreased = bestNewDistance < distances[v];
        distances[v] = bestNewDistance;
        parent[v] = bestNewParent;
        return decreased;
    };
    auto markAffected = [&](int v) {
        if (!isAffected[v]) {
            isAffected[v] = true;
            affectedVertices.push_back(v);
        }
    };

    for (int v : roots) {
        if (relax(v)) {
            markAffected(v);
        }
    }
    for (const auto &edge : insertedEdges) {
        int v = edge.to;
        if (v != source && relax(v)) {
            markAffected(v);
        }
    }

    // ========================================================================
    // PHASE 2: PROPAGATE THE UPDATE
    // ========================================================================
    // Iteratively propagate changes through the graph until convergence.
    // Each iteration:
    //   (a) Collect out-neighbors of all currently affected vertices as candidates.
    //   (b) For each candidate, recompute the best distance from all in-neighbors
    //       and keep it if it is better (monotone update).
    //   (c) If the distance decreased, mark the candidate as affected.
    // Distances only decrease, so the loop terminates, and a vertex that is
    // no longer reachable from the source keeps the INF it got in Phase 1.

    int iterationCount = 0;

    while (!affectedVertices.empty()) {
        ++iterationCount;
        if (iterationCount > numberOfNodes) {
            // Every sweep settles at least one more hop of every shortest
            // path, so this cannot happen.
            cout << "Error: SOSP update did not converge.\n";
            return false;
        }

        // --- 2a. Identify candidate vertices (out-neighbors of affected) ---
        vector<bool> isCandidate(numberOfNodes, false);
        vector<int> candidateVertices;

        for (int affectedVertex : affectedVertices) {
            isAffected[affectedVertex] = false; // Clear affected flag

            for (const auto &outNeighbor : outAdjacency[affectedVertex]) {
                int neighborVertex = outNeighbor.vertex;

                // CRITICAL: Never update the source vertex
                if (neighborVertex == source) {
                    continue;
                }

                if (!isCandidate[neighborVertex]) {
                    isCandidate[neighborVertex] = true;
                    candidateVertices.push_back(neighborVertex);
                }
            }
        }

        affectedVertices.clear();

        // --- 2b. Update distances of candidate vertices ---
        for (int candidateVertex : candidateVertices) {
            if (relax(candidateVertex)) {
                markAffected(candidateVertex);
            }
        }
    }

    // ========================================================================
    // WRITE OUTPUT
    // ========================================================================

    // Create output directories if needed
    filesystem::path distOutPath(distancesOutputPath);
    if (!distOutPath.parent_path().empty()) {
        filesystem::create_directories(distOutPath.parent_path());
    }

    filesystem::path treeOutPath(treeOutputPath);
    if (!treeOutPath.parent_path().empty()) {
        filesystem::create_directories(treeOutPath.parent_path());
    }

    // Write updated distances (same format as Dijkstra output)
    ofstream distancesOut(distancesOutputPath);
    if (!distancesOut.is_open()) {
        cout << "Error: Could not write updated distances file.\n";
        return false;
    }

    for (int i = 0; i < numberOfNodes; ++i) {
        distancesOut << i << " ";
        if (distances[i] >= INF_VALUE / 2) {
            distancesOut << "INF";
        } else {
            distancesOut << distances[i];
        }
        distancesOut << "\n";
    }

    // Write updated SSSP tree (same format as Dijkstra output)
    ofstream treeOut(treeOutputPath);
    if (!treeOut.is_open()) {
        cout << "Error: Could not write updated SSSP tree file.\n";
        return false;
    }

    for (int i = 0; i < numberOfNodes; ++i) {
        treeOut << i << " " << parent[i] << "\n";
    }

    return true;
}
