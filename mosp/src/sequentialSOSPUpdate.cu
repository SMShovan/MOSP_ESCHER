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
 *   Insertions are processed BEFORE deletions for each vertex.
 *
 *   For each inserted edge (u, v) with weight w:
 *     If dist[u] + w < dist[v], update dist[v] and parent[v], mark v affected.
 *
 *   For each deleted edge (u, v):
 *     If parent[v] == u (tree edge deleted), search all in-neighbors of v
 *     in the UPDATED graph for the best alternative parent. Mark v affected.
 *     Non-tree-edge deletions are safely ignored.
 *
 * --- Phase 2: Propagate the Update ---
 *   Iteratively propagate changes until no more vertices are affected:
 *     1. Collect all out-neighbors of affected vertices as candidates.
 *     2. For each candidate, recompute the best distance from ALL in-neighbors.
 *     3. Always update the parent to the current best (fixes parent consistency).
 *     4. If the distance changed, mark the candidate as newly affected.
 *
 * ============================================================================
 * CORRECTNESS FIXES OVER THE ORIGINAL PAPER
 * ============================================================================
 *
 * 1. SOURCE VERTEX PROTECTION: The source vertex (distance = 0) is never
 *    updated, even if it appears as a candidate during propagation.
 *
 * 2. PARENT CONSISTENCY: The paper only updates parent when distance changes.
 *    We always update parent to the current best in-neighbor, even when the
 *    distance stays the same. This prevents stale parent pointers when the
 *    old parent's distance increased but an equally-good alternative exists.
 *
 * 3. SAFETY ITERATION LIMIT: A maximum iteration count (numberOfNodes)
 *    prevents infinite loops in edge cases involving disconnected components.
 *
 * ============================================================================
 */

#include "sequentialSOSPUpdate.cuh"

#include "read.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
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
        if (candidateDistance < bestDistance) {
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
    // Identify initially affected vertices from the batch of changes.
    // Process insertions BEFORE deletions so that a newly inserted better path
    // can protect a vertex from needing an alternative parent search.

    vector<bool> isAffected(numberOfNodes, false);
    vector<int> affectedVertices;

    // --- 1a. Process insertions ---
    // We use the actual weight from the UPDATED adjacency list (not the raw
    // insertion record) because duplicate insertions for the same edge are
    // resolved in Phase 0e by keeping the last write. Using the adjacency
    // weight ensures consistency with the final graph topology.
    for (const auto &edge : insertedEdges) {
        int u = edge.from;
        int v = edge.to;

        // Skip if source vertex of insertion is unreachable
        if (distances[u] >= INF_VALUE / 2) {
            continue;
        }

        // Look up the actual edge weight from the updated adjacency list
        long long actualWeight = -1;
        for (const auto &neighbor : outAdjacency[u]) {
            if (neighbor.vertex == v) {
                actualWeight = neighbor.weight;
                break;
            }
        }
        if (actualWeight < 0) {
            continue; // edge was deleted and not re-inserted
        }

        long long newDistance = distances[u] + actualWeight;
        if (newDistance < distances[v]) {
            distances[v] = newDistance;
            parent[v] = u;
            if (!isAffected[v]) {
                isAffected[v] = true;
                affectedVertices.push_back(v);
            }
        }
    }

    // --- 1b. Process deletions ---
    for (const auto &edge : deletedEdges) {
        int u = edge.from;
        int v = edge.to;

        // Only act if the deleted edge was a tree edge (parent[v] == u).
        // Non-tree-edge deletions do not affect shortest distances.
        if (parent[v] != u) {
            continue;
        }

        // The tree edge to v was removed. Find the best alternative parent
        // among v's in-neighbors in the UPDATED graph (deleted edge already
        // removed, inserted edges already added to the adjacency lists).
        int bestAlternativeParent = -1;
        long long bestAlternativeDistance = INF_VALUE;
        findBestParent(v, inAdjacency, distances, INF_VALUE,
                        bestAlternativeParent, bestAlternativeDistance);

        parent[v] = bestAlternativeParent;
        distances[v] = bestAlternativeDistance;

        if (!isAffected[v]) {
            isAffected[v] = true;
            affectedVertices.push_back(v);
        }
    }

    // --- 1c. Process weight increases on existing edges ---
    // When an insertion overwrites an existing edge with a HIGHER weight,
    // the old shortest path through that edge may no longer be valid.
    // If the affected edge was a tree edge (parent[v] == u), re-evaluate
    // vertex v's distance from ALL in-neighbors, just like a deletion.
    for (const auto &wi : weightIncreases) {
        int u = wi.from;
        int v = wi.to;

        if (parent[v] != u) {
            continue;
        }

        int bestNewParent = -1;
        long long bestNewDistance = INF_VALUE;
        findBestParent(v, inAdjacency, distances, INF_VALUE,
                        bestNewParent, bestNewDistance);

        parent[v] = bestNewParent;
        distances[v] = bestNewDistance;

        if (!isAffected[v]) {
            isAffected[v] = true;
            affectedVertices.push_back(v);
        }
    }

    // ========================================================================
    // PHASE 2: PROPAGATE THE UPDATE
    // ========================================================================
    // Iteratively propagate changes through the graph until convergence.
    // Each iteration:
    //   (a) Collect out-neighbors of all currently affected vertices as candidates.
    //   (b) For each candidate, recompute the best distance from all in-neighbors.
    //   (c) If the distance or parent changed, mark the candidate as affected.

    int iterationCount = 0;
    const int maxIterations = numberOfNodes; // Safety limit

    while (!affectedVertices.empty() && iterationCount < maxIterations) {
        ++iterationCount;

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
            int bestNewParent = -1;
            long long bestNewDistance = INF_VALUE;
            findBestParent(candidateVertex, inAdjacency, distances, INF_VALUE,
                           bestNewParent, bestNewDistance);

            // ALWAYS update the parent to the current best (parent consistency fix).
            // Only propagate further if the DISTANCE actually changed.
            bool distanceChanged = (bestNewDistance != distances[candidateVertex]);

            parent[candidateVertex] = bestNewParent;
            distances[candidateVertex] = bestNewDistance;

            if (distanceChanged) {
                if (!isAffected[candidateVertex]) {
                    isAffected[candidateVertex] = true;
                    affectedVertices.push_back(candidateVertex);
                }
            }
        }
    }

    if (iterationCount >= maxIterations && !affectedVertices.empty()) {
        cout << "Warning: SOSP update reached maximum iteration limit ("
             << maxIterations << "). Running reachability check.\n";
    }

    // ========================================================================
    // POST-PROCESSING: REACHABILITY CHECK
    // ========================================================================
    // If edge deletions disconnected part of the graph from the source, the
    // iterative propagation may not converge (distances keep increasing in
    // a cycle). A BFS from the source on the updated graph identifies all
    // reachable vertices. Unreachable vertices get dist = INF, parent = -1.

    {
        vector<bool> reachable(numberOfNodes, false);
        queue<int> bfsQueue;
        reachable[source] = true;
        bfsQueue.push(source);

        while (!bfsQueue.empty()) {
            int current = bfsQueue.front();
            bfsQueue.pop();
            for (const auto &neighbor : outAdjacency[current]) {
                if (!reachable[neighbor.vertex]) {
                    reachable[neighbor.vertex] = true;
                    bfsQueue.push(neighbor.vertex);
                }
            }
        }

        for (int v = 0; v < numberOfNodes; ++v) {
            if (!reachable[v]) {
                distances[v] = INF_VALUE;
                parent[v] = -1;
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
