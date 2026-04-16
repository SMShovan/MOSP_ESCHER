/**
 * @file Dijkstra.cu
 * @brief Single-objective Dijkstra shortest path tree generator.
 */

#include "dijkstra.cuh"
#include "read.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/**
 * @brief Compute single-objective shortest paths and write output files.
 *
 * @param inputFile Path to the Matrix Market input file.
 * @param objectiveNumber 0-indexed objective to use as edge weight.
 * @param source Source vertex (0-indexed).
 * @param distanceOutputPath Output path for distance values.
 * @param treeOutputPath Output path for shortest-path tree.
 * @return True on success; false otherwise.
 */
bool runDijkstra(
    const string &inputFile,
    int objectiveNumber,
    int source,
    const string &distanceOutputPath,
    const string &treeOutputPath
) {
    Graph graph;
    int numberOfObjectives = 0;
    if (!readMtx(inputFile, graph, numberOfObjectives)) {
        return false;
    }

    if (objectiveNumber < 0 || objectiveNumber >= numberOfObjectives) {
        cout << "Error: objectiveNumber out of range.\n";
        return false;
    }

    int n = static_cast<int>(graph.size());
    if (source < 0 || source >= n) {
        cout << "Error: source vertex out of range.\n";
        return false;
    }

    const long long INF = numeric_limits<long long>::max() / 4;
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);

    using Node = pair<long long, int>;
    priority_queue<Node, vector<Node>, greater<Node>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) {
            continue;
        }

        for (const auto &edge : graph[u]) {
            int v = edge.to;
            int w = edge.weights[objectiveNumber];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    filesystem::path distancePath(distanceOutputPath);
    if (!distancePath.parent_path().empty()) {
        filesystem::create_directories(distancePath.parent_path());
    }
    ofstream distOut(distanceOutputPath);
    if (!distOut.is_open()) {
        cout << "Error: Could not write distance output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        distOut << i << " ";
        if (dist[i] >= INF / 2) {
            distOut << "INF";
        } else {
            distOut << dist[i];
        }
        distOut << "\n";
    }

    filesystem::path treePath(treeOutputPath);
    if (!treePath.parent_path().empty()) {
        filesystem::create_directories(treePath.parent_path());
    }
    ofstream treeOut(treeOutputPath);
    if (!treeOut.is_open()) {
        cout << "Error: Could not write tree output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        treeOut << i << " " << parent[i] << "\n";
    }

    return true;
}

/**
 * @brief Compute single-objective shortest paths from CSR format and write output files.
 *
 * @param inputPrefix Base path to CSR files (e.g. "data/originalGraph/graphCsr").
 * @param objectiveNumber 0-indexed objective to use as edge weight.
 * @param source Source vertex (0-indexed).
 * @param distanceOutputPath Output path for distance values.
 * @param treeOutputPath Output path for shortest-path tree.
 * @return True on success; false otherwise.
 */
bool runDijkstraCSR(
    const string &inputPrefix,
    int objectiveNumber,
    int source,
    const string &distanceOutputPath,
    const string &treeOutputPath
) {
    Graph graph;
    int numberOfObjectives = 0;
    if (!readCSR(inputPrefix, graph, numberOfObjectives)) {
        return false;
    }

    if (objectiveNumber < 0 || objectiveNumber >= numberOfObjectives) {
        cout << "Error: objectiveNumber out of range.\n";
        return false;
    }

    int n = static_cast<int>(graph.size());
    if (source < 0 || source >= n) {
        cout << "Error: source vertex out of range.\n";
        return false;
    }

    const long long INF = numeric_limits<long long>::max() / 4;
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);

    using Node = pair<long long, int>;
    priority_queue<Node, vector<Node>, greater<Node>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) {
            continue;
        }

        for (const auto &edge : graph[u]) {
            int v = edge.to;
            int w = edge.weights[objectiveNumber];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    filesystem::path distancePath(distanceOutputPath);
    if (!distancePath.parent_path().empty()) {
        filesystem::create_directories(distancePath.parent_path());
    }
    ofstream distOut(distanceOutputPath);
    if (!distOut.is_open()) {
        cout << "Error: Could not write CSR distance output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        distOut << i << " ";
        if (dist[i] >= INF / 2) {
            distOut << "INF";
        } else {
            distOut << dist[i];
        }
        distOut << "\n";
    }

    filesystem::path treePath(treeOutputPath);
    if (!treePath.parent_path().empty()) {
        filesystem::create_directories(treePath.parent_path());
    }
    ofstream treeOut(treeOutputPath);
    if (!treeOut.is_open()) {
        cout << "Error: Could not write CSR tree output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        treeOut << i << " " << parent[i] << "\n";
    }

    return true;
}
