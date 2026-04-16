#ifndef READ_CUH
#define READ_CUH

#include <string>
#include <vector>

struct Edge {
    int to;
    std::vector<int> weights;
};

using Graph = std::vector<std::vector<Edge>>;

/** @brief Read a Matrix Market (.mtx) graph into an adjacency list. */
bool readMtx(const std::string &path, Graph &graph, int &numberOfObjectives);

/** @brief Read a graph in CSR format from prefixRowPtr.txt, prefixColInd.txt, prefixValues.txt */
bool readCSR(const std::string &prefix, Graph &graph, int &numberOfObjectives);

#endif
