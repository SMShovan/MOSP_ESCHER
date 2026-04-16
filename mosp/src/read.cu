/**
 * @file read.cu
 * @brief Matrix Market (.mtx) and CSR reader for multi-objective graphs.
 */

#include "read.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Read a Matrix Market graph with multiple objective weights.
 *
 * Each edge line is expected to contain: from to w1 w2 ... wK
 * where K is inferred from the first edge line.
 *
 * @param path Input .mtx file path.
 * @param graph Output adjacency list (0-indexed vertices).
 * @param numberOfObjectives Output number of objectives (weights per edge).
 * @return True on success; false otherwise.
 */
bool readMtx(const string &path, Graph &graph, int &numberOfObjectives) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "Error: Could not open input file.\n";
        return false;
    }

    string line;

    // Skip comments and header lines beginning with '%'.
    while (getline(file, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    if (file.fail()) {
        cout << "Error: Invalid Matrix Market header.\n";
        return false;
    }

    stringstream header(line);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    header >> rows >> cols >> nnz;

    if (rows <= 0 || cols <= 0 || nnz <= 0) {
        cout << "Error: Invalid matrix size.\n";
        return false;
    }

    graph.assign(rows, {});
    numberOfObjectives = 0;

    int edgesRead = 0;
    while (edgesRead < nnz && getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);
        vector<int> tokens;
        int value = 0;
        while (ss >> value) {
            tokens.push_back(value);
        }

        if (tokens.size() < 3) {
            cout << "Error: Edge line missing weights.\n";
            return false;
        }

        if (numberOfObjectives == 0) {
            numberOfObjectives = static_cast<int>(tokens.size()) - 2;
            if (numberOfObjectives <= 0) {
                cout << "Error: Invalid number of objectives.\n";
                return false;
            }
        } else if (static_cast<int>(tokens.size()) - 2 != numberOfObjectives) {
            cout << "Error: Inconsistent number of objectives.\n";
            return false;
        }

        int from = tokens[0];
        int to = tokens[1];
        if (from < 0 || from >= rows || to < 0 || to >= rows) {
            cout << "Error: Vertex index out of range.\n";
            return false;
        }

        vector<int> weights(tokens.begin() + 2, tokens.end());
        graph[from].push_back({to, weights});
        ++edgesRead;
    }

    if (edgesRead != nnz) {
        cout << "Error: Unexpected end of file.\n";
        return false;
    }

    return true;
}

/**
 * @brief Read a graph in CSR format from three files.
 *
 * Reads prefixRowPtr.txt, prefixColInd.txt, prefixValues.txt
 * and converts to adjacency list format.
 *
 * @param prefix Base path (e.g. "data/originalGraph/graphCsr").
 * @param graph Output adjacency list (0-indexed vertices).
 * @param numberOfObjectives Output number of objectives (weights per edge).
 * @return True on success; false otherwise.
 */
bool readCSR(const string &prefix, Graph &graph, int &numberOfObjectives) {
    string path_row = prefix + "RowPtr.txt";
    string path_col = prefix + "ColInd.txt";
    string path_val = prefix + "Values.txt";

    ifstream file_row(path_row);
    ifstream file_col(path_col);
    ifstream file_val(path_val);

    if (!file_row.is_open() || !file_col.is_open() || !file_val.is_open()) {
        cout << "Error: Could not open CSR files.\n";
        return false;
    }

    vector<int> row_ptr;
    int value;
    while (file_row >> value) {
        row_ptr.push_back(value);
    }

    if (row_ptr.size() < 2) {
        cout << "Error: Invalid row_ptr file.\n";
        return false;
    }

    int n = static_cast<int>(row_ptr.size()) - 1;
    int nnz = row_ptr.back();

    vector<int> col_ind;
    col_ind.reserve(nnz);
    while (file_col >> value) {
        col_ind.push_back(value);
    }

    if (static_cast<int>(col_ind.size()) != nnz) {
        cout << "Error: col_ind size mismatch.\n";
        return false;
    }

    string line;
    vector<vector<int>> values;
    values.reserve(nnz);
    numberOfObjectives = 0;

    while (getline(file_val, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<int> weights;
        int w;
        while (ss >> w) {
            weights.push_back(w);
        }
        if (numberOfObjectives == 0) {
            numberOfObjectives = static_cast<int>(weights.size());
        } else if (static_cast<int>(weights.size()) != numberOfObjectives) {
            cout << "Error: Inconsistent number of objectives.\n";
            return false;
        }
        values.push_back(weights);
    }

    if (static_cast<int>(values.size()) != nnz) {
        cout << "Error: values size mismatch.\n";
        return false;
    }

    graph.assign(n, {});
    for (int i = 0; i < n; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            graph[i].push_back({col_ind[k], values[k]});
        }
    }

    return true;
}
