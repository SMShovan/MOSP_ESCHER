/**
 * @file test_dynamicgraph_roundtrip.cu
 * @brief Load a hand-built 5-vertex CSR into @ref escher_mosp::DynamicGraph,
 *        dump it back via @c dumpToCSR, and verify it is byte-identical.
 *
 * This is the simplest correctness check that the adapter preserves the
 * graph topology and weights through the ESCHER-backed storage path.
 */

#include <iostream>
#include <vector>

#include "DynamicGraph.hpp"

namespace {

/**
 * @brief Pretty-print a CSR for diagnostic output.
 */
void printCsr(const char* label,
              const std::vector<int>& rowPtr,
              const std::vector<int>& colInd,
              const std::vector<std::vector<int>>& values) {
    std::cout << label << ":\n  rowPtr = [";
    for (int x : rowPtr) std::cout << x << " ";
    std::cout << "]\n  colInd = [";
    for (int x : colInd) std::cout << x << " ";
    std::cout << "]\n  values = {";
    for (const auto& w : values) {
        std::cout << "[";
        for (int x : w) std::cout << x << " ";
        std::cout << "] ";
    }
    std::cout << "}\n";
}

} // namespace

int main() {
    using namespace escher_mosp;

    //   Graph (5 vertices, 2 objectives, 7 directed edges):
    //   0 -> 1 [w0=3, w1=5]
    //   0 -> 3 [w0=9, w1=1]
    //   1 -> 2 [w0=4, w1=2]
    //   1 -> 4 [w0=7, w1=8]
    //   2 -> 3 [w0=2, w1=6]
    //   3 -> 4 [w0=1, w1=1]
    //   4 -> 0 [w0=5, w1=9]
    const int V = 5;
    const int K = 2;
    const std::vector<int> rowPtr = {0, 2, 4, 5, 6, 7};
    const std::vector<int> colInd = {1, 3, 2, 4, 3, 4, 0};
    const std::vector<std::vector<int>> values = {
        {3, 5}, {9, 1}, {4, 2}, {7, 8}, {2, 6}, {1, 1}, {5, 9}
    };

    try {
        DynamicGraph dg(V, K, /*payloadCapacity=*/4096);
        dg.loadFromCSR(rowPtr, colInd, values);

        std::vector<int> outRow, outCol;
        std::vector<std::vector<int>> outVal;
        dg.dumpToCSR(outRow, outCol, outVal);

        bool ok = (outRow == rowPtr) && (outCol == colInd) && (outVal == values);
        if (!ok) {
            printCsr("expected", rowPtr, colInd, values);
            printCsr("actual",   outRow, outCol, outVal);
            std::cout << "test_dynamicgraph_roundtrip: FAIL\n";
            return 1;
        }
        std::cout << "test_dynamicgraph_roundtrip: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "test_dynamicgraph_roundtrip: FAIL (exception: " << e.what() << ")\n";
        return 1;
    }
}
