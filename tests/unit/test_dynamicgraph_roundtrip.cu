/**
 * @file test_dynamicgraph_roundtrip.cu
 * @brief Load a hand-built 5-vertex CSR into @ref escher_mosp::DynamicGraph,
 *        dump it back via @c dumpToCSR, and verify it is byte-identical;
 *        then run delete / insert rounds and read the three CBSTs back
 *        after each (DynamicGraph::checkEscher): edge records must sit
 *        under the edge-id the host uses and edges leaving vertex 0 must
 *        read back whole.
 */

#include <iostream>
#include <random>
#include <set>
#include <utility>
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
        if (dg.checkEscher(std::cout) != 0) {
            std::cout << "test_dynamicgraph_roundtrip: FAIL (after load)\n";
            return 1;
        }

        // Delete the edges with ids 1 (0->1) and 3 (1->2), then insert two
        // edges of different sizes: ESCHER reuses the deleted slots and the
        // host must adopt the keys it chose.
        dg.deleteEdges({{0, 1}, {1, 2}});
        dg.insertEdges({{2, 4, {11, 12}}, {0, 2, {6, 7}}});
        if (dg.checkEscher(std::cout) != 0) {
            std::cout << "test_dynamicgraph_roundtrip: FAIL (after reuse)\n";
            return 1;
        }

        // Random rounds on a larger graph, deleted edges re-inserted later.
        std::mt19937 rng(5);
        const int V2 = 60;
        std::set<std::pair<int, int>> present;
        std::vector<int> rp(V2 + 1, 0), ci;
        std::vector<std::vector<int>> vals;
        for (int u = 0; u < V2; ++u) {
            for (int v = 0; v < V2; ++v) {
                if (u != v && rng() % 10 == 0) {
                    ci.push_back(v);
                    vals.push_back({static_cast<int>(rng() % 20),
                                    static_cast<int>(rng() % 20)});
                    present.insert({u, v});
                }
            }
            rp[u + 1] = static_cast<int>(ci.size());
        }
        DynamicGraph big(V2, K, 1 << 16);
        big.loadFromCSR(rp, ci, vals);
        for (int round = 0; round < 8; ++round) {
            std::vector<EdgeDelete> del;
            for (auto it = present.begin(); it != present.end();) {
                if (rng() % 8 == 0) {
                    del.push_back({it->first, it->second});
                    it = present.erase(it);
                } else {
                    ++it;
                }
            }
            big.deleteEdges(del);
            std::vector<EdgeInsert> ins;
            for (int k = 0; k < 40; ++k) {
                int u = rng() % V2, v = rng() % V2;
                if (u == v || !present.insert({u, v}).second) continue;
                ins.push_back({u, v, {static_cast<int>(rng() % 20),
                                      static_cast<int>(rng() % 20)}});
            }
            big.insertEdges(ins);
            if (big.checkEscher(std::cout) != 0) {
                std::cout << "test_dynamicgraph_roundtrip: FAIL (round "
                          << round << ")\n";
                return 1;
            }
        }
        std::cout << "test_dynamicgraph_roundtrip: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "test_dynamicgraph_roundtrip: FAIL (exception: " << e.what() << ")\n";
        return 1;
    }
}
