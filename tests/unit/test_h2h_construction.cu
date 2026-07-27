/**
 * @file test_h2h_construction.cu
 * @brief The ESCHER-backed hypergraph's h2h view must equal a brute-force
 *        line graph, both in the host shadow and in the device CSR.
 */

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"

using namespace escher_mosp;

static int failures = 0;

static std::vector<std::set<int>> deviceRowsAsSets(
    const hsosp::DeviceH2H& dev, int m) {
    std::vector<long long> rowStart(m);
    std::vector<int> deg(m);
    cudaMemcpy(rowStart.data(), dev.d_rowStart, sizeof(long long) * m,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(deg.data(), dev.d_deg, sizeof(int) * m,
               cudaMemcpyDeviceToHost);
    std::vector<int> colInd(static_cast<std::size_t>(dev.capEntries));
    cudaMemcpy(colInd.data(), dev.d_colInd, sizeof(int) * dev.capEntries,
               cudaMemcpyDeviceToHost);
    std::vector<std::set<int>> rows(m);
    for (int i = 0; i < m; ++i) {
        for (int e = 0; e < deg[i]; ++e) {
            rows[i].insert(colInd[rowStart[i] + e] + 1);   // back to 1-based
        }
    }
    return rows;
}

int main() {
    std::mt19937_64 meta(4242);
    for (int cfg = 0; cfg < 25; ++cfg) {
        GenParams gp;
        gp.numHyperedges = 50 + static_cast<int>(meta() % 800);
        gp.numVertices = 40 + static_cast<int>(meta() % 600);
        gp.cMin = 1;
        gp.cMax = 2 + static_cast<int>(meta() % 8);
        gp.poolSize = 16 + static_cast<int>(meta() % 96);
        gp.bridgeFrac = 0.1;
        gp.seed = meta();

        GeneratedHypergraph g = generateHypergraph(gp);
        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = static_cast<int>(g.rows.size()) + 64;
        DynamicHypergraph dh(g.numVertices, caps);
        dh.bulkLoad(std::move(g.rows), std::move(g.weights), g.sourceHe,
                    g.targetHe);
        HostHypergraph& hg = dh.host();

        // Shadow == brute force.
        auto bf = hg.bruteForceH2H();
        auto cur = hg.h2h;
        for (auto& r : cur) std::sort(r.begin(), r.end());
        if (bf != cur) {
            std::printf("FAIL cfg %d: shadow h2h != brute force\n", cfg);
            ++failures;
            continue;
        }

        // Device CSR == shadow.
        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.5);
        auto devRows = deviceRowsAsSets(dev, hg.maxId());
        for (int id = 1; id <= hg.maxId(); ++id) {
            std::set<int> want(hg.h2h[id - 1].begin(), hg.h2h[id - 1].end());
            if (devRows[id - 1] != want) {
                std::printf("FAIL cfg %d: device row %d != shadow\n", cfg,
                            id);
                ++failures;
                break;
            }
        }
    }
    std::printf("test_h2h_construction: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
