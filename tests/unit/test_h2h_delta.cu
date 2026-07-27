/**
 * @file test_h2h_delta.cu
 * @brief After every batch, the incrementally maintained h2h (host shadow
 *        AND resident device CSR) must equal a from-scratch rebuild.
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
            rows[i].insert(colInd[rowStart[i] + e] + 1);
        }
    }
    return rows;
}

int main() {
    std::mt19937_64 meta(31337);
    for (int cfg = 0; cfg < 15 && failures == 0; ++cfg) {
        GenParams gp;
        gp.numHyperedges = 80 + static_cast<int>(meta() % 500);
        gp.numVertices = 60 + static_cast<int>(meta() % 400);
        gp.cMin = 1;
        gp.cMax = 2 + static_cast<int>(meta() % 7);
        gp.poolSize = 16 + static_cast<int>(meta() % 64);
        gp.bridgeFrac = 0.1;
        gp.seed = meta();

        GeneratedHypergraph g = generateHypergraph(gp);
        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = static_cast<int>(g.rows.size()) + 2048;
        caps.headroomFactor = 2.0;
        DynamicHypergraph dh(g.numVertices, caps);
        dh.bulkLoad(std::move(g.rows), std::move(g.weights), g.sourceHe,
                    g.targetHe);
        HostHypergraph& hg = dh.host();

        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.3);

        for (int bi = 0; bi < 6 && failures == 0; ++bi) {
            BatchParams bp;
            bp.size = 5 + static_cast<int>(meta() % 80);
            bp.delPct = static_cast<double>(meta() % 101);
            bp.kind =
                (meta() % 2) ? BatchKind::Hyperedge : BatchKind::Vertex;
            bp.seed = meta();
            HgBatch batch = generateBatch(hg, gp, bp, {}, {});

            DynamicHypergraph::BatchResult br = dh.applyBatch(batch);
            if (!hsosp::applyDeltaToDevice(dev, hg, br.delta)) {
                hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.3);
            }

            auto bf = hg.bruteForceH2H();
            auto cur = hg.h2h;
            for (auto& r : cur) std::sort(r.begin(), r.end());
            if (bf != cur) {
                std::printf("FAIL cfg %d batch %d: shadow != rebuild\n", cfg,
                            bi);
                ++failures;
                break;
            }
            auto devRows = deviceRowsAsSets(dev, hg.maxId());
            for (int id = 1; id <= hg.maxId(); ++id) {
                std::set<int> want(hg.h2h[id - 1].begin(),
                                   hg.h2h[id - 1].end());
                if (devRows[id - 1] != want) {
                    std::printf(
                        "FAIL cfg %d batch %d: device row %d != shadow "
                        "(%zu vs %zu entries)\n",
                        cfg, bi, id, devRows[id - 1].size(), want.size());
                    ++failures;
                    break;
                }
            }
        }
    }
    std::printf("test_h2h_delta: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
