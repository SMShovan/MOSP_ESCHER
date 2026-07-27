/**
 * @file test_hsosp_matches_dijkstra.cu
 * @brief GPU recompute and dynamic update must both match host Dijkstra,
 *        including a forced-disconnection case that exercises the fallback.
 */

#include <cstdio>
#include <random>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"

using namespace escher_mosp;

static int failures = 0;

static bool distsMatch(const hsosp::HsospState& st,
                       const HostHypergraph& hg, const char* what, int cfg) {
    std::vector<long long> got;
    st.downloadDistances(got, hg.maxId());
    std::vector<long long> truth = hg.dijkstra(hg.sourceHe);
    for (int i = 0; i < hg.maxId(); ++i) {
        if (got[i] != truth[i]) {
            std::printf("FAIL cfg %d: %s mismatch at he %d (got %lld want "
                        "%lld)\n",
                        cfg, what, i + 1, got[i], truth[i]);
            ++failures;
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937_64 meta(777);
    hsosp::UpdateConfig ucfg;
    ucfg.maxIterations = 256;

    // ---- randomized configurations --------------------------------------
    for (int cfg = 0; cfg < 20 && failures == 0; ++cfg) {
        GenParams gp;
        gp.numHyperedges = 100 + static_cast<int>(meta() % 1000);
        gp.numVertices = 80 + static_cast<int>(meta() % 800);
        gp.cMin = 1;
        gp.cMax = 2 + static_cast<int>(meta() % 8);
        gp.poolSize = 24 + static_cast<int>(meta() % 96);
        gp.bridgeFrac = 0.08;
        gp.wMax = 1 + static_cast<int>(meta() % 90);
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
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.4);
        hsosp::HsospState st;
        st.allocate(caps.maxHyperedges);

        hsosp::hsospRecompute(dev, st, hg.sourceHe, ucfg);
        if (!distsMatch(st, hg, "recompute", cfg)) continue;

        for (int bi = 0; bi < 4 && failures == 0; ++bi) {
            BatchParams bp;
            bp.size = 10 + static_cast<int>(meta() % 100);
            bp.delPct = static_cast<double>(meta() % 101);
            bp.kind =
                (meta() % 2) ? BatchKind::Hyperedge : BatchKind::Vertex;
            bp.seed = meta();
            HgBatch batch = generateBatch(hg, gp, bp, {}, {});

            DynamicHypergraph::BatchResult br = dh.applyBatch(batch);
            if (!hsosp::applyDeltaToDevice(dev, hg, br.delta)) {
                hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.4);
            }
            hsosp::hsospUpdate(dev, st, br.delta.seeds, br.delta.deadHe,
                               hg.sourceHe, ucfg);
            if (!distsMatch(st, hg, "update", cfg)) break;
        }
    }

    // ---- forced disconnection -------------------------------------------
    if (failures == 0) {
        std::vector<std::vector<int>> rows = {
            {0}, {0, 1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6},
        };
        std::vector<long long> ws = {0, 5, 7, 3, 11, 2, 0};
        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = 32;
        DynamicHypergraph dh(7, caps);
        dh.bulkLoad(std::move(rows), std::move(ws), 1, 7);
        HostHypergraph& hg = dh.host();

        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.4);
        hsosp::HsospState st;
        st.allocate(caps.maxHyperedges);
        hsosp::hsospRecompute(dev, st, hg.sourceHe, ucfg);

        HgBatch batch;
        batch.heDelete.push_back(4);   // the bridge hyperedge
        DynamicHypergraph::BatchResult br = dh.applyBatch(batch);
        if (!hsosp::applyDeltaToDevice(dev, hg, br.delta)) {
            hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.4);
        }
        hsosp::hsospUpdate(dev, st, br.delta.seeds, br.delta.deadHe,
                           hg.sourceHe, ucfg);
        distsMatch(st, hg, "disconnect", 9999);
    }

    std::printf("test_hsosp_matches_dijkstra: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
