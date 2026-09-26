/**
 * @file test_hsosp_matches_dijkstra.cu
 * @brief GPU recompute and dynamic update must both match the independent
 *        oracle (Dijkstra on the line graph rebuilt from the incidence
 *        lists), distances and shortest-path parents, including a
 *        forced-disconnection case.
 */

#include <cstdio>
#include <random>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"
#include "test_util.cuh"

using namespace escher_mosp;

static int failures = 0;

static bool distsMatch(const hsosp::HsospState& st,
                       const HostHypergraph& hg, const char* what, int cfg) {
    LineGraphCSR lg = rebuildLineGraph(hg);
    SospCheck c = testutil::checkState(st, hg, lg);
    if (!c.ok()) {
        std::printf("FAIL cfg %d: %s: %lld distance mismatches (first he %d: "
                    "got %lld want %lld), %lld parent errors\n",
                    cfg, what, c.distMismatches, c.firstBadId, c.firstGot,
                    c.firstWant, c.parentErrors);
        ++failures;
        return false;
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

    // ---- targeted placement deletes SOSP-tree parents ------------------
    // (The device stores 0-based node indices; generateBatch treated them as
    // 1-based ids and deleted the hyperedge before each parent.)
    {
        GenParams gp;
        gp.numHyperedges = 2000;
        gp.numVertices = 1500;
        gp.cMin = 2;
        gp.cMax = 5;
        gp.poolSize = 64;
        gp.seed = 99;
        GeneratedHypergraph g = generateHypergraph(gp);
        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = static_cast<int>(g.rows.size()) + 64;
        DynamicHypergraph dh(g.numVertices, caps);
        dh.bulkLoad(std::move(g.rows), std::move(g.weights), g.sourceHe,
                    g.targetHe);
        HostHypergraph& hg = dh.host();
        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.4);
        hsosp::HsospState st;
        st.allocate(caps.maxHyperedges);
        hsosp::hsospRecompute(dev, st, hg.sourceHe, ucfg);
        std::vector<int> parents0, parents;
        st.downloadParents(parents0, hg.maxId());   // 0-based node indices
        st.downloadParentIds(parents, hg.maxId());  // 1-based ids
        std::vector<char> isParent(hg.maxId() + 1, 0);
        for (int p : parents0)
            if (p >= 0) isParent[p + 1] = 1;
        BatchParams bp;
        bp.size = 200;
        bp.delPct = 100;
        bp.placement = Placement::Targeted;
        bp.seed = 5;
        HgBatch batch = generateBatch(hg, gp, bp, {}, parents);
        int notParent = 0;
        for (int id : batch.heDelete) notParent += !isParent[id];
        if (batch.heDelete.empty() || notParent != 0) {
            std::printf("FAIL: targeted batch: %d of %zu deletions are not "
                        "SOSP-tree parents\n",
                        notParent, batch.heDelete.size());
            ++failures;
        }
    }

    // ---- non-positive weights are rejected ----------------------------
    // (With two adjacent zero-weight hyperedges cut off from the source the
    // update kept a stale finite distance: {1,2} and {2,3} of weight 0
    // reachable only through {0,1}, delete {0,1}.)
    {
        auto expectThrow = [&](const char* what, auto&& fn) {
            try {
                fn();
            } catch (const std::exception&) {
                return;
            }
            std::printf("FAIL: %s accepted\n", what);
            ++failures;
        };
        expectThrow("zero-weight hyperedge in bulkLoad", [] {
            DynamicHypergraph::Caps caps;
            caps.maxHyperedges = 16;
            DynamicHypergraph dh(5, caps);
            dh.bulkLoad({{0}, {0, 1}, {1, 2}, {2, 3}, {3}}, {0, 5, 0, 0, 0},
                        1, 5);
        });
        expectThrow("zero-weight inserted hyperedge", [] {
            DynamicHypergraph::Caps caps;
            caps.maxHyperedges = 16;
            DynamicHypergraph dh(5, caps);
            dh.bulkLoad({{0}, {0, 1}, {1, 2}, {3}}, {0, 5, 4, 0}, 1, 4);
            HgBatch b;
            b.heInsert.push_back({{2, 3}, 0});
            dh.applyBatch(b);
        });
    }

    std::printf("test_hsosp_matches_dijkstra: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
