/**
 * @file test_h2h_delta.cu
 * @brief After every batch, the incrementally maintained h2h (host shadow
 *        AND resident device CSR) must equal the line graph rebuilt from the
 *        incidence lists (independent oracle; rows compared as multisets).
 */

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"
#include "test_util.cuh"

using namespace escher_mosp;

static int failures = 0;
static int rebuilds = 0;

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

        // Every third configuration has no spare CSR capacity (headroom 1),
        // so rows relocate into a full tail and the overflow rebuild runs.
        const double headroom = (cfg % 3 == 0) ? 1.0 : 1.3;
        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, headroom);

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
                ++rebuilds;
                hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, headroom);
            }

            LineGraphCSR lg = rebuildLineGraph(hg);
            if (testutil::shadowRowMismatches(hg, lg) != 0) {
                std::printf("FAIL cfg %d batch %d: shadow != rebuild\n", cfg,
                            bi);
                ++failures;
                break;
            }
            if (testutil::deviceRowMismatches(dev, lg, "delta") != 0) {
                std::printf("FAIL cfg %d batch %d: device CSR != rebuild\n",
                            cfg, bi);
                ++failures;
                break;
            }
        }
    }
    std::printf("device CSR overflow rebuilds: %d\n", rebuilds);
    std::printf("test_h2h_delta: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
