/**
 * @file test_h2h_delta.cu
 * @brief After every batch, the resident device CSR (patched with the
 *        line-graph delta derived on the GPU) must equal the line graph
 *        rebuilt from the incidence lists (independent oracle; rows
 *        compared as multisets), and the device incidence mirror must
 *        equal the host's.
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
static int reuploads = 0;

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
        LineGraphCSR lg0 = dh.bulkLoad(std::move(g.rows),
                                       std::move(g.weights), g.sourceHe,
                                       g.targetHe);
        HostHypergraph& hg = dh.host();

        // Every third configuration has no spare capacity (headroom 1), so
        // rows relocate into a full tail: the CSR overflow rebuild and the
        // re-upload of the incidence mirror run.
        const double headroom = (cfg % 3 == 0) ? 1.0 : 1.3;
        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, lg0, caps.maxHyperedges, headroom);

        for (int bi = 0; bi < 6 && failures == 0; ++bi) {
            BatchParams bp;
            bp.size = 5 + static_cast<int>(meta() % 80);
            bp.delPct = static_cast<double>(meta() % 101);
            bp.kind =
                (meta() % 2) ? BatchKind::Hyperedge : BatchKind::Vertex;
            bp.seed = meta();
            HgBatch batch = generateBatch(hg, gp, bp, {}, {});

            hsosp::BatchTimes bt =
                hsosp::applyBatch(dh, dev, batch, headroom);
            if (bt.csrRebuilt) ++rebuilds;
            reuploads += bt.mirrorReuploads;

            LineGraphCSR lg = rebuildLineGraph(hg);
            if (hsosp::incidenceMirrorMismatches(dev.inc, hg) != 0) {
                std::printf("FAIL cfg %d batch %d: device incidence != "
                            "host\n", cfg, bi);
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
    std::printf("device CSR overflow rebuilds: %d, incidence mirror "
                "re-uploads: %d\n",
                rebuilds, reuploads);
    if (rebuilds == 0 || reuploads == 0) {
        std::printf("FAIL: the overflow paths did not run\n");
        ++failures;
    }
    std::printf("test_h2h_delta: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
