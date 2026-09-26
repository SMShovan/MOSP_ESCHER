/**
 * @file test_h2h_construction.cu
 * @brief The ESCHER-backed hypergraph's h2h view must equal the line graph
 *        rebuilt from the incidence lists (independent oracle), both in the
 *        host shadow and in the device CSR (rows compared as multisets, so
 *        duplicate entries are caught).
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <random>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"
#include "test_util.cuh"

using namespace escher_mosp;

static int failures = 0;

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

        LineGraphCSR lg = rebuildLineGraph(hg);
        if (testutil::shadowRowMismatches(hg, lg) != 0) {
            std::printf("FAIL cfg %d: shadow h2h != rebuilt line graph\n",
                        cfg);
            ++failures;
            continue;
        }

        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.5);
        if (testutil::deviceRowMismatches(dev, lg, "construction") != 0) {
            std::printf("FAIL cfg %d: device CSR != rebuilt line graph\n",
                        cfg);
            ++failures;
        }
    }
    // entryHeadroom below 1 (or NaN) must be rejected, not overrun memory.
    {
        std::vector<std::vector<int>> rows = {{0}, {0, 1}, {1, 2}, {2}};
        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = 16;
        DynamicHypergraph dh(3, caps);
        dh.bulkLoad(std::move(rows), {0, 3, 4, 0}, 1, 4);
        for (double bad : {0.0, 0.5, std::nan("")}) {
            hsosp::DeviceH2H dev;
            bool threw = false;
            try {
                hsosp::buildDeviceH2H(dev, dh.host(), caps.maxHyperedges,
                                      bad);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            if (!threw) {
                std::printf("FAIL: entryHeadroom %g accepted\n", bad);
                ++failures;
            }
        }
    }
    std::printf("test_h2h_construction: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
