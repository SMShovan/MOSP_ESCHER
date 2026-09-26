/**
 * @file test_hsosp_scale.cu
 * @brief The H-SOSP pipeline above 65,535 CBST records (h2v, v2h and h2h
 *        each hold more), over consecutive insert-only, deletion-heavy,
 *        mixed and vertex batches. After every batch the device CSR rows,
 *        the SOSP distances and parents, and the contents of the three
 *        ESCHER CBSTs are compared with the independent oracle.
 */

#include <cstdio>
#include <iostream>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"
#include "test_util.cuh"

using namespace escher_mosp;

int main() {
    int failures = 0;
    try {
        GenParams gp;
        gp.numHyperedges = 75000;
        gp.numVertices = 80000;
        gp.cMin = 2;
        gp.cMax = 6;
        gp.poolSize = 384;
        gp.bridgeFrac = 0.05;
        gp.wMax = 100;
        gp.seed = 20260925;
        GeneratedHypergraph g = generateHypergraph(gp);

        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = static_cast<int>(g.rows.size()) + 40000;
        caps.headroomFactor = 1.5;
        caps.extraPayloadInts = 8 << 20;
        DynamicHypergraph dh(g.numVertices, caps);
        dh.bulkLoad(std::move(g.rows), std::move(g.weights), g.sourceHe,
                    g.targetHe);
        HostHypergraph& hg = dh.host();

        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.5);
        hsosp::HsospState st;
        st.allocate(caps.maxHyperedges);
        hsosp::UpdateConfig ucfg;
        hsosp::hsospRecompute(dev, st, hg.sourceHe, ucfg);

        struct Step {
            const char* name;
            BatchKind kind;
            int size;
            double delPct;
        };
        const Step steps[] = {
            {"load", BatchKind::Hyperedge, 0, 0},
            {"hyperedge insert-only", BatchKind::Hyperedge, 3000, 0},
            {"hyperedge delete-only", BatchKind::Hyperedge, 3000, 100},
            {"hyperedge mixed", BatchKind::Hyperedge, 6000, 50},
            {"vertex mixed", BatchKind::Vertex, 6000, 50},
            {"vertex delete-heavy", BatchKind::Vertex, 4000, 90},
            {"hyperedge delete-heavy", BatchKind::Hyperedge, 8000, 80},
            {"hyperedge insert-heavy", BatchKind::Hyperedge, 8000, 20},
        };
        for (std::size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i) {
            const Step& s = steps[i];
            if (s.size > 0) {
                BatchParams bp;
                bp.size = s.size;
                bp.delPct = s.delPct;
                bp.kind = s.kind;
                bp.seed = 1000 + i;
                HgBatch batch = generateBatch(hg, gp, bp, {}, {});
                DynamicHypergraph::BatchResult br = dh.applyBatch(batch);
                if (!hsosp::applyDeltaToDevice(dev, hg, br.delta))
                    hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.5);
                hsosp::hsospUpdate(dev, st, br.delta.seeds, br.delta.deadHe,
                                   hg.sourceHe, ucfg);
            }
            LineGraphCSR lg = rebuildLineGraph(hg);
            const long long rowBad =
                testutil::deviceRowMismatches(dev, lg, s.name);
            SospCheck c = testutil::checkState(st, hg, lg);
            const long long escherBad = dh.checkEscher(lg, std::cout);
            const bool ok = rowBad == 0 && c.ok() && escherBad == 0;
            std::printf("%s  %-24s ids %d: device rows %lld bad, distances "
                        "%lld bad, parents %lld bad, ESCHER %lld violations\n",
                        ok ? "PASS" : "FAIL", s.name, hg.maxId(), rowBad,
                        c.distMismatches, c.parentErrors, escherBad);
            if (!ok) ++failures;
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        ++failures;
    }
    std::printf("test_hsosp_scale: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
