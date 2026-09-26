/**
 * @file hsospStress.cu
 * @brief Randomized correctness harness for the full H-SOSP pipeline
 *        (ESCHER routing + device CSR + SOSP update) against an independent
 *        oracle: Dijkstra on the line graph rebuilt from the incidence lists
 *        (HypergraphOracle.hpp).
 *
 * Mirrors the role of stressTest / parallelStressTest in the MOSP project:
 * many random configurations, exact comparison of distances, a check of
 * the shortest-path parents, of the device CSR rows and (with
 * --check-escher) of the ESCHER CBST contents; non-zero exit and a
 * reproduction seed on the first failure.
 *
 * The update budget is disabled, so every batch runs the incremental
 * update (a fallback would be a from-scratch recompute).
 *
 * Usage: hsospStress [--configs N] [--seed S] [--check-escher]
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "HypergraphOracle.hpp"
#include "hsosp.cuh"

using namespace escher_mosp;

int main(int argc, char** argv) {
    int configs = 100;
    unsigned long long seed = 987654321ull;
    bool checkEscher = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--configs") && i + 1 < argc)
            configs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = std::stoull(argv[++i]);
        else if (!std::strcmp(argv[i], "--check-escher"))
            checkEscher = true;
        else {
            std::fprintf(stderr,
                         "usage: hsospStress [--configs N] [--seed S] "
                         "[--check-escher]\n");
            return 2;
        }
    }

    std::mt19937_64 meta(seed);
    int failures = 0, fallbacks = 0, batches = 0;

    for (int cfg = 0; cfg < configs; ++cfg) {
        GenParams gp;
        gp.numHyperedges = 60 + static_cast<int>(meta() % 1500);
        gp.numVertices = 40 + static_cast<int>(meta() % 1200);
        gp.cMin = 1 + static_cast<int>(meta() % 2);
        gp.cMax = gp.cMin + 1 + static_cast<int>(meta() % 7);
        gp.poolSize = 16 + static_cast<int>(meta() % 128);
        gp.bridgeFrac = 0.02 + 0.25 * (meta() % 100) / 100.0;
        gp.wMax = 1 + static_cast<int>(meta() % 100);
        gp.seed = meta();

        GeneratedHypergraph g = generateHypergraph(gp);

        DynamicHypergraph::Caps caps;
        caps.maxHyperedges = static_cast<int>(g.rows.size()) + 4096;
        caps.headroomFactor = 2.0;
        caps.extraPayloadInts = 1 << 20;

        DynamicHypergraph dh(g.numVertices, caps);
        LineGraphCSR lg0 = dh.bulkLoad(std::move(g.rows),
                                       std::move(g.weights), g.sourceHe,
                                       g.targetHe);
        HostHypergraph& hg = dh.host();

        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, lg0, caps.maxHyperedges, 1.5);
        hsosp::HsospState st;
        st.allocate(caps.maxHyperedges);

        hsosp::UpdateConfig ucfg;
        // No update budget: every batch exercises the incremental path (the
        // budget fallback is the recompute, tested on its own).
        ucfg.maxIterations = 1 << 30;
        ucfg.workBudget = 1e30;
        hsosp::hsospRecompute(dev, st, hg.sourceHe, ucfg);

        // Initial solve must already match the oracle.
        {
            LineGraphCSR lg = rebuildLineGraph(hg);
            std::vector<long long> got;
            std::vector<int> par;
            st.downloadDistances(got, hg.maxId());
            st.downloadParents(par, hg.maxId());
            SospCheck c = checkSosp(
                hg, lg, referenceDistances(hg, lg, hg.sourceHe), got, par);
            if (!c.ok()) {
                std::printf("FAIL cfg %d (seed %llu): initial solve\n", cfg,
                            (unsigned long long)gp.seed);
                ++failures;
                continue;
            }
        }

        const int nBatches = 3;
        for (int bi = 0; bi < nBatches && failures == 0; ++bi) {
            BatchParams bp;
            bp.size = 5 + static_cast<int>(meta() % 120);
            bp.delPct = static_cast<double>(meta() % 101);
            bp.kind =
                (meta() % 2) ? BatchKind::Hyperedge : BatchKind::Vertex;
            switch (meta() % 4) {
                case 0: bp.placement = Placement::Random; break;
                case 1: bp.placement = Placement::Targeted; break;
                case 2: bp.placement = Placement::Near; break;
                default: bp.placement = Placement::Far; break;
            }
            bp.seed = meta();

            std::vector<long long> distSnap;
            std::vector<int> parentSnap;
            if (bp.placement != Placement::Random) {
                st.downloadDistances(distSnap, hg.maxId());
                st.downloadParentIds(parentSnap, hg.maxId());
            }
            HgBatch batch = generateBatch(hg, gp, bp, distSnap, parentSnap);

            hsosp::applyBatch(dh, dev, batch, 1.5);
            ++batches;
            hsosp::UpdateStats us = hsosp::hsospUpdate(dev, st, hg.sourceHe, ucfg);
            if (us.fallbackRecompute) ++fallbacks;

            // Oracle: line graph rebuilt from the incidence lists.
            LineGraphCSR lg = rebuildLineGraph(hg);
            const char* what = nullptr;
            long long bad = 0;
            {
                auto devRows = hsosp::downloadRows(dev, hg.maxId());
                for (int id = 1; id <= hg.maxId() && !bad; ++id)
                    if (devRows[id - 1] !=
                        std::vector<int>(lg.row(id),
                                         lg.row(id) + lg.degree(id)))
                        bad = 1, what = "device CSR";
            }
            SospCheck c;
            if (!bad) {
                std::vector<long long> got;
                std::vector<int> par;
                st.downloadDistances(got, hg.maxId());
                st.downloadParents(par, hg.maxId());
                c = checkSosp(hg, lg, referenceDistances(hg, lg, hg.sourceHe),
                              got, par);
                if (!c.ok()) bad = 1, what = "distances / parents";
            }
            if (!bad && checkEscher && dh.checkEscher(lg, std::cout) != 0)
                bad = 1, what = "ESCHER contents";
            if (bad) {
                std::printf(
                    "FAIL cfg %d batch %d (cfgSeed %llu batchSeed %llu "
                    "kind=%s place=%s size=%d del=%.0f): %s\n",
                    cfg, bi, (unsigned long long)gp.seed,
                    (unsigned long long)bp.seed, toString(bp.kind),
                    toString(bp.placement), bp.size, bp.delPct, what);
                if (c.distMismatches || c.parentErrors)
                    std::printf("  %lld distance mismatches (first he %d: "
                                "got %lld want %lld), %lld parent errors\n",
                                c.distMismatches, c.firstBadId, c.firstGot,
                                c.firstWant, c.parentErrors);
                ++failures;
                break;
            }
        }
        if (failures) break;
    }

    std::printf("hsospStress: %d configs, %d batches, %d fallbacks, "
                "%d failures\n",
                configs, batches, fallbacks, failures);
    return failures == 0 ? 0 : 1;
}
