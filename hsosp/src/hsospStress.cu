/**
 * @file hsospStress.cu
 * @brief Randomized correctness harness for the full H-SOSP pipeline
 *        (ESCHER routing + device CSR + SOSP update) against host Dijkstra.
 *
 * Mirrors the role of stressTest / parallelStressTest in the MOSP project:
 * many random configurations, exact comparison, non-zero exit and a
 * reproduction seed on the first failure.
 *
 * Usage: hsospStress [--configs N] [--seed S]
 */

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"

using namespace escher_mosp;

int main(int argc, char** argv) {
    int configs = 100;
    unsigned long long seed = 987654321ull;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--configs") && i + 1 < argc)
            configs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = std::stoull(argv[++i]);
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
        dh.bulkLoad(std::move(g.rows), std::move(g.weights), g.sourceHe,
                    g.targetHe);
        HostHypergraph& hg = dh.host();

        hsosp::DeviceH2H dev;
        hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.5);
        hsosp::HsospState st;
        st.allocate(caps.maxHyperedges);

        hsosp::UpdateConfig ucfg;
        ucfg.maxIterations = 256;
        hsosp::hsospRecompute(dev, st, hg.sourceHe, ucfg);

        // Initial solve must already match Dijkstra.
        {
            std::vector<long long> got;
            st.downloadDistances(got, hg.maxId());
            std::vector<long long> truth = hg.dijkstra(hg.sourceHe);
            if (got != truth) {
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
                st.downloadParents(parentSnap, hg.maxId());
            }
            HgBatch batch = generateBatch(hg, gp, bp, distSnap, parentSnap);

            DynamicHypergraph::BatchResult br = dh.applyBatch(batch);
            ++batches;
            if (!hsosp::applyDeltaToDevice(dev, hg, br.delta)) {
                hsosp::buildDeviceH2H(dev, hg, caps.maxHyperedges, 1.5);
            }
            hsosp::UpdateStats us = hsosp::hsospUpdate(
                dev, st, br.delta.seeds, br.delta.deadHe, hg.sourceHe, ucfg);
            if (us.fallbackRecompute) ++fallbacks;

            // Shadow adjacency must equal a brute-force rebuild.
            {
                auto bf = hg.bruteForceH2H();
                for (auto& r : bf) std::sort(r.begin(), r.end());
                auto cur = hg.h2h;
                for (auto& r : cur) std::sort(r.begin(), r.end());
                if (bf != cur) {
                    std::printf(
                        "FAIL cfg %d batch %d (seed %llu): h2h shadow\n",
                        cfg, bi, (unsigned long long)bp.seed);
                    ++failures;
                    break;
                }
            }

            std::vector<long long> got;
            st.downloadDistances(got, hg.maxId());
            std::vector<long long> truth = hg.dijkstra(hg.sourceHe);
            if (got != truth) {
                std::printf(
                    "FAIL cfg %d batch %d (cfgSeed %llu batchSeed %llu "
                    "kind=%s place=%s size=%d del=%.0f)\n",
                    cfg, bi, (unsigned long long)gp.seed,
                    (unsigned long long)bp.seed, toString(bp.kind),
                    toString(bp.placement), bp.size, bp.delPct);
                for (int id = 1; id <= hg.maxId() && id <= 100000; ++id) {
                    if (got[id - 1] != truth[id - 1]) {
                        std::printf("  first mismatch he %d: got %lld "
                                    "want %lld\n",
                                    id, got[id - 1], truth[id - 1]);
                        break;
                    }
                }
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
