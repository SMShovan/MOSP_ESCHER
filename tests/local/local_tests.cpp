/**
 * @file local_tests.cpp
 * @brief GPU-free validation of the hypergraph host core.
 *
 * Compiled with plain g++ (see tests/local/Makefile). Validates, over many
 * randomized configurations:
 *   1. bulk h2h build == brute-force line graph;
 *   2. incrementally maintained h2h == rebuilt-from-scratch line graph
 *      after every batch (all four op kinds);
 *   3. the NET H2HDelta applied to the pre-batch adjacency reproduces the
 *      post-batch adjacency (this is exactly what the device CSR does);
 *   4. the sequential emulation of the device SOSP update matches Dijkstra
 *      after every batch (including disconnections and id recycling).
 *
 * These mirror tests/unit/*.cu which run the same checks through the real
 * ESCHER + CUDA path on the cluster.
 */

#include "HostHypergraph.hpp"
#include "HypergraphGen.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace escher_mosp;

static int failures = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                  \
            std::printf(__VA_ARGS__);                                         \
            std::printf("\n");                                                \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

static std::vector<std::set<int>> asSets(
    const std::vector<std::vector<int>>& rows) {
    std::vector<std::set<int>> out(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        out[i] = std::set<int>(rows[i].begin(), rows[i].end());
    return out;
}

int main() {
    std::mt19937_64 metaRng(20260725);

    const int CONFIGS = 60;
    int totalBatches = 0, fallbacks = 0;

    for (int cfg = 0; cfg < CONFIGS; ++cfg) {
        GenParams gp;
        gp.numHyperedges = 40 + static_cast<int>(metaRng() % 400);
        gp.numVertices = 30 + static_cast<int>(metaRng() % 300);
        gp.cMin = 1 + static_cast<int>(metaRng() % 2);
        gp.cMax = gp.cMin + 1 + static_cast<int>(metaRng() % 6);
        gp.poolSize = 8 + static_cast<int>(metaRng() % 64);
        gp.bridgeFrac = 0.02 + 0.2 * (metaRng() % 100) / 100.0;
        gp.wMin = 1;
        gp.wMax = 1 + static_cast<int>(metaRng() % 100);
        gp.seed = metaRng();

        GeneratedHypergraph g = generateHypergraph(gp);
        HostHypergraph hg;
        {
            auto rows = g.rows;       // keep a copy for the build check
            auto ws = g.weights;
            hg.buildFrom(g.numVertices, std::move(rows), std::move(ws));
        }
        hg.sourceHe = g.sourceHe;
        hg.targetHe = g.targetHe;

        // ---- 1. build == brute force -----------------------------------
        auto bf = asSets(hg.bruteForceH2H());
        auto cur = asSets(hg.h2h);
        CHECK(bf == cur, "cfg %d: bulk h2h != brute force", cfg);

        // Initial SOSP state via recompute (the device does the same).
        std::vector<long long> dist;
        std::vector<int> parent;
        emulateSospRecompute(hg, dist, parent, hg.maxId() + 2);
        {
            auto truth = hg.dijkstra(hg.sourceHe);
            CHECK(truth == dist, "cfg %d: initial recompute != dijkstra", cfg);
        }

        // ---- batches ----------------------------------------------------
        const int BATCHES = 4;
        for (int bi = 0; bi < BATCHES; ++bi) {
            BatchParams bp;
            bp.size = 5 + static_cast<int>(metaRng() % 60);
            bp.delPct = static_cast<double>(metaRng() % 101);
            bp.kind = (metaRng() % 2) ? BatchKind::Hyperedge
                                      : BatchKind::Vertex;
            switch (metaRng() % 4) {
                case 0: bp.placement = Placement::Random; break;
                case 1: bp.placement = Placement::Targeted; break;
                case 2: bp.placement = Placement::Near; break;
                default: bp.placement = Placement::Far; break;
            }
            bp.seed = metaRng();

            HgBatch batch = generateBatch(hg, gp, bp, dist, parent);
            std::vector<int> finalIds =
                hg.reserveIds(static_cast<int>(batch.heInsert.size()));

            // Pre-batch adjacency snapshot for the delta-replay check.
            auto preH2h = asSets(hg.h2h);

            H2HDelta delta;
            EscherHorizOps ops;
            hg.applyBatch(batch, finalIds, delta, ops);
            ++totalBatches;

            // ---- 2. incremental == rebuilt ------------------------------
            auto bf2 = asSets(hg.bruteForceH2H());
            auto cur2 = asSets(hg.h2h);
            CHECK(bf2 == cur2, "cfg %d batch %d: incremental h2h broke", cfg,
                  bi);

            // ---- 3. delta replay reproduces post-state ------------------
            {
                auto replay = preH2h;
                replay.resize(hg.maxId());
                for (auto [a, b] : delta.delEdges) {
                    replay[a - 1].erase(b);
                    replay[b - 1].erase(a);
                }
                for (auto [a, b] : delta.insEdges) {
                    replay[a - 1].insert(b);
                    replay[b - 1].insert(a);
                }
                CHECK(replay == cur2,
                      "cfg %d batch %d: delta replay != post adjacency", cfg,
                      bi);
            }

            // ---- pair-count bookkeeping ---------------------------------
            long long pairs = 0;
            for (auto& s : cur2) pairs += static_cast<long long>(s.size());
            CHECK(pairs / 2 == hg.h2hPairCount,
                  "cfg %d batch %d: pair count %lld != tracked %lld", cfg, bi,
                  pairs / 2, hg.h2hPairCount);

            // ---- 4. dynamic update == dijkstra --------------------------
            int it = emulateSospUpdate(hg, dist, parent, delta.seeds, 512);
            if (it < 0) ++fallbacks;
            auto truth = hg.dijkstra(hg.sourceHe);
            for (int id = 1; id <= hg.maxId(); ++id) {
                if (truth[id - 1] != dist[id - 1]) {
                    CHECK(false,
                          "cfg %d batch %d: dist mismatch at he %d "
                          "(got %lld want %lld) kind=%s place=%s",
                          cfg, bi, id, dist[id - 1], truth[id - 1],
                          toString(bp.kind), toString(bp.placement));
                    break;
                }
            }

            // Parent consistency: dist[v] == dist[parent] + w[v].
            for (int id = 1; id <= hg.maxId(); ++id) {
                if (id == hg.sourceHe || !hg.alive[id - 1]) continue;
                if (dist[id - 1] >= HostHypergraph::INF / 2) continue;
                int p = parent[id - 1];
                CHECK(p >= 1 && p <= hg.maxId(),
                      "cfg %d batch %d: reachable he %d has no parent", cfg,
                      bi, id);
                if (p >= 1 && p <= hg.maxId()) {
                    CHECK(dist[id - 1] == dist[p - 1] + hg.heW[id - 1],
                          "cfg %d batch %d: parent invariant broken at %d",
                          cfg, bi, id);
                }
            }
        }
    }

    // ---- forced disconnection scenario ---------------------------------
    {
        // Two pools joined by a single bridge hyperedge; deleting the
        // bridge must drive the far side to INF via the fallback path.
        HostHypergraph hg;
        std::vector<std::vector<int>> rows = {
            {0},          // 1: virtual source {s}
            {0, 1, 2},    // 2
            {2, 3},       // 3
            {3, 4},       // 4: bridge
            {4, 5},       // 5
            {5, 6},       // 6
            {6},          // 7: virtual target
        };
        std::vector<long long> ws = {0, 5, 7, 3, 11, 2, 0};
        hg.buildFrom(7, std::move(rows), std::move(ws));
        hg.sourceHe = 1;
        hg.targetHe = 7;

        std::vector<long long> dist;
        std::vector<int> parent;
        emulateSospRecompute(hg, dist, parent, 64);
        CHECK(dist[6] == 5 + 7 + 3 + 11 + 2 + 0,
              "forced: pre-delete target dist wrong (%lld)", dist[6]);

        HgBatch batch;
        batch.heDelete.push_back(4);
        H2HDelta delta;
        EscherHorizOps ops;
        hg.applyBatch(batch, {}, delta, ops);
        int it = emulateSospUpdate(hg, dist, parent, delta.seeds, 512);
        (void)it;
        CHECK(dist[4] >= HostHypergraph::INF / 2 &&
                  dist[5] >= HostHypergraph::INF / 2 &&
                  dist[6] >= HostHypergraph::INF / 2,
              "forced: disconnected side not INF");
        auto truth = hg.dijkstra(hg.sourceHe);
        CHECK(truth == dist, "forced: update != dijkstra after disconnect");
    }

    std::printf("local_tests: %d configs, %d batches, %d fallbacks, "
                "%d failures\n",
                CONFIGS, totalBatches, fallbacks, failures);
    return failures == 0 ? 0 : 1;
}
