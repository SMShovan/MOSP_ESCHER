/**
 * @file local_tests.cpp
 * @brief GPU-free validation of the hypergraph host core.
 *
 * Compiled with plain g++ (see tests/local/Makefile). Validates, over many
 * randomized configurations:
 *   1. HostHypergraph::lineGraph (used to build the device CSR) == the
 *      brute-force line graph == the oracle's rebuild;
 *   2. after every batch (all four op kinds) the incidence model is
 *      consistent (v2h is the transpose of heVerts) and the shipped
 *      IncidenceBatch describes it (pre / post rows, post lengths);
 *   3. the GPU's delta rule, run here on the host from the IncidenceBatch
 *      (candidates from the incidence lists of every changed incidence
 *      before and after the batch, classified by pre / post overlap),
 *      gives exactly the difference of the line graphs before and after;
 *   4. the sequential emulation of the device SOSP update matches Dijkstra
 *      and the canonical tree after every batch (including disconnections
 *      and id recycling).
 *
 * The tests in tests/unit/*.cu run the same checks through the real ESCHER
 * + CUDA path.
 */

#include "HostHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "HypergraphOracle.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

static std::vector<std::vector<int>> rowsOf(const LineGraphCSR& lg) {
    std::vector<std::vector<int>> out(lg.numIds);
    for (int id = 1; id <= lg.numIds; ++id)
        out[id - 1].assign(lg.row(id), lg.row(id) + lg.degree(id));
    return out;
}

static bool overlap(const int* a, int la, const int* b, int lb) {
    int i = 0, j = 0;
    while (i < la && j < lb) {
        if (a[i] == b[j]) return true;
        if (a[i] < b[j]) ++i;
        else ++j;
    }
    return false;
}

/** Host version of the GPU delta rule (hsospDelta.cu deriveDelta). */
static H2HDelta deriveOnHost(const std::vector<std::vector<int>>& preV2h,
                             const HostHypergraph& hg,
                             const std::vector<std::vector<int>>& preHeVerts,
                             const IncidenceBatch& inc) {
    // Net sign per (vertex, hyperedge).
    std::map<std::uint64_t, int> net;
    for (std::size_t i = 0; i < inc.incKey.size(); ++i)
        net[inc.incKey[i]] += inc.incSign[i];
    std::set<std::pair<int, int>> cand;
    for (auto [key, sign] : net) {
        if (sign == 0) continue;
        const int v = static_cast<int>(key >> 32);
        const int h = static_cast<int>(key & 0xffffffffu);
        for (int o : preV2h[v])
            if (o != h) cand.emplace(std::min(h, o), std::max(h, o));
        for (int o : hg.v2h[v])
            if (o != h) cand.emplace(std::min(h, o), std::max(h, o));
    }
    auto preRow = [&](int id) -> const std::vector<int>& {
        static const std::vector<int> empty;
        return id <= static_cast<int>(preHeVerts.size()) ? preHeVerts[id - 1]
                                                         : empty;
    };
    auto postRow = [&](int id) -> const std::vector<int>& {
        static const std::vector<int> empty;
        return hg.alive[id - 1] ? hg.heVerts[id - 1] : empty;
    };
    H2HDelta d;
    for (auto [a, b] : cand) {
        const auto &pa = preRow(a), &pb = preRow(b);
        const auto &qa = postRow(a), &qb = postRow(b);
        const bool pre = overlap(pa.data(), static_cast<int>(pa.size()),
                                 pb.data(), static_cast<int>(pb.size()));
        const bool post = overlap(qa.data(), static_cast<int>(qa.size()),
                                  qb.data(), static_cast<int>(qb.size()));
        if (post && !pre) d.insEdges.emplace_back(a, b);
        if (pre && !post) d.delEdges.emplace_back(a, b);
    }
    return d;
}

/** Distances and 1-based parents against the oracle (canonical tree). */
static void checkAgainstOracle(const HostHypergraph& hg,
                               const LineGraphCSR& lg,
                               const std::vector<long long>& dist,
                               const std::vector<int>& parent, int cfg,
                               int bi) {
    std::vector<int> parent0(parent.size());
    for (std::size_t i = 0; i < parent.size(); ++i)
        parent0[i] = parent[i] >= 1 ? parent[i] - 1 : -1;
    SospCheck c = checkSosp(hg, lg, referenceDistances(hg, lg, hg.sourceHe),
                            dist, parent0);
    CHECK(c.ok(),
          "cfg %d batch %d: %lld distance mismatches (first he %d: got %lld "
          "want %lld), %lld parent errors",
          cfg, bi, c.distMismatches, c.firstBadId, c.firstGot, c.firstWant,
          c.parentErrors);
}

int main() {
    std::mt19937_64 metaRng(20260725);

    const int CONFIGS = 60;
    int totalBatches = 0;

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
        hg.buildFrom(g.numVertices, std::move(g.rows), std::move(g.weights));
        hg.sourceHe = g.sourceHe;
        hg.targetHe = g.targetHe;

        // ---- 1. line graph == brute force == oracle rebuild ------------
        LineGraphCSR lg = hg.lineGraph();
        CHECK(rowsOf(lg) == hg.bruteForceH2H(),
              "cfg %d: lineGraph != brute force", cfg);
        CHECK(rowsOf(lg) == rowsOf(rebuildLineGraph(hg)),
              "cfg %d: lineGraph != oracle rebuild", cfg);

        // Initial SOSP state via recompute (the device does the same).
        std::vector<long long> dist;
        std::vector<int> parent;
        emulateSospRecompute(hg, lg, dist, parent, hg.maxId() + 2);
        checkAgainstOracle(hg, lg, dist, parent, cfg, -1);

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

            const LineGraphCSR pre = lg;
            const auto preV2h = hg.v2h;
            std::vector<std::vector<int>> preHeVerts(hg.maxId());
            for (int id = 1; id <= hg.maxId(); ++id)
                if (hg.alive[id - 1]) preHeVerts[id - 1] = hg.heVerts[id - 1];

            IncidenceBatch inc;
            EscherHorizOps ops;
            hg.applyBatch(batch, finalIds, inc, ops);
            ++totalBatches;
            lg = hg.lineGraph();

            // ---- 2. incidence model and the shipped batch --------------
            {
                std::vector<std::vector<int>> v2h(hg.numVertices);
                for (int id = 1; id <= hg.maxId(); ++id)
                    if (hg.alive[id - 1])
                        for (int v : hg.heVerts[id - 1]) v2h[v].push_back(id);
                bool same = true;
                for (int v = 0; v < hg.numVertices; ++v) {
                    std::vector<int> got = hg.v2h[v];
                    std::sort(got.begin(), got.end());
                    same = same && got == v2h[v];
                }
                CHECK(same, "cfg %d batch %d: v2h != transpose of heVerts",
                      cfg, bi);
                for (std::size_t t = 0; t < inc.touched.size(); ++t) {
                    const int id = inc.touched[t];
                    std::vector<int> preR(inc.preVals.begin() + inc.preOff[t],
                                          inc.preVals.begin() +
                                              inc.preOff[t + 1]);
                    std::vector<int> postR(
                        inc.postVals.begin() + inc.postOff[t],
                        inc.postVals.begin() + inc.postOff[t + 1]);
                    const std::vector<int> wantPre =
                        id <= static_cast<int>(preHeVerts.size())
                            ? preHeVerts[id - 1]
                            : std::vector<int>{};
                    const std::vector<int> wantPost =
                        hg.alive[id - 1] ? hg.heVerts[id - 1]
                                         : std::vector<int>{};
                    CHECK(preR == wantPre && postR == wantPost,
                          "cfg %d batch %d: shipped rows of he %d wrong", cfg,
                          bi, id);
                }
                for (std::size_t i = 0; i < inc.touchedVertices.size(); ++i)
                    CHECK(inc.touchedVertexLen[i] ==
                              static_cast<int>(
                                  hg.v2h[inc.touchedVertices[i]].size()),
                          "cfg %d batch %d: post length of vertex %d wrong",
                          cfg, bi, inc.touchedVertices[i]);
            }

            // ---- 3. delta rule == difference of the line graphs --------
            H2HDelta delta = lineGraphDelta(pre, lg);
            {
                H2HDelta derived = deriveOnHost(preV2h, hg, preHeVerts, inc);
                std::sort(derived.insEdges.begin(), derived.insEdges.end());
                std::sort(derived.delEdges.begin(), derived.delEdges.end());
                CHECK(derived.insEdges == delta.insEdges &&
                          derived.delEdges == delta.delEdges,
                      "cfg %d batch %d: derived delta (+%zu -%zu) != line "
                      "graph difference (+%zu -%zu)",
                      cfg, bi, derived.insEdges.size(),
                      derived.delEdges.size(), delta.insEdges.size(),
                      delta.delEdges.size());
            }

            // ---- 4. dynamic update == dijkstra (canonical tree) --------
            delta.newHe = inc.newHe;
            delta.deadHe = inc.deadHe;
            emulateSospUpdate(hg, lg, dist, parent, delta);
            checkAgainstOracle(hg, lg, dist, parent, cfg, bi);
        }
    }

    // ---- forced disconnection scenario ---------------------------------
    {
        // Two pools joined by a single bridge hyperedge; deleting the
        // bridge must drive the far side to INF.
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

        LineGraphCSR lg = hg.lineGraph();
        std::vector<long long> dist;
        std::vector<int> parent;
        emulateSospRecompute(hg, lg, dist, parent, 64);
        CHECK(dist[6] == 5 + 7 + 3 + 11 + 2 + 0,
              "forced: pre-delete target dist wrong (%lld)", dist[6]);

        HgBatch batch;
        batch.heDelete.push_back(4);
        IncidenceBatch inc;
        EscherHorizOps ops;
        const LineGraphCSR pre = lg;
        hg.applyBatch(batch, {}, inc, ops);
        lg = hg.lineGraph();
        H2HDelta delta = lineGraphDelta(pre, lg);
        delta.newHe = inc.newHe;
        delta.deadHe = inc.deadHe;
        emulateSospUpdate(hg, lg, dist, parent, delta);
        CHECK(dist[4] >= HostHypergraph::INF / 2 &&
                  dist[5] >= HostHypergraph::INF / 2 &&
                  dist[6] >= HostHypergraph::INF / 2,
              "forced: disconnected side not INF");
        checkAgainstOracle(hg, lg, dist, parent, -1, 0);
    }

    // ---- ops on the virtual source / target are skipped ----------------
    // (Deleting the source left a dead node at distance 0 and recycled its
    // id for a weighted hyperedge that then acted as the source; vertex
    // ops on the target made it a free bridge.)
    {
        // (Two vertices in each virtual row, so that a vertex deletion is
        // not already refused for emptying the row.)
        HostHypergraph hg;
        hg.buildFrom(8, {{0, 5}, {0, 1}, {1, 2}, {2, 3}, {3, 7}},
                     {0, 2, 3, 4, 0});
        hg.sourceHe = 1;
        hg.targetHe = 5;
        HgBatch batch;
        batch.heDelete = {1, 5};
        batch.vtxInsert = {{5, 6}, {1, 6}, {4, 6}};
        batch.vtxDelete = {{5, 7}, {1, 5}};
        batch.heInsert.push_back({{2, 3}, 50});
        IncidenceBatch inc;
        EscherHorizOps ops;
        const std::vector<int> ids = hg.reserveIds(1);
        hg.applyBatch(batch, ids, inc, ops);
        CHECK(inc.skippedOps == 6, "virtual: %d ops skipped, want 6",
              inc.skippedOps);
        CHECK(hg.alive[0] && hg.alive[4] &&
                  hg.heVerts[0] == (std::vector<int>{0, 5}) &&
                  hg.heVerts[4] == (std::vector<int>{3, 7}),
              "virtual: source or target changed");
        CHECK(ids[0] == 6 && hg.freeIds.empty(),
              "virtual: inserted hyperedge got id %d", ids[0]);
        CHECK(hg.heVerts[3] == (std::vector<int>{2, 3, 6}),
              "virtual: vertex insert into a real hyperedge lost");
        LineGraphCSR lg = hg.lineGraph();
        std::vector<long long> dist;
        std::vector<int> parent;
        emulateSospRecompute(hg, lg, dist, parent, 64);
        CHECK(dist[0] == 0 && dist[4] == 2 + 3 + 4,
              "virtual: dist(source) %lld, dist(target) %lld", dist[0],
              dist[4]);
        checkAgainstOracle(hg, lg, dist, parent, -2, 0);
    }

    // ---- batches for a hypergraph without real hyperedges ---------------
    // (A vertex batch read pool[0] of an empty pool.)
    {
        HostHypergraph hg;
        hg.buildFrom(4, {{0}, {3}}, {0, 0});
        hg.sourceHe = 1;
        hg.targetHe = 2;
        GenParams gp;
        gp.numVertices = 4;
        gp.poolSize = 4;
        BatchParams bp;
        bp.size = 5;
        bp.delPct = 50;
        bp.kind = BatchKind::Vertex;
        HgBatch b = generateBatch(hg, gp, bp, {}, {});
        CHECK(b.totalOps() == 0, "empty pool: vertex batch has %zu ops",
              b.totalOps());
        bp.kind = BatchKind::Hyperedge;
        b = generateBatch(hg, gp, bp, {}, {});
        CHECK(b.heDelete.empty() && !b.heInsert.empty(),
              "empty pool: hyperedge batch has %zu deletions, %zu "
              "insertions", b.heDelete.size(), b.heInsert.size());
    }

    // ---- real-hypergraph loader: 64-bit vertex ids -----------------------
    // (Ids were truncated to int: 4294967297 became 1 and an out-of-range
    // token became -1, merging distinct vertices.)
    {
        const std::string path =
            (std::filesystem::temp_directory_path() /
             ("local_tests_" + std::to_string(::getpid()) + ".hg"))
                .string();
        auto load = [&](const char* text, GeneratedHypergraph& g) {
            {
                std::ofstream out(path);
                out << text;
            }
            try {
                g = loadHypergraphFile(path, 100, 1);
            } catch (const std::runtime_error&) {
                return false;
            }
            return true;
        };
        // Rows 2 and 3 are the file's first two lines (row 1: virtual s);
        // a row lists its vertices in order of raw id.
        auto disjoint = [](const GeneratedHypergraph& g) {
            std::vector<int> a = g.rows[1], b = g.rows[2];
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            std::vector<int> common;
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                                  std::back_inserter(common));
            return common.empty();
        };
        GeneratedHypergraph g;
        CHECK(load("4294967297 5\n1 7\n", g) && g.numVertices == 4 &&
                  disjoint(g),
              "loader: ids 4294967297 and 1 merged (n = %d)", g.numVertices);
        CHECK(load("-9223372036854775808 5\n9223372036854775807 5 -1\n",
                   g) &&
                  g.numVertices == 4,
              "loader: 64-bit extremes: n = %d, want 4", g.numVertices);
        CHECK(!load("99999999999999999999 5\n-1 8\n", g),
              "loader: id beyond 64 bits accepted");
        CHECK(!load("12 x5\n", g), "loader: non-numeric token accepted");
        CHECK(load("-3 5\n2 7\n5 -3 2\n", g) && g.numVertices == 4 &&
                  disjoint(g) && g.rows[3] == (std::vector<int>{0, 2, 1}),
              "loader: in-range ids: n = %d, want 4", g.numVertices);
        std::remove(path.c_str());
    }

    std::printf("local_tests: %d configs, %d batches, "
                "%d failures\n",
                CONFIGS, totalBatches, failures);
    return failures == 0 ? 0 : 1;
}
