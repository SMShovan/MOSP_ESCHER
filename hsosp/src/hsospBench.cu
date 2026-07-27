/**
 * @file hsospBench.cu
 * @brief End-to-end H-SOSP benchmark driver. One invocation runs the whole
 *        experiment matrix and appends one CSV row per measured batch.
 *
 * Usage:
 *   hsospBench --suite smoke --out results/results.csv
 *   hsospBench --suite full  --out results/results.csv
 *   hsospBench --suite full  --exp E1,E3 --reps 3 --seed 42
 *
 * Experiments (see docs/HSOSP.md):
 *   E1 time vs changed-batch size (hyperedge + incident-vertex batches)
 *   E2 time vs deletion percentage
 *   E3 scalability vs hypergraph size
 *   E4 effect of hyperedge cardinality
 *   E5 effect of h2h density (average line-graph degree)
 *   E7 change placement (random / targeted / near / far)
 * E6 (phase breakdown), E8 (speedup) and E9 (memory) are derived from the
 * CSV columns by scripts/plot_results.py; they need no dedicated runs.
 *
 * The CSV is flushed after every row so partial runs remain usable.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "DynamicHypergraph.hpp"
#include "HypergraphGen.hpp"
#include "hsosp.cuh"

using namespace escher_mosp;
using Clock = std::chrono::steady_clock;

namespace {

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}

struct CliOptions {
    std::string suite = "smoke";
    std::string out = "results/results.csv";
    std::string experiments = "E1,E2,E3,E4,E5,E7";
    int reps = 3;
    long long verifyMax = 300000;   ///< host-Dijkstra gate on hyperedge count
    int maxIterations = 512;
    unsigned long long seed = 20260725ull;
    bool listOnly = false;
};

struct DatasetCfg {
    std::string name;
    long long m = 0;        ///< real hyperedges
    int degTarget = 24;     ///< desired average h2h degree
    int cMin = 2, cMax = 16;
    int poolSize = 4096;
    double bridgeFrac = 0.05;
};

/** E[c^2] for c ~ U[cMin, cMax]; drives n = m * E[c^2] / degTarget. */
double expectedC2(int cMin, int cMax) {
    double mean = 0.5 * (cMin + cMax);
    double var = (std::pow(cMax - cMin + 1, 2.0) - 1.0) / 12.0;
    return var + mean * mean;
}

GenParams toGenParams(const DatasetCfg& d, unsigned long long seed) {
    GenParams g;
    g.numHyperedges = d.m;
    g.numVertices = static_cast<int>(
        std::max(1024.0, d.m * expectedC2(d.cMin, d.cMax) / d.degTarget));
    g.cMin = d.cMin;
    g.cMax = d.cMax;
    g.poolSize = d.poolSize;
    g.bridgeFrac = d.bridgeFrac;
    g.wMin = 1;
    g.wMax = 100;
    g.seed = seed;
    return g;
}

struct Scenario {
    std::string experiment;
    BatchKind kind = BatchKind::Hyperedge;
    int batchSize = 50000;
    double delPct = 50.0;
    Placement placement = Placement::Random;
};

unsigned long long mixSeed(unsigned long long a, unsigned long long b) {
    a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
    return a;
}

std::string nowString() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) {
        bool exists = false;
        {
            std::ifstream probe(path);
            exists = probe.good() && probe.peek() != EOF;
        }
        out_.open(path, std::ios::app);
        if (!out_.is_open()) {
            throw std::runtime_error("cannot open CSV for append: " + path);
        }
        if (!exists) {
            out_ << "timestamp,suite,experiment,dataset,m_hyperedges,"
                    "n_vertices,c_min,c_max,pool_size,bridge_frac,"
                    "avg_h2h_deg,h2h_pairs,batch_kind,batch_size,del_pct,"
                    "placement,rep,seed,t_escher_ms,t_delta_ms,"
                    "t_csr_apply_ms,t_sosp_update_ms,t_dynamic_total_ms,"
                    "t_static_ms,speedup,iters,fallback,seeds,max_frontier,"
                    "overflow_rebuilds,dev_mem_mb,escher_mb,graph_mb,"
                    "reachable_frac,verified,correct\n";
            out_.flush();
        }
    }
    std::ofstream& row() { return out_; }
    void endRow() {
        out_ << "\n";
        out_.flush();
    }

private:
    std::ofstream out_;
};

double deviceMemUsedMB() {
    std::size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return -1.0;
    return static_cast<double>(totalB - freeB) / (1024.0 * 1024.0);
}

/** Everything loaded for one dataset instance. */
struct LoadedDataset {
    GenParams gen;
    DynamicHypergraph dh;
    hsosp::DeviceH2H dev;
    hsosp::HsospState stateA;   // persistent dynamic state
    hsosp::HsospState stateB;   // static-recompute scratch
    double reachableFrac = 0.0;
    int overflowRebuilds = 0;

    LoadedDataset(int numVertices, const DynamicHypergraph::Caps& caps)
        : dh(numVertices, caps) {}
};

int globalFailures = 0;

std::unique_ptr<LoadedDataset> loadDataset(const DatasetCfg& d,
                                           const CliOptions& opt,
                                           unsigned long long seed,
                                           int plannedInsertTotal) {
    GenParams gen = toGenParams(d, seed);
    std::fprintf(stderr,
                 "[load] %s: m=%lld n=%d c=[%d,%d] pool=%d degTarget=%d\n",
                 d.name.c_str(), gen.numHyperedges, gen.numVertices, gen.cMin,
                 gen.cMax, gen.poolSize, d.degTarget);

    auto t0 = Clock::now();
    GeneratedHypergraph g = generateHypergraph(gen);

    DynamicHypergraph::Caps caps;
    caps.maxHyperedges =
        static_cast<int>(g.rows.size()) + plannedInsertTotal + 1024;
    caps.headroomFactor = 1.3;
    // Growth headroom: every planned insert appends into h2v/v2h/h2h.
    caps.extraPayloadInts =
        static_cast<long long>(plannedInsertTotal) *
            (gen.cMax + d.degTarget * 2 + 16) +
        (8LL << 20);

    auto ds = std::make_unique<LoadedDataset>(gen.numVertices, caps);
    ds->gen = gen;
    ds->dh.bulkLoad(std::move(g.rows), std::move(g.weights), g.sourceHe,
                    g.targetHe);
    double loadMs = msSince(t0);

    HostHypergraph& hg = ds->dh.host();
    std::fprintf(stderr,
                 "[load] %s: built in %.0f ms; h2h pairs=%lld avgDeg=%.1f\n",
                 d.name.c_str(), loadMs, hg.h2hPairCount,
                 hg.aliveCount ? 2.0 * hg.h2hPairCount / hg.aliveCount : 0.0);

    t0 = Clock::now();
    hsosp::buildDeviceH2H(ds->dev, hg, caps.maxHyperedges,
                          /*entryHeadroom=*/1.6);
    ds->stateA.allocate(caps.maxHyperedges);
    ds->stateB.allocate(caps.maxHyperedges);
    hsosp::UpdateConfig ucfg;
    ucfg.maxIterations = opt.maxIterations;
    hsosp::UpdateStats is =
        hsosp::hsospRecompute(ds->dev, ds->stateA, hg.sourceHe, ucfg);
    std::fprintf(stderr,
                 "[load] %s: device build+initial SOSP in %.0f ms "
                 "(%d iterations)\n",
                 d.name.c_str(), msSince(t0), is.iterations);

    // Reachable fraction from the initial solve.
    {
        std::vector<long long> dist;
        ds->stateA.downloadDistances(dist, hg.maxId());
        long long reach = 0;
        for (int i = 0; i < hg.maxId(); ++i) {
            if (dist[i] < HostHypergraph::INF / 2) ++reach;
        }
        ds->reachableFrac =
            hg.aliveCount ? static_cast<double>(reach) / hg.aliveCount : 0.0;
        std::fprintf(stderr, "[load] %s: reachable fraction %.3f\n",
                     d.name.c_str(), ds->reachableFrac);
    }
    return ds;
}

void runScenario(LoadedDataset& ds, const DatasetCfg& dcfg,
                 const Scenario& sc, const CliOptions& opt, CsvWriter& csv) {
    HostHypergraph& hg = ds.dh.host();
    hsosp::UpdateConfig ucfg;
    ucfg.maxIterations = opt.maxIterations;

    for (int rep = 0; rep < opt.reps; ++rep) {
        unsigned long long bseed = mixSeed(
            opt.seed, std::hash<std::string>{}(dcfg.name + sc.experiment +
                                              toString(sc.kind) +
                                              toString(sc.placement)) +
                          sc.batchSize * 131 +
                          static_cast<int>(sc.delPct) * 7 + rep);

        // Placement-aware batch generation needs the current SOSP state.
        std::vector<long long> distSnapshot;
        std::vector<int> parentSnapshot;
        if (sc.placement != Placement::Random) {
            ds.stateA.downloadDistances(distSnapshot, hg.maxId());
            ds.stateA.downloadParents(parentSnapshot, hg.maxId());
        }
        BatchParams bp;
        bp.size = sc.batchSize;
        bp.delPct = sc.delPct;
        bp.kind = sc.kind;
        bp.placement = sc.placement;
        bp.seed = bseed;
        HgBatch batch =
            generateBatch(hg, ds.gen, bp, distSnapshot, parentSnapshot);

        // ---- dynamic pipeline (timed) ----------------------------------
        double csrMs = 0.0, sospMs = 0.0;
        DynamicHypergraph::BatchResult br;
        try {
            br = ds.dh.applyBatch(batch);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s %s: applyBatch failed: %s\n",
                         dcfg.name.c_str(), sc.experiment.c_str(), e.what());
            ++globalFailures;
            return;
        }

        auto t0 = Clock::now();
        bool ok = hsosp::applyDeltaToDevice(ds.dev, hg, br.delta);
        csrMs = msSince(t0);
        if (!ok) {
            // Tail exhausted: rebuild from the shadow (counted, not timed
            // as part of the update) and re-run this batch's CSR stage as
            // a rebuild (the rebuild itself installs the post-batch state).
            ++ds.overflowRebuilds;
            std::fprintf(stderr,
                         "[warn] %s: device CSR overflow, rebuilding\n",
                         dcfg.name.c_str());
            hsosp::buildDeviceH2H(ds.dev, hg, ds.stateA.maxNodes, 1.6);
        }

        t0 = Clock::now();
        hsosp::UpdateStats us =
            hsosp::hsospUpdate(ds.dev, ds.stateA, br.delta.seeds,
                               br.delta.deadHe, hg.sourceHe, ucfg);
        sospMs = msSince(t0);

        // ---- static baseline (timed) -----------------------------------
        t0 = Clock::now();
        hsosp::UpdateStats rs =
            hsosp::hsospRecompute(ds.dev, ds.stateB, hg.sourceHe, ucfg);
        double staticMs = msSince(t0);
        (void)rs;

        // ---- correctness ----------------------------------------------
        long long mismatches =
            hsosp::compareDistances(ds.stateA, ds.stateB, hg.maxId());
        bool verified = true;
        bool correct = (mismatches == 0);
        if (hg.maxId() <= opt.verifyMax) {
            std::vector<long long> got;
            ds.stateA.downloadDistances(got, hg.maxId());
            std::vector<long long> truth = hg.dijkstra(hg.sourceHe);
            for (int i = 0; i < hg.maxId(); ++i) {
                if (got[i] != truth[i]) {
                    correct = false;
                    break;
                }
            }
        }
        if (!correct) {
            ++globalFailures;
            std::fprintf(stderr,
                         "[FAIL] %s %s rep %d: distance mismatch "
                         "(%lld vs static)\n",
                         dcfg.name.c_str(), sc.experiment.c_str(), rep,
                         mismatches);
        }

        const double dynTotal = br.escherMs + br.deltaMs + csrMs + sospMs;
        const double avgDeg =
            hg.aliveCount ? 2.0 * hg.h2hPairCount / hg.aliveCount : 0.0;

        auto& o = csv.row();
        o << nowString() << "," << opt.suite << "," << sc.experiment << ","
          << dcfg.name << "," << ds.gen.numHyperedges << ","
          << ds.gen.numVertices << "," << ds.gen.cMin << "," << ds.gen.cMax
          << "," << ds.gen.poolSize << "," << ds.gen.bridgeFrac << ","
          << avgDeg << "," << hg.h2hPairCount << "," << toString(sc.kind)
          << "," << sc.batchSize << "," << sc.delPct << ","
          << toString(sc.placement) << "," << rep << "," << bseed << ","
          << br.escherMs << "," << br.deltaMs << "," << csrMs << ","
          << sospMs << "," << dynTotal << "," << staticMs << ","
          << (dynTotal > 0 ? staticMs / dynTotal : 0.0) << ","
          << us.iterations << "," << (us.fallbackRecompute ? 1 : 0) << ","
          << us.seedCount << "," << us.maxFrontier << ","
          << ds.overflowRebuilds << "," << deviceMemUsedMB() << ","
          << ds.dh.escherDeviceBytes() / (1024.0 * 1024.0) << ","
          << (ds.dev.deviceBytes() + ds.stateA.deviceBytes() +
              ds.stateB.deviceBytes()) /
                 (1024.0 * 1024.0)
          << "," << ds.reachableFrac << "," << (verified ? 1 : 0) << ","
          << (correct ? 1 : 0);
        csv.endRow();

        std::fprintf(stderr,
                     "[row] %s %s %s dE=%d del=%.0f %s rep=%d: "
                     "dyn=%.1fms (escher=%.1f delta=%.1f csr=%.1f "
                     "sosp=%.1f) static=%.1fms speedup=%.2f iters=%d%s\n",
                     dcfg.name.c_str(), sc.experiment.c_str(),
                     toString(sc.kind), sc.batchSize, sc.delPct,
                     toString(sc.placement), rep, dynTotal, br.escherMs,
                     br.deltaMs, csrMs, sospMs, staticMs,
                     dynTotal > 0 ? staticMs / dynTotal : 0.0, us.iterations,
                     us.fallbackRecompute ? " FALLBACK" : "");
    }
}

bool wantExp(const CliOptions& opt, const char* e) {
    return opt.experiments.find(e) != std::string::npos;
}

} // namespace

int main(int argc, char** argv) {
    CliOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--suite") opt.suite = next("--suite");
        else if (a == "--out") opt.out = next("--out");
        else if (a == "--exp") opt.experiments = next("--exp");
        else if (a == "--reps") opt.reps = std::stoi(next("--reps"));
        else if (a == "--verify-max")
            opt.verifyMax = std::stoll(next("--verify-max"));
        else if (a == "--maxiter")
            opt.maxIterations = std::stoi(next("--maxiter"));
        else if (a == "--seed") opt.seed = std::stoull(next("--seed"));
        else if (a == "--list") opt.listOnly = true;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    const bool smoke = (opt.suite == "smoke");

    // ---- dataset tables --------------------------------------------------
    std::vector<DatasetCfg> mainSets;
    if (smoke) {
        mainSets.push_back({"SM-A", 20000, 16, 2, 8, 512, 0.05});
        mainSets.push_back({"SM-B", 50000, 24, 2, 16, 1024, 0.05});
    } else {
        mainSets.push_back({"HG-S", 1000000, 24, 2, 16, 4096, 0.05});
        mainSets.push_back({"HG-M", 5000000, 24, 2, 16, 8192, 0.05});
        mainSets.push_back({"HG-L", 10000000, 16, 2, 8, 8192, 0.05});
        mainSets.push_back({"HG-XL", 16000000, 12, 2, 8, 16384, 0.05});
        mainSets.push_back({"HG-C", 2000000, 24, 2, 64, 32768, 0.05});
    }
    const std::vector<int> batchSizes =
        smoke ? std::vector<int>{2000, 5000}
              : std::vector<int>{25000, 50000, 100000, 200000};
    const std::vector<double> delPcts =
        smoke ? std::vector<double>{25, 75}
              : std::vector<double>{20, 40, 60, 80};
    const int defBatch = smoke ? 5000 : 50000;

    if (opt.listOnly) {
        std::printf("suite=%s datasets=%zu experiments=%s reps=%d\n",
                    opt.suite.c_str(), mainSets.size(),
                    opt.experiments.c_str(), opt.reps);
        return 0;
    }

    // Ensure the output directory exists (best effort).
    {
        auto slash = opt.out.find_last_of('/');
        if (slash != std::string::npos) {
            std::string dir = "mkdir -p " + opt.out.substr(0, slash);
            int rc = std::system(dir.c_str());
            (void)rc;
        }
    }
    CsvWriter csv(opt.out);

    // Upper bound on inserted hyperedges per dataset instance, for
    // capacity planning (E1 + E2 + E7 share one instance).
    auto plannedInserts = [&](bool isMain) {
        long long total = 0;
        if (wantExp(opt, "E1")) {
            for (int b : batchSizes) total += 2LL * b * opt.reps;   // 2 kinds
        }
        if (isMain && wantExp(opt, "E2")) {
            for (double d : delPcts)
                total += static_cast<long long>(defBatch * (1.0 - d / 100.0) *
                                                opt.reps) + defBatch;
        }
        if (isMain && wantExp(opt, "E7")) total += 4LL * defBatch * opt.reps;
        return static_cast<int>(std::min<long long>(total + 65536,
                                                    50000000LL));
    };

    auto tStart = Clock::now();

    // ---- E1 / E2 / E7 on the named datasets ------------------------------
    if (wantExp(opt, "E1") || wantExp(opt, "E2") || wantExp(opt, "E7")) {
        for (std::size_t di = 0; di < mainSets.size(); ++di) {
            const DatasetCfg& d = mainSets[di];
            std::unique_ptr<LoadedDataset> ds;
            try {
                ds = loadDataset(d, opt, mixSeed(opt.seed, di + 1),
                                 plannedInserts(true));
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[error] load %s failed: %s\n",
                             d.name.c_str(), e.what());
                ++globalFailures;
                continue;
            }

            if (wantExp(opt, "E1")) {
                for (BatchKind kind :
                     {BatchKind::Hyperedge, BatchKind::Vertex}) {
                    for (int b : batchSizes) {
                        Scenario sc;
                        sc.experiment = "E1";
                        sc.kind = kind;
                        sc.batchSize = b;
                        sc.delPct = 50.0;
                        runScenario(*ds, d, sc, opt, csv);
                    }
                }
            }
            if (wantExp(opt, "E2")) {
                for (double del : delPcts) {
                    Scenario sc;
                    sc.experiment = "E2";
                    sc.batchSize = defBatch;
                    sc.delPct = del;
                    runScenario(*ds, d, sc, opt, csv);
                }
            }
            if (wantExp(opt, "E7") && (d.name == "HG-M" || smoke)) {
                for (Placement pl : {Placement::Random, Placement::Targeted,
                                     Placement::Near, Placement::Far}) {
                    Scenario sc;
                    sc.experiment = "E7";
                    sc.batchSize = defBatch;
                    sc.delPct = 50.0;
                    sc.placement = pl;
                    runScenario(*ds, d, sc, opt, csv);
                }
            }
        }
    }

    // ---- E3: size scaling ------------------------------------------------
    if (wantExp(opt, "E3")) {
        std::vector<long long> sizes =
            smoke ? std::vector<long long>{20000, 50000}
                  : std::vector<long long>{1000000, 2000000, 5000000,
                                           10000000, 16000000};
        for (std::size_t si = 0; si < sizes.size(); ++si) {
            DatasetCfg d{"E3-" + std::to_string(sizes[si] / 1000000) + "M",
                         sizes[si], 16, 2, 8, 8192, 0.05};
            if (smoke)
                d.name = "E3-" + std::to_string(sizes[si] / 1000) + "K";
            try {
                auto ds = loadDataset(d, opt, mixSeed(opt.seed, 100 + si),
                                      defBatch * opt.reps + 65536);
                Scenario sc;
                sc.experiment = "E3";
                sc.batchSize = defBatch;
                sc.delPct = 50.0;
                runScenario(*ds, d, sc, opt, csv);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[error] E3 %s: %s\n", d.name.c_str(),
                             e.what());
                ++globalFailures;
            }
        }
    }

    // ---- E4: cardinality -------------------------------------------------
    if (wantExp(opt, "E4")) {
        std::vector<int> cmaxes = smoke ? std::vector<int>{8, 32}
                                        : std::vector<int>{8, 16, 32, 64, 128};
        for (std::size_t ci = 0; ci < cmaxes.size(); ++ci) {
            DatasetCfg d{"E4-c" + std::to_string(cmaxes[ci]),
                         smoke ? 20000LL : 2000000LL, 24, 2, cmaxes[ci],
                         32768, 0.05};
            try {
                auto ds = loadDataset(d, opt, mixSeed(opt.seed, 200 + ci),
                                      defBatch * opt.reps + 65536);
                Scenario sc;
                sc.experiment = "E4";
                sc.batchSize = defBatch;
                sc.delPct = 50.0;
                runScenario(*ds, d, sc, opt, csv);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[error] E4 %s: %s\n", d.name.c_str(),
                             e.what());
                ++globalFailures;
            }
        }
    }

    // ---- E5: h2h density -------------------------------------------------
    if (wantExp(opt, "E5")) {
        std::vector<int> degs = smoke ? std::vector<int>{8, 32}
                                      : std::vector<int>{8, 16, 32, 64};
        for (std::size_t gi = 0; gi < degs.size(); ++gi) {
            DatasetCfg d{"E5-d" + std::to_string(degs[gi]),
                         smoke ? 20000LL : 2000000LL, degs[gi], 2, 16, 8192,
                         0.05};
            try {
                auto ds = loadDataset(d, opt, mixSeed(opt.seed, 300 + gi),
                                      defBatch * opt.reps + 65536);
                Scenario sc;
                sc.experiment = "E5";
                sc.batchSize = defBatch;
                sc.delPct = 50.0;
                runScenario(*ds, d, sc, opt, csv);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[error] E5 %s: %s\n", d.name.c_str(),
                             e.what());
                ++globalFailures;
            }
        }
    }

    std::fprintf(stderr, "[done] total wall time %.1f s, failures=%d\n",
                 msSince(tStart) / 1000.0, globalFailures);
    if (globalFailures > 0) {
        std::fprintf(stderr,
                     "hsospBench: %d scenario(s) failed correctness or "
                     "capacity checks\n",
                     globalFailures);
        return 1;
    }
    return 0;
}
