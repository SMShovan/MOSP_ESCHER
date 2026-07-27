/**
 * @file HypergraphGen.cpp
 * @brief Seeded pool-model hypergraph generator + change-batch generator.
 */

#include "HypergraphGen.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <random>
#include <unordered_set>

namespace escher_mosp {

namespace {

/** Sample @p count distinct ints from [lo, hi] into @p out. */
void sampleDistinct(std::mt19937_64& rng, int lo, int hi, int count,
                    std::vector<int>& out) {
    out.clear();
    const int range = hi - lo + 1;
    if (count >= range) {
        for (int v = lo; v <= hi; ++v) out.push_back(v);
        return;
    }
    std::unordered_set<int> used;
    std::uniform_int_distribution<int> dist(lo, hi);
    while (static_cast<int>(out.size()) < count) {
        int v = dist(rng);
        if (used.insert(v).second) out.push_back(v);
    }
}

} // namespace

const char* toString(BatchKind k) {
    return k == BatchKind::Hyperedge ? "hyperedge" : "vertex";
}

const char* toString(Placement p) {
    switch (p) {
        case Placement::Random:   return "random";
        case Placement::Targeted: return "targeted";
        case Placement::Near:     return "near";
        case Placement::Far:      return "far";
    }
    return "?";
}

GeneratedHypergraph generateHypergraph(const GenParams& p) {
    GeneratedHypergraph g;
    g.numVertices = p.numVertices;
    g.sourceVertex = 0;
    g.targetVertex = p.numVertices - 1;

    std::mt19937_64 rng(p.seed);
    const int numPools =
        std::max(1, (p.numVertices + p.poolSize - 1) / p.poolSize);
    std::uniform_int_distribution<int> poolDist(0, numPools - 1);
    std::uniform_int_distribution<int> cardDist(p.cMin, p.cMax);
    std::uniform_int_distribution<long long> wDist(p.wMin, p.wMax);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const long long m = p.numHyperedges;
    g.rows.reserve(static_cast<std::size_t>(m) + 2);
    g.weights.reserve(static_cast<std::size_t>(m) + 2);

    // Row 1: virtual source hyperedge {s}, weight 0 (meeting notes, Step 1).
    g.rows.push_back({g.sourceVertex});
    g.weights.push_back(0);

    std::vector<int> verts;
    for (long long i = 0; i < m; ++i) {
        const int pool = poolDist(rng);
        const int lo = pool * p.poolSize;
        const int hi = std::min(p.numVertices - 1, lo + p.poolSize - 1);
        int c = cardDist(rng);
        c = std::max(1, std::min(c, hi - lo + 1));
        sampleDistinct(rng, lo, hi, c, verts);

        // Bridge: swap one member for a vertex of the next pool.
        if (numPools > 1 && unif(rng) < p.bridgeFrac) {
            const int nPool = (pool + 1) % numPools;
            const int nLo = nPool * p.poolSize;
            const int nHi = std::min(p.numVertices - 1, nLo + p.poolSize - 1);
            std::uniform_int_distribution<int> nv(nLo, nHi);
            verts[0] = nv(rng);
        }
        g.rows.push_back(verts);
        g.weights.push_back(wDist(rng));
    }

    // Guarantee that the source / target vertices appear in at least one
    // real hyperedge so the virtual nodes are not isolated.
    if (m >= 1) {
        g.rows[1].push_back(g.sourceVertex);
        g.rows[static_cast<std::size_t>(m)].push_back(g.targetVertex);
    }

    // Last row: virtual target hyperedge {t}, weight 0.
    g.rows.push_back({g.targetVertex});
    g.weights.push_back(0);

    g.sourceHe = 1;
    g.targetHe = static_cast<int>(g.rows.size());
    return g;
}

bool writeHypergraphText(const GeneratedHypergraph& g,
                         const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << g.numVertices << " " << g.rows.size() << "\n";
    for (std::size_t i = 0; i < g.rows.size(); ++i) {
        out << g.weights[i] << " " << g.rows[i].size();
        for (int v : g.rows[i]) out << " " << v;
        out << "\n";
    }
    return static_cast<bool>(out);
}

HgBatch generateBatch(const HostHypergraph& hg, const GenParams& gen,
                      const BatchParams& bp,
                      const std::vector<long long>& dist,
                      const std::vector<int>& parent) {
    HgBatch batch;
    std::mt19937_64 rng(bp.seed);

    const int m = hg.maxId();
    const int nDel = static_cast<int>(bp.size * bp.delPct / 100.0 + 0.5);
    const int nIns = bp.size - nDel;

    // ---- Candidate pool of deletable / modifiable hyperedges -------------
    // Excludes the virtual source and target.
    std::vector<int> pool;
    pool.reserve(hg.aliveCount);
    if (bp.placement == Placement::Random) {
        for (int id = 1; id <= m; ++id) {
            if (hg.alive[id - 1] && id != hg.sourceHe && id != hg.targetHe)
                pool.push_back(id);
        }
    } else if (bp.placement == Placement::Targeted) {
        // Hyperedges that are SOSP-tree parents: deleting them is
        // guaranteed to disturb the tree (DynaMOSP-style targeted changes).
        std::vector<std::uint8_t> isParent(m + 1, 0);
        for (int id = 1; id <= m; ++id) {
            int p = (id - 1 < static_cast<int>(parent.size()))
                        ? parent[id - 1] : -1;
            if (p >= 1 && p <= m) isParent[p] = 1;
        }
        for (int id = 1; id <= m; ++id) {
            if (hg.alive[id - 1] && isParent[id] && id != hg.sourceHe &&
                id != hg.targetHe)
                pool.push_back(id);
        }
    } else {
        // Near / Far by distance quartile among reachable nodes.
        std::vector<long long> finite;
        for (int id = 1; id <= m; ++id) {
            if (hg.alive[id - 1] && id - 1 < static_cast<int>(dist.size()) &&
                dist[id - 1] < HostHypergraph::INF / 2)
                finite.push_back(dist[id - 1]);
        }
        std::sort(finite.begin(), finite.end());
        long long q = 0;
        if (!finite.empty()) {
            std::size_t k = (bp.placement == Placement::Near)
                                ? finite.size() / 4
                                : (finite.size() * 3) / 4;
            if (k >= finite.size()) k = finite.size() - 1;
            q = finite[k];
        }
        for (int id = 1; id <= m; ++id) {
            if (!hg.alive[id - 1] || id == hg.sourceHe || id == hg.targetHe)
                continue;
            if (id - 1 >= static_cast<int>(dist.size())) continue;
            long long d = dist[id - 1];
            if (d >= HostHypergraph::INF / 2) continue;
            if (bp.placement == Placement::Near ? (d <= q) : (d >= q))
                pool.push_back(id);
        }
    }
    if (pool.empty()) {
        for (int id = 1; id <= m; ++id) {
            if (hg.alive[id - 1] && id != hg.sourceHe && id != hg.targetHe)
                pool.push_back(id);
        }
    }
    std::uniform_int_distribution<std::size_t> poolPick(0, pool.size() - 1);

    if (bp.kind == BatchKind::Hyperedge) {
        // ---- Deletions: distinct ids from the pool -----------------------
        std::unordered_set<int> chosen;
        const int want = std::min<int>(nDel, static_cast<int>(pool.size()));
        while (static_cast<int>(chosen.size()) < want) {
            chosen.insert(pool[poolPick(rng)]);
        }
        batch.heDelete.assign(chosen.begin(), chosen.end());

        // ---- Insertions: pool-model hyperedges ---------------------------
        const int numPools =
            std::max(1, (gen.numVertices + gen.poolSize - 1) / gen.poolSize);
        std::uniform_int_distribution<int> poolDist(0, numPools - 1);
        std::uniform_int_distribution<int> cardDist(gen.cMin, gen.cMax);
        long long wLo = gen.wMin, wHi = gen.wMax;
        if (bp.placement == Placement::Targeted) {
            // Below-average weights raise the odds of improving the tree.
            wHi = std::max<long long>(wLo, (gen.wMin + gen.wMax) / 4);
        }
        std::uniform_int_distribution<long long> wDist(wLo, wHi);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::vector<int> verts;
        for (int i = 0; i < nIns; ++i) {
            HgBatch::HeIns ins;
            const int p = poolDist(rng);
            const int lo = p * gen.poolSize;
            const int hi =
                std::min(gen.numVertices - 1, lo + gen.poolSize - 1);
            int c = cardDist(rng);
            c = std::max(1, std::min(c, hi - lo + 1));
            sampleDistinct(rng, lo, hi, c, verts);
            if (numPools > 1 && unif(rng) < gen.bridgeFrac) {
                const int nPool = (p + 1) % numPools;
                const int nLo = nPool * gen.poolSize;
                const int nHi =
                    std::min(gen.numVertices - 1, nLo + gen.poolSize - 1);
                std::uniform_int_distribution<int> nv(nLo, nHi);
                verts[0] = nv(rng);
            }
            ins.vertices = verts;
            ins.weight = wDist(rng);
            batch.heInsert.push_back(std::move(ins));
        }
    } else {
        // ---- Incident-vertex batch --------------------------------------
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        const double delFrac = bp.delPct / 100.0;
        long long retries = 0;
        const long long maxRetries = 10LL * bp.size + 1000;
        for (int i = 0; i < bp.size; ++i) {
            const int id = pool[poolPick(rng)];
            const std::vector<int>& verts = hg.heVerts[id - 1];
            if (unif(rng) < delFrac) {
                if (verts.size() < 2) {
                    // Cardinality-1 hyperedges cannot lose a vertex; retry
                    // with a bound so degenerate hypergraphs terminate.
                    if (++retries > maxRetries) break;
                    --i;
                    continue;
                }
                std::uniform_int_distribution<std::size_t> vp(
                    0, verts.size() - 1);
                batch.vtxDelete.push_back({id, verts[vp(rng)]});
            } else {
                // Insert a vertex from the hyperedge's neighborhood pool.
                int anchor = verts.empty() ? 0 : verts[0];
                int p = anchor / gen.poolSize;
                int lo = p * gen.poolSize;
                int hi =
                    std::min(gen.numVertices - 1, lo + gen.poolSize - 1);
                std::uniform_int_distribution<int> vd(lo, hi);
                batch.vtxInsert.push_back({id, vd(rng)});
            }
        }
    }
    return batch;
}

} // namespace escher_mosp
