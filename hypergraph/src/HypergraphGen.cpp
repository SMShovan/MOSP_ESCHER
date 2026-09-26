/**
 * @file HypergraphGen.cpp
 * @brief Seeded pool-model hypergraph generator + change-batch generator.
 */

#include "HypergraphGen.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
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
    // With no real hyperedge left the pool is empty: a hyperedge batch
    // still gets its insertions, a vertex batch has nothing to change.
    if (pool.empty() && bp.kind == BatchKind::Vertex) return batch;
    std::uniform_int_distribution<std::size_t> poolPick(
        0, pool.empty() ? 0 : pool.size() - 1);

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

namespace {

enum class IdParse { Ok, NotANumber, OutOfRange };

/** Decimal integer at @p p ([+-]digits, the prefix std::strtoll would
 *  accept after the separators), advancing @p p past it. Faster than
 *  strtoll with errno; values outside the signed 64-bit range are
 *  reported, not saturated. */
IdParse parseVertexId(const char*& p, long long& out) {
    const char* q = p;
    const bool neg = (*q == '-');
    if (*q == '-' || *q == '+') ++q;
    if (*q < '0' || *q > '9') return IdParse::NotANumber;
    // Accumulate the magnitude; the negative range has one more value.
    const unsigned long long limit =
        static_cast<unsigned long long>(
            std::numeric_limits<long long>::max()) + (neg ? 1u : 0u);
    unsigned long long mag = 0;
    bool overflow = false;
    for (; *q >= '0' && *q <= '9'; ++q) {
        const unsigned d = static_cast<unsigned>(*q - '0');
        if (mag > (limit - d) / 10) overflow = true;
        else mag = mag * 10 + d;
    }
    p = q;
    if (overflow) return IdParse::OutOfRange;
    out = neg ? static_cast<long long>(0 - mag) : static_cast<long long>(mag);
    return IdParse::Ok;
}

} // namespace

GeneratedHypergraph loadHypergraphFile(const std::string& path,
                                       int maxCardinality,
                                       std::uint64_t seed) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open hypergraph file " + path);
    // Raw ids are read as 64-bit values (some collections use ids above
    // 2^31; truncating them to int merged distinct vertices) and renumbered
    // 0..n-1 in order of first appearance as each kept row is read.
    std::unordered_map<long long, int> remap;
    {
        // Pre-size the map from the file size (as the two-pass version
        // did from the row count) so that it does not rehash as it grows.
        in.seekg(0, std::ios::end);
        const std::streamoff bytes = in.tellg();
        in.seekg(0, std::ios::beg);
        if (bytes > 0) remap.reserve(static_cast<std::size_t>(bytes / 16));
    }
    std::vector<std::vector<int>> rows;
    std::string line;
    std::vector<long long> r;
    while (std::getline(in, line)) {
        r.clear();
        const char* p = line.c_str();
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\r') ++p;
            if (!*p) break;
            long long v = 0;
            switch (parseVertexId(p, v)) {
            case IdParse::NotANumber:
                throw std::runtime_error("non-numeric token in " + path +
                                         ": " + line);
            case IdParse::OutOfRange:
                throw std::runtime_error("vertex id out of range in " + path +
                                         ": " + line);
            case IdParse::Ok:
                break;
            }
            r.push_back(v);
        }
        if (r.empty()) continue;
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        if (static_cast<long long>(r.size()) > maxCardinality) continue;
        std::vector<int> row;
        row.reserve(r.size());
        for (long long v : r) {
            auto it = remap.find(v);
            if (it == remap.end()) {
                if (remap.size() >= static_cast<std::size_t>(
                                        std::numeric_limits<int>::max()))
                    throw std::runtime_error(
                        "more than 2^31 - 1 distinct vertices in " + path);
                it = remap.emplace(v, static_cast<int>(remap.size())).first;
            }
            row.push_back(it->second);
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw std::runtime_error("no hyperedge of at most " +
                                 std::to_string(maxCardinality) +
                                 " vertices in " + path);
    }
    GeneratedHypergraph g;
    g.numVertices = static_cast<int>(remap.size());
    std::vector<int> deg(g.numVertices, 0);
    for (const auto& row : rows)
        for (int v : row) ++deg[v];
    std::mt19937_64 rng(seed);
    g.sourceVertex = static_cast<int>(
        std::max_element(deg.begin(), deg.end()) - deg.begin());
    g.targetVertex =
        std::uniform_int_distribution<int>(0, g.numVertices - 1)(rng);
    std::uniform_int_distribution<long long> wDist(1, 100);
    g.rows.reserve(rows.size() + 2);
    g.weights.reserve(rows.size() + 2);
    g.rows.push_back({g.sourceVertex});
    g.weights.push_back(0);
    for (auto& row : rows) {
        g.rows.push_back(std::move(row));
        g.weights.push_back(wDist(rng));
    }
    g.rows.push_back({g.targetVertex});
    g.weights.push_back(0);
    g.sourceHe = 1;
    g.targetHe = static_cast<int>(g.rows.size());
    return g;
}

HgBatch generatePaperBatch(const HostHypergraph& hg, BatchKind kind,
                           int size, double delPct, double replaceFrac,
                           std::mt19937_64& rng) {
    HgBatch b;
    std::vector<int> pool;
    pool.reserve(hg.aliveCount);
    for (int id = 1; id <= hg.maxId(); ++id)
        if (hg.alive[id - 1] && id != hg.sourceHe && id != hg.targetHe)
            pool.push_back(id);
    if (pool.empty() || size <= 0) return b;
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    std::uniform_int_distribution<long long> wDist(1, 100);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    auto randomOf = [&](const std::vector<int>& v) {
        return v[std::uniform_int_distribution<std::size_t>(0, v.size() - 1)(
            rng)];
    };
    // A hyperedge sharing a vertex with id (through a random vertex of id).
    auto randomNeighbor = [&](int id) -> int {
        const std::vector<int>& verts = hg.heVerts[id - 1];
        for (int tries = 0; tries < 8 && !verts.empty(); ++tries) {
            const std::vector<int>& inc = hg.v2h[randomOf(verts)];
            if (inc.size() < 2) continue;
            const int x = randomOf(inc);
            if (x != id && x != hg.sourceHe && x != hg.targetHe) return x;
        }
        return 0;
    };
    const int nDel = static_cast<int>(size * delPct / 100.0 + 0.5);
    const int nIns = size - nDel;
    if (kind == BatchKind::Hyperedge) {
        std::unordered_set<int> chosen;
        const int want = std::min<int>(nDel, static_cast<int>(pool.size()));
        while (static_cast<int>(chosen.size()) < want)
            chosen.insert(pool[pick(rng)]);
        b.heDelete.assign(chosen.begin(), chosen.end());
        std::sort(b.heDelete.begin(), b.heDelete.end());
        for (int i = 0; i < nIns; ++i) {
            const int src = pool[pick(rng)];
            std::vector<int> verts = hg.heVerts[src - 1];
            const int nb = randomNeighbor(src);
            if (nb > 0 && verts.size() >= 2) {
                const std::vector<int>& nv = hg.heVerts[nb - 1];
                const int r = std::max(
                    1, static_cast<int>(std::lround(replaceFrac * verts.size())));
                std::shuffle(verts.begin(), verts.end(), rng);
                for (int k = 0; k < r && k < static_cast<int>(verts.size()); ++k)
                    verts[k] = randomOf(nv);
            }
            HgBatch::HeIns ins;
            ins.vertices = std::move(verts);
            ins.weight = wDist(rng);
            b.heInsert.push_back(std::move(ins));
        }
    } else {
        long long retries = 0;
        const long long maxRetries = 10LL * size + 1000;
        for (int i = 0; i < size; ++i) {
            const int id = pool[pick(rng)];
            const std::vector<int>& verts = hg.heVerts[id - 1];
            if (unif(rng) < delPct / 100.0) {
                if (verts.size() < 2) {
                    if (++retries > maxRetries) break;
                    --i;
                    continue;
                }
                b.vtxDelete.push_back({id, randomOf(verts)});
            } else {
                const int nb = randomNeighbor(id);
                if (nb <= 0) {
                    if (++retries > maxRetries) break;
                    --i;
                    continue;
                }
                b.vtxInsert.push_back({id, randomOf(hg.heVerts[nb - 1])});
            }
        }
    }
    return b;
}

} // namespace escher_mosp
