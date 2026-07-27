/**
 * @file HostHypergraph.cpp
 * @brief Host model + h2h delta rules for the dynamic hypergraph. Pure C++.
 *
 * The delta rules implement the maintenance discussed in the project
 * meeting: an h2h edge lives while the two hyperedges share at least one
 * vertex; a vertex removal deletes the edge only when the last common
 * vertex disappears; insertions are symmetric.
 */

#include "HostHypergraph.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace escher_mosp {

const long long HostHypergraph::INF =
    std::numeric_limits<long long>::max() / 4;

namespace {

inline std::uint64_t pairKey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint32_t>(b);
}

inline std::pair<int, int> orderedPair(int a, int b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

} // namespace

void HostHypergraph::removeValue_(std::vector<int>& v, int value) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] == value) {
            v[i] = v.back();
            v.pop_back();
            return;
        }
    }
}

bool HostHypergraph::overlaps_(int a, int b) const {
    const std::vector<int>& A = heVerts[a - 1];
    const std::vector<int>& B = heVerts[b - 1];
    std::size_t i = 0, j = 0;
    while (i < A.size() && j < B.size()) {
        if (A[i] == B[j]) return true;
        if (A[i] < B[j]) ++i; else ++j;
    }
    return false;
}

void HostHypergraph::buildFrom(int nVerts,
                               std::vector<std::vector<int>>&& rows,
                               std::vector<long long>&& weights) {
    assert(rows.size() == weights.size());
    numVertices = nVerts;
    heVerts = std::move(rows);
    heW = std::move(weights);
    const int m = static_cast<int>(heVerts.size());
    alive.assign(m, 1);
    freeIds.clear();
    aliveCount = m;

    for (auto& r : heVerts) {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
    }

    v2h.assign(numVertices, {});
    for (int id = 1; id <= m; ++id) {
        for (int v : heVerts[id - 1]) {
            assert(v >= 0 && v < numVertices);
            v2h[v].push_back(id);
        }
    }

    // h2h: union of v2h lists over each hyperedge's vertices.
    h2h.assign(m, {});
    h2hPairCount = 0;
    std::vector<int> scratch;
    for (int id = 1; id <= m; ++id) {
        scratch.clear();
        for (int v : heVerts[id - 1]) {
            for (int other : v2h[v]) {
                if (other != id) scratch.push_back(other);
            }
        }
        std::sort(scratch.begin(), scratch.end());
        scratch.erase(std::unique(scratch.begin(), scratch.end()),
                      scratch.end());
        h2h[id - 1] = scratch;
        h2hPairCount += static_cast<long long>(scratch.size());
    }
    h2hPairCount /= 2;
}

std::vector<int> HostHypergraph::reserveIds(int count) const {
    std::vector<int> ids;
    ids.reserve(count);
    int fromFree = std::min<int>(count, static_cast<int>(freeIds.size()));
    // LIFO: take from the back of the free list.
    for (int i = 0; i < fromFree; ++i) {
        ids.push_back(freeIds[freeIds.size() - 1 - i]);
    }
    int next = maxId() + 1;
    for (int i = fromFree; i < count; ++i) {
        ids.push_back(next++);
    }
    return ids;
}

void HostHypergraph::applyBatch(const HgBatch& b,
                                const std::vector<int>& finalIds,
                                H2HDelta& delta, EscherHorizOps& ops) {
    assert(finalIds.size() == b.heInsert.size());
    delta = H2HDelta{};
    ops = EscherHorizOps{};

    // Net edge effect: +1 insert, -1 delete; cancels to 0 for churn.
    std::unordered_map<std::uint64_t, int> edgeNet;
    edgeNet.reserve(b.totalOps() * 8 + 16);
    std::vector<int> seedPool;   // candidate seeds, filtered at the end
    std::unordered_set<int> recreated;

    auto touchEdge = [&](int a, int bId, int d) {
        edgeNet[pairKey(a, bId)] += d;
        seedPool.push_back(a);
        seedPool.push_back(bId);
    };

    // ---------------- Phase 1: hyperedge deletions -----------------------
    for (int id : b.heDelete) {
        if (id < 1 || id > maxId() || !alive[id - 1]) {
            ++delta.skippedOps;
            continue;
        }
        // Kill all incident h2h edges.
        for (int nb : h2h[id - 1]) {
            touchEdge(id, nb, -1);
            removeValue_(h2h[nb - 1], id);
            ops.h2hUnfill.emplace_back(nb, id);
        }
        h2hPairCount -= static_cast<long long>(h2h[id - 1].size());
        h2h[id - 1].clear();

        // Remove from v2h rows.
        for (int v : heVerts[id - 1]) {
            removeValue_(v2h[v], id);
            ops.v2hUnfill.emplace_back(v + 1, id);
        }
        heVerts[id - 1].clear();
        alive[id - 1] = 0;
        --aliveCount;
        freeIds.push_back(id);
        delta.deletedAtAnyPoint.push_back(id);
    }

    // ---------------- Phase 2: incident vertex deletions ------------------
    for (const auto& c : b.vtxDelete) {
        if (c.heId < 1 || c.heId > maxId() || !alive[c.heId - 1] ||
            c.vertex < 0 || c.vertex >= numVertices) {
            ++delta.skippedOps;
            continue;
        }
        std::vector<int>& verts = heVerts[c.heId - 1];
        auto it = std::lower_bound(verts.begin(), verts.end(), c.vertex);
        if (it == verts.end() || *it != c.vertex) {
            ++delta.skippedOps;
            continue;
        }
        if (verts.size() == 1) {
            // Never reduce a hyperedge below one vertex (would be an
            // implicit hyperedge deletion; callers use heDelete for that).
            ++delta.skippedOps;
            continue;
        }
        verts.erase(it);
        removeValue_(v2h[c.vertex], c.heId);
        ops.h2vUnfill.emplace_back(c.heId, c.vertex + 1);
        ops.v2hUnfill.emplace_back(c.vertex + 1, c.heId);

        // Only hyperedges that also contain the removed vertex can lose
        // their edge to c.heId (meeting rule: edge dies when the LAST
        // common vertex disappears).
        for (int other : v2h[c.vertex]) {
            if (other == c.heId) continue;
            // Adjacent right now?
            bool adjacent = false;
            for (int nb : h2h[c.heId - 1]) {
                if (nb == other) { adjacent = true; break; }
            }
            if (!adjacent) continue;
            if (!overlaps_(c.heId, other)) {
                touchEdge(c.heId, other, -1);
                removeValue_(h2h[c.heId - 1], other);
                removeValue_(h2h[other - 1], c.heId);
                ops.h2hUnfill.emplace_back(c.heId, other);
                ops.h2hUnfill.emplace_back(other, c.heId);
                --h2hPairCount;
            }
        }
    }

    // ---------------- Phase 3: incident vertex insertions -----------------
    for (const auto& c : b.vtxInsert) {
        if (c.heId < 1 || c.heId > maxId() || !alive[c.heId - 1] ||
            c.vertex < 0 || c.vertex >= numVertices) {
            ++delta.skippedOps;
            continue;
        }
        std::vector<int>& verts = heVerts[c.heId - 1];
        auto it = std::lower_bound(verts.begin(), verts.end(), c.vertex);
        if (it != verts.end() && *it == c.vertex) {
            ++delta.skippedOps;   // already a member
            continue;
        }
        // New h2h edges to hyperedges containing this vertex that were not
        // adjacent before.
        for (int other : v2h[c.vertex]) {
            if (other == c.heId) continue;
            bool adjacent = false;
            for (int nb : h2h[c.heId - 1]) {
                if (nb == other) { adjacent = true; break; }
            }
            if (!adjacent) {
                touchEdge(c.heId, other, +1);
                h2h[c.heId - 1].push_back(other);
                h2h[other - 1].push_back(c.heId);
                ops.h2hFill.emplace_back(c.heId, other);
                ops.h2hFill.emplace_back(other, c.heId);
                ++h2hPairCount;
            }
        }
        verts.insert(it, c.vertex);
        v2h[c.vertex].push_back(c.heId);
        ops.h2vFill.emplace_back(c.heId, c.vertex + 1);
        ops.v2hFill.emplace_back(c.vertex + 1, c.heId);
    }

    // ---------------- Phase 4: hyperedge insertions -----------------------
    std::unordered_set<int> newHeSet;
    for (std::size_t i = 0; i < b.heInsert.size(); ++i) {
        newHeSet.insert(finalIds[i]);
    }
    for (std::size_t i = 0; i < b.heInsert.size(); ++i) {
        const int id = finalIds[i];
        assert(id >= 1);
        if (id > maxId()) {
            heVerts.resize(id);
            heW.resize(id, 0);
            alive.resize(id, 0);
            h2h.resize(id);
        }
        assert(!alive[id - 1] && "hyperedge id collision on insert");
        // Recycled ids are removed from the free list.
        for (std::size_t f = 0; f < freeIds.size(); ++f) {
            if (freeIds[f] == id) {
                freeIds[f] = freeIds.back();
                freeIds.pop_back();
                recreated.insert(id);
                break;
            }
        }

        std::vector<int> verts = b.heInsert[i].vertices;
        std::sort(verts.begin(), verts.end());
        verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
        assert(!verts.empty());

        heVerts[id - 1] = verts;
        heW[id - 1] = b.heInsert[i].weight;
        alive[id - 1] = 1;
        ++aliveCount;
        delta.newHe.push_back(id);

        // Neighbors: every alive hyperedge sharing a vertex.
        std::vector<int> nbs;
        for (int v : verts) {
            for (int other : v2h[v]) {
                if (other != id) nbs.push_back(other);
            }
        }
        std::sort(nbs.begin(), nbs.end());
        nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
        for (int nb : nbs) {
            touchEdge(id, nb, +1);
            h2h[id - 1].push_back(nb);
            h2h[nb - 1].push_back(id);
            // The new row itself is inserted wholesale into the h2h CBST
            // (with its final neighbor list) by DynamicHypergraph; only
            // pre-existing rows need a fill. Edges between two new
            // hyperedges are covered by both new rows.
            if (newHeSet.find(nb) == newHeSet.end()) {
                ops.h2hFill.emplace_back(nb, id);
            }
            ++h2hPairCount;
        }

        for (int v : verts) {
            v2h[v].push_back(id);
            ops.v2hFill.emplace_back(v + 1, id);
        }
        seedPool.push_back(id);
    }

    // ---------------- Net delta + seeds -----------------------------------
    for (const auto& kv : edgeNet) {
        if (kv.second == 0) continue;
        int a = static_cast<int>(kv.first >> 32);
        int bId = static_cast<int>(kv.first & 0xffffffffu);
        if (kv.second > 0) {
            delta.insEdges.push_back(orderedPair(a, bId));
        } else {
            delta.delEdges.push_back(orderedPair(a, bId));
        }
    }
    for (int id : delta.deletedAtAnyPoint) {
        if (!alive[id - 1]) delta.deadHe.push_back(id);
    }
    // Recreated nodes may keep identical adjacency but a different weight;
    // they and their neighbors must be reseeded.
    for (int id : recreated) {
        seedPool.push_back(id);
        for (int nb : h2h[id - 1]) seedPool.push_back(nb);
    }

    std::sort(seedPool.begin(), seedPool.end());
    seedPool.erase(std::unique(seedPool.begin(), seedPool.end()),
                   seedPool.end());
    for (int id : seedPool) {
        if (id >= 1 && id <= maxId() && alive[id - 1] && id != sourceHe) {
            delta.seeds.push_back(id);
        }
    }
}

std::vector<long long> HostHypergraph::dijkstra(int sourceId) const {
    const int m = maxId();
    std::vector<long long> dist(m, INF);
    if (sourceId < 1 || sourceId > m || !alive[sourceId - 1]) return dist;

    using QE = std::pair<long long, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    dist[sourceId - 1] = 0;
    pq.push({0, sourceId});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u - 1]) continue;
        for (int nb : h2h[u - 1]) {
            long long nd = d + heW[nb - 1];
            if (nd < dist[nb - 1]) {
                dist[nb - 1] = nd;
                pq.push({nd, nb});
            }
        }
    }
    return dist;
}

std::vector<std::vector<int>> HostHypergraph::bruteForceH2H() const {
    const int m = maxId();
    std::vector<std::vector<int>> out(m);
    std::vector<std::vector<int>> byVertex(numVertices);
    for (int id = 1; id <= m; ++id) {
        if (!alive[id - 1]) continue;
        for (int v : heVerts[id - 1]) byVertex[v].push_back(id);
    }
    for (int v = 0; v < numVertices; ++v) {
        for (int a : byVertex[v]) {
            for (int c : byVertex[v]) {
                if (a != c) out[a - 1].push_back(c);
            }
        }
    }
    for (auto& r : out) {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sequential emulation of the device update (see hsosp/src/hsospDevice.cu).
// ---------------------------------------------------------------------------

namespace {

int emulateLoop(const HostHypergraph& hg, std::vector<long long>& dist,
                std::vector<int>& parent, std::vector<int> candidates,
                int maxIterations) {
    const long long INF = HostHypergraph::INF;
    const int source = hg.sourceHe;
    int iterations = 0;
    std::vector<int> affected;
    std::vector<std::uint8_t> inCand(hg.maxId() + 1, 0);

    while (!candidates.empty() && iterations < maxIterations) {
        ++iterations;
        affected.clear();
        for (int v : candidates) {
            inCand[v] = 0;
            if (v == source) continue;
            long long best = INF;
            int bestP = -1;
            for (int p : hg.h2h[v - 1]) {
                long long dp = dist[p - 1];
                if (dp >= INF / 2) continue;
                long long cd = dp + hg.heW[v - 1];
                if (cd < best) { best = cd; bestP = p; }
            }
            if (best != dist[v - 1]) {
                dist[v - 1] = best;
                parent[v - 1] = bestP;
                affected.push_back(v);
            } else {
                parent[v - 1] = bestP;
            }
        }
        candidates.clear();
        for (int u : affected) {
            for (int nb : hg.h2h[u - 1]) {
                if (nb == source) continue;
                if (!inCand[nb]) { inCand[nb] = 1; candidates.push_back(nb); }
            }
        }
    }
    return candidates.empty() ? iterations : -1;
}

} // namespace

int emulateSospRecompute(const HostHypergraph& hg,
                         std::vector<long long>& dist,
                         std::vector<int>& parent, int maxIterations) {
    const long long INF = HostHypergraph::INF;
    dist.assign(hg.maxId(), INF);
    parent.assign(hg.maxId(), -1);
    if (hg.sourceHe >= 1 && hg.sourceHe <= hg.maxId()) {
        dist[hg.sourceHe - 1] = 0;
    }
    std::vector<int> firstCands;
    for (int nb : hg.h2h[hg.sourceHe - 1]) firstCands.push_back(nb);
    return emulateLoop(hg, dist, parent, std::move(firstCands),
                       maxIterations);
}

int emulateSospUpdate(const HostHypergraph& hg, std::vector<long long>& dist,
                      std::vector<int>& parent, const std::vector<int>& seeds,
                      int maxIterations) {
    // Arrays may need to grow when the batch introduced fresh ids.
    if (static_cast<int>(dist.size()) < hg.maxId()) {
        dist.resize(hg.maxId(), HostHypergraph::INF);
        parent.resize(hg.maxId(), -1);
    }
    // Dead nodes drop out of every adjacency list; pin them at INF.
    for (int id = 1; id <= hg.maxId(); ++id) {
        if (!hg.alive[id - 1]) {
            dist[id - 1] = HostHypergraph::INF;
            parent[id - 1] = -1;
        }
    }
    int it = emulateLoop(hg, dist, parent, seeds, maxIterations);
    if (it < 0) {
        // Convergence cap hit (stale-loop in a disconnected component).
        // Fall back to a full recompute, exactly like the device path.
        emulateSospRecompute(hg, dist, parent,
                             std::max(maxIterations, hg.maxId() + 1));
        return -1;
    }
    return it;
}

} // namespace escher_mosp
