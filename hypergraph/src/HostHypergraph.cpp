/**
 * @file HostHypergraph.cpp
 * @brief Host incidence model of the dynamic hypergraph. Pure C++.
 *
 * applyBatch validates the ops and turns them into incidence changes
 * (vertex, hyperedge, +/-1) plus the pre- and post-batch vertex lists of
 * the touched hyperedges; the GPU derives the net line-graph delta from
 * them (an h2h edge lives while the two hyperedges share at least one
 * vertex).
 */

#include "HostHypergraph.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace escher_mosp {

const long long HostHypergraph::INF =
    std::numeric_limits<long long>::max() / 4;

namespace {

// Below this many rows the OpenMP team costs more than it saves.
constexpr int kParallelMin = 1 << 14;

void removeValue(std::vector<int>& v, int value) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] == value) {
            v[i] = v.back();
            v.pop_back();
            return;
        }
    }
}

} // namespace

bool LineGraphCSR::adjacent(int a, int b) const {
    if (a < 1 || a > numIds || b < 1 || b > numIds) return false;
    const int* r = row(a);
    return std::binary_search(r, r + degree(a), b);
}

void HostHypergraph::freeListPush_(int id) {
    if (static_cast<int>(freePos.size()) <= id) freePos.resize(id + 1, -1);
    freePos[id] = static_cast<int>(freeIds.size());
    freeIds.push_back(id);
}

// O(1) removal (the original scanned the whole free list for every
// inserted hyperedge: 16 s per 200K batch on DBLP).
void HostHypergraph::freeListRemove_(int id) {
    if (id >= static_cast<int>(freePos.size()) || freePos[id] < 0) return;
    const int pos = freePos[id];
    const int last = freeIds.back();
    freeIds[pos] = last;
    freePos[last] = pos;
    freeIds.pop_back();
    freePos[id] = -1;
}

void HostHypergraph::buildFrom(int nVerts,
                               std::vector<std::vector<int>>&& rows,
                               std::vector<long long>&& weights) {
    if (rows.size() != weights.size())
        throw std::invalid_argument("HostHypergraph::buildFrom: one weight "
                                    "per hyperedge required");
    numVertices = nVerts;
    heVerts = std::move(rows);
    heW = std::move(weights);
    const int m = static_cast<int>(heVerts.size());
    alive.assign(m, 1);
    freeIds.clear();
    freePos.clear();
    aliveCount = m;

#pragma omp parallel for schedule(dynamic, 4096) if (m > kParallelMin)
    for (int i = 0; i < m; ++i) {
        std::vector<int>& r = heVerts[i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
    }
    for (int id = 1; id <= m; ++id)
        for (int v : heVerts[id - 1])
            if (v < 0 || v >= numVertices)
                throw std::invalid_argument(
                    "HostHypergraph::buildFrom: vertex id out of range");

    // v2h: counting pass, then fill in id order.
    std::vector<int> count(numVertices, 0);
    for (int id = 1; id <= m; ++id)
        for (int v : heVerts[id - 1]) ++count[v];
    v2h.assign(numVertices, {});
    for (int v = 0; v < numVertices; ++v) v2h[v].reserve(count[v]);
    for (int id = 1; id <= m; ++id)
        for (int v : heVerts[id - 1]) v2h[v].push_back(id);
    h2hPairCount = 0;   // set by the owner once the line graph is built
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
                                IncidenceBatch& inc, EscherHorizOps& ops) {
    if (finalIds.size() != b.heInsert.size())
        throw std::invalid_argument(
            "HostHypergraph::applyBatch: one id per inserted hyperedge");
    inc = IncidenceBatch{};
    ops = EscherHorizOps{};

    // Touched hyperedges (pre-batch rows snapshotted on first touch) and
    // touched vertices.
    std::unordered_map<int, int> touchedSlot;
    touchedSlot.reserve(b.totalOps() * 2 + 16);
    std::vector<std::vector<int>> preRows;
    auto touchHe = [&](int id) {
        if (touchedSlot.emplace(id, static_cast<int>(inc.touched.size()))
                .second) {
            inc.touched.push_back(id);
            preRows.push_back(id <= maxId() ? heVerts[id - 1]
                                            : std::vector<int>{});
        }
    };
    std::unordered_set<int> touchedVertexSet;
    auto incidence = [&](int v, int h, int sign) {
        inc.incKey.push_back((static_cast<std::uint64_t>(
                                  static_cast<std::uint32_t>(v))
                              << 32) |
                             static_cast<std::uint32_t>(h));
        inc.incSign.push_back(sign);
        if (touchedVertexSet.insert(v).second)
            inc.touchedVertices.push_back(v);
    };
    std::vector<int> deletedAtAnyPoint;

    // ---------------- Phase 1: hyperedge deletions -----------------------
    for (int id : b.heDelete) {
        if (id < 1 || id > maxId() || !alive[id - 1]) {
            ++inc.skippedOps;
            continue;
        }
        touchHe(id);
        for (int v : heVerts[id - 1]) {
            removeValue(v2h[v], id);
            ops.v2hUnfill.emplace_back(v + 1, id);
            incidence(v, id, -1);
        }
        heVerts[id - 1].clear();
        alive[id - 1] = 0;
        --aliveCount;
        freeListPush_(id);
        deletedAtAnyPoint.push_back(id);
    }

    // ---------------- Phase 2: incident vertex deletions ------------------
    for (const auto& c : b.vtxDelete) {
        if (c.heId < 1 || c.heId > maxId() || !alive[c.heId - 1] ||
            c.vertex < 0 || c.vertex >= numVertices) {
            ++inc.skippedOps;
            continue;
        }
        std::vector<int>& verts = heVerts[c.heId - 1];
        auto it = std::lower_bound(verts.begin(), verts.end(), c.vertex);
        if (it == verts.end() || *it != c.vertex) {
            ++inc.skippedOps;
            continue;
        }
        if (verts.size() == 1) {
            // Never reduce a hyperedge below one vertex (would be an
            // implicit hyperedge deletion; callers use heDelete for that).
            ++inc.skippedOps;
            continue;
        }
        touchHe(c.heId);
        verts.erase(it);
        removeValue(v2h[c.vertex], c.heId);
        ops.h2vUnfill.emplace_back(c.heId, c.vertex + 1);
        ops.v2hUnfill.emplace_back(c.vertex + 1, c.heId);
        incidence(c.vertex, c.heId, -1);
    }

    // ---------------- Phase 3: incident vertex insertions -----------------
    for (const auto& c : b.vtxInsert) {
        if (c.heId < 1 || c.heId > maxId() || !alive[c.heId - 1] ||
            c.vertex < 0 || c.vertex >= numVertices) {
            ++inc.skippedOps;
            continue;
        }
        std::vector<int>& verts = heVerts[c.heId - 1];
        auto it = std::lower_bound(verts.begin(), verts.end(), c.vertex);
        if (it != verts.end() && *it == c.vertex) {
            ++inc.skippedOps;   // already a member
            continue;
        }
        touchHe(c.heId);
        verts.insert(it, c.vertex);
        v2h[c.vertex].push_back(c.heId);
        ops.h2vFill.emplace_back(c.heId, c.vertex + 1);
        ops.v2hFill.emplace_back(c.vertex + 1, c.heId);
        incidence(c.vertex, c.heId, +1);
    }

    // ---------------- Phase 4: hyperedge insertions -----------------------
    for (std::size_t i = 0; i < b.heInsert.size(); ++i) {
        const int id = finalIds[i];
        if (id < 1)
            throw std::invalid_argument(
                "HostHypergraph::applyBatch: invalid hyperedge id");
        if (id > maxId()) {
            heVerts.resize(id);
            heW.resize(id, 0);
            alive.resize(id, 0);
        }
        if (alive[id - 1])
            throw std::logic_error(
                "HostHypergraph::applyBatch: hyperedge id collision on insert");
        std::vector<int> verts = b.heInsert[i].vertices;
        std::sort(verts.begin(), verts.end());
        verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
        if (verts.empty() || verts.front() < 0 || verts.back() >= numVertices)
            throw std::invalid_argument(
                "HostHypergraph::applyBatch: inserted hyperedge without "
                "vertices or with a vertex out of range");
        touchHe(id);
        freeListRemove_(id);   // recycled ids leave the free list
        heVerts[id - 1] = verts;
        heW[id - 1] = b.heInsert[i].weight;
        alive[id - 1] = 1;
        ++aliveCount;
        inc.newHe.push_back(id);
        inc.newW.push_back(b.heInsert[i].weight);
        for (int v : verts) {
            v2h[v].push_back(id);
            ops.v2hFill.emplace_back(v + 1, id);
            incidence(v, id, +1);
        }
    }

    for (int id : deletedAtAnyPoint)
        if (!alive[id - 1]) inc.deadHe.push_back(id);

    // Pre / post rows of the touched hyperedges, post lengths of the
    // touched vertices.
    inc.preOff.assign(1, 0);
    inc.postOff.assign(1, 0);
    for (std::size_t t = 0; t < inc.touched.size(); ++t) {
        const int id = inc.touched[t];
        inc.preVals.insert(inc.preVals.end(), preRows[t].begin(),
                           preRows[t].end());
        inc.preOff.push_back(static_cast<int>(inc.preVals.size()));
        const std::vector<int>& post = heVerts[id - 1];
        inc.postVals.insert(inc.postVals.end(), post.begin(), post.end());
        inc.postOff.push_back(static_cast<int>(inc.postVals.size()));
    }
    inc.touchedVertexLen.reserve(inc.touchedVertices.size());
    for (int v : inc.touchedVertices)
        inc.touchedVertexLen.push_back(static_cast<int>(v2h[v].size()));
}

LineGraphCSR HostHypergraph::lineGraph() const {
    const int m = maxId();
    LineGraphCSR lg;
    lg.numIds = m;
    lg.offset.assign(static_cast<std::size_t>(m) + 1, 0);
    // Row of id: union of the incidence lists of its vertices. Two passes
    // (count, then write) keep the rows in one allocation.
    auto gather = [&](int id, std::vector<int>& out) {
        out.clear();
        if (!alive[id - 1]) return;
        for (int v : heVerts[id - 1])
            for (int other : v2h[v])
                if (other != id) out.push_back(other);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    };
#pragma omp parallel if (m > kParallelMin)
    {
        std::vector<int> scratch;
#pragma omp for schedule(dynamic, 1024)
        for (int id = 1; id <= m; ++id) {
            gather(id, scratch);
            lg.offset[id] = static_cast<long long>(scratch.size());
        }
    }
    for (int id = 1; id <= m; ++id) lg.offset[id] += lg.offset[id - 1];
    lg.nbr.resize(static_cast<std::size_t>(lg.offset[m]));
#pragma omp parallel if (m > kParallelMin)
    {
        std::vector<int> scratch;
#pragma omp for schedule(dynamic, 1024)
        for (int id = 1; id <= m; ++id) {
            gather(id, scratch);
            std::copy(scratch.begin(), scratch.end(),
                      lg.nbr.begin() + lg.offset[id - 1]);
        }
    }
    return lg;
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

int emulateLoop(const HostHypergraph& hg, const LineGraphCSR& lg,
                std::vector<long long>& dist,
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
            for (long long k = 0; k < lg.degree(v); ++k) {
                const int p = lg.row(v)[k];
                long long dp = dist[p - 1];
                if (dp >= INF / 2) continue;
                long long cd = dp + hg.heW[v - 1];
                if (cd < best || (cd == best && p < bestP)) {
                    best = cd;
                    bestP = p;
                }
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
            for (long long k = 0; k < lg.degree(u); ++k) {
                const int nb = lg.row(u)[k];
                if (nb == source) continue;
                if (!inCand[nb]) { inCand[nb] = 1; candidates.push_back(nb); }
            }
        }
    }
    return candidates.empty() ? iterations : -1;
}

} // namespace

int emulateSospRecompute(const HostHypergraph& hg, const LineGraphCSR& lg,
                         std::vector<long long>& dist,
                         std::vector<int>& parent, int maxIterations) {
    const long long INF = HostHypergraph::INF;
    dist.assign(hg.maxId(), INF);
    parent.assign(hg.maxId(), -1);
    if (hg.sourceHe >= 1 && hg.sourceHe <= hg.maxId()) {
        dist[hg.sourceHe - 1] = 0;
    }
    std::vector<int> firstCands;
    for (long long k = 0; k < lg.degree(hg.sourceHe); ++k)
        firstCands.push_back(lg.row(hg.sourceHe)[k]);
    return emulateLoop(hg, lg, dist, parent, std::move(firstCands),
                       maxIterations);
}

int emulateSospUpdate(const HostHypergraph& hg, const LineGraphCSR& lg,
                      std::vector<long long>& dist, std::vector<int>& parent,
                      const H2HDelta& delta) {
    const long long INF = HostHypergraph::INF;
    const int m = hg.maxId();
    if (static_cast<int>(dist.size()) < m) {
        dist.resize(m, INF);
        parent.resize(m, -1);
    }
    // Roots (pre-batch tree): deleted tree edges, new / recreated / dead.
    std::vector<std::uint8_t> inv(m + 1, 0);
    for (auto [a, b] : delta.delEdges) {
        if (parent[b - 1] == a) inv[b] = 1;
        if (parent[a - 1] == b) inv[a] = 1;
    }
    for (int id : delta.newHe) inv[id] = 1;
    for (int id : delta.deadHe) inv[id] = 1;
    // Subtrees of the roots (children lists of the pre-batch tree).
    std::vector<std::vector<int>> children(m + 1);
    for (int id = 1; id <= m; ++id)
        if (parent[id - 1] >= 1) children[parent[id - 1]].push_back(id);
    std::vector<int> stack;
    for (int id = 1; id <= m; ++id)
        if (inv[id]) stack.push_back(id);
    while (!stack.empty()) {
        const int u = stack.back();
        stack.pop_back();
        for (int c : children[u])
            if (!inv[c]) {
                inv[c] = 1;
                stack.push_back(c);
            }
    }
    int invalidated = 0;
    for (int id = 1; id <= m; ++id) {
        if (inv[id] && id != hg.sourceHe) {
            dist[id - 1] = INF;
            parent[id - 1] = -1;
            ++invalidated;
        }
    }
    // (distance, parent id) comparisons: ties go to the lower id.
    using Key = std::pair<long long, int>;
    std::priority_queue<std::pair<Key, int>, std::vector<std::pair<Key, int>>,
                        std::greater<std::pair<Key, int>>>
        pq;
    auto offer = [&](int v, long long d, int p) {
        if (v == hg.sourceHe || !hg.alive[v - 1]) return;
        if (Key(d, p) < Key(dist[v - 1], parent[v - 1] < 0 ? INT32_MAX
                                                          : parent[v - 1])) {
            dist[v - 1] = d;
            parent[v - 1] = p;
            pq.push({Key(d, p), v});
        }
    };
    for (int id = 1; id <= m; ++id) {
        if (!inv[id] || !hg.alive[id - 1]) continue;
        for (long long k = 0; k < lg.degree(id); ++k) {
            const int p = lg.row(id)[k];
            if (dist[p - 1] < INF) offer(id, dist[p - 1] + hg.heW[id - 1], p);
        }
    }
    for (auto [a, b] : delta.insEdges) {
        if (dist[a - 1] < INF) offer(b, dist[a - 1] + hg.heW[b - 1], a);
        if (dist[b - 1] < INF) offer(a, dist[b - 1] + hg.heW[a - 1], b);
    }
    while (!pq.empty()) {
        const auto [key, u] = pq.top();
        pq.pop();
        if (key != Key(dist[u - 1], parent[u - 1])) continue;
        for (long long k = 0; k < lg.degree(u); ++k) {
            const int v = lg.row(u)[k];
            offer(v, dist[u - 1] + hg.heW[v - 1], u);
        }
    }
    return invalidated;
}

} // namespace escher_mosp
