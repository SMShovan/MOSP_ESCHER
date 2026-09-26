/**
 * @file HypergraphOracle.cpp
 * @brief Independent line-graph rebuild + Dijkstra (see HypergraphOracle.hpp).
 */

#include "HypergraphOracle.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>

namespace escher_mosp {

bool LineGraphCSR::adjacent(int a, int b) const {
    if (a < 1 || a > numIds || b < 1 || b > numIds) return false;
    const int* r = row(a);
    return std::binary_search(r, r + degree(a), b);
}

LineGraphCSR rebuildLineGraph(const HostHypergraph& hg) {
    const int m = hg.maxId();
    const int n = hg.numVertices;

    // Vertex -> alive hyperedges, from heVerts only.
    std::vector<long long> vOff(static_cast<std::size_t>(n) + 1, 0);
    for (int id = 1; id <= m; ++id) {
        if (!hg.alive[id - 1]) continue;
        for (int v : hg.heVerts[id - 1]) ++vOff[v + 1];
    }
    for (int v = 0; v < n; ++v) vOff[v + 1] += vOff[v];
    std::vector<int> vHe(static_cast<std::size_t>(vOff[n]));
    {
        std::vector<long long> cur(vOff.begin(), vOff.end() - 1);
        for (int id = 1; id <= m; ++id) {
            if (!hg.alive[id - 1]) continue;
            for (int v : hg.heVerts[id - 1]) vHe[cur[v]++] = id;
        }
    }

    // Neighbour rows: union of the vertex lists, sorted and deduplicated.
    // Two passes (count, then write) keep the rows in one allocation.
    LineGraphCSR lg;
    lg.numIds = m;
    lg.offset.assign(static_cast<std::size_t>(m) + 1, 0);
    auto gather = [&](int id, std::vector<int>& out) {
        out.clear();
        if (!hg.alive[id - 1]) return;
        for (int v : hg.heVerts[id - 1])
            for (long long k = vOff[v]; k < vOff[v + 1]; ++k)
                if (vHe[k] != id) out.push_back(vHe[k]);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    };
#pragma omp parallel
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
#pragma omp parallel
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

std::vector<long long> referenceDistances(const HostHypergraph& hg,
                                          const LineGraphCSR& lg,
                                          int sourceId) {
    const int m = lg.numIds;
    std::vector<long long> dist(m, HostHypergraph::INF);
    if (sourceId < 1 || sourceId > m || !hg.alive[sourceId - 1]) return dist;
    using QE = std::pair<long long, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    dist[sourceId - 1] = 0;
    pq.push({0, sourceId});
    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u - 1]) continue;
        const int* r = lg.row(u);
        for (long long k = 0; k < lg.degree(u); ++k) {
            const int v = r[k];
            const long long nd = d + hg.heW[v - 1];
            if (nd < dist[v - 1]) {
                dist[v - 1] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

SospCheck checkSosp(const HostHypergraph& hg, const LineGraphCSR& lg,
                    const std::vector<long long>& reference,
                    const std::vector<long long>& dist,
                    const std::vector<int>& parent0) {
    SospCheck c;
    const int m = lg.numIds;
    const long long INF = HostHypergraph::INF;
    for (int id = 1; id <= m; ++id) {
        const long long got =
            id - 1 < static_cast<int>(dist.size()) ? dist[id - 1] : INF;
        if (got != reference[id - 1]) {
            if (c.distMismatches == 0) {
                c.firstBadId = id;
                c.firstGot = got;
                c.firstWant = reference[id - 1];
            }
            ++c.distMismatches;
        }
    }
    if (parent0.empty() || c.distMismatches != 0) return c;
    for (int id = 1; id <= m; ++id) {
        if (id == hg.sourceHe) continue;
        const int p = parent0[id - 1] + 1;   // 1-based id or 0 (none)
        if (!hg.alive[id - 1] || reference[id - 1] >= INF) {
            if (p != 0) ++c.parentErrors;
            continue;
        }
        // The canonical parent: the lowest-id neighbour on a shortest path
        // (rows are sorted, so the first tight neighbour).
        int canonical = 0;
        const int* r = lg.row(id);
        for (long long k = 0; k < lg.degree(id) && canonical == 0; ++k)
            if (reference[r[k] - 1] + hg.heW[id - 1] == reference[id - 1])
                canonical = r[k];
        if (p != canonical) ++c.parentErrors;
    }
    return c;
}

} // namespace escher_mosp
