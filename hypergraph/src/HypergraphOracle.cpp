/**
 * @file HypergraphOracle.cpp
 * @brief Independent line-graph rebuild + Dijkstra (see HypergraphOracle.hpp).
 */

#include "HypergraphOracle.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>

#ifdef _OPENMP
#include <parallel/algorithm>
#endif

namespace escher_mosp {

namespace {
// Below this many rows the OpenMP team costs more than it saves.
constexpr int kParallelMin = 1 << 14;
} // namespace

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

    // Every ordered pair (a, b), a != b, of hyperedges sharing a vertex,
    // emitted per vertex, then sorted and deduplicated.
    std::vector<long long> pairOff(static_cast<std::size_t>(n) + 1, 0);
    for (int v = 0; v < n; ++v) {
        const long long d = vOff[v + 1] - vOff[v];
        pairOff[v + 1] = pairOff[v] + d * (d - 1);
    }
    std::vector<std::uint64_t> pairs(static_cast<std::size_t>(pairOff[n]));
#pragma omp parallel for schedule(dynamic, 1024) if (n > kParallelMin)
    for (int v = 0; v < n; ++v) {
        long long k = pairOff[v];
        for (long long i = vOff[v]; i < vOff[v + 1]; ++i)
            for (long long j = vOff[v]; j < vOff[v + 1]; ++j)
                if (i != j)
                    pairs[k++] = (static_cast<std::uint64_t>(vHe[i]) << 32) |
                                 static_cast<std::uint32_t>(vHe[j]);
    }
#ifdef _OPENMP
    if (pairs.size() > (1u << 20))
        __gnu_parallel::sort(pairs.begin(), pairs.end());
    else
#endif
        std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    LineGraphCSR lg;
    lg.numIds = m;
    lg.offset.assign(static_cast<std::size_t>(m) + 1, 0);
    lg.nbr.resize(pairs.size());
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        ++lg.offset[pairs[k] >> 32];
        lg.nbr[k] = static_cast<int>(pairs[k] & 0xffffffffu);
    }
    for (int id = 1; id <= m; ++id) lg.offset[id] += lg.offset[id - 1];
    return lg;
}

H2HDelta lineGraphDelta(const LineGraphCSR& pre, const LineGraphCSR& post) {
    H2HDelta d;
    const int m = std::max(pre.numIds, post.numIds);
    for (int a = 1; a <= m; ++a) {
        const int* r0 = a <= pre.numIds ? pre.row(a) : nullptr;
        const int* e0 = r0 ? r0 + pre.degree(a) : nullptr;
        const int* r1 = a <= post.numIds ? post.row(a) : nullptr;
        const int* e1 = r1 ? r1 + post.degree(a) : nullptr;
        // Merge of the two sorted rows; keep b > a only.
        while (r0 != e0 || r1 != e1) {
            if (r1 == e1 || (r0 != e0 && *r0 < *r1)) {
                if (*r0 > a) d.delEdges.emplace_back(a, *r0);
                ++r0;
            } else if (r0 == e0 || *r1 < *r0) {
                if (*r1 > a) d.insEdges.emplace_back(a, *r1);
                ++r1;
            } else {
                ++r0;
                ++r1;
            }
        }
    }
    return d;
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
