/**
 * @file DynamicHypergraph.cpp
 * @brief ESCHER routing for the dynamic hypergraph (see header for design).
 */

#include "DynamicHypergraph.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

#include "escher_errors.hpp"
#include "flatten.hpp"
#include "structure.hpp"

namespace escher_mosp {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}

/** Group raw (rowKey, value) pairs and issue one batched fill / unfill. */
void flushGrouped(const std::vector<std::pair<int, int>>& rawOps,
                  CBSTOperations& cbst, bool isFill) {
    if (rawOps.empty()) return;
    // std::map keeps keys sorted, which keeps the CBST call deterministic.
    std::map<int, std::vector<int>> byRow;
    for (const auto& op : rawOps) byRow[op.first].push_back(op.second);

    std::vector<int> keys;
    std::vector<int> payload;
    std::vector<int> prefix;
    keys.reserve(byRow.size());
    prefix.reserve(byRow.size());
    int run = 0;
    for (auto& kv : byRow) {
        keys.push_back(kv.first);
        for (int v : kv.second) payload.push_back(v);
        run += static_cast<int>(kv.second.size());
        prefix.push_back(run);
    }
    if (isFill) {
        cbst.fill(keys, payload, prefix);
    } else {
        unfillCBST(keys, payload, prefix,
                   const_cast<CBSTContext&>(cbst.context()));
    }
}

/** Build flatten inputs + occupancy counts and construct a CBST sized to
 *  flatSize * headroom + extra (capped at INT_MAX ints ~ 8 GiB). */
std::unique_ptr<CBSTOperations> constructFromRows(
    const char* name, const std::vector<std::vector<int>>& rows,
    double headroom, long long extra) {
    auto [flat, offsets] = flatten2DVector(rows);
    long long capacity = static_cast<long long>(
                             static_cast<double>(flat.size()) * headroom) +
                         extra;
    if (capacity > 2000000000LL) capacity = 2000000000LL;   // int-indexed
    if (capacity < static_cast<long long>(flat.size())) {
        throw escher::EscherError(std::string("DynamicHypergraph: ") + name +
                                  " initial payload exceeds the 2^31 int "
                                  "payload limit of the CBST core");
    }
    auto cbst = std::make_unique<CBSTOperations>(
        name, static_cast<int>(capacity), 4);
    std::vector<int> keys(rows.size());
    std::vector<int> counts(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        keys[i] = static_cast<int>(i) + 1;
        counts[i] = static_cast<int>(rows[i].size());
    }
    cbst->construct(keys.data(), offsets.data(),
                    static_cast<int>(rows.size()), flat.data(),
                    static_cast<int>(flat.size()), counts.data());
    return cbst;
}

} // namespace

struct DynamicHypergraph::Impl {
    int numVertices = 0;
    Caps caps;
    HostHypergraph host;

    std::unique_ptr<CBSTOperations> h2v;
    std::unique_ptr<CBSTOperations> v2h;
    std::unique_ptr<CBSTOperations> h2h;

    /// heId -> key of its row in the h2h CBST (0 = none).
    std::vector<int> h2hKeyOfHe;
};

DynamicHypergraph::DynamicHypergraph(int numVertices, const Caps& caps)
    : pImpl(std::make_unique<Impl>()) {
    if (numVertices <= 0 || caps.maxHyperedges <= 0) {
        throw escher::EscherError(
            "DynamicHypergraph: numVertices and maxHyperedges must be positive");
    }
    pImpl->numVertices = numVertices;
    pImpl->caps = caps;
}

DynamicHypergraph::~DynamicHypergraph() = default;

HostHypergraph& DynamicHypergraph::host() { return pImpl->host; }
const HostHypergraph& DynamicHypergraph::host() const { return pImpl->host; }

long long DynamicHypergraph::escherDeviceBytes() const {
    // Payload buffers dominate; add the per-record node/key arrays.
    auto cbstBytes = [](const CBSTOperations* c) -> long long {
        if (!c) return 0;
        const CBSTContext& ctx = c->context();
        return static_cast<long long>(ctx.fixedSize) * sizeof(int) +
               static_cast<long long>(ctx.numRecords) *
                   (sizeof(CBSTNode) + 4 * sizeof(int));
    };
    return cbstBytes(pImpl->h2v.get()) + cbstBytes(pImpl->v2h.get()) +
           cbstBytes(pImpl->h2h.get());
}

void DynamicHypergraph::bulkLoad(std::vector<std::vector<int>>&& rows,
                                 std::vector<long long>&& weights,
                                 int sourceHe, int targetHe) {
    Impl& im = *pImpl;
    im.host.buildFrom(im.numVertices, std::move(rows), std::move(weights));
    im.host.sourceHe = sourceHe;
    im.host.targetHe = targetHe;

    const int m = im.host.maxId();
    if (m > im.caps.maxHyperedges) {
        throw escher::EscherError(
            "DynamicHypergraph::bulkLoad: more hyperedges than maxHyperedges");
    }

    const double hf = im.caps.headroomFactor;
    const long long extra = im.caps.extraPayloadInts;

    // ---- h2v: incident vertex list per hyperedge (vertices stored +1) ----
    {
        std::vector<std::vector<int>> h2vRows(m);
        for (int id = 1; id <= m; ++id) {
            h2vRows[id - 1].reserve(im.host.heVerts[id - 1].size());
            for (int v : im.host.heVerts[id - 1])
                h2vRows[id - 1].push_back(v + 1);
        }
        im.h2v = constructFromRows("h2v", h2vRows, hf, extra);
    }

    // ---- v2h: incident hyperedge list per vertex -------------------------
    {
        std::vector<std::vector<int>> v2hRows(im.numVertices);
        for (int v = 0; v < im.numVertices; ++v) v2hRows[v] = im.host.v2h[v];
        im.v2h = constructFromRows("v2h", v2hRows, hf, extra);
    }

    // ---- h2h: neighboring hyperedge list per hyperedge -------------------
    {
        std::vector<std::vector<int>> h2hRows(m);
        for (int id = 1; id <= m; ++id) h2hRows[id - 1] = im.host.h2h[id - 1];
        im.h2h = constructFromRows("h2h", h2hRows, hf, extra);
        im.h2hKeyOfHe.assign(im.caps.maxHyperedges + 1, 0);
        for (int id = 1; id <= m; ++id) im.h2hKeyOfHe[id] = id;
    }
}

DynamicHypergraph::BatchResult DynamicHypergraph::applyBatch(
    const HgBatch& batch) {
    Impl& im = *pImpl;
    BatchResult res;

    // ---------------------------------------------------------------
    // 1. Vertical h2v insert FIRST: the returned mapping decides the
    //    final ids of the inserted hyperedges (ESCHER id reassignment).
    // ---------------------------------------------------------------
    std::vector<int> finalIds;
    {
        auto t0 = Clock::now();
        const int K = static_cast<int>(batch.heInsert.size());
        if (K > 0) {
            std::vector<int> tentative = im.host.reserveIds(K);
            std::vector<int> payload;
            std::vector<int> prefix;
            prefix.reserve(K);
            int run = 0;
            for (int i = 0; i < K; ++i) {
                std::vector<int> verts = batch.heInsert[i].vertices;
                std::sort(verts.begin(), verts.end());
                verts.erase(std::unique(verts.begin(), verts.end()),
                            verts.end());
                for (int v : verts) payload.push_back(v + 1);
                run += static_cast<int>(verts.size());
                prefix.push_back(run);
            }
            InsertMapping mapping =
                im.h2v->insert(tentative, payload, prefix);
            finalIds = mapping.itemToKey;
            for (int id : finalIds) {
                if (id < 1 || id > im.caps.maxHyperedges) {
                    throw escher::EscherError(
                        "DynamicHypergraph: adopted hyperedge id out of "
                        "range; raise Caps.maxHyperedges");
                }
            }
        }
        res.escherMs += msSince(t0);
    }

    // ---------------------------------------------------------------
    // 2. Host shadow update + delta extraction (with the final ids).
    // ---------------------------------------------------------------
    EscherHorizOps ops;
    {
        auto t0 = Clock::now();
        im.host.applyBatch(batch, finalIds, res.delta, ops);
        res.deltaMs = msSince(t0);
    }

    // ---------------------------------------------------------------
    // 3. Remaining ESCHER maintenance, unfills/erases before fills.
    // ---------------------------------------------------------------
    {
        auto t0 = Clock::now();

        // Vertical deletes on h2v (avail propagation, slots become
        // reusable for future best-fit inserts).
        if (!res.delta.deadHe.empty()) {
            im.h2v->erase(res.delta.deadHe);
        }

        // Horizontal removals.
        flushGrouped(ops.h2vUnfill, *im.h2v, /*isFill=*/false);
        flushGrouped(ops.v2hUnfill, *im.v2h, /*isFill=*/false);
        {
            // Translate h2h row ids to h2h keys (pre-insert mapping).
            std::vector<std::pair<int, int>> translated;
            translated.reserve(ops.h2hUnfill.size());
            for (auto [row, val] : ops.h2hUnfill) {
                const int key = im.h2hKeyOfHe[row];
                if (key > 0) translated.emplace_back(key, val);
            }
            flushGrouped(translated, *im.h2h, /*isFill=*/false);
        }

        // Vertical deletes on h2h.
        if (!res.delta.deadHe.empty()) {
            std::vector<int> keys;
            keys.reserve(res.delta.deadHe.size());
            for (int id : res.delta.deadHe) {
                const int key = im.h2hKeyOfHe[id];
                if (key > 0) {
                    keys.push_back(key);
                    im.h2hKeyOfHe[id] = 0;
                }
            }
            if (!keys.empty()) im.h2h->erase(keys);
        }

        // Horizontal additions.
        flushGrouped(ops.h2vFill, *im.h2v, /*isFill=*/true);
        flushGrouped(ops.v2hFill, *im.v2h, /*isFill=*/true);
        {
            std::vector<std::pair<int, int>> translated;
            translated.reserve(ops.h2hFill.size());
            for (auto [row, val] : ops.h2hFill) {
                const int key = im.h2hKeyOfHe[row];
                if (key > 0) translated.emplace_back(key, val);
            }
            flushGrouped(translated, *im.h2h, /*isFill=*/true);
        }

        // Vertical inserts on h2h: one row per new hyperedge with its
        // final neighbor list; adopt whatever keys the best-fit returns.
        if (!res.delta.newHe.empty()) {
            std::vector<int> tentative;
            std::vector<int> payload;
            std::vector<int> prefix;
            tentative.reserve(res.delta.newHe.size());
            int run = 0;
            for (int id : res.delta.newHe) {
                tentative.push_back(id);
                for (int nb : im.host.h2h[id - 1]) payload.push_back(nb);
                run += static_cast<int>(im.host.h2h[id - 1].size());
                prefix.push_back(run);
            }
            InsertMapping mapping =
                im.h2h->insert(tentative, payload, prefix);
            for (std::size_t i = 0; i < res.delta.newHe.size(); ++i) {
                im.h2hKeyOfHe[res.delta.newHe[i]] = mapping.itemToKey[i];
            }
        }

        res.escherMs += msSince(t0);
    }

    return res;
}

} // namespace escher_mosp
