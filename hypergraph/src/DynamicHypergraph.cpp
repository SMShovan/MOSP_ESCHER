/**
 * @file DynamicHypergraph.cpp
 * @brief ESCHER routing for the dynamic hypergraph (see header for design).
 */

#include "DynamicHypergraph.hpp"

#include <algorithm>
#include <chrono>
#include <ostream>
#include <stdexcept>
#include <string>

#include "escher_errors.hpp"
#include "flatten.hpp"
#include "integrity.hpp"
#include "structure.hpp"

namespace escher_mosp {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}

/** Rows of a batched fill / unfill: row keys, their values back to back
 *  and the inclusive prefix of the value counts (the CBST call format). */
struct GroupedOps {
    std::vector<int> keys;
    std::vector<int> payload;
    std::vector<int> prefix;

    /** Appends @p value to row @p key (a new row unless it is the last). */
    void add(int key, int value) {
        if (keys.empty() || keys.back() != key) {
            keys.push_back(key);
            prefix.push_back(prefix.empty() ? 0 : prefix.back());
        }
        payload.push_back(value);
        ++prefix.back();
    }
};

void flush(const GroupedOps& g, CBSTOperations& cbst, bool isFill) {
    if (g.keys.empty()) return;
    if (isFill) {
        cbst.fill(g.keys, g.payload, g.prefix);
    } else {
        unfillCBST(g.keys, g.payload, g.prefix,
                   const_cast<CBSTContext&>(cbst.context()));
    }
}

/** Groups raw (rowKey, value) pairs by row (a stable sort, so the values of
 *  a row keep their order and the call is deterministic) and issues one
 *  batched fill / unfill. (The original grouped through a std::map of
 *  vectors: about 330 ms per call for the 2.5M h2h pairs of a 50K DBLP
 *  batch.) */
void flushGrouped(std::vector<std::pair<int, int>>& rawOps,
                  CBSTOperations& cbst, bool isFill) {
    if (rawOps.empty()) return;
    std::stable_sort(rawOps.begin(), rawOps.end(),
                     [](const std::pair<int, int>& a,
                        const std::pair<int, int>& b) {
                         return a.first < b.first;
                     });
    GroupedOps g;
    g.payload.reserve(rawOps.size());
    for (const auto& op : rawOps) g.add(op.first, op.second);
    flush(g, cbst, isFill);
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
    /// finishBatch scratch: heId -> index in the batch's new hyperedges
    /// (-1 between batches).
    std::vector<int> newSlotOfHe;
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

long long DynamicHypergraph::checkEscher(const LineGraphCSR& lg,
                                       std::ostream& log) const {
    const Impl& im = *pImpl;
    const HostHypergraph& hg = im.host;
    const int m = hg.maxId();
    long long errors = 0;

    // h2v: key = hyperedge id, values = incident vertices + 1.
    std::vector<std::vector<int>> h2vRows(m);
    for (int id = 1; id <= m; ++id) {
        if (!hg.alive[id - 1]) continue;
        for (int v : hg.heVerts[id - 1]) h2vRows[id - 1].push_back(v + 1);
    }
    errors += checkTreeRows(im.h2v->context(), h2vRows, true, "h2v", log);
    errors += checkSubtreeAvail(im.h2v->context(), "h2v", log);

    // v2h: key = vertex + 1, values = incident alive hyperedge ids.
    std::vector<std::vector<int>> v2hRows(im.numVertices);
    for (int id = 1; id <= m; ++id) {
        if (!hg.alive[id - 1]) continue;
        for (int v : hg.heVerts[id - 1]) v2hRows[v].push_back(id);
    }
    errors += checkTreeRows(im.v2h->context(), v2hRows, true, "v2h", log);

    // h2h: key = h2hKeyOfHe[id], values = line-graph neighbours of id.
    std::vector<std::vector<int>> h2hRows;
    for (int id = 1; id <= m; ++id) {
        const int key = im.h2hKeyOfHe[id];
        if (!hg.alive[id - 1]) {
            if (key != 0) {
                log << "[check] h2h: dead hyperedge " << id
                    << " still owns key " << key << "  <-- FAILED\n";
                ++errors;
            }
            continue;
        }
        if (key < 1) {
            log << "[check] h2h: alive hyperedge " << id
                << " has no h2h key  <-- FAILED\n";
            ++errors;
            continue;
        }
        if (key > static_cast<int>(h2hRows.size())) h2hRows.resize(key);
        h2hRows[key - 1].assign(lg.row(id), lg.row(id) + lg.degree(id));
    }
    errors += checkTreeRows(im.h2h->context(), h2hRows, true, "h2h", log);
    errors += checkSubtreeAvail(im.h2h->context(), "h2h", log);
    return errors;
}

LineGraphCSR DynamicHypergraph::bulkLoad(std::vector<std::vector<int>>&& rows,
                                 std::vector<long long>&& weights,
                                 int sourceHe, int targetHe) {
    Impl& im = *pImpl;
    if (rows.size() != weights.size()) {
        throw escher::EscherError(
            "DynamicHypergraph::bulkLoad: one weight per hyperedge required");
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const int id = static_cast<int>(i) + 1;
        const bool isVirtual = (id == sourceHe || id == targetHe);
        if (isVirtual ? weights[i] != 0 : weights[i] < 1) {
            throw escher::EscherError(
                "DynamicHypergraph::bulkLoad: hyperedge " +
                std::to_string(id) + " has weight " +
                std::to_string(weights[i]) +
                "; weights must be >= 1 (0 only for the virtual source and "
                "target)");
        }
    }
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
    LineGraphCSR lg = im.host.lineGraph();
    im.host.h2hPairCount = lg.numEntries() / 2;
    {
        std::vector<std::vector<int>> h2hRows(m);
        for (int id = 1; id <= m; ++id)
            h2hRows[id - 1].assign(lg.row(id), lg.row(id) + lg.degree(id));
        im.h2h = constructFromRows("h2h", h2hRows, hf, extra);
        im.h2hKeyOfHe.assign(im.caps.maxHyperedges + 1, 0);
        for (int id = 1; id <= m; ++id) im.h2hKeyOfHe[id] = id;
    }
    return lg;
}

DynamicHypergraph::BatchResult DynamicHypergraph::beginBatch(
    const HgBatch& batch) {
    Impl& im = *pImpl;
    BatchResult res;
    // Validate the inserted hyperedges before any structure is modified.
    for (const auto& ins : batch.heInsert) {
        if (ins.weight < 1) {
            throw escher::EscherError(
                "DynamicHypergraph::applyBatch: inserted hyperedge weight " +
                std::to_string(ins.weight) + " < 1; weights must be positive");
        }
        if (ins.vertices.empty()) {
            throw escher::EscherError(
                "DynamicHypergraph::applyBatch: inserted hyperedge without "
                "vertices");
        }
        for (int v : ins.vertices) {
            if (v < 0 || v >= im.numVertices) {
                throw escher::EscherError(
                    "DynamicHypergraph::applyBatch: inserted hyperedge with "
                    "vertex " + std::to_string(v) + " out of range");
            }
        }
    }

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
    // 2. Host incidence update (with the final ids); the line-graph
    //    delta is derived from these incidence changes on the GPU.
    // ---------------------------------------------------------------
    {
        auto t0 = Clock::now();
        im.host.applyBatch(batch, finalIds, res.inc, res.ops);
        res.deltaMs = msSince(t0);
    }
    return res;
}

double DynamicHypergraph::finishBatch(
    BatchResult& res, const std::vector<std::uint64_t>& directedKeys) {
    Impl& im = *pImpl;
    const HostHypergraph& hg = im.host;
    auto t0 = Clock::now();
    const std::vector<int>& deadHe = res.inc.deadHe;
    const std::vector<int>& newHe = res.inc.newHe;

    // Vertical deletes on h2v (avail propagation, slots become reusable
    // for future best-fit inserts).
    if (!deadHe.empty()) im.h2v->erase(deadHe);

    // h2h operations from the line-graph delta: directed keys
    // row << 33 | isInsert << 32 | col (0-based), sorted, so every row's
    // changes are one run with its deletions first and the fill / unfill
    // groups come out in row order without a further sort. Rows of dead
    // hyperedges are erased below (no unfill); rows of new hyperedges are
    // inserted whole (no fill).
    std::vector<int>& newSlot = im.newSlotOfHe;
    if (static_cast<int>(newSlot.size()) <= hg.maxId())
        newSlot.resize(hg.maxId() + 1, -1);
    for (std::size_t i = 0; i < newHe.size(); ++i)
        newSlot[newHe[i]] = static_cast<int>(i);
    GroupedOps h2hUnfill, h2hFill;
    std::vector<int> newRowStart(newHe.size(), 0), newRowLen(newHe.size(), 0);
    std::vector<int> newRowVals;
    long long insKeys = 0, delKeys = 0;
    for (std::uint64_t k : directedKeys) {
        const int row = static_cast<int>(k >> 33) + 1;
        const bool ins = (k >> 32) & 1ull;
        const int val = static_cast<int>(k & 0xffffffffu) + 1;
        if (!ins) {
            ++delKeys;
            if (hg.alive[row - 1]) {
                const int key = im.h2hKeyOfHe[row];
                if (key > 0) h2hUnfill.add(key, val);
            }
        } else {
            ++insKeys;
            const int slot = newSlot[row];
            if (slot >= 0) {
                if (newRowLen[slot]++ == 0)
                    newRowStart[slot] = static_cast<int>(newRowVals.size());
                newRowVals.push_back(val);
            } else {
                const int key = im.h2hKeyOfHe[row];
                if (key > 0) h2hFill.add(key, val);
            }
        }
    }
    for (int id : newHe) newSlot[id] = -1;
    im.host.h2hPairCount += (insKeys - delKeys) / 2;

    // Horizontal removals.
    flushGrouped(res.ops.h2vUnfill, *im.h2v, /*isFill=*/false);
    flushGrouped(res.ops.v2hUnfill, *im.v2h, /*isFill=*/false);
    flush(h2hUnfill, *im.h2h, /*isFill=*/false);

    // Vertical deletes on h2h.
    if (!deadHe.empty()) {
        std::vector<int> keys;
        keys.reserve(deadHe.size());
        for (int id : deadHe) {
            const int key = im.h2hKeyOfHe[id];
            if (key > 0) {
                keys.push_back(key);
                im.h2hKeyOfHe[id] = 0;
            }
        }
        if (!keys.empty()) im.h2h->erase(keys);
    }

    // Horizontal additions.
    flushGrouped(res.ops.h2vFill, *im.h2v, /*isFill=*/true);
    flushGrouped(res.ops.v2hFill, *im.v2h, /*isFill=*/true);
    flush(h2hFill, *im.h2h, /*isFill=*/true);

    // Vertical inserts on h2h: one row per new hyperedge with its final
    // neighbour list; adopt whatever keys the best-fit returns.
    if (!newHe.empty()) {
        std::vector<int> tentative;
        std::vector<int> payload;
        std::vector<int> prefix;
        tentative.reserve(newHe.size());
        int run = 0;
        payload.reserve(newRowVals.size());
        for (std::size_t i = 0; i < newHe.size(); ++i) {
            tentative.push_back(newHe[i]);
            payload.insert(payload.end(),
                           newRowVals.begin() + newRowStart[i],
                           newRowVals.begin() + newRowStart[i] +
                               newRowLen[i]);
            run += newRowLen[i];
            prefix.push_back(run);
        }
        InsertMapping mapping = im.h2h->insert(tentative, payload, prefix);
        for (std::size_t i = 0; i < newHe.size(); ++i)
            im.h2hKeyOfHe[newHe[i]] = mapping.itemToKey[i];
    }
    return msSince(t0);
}

} // namespace escher_mosp
