/**
 * @file integrity.cu
 * @brief Host read-back checker for CBST contents (see integrity.hpp).
 *
 * Deliberately independent of the device code: it re-implements the BST
 * search and the row walk (values, zero / INT_MIN terminators, negative
 * chain pointers) on host copies of the arrays.
 */

#include "../include/integrity.hpp"
#include "../include/escher_errors.hpp"

#include <algorithm>
#include <climits>
#include <ostream>
#include <utility>

namespace {

struct HostTree {
    const CBSTContext* ctx;
    std::vector<CBSTNode> nodes;
    std::vector<int> avail;
    std::vector<int> flat;

    explicit HostTree(const CBSTContext& c) : ctx(&c) {
        const int n = c.numRecords;
        if (n <= 0 || c.d_nodes == nullptr) return;
        nodes.resize(n);
        avail.assign(n, 0);
        // Live rows lie in the used prefix of the payload.
        flat.resize(static_cast<std::size_t>(c.initialPayloadSize));
        ESCHER_CHECK_CUDA(cudaMemcpy(nodes.data(), c.d_nodes,
                                     sizeof(CBSTNode) * n,
                                     cudaMemcpyDeviceToHost));
        if (c.d_avail) {
            ESCHER_CHECK_CUDA(cudaMemcpy(avail.data(), c.d_avail,
                                         sizeof(int) * n,
                                         cudaMemcpyDeviceToHost));
        }
        ESCHER_CHECK_CUDA(cudaMemcpy(flat.data(), c.d_flatPayload,
                                     sizeof(int) * flat.size(),
                                     cudaMemcpyDeviceToHost));
    }

    int size() const { return static_cast<int>(nodes.size()); }
    bool live(int t) const { return avail[t] == 0 && nodes[t].index >= 1; }

    int child(const CBSTNode* p) const {
        return p == nullptr ? -1 : static_cast<int>(p - ctx->d_nodes);
    }

    // Same walk as the device kernels: compare with node->index, go
    // left / right.
    int find(int key) const {
        int t = size() > 0 ? 0 : -1;
        for (int depth = 0; t >= 0; ++depth) {
            if (t >= size() || depth > 64) return -1;
            if (nodes[t].index == key) return t;
            t = nodes[t].index > key ? child(nodes[t].left)
                                     : child(nodes[t].right);
        }
        return -1;
    }

    // One payload segment of a row: [begin, end] inclusive, end = the
    // terminator or chain pointer.
    struct Segment {
        long long begin, end;
    };

    // Reads a row following chain pointers. Returns false if the walk
    // leaves the payload or does not terminate.
    bool readRow(int t, std::vector<int>& row,
                 std::vector<Segment>* segments) const {
        row.clear();
        long long loc = nodes[t].value, segBegin = loc;
        const long long limit = static_cast<long long>(flat.size());
        for (long long steps = 0; steps <= limit; ++steps) {
            if (loc < 0 || loc >= limit) return false;
            const int v = flat[loc];
            if (v < 0 && v != INT_MIN) {
                if (segments) segments->push_back({segBegin, loc});
                loc = -static_cast<long long>(v);
                segBegin = loc;
                continue;
            }
            if (v == 0 || v == INT_MIN) {
                if (segments) segments->push_back({segBegin, loc});
                return true;
            }
            row.push_back(v);
            ++loc;
        }
        return false;
    }
};

void printRow(std::ostream& log, const std::vector<int>& r) {
    log << "[";
    for (std::size_t i = 0; i < r.size() && i < 12; ++i)
        log << (i ? " " : "") << r[i];
    if (r.size() > 12) log << " ... (" << r.size() << " values)";
    log << "]";
}

} // namespace

long long checkTreeRows(const CBSTContext& ctx,
                        const std::vector<std::vector<int>>& expected,
                        bool orderInsensitive, const char* name,
                        std::ostream& log) {
    HostTree tree(ctx);
    const int maxDetails = 5;
    long long rowErrors = 0, reachErrors = 0, metaErrors = 0,
              overlapErrors = 0;
    int details = 0;

    // 1. Rows by key, searched from the root.
    int maxKey = static_cast<int>(expected.size());
    for (int t = 0; t < tree.size(); ++t)
        if (tree.live(t)) maxKey = std::max(maxKey, tree.nodes[t].index);
    std::vector<int> got, want;
    for (int key = 1; key <= maxKey; ++key) {
        want.clear();
        if (key <= static_cast<int>(expected.size())) want = expected[key - 1];
        const int t = tree.find(key);
        bool ok = true;
        got.clear();
        if (t >= 0 && tree.live(t)) ok = tree.readRow(t, got, nullptr);
        if (orderInsensitive) {
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
        }
        if (!ok || got != want) {
            ++rowErrors;
            if (details++ < maxDetails) {
                log << "[check] " << name << " key " << key
                    << (t < 0 ? " (not found)" : "")
                    << (ok ? "" : " (unterminated row)") << ": expected ";
                printRow(log, want);
                log << " got ";
                printRow(log, got);
                log << "\n";
            }
        }
    }

    // 2. Every live node is reachable by a search for its own key.
    for (int t = 0; t < tree.size(); ++t) {
        if (tree.live(t) && tree.find(tree.nodes[t].index) != t) {
            ++reachErrors;
            if (details++ < maxDetails)
                log << "[check] " << name << " node " << t << " (key "
                    << tree.nodes[t].index
                    << ") is not reachable by BST search\n";
        }
    }

    // 3. Tail-segment metadata and 4. disjoint payload segments.
    std::vector<std::pair<long long, long long>> extents;
    std::vector<HostTree::Segment> segments;
    for (int t = 0; t < tree.size(); ++t) {
        if (!tree.live(t)) continue;
        const CBSTNode& nd = tree.nodes[t];
        segments.clear();
        if (!tree.readRow(t, got, &segments)) continue;  // reported above
        const long long tb = nd.tailBase, occ = nd.occupancy,
                        cap = nd.tailCapacity;
        bool metaOk = segments.back().begin == tb && occ >= 0 && occ <= cap &&
                      tb + cap < static_cast<long long>(tree.flat.size());
        if (metaOk) {
            const long long liveInTail =
                segments.back().end - segments.back().begin;
            metaOk = liveInTail == occ;
            for (long long p = tb + occ + 1; metaOk && p <= tb + cap; ++p)
                if (tree.flat[p] > 0 ||
                    (tree.flat[p] < 0 && tree.flat[p] != INT_MIN))
                    metaOk = false;
        }
        if (!metaOk) {
            ++metaErrors;
            if (details++ < maxDetails)
                log << "[check] " << name << " key " << nd.index
                    << ": tail metadata (tailBase " << tb << ", occupancy "
                    << occ << ", tailCapacity " << cap
                    << ") does not match the payload\n";
        }
        for (std::size_t s = 0; s < segments.size(); ++s) {
            long long b = segments[s].begin, e = segments[s].end;
            // The first segment owns `length` slots.
            if (s == 0) e = std::max(e, b + nd.length - 1);
            if (s + 1 == segments.size() && metaOk) e = std::max(e, tb + cap);
            extents.push_back({b, e});
        }
    }
    std::sort(extents.begin(), extents.end());
    for (std::size_t i = 1; i < extents.size(); ++i) {
        if (extents[i].first <= extents[i - 1].second) {
            ++overlapErrors;
            if (details++ < maxDetails)
                log << "[check] " << name << " payload segments overlap: ["
                    << extents[i - 1].first << ", " << extents[i - 1].second
                    << "] and [" << extents[i].first << ", "
                    << extents[i].second << "]\n";
        }
    }

    const long long total =
        rowErrors + reachErrors + metaErrors + overlapErrors;
    log << "[check] " << name << ": " << maxKey << " keys, " << rowErrors
        << " wrong rows, " << reachErrors << " unreachable nodes, "
        << metaErrors << " metadata violations, " << overlapErrors
        << " overlapping segments" << (total ? "  <-- FAILED" : "") << "\n";
    return total;
}

long long checkSubtreeAvail(const CBSTContext& ctx, const char* name,
                            std::ostream& log) {
    const int n = ctx.numRecords;
    if (n <= 0 || ctx.d_avail == nullptr) return 0;
    std::vector<int> avail(n), sub(n);
    ESCHER_CHECK_CUDA(cudaMemcpy(avail.data(), ctx.d_avail, sizeof(int) * n,
                                 cudaMemcpyDeviceToHost));
    ESCHER_CHECK_CUDA(cudaMemcpy(sub.data(), ctx.d_subtreeAvail,
                                 sizeof(int) * n, cudaMemcpyDeviceToHost));
    long long bad = 0;
    for (int i = 0; i < n; ++i) {
        const long long l = 2LL * i + 1 < n ? sub[2 * i + 1] : 0;
        const long long r = 2LL * i + 2 < n ? sub[2 * i + 2] : 0;
        if (sub[i] != avail[i] + l + r) {
            if (bad < 5)
                log << "[check] " << name << " subtreeAvail[" << i
                    << "] = " << sub[i] << ", expected " << avail[i] + l + r
                    << "\n";
            ++bad;
        }
    }
    log << "[check] " << name << ": " << bad
        << " inconsistent subtreeAvail entries" << (bad ? "  <-- FAILED" : "")
        << "\n";
    return bad;
}
