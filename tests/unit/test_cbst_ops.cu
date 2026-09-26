/**
 * @file test_cbst_ops.cu
 * @brief Content test for libescher_core: every CBST operation is mirrored
 *        on a host model and the tree is read back and compared with it
 *        (checkTreeRows, checkSubtreeAvail) after every step.
 *
 * Usage: test_cbst_ops <scenario>...   (no argument: all scenarios)
 *
 * Scenarios (each targets one defect of the original CBST code; "random"
 * mixes every operation at sizes on both sides of 65,536 records):
 *   scale        construct n = 1..1100 and n = 65,535 .. 2^20: every key
 *                must be found (int32 overflow of the in-order rank formula)
 *   reuse        erase, then reuse-insert twice: subtreeAvail must stay
 *                consistent and the returned keys distinct and nonzero
 *   terminator   rows whose length is a multiple of 4 (no INT_MIN
 *                terminator was written for them)
 *   erase        erase an inner node, then fill / unfill / erase keys of its
 *                left subtree (the erased node's key was overwritten by -1)
 *   surplus      a surplus insert rebuilds the tree; later fills must append
 *                (the rebuild reset occupancy / tail metadata)
 *   bestfit      reused slots must follow the best-fit matching and hold
 *                the whole item (items were given slots in BST order and
 *                truncated)
 *   unfill-chain unfill on a chained row, then fill (tail occupancy was
 *                decremented by removals from every segment)
 *   random       random erase / insert / fill / unfill sequences with rows
 *                of 1..3,000 values (thread, warp and block kernels)
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "escher_errors.hpp"
#include "flatten.hpp"
#include "integrity.hpp"
#include "structure.hpp"

namespace {

/** CBST plus the host model it must match. */
class Harness {
public:
    Harness(const char* name, long long payloadCapacity)
        : name_(name), op_(name, static_cast<int>(payloadCapacity), 4) {}

    void construct(const std::vector<std::vector<int>>& rows) {
        auto [flat, offsets] = flatten2DVector(rows);
        std::vector<int> keys(rows.size()), counts(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            keys[i] = static_cast<int>(i) + 1;
            counts[i] = static_cast<int>(rows[i].size());
        }
        op_.construct(keys.data(), offsets.data(),
                      static_cast<int>(rows.size()), flat.data(),
                      static_cast<int>(flat.size()), counts.data());
        model_ = rows;
    }

    void fill(const std::map<int, std::vector<int>>& adds) {
        std::vector<int> keys, payload, prefix;
        flattenOps(adds, keys, payload, prefix);
        if (keys.empty()) return;
        op_.fill(keys, payload, prefix);
        for (const auto& [k, vals] : adds)
            model_[k - 1].insert(model_[k - 1].end(), vals.begin(),
                                 vals.end());
    }

    /** Removes every occurrence of the listed values from each row. */
    void unfill(const std::map<int, std::vector<int>>& removals) {
        std::vector<int> keys, payload, prefix;
        flattenOps(removals, keys, payload, prefix);
        if (keys.empty()) return;
        unfillCBST(keys, payload, prefix,
                   const_cast<CBSTContext&>(op_.context()));
        for (const auto& [k, vals] : removals) {
            std::vector<int>& row = model_[k - 1];
            std::set<int> drop(vals.begin(), vals.end());
            row.erase(std::remove_if(row.begin(), row.end(),
                                     [&](int v) { return drop.count(v); }),
                      row.end());
        }
    }

    void erase(const std::vector<int>& keys) {
        op_.erase(keys);
        for (int k : keys) model_[k - 1].clear();
    }

    /** Inserts rows; returns the keys the CBST assigned. */
    std::vector<int> insert(const std::vector<std::vector<int>>& items) {
        std::vector<int> tentative, payload, prefix;
        int run = 0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            tentative.push_back(static_cast<int>(model_.size() + i) + 1);
            payload.insert(payload.end(), items[i].begin(), items[i].end());
            run += static_cast<int>(items[i].size());
            prefix.push_back(run);
        }
        InsertMapping m = op_.insert(tentative, payload, prefix);
        std::set<int> distinct;
        for (std::size_t i = 0; i < items.size(); ++i) {
            const int key = m.itemToKey[i];
            if (key < 1 || !distinct.insert(key).second || live(key)) {
                log_ << "[check] " << name_ << " insert item " << i
                     << " got key " << key
                     << " (zero, duplicate or a live key)  <-- FAILED\n";
                ++errors;
                continue;
            }
            if (key > static_cast<int>(model_.size())) model_.resize(key);
            model_[key - 1] = items[i];
        }
        return m.itemToKey;
    }

    /** Compares the tree with the model; prints the details on failure. */
    void check(const std::string& step) {
        const long long before = errors;
        log_ << "-- " << name_ << ": " << step << "\n";
        errors += checkTreeRows(op_.context(), model_, true, name_, log_);
        errors += checkSubtreeAvail(op_.context(), name_, log_);
        if (errors != before) std::cout << log_.str();
        log_.str("");
    }

    bool live(int key) const {
        return key >= 1 && key <= static_cast<int>(model_.size()) &&
               !model_[key - 1].empty();
    }
    int numKeys() const { return static_cast<int>(model_.size()); }
    const std::vector<int>& row(int key) const { return model_[key - 1]; }

    long long errors = 0;

private:
    static void flattenOps(const std::map<int, std::vector<int>>& ops,
                           std::vector<int>& keys, std::vector<int>& payload,
                           std::vector<int>& prefix) {
        int run = 0;
        for (const auto& [k, vals] : ops) {
            if (vals.empty()) continue;
            keys.push_back(k);
            payload.insert(payload.end(), vals.begin(), vals.end());
            run += static_cast<int>(vals.size());
            prefix.push_back(run);
        }
    }

    const char* name_;
    CBSTOperations op_;
    std::vector<std::vector<int>> model_;
    std::ostringstream log_;
};

std::vector<int> iotaRow(int from, int len) {
    std::vector<int> r(len);
    for (int i = 0; i < len; ++i) r[i] = from + i;
    return r;
}

long long scenarioScale() {
    std::vector<int> sizes;
    for (int n = 1; n <= 1100; ++n) sizes.push_back(n);
    for (int n : {65535, 65536, 65537, 70000, 131071, 200000, 1 << 20})
        sizes.push_back(n);
    for (int n : sizes) {
        std::vector<std::vector<int>> rows(n);
        for (int i = 0; i < n; ++i) rows[i] = iotaRow(3 * i + 1, 1 + i % 3);
        Harness h("scale", 12LL * n + 1024);
        h.construct(rows);
        h.check("construct n=" + std::to_string(n));
        if (h.errors) return h.errors;
    }
    return 0;
}

long long scenarioReuse() {
    const int n = 1000;
    std::mt19937_64 rng(7);
    std::vector<std::vector<int>> rows(n);
    for (int i = 0; i < n; ++i) rows[i] = iotaRow(10 * i + 1, 1 + i % 5);
    Harness h("reuse", 64LL * n);
    h.construct(rows);
    std::set<int> victims;
    while (victims.size() < 59) victims.insert(1 + static_cast<int>(rng() % n));
    h.erase(std::vector<int>(victims.begin(), victims.end()));
    h.check("erase 59");
    std::vector<std::vector<int>> items;
    for (int i = 0; i < 20; ++i) items.push_back(iotaRow(100000 + 7 * i, 1));
    h.insert(items);
    h.check("reuse-insert 20");
    items.clear();
    for (int i = 0; i < 40; ++i) items.push_back(iotaRow(200000 + 7 * i, 2));
    h.insert(items);
    h.check("insert 40 (39 reusable slots, 1 surplus)");
    return h.errors;
}

long long scenarioTerminator() {
    Harness h("terminator", 4096);
    h.construct({{1, 2, 3, 4}, {5, 6}, iotaRow(7, 8), {15, 16, 17}});
    h.check("construct");
    h.fill({{1, {9}}, {3, {100, 101}}});
    h.check("fill rows of length 4 and 8");
    h.unfill({{1, {2}}, {3, {8, 9}}});
    h.check("unfill");
    return h.errors;
}

long long scenarioErase() {
    std::vector<std::vector<int>> rows(15);
    for (int k = 1; k <= 15; ++k) rows[k - 1] = {10 * k};
    Harness h("erase", 4096);
    h.construct(rows);
    h.erase({8});   // the root of a 15-node CBST
    h.check("erase root key 8");
    h.fill({{3, {999}}, {12, {777}}});
    h.check("fill keys 3 and 12");
    h.unfill({{5, {50}}});
    h.check("unfill key 5");
    h.erase({3});
    h.check("erase key 3");
    return h.errors;
}

long long scenarioSurplus() {
    Harness h("surplus", 4096);
    h.construct({{10}, {20, 21, 22}, {30}});
    h.insert({{40, 41}});   // no deleted slot: surplus append + rebuild
    h.check("surplus insert");
    h.fill({{2, {777}}, {1, {11, 12, 13, 14, 15, 16}}});
    h.check("fill after rebuild");
    h.fill({{4, {42}}});
    h.check("fill the new row");
    return h.errors;
}

long long scenarioBestFit() {
    Harness h("bestfit", 4096);
    h.construct({{10}, {20}, {30}, {40}, iotaRow(500, 13), {60}});
    h.erase({2, 5});   // capacities 3 and 15
    h.check("erase keys 2 and 5");
    h.insert({iotaRow(100, 10), {7, 8}});
    h.check("insert a 10-value and a 2-value item");
    return h.errors;
}

long long scenarioUnfillChain() {
    Harness h("unfill-chain", 4096);
    h.construct({{1, 2}, {50}});
    h.fill({{1, {3, 4, 5}}});   // overflows into a chained segment
    h.check("fill into a chain");
    h.unfill({{1, {1}}});        // removal from the head segment
    h.check("unfill from the head segment");
    h.fill({{1, {6}}});
    h.check("fill after unfill");
    h.unfill({{1, {5, 6}}});
    h.fill({{1, {7, 8, 9, 10, 11, 12, 13}}});
    h.check("unfill tail, fill again");
    return h.errors;
}

long long scenarioRandom() {
    long long errors = 0;
    for (int n : {300, 5000, 70000}) {
        std::mt19937_64 rng(1234 + n);
        auto rnd = [&](int lo, int hi) {
            return lo + static_cast<int>(rng() % (hi - lo + 1));
        };
        // Row lengths: mostly short, some warp-sized, a few block-sized.
        auto rowLen = [&]() {
            int r = rnd(0, 99);
            return r < 85 ? rnd(1, 12) : (r < 98 ? rnd(32, 200)
                                                 : rnd(1024, 3000));
        };
        int nextValue = 1;
        auto freshRow = [&](int len) {
            std::vector<int> r(len);
            for (int& v : r) v = nextValue++;
            return r;
        };
        std::vector<std::vector<int>> rows(n);
        for (auto& r : rows) r = freshRow(rowLen());
        Harness h("random", 400LL * n + (8 << 20));
        h.construct(rows);
        h.check("construct n=" + std::to_string(n));
        for (int round = 0; round < 6 && h.errors == 0; ++round) {
            // Erase ~4% of the live keys.
            std::set<int> er;
            for (int i = 0; i < n / 25; ++i) {
                int k = rnd(1, h.numKeys());
                if (h.live(k)) er.insert(k);
            }
            h.erase(std::vector<int>(er.begin(), er.end()));
            // Unfill part of ~5% of the rows (some from chained rows).
            std::map<int, std::vector<int>> rem;
            for (int i = 0; i < n / 20; ++i) {
                int k = rnd(1, h.numKeys());
                if (!h.live(k) || rem.count(k)) continue;
                const auto& r = h.row(k);
                std::vector<int> vals;
                for (int v : r)
                    if (rng() % 3 == 0) vals.push_back(v);
                if (vals.size() == r.size()) vals.pop_back();   // keep >= 1
                if (!vals.empty()) rem[k] = vals;
            }
            h.unfill(rem);
            // Fill ~6% of the rows, some past their capacity (chains).
            std::map<int, std::vector<int>> add;
            for (int i = 0; i < n / 16; ++i) {
                int k = rnd(1, h.numKeys());
                if (!h.live(k) || rem.count(k) || add.count(k)) continue;
                add[k] = freshRow(rowLen() / (rng() % 4 == 0 ? 1 : 3) + 1);
            }
            h.fill(add);
            // Insert ~5% new rows: reuse deleted slots and append surplus.
            std::vector<std::vector<int>> items;
            for (int i = 0; i < n / 20; ++i) items.push_back(freshRow(rowLen()));
            h.insert(items);
            h.check("round " + std::to_string(round) + " n=" +
                    std::to_string(n));
        }
        errors += h.errors;
        if (errors) break;
    }
    return errors;
}

struct Scenario {
    const char* name;
    long long (*run)();
};

const Scenario kScenarios[] = {
    {"scale", scenarioScale},       {"reuse", scenarioReuse},
    {"terminator", scenarioTerminator}, {"erase", scenarioErase},
    {"surplus", scenarioSurplus},   {"bestfit", scenarioBestFit},
    {"unfill-chain", scenarioUnfillChain}, {"random", scenarioRandom},
};

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> wanted(argv + 1, argv + argc);
    int failed = 0, ran = 0;
    for (const Scenario& s : kScenarios) {
        if (!wanted.empty() &&
            std::find(wanted.begin(), wanted.end(), s.name) == wanted.end())
            continue;
        ++ran;
        long long errors = 0;
        try {
            errors = s.run();
        } catch (const std::exception& e) {
            std::cout << "exception: " << e.what() << "\n";
            errors = 1;
        }
        std::cout << (errors ? "FAIL  " : "PASS  ") << s.name;
        if (errors) std::cout << " (" << errors << " violations)";
        std::cout << "\n";
        if (errors) ++failed;
    }
    if (ran == 0) {
        std::cout << "unknown scenario\n";
        return 2;
    }
    std::cout << "test_cbst_ops: " << (ran - failed) << "/" << ran
              << " scenarios passed\n";
    return failed ? 1 : 0;
}
