/**
 * @file test_cbst_smoke.cu
 * @brief Smoke test for @c libescher_core: construct a CBST at a variety of
 *        sizes (including non-power-of-two) and run a basic insert/erase round.
 *
 * Covers the build_tree.cu tree-position formula for N in
 * {1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000} which was flagged as fragile
 * under non-power-of-two tree sizes during code review.
 */

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <vector>

#include "escher_errors.hpp"
#include "flatten.hpp"
#include "structure.hpp"

namespace {

/**
 * @brief Construct a CBST with @p n records of a tiny 2-int payload,
 *        then insert one extra record and erase one existing record.
 *
 * @return True if the sequence completes without throwing.
 */
bool runOneSize(int n) {
    // Build n records, each of size 2: [k, k*10].
    std::vector<std::vector<int>> rows(n);
    for (int i = 0; i < n; ++i) rows[i] = {i + 1, (i + 1) * 10};

    auto [flatValues, startOffsets] = flatten2DVector(rows);
    std::vector<int> keys(n);
    std::iota(keys.begin(), keys.end(), 1);

    CBSTOperations op("smoke", /*payloadCapacity=*/8192, /*alignment=*/4);
    op.construct(keys.data(), startOffsets.data(), n,
                 flatValues.data(), static_cast<int>(flatValues.size()));

    // Erase the last record (if any) and insert a fresh one. Exercises
    // CBSTOperations::erase -> deleteCBST and ::insert -> insertCBST.
    if (n >= 1) {
        std::vector<int> toErase = {n};
        op.erase(toErase);

        std::vector<int> newKeys       = {n + 1};
        std::vector<int> newPayload    = {n + 1, (n + 1) * 10};
        std::vector<int> newPrefixSize = {2};
        op.insert(newKeys, newPayload, newPrefixSize);
    }
    return true;
}

} // namespace

int main() {
    const int sizes[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000};
    int passed = 0;
    for (int n : sizes) {
        try {
            if (runOneSize(n)) {
                std::cout << "[smoke] PASS n=" << n << "\n";
                ++passed;
            } else {
                std::cout << "[smoke] FAIL n=" << n << " (runOneSize returned false)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "[smoke] FAIL n=" << n << " threw: " << e.what() << "\n";
        }
    }
    const int total = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));
    std::cout << "test_cbst_smoke: " << passed << "/" << total << " passed\n";
    return (passed == total) ? 0 : 1;
}
