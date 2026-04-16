#ifndef BINNING_HPP
#define BINNING_HPP

#include <vector>

/** Degree binning (technique 04): split item indices into small / medium / large
 *  by per-item size for load-balanced kernel launches (thread / warp / block). */

/** Default thresholds (warp size and heavy row). */
constexpr int BINNING_SMALL_THRESHOLD = 32;   /* degree < 32 -> thread */
constexpr int BINNING_HEAVY_THRESHOLD = 1024; /* degree >= 1024 -> block */

/**
 * Compute degree bin indices from a prefix-sum array.
 * degree[i] = prefixSizes[i] - (i==0 ? 0 : prefixSizes[i-1]).
 * Output: smallOut, mediumOut, largeOut contain indices in [0, K-1].
 */
void binByDegree(const int* prefixSizes,
                 int K,
                 int thSmall,
                 int thLarge,
                 std::vector<int>& smallOut,
                 std::vector<int>& mediumOut,
                 std::vector<int>& largeOut);

#endif /* BINNING_HPP */
