#ifndef ESCHER_MOSP_TEST_UTIL_CUH
#define ESCHER_MOSP_TEST_UTIL_CUH

/**
 * @file test_util.cuh
 * @brief Shared checks for the H-SOSP tests: device CSR rows and SOSP
 *        results against the independent oracle (HypergraphOracle.hpp).
 */

#include <algorithm>
#include <cstdio>
#include <vector>

#include "HypergraphOracle.hpp"
#include "hsosp.cuh"

namespace escher_mosp {
namespace testutil {

/** Number of device rows that differ from the oracle's line graph. */
inline long long deviceRowMismatches(const hsosp::DeviceH2H& dev,
                                     const LineGraphCSR& lg,
                                     const char* what) {
    auto rows = hsosp::downloadRows(dev, lg.numIds);
    long long bad = 0;
    for (int id = 1; id <= lg.numIds; ++id) {
        std::vector<int> want(lg.row(id), lg.row(id) + lg.degree(id));
        if (rows[id - 1] != want) {
            if (bad++ < 3)
                std::printf("  %s: device row %d has %zu entries, oracle %zu\n",
                            what, id, rows[id - 1].size(), want.size());
        }
    }
    return bad;
}

/** Number of rows of @p a that differ from @p b (both sorted). */
inline long long lineGraphMismatches(const LineGraphCSR& a,
                                     const LineGraphCSR& b) {
    if (a.numIds != b.numIds) return 1;
    long long bad = 0;
    for (int id = 1; id <= a.numIds; ++id)
        if (!std::equal(a.row(id), a.row(id) + a.degree(id), b.row(id),
                        b.row(id) + b.degree(id)))
            ++bad;
    return bad;
}

/** Device distances and parents of @p st against the oracle. */
inline SospCheck checkState(const hsosp::HsospState& st,
                            const HostHypergraph& hg,
                            const LineGraphCSR& lg) {
    std::vector<long long> dist;
    std::vector<int> parent;
    st.downloadDistances(dist, hg.maxId());
    st.downloadParents(parent, hg.maxId());
    auto ref = referenceDistances(hg, lg, hg.sourceHe);
    return checkSosp(hg, lg, ref, dist, parent);
}

} // namespace testutil
} // namespace escher_mosp

#endif // ESCHER_MOSP_TEST_UTIL_CUH
