#ifndef ESCHER_MOSP_HSOSP_INTERNAL_CUH
#define ESCHER_MOSP_HSOSP_INTERNAL_CUH

/**
 * @file hsospInternal.cuh
 * @brief Helpers shared by the H-SOSP translation units (not public API).
 */

#include <stdexcept>
#include <string>

#include "hsosp.cuh"

#define HSOSP_CUDA_CHECK(call)                                                \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err__) + " at " +     \
                                     __FILE__ + ":" +                         \
                                     std::to_string(__LINE__));               \
        }                                                                     \
    } while (0)

namespace escher_mosp {
namespace hsosp {
namespace detail {

inline int gridFor(long long n, int block) {
    return static_cast<int>((n + block - 1) / block);
}

/** The CSR part of buildDeviceH2H (the incidence mirror is kept); also
 *  the rebuild after a tail overflow. */
void buildCsr(DeviceH2H& dev, const HostHypergraph& hg,
              const LineGraphCSR& lg, int maxNodes, double entryHeadroom);

/**
 * Applies the pairs in dev.delta to the CSR (device sort by row + one warp
 * per row) and scatters the weights of the new nodes; afterwards
 * dev.delta.d_sortedKeys holds the sorted directed keys. Returns false if
 * the tail region overflowed.
 */
bool applyDeltaPairs(DeviceH2H& dev);

} // namespace detail
} // namespace hsosp
} // namespace escher_mosp

#endif // ESCHER_MOSP_HSOSP_INTERNAL_CUH
