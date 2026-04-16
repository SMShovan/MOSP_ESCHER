#ifndef ESCHER_ERRORS_HPP
#define ESCHER_ERRORS_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace escher {

/**
 * @brief Exception type thrown by the ESCHER core on CUDA failures and
 *        internal invariant violations.
 *
 * Prefer this over abrupt @c exit(-1) so callers (including MOSP) can catch,
 * log, and continue when a single dynamic update fails without tearing down
 * the whole process.
 */
class EscherError : public std::runtime_error {
public:
    explicit EscherError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Wrap a CUDA runtime call and throw @c EscherError on failure.
 *
 * Use the @c ESCHER_CHECK_CUDA macro at the call site so the thrown message
 * carries file and line information.
 */
inline void checkCudaImpl(cudaError_t err, const char* file, int line,
                          const char* expr) {
    if (err != cudaSuccess) {
        std::string msg = "CUDA error at ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        msg += " (";
        msg += expr;
        msg += "): ";
        msg += cudaGetErrorString(err);
        throw EscherError(msg);
    }
}

} // namespace escher

#define ESCHER_CHECK_CUDA(expr)                                                \
    ::escher::checkCudaImpl((expr), __FILE__, __LINE__, #expr)

#endif // ESCHER_ERRORS_HPP
