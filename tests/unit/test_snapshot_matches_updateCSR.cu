/**
 * @file test_snapshot_matches_updateCSR.cu
 * @brief End-to-end correctness test: ESCHER-backed @c updateGraphWithESCHER
 *        must produce byte-identical output to the legacy @c updateGraphCSR
 *        for the same inputs.
 *
 * Steps:
 *   1. Generate a random CSR and insert/delete lists into a fresh temp dir.
 *   2. Run the legacy path: @c updateGraphCSR writes one set of updated files.
 *   3. Run the ESCHER path: @c updateGraphWithESCHER writes a parallel set.
 *   4. Diff the two sets line-by-line.
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "generateChangedEdges.cuh"
#include "generateGraphCSR.cuh"
#include "updateGraphCSR.cuh"
#include "updateGraphWithESCHER.hpp"

namespace {

/**
 * @brief Read an entire file into a string for equality comparison.
 */
std::string slurp(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return "<missing: " + path + ">";
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/**
 * @brief Compare two CSR triples file-by-file and print mismatches.
 */
bool diffCsrTriple(const std::string& prefixA, const std::string& prefixB) {
    bool ok = true;
    for (const char* suffix : {"RowPtr.txt", "ColInd.txt", "Values.txt"}) {
        const std::string a = slurp(prefixA + suffix);
        const std::string b = slurp(prefixB + suffix);
        if (a != b) {
            std::cout << "  DIFF " << suffix << "\n";
            std::cout << "  --- legacy (" << prefixA << suffix << "):\n" << a;
            std::cout << "  --- escher (" << prefixB << suffix << "):\n" << b;
            ok = false;
        }
    }
    return ok;
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    // Fixed seed so the test is reproducible independent of the MOSP
    // stress-test harness.
    const int V      = 30;
    const int E      = 80;
    const int K      = 2;
    const int chg    = 15;
    const int seed   = 424242;
    const int chgSeed = 737373;

    const std::string baseDir       = "tests/tmp/test_snapshot";
    const std::string originalPrefix = baseDir + "/original/graphCsr";
    const std::string legacyPrefix   = baseDir + "/legacy/updatedCsr";
    const std::string escherPrefix   = baseDir + "/escher/updatedCsr";
    const std::string insertPath     = baseDir + "/edges/insert.txt";
    const std::string deletePath     = baseDir + "/edges/delete.txt";

    fs::create_directories(baseDir + "/original");
    fs::create_directories(baseDir + "/legacy");
    fs::create_directories(baseDir + "/escher");
    fs::create_directories(baseDir + "/edges");

    if (!generateGraphCSR(V, E, /*directed=*/true, originalPrefix,
                          K, /*objMin=*/1, /*objMax=*/100, seed)) {
        std::cout << "test_snapshot_matches_updateCSR: FAIL (generateGraphCSR)\n";
        return 1;
    }
    if (!generateChangedEdges(1, 100, K, V, chg,
                              /*insertPct=*/60.0, /*deletePct=*/40.0,
                              /*directed=*/true, /*existOnly=*/true,
                              /*allowDup=*/false, /*selfLoop=*/false,
                              originalPrefix, insertPath, deletePath, chgSeed)) {
        std::cout << "test_snapshot_matches_updateCSR: FAIL (generateChangedEdges)\n";
        return 1;
    }

    // Legacy path
    if (!updateGraphCSR(originalPrefix, legacyPrefix,
                        insertPath, deletePath, /*directed=*/true)) {
        std::cout << "test_snapshot_matches_updateCSR: FAIL (updateGraphCSR)\n";
        return 1;
    }

    // ESCHER path
    try {
        if (!escher_mosp::updateGraphWithESCHER(
                originalPrefix, escherPrefix,
                insertPath, deletePath,
                /*payloadCapacity=*/65536, /*directed=*/true)) {
            std::cout << "test_snapshot_matches_updateCSR: FAIL (updateGraphWithESCHER returned false)\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "test_snapshot_matches_updateCSR: FAIL (threw: " << e.what() << ")\n";
        return 1;
    }

    if (!diffCsrTriple(legacyPrefix, escherPrefix)) {
        std::cout << "test_snapshot_matches_updateCSR: FAIL (CSR mismatch)\n";
        return 1;
    }

    std::cout << "test_snapshot_matches_updateCSR: PASS\n";
    return 0;
}
