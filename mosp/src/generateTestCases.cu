/**
 * @file generateTestCases.cu
 * @brief Generate 10 deterministic test cases for dynamic shortest-path validation.
 */

#include "generateTestCases.cuh"

#include "dijkstra.cuh"
#include "generateChangedEdges.cuh"
#include "generateGraphCSR.cuh"
#include "sequentialSOSPUpdate.cuh"
#include "updateGraphCSR.cuh"
#include "updateGraphWithESCHER.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace {

/**
 * @brief Compare two distance files line by line.
 *
 * @param groundTruthPath Path to Dijkstra ground-truth distances.
 * @param sospUpdatePath  Path to SOSP update distances.
 * @return True if every line matches; false otherwise.
 */
bool compareDistanceFiles(const string &groundTruthPath, const string &sospUpdatePath) {
    ifstream groundTruthFile(groundTruthPath);
    ifstream sospUpdateFile(sospUpdatePath);

    if (!groundTruthFile.is_open() || !sospUpdateFile.is_open()) {
        return false;
    }

    string groundTruthLine, sospUpdateLine;
    while (getline(groundTruthFile, groundTruthLine) &&
           getline(sospUpdateFile, sospUpdateLine)) {
        if (groundTruthLine != sospUpdateLine) {
            return false;
        }
    }

    // Both files should end at the same point
    if (getline(groundTruthFile, groundTruthLine) ||
        getline(sospUpdateFile, sospUpdateLine)) {
        return false;
    }

    return true;
}

struct TestCaseParams {
    int numberOfNodes;
    int numberOfEdges;
    int numberOfObjectives;
    int objectiveStartRange;
    int objectiveEndRange;
    int numberOfChangedEdges;
    double insertionPercentage;
    double deletionPercentage;
    int objectiveNumber;
    int source;
    unsigned int graphSeed;
    unsigned int changeSeed;
};

} // namespace

/**
 * @brief Generate 10 deterministic test case directories.
 *
 * @details
 * Each test case directory (`baseDir/testCaseN/`) contains:
 * - `originalGraph/graphCsrRowPtr.txt`, `...ColInd.txt`, `...Values.txt`
 * - `changedEdges/insert.txt`, `delete.txt`
 * - `updatedGraph/updatedGraphCsrRowPtr.txt`, `...ColInd.txt`, `...Values.txt`
 * - `expected/distancesOriginal.txt`, `SSSPTreeOriginal.txt`
 * - `expected/distancesUpdated.txt`, `SSSPTreeUpdated.txt`
 * - `expected/distancesSospUpdate.txt`, `SSSPTreeSospUpdate.txt`
 *
 * All random generation uses fixed seeds for reproducibility.
 *
 * @param baseDir Root directory for test output (e.g. "tests").
 * @return True if all 10 test cases generated successfully; false otherwise.
 */
bool generateTestCases(const string &baseDir) {
    const TestCaseParams cases[10] = {
        // Case 0: small graph, 1 objective, balanced changes, obj 0
        {5, 6, 1, 1, 10, 2, 50, 50, 0, 0, 1001, 2001},
        // Case 1: small graph, 2 objectives, insertion-heavy, obj 1
        {5, 7, 2, 1, 15, 3, 70, 30, 1, 0, 1002, 2002},
        // Case 2: medium graph, 3 objectives, balanced, obj 2
        {8, 12, 3, 1, 20, 4, 50, 50, 2, 0, 1003, 2003},
        // Case 3: medium graph, 2 objectives, insertion-heavy, obj 0
        {10, 15, 2, 1, 10, 5, 80, 20, 0, 0, 1004, 2004},
        // Case 4: small graph, 3 objectives, deletion-heavy, obj 1
        {6, 10, 3, 1, 9, 4, 30, 70, 1, 0, 1005, 2005},
        // Case 5: medium graph, 1 objective, insert-only, obj 0
        {12, 20, 1, 1, 50, 4, 100, 0, 0, 0, 1006, 2006},
        // Case 6: small graph, 2 objectives, delete-only, obj 1
        {7, 12, 2, 1, 25, 3, 0, 100, 1, 0, 1007, 2007},
        // Case 7: larger graph, 3 objectives, mixed, obj 2
        {15, 25, 3, 1, 30, 8, 60, 40, 2, 0, 1008, 2008},
        // Case 8: tiny graph, 1 objective, balanced, obj 0
        {4, 5, 1, 1, 5, 2, 50, 50, 0, 0, 1009, 2009},
        // Case 9: larger graph, 3 objectives, deletion-heavy, obj 0
        {20, 35, 3, 1, 100, 10, 40, 60, 0, 0, 1010, 2010},
    };

    for (int i = 0; i < 10; ++i) {
        const auto &tc = cases[i];
        string caseDir = baseDir + "/testCase" + to_string(i);
        string originalPrefix = caseDir + "/originalGraph/graphCsr";
        string updatedPrefix = caseDir + "/updatedGraph/updatedGraphCsr";
        string insertPath = caseDir + "/changedEdges/insert.txt";
        string deletePath = caseDir + "/changedEdges/delete.txt";
        string expectedDir = caseDir + "/expected";

        cout << "Generating test case " << i << "...\n";

        if (!generateGraphCSR(
                tc.numberOfNodes,
                tc.numberOfEdges,
                true,
                originalPrefix,
                tc.numberOfObjectives,
                tc.objectiveStartRange,
                tc.objectiveEndRange,
                tc.graphSeed)) {
            cout << "Error: Failed to generate graph for test case " << i << "\n";
            return false;
        }

        if (!generateChangedEdges(
                tc.objectiveStartRange,
                tc.objectiveEndRange,
                tc.numberOfObjectives,
                tc.numberOfNodes,
                tc.numberOfChangedEdges,
                tc.insertionPercentage,
                tc.deletionPercentage,
                true,
                true,
                true,
                false,
                originalPrefix,
                insertPath,
                deletePath,
                tc.changeSeed)) {
            cout << "Error: Failed to generate changed edges for test case " << i << "\n";
            return false;
        }

        if (!escher_mosp::updateGraphWithESCHER(
                originalPrefix,
                updatedPrefix,
                insertPath,
                deletePath,
                /*payloadCapacity=*/65536,
                /*directed=*/true)) {
            cout << "Error: Failed to update graph for test case " << i << "\n";
            return false;
        }

        if (!runDijkstraCSR(
                originalPrefix,
                tc.objectiveNumber,
                tc.source,
                expectedDir + "/distancesOriginal.txt",
                expectedDir + "/SSSPTreeOriginal.txt")) {
            cout << "Error: Failed Dijkstra on original for test case " << i << "\n";
            return false;
        }

        if (!runDijkstraCSR(
                updatedPrefix,
                tc.objectiveNumber,
                tc.source,
                expectedDir + "/distancesUpdated.txt",
                expectedDir + "/SSSPTreeUpdated.txt")) {
            cout << "Error: Failed Dijkstra on updated for test case " << i << "\n";
            return false;
        }

        if (!sequentialSOSPUpdate(
                originalPrefix,
                expectedDir + "/distancesOriginal.txt",
                expectedDir + "/SSSPTreeOriginal.txt",
                insertPath,
                deletePath,
                tc.objectiveNumber,
                tc.source,
                expectedDir + "/distancesSospUpdate.txt",
                expectedDir + "/SSSPTreeSospUpdate.txt")) {
            cout << "Error: Failed SOSP update for test case " << i << "\n";
            return false;
        }

        // --- Verify: compare SOSP update distances with ground truth ---
        bool distancesMatch = compareDistanceFiles(
            expectedDir + "/distancesUpdated.txt",
            expectedDir + "/distancesSospUpdate.txt"
        );

        if (distancesMatch) {
            cout << "Test case " << i << ": PASS (distances match ground truth)\n";
        } else {
            cout << "Test case " << i << ": FAIL (distances differ from ground truth)\n";
        }
    }

    // --- Final summary ---
    int passCount = 0;
    for (int i = 0; i < 10; ++i) {
        string expectedDir = baseDir + "/testCase" + to_string(i) + "/expected";
        if (compareDistanceFiles(
                expectedDir + "/distancesUpdated.txt",
                expectedDir + "/distancesSospUpdate.txt")) {
            ++passCount;
        }
    }

    cout << "\n=== Test Summary: " << passCount << "/10 passed ===\n";
    return true;
}
