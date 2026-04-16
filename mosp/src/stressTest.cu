/**
 * @file stressTest.cu
 * @brief Randomized stress test for the sequential SOSP update algorithm.
 *
 * Generates random graphs with random changes, runs both full Dijkstra
 * (ground truth) and the SOSP update algorithm, and compares distances.
 * Uses random seeds for each iteration to test robustness.
 */

#include "dijkstra.cuh"
#include "generateChangedEdges.cuh"
#include "generateGraphCSR.cuh"
#include "sequentialSOSPUpdate.cuh"
#include "updateGraphCSR.cuh"
#include "updateGraphWithESCHER.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace std;

namespace {

bool compareDistanceFiles(const string &pathA, const string &pathB) {
    ifstream fileA(pathA);
    ifstream fileB(pathB);
    if (!fileA.is_open() || !fileB.is_open()) return false;

    string lineA, lineB;
    while (getline(fileA, lineA) && getline(fileB, lineB)) {
        if (lineA != lineB) return false;
    }
    return !getline(fileA, lineA) && !getline(fileB, lineB);
}

} // namespace

int main() {
    const int totalRuns = 100;
    const string baseDir = "stressTest";

    mt19937 rng(random_device{}());
    uniform_int_distribution<int> nodesDist(4, 30);
    uniform_int_distribution<int> objDist(1, 3);
    uniform_int_distribution<int> weightDist(1, 50);
    uniform_int_distribution<int> changeDist(1, 10);
    uniform_int_distribution<int> pctDist(0, 100);
    uniform_int_distribution<unsigned int> seedDist(1, 999999);

    int passCount = 0;
    int failCount = 0;

    for (int run = 0; run < totalRuns; ++run) {
        int numberOfNodes = nodesDist(rng);
        int minEdges = numberOfNodes - 1;
        int maxEdges = min(numberOfNodes * (numberOfNodes - 1), minEdges + 40);
        uniform_int_distribution<int> edgeDist(minEdges, maxEdges);
        int numberOfEdges = edgeDist(rng);
        int numberOfObjectives = objDist(rng);
        int objectiveEndRange = weightDist(rng);
        int objectiveIndex = uniform_int_distribution<int>(0, numberOfObjectives - 1)(rng);
        int numberOfChangedEdges = changeDist(rng);
        int insertPct = pctDist(rng);
        int deletePct = 100 - insertPct;
        unsigned int graphSeed = seedDist(rng);
        unsigned int changeSeed = seedDist(rng);

        string dir = baseDir + "/run" + to_string(run);
        string originalPrefix = dir + "/originalGraph/graphCsr";
        string updatedPrefix = dir + "/updatedGraph/updatedGraphCsr";
        string insertPath = dir + "/changedEdges/insert.txt";
        string deletePath = dir + "/changedEdges/delete.txt";
        string expectedDir = dir + "/expected";

        bool ok = true;

        ok = ok && generateGraphCSR(numberOfNodes, numberOfEdges, true,
                                     originalPrefix, numberOfObjectives,
                                     1, objectiveEndRange, graphSeed);

        ok = ok && generateChangedEdges(1, objectiveEndRange, numberOfObjectives,
                                         numberOfNodes, numberOfChangedEdges,
                                         insertPct, deletePct, true, true, true,
                                         false, originalPrefix, insertPath,
                                         deletePath, changeSeed);

        ok = ok && escher_mosp::updateGraphWithESCHER(
                        originalPrefix, updatedPrefix,
                        insertPath, deletePath,
                        /*payloadCapacity=*/65536, /*directed=*/true);

        ok = ok && runDijkstraCSR(originalPrefix, objectiveIndex, 0,
                                   expectedDir + "/distancesOriginal.txt",
                                   expectedDir + "/SSSPTreeOriginal.txt");

        ok = ok && runDijkstraCSR(updatedPrefix, objectiveIndex, 0,
                                   expectedDir + "/distancesUpdated.txt",
                                   expectedDir + "/SSSPTreeUpdated.txt");

        ok = ok && sequentialSOSPUpdate(originalPrefix,
                                         expectedDir + "/distancesOriginal.txt",
                                         expectedDir + "/SSSPTreeOriginal.txt",
                                         insertPath, deletePath,
                                         objectiveIndex, 0,
                                         expectedDir + "/distancesSospUpdate.txt",
                                         expectedDir + "/SSSPTreeSospUpdate.txt");

        if (!ok) {
            cout << "Run " << run << ": ERROR (pipeline failure)\n";
            ++failCount;
            continue;
        }

        bool match = compareDistanceFiles(
            expectedDir + "/distancesUpdated.txt",
            expectedDir + "/distancesSospUpdate.txt"
        );

        if (match) {
            ++passCount;
        } else {
            cout << "Run " << run << ": FAIL (nodes=" << numberOfNodes
                 << " edges=" << numberOfEdges
                 << " objs=" << numberOfObjectives
                 << " objIdx=" << objectiveIndex
                 << " changes=" << numberOfChangedEdges
                 << " ins%=" << insertPct
                 << " graphSeed=" << graphSeed
                 << " changeSeed=" << changeSeed << ")\n";
            ++failCount;
        }
    }

    cout << "\n=== Stress Test: " << passCount << "/" << totalRuns
         << " passed, " << failCount << " failed ===\n";

    return failCount > 0 ? 1 : 0;
}
