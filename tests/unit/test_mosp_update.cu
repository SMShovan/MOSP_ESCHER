/**
 * @file test_mosp_update.cu
 * @brief Deterministic MOSP-half regressions: sequentialSOSPUpdate and
 *        parallelSOSPUpdate must reproduce the Dijkstra distances and the
 *        (lowest-id) Dijkstra tree of the updated graph.
 *
 *  - disconnect: 0->1 (1), 0->3 (50), 1->2 (1), 2->1 (1), 3->1 (50); delete
 *    0->1. Vertices 1 and 2 stay reachable only through 3 (d = 100, 101);
 *    the original update counted to infinity around the 1<->2 cycle, hit its
 *    iteration cap and returned d(1) = 7, d(2) = 6.
 *  - delete-all: every edge of a small graph is deleted; only the source
 *    stays reachable (the original pipeline could not read the empty CSR).
 *  - ties: an insertion creates a second shortest path to a vertex; its
 *    parent must become the lower id (canonical tree).
 */

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "dijkstra.cuh"
#include "parallelSOSPUpdate.cuh"
#include "sequentialSOSPUpdate.cuh"

namespace {

struct Edge {
    int u, v, w;
};

void writeCsr(const std::string& prefix, int n, std::vector<Edge> edges) {
    std::vector<std::vector<Edge>> out(n);
    for (const Edge& e : edges) out[e.u].push_back(e);
    std::ofstream rp(prefix + "RowPtr.txt"), ci(prefix + "ColInd.txt"),
        va(prefix + "Values.txt");
    int k = 0;
    rp << 0;
    for (int u = 0; u < n; ++u) {
        for (const Edge& e : out[u]) {
            ci << e.v << "\n";
            va << e.w << "\n";
            ++k;
        }
        rp << " " << k;
    }
    rp << "\n";
}

void writeChanges(const std::string& path, const std::vector<Edge>& edges,
                  bool withWeights) {
    std::ofstream f(path);
    for (const Edge& e : edges) {
        f << e.u << " " << e.v;
        if (withWeights) f << " " << e.w;
        f << "\n";
    }
}

std::string slurp(const std::string& path) {
    std::ifstream in(path);
    std::string s, all;
    while (std::getline(in, s)) all += s + "\n";
    return all;
}

/** Runs both updates for one case; returns the number of mismatches. */
int runCase(const char* name, int n, const std::vector<Edge>& before,
            const std::vector<Edge>& inserts,
            const std::vector<Edge>& deletes) {
    const std::string dir = std::string("mosp_update/") + name + "/";
    std::filesystem::create_directories(dir);
    std::vector<Edge> after;
    for (const Edge& e : before) {
        bool gone = false;
        for (const Edge& d : deletes) gone |= (d.u == e.u && d.v == e.v);
        if (!gone) after.push_back(e);
    }
    after.insert(after.end(), inserts.begin(), inserts.end());
    writeCsr(dir + "g", n, before);
    writeCsr(dir + "u", n, after);
    writeChanges(dir + "insert.txt", inserts, true);
    writeChanges(dir + "delete.txt", deletes, false);

    int bad = 0;
    bool ok = runDijkstraCSR(dir + "g", 0, 0, dir + "dOrig.txt",
                             dir + "tOrig.txt") &&
              runDijkstraCSR(dir + "u", 0, 0, dir + "dTruth.txt",
                             dir + "tTruth.txt");
    if (!ok) {
        std::printf("FAIL  %s: Dijkstra could not run\n", name);
        return 1;
    }
    const std::string truthD = slurp(dir + "dTruth.txt");
    const std::string truthT = slurp(dir + "tTruth.txt");
    struct Update {
        const char* name;
        bool (*fn)(const std::string&, const std::string&,
                   const std::string&, const std::string&,
                   const std::string&, int, int, const std::string&,
                   const std::string&);
    };
    const Update updates[] = {{"sequential", sequentialSOSPUpdate},
                              {"parallel", parallelSOSPUpdate}};
    for (const Update& up : updates) {
        const std::string d = dir + up.name + "D.txt";
        const std::string t = dir + up.name + "T.txt";
        bool ran = up.fn(dir + "g", dir + "dOrig.txt", dir + "tOrig.txt",
                         dir + "insert.txt", dir + "delete.txt", 0, 0, d, t);
        const bool match = ran && slurp(d) == truthD && slurp(t) == truthT;
        std::printf("%s  %s %s\n", match ? "PASS" : "FAIL", name, up.name);
        if (!match) {
            std::printf("  expected distances:\n%s  got:\n%s", truthD.c_str(),
                        slurp(d).c_str());
            ++bad;
        }
    }
    return bad;
}

} // namespace

int main() {
    int failures = 0;
    failures += runCase("disconnect", 4,
                        {{0, 1, 1}, {0, 3, 50}, {1, 2, 1}, {2, 1, 1},
                         {3, 1, 50}},
                        {}, {{0, 1, 0}});
    failures += runCase("delete-all", 5,
                        {{0, 1, 3}, {1, 2, 4}, {0, 3, 2}, {3, 4, 7}}, {},
                        {{0, 1, 0}, {1, 2, 0}, {0, 3, 0}, {3, 4, 0}});
    // 0->1 (5), 0->2 (1), 2->3 (1), 1->4 (2), 3->4 (2): d(4) = 4 via 3.
    // Inserting 2->1 (1) gives d(1) = 2 and a second tight path to 4 (via
    // 1, also 4); the parent of 4 must become the lower id, 1.
    failures += runCase("ties", 5,
                        {{0, 1, 5}, {0, 2, 1}, {2, 3, 1}, {1, 4, 2},
                         {3, 4, 2}},
                        {{2, 1, 1}}, {});
    std::printf("test_mosp_update: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
