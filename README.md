# escher-mosp

Multi-Objective Shortest Path (MOSP) on GPU, with every edge insertion and
deletion routed through the **ESCHER** dynamic hypergraph data structure
(Complete Binary Search Trees on GPU). This repository unifies two sibling
research projects — the ESCHER CBST core and the MOSP CUDA algorithm — into
a single build tree so that MOSP now uses our own data structure as its
authoritative dynamic graph store.

## NEW: H-SOSP — dynamic shortest paths on hypergraphs

The `hypergraph/` and `hsosp/` modules implement the project defined in the
July 2026 meeting: single-objective shortest paths on a weighted dynamic
hypergraph via its h2h (line graph) representation, with ESCHER as the
authoritative store and the parallel SOSP-update framework as the engine.
See **[docs/HSOSP.md](docs/HSOSP.md)** for the model, experiments, and the
one-command cluster pipeline:

```bash
CUDA_ARCH=sm_80 ./scripts/run_experiments.sh smoke   # sanity pass
CUDA_ARCH=sm_80 ./scripts/run_experiments.sh full    # paper run: CSV + all figures
```

## Layout

```
escher-mosp/
├── escher/          libescher_core: the CBST data structure (motif-free subset of ESCHER-GPU)
│   ├── include/     structure.hpp, binning.hpp, flatten.hpp, escher_errors.hpp, printUtils.hpp
│   ├── kernel/      build_tree, find, payload, insert_reuse, delete_avail, unfill
│   ├── structure/   operations.cu (construct / insert / erase / fill / unfill)
│   └── utils/       flatten.cpp, binning.cpp, printUtils.cpp
│
├── graph/           DynamicGraph adapter: the simple API MOSP calls into
│   ├── include/     DynamicGraph.hpp, GraphSnapshot.hpp, updateGraphWithESCHER.hpp
│   └── src/         DynamicGraph.cpp, snapshot.cu, updateGraphWithESCHER.cpp
│
├── mosp/            MOSP-CUDA sources, re-targeted to call through the adapter
│   ├── headers/     (unchanged .cuh files)
│   └── src/         main, stressTest, parallelStressTest, generateTestCases
│                    and the Dijkstra / sequentialSOSPUpdate / parallelSOSPUpdate
│                    kernels (unchanged bodies; only the graph-update call site
│                    was swapped)
│
├── tests/unit/      Smoke + round-trip + equivalence tests
├── tests/run_all.sh Test driver (cluster-side)
├── scripts/         sync_to_cluster.sh, build_on_cluster.sh
├── docs/            ARCHITECTURE.md, MIGRATION_NOTES.md (hand-authored)
├── Makefile         Unified build
└── Doxyfile         `make docs` → docs/html/
```

## Architecture in one picture

```
  [ main.cu / stressTest / generateTestCases ]
                │ (call once per update batch)
                ▼
  escher_mosp::updateGraphWithESCHER(originalPrefix, updatedPrefix, ...)
                │
                │    readCSR        (unchanged MOSP reader)
                │    DynamicGraph::loadFromCSR
                │    DynamicGraph::deleteEdges   ──► CBSTOperations::erase
                │                                    unfillCBST
                │    DynamicGraph::insertEdges   ──► CBSTOperations::insert
                │                                    CBSTOperations::fill
                │    DynamicGraph::dumpToCSR     ──► writes updatedPrefix{RowPtr,ColInd,Values}.txt
                ▼
  [ Dijkstra / sequentialSOSPUpdate / parallelSOSPUpdate / parallelCombinedGraph ]
            unchanged — consume the updated CSR files as before
```

- The **ESCHER CBST** is the authoritative store between update batches. Every
  insert and delete goes through `CBSTOperations::insert` / `::erase` / `::fill`
  or `unfillCBST`.
- The **host shadow** of the adjacency topology is kept in lock-step so
  `dumpToCSR` and `snapshot(objective)` run in linear time without reading
  the CBST back.
- The **MOSP CUDA kernels** (collectCandidates / updateDistances / bfs /
  markUnreachable / Dijkstra relaxation) are untouched. They still read a
  device CSR that MOSP allocates from the updated files on disk. A
  `GraphSnapshot` API is also provided for a future rewiring that skips the
  disk round-trip entirely.
- **Motif counting code from ESCHER is dropped** — we do not need 30-bin
  triangle motifs here. Just the CBST core.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details and
[docs/MIGRATION_NOTES.md](docs/MIGRATION_NOTES.md) for every logic fix
applied to the upstream ESCHER code.

## Requirements

- CUDA Toolkit 12.5+ (Thrust is bundled)
- NVIDIA GPU with compute capability ≥ 7.0 (default `-arch=sm_70`)
- C++17 host compiler (gcc 9+ or clang 10+)
- `doxygen` (optional, for `make docs`)

On macOS you can still preprocess every translation unit with
`make syntax-check`; the final link and run happen on the Linux cluster.

## Build

```bash
make                     # libescher_core.a + main + stressTest + parallelStressTest + unit tests
make run                 # build and run main
make tests               # build unit test binaries
make stressTest          # build just the sequential stress test
make parallelStressTest  # build just the CUDA parallel stress test
make clean               # remove build/ and bin/
make docs                # Doxygen HTML in docs/html/
make syntax-check        # nvcc -E on every TU (no link; works on macOS)

# Override the CUDA architecture (for newer GPUs on the cluster):
make CUDA_ARCH=sm_80
```

## Cluster workflow (mill.mst.edu)

```bash
# On MacBook:
./scripts/sync_to_cluster.sh                # rsync source tree

# On the cluster:
ssh sskg8@mill.mst.edu
cd ~/escher-mosp
./scripts/build_on_cluster.sh                # module load cuda + make
./tests/run_all.sh                           # runs unit tests, main, both stress tests
```

## Test strategy

Three unit tests and the original MOSP stress/test-case harness:

| Binary                                        | What it proves                                                                                  |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------|
| `bin/test_cbst_smoke`                         | libescher_core constructs/insert/erase work for N ∈ {1,2,3,7,8,9,15,16,17,100,1000}              |
| `bin/test_dynamicgraph_roundtrip`             | `loadFromCSR` → `dumpToCSR` is byte-identical for a hand-built 5-vertex / 2-objective graph      |
| `bin/test_snapshot_matches_updateCSR`         | **ESCHER-backed** `updateGraphWithESCHER` produces byte-identical CSR to the legacy `updateGraphCSR` path |
| `bin/main`                                    | Runs the full MOSP pipeline. `tests/testCase1..10/expected/` are generated and verified end-to-end |
| `bin/stressTest`                              | 100 random configurations: sequentialSOSPUpdate (on ESCHER-backed graph) must match Dijkstra    |
| `bin/parallelStressTest`                      | 100 random configurations: parallelSOSPUpdate must match Dijkstra                                |

The equivalence test (`test_snapshot_matches_updateCSR`) is the single most
important correctness check: if it passes, every downstream MOSP consumer
sees exactly the same CSR whether the update went through the legacy in-memory
path or through ESCHER.

## Key decisions

1. **Snapshot-to-CSR at the update boundary**, not per-kernel. MOSP's kernels
   stay untouched; we replace only the graph-update step. Rewriting kernels
   to walk CBST pointers is possible but not necessary to claim the integration.
2. **Regular graph as a hypergraph of 2-vertex hyperedges**. `edgesCBST` has
   one record per directed edge with fixed payload `[src, dst, w_0 … w_{K-1}]`;
   `outAdjCBST` and `inAdjCBST` hold variable-length edge-id lists keyed by
   source / destination vertex.
3. **Host shadow for linear-time snapshot**. Every ESCHER operation also
   mirrors into a `std::vector<std::vector<int>>` shadow so dumps don't need
   to read the CBST back. The claim "MOSP uses ESCHER" holds because every
   write still goes through ESCHER; the read path is a performance choice.
4. **Motif counting dropped**. ESCHER's `HMotifCount*`, `type1/2/3`,
   `coarseTriangle`, and `motif_utils.cuh` are not part of the unified tree.
5. **Default arch `sm_70`** (Volta/V100), overridable. Matches MOSP-CUDA's
   original default and works with `--extended-lambda`.

## Where each public symbol lives

| Public API                              | Header                                 |
|------------------------------------------|----------------------------------------|
| `escher_mosp::DynamicGraph`              | `graph/include/DynamicGraph.hpp`       |
| `escher_mosp::GraphSnapshot`             | `graph/include/GraphSnapshot.hpp`      |
| `escher_mosp::updateGraphWithESCHER`     | `graph/include/updateGraphWithESCHER.hpp` |
| `CBSTOperations` (construct/insert/erase/fill) | `escher/include/structure.hpp`   |
| `unfillCBST` free function               | `escher/include/structure.hpp`         |
| `flatten2DVector`                        | `escher/include/flatten.hpp`           |
| `escher::EscherError`, `ESCHER_CHECK_CUDA` | `escher/include/escher_errors.hpp`   |
