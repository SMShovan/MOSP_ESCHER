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
CUDA_ARCH=sm_80 ./scripts/run_experiments.sh full    # synthetic suite: CSV + all figures
# A real hypergraph (one hyperedge per line) with the paper's preprocessing
# and batch model, every batch checked by the independent oracle:
./bin/hsospBench --hg coauth.hg --kind hyperedge --batch 50000 --batches 3 \
    --verify all --out results/dblp.csv
```

## Correctness and performance changes

[CHANGES.md](CHANGES.md) lists every fix and optimization made on top of
the original code (base: tag `baseline-2026-09`), each with its regression
test and before/after numbers; [results/README.md](results/README.md) has
the before/after tables on coauth-DBLP and coauth-MAG-Geology. In short:

- The ESCHER CBST core had the defects fixed earlier in ESCHER-GPU (a
  32-bit rank overflow above 65,535 records, stale subtree counts, missing
  row terminators, erased keys, metadata lost on rebuild, truncated
  best-fit inserts, a wrong tail occupancy after unfill); at the paper's
  scale the original crashed or corrupted the h2v / v2h / h2h contents.
- The H-SOSP update counted to infinity after deletions, so every
  realistic batch ran its 512-iteration cap and then a full recompute. It
  is now exact and incremental (subtree invalidation, pull, push with a
  packed atomicMin; a budget falls back to the recompute, timed).
- The MOSP half was an unfixed copy of MOSP-CUDA with the same
  count-to-infinity defect (wrong, too small distances); the fixed update
  was ported.
- Unification (the line-graph delta) runs on the GPU from the incidence
  changes; the CSR apply, the pull kernels and the ESCHER operations were
  rewritten for throughput. On DBLP the dynamic time per 50K hyperedge
  batch went from 13.4 s to 0.17 s (27-114x across the measured
  configurations).
- Tests fail when results are wrong: an independent oracle (line graph
  rebuilt from the vertex lists, Dijkstra, canonical tree), ESCHER content
  read-back against a host model, tests above 65,535 records. `make test`
  runs everything.

### How this implementation differs from the paper

- **Unification input.** The paper derives the line-graph changes on the
  GPU "using the same v2h and h2v lookups" as ESCHER. Here the GPU keeps a
  separate slack-row mirror of the incidence (`DeviceIncidence`) and
  derives the net delta from it; the ESCHER h2v / v2h / h2h CBSTs are
  maintained in every batch (and checked by the tests) but nothing on the
  shortest-path path reads them, as in the original code.
- **Update algorithm.** The pull-based update with an iteration cap
  described in the paper does not converge after deletions (stale
  distances rise by at least 1 per iteration). The update here first
  invalidates the pre-batch subtrees of deleted tree edges and of new or
  dead nodes, then pulls and pushes; it is exact without a cap. The budget
  fallback to the recompute is kept and counted in the update time.
- **Ties.** Parents are canonical (lowest id among the tight neighbours)
  in the update, the recompute and the oracle.
- **Weights.** Integer hyperedge weights in [1, 2^28] (other weights are
  rejected; the paper's weights are positive reals, and the cap keeps
  every path cost far from 64-bit overflow). Real datasets get U[1,100]
  weights.
- **Baseline.** The static baseline is the GPU recompute with the same
  kernels (as in the paper). Those kernels became about 4x faster (P3 in
  CHANGES.md), which lowers the reported speedups.
- **Scale.** The CBST payloads are `int`-indexed, and `constructFromRows`
  caps each at 2.0 x 10^9 values, so Orkut, AMiner and MAG (about 2.1-2.9
  x 10^9 h2h entries, estimated from the paper's Table I; not loaded
  here) would be rejected, independently of GPU memory. Only DBLP and
  Geology were measured.
- **Results.** On an RTX A5000 the dynamic time per batch (maintenance +
  unification + CSR apply + update, the paper's metric) is below the
  recompute only for DBLP vertex batches of 25K (1.28x); elsewhere it is
  recompute/dynamic = 0.08-0.85, against 1.3-12.1x reported for DBLP
  in the paper. The update stage alone is 1.4-25x faster than the
  recompute; ESCHER maintenance and unification dominate. See
  results/README.md.

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
│   ├── headers/     MOSP-CUDA headers, plus deviceArray.cuh and sospUpdateGpu.cuh
│   │                of the ported SOSP update
│   └── src/         main, stressTest, parallelStressTest, generateTestCases
│                    (graph-update call site swapped to the adapter); Dijkstra,
│                    sequentialSOSPUpdate and parallelSOSPUpdate with the fixed
│                    update ported from MOSP-CUDA (sospUpdateGpu.cu) and a
│                    lowest-id tie-break
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
            consume the updated CSR files as before; parallelSOSPUpdate
            runs the ported sospUpdateGpu engine
```

- The **ESCHER CBST** is the authoritative store between update batches. Every
  insert and delete goes through `CBSTOperations::insert` / `::erase` / `::fill`
  or `unfillCBST`.
- The **host shadow** of the adjacency topology is kept in lock-step so
  `dumpToCSR` and `snapshot(objective)` run in linear time without reading
  the CBST back.
- The **MOSP SOSP update** is the fixed engine ported from MOSP-CUDA
  (`sospUpdateGpu`): it finds the roots from the change list, invalidates
  their subtrees by pointer jumping, runs a pull pass, then a monotone
  near-far push with a packed 64-bit atomicMin (see CHANGES.md; the
  original collect / update / BFS / mark-unreachable kernels counted to
  infinity after deletions). It still reads a device CSR that MOSP
  allocates from the updated files on disk. A `GraphSnapshot` API is also
  provided for a future rewiring that skips the disk round-trip entirely.
- **Motif counting code from ESCHER is dropped** — we do not need 30-bin
  triangle motifs here. Just the CBST core.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details and
[docs/MIGRATION_NOTES.md](docs/MIGRATION_NOTES.md) for every logic fix
applied to the upstream ESCHER code.

## Requirements

- CUDA Toolkit 12.5+ (Thrust and CUB are bundled); tested with 12.9 and
  13.1
- NVIDIA GPU with compute capability ≥ 7.0 (default `-arch=sm_86`; set
  `CUDA_ARCH` for other GPUs, e.g. `sm_80` for A100; CUDA 13 no longer
  targets `sm_70`)
- C++17 host compiler with OpenMP (gcc 9+)
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
make test                # build everything and run tests/run_tests.sh

# Override the CUDA architecture (default sm_86), the compiler or the
# optimization level (default -O3, host and device):
make CUDA_ARCH=sm_80
make NVCC=/usr/local/cuda/bin/nvcc OPT=-O2
```

Objects depend on the headers they include and on the compiler flags in
use, so changing a header or `CUDA_ARCH` rebuilds what is affected.

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

`make test` (or `./tests/run_all.sh`) builds everything and runs
`tests/run_tests.sh`, which prints PASS / FAIL per case and exits non-zero
on any failure:

| Case | What it checks |
|---|---|
| `test_cbst_smoke`, `test_cbst_ops <scenario>` | every CBST operation mirrored on a host model; contents, reachability, tail metadata and subtree counts after each step (up to 2^20 records) |
| `test_dynamicgraph_roundtrip` | `DynamicGraph` delete / insert rounds, ESCHER contents |
| `test_snapshot_matches_updateCSR` | ESCHER-backed `updateGraphWithESCHER` == the legacy `updateGraphCSR` path |
| `test_h2h_construction`, `test_h2h_delta` | device CSR and incidence mirror == the line graph rebuilt from the vertex lists, after every batch (overflow paths included) |
| `test_hsosp_matches_dijkstra`, `test_hsosp_scale` | update, recompute and fallback == Dijkstra on the rebuilt line graph, canonical parents; ESCHER contents at 75K hyperedges |
| `local_tests` | the host core without a GPU (tests/local) |
| `hsospStress`, `hsospBench` smoke and real-file cases | randomized pipeline runs, every batch against the independent oracle |
| `test_mosp_update`, `main`, `stressTest`, `parallelStressTest` | MOSP SOSP updates == Dijkstra (distances and trees), disconnecting deletions |

## Key decisions

1. **Snapshot-to-CSR at the update boundary**, not per-kernel. The ESCHER
   integration replaces only the graph-update step; MOSP's kernels read the
   CSR as before. Rewriting kernels to walk CBST pointers is possible but
   not necessary to claim the integration. (The SOSP update itself was
   later replaced by the fixed engine ported from MOSP-CUDA, independently
   of ESCHER; see CHANGES.md.)
2. **Regular graph as a hypergraph of 2-vertex hyperedges**. `edgesCBST` has
   one record per directed edge with fixed payload
   `[src+1, dst+1, w_0+1 … w_{K-1}+1]` (every field is shifted by one
   because 0 ends a CBST row); edge ids are the keys ESCHER's insert
   assigns. `outAdjCBST` and `inAdjCBST` hold variable-length edge-id lists
   keyed by source / destination vertex.
3. **Host shadow for linear-time snapshot**. Every ESCHER operation also
   mirrors into a `std::vector<std::vector<int>>` shadow so dumps don't need
   to read the CBST back. The claim "MOSP uses ESCHER" holds because every
   write still goes through ESCHER; the read path is a performance choice.
4. **Motif counting dropped**. ESCHER's `HMotifCount*`, `type1/2/3`,
   `coarseTriangle`, and `motif_utils.cuh` are not part of the unified tree.
5. **Default arch `sm_86`** (Ampere RTX A5000 / A40, the development
   machine), overridable with `CUDA_ARCH`.

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
