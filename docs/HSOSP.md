# H-SOSP: Dynamic Single-Objective Shortest Path on Hypergraphs

This module implements the project defined in the July 2026 meeting: the
single-source shortest path problem on a weighted, undirected hypergraph in
the fully dynamic setting, on GPU, with **ESCHER** (the CBST core in
`escher/`) as the authoritative dynamic hypergraph store and the **MOSP**
project's parallel SOSP-update framework as the update engine.

## Problem model (from the meeting notes)

- Hyperedge `h_i` carries one non-negative weight `w_i`.
- The hypergraph is converted to the **h2h structure** (line graph): two
  hyperedges are adjacent iff they share at least one vertex.
- Stepping into `h_j` costs `w_j`; a path's cost is the sum of the weights
  of the hyperedges entered. Consequently every in-edge of line-graph node
  `j` has weight `w_j`, so the device graph stores one symmetric adjacency
  plus a per-node weight array.
- Step 1 of the notes: virtual hyperedge `h_0 = {s}` (weight 0) for the
  source and `h_{n+1} = {t}` for the target. The SOSP tree is computed from
  `h_0`; `dist(t) = dist(h_{n+1})`.
- Dynamics: hyperedge insertion/deletion (vertical ops) and incident-vertex
  insertion/deletion (horizontal ops). An h2h edge dies exactly when the
  last common vertex of the two hyperedges disappears.

## Architecture

```
 generator (pool model)          batches (he/vertex, del%, placement)
        │                                   │
        ▼                                   ▼
 DynamicHypergraph  ──ESCHER──►  h2v / v2h / h2h CBSTs   (t_escher_ms)
        │ host shadow (HostHypergraph)
        ├── H2HDelta (net edge ins/del, new/dead nodes, seeds)  (t_delta_ms)
        ▼
 DeviceH2H: resident slack-CSR line graph  ◄── applyDeltaToDevice (t_csr_apply_ms)
        ▼
 hsospUpdate: seeded candidate/affected propagation        (t_sosp_update_ms)
 hsospRecompute: static baseline from blank                (t_static_ms)
```

- `hypergraph/include/HostHypergraph.hpp` — host shadow + delta rules
  (pure C++, unit-tested off-GPU via `tests/local/`).
- `hypergraph/include/DynamicHypergraph.hpp` — ESCHER routing. Hyperedge
  ids adopt the keys returned by the h2v `insertCBST` best-fit mapping
  (the ESCHER paper's id-reassignment scheme); the h2h CBST keeps its own
  key space with a host-side translation.
- `hsosp/include/hsosp.cuh` — device line graph + node-weighted SOSP
  kernels (adapted from `mosp/src/parallelSOSPUpdate.cu`, originals
  untouched).
- Convergence: positive weights make any fixed point of the relaxation
  correct, so no reachability BFS is needed on the happy path. If a batch
  disconnects a region, its stale distances keep increasing until the
  iteration cap (`--maxiter`, default 512) triggers a full recompute
  fallback (counted in the CSV as `fallback`).

## Binaries

| Binary | Purpose |
|---|---|
| `bin/hsospBench`  | experiment matrix -> CSV (`--suite smoke|full`) |
| `bin/hsospStress` | randomized full-pipeline check vs host Dijkstra |
| `bin/test_h2h_construction` | shadow + device CSR == brute-force line graph |
| `bin/test_h2h_delta` | incremental maintenance == rebuild after every batch |
| `bin/test_hsosp_matches_dijkstra` | update + recompute == Dijkstra (incl. disconnects) |

## Experiments

Synthetic hypergraphs only (per project decision). The clustered-pool
generator controls the average h2h degree through the vertex count
(`n ~ m * E[c^2] / degTarget`); pools give locality and connectivity.
Named configurations (full suite): HG-S (1M hyperedges), HG-M (5M),
HG-L (10M), HG-XL (16M), HG-C (2M, cardinality up to 64).

| Exp | Sweep | Figure(s) |
|---|---|---|
| E1 | batch size 25K-200K, hyperedge + vertex batches | `time_vs_DeltaE_*.pdf`, `<ds>_base_vs_DeltaE.pdf` |
| E2 | deletion percentage 20-80% | `<ds>-del-vary.pdf` |
| E3 | hypergraph size 1M-16M | `time_vs_size.pdf`, `memory_plot.pdf` |
| E4 | max cardinality 8-128 | `time_vs_cardinality.pdf` |
| E5 | average h2h degree 8-64 | `time_vs_density.pdf` |
| E6 | (derived) phase breakdown | `stacked_percentage.pdf` |
| E7 | placement random/targeted/near/far | `placement_time.pdf` |
| E8 | (derived) speedup vs static recompute | `Speedup.pdf` |
| E9 | (derived) memory | `memory_plot.pdf` |

Every measured batch appends one CSV row (flushed immediately); the row
carries all phase timings, iteration counts, seed values, memory, and the
correctness verdict (compared against the static recompute on every batch,
and against host Dijkstra whenever `m <= --verify-max`).

## Running on the cluster

```bash
# MacBook:
./scripts/sync_to_cluster.sh

# Cluster (one command; smoke first, then the real run):
ssh sskg8@mill.mst.edu
cd ~/escher-mosp
CUDA_ARCH=sm_80 ./scripts/run_experiments.sh smoke
CUDA_ARCH=sm_80 ./scripts/run_experiments.sh full
```

`run_experiments.sh` builds, runs all unit tests + the stress harness
(hard gate: the benchmark never runs on a broken build), executes the
suite into `results/<suite>_<timestamp>/results.csv`, and renders all
figures into `results/<suite>_<timestamp>/figures/`. The log of the whole
run is kept next to the CSV.

## Fixes applied to the existing code (see git diff for detail)

1. `escher/structure/operations.cu`: the reusable scratch buffers
   (`d_insertKeys` / `d_insertPayload` / ...) were sized once at
   construction and silently overflowed for batches larger than the
   initial record count; they now grow on demand.
2. `constructCBST` never initialized node occupancy (the "fixup pass"
   mentioned in build_tree.cu did not exist), so the first `fillCBST` on a
   row overwrote construct-time payload. Callers can now pass true per-row
   counts; `DynamicGraph` and `DynamicHypergraph` do.
3. `insertCBST`'s surplus reconstruction sorted up to `numRecords`
   (key, offset) pairs on the host per batch; the sort now runs on the
   device via Thrust.
4. `DynamicGraph::insertEdges` silently leaked a parallel edge record when
   re-inserting an existing (src,dst); it now fails loudly (upsert is
   handled one level up, which pre-deletes).
5. `Makefile`: `test_snapshot_matches_updateCSR` did not link
   (missing SOSP objects for `generateTestCases.cu`); fixed.
6. `escher/utils/binning.cpp`: missing `<cstddef>` include broke the build
   on newer host compilers.
