# H-SOSP: Dynamic Single-Objective Shortest Path on Hypergraphs

This module implements the project defined in the July 2026 meeting: the
single-source shortest path problem on a weighted, undirected hypergraph in
the fully dynamic setting, on GPU, with **ESCHER** (the CBST core in
`escher/`) as the authoritative dynamic hypergraph store and the **MOSP**
project's parallel SOSP-update framework as the update engine.

## Problem model (from the meeting notes)

- Hyperedge `h_i` carries one positive weight `w_i >= 1` (the paper's
  ω → R>0; integer weights here). Only the virtual source and target
  hyperedges have weight 0. Zero or negative weights are rejected:
  two adjacent zero-weight hyperedges cut off from the source would keep a
  stale finite distance.
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
 generator / file loader          batches (he/vertex, del%, placement)
        │                                   │
        ▼                                   ▼
 DynamicHypergraph::beginBatch: h2v insert (decides new ids)  (t_escher_ms)
        │ host incidence model (HostHypergraph): validates the ops,
        │ yields the incidence changes (v, h, ±1) + touched rows   (t_delta_ms)
        ▼
 GPU unification (hsospDelta.cu): candidates from the vertex
 lists before / after the batch, classified by overlap
 → net line-graph delta in dev.delta                            (t_delta_ms)
        ▼
 DeviceH2H: resident slack-CSR line graph ◄── device sort +
 warp-per-row apply (rebuilt on a tail overflow)             (t_csr_apply_ms)
        ▼
 DynamicHypergraph::finishBatch: remaining h2v / v2h / h2h
 CBST maintenance from the sorted net delta                   (t_escher_ms)
        ▼
 hsospUpdate: invalidate subtrees, pull, push              (t_sosp_update_ms)
 hsospRecompute: static baseline from blank                (t_static_ms)
```

`hsosp::applyBatch` runs the first four stages. The line graph itself is
not kept on the host: the host holds the incidence (the sorted vertex list
of every hyperedge, the hyperedge list of every vertex, weights, liveness,
free ids) and the GPU holds a mirror of it (`DeviceIncidence`) from which
it derives the net line-graph delta. The rule is order-free and exact: a
pair (a, b) can only change if a shared vertex was added to or removed
from one of them, so every changed pair is a candidate of some changed
incidence (v, h) (the other hyperedge is in v's list before or after the
batch), and each candidate is inserted if a and b overlap only after the
batch, deleted if only before.

- `hypergraph/include/HostHypergraph.hpp` — host incidence model and the
  batch rules (pure C++, unit-tested off-GPU via `tests/local/`, which also
  runs a host copy of the delta rule against the line-graph difference).
- `hypergraph/include/DynamicHypergraph.hpp` — ESCHER routing. Hyperedge
  ids adopt the keys returned by the h2v `insertCBST` best-fit mapping
  (the ESCHER paper's id-reassignment scheme); the h2h CBST keeps its own
  key space with a host-side translation.
- `hsosp/include/hsosp.cuh` — device line graph + node-weighted SOSP
  kernels (adapted from `mosp/src/parallelSOSPUpdate.cu`, originals
  untouched).
- Update (exact): the pre-batch shortest-path subtree of every deleted
  tree edge and of every new, recreated or dead node is invalidated
  (pointer jumping), invalidated nodes pull their best neighbour, inserted
  pairs are relaxed, and decreases are pushed to convergence with a packed
  (distance, parent) atomicMin. No stale distance survives, so there is no
  count-to-infinity and no iteration cap; parent ties go to the lowest id.
  An update that exceeds its budget (`--work-budget` x adjacency entries
  relaxations, default 1, or `--maxiter` push iterations, default 4,096)
  falls back to the recompute inside the timed update (CSV: `fallback`,
  `fallback_iters`).

## Binaries

| Binary | Purpose |
|---|---|
| `bin/hsospBench`  | experiment matrix -> CSV (`--suite smoke|full`) |
| `bin/hsospStress` | randomized full-pipeline check vs host Dijkstra |
| `bin/test_h2h_construction` | line graph + device CSR + incidence mirror == oracle |
| `bin/test_h2h_delta` | device CSR (GPU-derived delta) and incidence mirror == rebuild after every batch |
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
correctness verdict. Every batch is compared with the static recompute
(`mismatch_static`; same kernels, so not an independent check) and, when the
hypergraph has at most `--verify-max` hyperedge ids (default 5,000,000), with
an independent oracle: Dijkstra on the line graph rebuilt from the incidence
lists, plus a check of the shortest-path tree (`oracle` = `host`,
`mismatch_oracle`, `parent_errors`). `verified` is 1 only when that oracle
ran and agreed; `correct` requires both checks that ran to pass. `iters` is
the update's push iterations, `fallback_iters` the rounds of a fallback
recompute, `invalidated` the invalidated nodes, `update_work` the edge
relaxations, `static_iters` the baseline's rounds. `t_csr_apply_ms` includes
a rebuild after a tail overflow (`overflow_rebuilds`). `dev_mem_mb` is the
device-wide used memory (`cudaMemGetInfo`, other processes included);
`escher_mb` and `graph_mb` are this process's structures.

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
