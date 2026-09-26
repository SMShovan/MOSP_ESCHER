# Changes on `fix/correctness-perf`

Base: `baseline-2026-09` (the original `main`, 80cee62). Every commit builds
and passes `make test`. Hardware for all measurements: RTX A5000 (sm_86,
24 GB), 28-core / 56-thread host, CUDA 13.1.

## Build and tests

- **Build** (`Makefile`): `CUDA_ARCH` defaults to `sm_86` (it was `sm_70`,
  which CUDA 13 cannot target); host and device code at `-O3` (`OPT`),
  `-lineinfo`; `nvcc -MMD -MP` header dependencies and a flags stamp
  (`build/.flags`), so editing a header or changing `CUDA_ARCH` rebuilds what
  it affects (the original had no header dependencies); `NVCC` falls back to
  `/usr/local/cuda/bin/nvcc`; host OpenMP for the oracles.
- **`make test`** (`tests/run_tests.sh`) runs every unit test, the H-SOSP
  stress harness and the MOSP harnesses in a temporary directory, prints
  PASS/FAIL per case and exits non-zero on any failure. Known defects were
  listed as XFAIL (a case that starts passing while listed is itself a
  failure) and removed from the list by the commit that fixes them. The
  MOSP stress tests take `[seed] [runs]` and run with fixed seeds.

## Test oracles

- **CBST content checker** (`escher/structure/integrity.cu`): copies a tree
  to the host and compares every row, found by a BST search from the root
  and read through its overflow chain, with a host model; also checks that
  every live node is reachable, the tail-segment metadata that fill and
  unfill rely on, non-overlapping payload segments and the `subtreeAvail`
  sums.
- **`test_cbst_ops`**: every CBST operation mirrored on a host model and
  checked after each step. One scenario per defect of the original code
  (sizes up to 2^20, reuse, terminators, erase, surplus rebuild, best fit,
  chained unfill) plus random operation sequences at 300 / 5,000 / 70,000
  records with rows of up to 3,000 values (thread, warp and block kernels).
- **Independent H-SOSP oracle** (`hypergraph/src/HypergraphOracle.cpp`): the
  line graph is rebuilt from the incidence lists alone (not from the
  maintained h2h lists, the CBSTs or the device CSR), distances come from a
  binary-heap Dijkstra on it, and the shortest-path tree is checked (every
  reachable node's parent is alive, adjacent and tight). All H-SOSP tests and
  `hsospStress` use it; device CSR rows are compared as multisets (the
  original compared `std::set`s, which hid duplicate entries).
- **ESCHER contents in the H-SOSP pipeline**:
  `DynamicHypergraph::checkEscher` compares the h2v, v2h and h2h CBSTs with
  the host model and the rebuilt line graph (`hsospStress --check-escher`).
- **`test_hsosp_scale`**: 75,000 hyperedges / 80,000 vertices (every CBST
  above 65,535 records), seven consecutive batches (insert-only, delete-only,
  mixed, vertex, delete-heavy, insert-heavy); device rows, distances,
  parents and ESCHER contents checked after each.

The oracles fail when results are wrong. With temporary mutations (not
committed) `make test` failed:

| Mutation | Caught by |
|---|---|
| pull kernel adds the neighbour's weight instead of the node's | `test_hsosp_matches_dijkstra` (125 wrong distances), `hsospStress` (50/50 configurations) |
| pull kernel records a wrong (non-tight) parent, distances unchanged | `hsospStress` (parent check, 15 configurations) |
| CSR apply skips the last deletion of each row | `hsospStress` (device CSR rows) |
| (after E1-E7) unfill ignores the last value of each removal list | `test_cbst_ops` (4 scenarios), `hsospStress --check-escher`, `test_hsosp_scale` |
| (after E8) `DynamicGraph` keeps its own edge ids instead of ESCHER's keys | `test_dynamicgraph_roundtrip` |

On the original code the new cases fail as expected: every `test_cbst_ops`
scenario, `test_hsosp_scale` (abort on a colliding hyperedge id),
`hsospStress --check-escher` (wrong h2v / v2h / h2h rows after one batch),
and `parallelStressTest` seed 6 (a wrong MOSP distance in run 195).

## Correctness: ESCHER core

The CBST code is the motif-free copy of ESCHER-GPU that predates the fixes
made there; these commits port those fixes, adapted to this copy's API
(exceptions instead of `exit`, caller-supplied row occupancy).

- **E1 CBST rank overflow** (`kernel/device_utils.cuh`, used by
  `build_tree.cu`, `operations.cu` `setInitialOccupancy`, `insert_reuse.cu`
  `extractKeysFromPositions`). The in-order rank
  `(2*(pos+1-2^k)+1) * 2^log2(n) / 2^k` was computed in `int` and overflowed
  for n > 65,535: the tree lost its BST order (at n = 65,536 only 49,152 of
  the keys were found; at 2^20 far fewer) and negative ranks indexed the key
  and offset arrays out of bounds. The three copies now share
  `cbstRankOfPosition`, a 64-bit shift. With the original code the H-SOSP
  pipeline aborts on the first batch at 70K hyperedges (colliding
  hyperedge id) and crashes in construct at 1M. Regression:
  `test_cbst_ops scale` (every key of n = 1..1,100 and 65,535..2^20).
- **E2 subtreeAvail recomputation after insert** (`operations.cu`
  `insertCBST`). The bottom-up loop advanced `levelStart` both in the `for`
  header and in its body, so each launch reduced two tree levels and
  parents read children written by the same launch. The counts went
  stale; the next insert located invalid slots and returned key 0 and
  duplicate keys, which `DynamicHypergraph` rejected (`test_h2h_delta`
  aborted on sm_86 in 3 of 3 runs, and `scripts/run_experiments.sh` stopped
  at that gate). Insert and erase now share one helper that reduces one
  level per launch (ESCHER-GPU fixed the same loop). Regression:
  `test_h2h_delta` (no longer XFAIL) and the `subtreeAvail` check after every
  step of `test_cbst_ops`.
- **E3 flatten terminator** (`utils/flatten.cpp`, `kernel/payload.cu`). A
  row whose length is a multiple of 4 was padded to exactly that length,
  so it had no INT_MIN terminator: readers ran into the next row (row
  {1,2,3,4} read back as [1,2,3,4,5,6]), construct gave it
  `tailCapacity = len - 1 < occupancy`, and a fill then wrote its
  back-pointer over the row's last value and `allocateSpace` read
  `values[-1]` (compute-sanitizer: invalid global read, hit by the repo's
  own `test_hsosp_matches_dijkstra` and `hsospStress`). About a quarter of
  the h2v / h2h rows are affected. Rows are now padded to the next multiple
  of 4 above their length (the device rule), and the fill kernels clamp the
  free space of a row at 0. Regression: `test_cbst_ops terminator`.
- **E4 erase overwrote the key** (`kernel/delete_avail.cu`). `applyDeletes`
  set the deleted node's key to -1, so every search passing that node went
  right and the keys of its left subtree became unfindable until the next
  rebuild; later fills, unfills and erases of those keys were silently
  skipped (erasing the root of a 15-key tree made 7 keys unfindable). The
  node now keeps its key and deletion is recorded in `avail[]` only; the
  surplus rebuild drops nodes by `avail[]`. Regression:
  `test_cbst_ops erase`.
- **E5 surplus rebuild reset the row metadata** (`operations.cu`
  `insertCBST`, `kernel/build_tree.cu`). An insert with more items than
  reusable slots appends the rest and rebuilds the tree; the rebuild went
  through (key, offset) pairs and `storeItemsIntoNodes`, which set every
  node's occupancy to 0 and its tail to the first segment, so the next
  fill overwrote each row from its base and chained segments were lost
  (row [20,21,22] + surplus insert + fill 777 read back as [777]). H-SOSP
  takes this path in almost every batch (the study measured h2h rows with
  correct content falling from 78.7% to 32.7% over three DBLP batches). The
  rebuild now keeps the full record of every live node (brought into key
  order through its in-order rank) and gives surplus rows their real
  occupancy and tail. Regression: `test_cbst_ops surplus`.
- **E6 best-fit result discarded** (`operations.cu` `insertCBST`,
  `kernel/insert_reuse.cu`). The capacities of the deleted slots were sorted
  without their slot ids; the matched items were then re-sorted by index
  and the k-th one written into the k-th deleted slot in BST order. Items
  larger than that slot were truncated (the relocation plan the kernels
  wrote was never consumed): a 10-value item stored in a 3-value slot kept
  3 values. The slot ids now travel through the sort and each matched item
  goes to the slot the prefix-max matching chose, which always fits; the
  rest of a reused slot is cleared. Regression: `test_cbst_ops bestfit`
  (and `reuse`, which also needed E4).
- **E7 unfill and the tail occupancy** (`kernel/unfill.cu`). On a chained
  row, unfill subtracted the removals from every segment from `occupancy`,
  which counts the live entries of the tail segment only; the next fill
  wrote at `tailBase + occupancy` over live data (row [1,2] + fill 3,4,5 +
  unfill 1 + fill 6 read back as [2,3,4,6]). `occupancy` is now set from the
  compacted tail segment (one serial helper shared by the thread, warp and
  block kernels). The unused `unfillKernel` and `insertNode` kernels, which
  carried the same defects, were removed. Regression:
  `test_cbst_ops unfill-chain`.

With E1-E7, every `test_cbst_ops` scenario (including the random sequences
at 70,000 records), `test_hsosp_scale` and `hsospStress --check-escher`
pass: after every batch the three CBSTs hold exactly the host model's rows.
- **E8 `DynamicGraph::insertEdges` ignored the insert mapping**
  (`graph/src/DynamicGraph.cpp`). Host edge ids came from a LIFO free list
  while `insertCBST` stored each record under the key its best-fit reuse
  chose; the returned mapping was discarded, so host ids and ESCHER keys
  diverged and a later erase by host id removed another edge's record. Edge
  records also started with the source vertex, and 0 ends a CBST row, so
  edges leaving vertex 0 read back empty. The insert now runs first and its
  keys become the edge ids (no host free list); records are stored as
  [src+1, dst+1, w_0+1, ...] and negative weights are rejected. MOSP
  results were not affected (its CSR files come from the host shadow).
  Regression: `test_dynamicgraph_roundtrip` (delete / insert rounds with
  `DynamicGraph::checkEscher`).
- **E9 `buildDeviceH2H` headroom** (`hsosp/src/hsospDevice.cu`). An
  `entryHeadroom` below 1 made the colInd capacity smaller than the row
  layout and the host fill overran its vector (0.0: segmentation fault,
  0.5: heap corruption). Values below 1 (and NaN) are now rejected with
  `std::invalid_argument`, and the capacity is never below the layout.
  Regression: `test_h2h_construction`; `test_h2h_delta` builds every third
  configuration with headroom 1 so the overflow-rebuild path runs.

## Correctness: MOSP half (`mosp/`)

`mosp/` was a copy of the unfixed MOSP-CUDA (`ac29545`). The fixed SOSP
update engine of MOSP-CUDA was ported (its `sospUpdateGpu`, the invalidation
in `sequentialSOSPUpdate` and the lowest-id tie rule), keeping this
repository's ESCHER adapter (`updateGraphWithESCHER`) at the call sites and
leaving out MOSP-CUDA's instrumentation and in-memory drivers.

- **M1 count-to-infinity in the SOSP update** (`parallelSOSPUpdate.cu`,
  `sequentialSOSPUpdate.cu`). After a deletion, vertices that lost their
  tree path kept stale finite distances that grew around cycles; the loop
  was capped at n iterations and followed by a reachability BFS that only
  reset unreachable vertices, so reachable vertices on a stale cycle kept
  distances that were too small (the 4-vertex example in
  `test_mosp_update`: d(1) = 7, d(2) = 6 instead of 100, 101; about 1 in 800
  random stress configurations). Now: the subtree of every deleted or
  weight-increased tree edge is invalidated (pointer jumping on the GPU, a
  children walk on the host), invalidated vertices and heads of inserted
  edges pull their best (distance, id), and a monotone push (near-far
  worklist with a packed 64-bit atomicMin on the GPU) propagates the
  decreases. There is no iteration cap and no reachability pass. Parent
  ties go to the lowest vertex id everywhere (Dijkstra included), so the
  updated trees equal the Dijkstra trees exactly and the stress tests now
  compare the tree files too.
- **M2 a batch that deletes every edge** (`Dijkstra.cu`). The updated CSR
  then has an empty values file, `readCSR` infers 0 objectives and
  `runDijkstraCSR` rejected every objective index, so the stress harness
  reported a pipeline failure. The objective index is only range-checked
  when the graph has weights.
- `main` ignored the result of every step and `generateTestCases` returned
  true even when a case did not match Dijkstra; both now report failure
  (and `generateGraph` creates its output directory, which `main` needs on
  a fresh checkout). The stress tests print the run parameters of a
  pipeline error.

Regression: `test_mosp_update` (disconnection, delete-all, tie cases for
both updates; the first two fail on the original code), `parallelStressTest`
seed 6 (failed on the original), and the stress tests' tree comparison.
Additional check (not in `make test`): 5 x 400 configurations of each stress
test with other seeds, 0 failures.

## Correctness: H-SOSP

- **S3 non-positive weights** (`hypergraph/src/DynamicHypergraph.cpp`,
  docs). The documentation allowed non-negative weights and nothing checked
  them; two adjacent zero-weight hyperedges cut off from the source keep a
  stale finite distance, a fixed point of the relaxation, so the update
  converged to wrong distances without a fallback (reproduced: 3 wrong
  distances after one deletion). As in the paper (ω → R>0), weights must be
  >= 1; only the virtual source and target have weight 0. `bulkLoad` and
  `applyBatch` reject other weights (and empty inserted hyperedges) before
  touching any structure. Regression: `test_hsosp_matches_dijkstra`.
- **S4 "targeted" change placement** (`hsospBench.cu`, `hsospStress.cu`,
  `HypergraphGen.hpp`). Device parents are 0-based node indices but
  `generateBatch` reads them as 1-based hyperedge ids, so the targeted
  placement (experiment E7, not in the paper) deleted the hyperedge before
  each SOSP-tree parent. `HsospState::downloadParentIds` converts, the call
  sites use it and the convention is documented. Regression:
  `test_hsosp_matches_dijkstra` (every deletion of a targeted batch is a
  tree parent; 153 of 200 were not with the old conversion).
- **S2 verification in `hsospBench`**. Correctness was "update == static
  recompute", two results of the same kernels; after the fallback (every
  realistic batch, see S1) the update *was* the recompute, so the check could
  not fail. Host Dijkstra ran only up to 300K hyperedges and the `verified`
  column was hard-coded to 1. Now every batch of a hypergraph with at most
  `--verify-max` ids (default 5M) is also checked against the independent
  oracle (distances and shortest-path tree), and the CSV reports what was
  checked: `oracle` (host / none), `mismatch_static`, `mismatch_oracle`,
  `parent_errors`; `verified` is 1 only when the oracle ran and agreed. The
  smoke suite runs in `make test`.

## Benchmark driver

- **Real hypergraphs** (`hsospBench --hg FILE`). The repository had no
  loader for the paper's datasets (synthetic generators only). The new mode
  reads one hyperedge per line with the paper's preprocessing (duplicate
  vertices merged, hyperedges above `--maxcard`, default 25, dropped, vertex
  ids renumbered; weights U[1,100]; source = a vertex of maximum degree,
  target = a random vertex) and runs consecutive batches with the paper's
  batch model (`generatePaperBatch`: deletions of random hyperedges;
  insertions clone a hyperedge and replace about 30% of its vertices by
  vertices of a neighbouring hyperedge; vertex batches remove a member or
  adopt a neighbour's vertex), `--batch`, `--batches`, `--del`, `--kind`.
  Every batch goes through the same routine as the synthetic suite (dynamic
  pipeline, static recompute, checks) and gives one CSV row; `--verify`
  selects the batches the independent oracle checks (all / first / none).
  On coauth-DBLP this gives 2,466,792 hyperedges and 1,924,991 vertices
  (the paper lists 2,466,661 / 1,924,991).
- **S1 count-to-infinity and the every-batch fallback**
  (`hsosp/src/hsospDevice.cu`). The update re-evaluated seed nodes by pull
  with no invalidation: after a deletion, nodes that lost their tree path
  kept stale finite distances that rose by at least 1 per iteration and
  never converged, so every batch with a disconnection ran 512 capped
  iterations and then a full recompute (all baseline DBLP batches did;
  the update was never cheaper than the recompute it is compared with).
  `hsospUpdate` is now exact and incremental: roots (the endpoint of every
  deleted tree edge; new, recreated and dead nodes) → pointer-jumping
  invalidation of their pre-batch subtrees → one warp-cooperative pull per
  invalidated node and relaxation of both directions of every inserted pair
  → push to convergence with a packed 64-bit atomicMin on
  (distance << 32 | parent) and epoch-stamped frontiers. After invalidation
  every finite distance is realised by a path of the new graph, so values
  only decrease and no cap is needed. A work budget (default: as many edge
  relaxations as the graph has adjacency entries) and an iteration budget
  (4,096) fall back to the recompute, whose time is part of the reported
  update time; so does a distance that would not fit 32 bits (the packed
  word). Ties go to the lowest parent id in the update and the recompute,
  so parents are canonical and the oracle now checks them exactly. The
  host emulation (`emulateSospUpdate`, GPU-free `tests/local`) implements
  the same algorithm. The update needs the batch's pairs on the device;
  `applyDeltaToDevice` keeps them in `DeviceH2H::delta` (the seeds list is
  no longer used).
  DBLP, 50K hyperedge batches (other stages unchanged at this commit): SOSP
  stage 5.1-5.8 ms (batches 2-3; 12-13 push iterations, no fallback) against
  473-482 ms for the original (512 capped iterations + recompute); batch 1
  still pays a ~0.38 s host allocator stall (see P5). All three batches
  matched the oracle (distances and canonical parents).
  Regression: `hsospStress` (1,500 batches, budget disabled so every batch
  takes the incremental path; 0 failures), `test_hsosp_matches_dijkstra`
  (alternating batches with a zero budget exercise the fallback),
  `test_hsosp_scale`.
