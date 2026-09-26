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
