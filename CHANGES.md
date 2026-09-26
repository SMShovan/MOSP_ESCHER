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
