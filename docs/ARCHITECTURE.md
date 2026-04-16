# Architecture

This document walks through the escher-mosp source layout in execution order
and explains the contract at each boundary.

## 1. libescher_core (`escher/`)

The CBST data structure, extracted from ESCHER-GPU-main with every motif
counting file removed. Everything here is reusable by any graph workload,
not just MOSP.

### Public types (`escher/include/structure.hpp`)

- `struct CBSTNode` — tree node with an index (key), start offset into the
  flat payload, tail bookkeeping (base, capacity, occupancy) and three
  pointer-based tree links (left / right / parent).
- `struct CBSTContext` — device-resident buffers plus sizing/alignment
  metadata. This is what every free function (`constructCBST`, `insertCBST`,
  etc.) mutates.
- `struct InsertMapping` — returned by `insertCBST`: for each new item it
  tells the caller which CBST key it actually landed on (may be a recycled
  slot's key rather than the requested key).
- `class CBSTOperations` — RAII wrapper owning a `CBSTContext`. Provides
  `construct`, `insert` (with best-fit slot reuse), `fill` (append-only),
  `erase` (delete by key) and `findAndPrint` (debug).

### Operations (`escher/structure/operations.cu`)

- `constructCBST` — allocate device buffers, upload the initial flat
  payload, build the balanced binary tree, populate nodes.
- `fillCBST` — append new data to the tail segments of existing keys.
- `insertCBST` — add new records, reusing deleted slots via a GPU-parallel
  best-fit match (Thrust sort / scan / copy_if). Items that do not find a
  matching slot get appended at `spaceAvailableFrom`.
- `deleteCBST` — mark nodes as available; bottom-up reduction rebuilds the
  subtree-available array for future best-fit lookups.
- `unfillCBST` — scan payloads and zero out specific values (used by
  `DynamicGraph::deleteEdges` to remove edge IDs from an adjacency list
  without erasing the whole vertex record).

### Kernels (`escher/kernel/`)

- `build_tree.cu` — builds the empty tree (`buildEmptyBinaryTree`) and
  stores items into the tree nodes (`storeItemsIntoNodes`). Contains the
  tree-position → in-order-rank formula flagged as fragile for
  non-power-of-two sizes; `test_cbst_smoke` exercises that path for
  N ∈ {1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000}.
- `find.cu` — BST lookup and payload printing. Contains a local,
  bounded-hops `readFlat` chain walker (was previously in the motif-specific
  `motif_utils.cuh`; see `MIGRATION_NOTES.md`).
- `payload.cu` — `insertNode*` family and `allocateSpace` for appending to
  full tail segments.
- `insert_reuse.cu` — GPU best-fit reuse (locateReusableSlots,
  lowerBoundKernel, applyReuse variants).
- `delete_avail.cu` — locate and apply deletes, reduce availability
  level-by-level.
- `unfill.cu` — value-level payload removal for existing keys.

## 2. DynamicGraph adapter (`graph/`)

The simplified API MOSP talks to. Keeps the ESCHER core one layer away from
MOSP so the code path is easy to follow.

### `graph/include/DynamicGraph.hpp`

Holds the pImpl-based `DynamicGraph` class. Public API:

```cpp
DynamicGraph(int numVertices, int numObjectives, int payloadCapacity);
void loadFromCSR(rowPtr, colInd, values);
void insertEdges(const std::vector<EdgeInsert>& edges);
void deleteEdges(const std::vector<EdgeDelete>& edges);
GraphSnapshot snapshot(int objective) const;
void dumpToCSR(rowPtr, colInd, values) const;
```

### `graph/src/DynamicGraph.cpp` (host orchestration)

`DynamicGraph::Impl` owns:

- `edgesCBST` — one record per directed edge, key = 1-based edge id,
  payload = `[src, dst, w_0 … w_{K-1}]` (fixed stride).
- `outAdjCBST` — one record per source vertex (always present, even for
  isolated vertices), key = vertex+1, payload = variable-length edge id list.
- `inAdjCBST` — one record per destination vertex, same shape as out.
- `outAdjShadow` / `inAdjShadow` — host mirror of the adjacency payloads.
- `edgeSrc / edgeDst / edgeWeights` — per-edge metadata indexed by (edge_id - 1).
- `freeEdgeIds` — LIFO free-list driving ESCHER's best-fit reuse.
- `edgeIdBySrcDst` — `unordered_map<(src,dst), edge_id>` used by `deleteEdges`.

`insertEdges` calls `edgesCBST->insert(...)` for best-fit slot reuse and
`outAdjCBST->fill(...)` / `inAdjCBST->fill(...)` to grow the adjacency lists.
`deleteEdges` calls `unfillCBST(...)` on each adjacency CBST and
`edgesCBST->erase(...)` on the edge records.

### `graph/src/snapshot.cu` (device materialization)

Implements `DynamicGraph::snapshot(objective)`: walks the host shadow to
build CSR triples, then uploads them to freshly allocated device buffers
owned by the returned `GraphSnapshot`. Also defines `GraphSnapshot`'s move
constructor / assignment / destructor so the cudaFree calls live in a `.cu`
translation unit.

### `graph/src/updateGraphWithESCHER.cpp` (MOSP entry point)

Drop-in replacement for MOSP's legacy `updateGraphCSR`. Same signature,
same on-disk output format. Routes the reading through `readCSR`, the
delete/insert batches through `DynamicGraph`, and the writing through
`dumpToCSR` plus a tiny file writer. The equivalence is validated by
`tests/unit/test_snapshot_matches_updateCSR.cu`.

## 3. MOSP (`mosp/`)

Unchanged except for a single call-site swap in four files:

- `mosp/src/main.cu` — step 3 now calls `updateGraphWithESCHER`.
- `mosp/src/stressTest.cu` — same swap.
- `mosp/src/parallelStressTest.cu` — same swap.
- `mosp/src/generateTestCases.cu` — same swap.

All CUDA kernels (`parallelSOSPUpdate` collect/update/bfs/mark,
`sequentialSOSPUpdate` host baseline, `Dijkstra`, `parallelCombinedGraph`)
are untouched. They continue to read the updated CSR files from disk —
those files are now written by the ESCHER-backed adapter.

The legacy `updateGraphCSR.cu` stays in the tree because
`test_snapshot_matches_updateCSR` needs it as a reference implementation.

## 4. Tests (`tests/unit/`)

- `test_cbst_smoke.cu` — construct/insert/erase at many sizes.
- `test_dynamicgraph_roundtrip.cu` — loadFromCSR → dumpToCSR byte equality.
- `test_snapshot_matches_updateCSR.cu` — end-to-end equivalence of the
  ESCHER path and the legacy path.
- `tests/run_all.sh` — shell driver that also invokes `main`, `stressTest`,
  `parallelStressTest`.

## 5. Build (`Makefile`)

- `libescher_core.a` static archive from `escher/**/*.{cu,cpp}`.
- `bin/main`, `bin/stressTest`, `bin/parallelStressTest` link against the
  library and the graph-adapter objects.
- Each unit test links only what it needs; `test_cbst_smoke` does not pull
  in MOSP or the graph adapter, keeping it minimal.
- `make syntax-check` runs `nvcc -E` on every TU without linking — usable
  on macOS during development.
