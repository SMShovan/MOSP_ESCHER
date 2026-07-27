# Plan: Dynamic Single-Objective Shortest Path on Hypergraphs (H-SOSP) with ESCHER on GPU

Prepared from: the meeting video + handwritten notes (Meeting/video1989311071.mp4), the escher-mosp
codebase, the DynaMOSP paper (MOSP GPU), the ESCHER/Dynamic Hypergraph papers (conference + TKDE),
and `Our plotting style.ipynb`.

---

## 1. Problem statement (what the meeting defines)

We solve the **single-objective, single-source shortest path problem on a weighted hypergraph**,
in the **fully dynamic** setting, on GPU, reusing two existing assets:

- **ESCHER** (CBST) as the authoritative dynamic hypergraph store, and
- the **parallel SOSP update algorithm** from the MOSP project as the update engine.

Model, exactly as in the handwritten notes:

- Hypergraph `H = (V, E_H)` with hyperedges `h_1 .. h_m`. Each hyperedge `h_i` carries one
  non-negative weight `w_i` (single objective). Per the meeting discussion, the hypergraph is
  treated as **undirected** for this phase (the notes sketch directed variants; the decision
  in the meeting was to take the undirected case first).
- Two hyperedges are neighbors iff they share at least one vertex (stated explicitly in the
  meeting: an h2h edge exists while a common vertex exists, and is deleted only when the last
  common vertex disappears). This gives the **h2h (line graph) representation**: one node per
  hyperedge, one symmetric pair of directed edges per overlapping pair.
- **Cost model (confirmed):** stepping from `h_i` into a neighboring `h_j` costs `w_j`
  (the weight of the hyperedge being entered). The cost of a path is the sum of weights of the
  hyperedges entered along it.
- **Step 1 of the notes:** add a virtual hyperedge `h_0 = {s}` with weight 0 for the source
  vertex `s`, and a virtual hyperedge `h_{m+1} = {t}` containing only the target `t`.
  Then `dist(s → t)` in the hypergraph equals the SOSP distance from node `h_0` to node
  `h_{m+1}` in the h2h graph. Computing the full **SOSP tree rooted at `h_0`** gives distances
  to every hyperedge (and hence, to every vertex: `dist(v) = min over h ∋ v of dist(h)`).
- **Dynamics ("for Δ vertex/edge ins/del, process as we do"):**
  - *Vertical ops:* hyperedge insertion / deletion.
  - *Horizontal ops:* incident-vertex insertion / deletion inside existing hyperedges.
  - Every hypergraph change maps to a **batch of edge insertions/deletions in the h2h graph**,
    which is then consumed by the existing parallel SOSP-update algorithm
    (Step 1: process changed edges, Step 2: iterative propagation), unchanged in spirit.

Baseline (confirmed): **static recompute** — after each change batch, recompute SOSP from
scratch on the updated h2h graph on GPU. Speedup = static time / dynamic-update time.

Positioning (from the meeting): to the team's knowledge there is **no existing parallel or
dynamic algorithm for shortest paths on hypergraphs** in this formulation, which is the
novelty claim the experiments must support (dynamic vs recompute speedup, scalability, and
data structure overheads).

Datasets (confirmed): **synthetic hypergraphs only** for this phase. The real graph datasets in
`dataset/` stay out of v1 (kept as an optional extension at the end of this plan).

---

## 2. Approach overview

```
                     ┌──────────────────────────────────────────────┐
   synthetic         │      DynamicHypergraph  (new adapter)        │
   generator ──────► │  h2vCBST   v2hCBST   h2hCBST   (ESCHER core) │
                     │  + host shadows + overlap multiplicities     │
   change  ────────► │  insert/delete hyperedges                    │
   batches           │  insert/delete incident vertices             │
                     │        │                                     │
                     │        ▼                                     │
                     │  Δ(h2h edges)  = derived ins/del on line grph│
                     └────────┼─────────────────────────────────────┘
                              ▼
             resident device h2h CSR (updated in place)
                              ▼
        parallelSOSPUpdate kernels (existing, refactored to in-memory API)
                              ▼
              distances[h], parent[h]  (SOSP tree over hyperedges)
```

Key design decisions:

1. **h2h is the graph the SOSP kernels see.** The existing `collectCandidatesKernel` /
   `updateDistancesKernel` / BFS reachability kernels are reused as-is on the h2h CSR; only the
   surrounding I/O changes. This is the strongest form of "utilize our existing project".
2. **ESCHER stays the authoritative store.** All three mappings from the ESCHER paper are
   instantiated as CBSTs: `h2v` (incident vertex list per hyperedge), `v2h` (incident hyperedge
   list per vertex), `h2h` (neighbor hyperedge list per hyperedge). Every dynamic update flows
   through `CBSTOperations::insert / ::erase / ::fill / unfillCBST`, mirroring how
   `DynamicGraph` already does it for ordinary graphs.
3. **Overlap multiplicity is the h2h edge support.** For a pair `(h_i, h_j)` we track
   `ov[i][j] = |h_i ∩ h_j|`. An h2h edge exists iff `ov > 0`. A vertex removal decrements
   support and only deletes the h2h edge when support hits 0; hyperedge deletion removes all
   its h2h edges. This makes horizontal and vertical dynamics compose correctly.
4. **Host shadow first, GPU delta kernels second.** Following the proven `DynamicGraph`
   pattern, v1 computes the h2h delta on the host shadow (`v2hShadow` makes it a linear pass
   over the touched vertices) while ESCHER device structures are updated in lock-step. A GPU
   delta-extraction kernel family (traversing v2h payloads with atomics on an open-addressing
   overlap table) is planned as an optimization milestone; the benchmark timing schema already
   separates "DS maintenance" from "SOSP update" so both variants slot in without changing
   the experiment harness.
5. **Resident device h2h CSR with slack.** `parallelSOSPUpdate` today rebuilds and re-uploads
   the whole CSR from disk on every call, which would swamp all measurements. The new in-memory
   path keeps the h2h CSR resident on device across batches and applies the (small) delta with
   lightweight kernels; a full rebuild path is kept as the correctness fallback and for
   occasional compaction.

---

## 3. Data structure design (`hypergraph/` — new module)

`escher_mosp::DynamicHypergraph` (pImpl, mirrors `DynamicGraph`):

ESCHER-backed (authoritative, all device-resident):

- `h2vCBST` — key = hyperedge id (1-based), payload = incident vertex list.
- `v2hCBST` — key = vertex id + 1, payload = incident hyperedge id list.
- `h2hCBST` — key = hyperedge id, payload = neighbor hyperedge id list
  (tombstoned via `unfillCBST` on deletion, exactly like the crossed-out slot in the
  meeting notes' `h_4` row).

Host state (lock-step shadows, same philosophy as `DynamicGraph`):

- `heVertices[i]` — vertex list per hyperedge; `heWeight[i]` — hyperedge weight;
  `heAlive[i]` — liveness flag; LIFO free-list of hyperedge ids for ESCHER best-fit reuse.
- `v2hShadow[v]` — hyperedge ids containing `v`.
- `h2hShadow[i]` — `unordered_map<int,int>` neighbor → overlap multiplicity
  (memory-bounded: total size = Σ h2h degree).
- Virtual nodes: `h_0` (source, weight 0) and `h_{m+1}` (target) are ordinary rows that the
  generator emits; they participate in h2h maintenance like any hyperedge.

Public API:

```cpp
DynamicHypergraph(int numVertices, int maxHyperedges, long long payloadCapacity);
void  bulkLoad(const HypergraphData& hg);              // initial build (host + ESCHER + device CSR)
void  insertHyperedges(const std::vector<HyperedgeInsert>&);   // {vertices, weight}
void  deleteHyperedges(const std::vector<int>& ids);
void  insertIncidentVertices(const std::vector<IncidentChange>&); // {heId, vertexId}
void  deleteIncidentVertices(const std::vector<IncidentChange>&);
const H2HDelta& lastDelta() const;   // ins/del edge lists + weights for the SOSP updater
DeviceH2HView deviceView();          // resident device CSR (in+out adjacency, weights)
void  rebuildDeviceCSR();            // compaction / correctness fallback
Stats stats() const;                 // m, |h2h| edges, avg degree, device bytes
```

Delta rules (host pass over touched vertices only):

- insert hyperedge `h_new` with set `S`: for each `v ∈ S`, for each `h ∈ v2hShadow[v]`:
  `ov[h_new][h]++`; on 0→1 emit h2h edge pair (`h → h_new` with weight `w_new`,
  `h_new → h` with weight `w_h`).
- delete hyperedge `h`: emit deletion of all h2h edges incident to `h`; decrement partner maps.
- incident vertex insert/delete on `h`: same, restricted to one vertex.
- hyperedge weight change (optional op): emits weight updates on all in-edges of that node —
  supported by the SOSP updater's existing "weight increase / decrease" handling.

Complexities match the ESCHER paper style: a batch touching `χ` hyperedges with max
cardinality `c_max` and max vertex degree `d_max` costs
`O(χ · c_max · d_max)` shadow work + the usual `O((χ/p)·log m)` ESCHER tree operations.

---

## 4. Algorithms (`hsosp/` — new module)

1. **Initial SOSP** on the h2h CSR: reuse the trick already used by
   `parallelCombinedGraph`: run the parallel SOSP update kernels from a blank state
   (all INF, `dist[h_0]=0`) — i.e., GPU Bellman-Ford with frontier dedup. No new kernel needed.
2. **Dynamic H-SOSP update** per change batch (the deliverable algorithm):
   - Step A: apply batch to `DynamicHypergraph` (ESCHER ops + shadows) → `H2HDelta`.
   - Step B: apply delta to resident device CSR (small kernels / slack rows).
   - Step C: existing Phase-1 host processing of changed h2h edges (insert relaxations,
     deletion re-parenting) on the hyperedge-level `dist` / `parent` arrays.
   - Step D: existing Phase-2 iterative CUDA propagation (collect candidates → update
     distances) until convergence + BFS reachability fix for disconnections.
   Every step is timed with `cudaEvent`s and reported separately.
3. **Static baseline**: GPU recompute-from-blank (same kernels) on the updated h2h CSR.
   Timing excludes hypergraph maintenance (favorable to the baseline, mirroring the MoCHy
   comparison methodology).
4. **Ground truth**: host Dijkstra on the h2h graph (size-gated) for correctness checks;
   reused from `Dijkstra.cu` conceptually, implemented for in-memory h2h.

Refactor of the existing code (explicitly allowed by you):

- Extract the body of `parallelSOSPUpdate` into an in-memory library function
  `sospUpdateInMemory(DeviceGraphView, HostChangeSet, DistTree&, PhaseTimings&)`;
  the current file-based `parallelSOSPUpdate(...)` becomes a thin wrapper (all existing
  tests and binaries keep working, `test_snapshot_matches_updateCSR` stays green).
- Audit items I will fix while in there:
  - `DynamicGraph::insertEdges` performs no existence check, so re-inserting an existing
    `(u,v)` leaks a parallel edge record and stale map entry (upsert is only handled one
    layer up in `updateGraphWithESCHER`). Fix at the `DynamicGraph` level; same guard goes
    into `DynamicHypergraph`.
  - CBST `payloadCapacity` is a hard preallocation (65536 in `main.cu`) with no overflow
    diagnostics surfaced — add explicit capacity checks + clear `EscherError` messages, and
    compute capacities from generator parameters in the benchmark driver.
  - Replace hard-coded constants in `main.cu` with CLI arguments for the new binaries;
    seed every RNG (`generateChangedEdges` already accepts a seed — the new generators will too).
  - Full-graph H2D copies per call and disk round-trips disappear on the benchmark path
    (resident device CSR); host-file path retained for the pipeline demo.

---

## 5. Synthetic hypergraph generation (`hsosp/src/generateHypergraph.cu`)

Uniform random vertex sampling makes the expected h2h degree explode
(`E[deg] ≈ m·c²/n`, e.g. m=10M, n=m/3, c=16 gives degree ≈ 770 → 7.7B h2h edges). So the
generator uses a **clustered-pool model**, which both keeps h2h density controllable and gives
realistic community structure:

- Vertices are partitioned into pools of size `p`. Each hyperedge picks a home pool and
  samples its `c` vertices inside it; with probability `β` one vertex is drawn from a
  neighboring pool (bridges keep the h2h graph connected and let us dial inter-cluster paths).
- Parameters: `n` vertices, `m` hyperedges, cardinality `c ~ U[c_min, c_max]`, weights
  `w ~ U[1, 100]`, pool size `p`, bridge fraction `β`, seed.
  Expected h2h degree ≈ `(m·p/n)·(1 − exp(−c²/p))` + bridge term — the driver prints the
  measured value and it lands in the CSV.
- Source `s` = a vertex in pool 0; generator emits `h_0 = {s}` (w=0) and `h_{m+1} = {t}`
  with `t` in the last pool. Reachable fraction from `h_0` is measured and logged.
- Output format: one file, `m` lines of `w_i c_i v_1 … v_{c_i}` + a small JSON sidecar with
  the parameters — read by both the benchmark driver and the tests.

**Change-batch generator** (`generateChangedHyperedges.cu`), mirroring `generateChangedEdges`:

- Hyperedge-level batches: `ΔE ∈ {25K, 50K, 100K, 200K}` with deletion percentage
  `∈ {20, 40, 60, 80}`; deletions sampled from live hyperedge ids, insertions drawn from the
  same pool model. A `--targeted` mode samples deletions only from current SOSP-tree
  hyperedges and gives insertions weights below the average (the DynaMOSP "targeted changes"
  methodology), and a `--near/--far` mode restricts changed hyperedges by their current
  distance from `h_0` relative to the h2h diameter (the "low/high" experiment).
- Incident-vertex batches: pick random live hyperedges, insert a random pool vertex or delete
  a random member (never below cardinality 1), same sizes.
- Everything seeded; batches are written to files so a run is fully reproducible.

**Named configurations** (pilot-calibrated on the A100 80GB; targets: h2h edge count is the
memory driver at ~32 B/edge for in+out CSR):

| Name  | m (hyperedges) | n (vertices) | c_max | pool p | target avg h2h deg | est. h2h edges |
|-------|---------------|--------------|-------|--------|--------------------|----------------|
| HG-S  | 1M            | 333K         | 16    | 4K     | ~16                | ~16M           |
| HG-M  | 5M            | 1.7M         | 16    | 8K     | ~24                | ~120M          |
| HG-L  | 10M           | 3.3M         | 8     | 8K     | ~16                | ~160M          |
| HG-XL | 20M           | 6.7M         | 8     | 16K    | ~12                | ~240M          |
| HG-C  | 5M            | 1.7M         | 64    | 32K    | ~48 (high card)    | ~240M          |

The first cluster run executes a calibration pass that measures actual degrees/memory and, if
needed, auto-scales pool sizes to hit the targets before the main experiment matrix runs.

---

## 6. Experiment design (following the papers' style + my additions)

Every experiment: 3 repetitions after 1 warmup, `cudaEvent` timing, means plotted, all raw
rows kept in the CSV. Fixed defaults unless swept: batch 50K, 50% deletion, hyperedge-level
changes, random placement, dataset HG-M.

- **E1. Time vs batch size** (ΔE = 25K/50K/100K/200K) for Dynamic vs Static across all five
  configs; both hyperedge-level and incident-vertex-level variants.
  → grouped bar per dataset, hue = ΔE (rocket palette, `time_vs_DeltaE.pdf` style), plus a
  per-dataset Dynamic-vs-Static line plot (`{dataset}_base_vs_DeltaE.pdf` style).
- **E2. Time vs deletion percentage** (20/40/60/80 at ΔE=50K): Dynamic vs Static
  → line plots with markers per dataset (`{dataset}-del-vary.pdf` style).
- **E3. Scalability vs hypergraph size**: m ∈ {1M, 2M, 5M, 10M, 20M} (pool model, fixed
  c_max=8, fixed ΔE=50K) → bar plot with value labels (`time_vs_vertexcount.pdf` style).
- **E4. Effect of cardinality**: c_max ∈ {8, 16, 32, 64, 128} on HG-M-sized hypergraphs
  → bar plot (`time_vs_length.pdf` style). Higher cardinality densifies h2h; insertion
  overflow (ESCHER case 2) cost shows here.
- **E5. Effect of overlap density** (my addition): sweep pool size / bridge fraction to vary
  avg h2h degree at fixed m — measures how line-graph density drives both maintenance and
  propagation cost. → line plot, x = measured avg h2h degree.
- **E6. Phase/time breakdown**: stacked percentage bars per dataset:
  ESCHER maintenance | delta extraction | CSR apply | SOSP step 1 | SOSP step 2
  (`stacked_percentage.pdf` style).
- **E7. Change placement (near vs far)**: targeted batches near `h_0` vs far (by h2h depth)
  → grouped bars, mirrors the DynaMOSP "targeted changed edges" figure.
- **E8. Speedup summary**: Dynamic vs Static across all configs and batch sizes
  → speedup bar chart with `bar_label` annotations (`Speedup.pdf` style) + avg/max speedup
  table for the text.
- **E9. Memory**: device bytes for ESCHER structures + h2h CSR vs m
  → line/bar (`memory_plot.pdf` style).
- **E10. Correctness gate** (reported, not plotted): every benchmarked configuration at small
  scale (m ≤ 100K) is verified against host Dijkstra; the stress harness runs 100 random
  configurations like the existing `stressTest`/`parallelStressTest` and the CSV records
  pass/fail. The full run aborts if any correctness check fails.

**CSV schema** (`results/results.csv`, one row per measured run):

```
timestamp, dataset, m, n, c_max, pool, beta, avg_h2h_deg, h2h_edges,
batch_kind(hyperedge|vertex), batch_size, del_pct, placement(random|targeted|near|far),
rep, t_escher_ms, t_delta_ms, t_csr_apply_ms, t_sosp_p1_ms, t_sosp_p2_ms,
t_dynamic_total_ms, t_static_ms, speedup, sosp_iters, affected_max, frontier_peak,
dev_mem_mb, reachable_frac, correct(0|1|skipped), seed
```

---

## 7. Single-run cluster pipeline

`scripts/run_experiments.sh` (direct interactive run on mill.mst.edu, per your answer):

```
module load cuda-toolkit/12.5        # guarded: skipped if module absent
make -j all                          # existing binaries + new: bin/hsospBench, bin/hsospStress
./tests/run_all.sh --quick           # existing tests + new hypergraph unit tests (gate)
./bin/hsospStress  --configs 100     # correctness gate vs Dijkstra (gate)
./bin/hsospBench   --suite full --out results/results.csv
python3 scripts/plot_results.py results/results.csv results/figures/
```

- `--suite smoke` variant (~5 min, tiny configs) to validate the whole chain before the long
  run; `--suite full` is the paper run (est. a few hours, printed per-experiment ETA).
- `hsospBench` writes the CSV **incrementally** (flush after every row) so a partial run
  still yields plottable data; it also snapshots the exact generator/batch seeds used.
- `plot_results.py`: pandas + seaborn, one function per figure, faithful to the notebook:
  `sns.barplot(palette='rocket', dodge=True)` grouped bars with y-grid behind bars and framed
  legends; `sns.lineplot(marker='o')` for del-vary and density sweeps; stacked percentage
  bars; `bar_label` annotations on speedup plots; every figure saved as PDF with
  `bbox_inches='tight'`, figsize (6,4). Figure filenames follow your conventions
  (`time_vs_DeltaE.pdf`, `HG-M-del-vary.pdf`, `stacked_percentage.pdf`, `Speedup.pdf`, …).
  If seaborn is missing on the cluster it pip-installs `--user` pinned versions.

Nothing runs on your MacBook; you rsync with the existing `sync_to_cluster.sh` (I will extend
its excludes for `results/`) and execute one script.

---

## 8. Repository changes summary

```
hypergraph/
  include/DynamicHypergraph.hpp     new adapter (h2v, v2h, h2h CBSTs + shadows + delta)
  src/DynamicHypergraph.cpp         host orchestration + ESCHER routing
  src/h2hDeviceView.cu              resident device CSR + delta-apply kernels
hsosp/
  include/hsosp.cuh                 in-memory SOSP update + recompute API
  src/sospInMemory.cu               refactored kernels/driver (kernels unchanged)
  src/hypergraphDijkstra.cpp        host ground truth
  src/generateHypergraph.cu         pool-model generator
  src/generateChangedHyperedges.cu  batch generator (targeted / near-far modes)
  src/hsospBench.cu                 experiment driver → CSV
  src/hsospStress.cu                randomized correctness harness
tests/unit/
  test_h2h_construction.cu          h2h == brute-force line graph on small inputs
  test_h2h_delta.cu                 batched delta == rebuild-from-scratch line graph
  test_hsosp_matches_dijkstra.cu    dynamic update == Dijkstra recompute
scripts/
  run_experiments.sh                single-run pipeline (smoke/full)
  plot_results.py                   seaborn figures from CSV
Makefile                            new targets wired in; existing targets untouched
```

Existing MOSP/ESCHER code: kernels untouched; `parallelSOSPUpdate.cu` refactored into
wrapper + library function; small bug fixes listed in §4; all current tests must stay green
(`test_snapshot_matches_updateCSR` remains the regression anchor for the old path).

---

## 9. Validation strategy

1. Unit: CBST-backed h2h equals a brute-force line graph builder for randomized small
   hypergraphs (including overlap multiplicities and tombstone reuse after deletions).
2. Delta: for random batch sequences, incrementally maintained h2h equals
   built-from-scratch h2h after every batch.
3. Algorithm: dynamic H-SOSP distances equal host Dijkstra on the updated h2h graph across
   100 random configurations (both batch kinds, all deletion percentages, disconnections
   included — the BFS reachability pass must mark INF correctly).
4. Pipeline: smoke suite runs end to end on the cluster producing a CSV and all figures
   before the full suite is attempted.
5. In this cloud workspace (no GPU) I will keep everything compiling via `nvcc`
   host-compilation (`make syntax-check` extended to the new TUs) and run the host-side
   logic (generator, shadows, delta, Dijkstra) natively with a CPU test harness, so what you
   copy to the cluster has the highest possible chance of building and passing first try.

---

## 10. Milestones (implementation order)

1. **M1 — Hypergraph core**: generator + `DynamicHypergraph` (bulk load, h2h build, shadows,
   ESCHER routing) + unit tests. *(largest single chunk)*
2. **M2 — SOSP refactor**: in-memory API extraction, initial-SOSP-from-blank, host Dijkstra,
   equivalence tests on static hypergraphs.
3. **M3 — Dynamic pipeline**: batch application, delta → device CSR, update algorithm,
   stress harness green.
4. **M4 — Benchmark driver**: experiment matrix, timing instrumentation, CSV.
5. **M5 — Plots + runner**: `plot_results.py`, `run_experiments.sh`, smoke suite, docs
   (README section + ARCHITECTURE addendum).
6. **M6 (stretch) — GPU delta extraction**: kernel family replacing the host delta pass,
   plus an E6 re-run showing the maintenance share shrink.

Open items I flagged and will resolve during implementation (will not block M1):

- Exact per-block layout for h2h CBST payloads (neighbor id + cached weight vs id-only with a
  separate weight array — leaning id-only + `heWeight[]` lookup so weight changes are O(1)).
- Whether `h_{m+1}` (target) participates in all experiments or only in the pipeline demo
  (SOSP tree from `h_0` is the measured object either way, matching the notes).
- Auto-calibration bounds for HG-XL so the resident CSR + ESCHER buffers stay under 60 GB.

## 11. Optional extension (explicitly out of v1 scope, per your instruction)

Deriving hypergraphs from the real graphs in `dataset/` (e.g., 1-hop neighborhoods or the
synth-clusters communities as hyperedges) plugs into the same pipeline later: the reader
already accepts the generator's file format, so only one converter script is needed when you
want real-data figures.
