# H-SOSP before / after on real hypergraphs

Measured on 2026-09-26 on one RTX A5000 (24 GB, sm_86) used exclusively
during each run, host 28 cores / 56 threads (shared with other jobs, load
average about 8 during the runs), CUDA 12.9 (final build) and 13.1
(baseline build).

## Builds

- **original+E1E2**: the original code (tag `baseline-2026-09`) with only
  the two fixes without which it cannot run at this scale (E1, the 32-bit
  CBST rank that overflows above 65,535 records, and E2, the stale subtree
  counts after insert; see [CHANGES.md](../CHANGES.md)), plus the
  real-hypergraph mode of `hsospBench` (loader, paper batch model,
  oracle) so that it runs the same batch model. Original build flags
  (`-O2`, `-arch=sm_86`), CUDA 13.1. The unmodified original does not
  complete: it aborts on the first batch (colliding hyperedge id) or
  crashes in construct. Its results are correct only because every batch
  falls back to a full recompute; its ESCHER contents are not
  (CHANGES.md, E3-E7). The build is
  [baseline_original_E1E2.patch](baseline_original_E1E2.patch) applied to
  the tag:

  ```bash
  git worktree add ../baseline baseline-2026-09
  git -C ../baseline apply "$PWD/results/baseline_original_E1E2.patch"
  make -C ../baseline CUDA_ARCH=sm_86 NVCC=/usr/local/cuda-13.1/bin/nvcc \
      bin/hsospBench
  ```
- **final**: this branch, measured at commit `07bbceb` (later commits
  only zero allocations for the sanitizer, replace a deprecated functor
  and change documentation).

Both run the same batches: the batch generator is seeded and takes the
same decisions on the same hypergraph state.

## Method

`hsospBench --hg FILE --kind K --batch B --batches 3 --del 50` per
configuration: load, initial SSSP, then three consecutive batches of B
changes with 50 % deletions (paper's batch model, see
[docs/HSOSP.md](../docs/HSOSP.md)). Each configuration was run three
times; tables give the median over the three runs of the per-run mean
over the three batches.

- **dynamic**: the paper's per-batch time: ESCHER maintenance of the h2v,
  v2h and h2h CBSTs + unification (line-graph delta) + CSR apply + SOSP
  update, with a budget fallback to the recompute included.
- **static**: the baseline of the paper, a GPU recompute from blank on the
  updated graph with the same kernels (they are about 4x faster in the
  final build, P3, which lowers the static/dynamic ratio).
- **load + 3 batches**: in-process end-to-end time: reading and
  preprocessing the file, building ESCHER, the device graph and the
  initial SSSP, and the dynamic time of the three batches (the static
  baseline runs and the checks are excluded).
- **fallbacks**: batches whose update exceeded its budget and ended with
  the recompute (the original falls back in every batch); **iters**: push
  iterations of the update (original: rounds of its final recompute).

Correctness: every batch of the timing runs was compared with the static
recompute (0 mismatches). Separate runs with `--verify all` checked every
batch of every configuration against the independent oracle (Dijkstra on
the line graph rebuilt from the vertex lists, and the canonical
shortest-path tree): 24 of 24 batches on DBLP and 24 of 24 on Geology
agree, distances and parents (`data/*_verify.csv`).

## Summary

coauth-DBLP (2,466,792 hyperedges, 1,924,991 vertices, 125.4M line-graph pairs):

| batch | dynamic ms, original+E1E2 | dynamic ms, final | speedup | load + 3 batches s, original+E1E2 | final | speedup | update stage ms, final | static ms, final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hyperedge 25K | 6,835 | 107 | 64x | 31.7 | 5.33 | 5.9x | 3.2 | 78.3 |
| hyperedge 50K | 13,425 | 169 | 80x | 51.4 | 5.53 | 9.3x | 6.2 | 79.3 |
| hyperedge 100K | 26,405 | 275 | 96x | 90.5 | 5.84 | 15.5x | 10.1 | 79.9 |
| hyperedge 200K | 53,427 | 470 | 114x | 171.4 | 6.42 | 26.7x | 20.2 | 82.0 |
| vertex 25K | 2,031 | 61 | 34x | 17.3 | 5.20 | 3.3x | 2.7 | 77.6 |
| vertex 50K | 3,744 | 92 | 41x | 22.4 | 5.24 | 4.3x | 3.8 | 78.2 |
| vertex 100K | 7,345 | 144 | 51x | 33.2 | 5.38 | 6.2x | 5.7 | 78.3 |
| vertex 200K | 14,448 | 250 | 58x | 54.5 | 5.72 | 9.5x | 10.5 | 78.6 |

coauth-MAG-Geology (1,203,895 hyperedges, 1,256,385 vertices, 37.7M
line-graph pairs):

| batch | dynamic ms, original+E1E2 | dynamic ms, final | speedup | load + 3 batches s, original+E1E2 | final | speedup | update stage ms, final | static ms, final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hyperedge 25K | 3,385 | 70 | 48x | 14.5 | 2.34 | 6.2x | 2.3 | 26.8 |
| hyperedge 50K | 6,851 | 113 | 60x | 24.8 | 2.51 | 9.9x | 3.3 | 26.7 |
| hyperedge 100K | 13,831 | 199 | 69x | 45.8 | 2.75 | 16.7x | 17.7 | 27.4 |
| hyperedge 200K | 30,090 | 337 | 89x | 94.5 | 3.14 | 30.1x | 19.6 | 27.2 |
| vertex 25K | 962 | 35 | 27x | 7.2 | 2.25 | 3.2x | 1.5 | 27.3 |
| vertex 50K | 1,886 | 63 | 30x | 10.0 | 2.30 | 4.3x | 2.3 | 27.4 |
| vertex 100K | 3,762 | 96 | 39x | 15.6 | 2.41 | 6.5x | 5.9 | 26.7 |
| vertex 200K | 7,434 | 165 | 45x | 26.6 | 2.65 | 10.1x | 6.6 | 26.9 |

What the numbers say:

- The dynamic time per batch is 27-114x lower than in the original; load
  plus three batches is 3-30x faster (load itself: DBLP 11.2 → 5.0 s,
  Geology 4.3 → 2.1 s).
- The static recompute is 4.3x (DBLP) and 3.7x (Geology) faster than in
  the original (338 → 78 ms, 100 → 27 ms), because it shares the pull
  kernels rewritten in P3.
- Against that faster recompute, the dynamic path wins only for DBLP
  vertex batches of 25K (static/dynamic 1.28x); in the other
  configurations static/dynamic is 0.08-0.85x, i.e. recomputing is faster
  than updating once ESCHER maintenance and unification are counted, as
  the paper counts them. The paper reports 1.3-12.1x on DBLP (A100). The
  SOSP update stage alone takes 1.5-20 ms, 1.4-25x less than the
  recompute; ESCHER maintenance (50-70 % of the dynamic time) and
  unification (20-40 %) dominate.
- Geology hyperedge batches of 100K and 200K invalidate about 40 % of the
  nodes in one of their three batches; the update then exceeds its work
  budget and falls back to the recompute (3 of 9 batches each), as
  designed. No DBLP batch falls back.

## Per-stage detail

Medians as above; stage times in ms; `static/dynamic` is the speedup of
the dynamic path over the recompute in the paper's sense.

coauth-DBLP:

| batch | build | dynamic ms | ESCHER | unification | CSR | update | static ms | static/dynamic | fallbacks | iters | load s | load+dynamic s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hyperedge 25K | original+E1E2 | 6,835 | 2,257 | 2,759 | 1,364 | 455 | 338 | 0.05x | 9/9 | 17-20 | 11.2 | 31.67 |
| hyperedge 25K | final | 107 | 66.9 | 31.2 | 6.2 | 3.2 | 78.3 | 0.73x | 0/9 | 12-13 | 5.0 | 5.33 |
| hyperedge 50K | original+E1E2 | 13,425 | 4,675 | 5,655 | 2,321 | 770 | 340 | 0.03x | 9/9 | 18-19 | 11.2 | 51.44 |
| hyperedge 50K | final | 169 | 98.8 | 53.6 | 10.2 | 6.2 | 79.3 | 0.47x | 0/9 | 12-18 | 5.0 | 5.53 |
| hyperedge 100K | original+E1E2 | 26,405 | 9,491 | 11,713 | 4,065 | 1,108 | 346 | 0.01x | 9/9 | 16-19 | 11.2 | 90.50 |
| hyperedge 100K | final | 275 | 155 | 93.3 | 16.4 | 10.1 | 79.9 | 0.29x | 0/9 | 15-18 | 5.0 | 5.84 |
| hyperedge 200K | original+E1E2 | 53,427 | 19,059 | 25,074 | 7,959 | 1,334 | 345 | 0.01x | 9/9 | 17-20 | 11.1 | 171.38 |
| hyperedge 200K | final | 470 | 255 | 167 | 28.2 | 20.2 | 82.0 | 0.17x | 0/9 | 14-19 | 5.0 | 6.42 |
| vertex 25K | original+E1E2 | 2,031 | 571 | 690 | 341 | 427 | 335 | 0.16x | 9/9 | 19-20 | 11.2 | 17.31 |
| vertex 25K | final | 60.6 | 42.7 | 12.4 | 2.6 | 2.7 | 77.6 | 1.28x | 0/9 | 11-15 | 5.0 | 5.20 |
| vertex 50K | original+E1E2 | 3,744 | 1,186 | 1,442 | 662 | 455 | 333 | 0.09x | 9/9 | 18-20 | 11.1 | 22.36 |
| vertex 50K | final | 92.2 | 61.2 | 23.4 | 3.8 | 3.8 | 78.2 | 0.85x | 0/9 | 13-15 | 5.0 | 5.24 |
| vertex 100K | original+E1E2 | 7,345 | 2,458 | 3,012 | 1,378 | 492 | 335 | 0.05x | 9/9 | 16-20 | 11.1 | 33.16 |
| vertex 100K | final | 144 | 86.5 | 44.8 | 5.9 | 5.7 | 78.3 | 0.55x | 0/9 | 13-17 | 4.9 | 5.38 |
| vertex 200K | original+E1E2 | 14,448 | 5,024 | 6,149 | 2,734 | 541 | 340 | 0.02x | 9/9 | 16-19 | 11.1 | 54.45 |
| vertex 200K | final | 250 | 143 | 86.6 | 9.8 | 10.5 | 78.6 | 0.31x | 0/9 | 15-17 | 5.0 | 5.72 |

coauth-MAG-Geology:

| batch | build | dynamic ms | ESCHER | unification | CSR | update | static ms | static/dynamic | fallbacks | iters | load s | load+dynamic s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hyperedge 25K | original+E1E2 | 3,385 | 1,174 | 1,394 | 652 | 158 | 101 | 0.03x | 9/9 | 21-24 | 4.3 | 14.49 |
| hyperedge 25K | final | 70.0 | 40.7 | 23.0 | 3.7 | 2.3 | 26.8 | 0.38x | 0/9 | 16-17 | 2.1 | 2.34 |
| hyperedge 50K | original+E1E2 | 6,851 | 2,233 | 3,188 | 1,027 | 331 | 99.8 | 0.01x | 9/9 | 22-26 | 4.3 | 24.84 |
| hyperedge 50K | final | 113 | 64.4 | 40.2 | 5.3 | 3.3 | 26.7 | 0.24x | 0/9 | 15-19 | 2.2 | 2.51 |
| hyperedge 100K | original+E1E2 | 13,831 | 4,911 | 6,501 | 1,990 | 430 | 102 | 0.01x | 9/9 | 20-24 | 4.3 | 45.76 |
| hyperedge 100K | final | 199 | 102 | 70.6 | 8.9 | 17.7 | 27.4 | 0.14x | 3/9 | 2-20 | 2.2 | 2.75 |
| hyperedge 200K | original+E1E2 | 30,090 | 9,826 | 15,720 | 4,039 | 505 | 103 | 0.00x | 9/9 | 21-22 | 4.2 | 94.51 |
| hyperedge 200K | final | 337 | 175 | 128 | 14.5 | 19.6 | 27.2 | 0.08x | 3/9 | 4-23 | 2.1 | 3.14 |
| vertex 25K | original+E1E2 | 962 | 303 | 361 | 159 | 140 | 98.8 | 0.10x | 9/9 | 25-26 | 4.3 | 7.16 |
| vertex 25K | final | 35.3 | 21.5 | 10.9 | 1.5 | 1.5 | 27.3 | 0.77x | 0/9 | 12-18 | 2.1 | 2.25 |
| vertex 50K | original+E1E2 | 1,886 | 653 | 740 | 341 | 151 | 99.3 | 0.05x | 9/9 | 24-26 | 4.3 | 9.97 |
| vertex 50K | final | 63.4 | 38.6 | 20.5 | 2.0 | 2.3 | 27.4 | 0.43x | 0/9 | 15-21 | 2.1 | 2.30 |
| vertex 100K | original+E1E2 | 3,762 | 1,340 | 1,555 | 690 | 177 | 101 | 0.03x | 9/9 | 24-25 | 4.3 | 15.57 |
| vertex 100K | final | 96.0 | 49.9 | 37.6 | 2.9 | 5.9 | 26.7 | 0.28x | 0/9 | 18-22 | 2.1 | 2.41 |
| vertex 200K | original+E1E2 | 7,434 | 2,750 | 3,243 | 1,143 | 299 | 97.8 | 0.01x | 9/9 | 20-24 | 4.3 | 26.61 |
| vertex 200K | final | 165 | 82.7 | 71.5 | 4.6 | 6.6 | 26.9 | 0.16x | 0/9 | 18-23 | 2.1 | 2.65 |

## Not measured

- Orkut, AMiner and MAG: their h2h CBST payloads (about 2.1, 2.5 and 2.9 x
  10^9 values) exceed the `int`-indexed CBST payload limit of 2^31, so
  `bulkLoad` rejects them regardless of GPU memory. Threads (about 1.55 x
  10^9) would fit that limit but was not downloaded (size and time
  budget).
- The synthetic suites (`--suite full`, HG-S ... HG-XL) were not rerun.

## Files and reproduction

- `data/<dataset>_original_E1E2.csv`, `data/<dataset>_final.csv`: every
  batch of the timing runs (column `run` = 1..3, `rep` = batch index).
  The baseline CSVs lack the final build's columns `fallback_iters`,
  `invalidated`, `update_work` and `static_iters`; they already carry
  `oracle`, `mismatch_static`, `mismatch_oracle` and `parent_errors`,
  which the benchmark's real-hypergraph mode added to both builds.
- `data/<dataset>_<build>_e2e.csv`: the `[e2e]` line of every timing run
  (load time, batches, sums of the dynamic and static times, process wall
  time; ms), the source of the load columns.
- `data/<dataset>_verify.csv`: the oracle runs (`verified` = 1).
- `baseline_original_E1E2.patch`: the original+E1E2 build (see Builds).

The inputs are Benson et al.'s coauth-DBLP-full and coauth-MAG-Geology-full
simplicial datasets, which list a simplex once per timestamp. The paper
uses each distinct simplex once, and `hsospBench` does not remove repeated
lines, so `scripts/benson_to_hg.py` writes one line per distinct vertex
set (vertices sorted and renumbered 1..V in id order): DBLP 3,700,681
simplices → 2,467,389 lines, Geology 1,203,895 lines. `--maxcard 25`
(the default) then keeps 2,466,792 and 1,203,895 hyperedges.

```bash
./scripts/benson_to_hg.py coauth-DBLP-full/coauth-DBLP-full coauth.hg
# one run of one configuration; the sweep ran every kind x batch size for
# runs r1..r3 with each build
./bin/hsospBench --hg coauth.hg --kind hyperedge --batch 50000 --batches 3 \
    --verify none --out final/hyperedge_50000_r1.csv \
    2> final/hyperedge_50000_r1.log
# the tables above, from the committed data
./scripts/summarize_hg.py \
    --build original+E1E2 results/data/dblp_original_E1E2.csv \
    --build final results/data/dblp_final.csv
# the same from directories of <kind>_<batch>_r<run>.csv / .log files;
# --merge writes the merged CSVs and the _e2e.csv files of results/data
./scripts/summarize_hg.py --build dblp_original_E1E2 base/ \
    --build dblp_final final/ --merge merged/
```
