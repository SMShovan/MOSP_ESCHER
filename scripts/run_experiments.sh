#!/usr/bin/env bash
#
# One-shot H-SOSP experiment pipeline for the cluster (mill.mst.edu).
# Builds everything, gates on correctness, runs the benchmark matrix into a
# CSV, and renders every figure with seaborn.
#
#   ./scripts/run_experiments.sh smoke     # ~5-10 min sanity pass (default)
#   ./scripts/run_experiments.sh full      # the paper run (several hours)
#
# Environment overrides:
#   HSOSP_RESULTS   output directory                (default: results/<suite>_<timestamp>)
#   HSOSP_REPS      repetitions per scenario        (default: 3)
#   HSOSP_EXP       experiment subset, e.g. "E1,E3" (default: all)
#   HSOSP_SEED      base RNG seed                   (default: 20260725)
#   CUDA_ARCH       nvcc arch                       (default: sm_70; use sm_80 on A100)
#
# Every step is logged; the script aborts on the first failure so a broken
# build or a correctness regression can never silently produce figures.

set -euo pipefail

SUITE="${1:-smoke}"
if [[ "$SUITE" != "smoke" && "$SUITE" != "full" ]]; then
    echo "usage: $0 [smoke|full]" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="${HSOSP_RESULTS:-results/${SUITE}_${STAMP}}"
REPS="${HSOSP_REPS:-3}"
EXP="${HSOSP_EXP:-E1,E2,E3,E4,E5,E7}"
SEED="${HSOSP_SEED:-20260725}"
ARCH="${CUDA_ARCH:-sm_70}"

mkdir -p "$RESULTS"
LOG="$RESULTS/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== H-SOSP pipeline: suite=$SUITE results=$RESULTS reps=$REPS exp=$EXP seed=$SEED ==="
date

# ---- 0. environment --------------------------------------------------------
if command -v module >/dev/null 2>&1; then
    module load cuda-toolkit/12.5 2>/dev/null || \
        echo "[env] cuda-toolkit module not loaded (nvcc must be on PATH)"
fi
nvcc --version | tail -1
nvidia-smi -L 2>/dev/null || echo "[env] nvidia-smi unavailable"

# ---- 1. build --------------------------------------------------------------
echo "=== build (CUDA_ARCH=$ARCH) ==="
make -j "$(nproc)" all CUDA_ARCH="$ARCH"

# ---- 2. correctness gates --------------------------------------------------
echo "=== unit tests ==="
./bin/test_cbst_smoke
./bin/test_dynamicgraph_roundtrip
./bin/test_snapshot_matches_updateCSR
./bin/test_h2h_construction
./bin/test_h2h_delta
./bin/test_hsosp_matches_dijkstra

echo "=== randomized stress: full pipeline vs Dijkstra ==="
STRESS_CONFIGS=100
[[ "$SUITE" == "smoke" ]] && STRESS_CONFIGS=30
./bin/hsospStress --configs "$STRESS_CONFIGS" --seed "$SEED"

# ---- 3. benchmark matrix ---------------------------------------------------
echo "=== benchmark suite ($SUITE) ==="
./bin/hsospBench --suite "$SUITE" --out "$RESULTS/results.csv" \
    --exp "$EXP" --reps "$REPS" --seed "$SEED"

# ---- 4. figures ------------------------------------------------------------
echo "=== figures ==="
if ! python3 -c "import seaborn, pandas, matplotlib" 2>/dev/null; then
    echo "[deps] installing seaborn/pandas/matplotlib (--user)"
    python3 -m pip install --user --quiet seaborn pandas matplotlib
fi
python3 scripts/plot_results.py "$RESULTS/results.csv" "$RESULTS/figures/"

echo "=== done ==="
date
echo "CSV:     $RESULTS/results.csv"
echo "Figures: $RESULTS/figures/"
echo "Log:     $LOG"
