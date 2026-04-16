#!/usr/bin/env bash
#
# End-to-end test runner for escher-mosp. Intended to be executed on the
# cluster (mill.mst.edu) after transfer, once CUDA is available:
#
#   module load cuda-toolkit/12.5
#   cd ~/escher-mosp
#   ./tests/run_all.sh
#
# Exits non-zero on the first failure so CI can surface problems clearly.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN="$ROOT/bin"

cd "$ROOT"

echo "=== building escher-mosp ==="
make -j all

echo
echo "=== unit tests ==="
"$BIN/test_cbst_smoke"
"$BIN/test_dynamicgraph_roundtrip"
"$BIN/test_snapshot_matches_updateCSR"

echo
echo "=== main pipeline ==="
"$BIN/main"

echo
echo "=== sequential stress test ==="
"$BIN/stressTest"

echo
echo "=== parallel stress test ==="
"$BIN/parallelStressTest"

echo
echo "=== all tests passed ==="
