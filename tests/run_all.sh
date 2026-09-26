#!/usr/bin/env bash
#
# Build escher-mosp and run the whole test suite (same as `make test`).
# Exits non-zero on the first build error or on any failed test case.
#
#   ./tests/run_all.sh                 # default CUDA_ARCH (sm_86)
#   CUDA_ARCH=sm_80 ./tests/run_all.sh # e.g. A100

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
make -j "$(nproc)" all ${CUDA_ARCH:+CUDA_ARCH="$CUDA_ARCH"}
./tests/run_tests.sh
