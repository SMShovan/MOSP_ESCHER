#!/usr/bin/env bash
#
# Sync the escher-mosp tree to mill.mst.edu for execution. Run from macOS
# (the development machine). Transfers source only; excludes build output
# and test scratch directories.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

REMOTE_HOST="${ESCHER_MOSP_REMOTE:-sskg8@mill.mst.edu}"
REMOTE_PATH="${ESCHER_MOSP_REMOTE_PATH:-~/escher-mosp/}"

rsync -avz --delete --progress \
    --exclude 'build/' \
    --exclude 'bin/' \
    --exclude 'data/' \
    --exclude 'output/' \
    --exclude 'tests/tmp/' \
    --exclude 'tests/testCase*/' \
    --exclude 'tests/local/local_tests' \
    --exclude 'docs/html/' \
    --exclude 'results/' \
    --exclude '.DS_Store' \
    "$ROOT/" \
    "$REMOTE_HOST:$REMOTE_PATH"
