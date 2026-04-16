#!/usr/bin/env bash
#
# Build escher-mosp on mill.mst.edu. Run after sync_to_cluster.sh.

set -euo pipefail

module load cuda-toolkit/12.5
cd ~/escher-mosp
make clean
make -j all
