#!/usr/bin/env bash
#
# Build escher-mosp on mill.mst.edu. Run after sync_to_cluster.sh.

set -euo pipefail

module load cuda-toolkit/12.5
cd ~/escher-mosp
make clean
# mill has V100 (sm_70) nodes; the Makefile default is sm_86.
make -j all CUDA_ARCH=${CUDA_ARCH:-sm_70}
