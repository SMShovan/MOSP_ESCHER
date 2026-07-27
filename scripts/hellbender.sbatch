#!/bin/bash
#
# SLURM wrapper for the H-SOSP pipeline on Hellbender (Mizzou RNet).
#
#   sbatch scripts/hellbender.sbatch smoke    # ~10-20 min sanity pass
#   sbatch scripts/hellbender.sbatch full     # the paper run
#
# Adjust --partition / --account / --gres to your allocation if needed
# (check `sinfo -o "%P %G %l"` and your usual submission flags).
#
#SBATCH --job-name=hsosp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_%j.log

set -euo pipefail

SUITE="${1:-full}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p results

echo "Host: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null || echo 'n/a')"

# Hellbender exposes CUDA as the `cuda` module.
if command -v module >/dev/null 2>&1 && ! command -v nvcc >/dev/null 2>&1; then
    module load cuda 2>/dev/null || module load cuda-toolkit/12.5 2>/dev/null || true
fi

CUDA_ARCH="${CUDA_ARCH:-sm_80}" ./scripts/run_experiments.sh "$SUITE"
