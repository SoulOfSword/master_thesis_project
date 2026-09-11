#!/bin/bash
# Submit the density-mosaic chain on Habrok (partition `regular`). Uses
# `sbatch --wait` so each stage blocks until the previous one finishes.
# The DMO stages only do something once the DMO data is on Habrok; without
# it they skip/fail per combo and the chain carries on (FP-only).
#
# Usage (run in tmux so SSH drops don't kill it):
#   tmux new -s pipeline
#   bash scripts/slurm/pipeline_density.sh
#   # detach with Ctrl-b d; reattach with `tmux attach -t pipeline`

set -euo pipefail

cd "$(dirname "$0")"

echo "[$(date +%H:%M:%S)] [1/5] build_catalogs..."
sbatch --wait build_catalogs.sbatch

echo "[$(date +%H:%M:%S)] [2/5] build_catalogs_dmo..."
sbatch --wait build_catalogs_dmo.sbatch

echo "[$(date +%H:%M:%S)] [3/5] compute_profiles..."
sbatch --wait compute_profiles.sbatch

echo "[$(date +%H:%M:%S)] [4/5] compute_gamma..."
sbatch --wait compute_gamma.sbatch

echo "[$(date +%H:%M:%S)] [5/5] compute_rcore..."
sbatch --wait compute_rcore.sbatch

echo
echo "[$(date +%H:%M:%S)] pipeline_density: all stages complete."
