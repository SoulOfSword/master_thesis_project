#!/bin/bash
# Submit the disc-only data chain in sequence (one job at a time).
# Requires: regular FP catalogs already built (build_catalogs.sbatch) AND
# MORDOR samples already built (build_mordor_samples.sbatch).
#
# Run inside tmux:
#   tmux new -s pipeline_disc
#   bash scripts/slurm/pipeline_disc.sh

set -euo pipefail

cd "$(dirname "$0")"

echo "[$(date +%H:%M:%S)] [1/5] filter_disc_catalogs..."
sbatch --wait filter_disc_catalogs.sbatch

echo "[$(date +%H:%M:%S)] [2/5] build_disc_catalogs_dmo..."
sbatch --wait build_disc_catalogs_dmo.sbatch

echo "[$(date +%H:%M:%S)] [3/5] compute_profiles_disc..."
sbatch --wait compute_profiles_disc.sbatch

echo "[$(date +%H:%M:%S)] [4/5] compute_gamma_disc..."
sbatch --wait compute_gamma_disc.sbatch

echo "[$(date +%H:%M:%S)] [5/5] compute_rcore_disc..."
sbatch --wait compute_rcore_disc.sbatch

echo
echo "[$(date +%H:%M:%S)] pipeline_disc complete."
