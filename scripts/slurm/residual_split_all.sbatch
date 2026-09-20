#!/bin/bash -l
#SBATCH --job-name=residuals_all
#SBATCH --partition=regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=costa@astro.rug.nl
#SBATCH --output=/scratch/s4636708/aida/derived/logs/slurm/residuals_all_%j.out
#
# Residual plots, raw and noise-floor-subtracted, of every galaxy (PV and cube: the
# cube version reads every full cube, too heavy for the login node) and of the
# calibration galaxies. Takes no variables, so no --export=ALL needed:
#   sbatch scripts/slurm/residual_split_all.sbatch

set -u
source "$HOME/software/habrok_env.sh"
cd /home2/s4636708/master_thesis_project

echo "[$(date +%F_%T)] all galaxies, PV and cube"
python -u scripts/plots/disc_quality/plot_residual_split.py --all both --workers "$SLURM_CPUS_PER_TASK"

echo "[$(date +%F_%T)] calibration galaxies"
python -u scripts/plots/disc_quality/plot_residual_split.py

echo "[$(date +%F_%T)] all done"
