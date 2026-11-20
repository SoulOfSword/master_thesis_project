#!/bin/bash
#SBATCH -p regular
#SBATCH -J profiles
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/profiles_%j.out
#
# Density profiles of the final sample (scripts/data/compute_profiles.py):
#   sbatch scripts/slurm/compute_profiles.sbatch

set -euo pipefail
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /home2/s4636708/master_thesis_project

python -u scripts/data/compute_profiles.py --ncpu "$SLURM_CPUS_PER_TASK"
