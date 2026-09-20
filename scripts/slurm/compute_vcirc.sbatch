#!/bin/bash
#SBATCH -p regular
#SBATCH -J vcirc
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/vcirc_%j.out
#
# Disc-plane circular-velocity profiles (scripts/data/compute_vcirc.py) for a
# list of galaxies -- the final sample, discs in both gas and stars. Pass the
# list (one "model snap subID" per line) on submission:
#   GALAXIES=/path/to/discs.txt sbatch scripts/slurm/compute_vcirc.sbatch

set -euo pipefail
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PROJECT=/home2/s4636708/master_thesis_project
: "${GALAXIES:?set GALAXIES=/path/to/list.txt when submitting}"
cd "$PROJECT"

python -u scripts/data/compute_vcirc.py --galaxies "$GALAXIES" --ncpu "$SLURM_CPUS_PER_TASK"
