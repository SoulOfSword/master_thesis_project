#!/bin/bash -l
#SBATCH --job-name=ebump
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s4636708/aida/derived/mordor_galaxies/ebump_%A.out

# Energy-bump metrics over a WHOLE snapshot (the --all batch path).
#   sbatch run_ebump.sbatch SIDM1 21      # z=4 SIDM1
#   sbatch run_ebump.sbatch CDM   21      # z=4 CDM
# (single galaxy is interactive WebAgg -> run the .py directly on a node,
#  not through this batch script.)
# One MORDOR decomposition per worker; big centrals take several GB each.

set -u
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # parallel over galaxies, not BLAS
cd /home2/s4636708/master_thesis_project

MODEL=${1:?usage: sbatch run_ebump.sbatch MODEL SNAP}
SNAP=${2:?usage: sbatch run_ebump.sbatch MODEL SNAP}

python3 -u scripts/plots/morphology/energy_bump_check.py \
    --all --model "$MODEL" --snap "$SNAP" --ncpu "$SLURM_CPUS_PER_TASK"
