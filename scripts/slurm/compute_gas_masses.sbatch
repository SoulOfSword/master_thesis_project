#!/bin/bash
#SBATCH -p regular
#SBATCH -J gasmass
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/gasmass_%j.out
#
# Gas masses by definition, resolved in radius, height and phase
# (scripts/data/compute_gas_masses.py), for the final sample built at run time:
#   sbatch scripts/slurm/compute_gas_masses.sbatch

set -euo pipefail
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /home2/s4636708/master_thesis_project

python -u scripts/data/compute_gas_masses.py --ncpu "$SLURM_CPUS_PER_TASK"
