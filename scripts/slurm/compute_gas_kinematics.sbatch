#!/bin/bash
#SBATCH -p regular
#SBATCH -J gaskin
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/gaskin_%j.out
#
# Neutral-gas kinematics of the MORDOR discs for the gas-disc selection
# (scripts/data/compute_gas_kinematics.py), for the mock manifest:
#   sbatch scripts/slurm/compute_gas_kinematics.sbatch
# or for a list of galaxies (one "model snap subID" per line):
#   GALAXIES=/path/to/list.txt sbatch --export=ALL scripts/slurm/compute_gas_kinematics.sbatch
# Needs the v_circ files of the same galaxies (compute_vcirc.sbatch) first.

set -euo pipefail
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /home2/s4636708/master_thesis_project

python -u scripts/data/compute_gas_kinematics.py ${GALAXIES:+--galaxies "$GALAXIES"} --ncpu "$SLURM_CPUS_PER_TASK"
