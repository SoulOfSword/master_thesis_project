#!/bin/bash -l
#SBATCH --job-name=ebump
#SBATCH --account=euhpc_r05_084
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/data/mordor_galaxies/ebump_%A.out

# Energy-bump metrics over a WHOLE snapshot (the --all batch path).
#   sbatch run_ebump.sbatch SIDM1 21      # z=4 SIDM1
#   sbatch run_ebump.sbatch CDM   21      # z=4 CDM
# (single galaxy is interactive WebAgg -> run the .py directly on a node,
#  not through this batch script.)

set -u
export PATH="$HOME/.local/bin:$PATH"        # login shell (-l) already loaded python
cd /leonardo/home/userexternal/acosta01/master_thesis_project

MODEL=${1:?usage: sbatch run_ebump.sbatch MODEL SNAP}
SNAP=${2:?usage: sbatch run_ebump.sbatch MODEL SNAP}

python3 -u scripts/plots/morphology/energy_bump_check.py \
    --all --model "$MODEL" --snap "$SNAP" --ncpu "$SLURM_CPUS_PER_TASK"
