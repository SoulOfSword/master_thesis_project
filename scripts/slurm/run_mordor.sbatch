#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J mordor
#SBATCH -t 06:00:00
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/mordor_%j.out
#
# Run MORDOR for one (or more) (MODEL, SNAP) combos. Edit MODELS / SNAPS
# below. The script handles MORDOR's own internal parallelism (extract
# + multi-worker batching), so this stays as a single big job per submit.
#
# For CDM snap 21 specifically you also need --base-path (because the
# CDM snapdir_021 lives in $SCRATCH, not $WORK). Edit the line marked
# BASE_PATH below.

set -euo pipefail
mkdir -p "/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm"

PROJECT=/leonardo/home/userexternal/acosta01/master_thesis_project
cd "$PROJECT"

# Edit to taste:
MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)
# BASE_PATH=$SCRATCH/master_thesis_project/data/aida_tng/L35n1080_CDM/output    # only needed for CDM snap 21

# Compose the args:
ARGS=(
    --models "${MODELS[@]}"
    --snaps  "${SNAPS[@]}"
    --max-workers 64
    --mem-per-worker 4G
)
# Uncomment if running CDM with the shadow tree:
ARGS+=(--base-path "$BASE_PATH")

echo "[$(date +%H:%M:%S)] run_mordor_all.py ${ARGS[*]}"
python scripts/mordor/run_mordor_all.py "${ARGS[@]}"
