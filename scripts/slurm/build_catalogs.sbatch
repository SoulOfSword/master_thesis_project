#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J build_cat
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/build_cat_%j.out
#
# Build FP catalogs for every (MODEL, SNAP) combo. One dcgp node, runs
# combos in parallel inside the job.

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/master_thesis_project
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)

N_PARALLEL=32

for MODEL in "${MODELS[@]}"; do
    for SNAP in "${SNAPS[@]}"; do
        echo "[$(date +%H:%M:%S)] dispatch build_catalog  MODEL=$MODEL  SNAP=$SNAP"
        python scripts/data/build_catalog.py --model "$MODEL" --snap "$SNAP" \
            > "$LOGS_DIR/build_cat_${SLURM_JOB_ID}_${MODEL}_${SNAP}.out" 2>&1 &
        while [ "$(jobs -r -p | wc -l)" -ge $N_PARALLEL ]; do sleep 0.5; done
    done
done
wait

echo "[$(date +%H:%M:%S)] build_catalogs all done"
