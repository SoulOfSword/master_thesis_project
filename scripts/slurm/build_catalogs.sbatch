#!/bin/bash
#SBATCH -p regular
#SBATCH -J build_cat
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/build_cat_%j.out
#
# Build FP catalogs for every (MODEL, SNAP) combo. One single-threaded
# python per combo, run in parallel inside the job (-c = one core per combo).

set -euo pipefail
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PROJECT=/home2/s4636708/master_thesis_project
LOGS_DIR="/scratch/s4636708/aida/derived/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)

N_PARALLEL=$SLURM_CPUS_PER_TASK

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
