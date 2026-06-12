#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J mordor_samples
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/mordor_samples_%j.out
#
# For each (MODEL, SNAP) combo: parse MORDOR ASCII output + join with the
# halo catalog + compute the interacting flag, then save one HDF5 sample
# per combo. One dcgp node, combos run in parallel inside the job.

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/master_thesis_project
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)

N_PARALLEL=16

for MODEL in "${MODELS[@]}"; do
    for SNAP in "${SNAPS[@]}"; do
        SNAP3=$(printf "%03d" "$SNAP")
        MORDOR_TXT="$SCRATCH/master_thesis_project/data/mordor_galaxies/outputs/snap_${SNAP3}/morphology_${MODEL}.txt"
        if [ ! -s "$MORDOR_TXT" ]; then
            echo "[$(date +%H:%M:%S)] skip ${MODEL} snap $SNAP — no MORDOR output at $MORDOR_TXT"
            continue
        fi
        echo "[$(date +%H:%M:%S)] dispatch build_mordor_sample  $MODEL snap $SNAP"
        python scripts/data/build_mordor_sample.py \
            --model "$MODEL" --snap "$SNAP" --mordor-txt "$MORDOR_TXT" \
            > "$LOGS_DIR/mordor_sample_${SLURM_JOB_ID}_${MODEL}_${SNAP}.out" 2>&1 &
        while [ "$(jobs -r -p | wc -l)" -ge $N_PARALLEL ]; do sleep 0.5; done
    done
done
wait

echo "[$(date +%H:%M:%S)] build_mordor_samples all done"
