#!/bin/bash
#SBATCH -p regular
#SBATCH -J mordor_samples
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/mordor_samples_%j.out
#
# For each (MODEL, SNAP) combo: parse MORDOR ASCII output + join with the
# halo catalog + compute the interacting flag, then save one HDF5 sample
# per combo. Combos run in parallel inside the job.

set -euo pipefail
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PROJECT=/home2/s4636708/master_thesis_project
DATA=/scratch/s4636708/aida/derived
LOGS_DIR="$DATA/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)

N_PARALLEL=$SLURM_CPUS_PER_TASK

for MODEL in "${MODELS[@]}"; do
    for SNAP in "${SNAPS[@]}"; do
        SNAP3=$(printf "%03d" "$SNAP")
        MORDOR_TXT="$DATA/mordor_galaxies/outputs/snap_${SNAP3}/morphology_${MODEL}.txt"
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
