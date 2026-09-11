#!/bin/bash
#SBATCH -p regular
#SBATCH -J filter_disc
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18
#SBATCH --mem=32G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/filter_disc_%j.out
#
# Filter FP catalogs down to MORDOR-disc galaxies. Requires FP catalogs
# (from build_catalogs.sbatch) AND mordor samples (from
# build_mordor_samples.sbatch) to exist beforehand.

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

MSTAR_TAG=1e+08
NDM_TAG=0

N_PARALLEL=$SLURM_CPUS_PER_TASK

for MODEL in "${MODELS[@]}"; do
    for SNAP in "${SNAPS[@]}"; do
        SNAP3=$(printf "%03d" "$SNAP")
        FP_CAT="$DATA/processed/catalogs/catalog_${MODEL}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}.hdf5"
        MORDOR_SAMPLE="$DATA/mordor_galaxies/samples/mordor_sample_${MODEL}_${SNAP3}.hdf5"
        if [ ! -f "$FP_CAT" ]; then
            echo "[$(date +%H:%M:%S)] skip $MODEL snap $SNAP — no FP catalog"
            continue
        fi
        if [ ! -f "$MORDOR_SAMPLE" ]; then
            echo "[$(date +%H:%M:%S)] skip $MODEL snap $SNAP — no MORDOR sample"
            continue
        fi
        echo "[$(date +%H:%M:%S)] dispatch filter_disc_catalog $MODEL snap $SNAP"
        python scripts/data/filter_disc_catalog.py \
            --catalog "$FP_CAT" --mordor-sample "$MORDOR_SAMPLE" \
            > "$LOGS_DIR/filter_disc_${SLURM_JOB_ID}_${MODEL}_${SNAP}.out" 2>&1 &
        while [ "$(jobs -r -p | wc -l)" -ge $N_PARALLEL ]; do sleep 0.5; done
    done
done
wait

echo "[$(date +%H:%M:%S)] filter_disc_catalogs all done"
