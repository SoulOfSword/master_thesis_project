#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J filter_disc
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/filter_disc_%j.out
#
# Filter FP catalogs down to MORDOR-disc galaxies. Requires FP catalogs
# (from build_catalogs.sbatch) AND mordor samples (from
# build_mordor_samples.sbatch) to exist beforehand.

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/master_thesis_project
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)

MSTAR_TAG=1e+08
NDM_TAG=0

N_PARALLEL=16

for MODEL in "${MODELS[@]}"; do
    for SNAP in "${SNAPS[@]}"; do
        SNAP3=$(printf "%03d" "$SNAP")
        FP_CAT="$SCRATCH/master_thesis_project/data/processed/catalogs/catalog_${MODEL}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}.hdf5"
        MORDOR_SAMPLE="$SCRATCH/master_thesis_project/data/mordor_galaxies/samples/mordor_sample_${MODEL}_${SNAP3}.hdf5"
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
