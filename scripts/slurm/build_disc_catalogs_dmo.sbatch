#!/bin/bash
#SBATCH -p regular
#SBATCH -J build_disc_dmo
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/build_disc_dmo_%j.out
#
# Build DMO catalogs matched to FP DISC catalogs (option b of the disc
# workflow). Requires disc FP catalogs from filter_disc_catalogs.sbatch, and
# the DMO group catalogs + SubhaloMatchingToDark on Habrok.

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
        FP_CAT="$DATA/processed/catalogs/catalog_${MODEL}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}_disc.hdf5"
        if [ ! -f "$FP_CAT" ]; then
            echo "[$(date +%H:%M:%S)] skip $MODEL snap $SNAP — no disc FP catalog"
            continue
        fi
        echo "[$(date +%H:%M:%S)] dispatch build_catalog_dmo (disc) $MODEL snap $SNAP"
        python scripts/data/build_catalog_dmo.py --fp-catalog "$FP_CAT" \
            > "$LOGS_DIR/build_disc_dmo_${SLURM_JOB_ID}_${MODEL}_${SNAP}.out" 2>&1 &
        while [ "$(jobs -r -p | wc -l)" -ge $N_PARALLEL ]; do sleep 0.5; done
    done
done
wait

echo "[$(date +%H:%M:%S)] build_disc_catalogs_dmo all done"
