#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J build_disc_dmo
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/build_disc_dmo_%j.out
#
# Build DMO catalogs matched to FP DISC catalogs (option b of the disc
# workflow). Requires disc FP catalogs from filter_disc_catalogs.sbatch.

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
        FP_CAT="$SCRATCH/master_thesis_project/data/processed/catalogs/catalog_${MODEL}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}_disc.hdf5"
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
