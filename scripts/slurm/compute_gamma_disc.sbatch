#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J gamma_disc
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/gamma_disc_%j.out
#
# gamma_DM per DISC catalog (FP + DMO). One dcgp node, combos in
# parallel.

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/master_thesis_project
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)
KINDS=("" "-Dark")

MSTAR_TAG=1e+08
NDM_TAG=0

R_OUTER_KIND=nfw_rs

N_PARALLEL=16

for MODEL in "${MODELS[@]}"; do
    for KIND in "${KINDS[@]}"; do
        for SNAP in "${SNAPS[@]}"; do
            SNAP3=$(printf "%03d" "$SNAP")
            CAT="$SCRATCH/master_thesis_project/data/processed/catalogs/catalog_${MODEL}${KIND}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}_disc.hdf5"
            PROF=$(ls -1 "$SCRATCH/master_thesis_project/data/processed/profiles/profiles_${MODEL}${KIND}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}"*_disc.hdf5 2>/dev/null | head -n1 || true)
            if [ ! -f "$CAT" ] || [ -z "${PROF:-}" ] || [ ! -f "${PROF}" ]; then
                echo "[$(date +%H:%M:%S)] skip ${MODEL}${KIND} snap $SNAP — missing inputs"
                continue
            fi
            echo "[$(date +%H:%M:%S)] dispatch compute_gamma (disc) $CAT"
            python scripts/data/compute_gamma.py --catalog "$CAT" --profiles "$PROF" \
                --r-outer-kind "$R_OUTER_KIND" \
                > "$LOGS_DIR/gamma_disc_${SLURM_JOB_ID}_${MODEL}${KIND}_${SNAP}.out" 2>&1 &
            while [ "$(jobs -r -p | wc -l)" -ge $N_PARALLEL ]; do sleep 0.5; done
        done
    done
done
wait

echo "[$(date +%H:%M:%S)] compute_gamma_disc all done"
