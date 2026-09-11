#!/bin/bash
#SBATCH -p regular
#SBATCH -J profiles_disc
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18
#SBATCH --mem=64G
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/profiles_disc_%j.out
#
# Slice Despali profile catalogs per DISC catalog (FP + DMO). Combos in
# parallel inside the job.
# Needs <run>/postprocessing/cat_halo_profiles_<snap>.hdf5 on Habrok.

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
KINDS=("" "-Dark")               # FP + DMO

MSTAR_TAG=1e+08
NDM_TAG=0

N_PARALLEL=$SLURM_CPUS_PER_TASK

for MODEL in "${MODELS[@]}"; do
    for KIND in "${KINDS[@]}"; do
        for SNAP in "${SNAPS[@]}"; do
            SNAP3=$(printf "%03d" "$SNAP")
            CAT="$DATA/processed/catalogs/catalog_${MODEL}${KIND}_${SNAP3}_mstar${MSTAR_TAG}_ndm${NDM_TAG}_disc.hdf5"
            if [ ! -f "$CAT" ]; then
                echo "[$(date +%H:%M:%S)] skip ${MODEL}${KIND} snap $SNAP — no disc catalog"
                continue
            fi
            echo "[$(date +%H:%M:%S)] dispatch compute_profiles (disc) $CAT"
            python scripts/data/compute_profiles.py --catalog "$CAT" \
                > "$LOGS_DIR/profiles_disc_${SLURM_JOB_ID}_${MODEL}${KIND}_${SNAP}.out" 2>&1 &
            while [ "$(jobs -r -p | wc -l)" -ge $N_PARALLEL ]; do sleep 0.5; done
        done
    done
done
wait

echo "[$(date +%H:%M:%S)] compute_profiles_disc all done"
