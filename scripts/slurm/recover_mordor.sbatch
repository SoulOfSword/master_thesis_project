#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J recover_mordor
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm/recover_mordor_%j.out
#
# Recover the stale cross-snapshot MORDOR galaxies (subhalo 0 at z>=1 plus
# a handful of others) after the flat Gal dirs were migrated to per-snap
# dirs. Re-extracts each missing central, runs MORDOR on it, splices the
# fresh row into the per-snap morphology table, then rebuilds the affected
# samples. CDM snap 21 particles are read from the SCRATCH shadow tree
# (handled inside recover_stale_mordor_galaxies.py).

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/master_thesis_project
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

MODELS=(CDM SIDM1 vSIDM)
SNAPS=(17 21 25 33 50)

echo "[$(date +%H:%M:%S)] recover: extract + MORDOR + patch morphology tables"
python scripts/data/recover_stale_mordor_galaxies.py --apply \
    --models "${MODELS[@]}" --snaps "${SNAPS[@]}"

echo "[$(date +%H:%M:%S)] rebuild affected samples from patched tables"
for SNAP in "${SNAPS[@]}"; do
    SNAP3=$(printf "%03d" "$SNAP")
    for MODEL in "${MODELS[@]}"; do
        MORDOR_TXT="$SCRATCH/master_thesis_project/data/mordor_galaxies/outputs/snap_${SNAP3}/morphology_${MODEL}.txt"
        [ -s "$MORDOR_TXT" ] || { echo "skip $MODEL snap $SNAP — no table"; continue; }
        echo "[$(date +%H:%M:%S)] build_mordor_sample $MODEL snap $SNAP"
        python scripts/data/build_mordor_sample.py \
            --model "$MODEL" --snap "$SNAP" --mordor-txt "$MORDOR_TXT"
    done
done

echo "[$(date +%H:%M:%S)] recover_mordor done"
