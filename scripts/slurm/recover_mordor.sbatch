#!/bin/bash
#SBATCH -p regular
#SBATCH -J recover_mordor
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=costa@astro.rug.nl
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/recover_mordor_%j.out
#
# Recover the stale cross-snapshot MORDOR galaxies (subhalo 0 at z>=1 plus
# a handful of others) after the flat Gal dirs were migrated to per-snap
# dirs. Re-extracts each missing central, runs MORDOR on it, splices the
# fresh row into the per-snap morphology table, then rebuilds the affected
# samples. Galaxies are processed one at a time (sequential), but they are
# the most massive centrals -> modest cores, generous memory.

set -euo pipefail
source "$HOME/software/habrok_env.sh"

PROJECT=/home2/s4636708/master_thesis_project
DATA=/scratch/s4636708/aida/derived
LOGS_DIR="$DATA/logs/slurm"
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
        MORDOR_TXT="$DATA/mordor_galaxies/outputs/snap_${SNAP3}/morphology_${MODEL}.txt"
        [ -s "$MORDOR_TXT" ] || { echo "skip $MODEL snap $SNAP — no table"; continue; }
        echo "[$(date +%H:%M:%S)] build_mordor_sample $MODEL snap $SNAP"
        python scripts/data/build_mordor_sample.py \
            --model "$MODEL" --snap "$SNAP" --mordor-txt "$MORDOR_TXT"
    done
done

echo "[$(date +%H:%M:%S)] recover_mordor done"
