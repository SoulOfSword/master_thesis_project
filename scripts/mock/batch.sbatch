#!/bin/bash -l
#SBATCH --job-name=mockpipe
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-423%100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=costa@astro.rug.nl
#SBATCH --output=/scratch/s4636708/aida/derived/martini/logs/mock_%A_%a.out

# One array task = CHUNK manifest lines (galaxies), run sequentially.
#   array range = 0 .. ceil(n_lines / CHUNK) - 1   (4237 lines, CHUNK=10 -> 0-423)
#   Habrok caps array indices at 1000: raise CHUNK if the manifest grows past ~10k.
#   %100 = at most 100 tasks (800 cores) running at once; override on submit:
#     sbatch --array=0-423%50 scripts/mock/batch.sbatch
# Independent tasks -> an array on `regular` schedules into gaps; a multi-node
# `parallel` job would have to wait for all its nodes at once.
# Sizing (Leonardo, 8 cpus): 10 galaxies = median 0.35 h, max 1.4 h, no OOM.
# 6 h leaves room for the noiseless+noisy cubes / extended-ring refits and
# stays in regularshort (<= 8 h, quickest to start).

set -u
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

PROJ=/home2/s4636708/master_thesis_project
MANIFEST="${MANIFEST:-/scratch/s4636708/aida/derived/martini/manifest.txt}"
CHUNK="${CHUNK:-10}"
# Stages: sphview, cube, barolo_3rings, barolo (final fit), kinematics.
# For now everything up to the 3-ring fit; the final fit later with
#   STAGES=barolo,kinematics sbatch scripts/mock/batch.sbatch
STAGES="${STAGES:-sphview,cube,barolo_3rings}"
cd "$PROJ"

start=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
end=$(( start + CHUNK - 1 ))
echo "[$(date +%F_%T)] task=$SLURM_ARRAY_TASK_ID host=$(hostname) lines=${start}..${end} chunk=$CHUNK"

# removed --skip-existing
sed -n "${start},${end}p" "$MANIFEST" | while read -r model snap sub; do
    [ -z "${model:-}" ] && continue
    echo "--- $(date +%T) $model snap=$snap sub=$sub ---"
    python3 -u scripts/mock/build_galaxy.py \
        --model "$model" --snap "$snap" --sub-id "$sub" \
        --ncpu "$SLURM_CPUS_PER_TASK" --stages "$STAGES" \
        || echo "!!! FAILED $model snap=$snap sub=$sub rc=$?"
done
echo "[$(date +%F_%T)] task=$SLURM_ARRAY_TASK_ID DONE"
