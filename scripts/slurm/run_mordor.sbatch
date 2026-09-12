#!/bin/bash
#SBATCH -p regular
#SBATCH -J mordor
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=320G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=costa@astro.rug.nl
#SBATCH -o /scratch/s4636708/aida/derived/logs/slurm/mordor_%j.out
#
# Run MORDOR for one (or more) (MODEL, SNAP) combos. Edit MODELS / SNAPS
# below. The script handles MORDOR's own internal parallelism (extract
# + multi-worker batching), so this stays as a single big job per submit.
# All workers are subprocesses of one master -> ONE node (extra nodes would
# sit idle): workers = -c, keep workers x --mem-per-worker within --mem.
#
# Leonardo: ~6 h for all 18 combos with 64 workers on one node; -t > 8 h
# lands in regularmedium. After a run, `jobinfo <jobid>` shows the real
# CPU/memory use -> right-size -c / --mem for the next one.

set -euo pipefail
source "$HOME/software/habrok_env.sh"
mkdir -p "/scratch/s4636708/aida/derived/logs/slurm"

PROJECT=/home2/s4636708/master_thesis_project
cd "$PROJECT"

# Edit to taste:
MODELS=(CDM SIDM1 vSIDM)
SNAPS=(67 50 33 25 21 17)

# Compose the args:
ARGS=(
    --models "${MODELS[@]}"
    --snaps  "${SNAPS[@]}"
    --max-workers "$SLURM_CPUS_PER_TASK"
    --mem-per-worker 4G
)

echo "[$(date +%H:%M:%S)] run_mordor_all.py ${ARGS[*]}"
python scripts/mordor/run_mordor_all.py "${ARGS[@]}"
