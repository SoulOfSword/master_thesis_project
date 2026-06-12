#!/bin/bash -l
#SBATCH --job-name=mockpipe
#SBATCH --account=euhpc_r05_084
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-423%20
#SBATCH --output=/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/data/martini/logs/mock_%A_%a.out

# login shell (-l) pulls in module load python/3.11.7 via ~/.bashrc
set -u
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

PROJ=/leonardo/home/userexternal/acosta01/master_thesis_project
MANIFEST="${MANIFEST:-/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/data/martini/manifest.txt}"
CHUNK="${CHUNK:-10}"
cd "$PROJ"

start=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
end=$(( start + CHUNK - 1 ))
echo "[$(date +%F_%T)] task=$SLURM_ARRAY_TASK_ID host=$(hostname) lines=${start}..${end} chunk=$CHUNK"

sed -n "${start},${end}p" "$MANIFEST" | while read -r model snap sub; do
    [ -z "${model:-}" ] && continue
    echo "--- $(date +%T) $model snap=$snap sub=$sub ---"
    python3 -u scripts/mock/build_galaxy.py \
        --model "$model" --snap "$snap" --sub-id "$sub" \
        --ncpu "$SLURM_CPUS_PER_TASK" --skip-existing \
        || echo "!!! FAILED $model snap=$snap sub=$sub rc=$?"
done
echo "[$(date +%F_%T)] task=$SLURM_ARRAY_TASK_ID DONE"
