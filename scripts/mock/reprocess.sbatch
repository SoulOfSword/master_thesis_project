#!/bin/bash -l
#SBATCH --job-name=replot
#SBATCH --account=euhpc_r05_084
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/acosta01/master_thesis_project/data/martini/logs/replot_%j.out

set -u
export PATH="$HOME/.local/bin:$PATH"
cd /leonardo/home/userexternal/acosta01/master_thesis_project
python3 -u scripts/mock/reprocess_kinematics.py --ncpu "$SLURM_CPUS_PER_TASK"
