#!/bin/bash -l
#SBATCH --job-name=replot
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=costa@astro.rug.nl
#SBATCH --output=/scratch/s4636708/aida/derived/martini/logs/replot_%j.out

set -u
source "$HOME/software/habrok_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # parallel over galaxies, not BLAS
cd /home2/s4636708/master_thesis_project
python3 -u scripts/mock/reprocess_kinematics.py --ncpu "$SLURM_CPUS_PER_TASK"
