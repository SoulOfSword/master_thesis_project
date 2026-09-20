#!/bin/bash -l
#SBATCH --job-name=asymmetry
#SBATCH --partition=regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=costa@astro.rug.nl
#SBATCH --output=/scratch/s4636708/aida/derived/logs/slurm/asymmetry_%j.out
#
# 3D asymmetry (He+2026) of every galaxy: reads every full cube, too heavy for the
# login node. Writes <processed>/gas_discs/asymmetry_3d.csv and
# figures/disc_quality/asymmetry_split.pdf + asymmetry_all.pdf.
# Takes no variables, so no --export=ALL needed:
#   sbatch scripts/slurm/asymmetry_all.sbatch
# Only after a running cube/fit job has finished:
#   sbatch --dependency=afterany:<jobid> scripts/slurm/asymmetry_all.sbatch

set -u
source "$HOME/software/habrok_env.sh"
cd /home2/s4636708/master_thesis_project

echo "[$(date +%F_%T)] 3D asymmetry, all galaxies"
python -u scripts/plots/disc_quality/plot_asymmetry.py --workers "$SLURM_CPUS_PER_TASK"
echo "[$(date +%F_%T)] done"
