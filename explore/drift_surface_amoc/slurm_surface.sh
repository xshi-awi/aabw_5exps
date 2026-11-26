#!/bin/bash
#SBATCH --job-name=drift_surf
#SBATCH --account=ba1066
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ba1066/a270064/cc_projects/aabw_5exps/explore/drift_surface_amoc/slurm_surface_%j.out
set -e
cd /work/ba1066/a270064/cc_projects/aabw_5exps/explore/drift_surface_amoc
module load python3/2022.01-gcc-11.2.0
/usr/bin/time -v python3 -s calc_surface_yearly.py
echo SURF_JOB_DONE
