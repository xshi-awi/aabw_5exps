#!/bin/bash
#SBATCH --job-name=aabw_d2_xcheck
#SBATCH --account=ba1066
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ba1066/a270064/cc_projects/aabw_5exps/data/slurm_crosscheck_%j.out
set -e
cd /work/ba1066/a270064/cc_projects/aabw_5exps/data
/usr/bin/time -v /home/a/a270064/sklearn-env/bin/python crosscheck_formation_climdiag.py
echo XCHECK_JOB_DONE
