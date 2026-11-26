#!/bin/bash
#SBATCH --job-name=drift_moc100
#SBATCH --account=ba1066
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --array=0-4
#SBATCH --output=/work/ba1066/a270064/cc_projects/aabw_5exps/explore/drift_surface_amoc/slurm_moc_100yr_%A_%a.out
set -e
cd /work/ba1066/a270064/cc_projects/aabw_5exps/explore/drift_surface_amoc
EXPS=(pi mh lig lgm mis)
PY=/home/a/a270064/sklearn-env/bin/python
/usr/bin/time -v $PY calc_moc_100yr.py ${EXPS[$SLURM_ARRAY_TASK_ID]}
echo MOC_JOB_DONE
