#!/bin/bash
#SBATCH --job-name=aabw_moc_geo
#SBATCH --account=ba1066
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/work/ba1066/a270064/cc_projects/aabw_5exps/data_moc/slurm_moc_%j.out
set -e
cd /work/ba1066/a270064/cc_projects/aabw_5exps/data_moc
PY=/home/a/a270064/sklearn-env/bin/python
[ "$1" = verify ] || /usr/bin/time -v $PY calc_moc_5exps.py
$PY verify_moc_binning.py
echo MOC_JOB_DONE
