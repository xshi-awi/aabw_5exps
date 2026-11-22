#!/bin/bash
#SBATCH --job-name=osisaf_clim
#SBATCH --partition=shared
#SBATCH --account=ba1066
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24GB
#SBATCH --time=00:40:00
#SBATCH --output=/work/ba1066/a270064/cc_projects/aabw_5exps/fig_seaice_obs/prep_osisaf_%j.log
# OSI SAF CDR v3.0 SH daily -> monthly 1979-2008 (about 9,400 files)
set -e
module load python3/2022.01-gcc-11.2.0
cd /work/ba1066/a270064/cc_projects/aabw_5exps
python fig_seaice_obs/prep_osisaf_clim.py
