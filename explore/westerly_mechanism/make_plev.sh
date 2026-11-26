#!/bin/bash
# Spectral model-level T, vorticity, divergence in {exp}/echam_clim.nc (100-yr ymonmean)
# -> Gaussian T63 pressure-level T (130), u (131), v (132), plus sea-level pressure.
# Run in the foreground. Output: work/plev_{exp}.nc; work/slp_{exp}.nc (only a cross-check, the analysis uses
# {exp}/100years/slp_mergetime.nc). ml_{exp}.nc is an intermediate and is deleted.
set -e
module load cdo
ROOT=/work/ba1066/a270064/cc_projects/aabw_5exps
OUT=$ROOT/explore/westerly_mechanism/work
mkdir -p $OUT
PLEV=100000,92500,85000,77500,70000,60000,50000,40000,30000,25000,20000,15000,10000
for e in pi mh lig lgm mis; do
  IN=$ROOT/$e/echam_clim.nc
  cdo -s -f nc4 sp2gp -selcode,130,152 $IN $OUT/tmp_t_${e}.nc
  cdo -s -f nc4 dv2uv -selcode,138,155 $IN $OUT/tmp_uv_${e}.nc
  cdo -s -f nc4 selcode,129,134 $IN $OUT/tmp_sfc_${e}.nc
  cdo -s -f nc4 merge $OUT/tmp_t_${e}.nc $OUT/tmp_uv_${e}.nc $OUT/tmp_sfc_${e}.nc $OUT/ml_${e}.nc
  cdo -s -f nc4 ml2pl,$PLEV $OUT/ml_${e}.nc $OUT/plev_${e}.nc
  cdo -s -f nc4 sealevelpressure $OUT/ml_${e}.nc $OUT/slp_${e}.nc
  rm -f $OUT/tmp_t_${e}.nc $OUT/tmp_uv_${e}.nc $OUT/tmp_sfc_${e}.nc $OUT/ml_${e}.nc
  ls -la $OUT/plev_${e}.nc $OUT/slp_${e}.nc
done
