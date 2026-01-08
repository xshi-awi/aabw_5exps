
# task

plot antarctic (south of 50s) mean climate (djf/jja) for 5 experiments (pi,mh,lig, lgm,mis)

## data path
/home/a/a270064/bb1029/wmt/exps/{pi,mh,lig,lgm,mis}

## github

https://github.com/xshi-awi/aabw_5exps

## data

a_ice: ice concentration
sst/sss: sea surface temperature/salinity
MLD1: mixed layer depth (only plot jja for mld1)
u10 (var165 in echam_clim.nc): 10m uwinds

saved in <var>.clim.nc in unstructured grid, 12 months.

## grid info

mesh path for pi/mh/lig: /home/a/a270064/bb1029/inputs/mesh_core2
mesh path for LGM: /home/a/a270064/bb1029/inputs/mesh_glac1d
mesh path  for MIS3: /home/a/a270064/bb1029/inputs/mesh_glac1d_38k
fesom.mesh.diag.nc in mesh path contain all needed info. (lat/lon might be in 弧度)

## others

I am writing a paper about AABW in 5 different simulations, in the paper result part, I would like to first present some large feature pattern of the simulated climate variables, for example sst/sss/density/mld/seaice/winds. (density need to be calculated from sst/sss).

please design the plots structure for me, they should be in high quality representing mean climate, and the differences between paleo (mh/lig/lgm/mis3) and pi, 要求高端大气上档次，能够达到顶刊要求。

jja mean and djf mean are enough

sea ice/mld : plot absolute values for each exp

except seaice/mld: plot anomalies in shading covered PI winds in contour lines

i recommend:
sst/sss/density: merged into one figure
mld/seaice: merged into one figure
u10: one figure


## Project Progress

### 2026-01-08 14:30

**Progress**: Git repository initialized and pushed to GitHub; plotting script improved

**Technical Stack**:
- Git repository setup with proper .gitignore for NetCDF and data files
- GitHub remote: https://github.com/xshi-awi/aabw_5exps
- Modified `plot_climate_patterns_jja.py`:
  - Removed all `plt.suptitle()` calls from figures 1-3
  - Added `add_cyclic_point()` function to handle longitude wraparound
  - Applied cyclic point using `np.concatenate()` to extend data/lon arrays
  - Fixed longitude discontinuity in all three figures by adding cyclic points before `ax.contourf()`

**Key Findings**:
- Longitude discontinuity at -180/180° boundary caused visual artifacts in polar stereographic projection
- Solution: append first longitude column to end with +360° offset for seamless wraparound

**Next Steps**:
- Run plotting script to generate updated figures without titles and discontinuities
- Verify visual quality of polar plots
- Consider creating DJF (austral summer) versions if needed
