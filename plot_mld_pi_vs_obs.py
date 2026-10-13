#!/usr/bin/env python
"""
Pre-industrial winter mixed layer depth against an observational estimate.

Reviewer 1 asked how well the model represents the present-day Southern Ocean,
and Jiping asked for this figure to drop the paleoclimate states and show PI
beside observations instead. Two panels, nothing else: no contours, no per-panel
annotation, and a colour bar labelled only "JJA mixed layer depth (m)".

The observational side is ORAS5, the ECMWF ocean reanalysis, JJA climatology
over 1990-2009, ensemble member opa0, from the DKRZ pool. ORAS5 is the closest
thing to an observational mixed layer field for this region: it assimilates the
same hydrography the model should be judged against, and unlike the Copernicus
products it actually carries a mixed layer diagnostic.

One caveat is stated in the caption rather than buried: the two use different
definitions. ORAS5 uses a 0.01 kg/m3 density threshold, our model a bulk
Richardson number criterion, so a point-by-point difference is not purely model
error.
"""
import glob
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

ORAS5 = '/pool/data/ICDC/ocean_syntheses/oras5/ORCA025/somxl010/opa0'
YEARS = range(1990, 2010)
MESH = '/home/a/a270064/bb1029/inputs/mesh_core2/fesom.mesh.diag.nc'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42})

GX = np.arange(-180, 180.5, 0.8)
GY = np.arange(-78, -57, 0.4)
MGX, MGY = np.meshgrid(GX, GY)


def circular(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(t), np.cos(t)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5, closed=True), transform=ax.transAxes)


def model_pi():
    m = xr.open_dataset(MESH)
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    lon = np.degrees(m['nodes'].isel(n2=0).values)
    m.close()
    d = xr.open_dataset('pi/MLD1_clim.nc')
    v = [x for x in d.data_vars][0]
    mld = np.abs(d[v].values)[[5, 6, 7]].mean(axis=0)     # JJA
    d.close()
    sel = lat < -58
    return griddata(np.column_stack([lon[sel], lat[sel]]), mld[sel],
                    (MGX, MGY), method='linear')


def oras5_jja():
    files = []
    for y in YEARS:
        for mo in ('06', '07', '08'):
            f = f'{ORAS5}/somxl010_ORAS5_1m_{y}{mo}_grid_T_02.nc'
            files += glob.glob(f)
    if not files:
        raise SystemExit('no ORAS5 files found under ' + ORAS5)
    print('averaging %d ORAS5 JJA months' % len(files))
    acc = None
    for f in files:
        d = xr.open_dataset(f, decode_times=False)
        a = d['somxl010'].isel(time_counter=0).values
        a = np.where(np.isfinite(a) & (a > 0) & (a < 1e4), a, np.nan)
        acc = a if acc is None else np.nansum(np.dstack([acc, a]), axis=2)
        if 'lat' not in dir():
            lat = d['nav_lat'].values
            lon = d['nav_lon'].values
        d.close()
    mean = acc / len(files)
    sel = (lat < -57) & np.isfinite(mean)
    return griddata(np.column_stack([lon[sel], lat[sel]]), mean[sel],
                    (MGX, MGY), method='linear')


panels = [('(a)  AWI-ESM2, pre-industrial', model_pi()),
          ('(b)  ORAS5 reanalysis, 1990-2009', oras5_jja())]

fig = plt.figure(figsize=(11.6, 6.3))
gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.05, right=0.96,
                      top=0.92, bottom=0.17)
proj = ccrs.SouthPolarStereo()
levels = np.arange(0, 801, 50)

for c, (title, grid) in enumerate(panels):
    ax = fig.add_subplot(gs[0, c], projection=proj)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    circular(ax)
    cf = ax.contourf(GX, GY, grid, levels=levels, cmap='YlGnBu',
                     extend='max', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--', x_inline=False, y_inline=True)
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.ylabel_style = {'size': 9, 'color': '#333333'}
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)

cax = fig.add_axes([0.30, 0.095, 0.40, 0.028])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                  ticks=np.arange(0, 801, 200))
cb.set_label('JJA mixed layer depth (m)', fontsize=13)
cb.ax.tick_params(labelsize=11)

fig.savefig('figures/figR17_mld_pi_vs_obs.pdf', dpi=400)
fig.savefig('figures/figR17_mld_pi_vs_obs.png', dpi=200)
print('saved figures/figR17_mld_pi_vs_obs.{pdf,png}')

# numbers for the caption: where they agree and where they do not
mod, obs = panels[0][1], panels[1][1]
ok = np.isfinite(mod) & np.isfinite(obs)
w = np.cos(np.radians(MGY))[ok]
print('\narea-weighted JJA mean south of 60S:')
print('  model %.0f m, ORAS5 %.0f m' % (np.average(mod[ok], weights=w),
                                        np.average(obs[ok], weights=w)))
for lab, thr in [('>400 m', 400), ('>600 m', 600)]:
    fm = np.average((mod[ok] > thr).astype(float), weights=w) * 100
    fo = np.average((obs[ok] > thr).astype(float), weights=w) * 100
    print('  area %s : model %.1f%%, ORAS5 %.1f%%' % (lab, fm, fo))
# Weddell sector, where the model convects
wed = ok & (MGX >= -60) & (MGX <= 30)
ww = np.cos(np.radians(MGY))[wed]
print('  Weddell 60W-30E mean: model %.0f m, ORAS5 %.0f m'
      % (np.average(mod[wed], weights=ww), np.average(obs[wed], weights=ww)))
