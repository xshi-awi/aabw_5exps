#!/usr/bin/env python
"""
Second SAM figure for Reviewer 3: the spatial mechanism.

Row 1  SAM composite zonal wind stress anomaly (high minus low SAM), JJA
Row 2  SAM composite sea ice concentration anomaly, JJA

Together these show the wind perturbation that drives the Ekman divergence, and
the sea ice response that carries the salt pump, for all five climate states.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
THRESH = 1.2
LAT_MAX = -45

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42})


def circular(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(t), np.cos(t)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5, closed=True), transform=ax.transAxes)


def sam_phases(exp):
    d = xr.open_dataset(f'{exp}/100years/slp_mergetime.nc')
    v = [x for x in d.data_vars if d[x].ndim == 3][0]
    lat = d['lat'].values
    zm = d[v].mean('lon')
    p40 = zm.isel(lat=int(np.argmin(np.abs(lat + 40)))).values
    p65 = zm.isel(lat=int(np.argmin(np.abs(lat + 65)))).values
    p40 = (p40 - p40.mean()) / p40.std()
    p65 = (p65 - p65.mean()) / p65.std()
    sam = (p40 - p65).reshape(-1, 12)[:, 5:8].mean(axis=1)
    d.close()
    m, s = sam.mean(), sam.std()
    return np.where(sam > m + THRESH * s)[0], np.where(sam < m - THRESH * s)[0]


fig = plt.figure(figsize=(21, 9.8))
gs = fig.add_gridspec(2, 5, hspace=0.42, wspace=0.06,
                      left=0.045, right=0.985, top=0.92, bottom=0.15)
proj = ccrs.SouthPolarStereo()
letters = 'abcdefghij'
p = 0
cf1 = cf2 = None

for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    hi, lo = sam_phases(e)

    # ---------------- row 1: zonal wind stress anomaly (regular ECHAM grid)
    d = xr.open_dataset(f'sam_wind/{e}_taux.nc')
    tx = d['var180'].values
    lat = d['lat'].values
    lon = d['lon'].values
    tx = tx.reshape(-1, 12, len(lat), len(lon))[:, 5:8].mean(axis=1)
    anom = tx[hi].mean(axis=0) - tx[lo].mean(axis=0)
    d.close()

    ax = fig.add_subplot(gs[0, c], projection=proj)
    ax.set_extent([-180, 180, -90, LAT_MAX], crs=ccrs.PlateCarree())
    circular(ax)
    lon_c = np.concatenate([lon, [360.0]])
    anom_c = np.concatenate([anom, anom[:, :1]], axis=1)
    cf1 = ax.contourf(lon_c, lat, anom_c, levels=np.arange(-0.12, 0.121, 0.01),
                      cmap='PuOr_r', extend='both', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                      linestyle='--', x_inline=False, y_inline=True)
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.ylabel_style = {'size': 9, 'color': '#333333'}
    ax.set_title(lab, fontsize=20, fontweight='bold', pad=8)
    ax.text(0.03, 0.97, f'({letters[p]})', transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))
    p += 1
    if c == 0:
        ax.text(-0.10, 0.5, 'Zonal wind stress', transform=ax.transAxes, rotation=90,
                va='center', ha='center', fontsize=16, fontweight='bold')

    # ---------------- row 2: sea ice concentration anomaly (FESOM nodes)
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[e]}/fesom.mesh.diag.nc')
    flat = np.degrees(m['nodes'].isel(n2=1).values)
    flon = np.degrees(m['nodes'].isel(n2=0).values)
    ai = xr.open_dataset(f'{e}/100years/a_ice_mergetime.nc')['a_ice'].values
    ai = ai.reshape(-1, 12, ai.shape[-1])[:, 5:8].mean(axis=1)
    dai = ai[hi].mean(axis=0) - ai[lo].mean(axis=0)
    m.close()

    sel = flat < -40
    gy = np.arange(-78, -39, 0.75)
    gx = np.arange(-180, 180.5, 1.0)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(np.column_stack([flon[sel], flat[sel]]), dai[sel], (GX, GY),
                    method='linear')

    ax = fig.add_subplot(gs[1, c], projection=proj)
    ax.set_extent([-180, 180, -90, LAT_MAX], crs=ccrs.PlateCarree())
    circular(ax)
    cf2 = ax.contourf(gx, gy, grid, levels=np.arange(-0.20, 0.201, 0.02),
                      cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                      linestyle='--', x_inline=False, y_inline=True)
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.ylabel_style = {'size': 9, 'color': '#333333'}
    ax.text(0.03, 0.97, f'({letters[p]})', transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))
    p += 1
    if c == 0:
        ax.text(-0.10, 0.5, 'Sea ice concentration', transform=ax.transAxes,
                rotation=90, va='center', ha='center', fontsize=16, fontweight='bold')

cax1 = fig.add_axes([0.20, 0.545, 0.60, 0.014])
cb1 = fig.colorbar(cf1, cax=cax1, orientation='horizontal',
                   ticks=np.arange(-0.12, 0.121, 0.04))
cb1.set_label('Zonal wind stress anomaly, high $-$ low SAM  (N m$^{-2}$)', fontsize=14)
cb1.ax.tick_params(labelsize=12)

cax2 = fig.add_axes([0.20, 0.055, 0.60, 0.015])
cb2 = fig.colorbar(cf2, cax=cax2, orientation='horizontal',
                   ticks=np.arange(-0.2, 0.21, 0.1))
cb2.set_label('Sea ice concentration anomaly, high $-$ low SAM', fontsize=14)
cb2.ax.tick_params(labelsize=12)

fig.savefig('figures/figR5_sam_wind_ice.pdf', dpi=400)
fig.savefig('figures/figR5_sam_wind_ice.png', dpi=200)
print('saved figures/figR5_sam_wind_ice.{pdf,png}')
