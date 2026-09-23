#!/usr/bin/env python
"""
Glacial mixed layer depth with a tightened colour scale.

Reviewer 1 comment 10 objected that the deep mixed layers are barely visible in
the glacial panels, because the shared 0-400 m scale of the original figure
saturates nothing and leaves the coastal signal indistinguishable from the
background. Plotting LGM and MIS3 on their own 0-300 m scale (the 99th percentile
of the winter field south of 55 S is 280 and 296 m) makes the coastal polynya
cells stand out, which is where the glacial dense water is formed.

The figure shows LGM and MIS3 on a 0-300 m scale with the 200 m contour marked,
over the region south of 60 S where the glacial dense water is formed. The
original 0-400 m version was dropped at Xiaoxu's request, since the point of the
figure is what the tightened scale reveals rather than the comparison of scales.
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

EXPS = ['lgm', 'mis']
LABELS = ['LGM', 'MIS3']
MESH = {'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42})


def circular(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(t), np.cos(t)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5, closed=True), transform=ax.transAxes)


def load(exp):
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc')
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    lon = np.degrees(m['nodes'].isel(n2=0).values)
    d = xr.open_dataset(f'{exp}/MLD1_clim.nc')
    v = [x for x in d.data_vars][0]
    mld = np.abs(d[v].values)[[5, 6, 7]].mean(axis=0)
    m.close(); d.close()
    sel = lat < -58
    gy = np.arange(-78, -57, 0.4)
    gx = np.arange(-180, 180.5, 0.8)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(np.column_stack([lon[sel], lat[sel]]), mld[sel], (GX, GY),
                    method='linear')
    return gx, gy, grid


data = {e: load(e) for e in EXPS}

fig = plt.figure(figsize=(11.5, 6.0))
gs = fig.add_gridspec(1, 2, wspace=0.08,
                      left=0.06, right=0.96, top=0.90, bottom=0.17)
proj = ccrs.SouthPolarStereo()
letters = 'ab'
p = 0

ROWS = [(300, np.arange(0, 301, 20), 'Tightened scale (0--300 m)', 200)]

cfs = []
for r, (vmax, levels, rowlab, contour) in enumerate(ROWS):
    for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
        gx, gy, grid = data[e]
        ax = fig.add_subplot(gs[r, c], projection=proj)
        ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        circular(ax)
        cf = ax.contourf(gx, gy, grid, levels=levels, cmap='YlGnBu',
                         extend='max', transform=ccrs.PlateCarree())
        if contour is not None:
            ax.contour(gx, gy, grid, levels=[contour], colors='#D55E00',
                       linewidths=1.6, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                          alpha=0.5, linestyle='--', x_inline=False, y_inline=True)
        gl.top_labels = gl.right_labels = gl.bottom_labels = False
        gl.ylabel_style = {'size': 9, 'color': '#333333'}
        if r == 0:
            ax.set_title(lab, fontsize=20, fontweight='bold', pad=8)
        ax.text(0.03, 0.97, f'({letters[p]})', transform=ax.transAxes, fontsize=15,
                fontweight='bold', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))
        p += 1
    cfs.append(cf)

cax = fig.add_axes([0.28, 0.085, 0.44, 0.022])
cb = fig.colorbar(cfs[0], cax=cax, orientation='horizontal',
                  ticks=np.arange(0, 301, 75))
cb.set_label('JJA mixed layer depth (m); orange contour = 200 m', fontsize=13)
cb.ax.tick_params(labelsize=11)

fig.savefig('figures/figR7_mld_glacial_zoom.pdf', dpi=400)
fig.savefig('figures/figR7_mld_glacial_zoom.png', dpi=200)
print('saved figures/figR7_mld_glacial_zoom.{pdf,png}')
