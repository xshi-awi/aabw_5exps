#!/usr/bin/env python
"""
Glacial mixed layer depth with a tightened colour scale.

Reviewer 1 comment 10 objected that the deep mixed layers are barely visible in
the glacial panels, because the shared 0-400 m scale of the original figure
saturates nothing and leaves the coastal signal indistinguishable from the
background. Plotting LGM and MIS3 on their own 0-300 m scale (the 99th percentile
of the winter field south of 55 S is 280 and 296 m) makes the coastal polynya
cells stand out, which is where the glacial dense water is formed.

The figure shows LGM and MIS3 on a 0-300 m scale over the region south of 60 S
where the glacial dense water is formed. The 200 m contour and the long colour
bar label were dropped at Xiaoxu's request, matching the other letter figures. The
original 0-400 m version was dropped at Xiaoxu's request, since the point of the
figure is what the tightened scale reveals rather than the comparison of scales.

2026-11-27: text enlarged for an A4 printout of the reply letter, where the figure
prints 5.81 in wide. Font sizes are given as printed points through pt(); the
figure is smaller (7.4 in) so lines print thicker too. Data and colours unchanged.
Previous version: figures/versions/figR7_mld_glacial_zoom_v20261127pre.*
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter
from scipy.interpolate import griddata

EXPS = ['lgm', 'mis']
LABELS = ['LGM', 'MIS3']
MESH = {'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42})

FIG_W, FIG_H = 7.4, 4.0
SCALE = 5.81 / FIG_W          # printed width in the letter / saved width


def pt(p):
    """native size that prints as p points in the letter"""
    return p / SCALE


class LatEvery10(LatitudeFormatter):
    """Inline latitude labels sit at 150E, where the 65S and 75S labels would
    cover coastal cells; only 70S and 80S are labelled. All gridlines stay."""
    def __call__(self, value, pos=None):
        return super().__call__(value, pos) if round(value) % 10 == 0 else ''


def regrid_closed(lon, lat, val, GX, GY):
    """griddata on (GX, GY), then close the seam at 180 deg (2026-11-26).
    Only the grid columns with |lon| > 178.5 are filled, from an interpolation
    in which points within 10 deg of the dateline are repeated at lon -/+ 360,
    and only at latitudes where the plain interpolation has data on both
    sides of the seam (|lon| ~ 178). Every other cell is the plain result."""
    old = griddata(np.column_stack([lon, lat]), val, (GX, GY), method='linear')
    lon = np.mod(lon + 180.0, 360.0) - 180.0
    e, w = lon > 170, lon < -170
    per = griddata(np.column_stack([np.concatenate([lon, lon[e] - 360.0, lon[w] + 360.0]),
                                    np.concatenate([lat, lat[e], lat[w]])]),
                   np.concatenate([val, val[e], val[w]]), (GX, GY), method='linear')
    gx = GX[0]
    seam = np.abs(gx) > 178.5
    inner = ~seam & (np.abs(gx) > 177.0)
    left = np.isfinite(old[:, inner & (gx < 0)]).all(axis=1)
    right = np.isfinite(old[:, inner & (gx > 0)]).all(axis=1)
    fill = np.isnan(old) & seam[None, :] & (left & right)[:, None] & np.isfinite(per)
    old[fill] = per[fill]
    return old


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
    grid = regrid_closed(lon[sel], lat[sel], mld[sel], GX, GY)
    return gx, gy, grid


data = {e: load(e) for e in EXPS}

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(1, 2, wspace=0.12,
                      left=0.05, right=0.95, top=0.89, bottom=0.21)
proj = ccrs.SouthPolarStereo()
letters = 'ab'
p = 0

ROWS = [(300, np.arange(0, 301, 20), 'Tightened scale (0--300 m)', None)]

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
        gl = ax.gridlines(draw_labels=True, linewidth=0.5 / SCALE, color='gray',
                          alpha=0.5, linestyle='--', x_inline=False, y_inline=True)
        gl.top_labels = gl.right_labels = gl.bottom_labels = False
        gl.yformatter = LatEvery10()
        gl.xlabel_style = {'size': pt(8), 'color': 'black'}
        gl.ylabel_style = {'size': pt(8), 'color': 'black'}
        ax.spines['geo'].set_linewidth(0.75 / SCALE)
        if r == 0:
            ax.set_title(lab, fontsize=pt(10), fontweight='bold', pad=6)
        ax.text(0.03, 0.97, f'({letters[p]})', transform=ax.transAxes, fontsize=pt(9.5),
                fontweight='bold', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))
        p += 1
    cfs.append(cf)

cax = fig.add_axes([0.27, 0.125, 0.46, 0.032])
cb = fig.colorbar(cfs[0], cax=cax, orientation='horizontal',
                  ticks=np.arange(0, 301, 75))
cb.set_label('JJA mixed layer depth (m)', fontsize=pt(8.5), fontweight='bold')
cb.ax.tick_params(labelsize=pt(8), labelcolor='black', width=0.6 / SCALE)
cb.outline.set_linewidth(0.6 / SCALE)

fig.savefig('figures/figR7_mld_glacial_zoom.pdf', dpi=400)
fig.savefig('figures/figR7_mld_glacial_zoom.png', dpi=310)
print('saved figures/figR7_mld_glacial_zoom.{pdf,png}')
