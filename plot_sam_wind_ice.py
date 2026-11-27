#!/usr/bin/env python
"""
Second SAM figure for Reviewer 3: the spatial mechanism.

Row 1  SAM composite zonal wind stress anomaly (high minus low SAM), JJA
Row 2  SAM composite sea ice concentration anomaly, JJA

Together these show the wind perturbation that drives the Ekman divergence, and
the sea ice response that carries the salt pump, for all five climate states.

2026-11-27 (A4 readability): drawn at the printed width 5.81 in, so every size below is
the printed size in pt. Gridline labels are drawn by hand: longitudes outside the outer
columns, two latitudes in panel (a) only. Old: figures/versions/*_v20261127pre.*
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
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'text.color': 'k', 'axes.labelcolor': 'k',
                     'xtick.color': 'k', 'ytick.color': 'k'})
PRINT_W = 5.81          # in, width of the figure in the letter
PNG_DPI = 723           # 5.81 in x 723 = 4200 px, the old pixel width
PC = ccrs.PlateCarree()


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


def gridlines(ax):
    ax.gridlines(draw_labels=False, linewidth=0.5, color='#555555', alpha=0.6,
                 linestyle='--')
    ax.spines['geo'].set_linewidth(0.6)


def lon_labels(ax, side):
    """60/120 deg labels just outside the circle, on the outer side of the figure."""
    sgn = -1 if side == 'W' else 1
    tr = PC._as_mpl_transform(ax)
    for lo in (60, 120):
        ax.annotate(f'{lo}°{side}', xy=(sgn * lo, LAT_MAX), xycoords=tr,
                    xytext=(sgn * 4, 0), textcoords='offset points', fontsize=8,
                    ha='right' if side == 'W' else 'left', va='center',
                    annotation_clip=False)


def lat_labels(ax):
    for la in (-60, -80):
        ax.text(150, la, f'{abs(la)}°S', transform=PC, ha='center', va='center',
                fontsize=8, zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3))


def panel_letter(ax, letter):
    """Above the top-left corner of the square axes, outside the circular map."""
    ax.text(-0.07, 1.04, f'({letter})', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='left')


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


# layout in inches (printed size)
FW = PRINT_W
left, right, gap = 0.72, 0.38, 0.05          # left/right hold the longitude labels
D = (FW - left - right - 4 * gap) / 5        # side of one square map
top, below, between, bottom = 0.24, 0.60, 0.14, 0.04   # below = colorbar + ticks + label
FH = top + D + below + between + D + below + bottom
y_row = [FH - top - D, bottom + below]       # bottom edge of row 1 and row 2
fig = plt.figure(figsize=(FW, FH))
proj = ccrs.SouthPolarStereo()
letters = 'abcdefghij'
p = 0
cf1 = cf2 = None


def map_axes(row, col):
    return fig.add_axes([(left + col * (D + gap)) / FW, y_row[row] / FH, D / FW, D / FH],
                        projection=proj)


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

    ax = map_axes(0, c)
    ax.set_extent([-180, 180, -90, LAT_MAX], crs=ccrs.PlateCarree())
    circular(ax)
    lon_c = np.concatenate([lon, [360.0]])
    anom_c = np.concatenate([anom, anom[:, :1]], axis=1)
    cf1 = ax.contourf(lon_c, lat, anom_c, levels=np.arange(-0.12, 0.121, 0.01),
                      cmap='PuOr_r', extend='both', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gridlines(ax)
    ax.set_title(lab, fontsize=10, fontweight='bold', pad=4)
    panel_letter(ax, letters[p])
    p += 1
    if c == 0:
        lon_labels(ax, 'W')
        lat_labels(ax)                  # same grid in every panel; label it once
        fig.text(0.31 / FW, (y_row[0] + D / 2) / FH, 'Zonal wind\nstress', rotation=90,
                 va='center', ha='right', fontsize=8.5, fontweight='bold',
                 multialignment='center', linespacing=1.1)
    if c == 4:
        lon_labels(ax, 'E')

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
    grid = regrid_closed(flon[sel], flat[sel], dai[sel], GX, GY)

    ax = map_axes(1, c)
    ax.set_extent([-180, 180, -90, LAT_MAX], crs=ccrs.PlateCarree())
    circular(ax)
    cf2 = ax.contourf(gx, gy, grid, levels=np.arange(-0.20, 0.201, 0.02),
                      cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gridlines(ax)
    panel_letter(ax, letters[p])
    p += 1
    if c == 0:
        lon_labels(ax, 'W')
        fig.text(0.31 / FW, (y_row[1] + D / 2) / FH, 'Sea ice\nconcentration', rotation=90,
                 va='center', ha='right', fontsize=8.5, fontweight='bold',
                 multialignment='center', linespacing=1.1)
    if c == 4:
        lon_labels(ax, 'E')

for row, cf, ticks, label in (
        (0, cf1, np.arange(-0.12, 0.121, 0.04),
         'Zonal wind stress anomaly, high $-$ low SAM  (N m$^{-2}$)'),
        (1, cf2, np.arange(-0.2, 0.21, 0.1),
         'Sea ice concentration anomaly, high $-$ low SAM')):
    cax = fig.add_axes([0.20, (y_row[row] - 0.10 - 0.09) / FH, 0.60, 0.09 / FH])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=ticks)
    cb.set_label(label, fontsize=8.5, fontweight='bold', labelpad=2)
    cb.ax.tick_params(labelsize=8, width=0.6, length=3)
    cb.outline.set_linewidth(0.6)

fig.savefig('figures/figR5_sam_wind_ice.pdf', dpi=400)
fig.savefig('figures/figR5_sam_wind_ice.png', dpi=PNG_DPI)
print('saved figures/figR5_sam_wind_ice.{pdf,png}')
