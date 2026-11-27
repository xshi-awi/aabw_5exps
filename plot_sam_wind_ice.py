#!/usr/bin/env python
"""
Second SAM figure for Reviewer 3: the spatial mechanism.

Row 1  SAM composite zonal wind stress anomaly (high minus low SAM), JJA
Row 2  SAM composite sea ice concentration anomaly, JJA

Together these show the wind perturbation that drives the Ekman divergence, and
the sea ice response that carries the salt pump, for all five climate states.

2026-11-27 (A4 readability): drawn at the printed width 5.81 in, so every size below is
the printed size in pt. Old: figures/versions/*_v20261127pre.*
2026-11-27 10:34 (user): every latitude circle and every meridian labelled in every
panel, small (6 pt). Meridian labels sit outside the circle along it; latitude labels
along 20E, where a search over meridians found the least hidden signal. Old: *_v20261127b.*
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
proj = ccrs.SouthPolarStereo()
import matplotlib.patheffects as patheffects
import matplotlib.ticker as mticker
GRID_LONS = [-180, -120, -60, 0, 60, 120]      # cartopy's default lines here, now fixed
GRID_LATS = [-80, -70, -60, -50]
FS_GRID = 6                                    # printed pt
LATLAB_LON = 20
# side of the meridian (L, R, C), and the gap in pt to the pole side of the circle (None = on it)
LAT_LAYOUT = {-80: ('C', 2.0), -70: ('C', 0.5), -60: ('L', None), -50: ('L', None)}
HALO = [patheffects.withStroke(linewidth=1.2, foreground='white')]


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
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='#555555', alpha=0.6,
                      linestyle='--')
    gl.xlocator = mticker.FixedLocator(GRID_LONS)
    gl.ylocator = mticker.FixedLocator(GRID_LATS)
    ax.spines['geo'].set_linewidth(0.6)


def lon_text(lo):
    if lo == 0:
        return '0°'
    if abs(lo) == 180:
        return '180°'
    return f'{abs(lo)}°{"E" if lo > 0 else "W"}'


def grid_labels(ax):
    """Each meridian labelled just outside the circle where it ends, written along the
    circle; each latitude circle labelled along LATLAB_LON (LAT_LAYOUT: side of the
    meridian, and whether the label sits on the pole side of its circle)."""
    for lo in GRID_LONS:
        dx, dy = proj.transform_point(lo, -60, PC)
        n = np.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        rot = np.degrees(np.arctan2(dy, dx)) - 90
        rot = (rot + 180) % 360 - 180
        if rot > 90:
            rot -= 180
        elif rot < -90:
            rot += 180
        ax.annotate(lon_text(lo), xy=(0.5 + 0.5 * dx, 0.5 + 0.5 * dy), xycoords='axes fraction',
                    xytext=(5 * dx, 5 * dy), textcoords='offset points', fontsize=FS_GRID,
                    ha='center', va='center', rotation=rot, rotation_mode='anchor',
                    annotation_clip=False)
    for la in GRID_LATS:
        side, gap = LAT_LAYOUT[la]
        pole = gap is not None
        px, py = proj.transform_point(LATLAB_LON, la, PC)
        up = py > 0
        ax.annotate(f'{abs(la)}°S', xy=(px, py), xycoords='data',
                    xytext=({'L': -2.0, 'R': 2.0, 'C': 0.0}[side],
                            ((-gap if up else gap) if pole else 0.0)),
                    textcoords='offset points', fontsize=FS_GRID,
                    ha={'L': 'right', 'R': 'left', 'C': 'center'}[side],
                    va=('top' if up else 'bottom') if pole else 'center',
                    path_effects=HALO, zorder=6, annotation_clip=False)


def panel_letter(ax, letter):
    """Above the top-left corner of the square axes, outside the circular map."""
    ax.text(-0.07, 1.0, f'({letter})', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom', ha='left')


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
over = 8.6 / 72                              # 120°E/W label beyond the square, 6 pt
left, right, gap = 0.03 + 0.27 + 0.04 + over, over + 0.02, 2 * over + 2.5 / 72
D = (FW - left - right - 4 * gap) / 5        # side of one square map
top, below, between, bottom = 0.36, 0.67, 0.17, 0.04   # below = 180° label + colour bar
FH = top + D + below + between + D + below + bottom
y_row = [FH - top - D, bottom + below]       # bottom edge of row 1 and row 2
fig = plt.figure(figsize=(FW, FH))
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
    ax.set_title(lab, fontsize=10, fontweight='bold', pad=12)
    panel_letter(ax, letters[p])
    grid_labels(ax)
    p += 1
    if c == 0:
        fig.text((left - 0.04 - over) / FW, (y_row[0] + D / 2) / FH, 'Zonal wind\nstress',
                 rotation=90, va='center', ha='right', fontsize=8.5, fontweight='bold',
                 multialignment='center', linespacing=1.1)

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
    grid_labels(ax)
    p += 1
    if c == 0:
        fig.text((left - 0.04 - over) / FW, (y_row[1] + D / 2) / FH, 'Sea ice\nconcentration',
                 rotation=90, va='center', ha='right', fontsize=8.5, fontweight='bold',
                 multialignment='center', linespacing=1.1)

for row, cf, ticks, label in (
        (0, cf1, np.arange(-0.12, 0.121, 0.04),
         'Zonal wind stress anomaly, high $-$ low SAM  (N m$^{-2}$)'),
        (1, cf2, np.arange(-0.2, 0.21, 0.1),
         'Sea ice concentration anomaly, high $-$ low SAM')):
    cax = fig.add_axes([0.20, (y_row[row] - 0.17 - 0.09) / FH, 0.60, 0.09 / FH])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=ticks)
    cb.set_label(label, fontsize=8.5, fontweight='bold', labelpad=2)
    cb.ax.tick_params(labelsize=8, width=0.6, length=3)
    cb.outline.set_linewidth(0.6)

fig.savefig('figures/figR5_sam_wind_ice.pdf', dpi=400)
fig.savefig('figures/figR5_sam_wind_ice.png', dpi=PNG_DPI)
print('saved figures/figR5_sam_wind_ice.{pdf,png}')
