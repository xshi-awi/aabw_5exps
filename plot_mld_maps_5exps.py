#!/usr/bin/env python
"""
Winter mixed layer depth for the five states, south of 60 S.

A maps-only cut of plot_convection_sites.py, made at Xiaoxu's request for the
response letter. The bar panels are dropped, and the interglacials and the
glacials get separate colour scales: the glacial mixed layers reach only about
300 m, so on the shared 0-800 m scale of the full figure their coastal cells are
invisible. The 0-300 m scale for LGM and MIS3 matches the glacial figure sent to
Reviewer 1, so the two can be read together.

PI, MH, LIG   0-800 m, 400 m contour   open-ocean convection in the gyres
LGM, MIS3     0-300 m, 200 m contour   discrete coastal cells only
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
SECTORS = [('Ross\n(180°W-60°W)', -180, -60),
           ('Weddell\n(60°W-79°E)', -60, 79),
           ('Adélie\n(79°E-180°E)', 79, 180)]
COLS = {'PI': '#000000', 'MH': '#E69F00', 'LIG': '#009E73',
        'LGM': '#0072B2', 'MIS3': '#CC79A7'}

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def circular(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(t), np.cos(t)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5, closed=True), transform=ax.transAxes)


def load(exp):
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc')
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    lon = np.degrees(m['nodes'].isel(n2=0).values)
    area = (m['nod_area'].isel(nz=0).values if 'nz' in m['nod_area'].dims
            else m['nod_area'].isel(nl=0).values)
    d = xr.open_dataset(f'{exp}/MLD1_clim.nc')
    v = [x for x in d.data_vars][0]
    mld = np.abs(d[v].values)[[5, 6, 7]].mean(axis=0)
    m.close(); d.close()
    return lat, lon, area, mld



data = {e: load(e) for e in EXPS}

fig = plt.figure(figsize=(21, 6.6))
gs = fig.add_gridspec(1, 5, wspace=0.10, left=0.04, right=0.99,
                      top=0.90, bottom=0.22)
proj = ccrs.SouthPolarStereo()
letters = 'abcde'

# interglacials keep the scale of the full figure; the glacials get their own,
# because their mixed layers never approach 400 m and the shared scale hides them
SCALE = {'pi':  (np.arange(0, 801, 50), 400, np.arange(0, 801, 200)),
         'mh':  (np.arange(0, 801, 50), 400, np.arange(0, 801, 200)),
         'lig': (np.arange(0, 801, 50), 400, np.arange(0, 801, 200)),
         'lgm': (np.arange(0, 301, 20), 200, np.arange(0, 301, 75)),
         'mis': (np.arange(0, 301, 20), 200, np.arange(0, 301, 75))}

cf_warm = cf_glac = None
for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    lat, lon, area, mld = data[e]
    levels, contour, _ = SCALE[e]
    ax = fig.add_subplot(gs[0, c], projection=proj)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    circular(ax)

    sel = lat < -58
    gy = np.arange(-78, -57, 0.4)
    gx = np.arange(-180, 180.5, 0.8)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(np.column_stack([lon[sel], lat[sel]]), mld[sel], (GX, GY),
                    method='linear')

    cf = ax.contourf(gx, gy, grid, levels=levels, cmap='YlGnBu',
                     extend='max', transform=ccrs.PlateCarree())
    ax.contour(gx, gy, grid, levels=[contour], colors='#D55E00', linewidths=1.8,
               transform=ccrs.PlateCarree())
    if e == 'lig':
        cf_warm = cf
    if e == 'mis':
        cf_glac = cf
    ax.add_feature(cfeature.LAND, facecolor='#999999', zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                      linestyle='--', x_inline=False, y_inline=True)
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.ylabel_style = {'size': 9, 'color': '#333333'}
    ax.set_title(lab, fontsize=20, fontweight='bold', pad=8)
    ax.text(0.03, 0.97, f'({letters[c]})', transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))
    a400 = area[(mld > 400) & (lat < -60)].sum()
    ax.text(0.5, -0.10, f'MLD>400 m: {a400/1e11:.1f}$\\times10^{{11}}$ m$^2$',
            transform=ax.transAxes, ha='center', fontsize=12, color='#333333')

cax1 = fig.add_axes([0.075, 0.095, 0.44, 0.030])
cb1 = fig.colorbar(cf_warm, cax=cax1, orientation='horizontal',
                   ticks=np.arange(0, 801, 200))
cb1.set_label('(a-c) JJA mixed layer depth (m); orange contour = 400 m', fontsize=14)
cb1.ax.tick_params(labelsize=12)

cax2 = fig.add_axes([0.625, 0.095, 0.30, 0.030])
cb2 = fig.colorbar(cf_glac, cax=cax2, orientation='horizontal',
                   ticks=np.arange(0, 301, 75))
cb2.set_label('(d-e) JJA mixed layer depth (m); orange contour = 200 m', fontsize=14)
cb2.ax.tick_params(labelsize=12)

fig.savefig('figures/figR15_mld_maps_5exps.pdf', dpi=400)
fig.savefig('figures/figR15_mld_maps_5exps.png', dpi=200)
print('saved figures/figR15_mld_maps_5exps.{pdf,png}')
