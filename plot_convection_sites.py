#!/usr/bin/env python
"""
Where does the model actually form its dense water?

Answers Reviewer 1 comment 23 ("clarify where the primary deep water formation
regions occur in this model"), Reviewer 3's main critique (open-ocean convection
versus shelf processes), and quantifies the glacial contraction that Reviewer 1
comment 10 questioned.

(a-e) JJA mixed layer depth for the five states, with the 400 m contour marked
      and the shelf break (1000 m isobath) drawn for reference.
(f)   Area with winter MLD deeper than 400 and 600 m, by sector, for PI.
(g)   Total deep-convection area across the five states.
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

fig = plt.figure(figsize=(21, 10.8))
gs = fig.add_gridspec(2, 5, height_ratios=[1.35, 1.0], hspace=0.60, wspace=0.10,
                      left=0.05, right=0.985, top=0.93, bottom=0.09)
proj = ccrs.SouthPolarStereo()
letters = 'abcdefg'
p = 0
cf = None

for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    lat, lon, area, mld = data[e]
    ax = fig.add_subplot(gs[0, c], projection=proj)
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    circular(ax)

    sel = lat < -48
    gy = np.arange(-78, -47, 0.5)
    gx = np.arange(-180, 180.5, 1.0)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(np.column_stack([lon[sel], lat[sel]]), mld[sel], (GX, GY),
                    method='linear')

    cf = ax.contourf(gx, gy, grid, levels=np.arange(0, 801, 50), cmap='YlGnBu',
                     extend='max', transform=ccrs.PlateCarree())
    ax.contour(gx, gy, grid, levels=[400], colors='#D55E00', linewidths=1.8,
               transform=ccrs.PlateCarree())
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
    a400 = area[(mld > 400) & (lat < -55)].sum()
    ax.text(0.5, -0.09, f'MLD>400 m: {a400/1e11:.1f}$\\times10^{{11}}$ m$^2$',
            transform=ax.transAxes, ha='center', fontsize=12, color='#333333')

cax = fig.add_axes([0.28, 0.492, 0.44, 0.013])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=np.arange(0, 801, 200))
cb.set_label('JJA mixed layer depth (m); orange contour = 400 m', fontsize=14)
cb.ax.tick_params(labelsize=12)

# --------------------------------------------------- (f) PI sector breakdown
ax = fig.add_subplot(gs[1, 0:2])
lat, lon, area, mld = data['pi']
x = np.arange(len(SECTORS))
for k, thr in enumerate([400, 600]):
    vals = []
    for name, lo, hi in SECTORS:
        sel = (mld > thr) & (lat < -55) & (lon >= lo) & (lon < hi)
        vals.append(area[sel].sum() / 1e11)
    ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34,
           color=['#0072B2', '#D55E00'][k], label=f'MLD > {thr} m')
    for i, v in enumerate(vals):
        ax.text(x[i] + (k - 0.5) * 0.36, v + 0.10, f'{v:.2f}', ha='center', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in SECTORS], fontsize=12)
ax.set_ylabel('Area ($10^{11}$ m$^2$)', fontsize=13)
ax.set_title('(f)  PI deep convection by sector', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.25, lw=0.6)

# --------------------------------------------------- (g) across states
ax = fig.add_subplot(gs[1, 2:4])
x = np.arange(len(EXPS))
for k, thr in enumerate([400, 600]):
    vals = []
    for e in EXPS:
        lat, lon, area, mld = data[e]
        vals.append(area[(mld > thr) & (lat < -55)].sum() / 1e11)
    ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34,
           color=['#0072B2', '#D55E00'][k], label=f'MLD > {thr} m')
    for i, v in enumerate(vals):
        ax.text(x[i] + (k - 0.5) * 0.36, v + 0.10, f'{v:.1f}', ha='center', fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=12)
ax.set_ylabel('Area ($10^{11}$ m$^2$)', fontsize=13)
ax.set_title('(g)  Deep-convection area, all states', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.25, lw=0.6)

# --------------------------------------------------- (h) max MLD
ax = fig.add_subplot(gs[1, 4])
vals = [np.nanmax(data[e][3][data[e][0] < -55]) for e in EXPS]
ax.bar(np.arange(len(EXPS)), vals, color=[COLS[l] for l in LABELS], width=0.62)
ax.set_xticks(np.arange(len(EXPS)))
ax.set_xticklabels(LABELS, fontsize=11, rotation=30)
ax.set_ylabel('Maximum MLD (m)', fontsize=13)
ax.set_title('(h)  Deepest winter\nmixed layer', fontsize=15, fontweight='bold')
for i, v in enumerate(vals):
    ax.text(i, v + 14, f'{v:.0f}', ha='center', fontsize=11)
ax.grid(axis='y', alpha=0.25, lw=0.6)
ax.set_ylim(0, max(vals) * 1.18)

for a in fig.axes:
    a.tick_params(labelsize=11)

fig.savefig('figures/figR6_convection_sites.pdf', dpi=400)
fig.savefig('figures/figR6_convection_sites.png', dpi=200)
print('saved figures/figR6_convection_sites.{pdf,png}')

print()
print('PI sector breakdown, area with MLD above threshold south of 55S (1e11 m2):')
lat, lon, area, mld = data['pi']
for thr in [400, 600]:
    tot = area[(mld > thr) & (lat < -55)].sum()
    print(f'  MLD > {thr} m, total {tot/1e11:.2f}')
    for name, lo, hi in SECTORS:
        sel = (mld > thr) & (lat < -55) & (lon >= lo) & (lon < hi)
        print(f'    {name.splitlines()[0]:10s} {area[sel].sum()/1e11:6.2f}'
              f'  ({area[sel].sum()/tot*100:4.1f}%)')
