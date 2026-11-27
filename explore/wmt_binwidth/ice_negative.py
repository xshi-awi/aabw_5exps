#!/usr/bin/env python
"""Why the winter (JJA) sea ice term of Fig. 2 is negative in some 0.1 classes (user question 11-27 12:44).
Per-node, per-0.02-bin JJA accumulations of fig_formation_location/data/acc_<exp>.nc (F*area of the months
whose surface sigma2 falls in the bin; F_ice = -beta*S*fsitherm, positive = densifying, i.e. ice growth and
brine; negative = melt water) are summed over a 0.1 class and split by the node's JJA ice concentration,
distance to the 15% edge and winter mixed layer. Writes ice_negative.txt and ice_negative_map.png."""
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import netCDF4 as nc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FL = os.path.join(HERE, '..', '..', 'fig_formation_location', 'data')
CASES = [('pi', 'Weddell', 36.7), ('mh', 'Weddell', 36.7), ('lig', 'Southern Ocean', 36.6)]
LAB = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG'}
REG = {'Weddell': (-60, 79), 'Southern Ocean': (-180, 180)}
out = []


def P(s=''):
    print(s)
    out.append(s)


fig = plt.figure(figsize=(7.0, 3.3))
for k, (e, reg, lo) in enumerate(CASES):
    a = nc.Dataset(f'{FL}/acc_{e}.nc')
    r = nc.Dataset(f'{FL}/regions_{e}.nc')
    edges = a['edges_sig2f'][:]
    kb = np.where((edges[:-1] >= lo - 1e-6) & (edges[:-1] < lo + 0.1 - 1e-6))[0]
    acc = a['acc_jja_sig2f_ice'][:][:, kb].sum(1)             # kg s-1 per node, time mean
    lat, lon = a['lat'][:], a['lon'][:]
    lon = np.where(lon > 180, lon - 360, lon)
    ai, mld = r['aice_jja'][:], r['mld1_jja'][:]
    assert np.allclose(r['lat'][:], lat)
    m = (lat <= -60) & (lon >= REG[reg][0]) & (lon <= REG[reg][1])
    G = lambda x: 1035 / 1e9 * x / 0.1                        # Sv
    neg, pos = acc < 0, acc > 0
    P(f'{LAB[e]} {reg}, JJA, sigma2 {lo:.1f}-{lo + 0.1:.1f}: sea ice term {G(acc[m].sum()):+.2f} Sv '
      f'= melt {G(acc[m & neg].sum()):+.2f} + growth {G(acc[m & pos].sum()):+.2f}')
    for nm, sel in [('ice-free (<0.15)', ai < 0.15), ('edge (0.15-0.5)', (ai >= 0.15) & (ai < 0.5)),
                    ('pack (0.5-0.9)', (ai >= 0.5) & (ai < 0.9)), ('close pack (>=0.9)', ai >= 0.9)]:
        P(f'    {nm:20s} melt {G(acc[m & neg & sel].sum()):+6.2f}  growth {G(acc[m & pos & sel].sum()):+6.2f} Sv')
    deep = mld > 400
    P(f'    melt where winter MLD > 400 m: {G(acc[m & neg & deep].sum()):+.2f} Sv; '
      f'melt-weighted mean latitude {np.average(lat[m & neg], weights=-acc[m & neg]):.1f}, '
      f'lon {np.average(lon[m & neg], weights=-acc[m & neg]):.1f}, '
      f'ice conc {np.average(ai[m & neg], weights=-acc[m & neg]):.2f}')
    # the same nodes in the neighbouring lighter class (36.6-36.7 or 36.5-36.6), to show why 0.2 hid it
    kb2 = np.where((edges[:-1] >= lo - 0.1 - 1e-6) & (edges[:-1] < lo - 1e-6))[0]
    acc2 = a['acc_jja_sig2f_ice'][:][:, kb2].sum(1)
    P(f'    lighter neighbour {lo - 0.1:.1f}-{lo:.1f}: sea ice term {G(acc2[m].sum()):+.2f} Sv; '
      f'0.2 bin {lo - 0.1:.1f}-{lo + 0.1:.1f} average {0.5 * (G(acc2[m].sum()) + G(acc[m].sum())):+.2f} Sv')
    P()
    # map: contribution per unit area, kg m-2 month-1
    ax = fig.add_subplot(1, 3, k + 1, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -55], crs=ccrs.PlateCarree())
    t = np.linspace(0, 2 * np.pi, 100)
    ax.set_boundary(mpath.Path(np.vstack([np.sin(t), np.cos(t)]).T * 0.5 + 0.5), transform=ax.transAxes)
    v = acc / a['area'][:] * 30 * 86400
    sel = np.abs(v) > 0.05
    sc = ax.scatter(lon[sel], lat[sel], c=v[sel], s=0.6, cmap='RdBu_r', vmin=-3, vmax=3, transform=ccrs.PlateCarree(),
                    rasterized=True, linewidths=0)
    tri = ax.tricontour(lon, lat, ai, levels=[0.15], colors='k', linewidths=0.6, transform=ccrs.PlateCarree())
    ax.tricontour(lon, lat, mld, levels=[400], colors='#7B3294', linewidths=0.6, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#cccccc', zorder=3)
    ax.set_title(f'({chr(97 + k)}) {LAB[e]}, $\\sigma_2$ {lo:.1f}–{lo + 0.1:.1f}', fontsize=9, fontweight='bold')
cax = fig.add_axes([0.25, 0.20, 0.5, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both')
cb.set_label('JJA sea ice density flux in this class (kg m$^{-2}$ month$^{-1}$); blue = melt', fontsize=8, fontweight='bold')
cb.ax.tick_params(labelsize=7.5)
fig.text(0.5, 0.015, 'black: JJA 15% ice edge; purple: JJA mixed layer 400 m', ha='center', fontsize=7.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.27, wspace=0.05)
fig.savefig(os.path.join(HERE, 'ice_negative_map.png'), dpi=300)
open(os.path.join(HERE, 'ice_negative.txt'), 'w').write('\n'.join(out) + '\n')
print('wrote ice_negative_map.png, ice_negative.txt')
