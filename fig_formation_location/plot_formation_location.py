#!/usr/bin/env python
"""Where each density class is made: JJA surface density flux restricted to
(a) the densest tail (sigma2 >= sigma_10 of each state) and (b) the published
peak bin, plus the regional split of the transformation (c-e).

Reads data/regions_<exp>.nc, data/G_<exp>.nc, data/key_<exp>.json written by
analyze_formation_location.py. Run in the foreground (cartopy):
    python3 -s plot_formation_location.py
    python3 -s plot_formation_location.py --no-ice-edge
Output: fig_formation_location.{png,pdf}; with --no-ice-edge the 15% sea-ice
edge contour and its legend entry are left out and the output is
fig_formation_location_noice.{png,pdf} (letter Fig. R12, user 2026-11-26).
"""
import json
import sys
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs

HERE = '/work/ba1066/a270064/cc_projects/aabw_5exps/fig_formation_location'
INP = '/home/a/a270064/bb1029/inputs'
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABEL = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
SURF_MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
             'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}
MONTH = 30 * 86400.0
UNIT = 'kg m$^{-2}$ month$^{-1}$'
LAT_EDGE = -55.0
# categorical slots, validated with the dataviz validator (light mode, adjacent pairs)
C_SHELF, C_CONV, C_ICE, C_FREE = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
C_COAST_PACK = '#86b6ef'      # lighter step of the shelf blue, drawn with hatching

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333', 'axes.linewidth': 0.8})
PROJ = ccrs.SouthPolarStereo()


def load(exp):
    r = nc.Dataset(f'{HERE}/data/regions_{exp}.nc')
    r.set_auto_mask(False)
    d = {k: r[k][:] for k in r.variables}
    d['peak_bin'] = r.peak_bin
    d['sig10'] = r.sig10
    r.close()
    m = nc.Dataset(f'{INP}/{SURF_MESH[exp]}/fesom.mesh.diag.nc')
    m.set_auto_mask(False)
    elem = m['elem'][:] - 1
    m.close()
    if elem.shape[0] != 3:
        elem = elem.T
    glob2sub = -np.ones(elem.max() + 1, int)
    glob2sub[d['node_index'].astype(int)] = np.arange(d['node_index'].size)
    t = glob2sub[elem].T
    t = t[(t >= 0).all(axis=1)]
    xy = PROJ.transform_points(ccrs.PlateCarree(), d['lon'], d['lat'])
    d['tri'] = mtri.Triangulation(xy[:, 0], xy[:, 1], t)
    d['key'] = json.load(open(f'{HERE}/data/key_{exp}.json'))
    return d


def circular(ax):
    th = np.linspace(0, 2 * np.pi, 200)
    c = np.vstack([np.sin(th), np.cos(th)]).T
    ax.set_boundary(mpath.Path(c * 0.5 + 0.5), transform=ax.transAxes)


def sym_limit(fields, sel):
    v = np.concatenate([np.abs(f[s][f[s] != 0]) for f, s in zip(fields, sel)])
    return float(np.percentile(v, 98))


def stacked(ax, vals, colors, hatches, xlab, ylab, title, letter):
    """vals: (nstate, nregion) in Sv; positives stacked up, negatives down."""
    x = np.arange(vals.shape[0])
    up = np.zeros(vals.shape[0])
    dn = np.zeros(vals.shape[0])
    for j in range(vals.shape[1]):
        v = vals[:, j]
        bottom = np.where(v >= 0, up, dn)
        ax.bar(x, v, 0.62, bottom=bottom, color=colors[j], hatch=hatches[j],
               edgecolor='white', linewidth=1.0)
        up += np.where(v >= 0, v, 0)
        dn += np.where(v < 0, v, 0)
    tot = vals.sum(axis=1)
    for i in range(len(x)):
        ax.text(x[i], up[i], f'{tot[i]:.1f}', ha='center', va='bottom', fontsize=9, color='#222222')
    ax.set_xticks(x)
    ax.set_xticklabels(xlab, fontsize=9)
    ax.set_ylabel(ylab)
    ax.axhline(0, color='#555555', lw=0.6)
    ax.set_title(title, fontsize=10, loc='left')
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.set_ylim(min(0, dn.min() * 1.1), up.max() * 1.15)


def main():
    ice_edge = '--no-ice-edge' not in sys.argv[1:]
    out = 'fig_formation_location' + ('' if ice_edge else '_noice')
    D = {e: load(e) for e in EXPS}
    fig = plt.figure(figsize=(15.5, 12.8))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.95], hspace=0.30, wspace=0.06,
                          left=0.045, right=0.925, top=0.93, bottom=0.07)
    rows = [('F_jja_dense_sig10', 'densest tail, $\\sigma_2 \\geq \\sigma_{10}$'),
            ('F_jja_peakbin', 'peak density class')]
    for ri, (var, rtitle) in enumerate(rows):
        sel = [(D[e]['lat'] < LAT_EDGE) for e in EXPS]
        if ri == 0:
            # all-positive and confined to a few hundred nodes: log colour scale, node dots
            norm = mpl.colors.LogNorm(0.1, 50)
            cmap = 'YlOrRd'
        else:
            vmax = sym_limit([D[e][var] * MONTH for e in EXPS], sel)
            norm = mpl.colors.Normalize(-vmax, vmax)
            cmap = 'RdBu_r'
        for ci, e in enumerate(EXPS):
            d = D[e]
            ax = fig.add_subplot(gs[ri, ci], projection=PROJ)
            ax.set_extent([-180, 180, -90, LAT_EDGE], ccrs.PlateCarree())
            circular(ax)
            ax.set_facecolor('#d0d0d0')
            f = d[var] * MONTH
            if ri == 0:
                ax.set_facecolor('#d0d0d0')
                ax.tripcolor(d['tri'], np.zeros_like(f), cmap='Greys', vmin=-1, vmax=20,
                             shading='flat', rasterized=True)
                k = (f >= 0.1) & (d['lat'] < LAT_EDGE + 1)
                o = np.argsort(f[k])
                xy = d['tri'].x[k][o], d['tri'].y[k][o]
                pc = ax.scatter(*xy, c=f[k][o], s=6, cmap=cmap, norm=norm, linewidths=0,
                                rasterized=True, zorder=3)
            else:
                pc = ax.tripcolor(d['tri'], f, shading='gouraud', cmap=cmap,
                                  norm=norm, rasterized=True)
            ax.gridlines(xlocs=[-180, -90, 0, 90], ylocs=[], color='#9a9a9a', lw=0.4, alpha=0.7)
            if ci == 0 and ri == 0:
                for lo, t in [(0, '0$^\\circ$'), (90, '90$^\\circ$E'), (180, '180$^\\circ$'), (-90, '90$^\\circ$W')]:
                    ax.text(lo, -57.3, t, transform=ccrs.PlateCarree(), fontsize=7, color='#555555',
                            ha='center', va='center')
            ax.tricontour(d['tri'], d['bottom_depth'], levels=[1000], colors='#262626', linewidths=0.6)
            if ice_edge:
                ax.tricontour(d['tri'], d['aice_jja'], levels=[0.15], colors='k', linewidths=1.1,
                              linestyles='--')
            if (d['mld1_jja'][d['lat'] < LAT_EDGE] > 400).any():
                ax.tricontour(d['tri'], d['mld1_jja'], levels=[400], colors='#7b3294', linewidths=1.3)
            ax.plot(np.linspace(-180, 180, 361), np.full(361, -60.0), transform=ccrs.PlateCarree(),
                    color='#555555', lw=0.7, ls=':')
            cls = (f'$\\sigma_2 \\geq$ {d["sig10"]:.2f}' if ri == 0 else f'$\\sigma_2$ {d["peak_bin"]}')
            ax.text(0.5, -0.07, cls, transform=ax.transAxes, ha='center', va='top', fontsize=10)
            if ri == 0:
                ax.set_title(LABEL[e], fontsize=13, fontweight='bold', pad=6)
            if ci == 0:
                ax.text(-0.10, 1.03, 'ab'[ri], transform=ax.transAxes, fontsize=13, fontweight='bold')
                ax.text(-0.085, 0.5, rtitle, transform=ax.transAxes, rotation=90, ha='center',
                        va='center', fontsize=10)
        cax = fig.add_axes([0.94, 0.70 - ri * 0.30, 0.009, 0.19])
        cb = fig.colorbar(pc, cax=cax, extend='both' if ri else 'max')
        cb.set_label(f'JJA density flux ({UNIT})' + ('\n(nodes with flux $\\geq$ 0.1)' if ri == 0 else ''),
                     fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    handles = [Line2D([], [], color='#262626', lw=0.6, label='1000 m isobath'),
               Line2D([], [], color='k', lw=1.1, ls='--', label='JJA 15% sea-ice edge'),
               Line2D([], [], color='#7b3294', lw=1.3, label='JJA MLD 400 m'),
               Line2D([], [], color='#555555', lw=0.7, ls=':', label='60$^\\circ$S (WMT domain)')]
    if not ice_edge:
        handles.pop(1)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.485, 0.998), ncol=len(handles),
               frameon=False, fontsize=9.5)

    # ------------------------------------------------ bars
    gsb = gs[2, :].subgridspec(1, 3, wspace=0.28)
    keys = [D[e]['key'] for e in EXPS]
    dep_col = [C_SHELF, C_CONV, C_ICE, C_FREE]
    dep_hat = [None, None, None, None]
    xl_peak = [f'{LABEL[k["exp"]]}\n{k["peak_bin"][0]:.1f}-{k["peak_bin"][1]:.1f}' for k in keys]
    xl_d = [f'{LABEL[k["exp"]]}\n$\\geq${k["sig10"]:.2f}' for k in keys]
    ax = fig.add_subplot(gsb[0])
    stacked(ax, np.array([k['depth1000']['peak_bin_jja'] for k in keys]), dep_col, dep_hat,
            xl_peak, 'JJA transformation (Sv)', 'Peak class, by depth and ice', 'c')
    ax = fig.add_subplot(gsb[1])
    stacked(ax, np.array([k['depth1000']['sig10_jja'] for k in keys]), dep_col, dep_hat,
            xl_d, 'JJA transformation (Sv)', 'Densest tail, by depth and ice', 'd')
    ax = fig.add_subplot(gsb[2])
    stacked(ax, np.array([k['coast200']['sig10_jja'] for k in keys]),
            [C_SHELF, C_COAST_PACK, C_CONV, C_ICE, C_FREE], [None, '////', None, None, None],
            xl_d, 'JJA transformation (Sv)', 'Densest tail, by distance to coast', 'e')
    leg = [Patch(facecolor=C_SHELF, label='shelf/slope (H < 1000 m) | c, d'),
           Patch(facecolor=C_SHELF, label='within 200 km of coast, ice < 0.7 | e'),
           Patch(facecolor=C_COAST_PACK, hatch='////', edgecolor='white',
                 label='within 200 km of coast, ice $\\geq$ 0.7 | e'),
           Patch(facecolor=C_CONV, label='deep/offshore, MLD > 400 m'),
           Patch(facecolor=C_ICE, label='deep/offshore, under sea ice'),
           Patch(facecolor=C_FREE, label='deep/offshore, ice-free')]
    fig.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=6, frameon=False,
               fontsize=8.6, handlelength=1.6, columnspacing=1.2)
    for ext, dpi in [('png', 200), ('pdf', 300)]:
        fig.savefig(f'{HERE}/{out}.{ext}', dpi=dpi)
    print(f'saved {out}.{{png,pdf}}')


if __name__ == '__main__':
    main()
