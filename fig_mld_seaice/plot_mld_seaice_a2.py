#!/usr/bin/env python3
"""Supplementary Fig. A2 (label mld): June-September MLD (a-e) and sea ice (f-j).

Same data, smoothing and sea-ice row as plot_figure2() in ../plot_climate_patterns_jja.py
(the figure in the manuscript since March 2026). Only change (user, 2026-11-26, Reviewer 3
L134): the MLD colour scale of the interglacial panels (a-c) now runs to 1000 m; the glacial
panels (d-e) keep the old 0-400 m scale. Run from the project root:
    python3 -s fig_mld_seaice/plot_mld_seaice_a2.py
The PI sea-ice panel still reads pi/a_ice_reg.nc (the overwritten 2090-2099 file); the user
decided (FS, 2026-11-25) not to change it yet. Do not switch to *_beta here without her.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
import plot_climate_patterns_jja as P

OUT = 'figures/fig02_climate_mld_seaice_jja_mld1000'


def mld_cmap_norm(vmax, n_blue=18):
    blues = plt.cm.Blues
    cols = np.vstack([[[1, 1, 1, 1]], blues(np.linspace(0.4, 1.0, n_blue))])
    cmap = matplotlib.colors.ListedColormap(cols)
    cmap.set_over(blues(1.0))
    lev = np.concatenate([[0, 50], np.linspace(50, vmax, n_blue)[1:]])
    return lev, cmap, matplotlib.colors.BoundaryNorm(lev, ncolors=len(lev) - 1)


def main():
    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 5, figure=fig, hspace=0.25, wspace=0.08,
                  left=0.06, right=0.94, top=0.92, bottom=0.1)
    cols = ['#0000FF', '#0040FF', '#0080FF', '#00BFFF', '#00FFFF', '#00FFB0', '#00FF60',
            '#00FF00', '#60FF00', '#BFFF00', '#FFFF00', '#FFD000', '#FFA000', '#FF6000',
            '#FF2000', '#FF0000', '#CC0000']
    cmap_ice = matplotlib.colors.LinearSegmentedColormap.from_list('BlAqGrYeOrRe', cols, N=256)
    cmap_ice.set_under('#0000FF')
    warm = mld_cmap_norm(1000.0)   # a-c
    cold = mld_cmap_norm(400.0)    # d-e, unchanged from the March figure

    panel = 0
    ims = {}
    axes = {}
    for row, var in enumerate(['MLD1', 'a_ice']):
        for col, exp in enumerate(P.exps):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
            axes[(row, col)] = ax
            data, lon, lat = P.load_variable_reg(exp, var)
            if var == 'MLD1':
                data = np.abs(data)
                lev, cmap, norm = warm if col < 3 else cold
                kw = dict(levels=lev, cmap=cmap, norm=norm, extend='max')
            else:
                data = np.where(data < (0.05 if exp in ['lgm', 'mis'] else 0.01), np.nan, data)
                kw = dict(levels=np.linspace(0.1, 1, 19), cmap=cmap_ice, extend='both')
            ds = P.smooth_data(data, sigma=1.5)
            dc, lonc = P.add_cyclic_point(ds, lon)
            lo2, la2 = np.meshgrid(lonc, lat)
            im = ax.contourf(lo2, la2, dc, transform=ccrs.PlateCarree(), **kw)
            ims[(row, col)] = im
            ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
            ax.coastlines(linewidth=0.5, zorder=3)
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
            ax.text(0.02, 0.98, f'({chr(97 + panel)})', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
            panel += 1
            if row == 0:
                ax.set_title(P.exp_labels[exp], fontsize=12, fontweight='bold')
            if col == 0:
                ax.text(-0.12, 0.5, 'MLD' if var == 'MLD1' else 'Sea Ice', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', rotation=90, va='center', ha='right')

    def cbar(row, c0, c1, label, frac=0.8):
        p0, p1 = axes[(row, c0)].get_position(), axes[(row, c1)].get_position()
        w = (p1.x1 - p0.x0) * frac
        x0 = (p0.x0 + p1.x1 - w) / 2
        cax = fig.add_axes([x0, p0.y0 - 0.045, w, 0.015])
        cb = fig.colorbar(ims[(row, c0)], cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
        return cb

    cb = cbar(0, 0, 2, 'MLD (m), PI, MH, LIG')
    cb.set_ticks([0, 50, 200, 400, 600, 800, 1000])
    cb = cbar(0, 3, 4, 'MLD (m), LGM, MIS3')
    cb.set_ticks([0, 50, 100, 200, 300, 400])
    cbar(1, 0, 4, 'Sea Ice (fraction)', frac=0.6)

    for ext in ('pdf', 'png'):
        plt.savefig(f'{OUT}.{ext}', dpi=300, bbox_inches='tight')
        print('saved', f'{OUT}.{ext}')
    plt.close()


if __name__ == '__main__':
    main()
