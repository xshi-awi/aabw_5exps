#!/usr/bin/env python3
"""Supplementary Fig. A2 (label mld): June-September MLD (a-e) and sea ice (f-j).

Same data, smoothing and sea-ice row as plot_figure2() in ../plot_climate_patterns_jja.py
(the figure in the manuscript since March 2026). Only change (user, 2026-11-26, Reviewer 3
L134): the MLD colour scale of the interglacial panels (a-c) now runs to 1000 m; the glacial
panels (d-e) keep the old 0-400 m scale. Run from the project root:
    python3 -s fig_mld_seaice/plot_mld_seaice_a2.py
The PI sea-ice panel still reads pi/a_ice_reg.nc (the overwritten 2090-2099 file); the user
decided (FS, 2026-11-25) not to change it yet. Do not switch to *_beta here without her.

2026-11-27: text enlarged for an A4 printout of the reply letter (printed 5.81 in wide).
Font sizes are given as printed points through pt(); the figure is 8 in wide instead of
20 in, so lines print thicker too. Data, levels, colours and panel order unchanged.
Previous version: figures/versions/fig02_climate_mld_seaice_jja_mld1000_v20261127pre.*

2026-11-27 (user, 10:34, letter only): `--mld-only` draws only the MLD panels with 6 pt
latitude/longitude labels, (a)-(c) on top and (d)-(e) centred below, each row with its own
colour bar, written to *_mldonly.{png,pdf} for letter_figs/fig02_mld1000_mldonly.png.
One-row version of 10:45: figures/versions/*_mldonly_v20261127_onerow.*
The default (both rows, manuscript A2) is unchanged.
    python3 -s fig_mld_seaice/plot_mld_seaice_a2.py --mld-only
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
from matplotlib.transforms import offset_copy
from matplotlib import patheffects
import plot_climate_patterns_jja as P

OUT = 'figures/fig02_climate_mld_seaice_jja_mld1000'

FIG_W, FIG_H = 8.0, 4.65
SAVED_W = 8.015               # width after bbox_inches='tight' (checked 2026-11-27)
SCALE = 5.81 / SAVED_W        # printed width in the letter / saved width


# --mld-only layout in inches: (a)-(c) on top, (d)-(e) centred below; the gaps between
# maps hold the side longitude labels (MO_SIDE wide on each side of a map)
MO_FIG_W, MO_FIG_H = 8.2, 7.04
MO_MAP, MO_GAP, MO_SIDE = 2.32, 0.353, 0.1667
MO_ROW_Y = (4.30, 0.78)       # bottom edge of the maps in the top and bottom row
MLDONLY_SAVED_W = 8.19        # width after bbox_inches='tight' (checked 2026-11-27)
LAT_LABEL_LON = -135          # latitude labels along 135W, the quietest diagonal in a-e


def mo_rect(r, c):
    """figure-fraction rectangle of map c in row r of the --mld-only layout"""
    n = 3 if r == 0 else 2
    x = MO_FIG_W / 2 - (n * MO_MAP + (n - 1) * MO_GAP) / 2 + c * (MO_MAP + MO_GAP)
    return [x / MO_FIG_W, MO_ROW_Y[r] / MO_FIG_H, MO_MAP / MO_FIG_W, MO_MAP / MO_FIG_H]


def pt(p):
    """native size that prints as p points in the letter"""
    return p / SCALE


def grid_labels(ax, fig):
    """6 pt labels for the gridlines the default gridliner draws on these maps
    (meridians every 60 deg; circles 80-50S, and 40S only as corner arcs).
    Each meridian is labelled just outside the frame where it leaves the map, the
    circles 80-50S along LAT_LABEL_LON with a thin white halo, and 40S just below the
    frame where its lower-left arc meets the bottom edge (inside, the arc is too short)."""
    pc = ccrs.PlateCarree()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    kw = dict(fontsize=pt(6), color='black')
    off = 1.5 / SCALE                     # gap between frame and label (points)
    for lon in (0, 60, 120, 180, -120, -60):
        dx, dy = ax.projection.transform_point(lon, -60, pc)   # meridians are rays
        tx = (x1 if dx > 0 else x0) / dx if abs(dx) > 1 else np.inf
        ty = (y1 if dy > 0 else y0) / dy if abs(dy) > 1 else np.inf
        t = min(tx, ty)
        fx, fy = (t * dx - x0) / (x1 - x0), (t * dy - y0) / (y1 - y0)
        name = {0: '0°', 180: '180°'}.get(lon, f'{abs(lon)}°' + ('E' if lon > 0 else 'W'))
        if tx < ty:                       # left or right edge, text along the edge
            sx = 1 if dx > 0 else -1
            ax.text(fx, fy, name, rotation=90, va='center', ha='left' if sx > 0 else 'right',
                    transform=offset_copy(ax.transAxes, fig=fig, x=sx * off, units='points'),
                    **kw)
        else:                             # top or bottom edge
            sy = 1 if dy > 0 else -1
            ax.text(fx, fy, name, ha='center', va='bottom' if sy > 0 else 'top',
                    transform=offset_copy(ax.transAxes, fig=fig, y=sy * off, units='points'),
                    **kw)
    halo = [patheffects.withStroke(linewidth=1.0 / SCALE, foreground='white')]
    for lat in (-80, -70, -60, -50):
        ax.text(LAT_LABEL_LON, lat, f'{-lat}°S', transform=pc, ha='center', va='center',
                path_effects=halo, zorder=6, **kw)
    r40 = np.hypot(*ax.projection.transform_point(0, -40, pc))
    xb = -np.sqrt(r40 ** 2 - y0 ** 2)     # lower-left 40S arc meets the bottom edge
    ax.text((xb - x0) / (x1 - x0), 0.0, '40°S', ha='center', va='top',
            transform=offset_copy(ax.transAxes, fig=fig, y=-off, units='points'), **kw)


def mld_cmap_norm(vmax, n_blue=18):
    blues = plt.cm.Blues
    cols = np.vstack([[[1, 1, 1, 1]], blues(np.linspace(0.4, 1.0, n_blue))])
    cmap = matplotlib.colors.ListedColormap(cols)
    cmap.set_over(blues(1.0))
    lev = np.concatenate([[0, 50], np.linspace(50, vmax, n_blue)[1:]])
    return lev, cmap, matplotlib.colors.BoundaryNorm(lev, ncolors=len(lev) - 1)


def main(mld_only=False):
    global SCALE
    if mld_only:
        SCALE = 5.81 / MLDONLY_SAVED_W
        fig = plt.figure(figsize=(MO_FIG_W, MO_FIG_H))
        rows = ['MLD1']
        top_pad = 10.5 / SCALE            # title and letters sit above the 0° label
    else:
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = GridSpec(2, 5, figure=fig, hspace=0.582, wspace=0.06,
                      left=0.05, right=0.995, top=0.9355, bottom=0.1335)
        rows = ['MLD1', 'a_ice']
        top_pad = 4
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
    for row, var in enumerate(rows):
        for col, exp in enumerate(P.exps):
            if mld_only:
                key = (0, col) if col < 3 else (1, col - 3)
                ax = fig.add_axes(mo_rect(*key), projection=ccrs.SouthPolarStereo())
            else:
                key = (row, col)
                ax = fig.add_subplot(gs[row, col], projection=ccrs.SouthPolarStereo())
            axes[key] = ax
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
            ims[key] = im
            ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
            ax.coastlines(linewidth=0.5 / SCALE, zorder=3)
            ax.gridlines(draw_labels=False, linewidth=0.5 / SCALE, alpha=0.5, linestyle='--')
            ax.spines['geo'].set_linewidth(0.75 / SCALE)
            # panel letter just above the map's top-left corner (on the title line in
            # row 1), so that at print size it does not hide the ice edge in (i), (j)
            ax.text(0.0, 1.0, f'({chr(97 + panel)})', fontsize=pt(9), fontweight='bold',
                    va='baseline', ha='left',
                    transform=offset_copy(ax.transAxes, fig=fig, y=top_pad, units='points'))
            panel += 1
            if row == 0:
                ax.set_title(P.exp_labels[exp], fontsize=pt(9), fontweight='bold', pad=top_pad)
            if col == 0 and not mld_only:   # MLD-only: titles and colour bars say MLD
                ax.text(-0.05, 0.5, 'MLD' if var == 'MLD1' else 'Sea Ice', transform=ax.transAxes,
                        fontsize=pt(9), fontweight='bold', rotation=90, va='center', ha='right')
            if mld_only:
                grid_labels(ax, fig)

    def cbar(row, c0, c1, label, frac=0.8):
        p0, p1 = axes[(row, c0)].get_position(), axes[(row, c1)].get_position()
        w = (p1.x1 - p0.x0) * frac
        x0 = (p0.x0 + p1.x1 - w) / 2
        gap_in = 0.22 if mld_only else 0.07          # inches below the maps
        gap, height = gap_in / fig.get_figheight(), 0.09 / fig.get_figheight()
        cax = fig.add_axes([x0, p0.y0 - gap - height, w, height])
        cb = fig.colorbar(ims[(row, c0)], cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=pt(8.5), fontweight='bold', labelpad=2)
        cb.ax.tick_params(labelsize=pt(8), labelcolor='black', width=0.6 / SCALE, pad=2)
        cb.outline.set_linewidth(0.6 / SCALE)
        return cb

    cb = cbar(0, 0, 2, 'MLD (m), PI, MH, LIG', frac=0.9)
    cb.set_ticks([0, 50, 200, 400, 600, 800, 1000])
    if mld_only:                  # the d-e bar is long enough here for all labels
        cb = cbar(1, 0, 1, 'MLD (m), LGM, MIS3', frac=0.9)
        cb.set_ticks([0, 50, 100, 200, 300, 400])
    else:
        cb = cbar(0, 3, 4, 'MLD (m), LGM, MIS3', frac=1.0)
        cb.set_ticks([0, 50, 100, 200, 300, 400])
        # at print size the 0 and 50 labels run together on this shorter bar; the 0 tick
        # mark stays, its label is left out (the a-c bar beside it shows 0 and 50)
        cb.set_ticklabels(['', '50', '100', '200', '300', '400'])
    if not mld_only:
        cbar(1, 0, 4, 'Sea Ice (fraction)', frac=0.6)

    out = OUT + ('_mldonly' if mld_only else '')
    for ext, dpi in (('pdf', 300), ('png', 690)):   # png keeps the old ~5500 px width
        plt.savefig(f'{out}.{ext}', dpi=dpi, bbox_inches='tight')
        print('saved', f'{out}.{ext}')
    plt.close()


if __name__ == '__main__':
    main(mld_only='--mld-only' in sys.argv[1:])
