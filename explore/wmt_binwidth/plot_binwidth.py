#!/usr/bin/env python
"""Trial figures for the user's question (11-27 11:43): what does manuscript Fig. 2 look like with
narrower sigma2 bins? Reads data/G_bw_<exp>.nc (calc_binwidth.py). Writes, in this folder,
  fig2_binwidth_test_<w>.png/pdf   Fig. 2 redrawn at bin width w (0p20 = the published figure)
  fig2_binwidth_test_overlay.png    total WMT at all four widths in the Fig. 2 layout
  fig2_binwidth_test_zoomPI.png     PI Southern Ocean (Fig. 2a), all terms, 36.3-37.3, four widths
Sizes are for a 5.81 in wide print (A4 letter): figures 7.0 in wide, scale 0.83, ticks print 7.6 pt."""
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPS = [('pi', 'PI'), ('mh', 'MH'), ('lig', 'LIG'), ('lgm', 'LGM'), ('mis', 'MIS3')]
REGS = ['Southern\nOcean', 'Ross\nSea', 'Weddell\nSea', 'Adélie\nLand']
W = [('0p20', 0.2), ('0p10', 0.1), ('0p05', 0.05), ('0p02', 0.02)]
COL = {'tot': '#111111', 'heat': '#D7263D', 'ice': '#1B998B', 'oth': '#2E86AB'}
WCOL = {'0p20': '#111111', '0p10': '#E69F00', '0p05': '#0072B2', '0p02': '#999999'}
WLW = {'0p20': 2.0, '0p10': 1.4, '0p05': 1.4, '0p02': 0.9}
FS_TK, FS_LB, FS_TI = 9.2, 10.0, 11.0
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.9,
                     'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'text.color': 'black',
                     'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})

D = {}
for e, _ in EXPS:
    g = nc.Dataset(f'{HERE}/data/G_bw_{e}.nc')
    for k, _w in W:
        G = g[f'G_{k}'][:].astype(float)            # year, region, comp(heat, ice, other), bin
        tot = G.sum(axis=2)
        sd = G.std(axis=0)
        D[(e, k)] = dict(c=g[f'center_{k}'][:], heat=G[:, :, 0].mean(0), ice=G[:, :, 1].mean(0),
                         oth=G[:, :, 2].mean(0), tot=tot.mean(0),
                         sd_heat=sd[:, 0], sd_ice=sd[:, 1], sd_oth=sd[:, 2],
                         sd_tot=np.sqrt((sd ** 2).sum(axis=1)))     # as the published figure (independent terms)
    g.close()


def frame(ax, e, ri, ci, letter):
    ax.set_xlim(*((35.5, 37.2) if e in ('pi', 'mh', 'lig') else (36.0, 38.5)))
    ax.axhline(0, color='#444444', lw=0.6, zorder=1)
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.tick_params(labelsize=FS_TK, length=3, pad=2)
    ax.text(0.04, 0.95, f'({letter})', transform=ax.transAxes, fontsize=FS_LB, fontweight='bold', va='top')
    if ri == 3:
        ax.set_xlabel(r'$\sigma_2$ (kg m$^{-3}$)', fontsize=FS_LB, fontweight='bold', labelpad=2)
    else:
        ax.tick_params(labelbottom=False)
    if ci == 0:
        ax.set_ylabel('WMT (Sv)', fontsize=FS_LB, fontweight='bold', labelpad=2)
        ax.text(-0.62, 0.5, REGS[ri], transform=ax.transAxes, rotation=90, ha='center', va='center',
                fontsize=FS_LB, fontweight='bold')
    if ri == 0:
        ax.set_title(EXPS[ci][1], fontsize=FS_TI, fontweight='bold', pad=4)


def grid(title):
    fig, axes = plt.subplots(4, 5, figsize=(7.0, 6.9))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.15, hspace=0.18, wspace=0.42)
    fig.suptitle(title, fontsize=FS_TI, fontweight='bold', y=0.985)
    return fig, axes


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{HERE}/{name}.{ext}', dpi=400)
    plt.close(fig)
    print('wrote', name)


# --------------------------------------------------------------- Fig. 2 at each width
for k, w in W:
    fig, axes = grid(r'Fig. 2 recomputed with $\sigma_2$ bins %.2f kg m$^{-3}$ wide%s'
                     % (w, ' (as published)' if k == '0p20' else ''))
    hs = []
    for ri in range(4):
        for ci, (e, _) in enumerate(EXPS):
            ax, d = axes[ri, ci], D[(e, k)]
            for key, lab, lw in (('tot', 'Total WMT', 1.9), ('heat', 'Heat', 1.5), ('ice', 'Sea-ice FW', 1.4),
                                 ('oth', 'Other FW', 1.4)):
                ax.fill_between(d['c'], d[key][ri] - d['sd_' + key][ri], d[key][ri] + d['sd_' + key][ri],
                                color=COL[key], alpha=0.18 if key != 'tot' else 0.25, lw=0)
                h, = ax.plot(d['c'], d[key][ri], color=COL[key], lw=lw, label=lab, zorder=5 if key == 'tot' else 4)
                if ri == 0 and ci == 0:
                    hs.append(h)
            frame(ax, e, ri, ci, chr(97 + ri * 5 + ci))
    fig.legend(handles=hs, loc='lower center', ncol=4, fontsize=FS_LB, frameon=False, bbox_to_anchor=(0.55, 0.0))
    save(fig, f'fig2_binwidth_test_{k}')

# --------------------------------------------------------------- overlay of the totals
fig, axes = grid(r'Total WMT of Fig. 2 for four $\sigma_2$ bin widths')
hs = []
for ri in range(4):
    for ci, (e, _) in enumerate(EXPS):
        ax = axes[ri, ci]
        for k, w in W[::-1]:
            d = D[(e, k)]
            h, = ax.plot(d['c'], d['tot'][ri], color=WCOL[k], lw=WLW[k], label='%.2f' % w,
                         zorder=6 if k == '0p20' else 5)
            if ri == 0 and ci == 0:
                hs.append(h)
        frame(ax, e, ri, ci, chr(97 + ri * 5 + ci))
leg = fig.legend(handles=hs[::-1], loc='lower center', ncol=4, fontsize=FS_LB, frameon=False,
                 bbox_to_anchor=(0.55, 0.0), title=r'bin width (kg m$^{-3}$)', title_fontsize=FS_LB)
save(fig, 'fig2_binwidth_test_overlay')

# --------------------------------------------------------------- zoom on Fig. 2a
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.30, wspace=0.08)
hs = []
for i, (k, w) in enumerate(W):
    ax, d = axes[i], D[('pi', k)]
    for key, lab, lw in (('tot', 'Total WMT', 1.9), ('heat', 'Heat', 1.5), ('ice', 'Sea-ice FW', 1.4),
                         ('oth', 'Other FW', 1.4)):
        h, = ax.plot(d['c'], d[key][0], color=COL[key], lw=lw, label=lab, marker='o', ms=1.6)
        if i == 0:
            hs.append(h)
    ax.set_xlim(36.3, 37.3)
    ax.axvline(37.0, color='#666666', lw=0.6, ls=':')
    ax.axhline(0, color='#444444', lw=0.6)
    ax.grid(True, alpha=0.25, ls='--', lw=0.5)
    ax.tick_params(labelsize=FS_TK, length=3, pad=2)
    ax.set_title('(%s) bins %.2f' % (chr(97 + i), w), fontsize=FS_LB, fontweight='bold', pad=3)
    ax.set_xlabel(r'$\sigma_2$ (kg m$^{-3}$)', fontsize=FS_LB, fontweight='bold', labelpad=2)
axes[0].set_ylabel('WMT (Sv)', fontsize=FS_LB, fontweight='bold')
fig.suptitle('PI, Southern Ocean south of 60°S, JJA (Fig. 2a); dots mark the bin centres', fontsize=FS_LB,
             fontweight='bold', y=0.98)
fig.legend(handles=hs, loc='lower center', ncol=4, fontsize=FS_LB, frameon=False, bbox_to_anchor=(0.53, -0.01))
save(fig, 'fig2_binwidth_test_zoomPI')
