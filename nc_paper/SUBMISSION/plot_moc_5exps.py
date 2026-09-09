#!/usr/bin/env python
"""
Global meridional overturning streamfunction for the five climate states,
from the model's own online MOC diagnostic (diag_gmoc.nc), averaged over the
final 20 diagnostic years.

Requested by Reviewer #3, who noted that the overturning can be computed
directly in the model and should be shown rather than inferred from ideal age.

Row 1: absolute streamfunction, all five states.
Row 2: anomalies relative to PI for MH, LIG, LGM, MIS3.
"""
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
PROD = '/work/ba1066/a270064/production'
NAVG = 20

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'pdf.fonttype': 42,
})


def load(exp):
    f = nc.Dataset(f'{PROD}/{exp}_age/diag_gmoc.nc')
    moc = np.array(f.variables['MOC'][:])
    lats = np.array(f.variables['lats'][:])
    deps = np.array(f.variables['deps'][:])
    return moc[-NAVG:].mean(axis=0), lats, deps


data = {e: load(e) for e in EXPS}

fig = plt.figure(figsize=(21, 9.4))
gs = fig.add_gridspec(2, 5, hspace=0.72, wspace=0.16,
                      left=0.055, right=0.985, top=0.92, bottom=0.17)

lev_abs = np.arange(-24, 24.1, 2)
lev_anm = np.arange(-12, 12.1, 1)
letters = 'abcdefghi'
p = 0

pi_m, pi_lat, pi_dep = data['pi']

for c, (e, lab) in enumerate(zip(EXPS, LABELS)):
    m, lats, deps = data[e]
    ax = fig.add_subplot(gs[0, c])
    cf = ax.contourf(lats, -deps, m, levels=lev_abs, cmap='RdBu_r',
                     norm=TwoSlopeNorm(vcenter=0, vmin=-24, vmax=24),
                     extend='both')
    ax.contour(lats, -deps, m, levels=lev_abs, colors='k',
               linewidths=0.25, alpha=0.35)
    ax.set_ylim(6000, 0)
    ax.set_xlim(-78, 80)
    ax.set_title(lab, fontsize=20, fontweight='bold', pad=8)
    ax.text(0.025, 0.055, f'({letters[p]})', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    p += 1
    # AABW cell strength
    kz = deps <= -2000
    js = lats < -30
    aabw = np.nanmin(m[np.ix_(kz, js)])
    ax.text(0.97, 0.055, f'AABW {aabw:.1f} Sv', transform=ax.transAxes,
            ha='right', fontsize=13, color='#111111')
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=15)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Latitude', fontsize=13)

cax = fig.add_axes([0.30, 0.545, 0.42, 0.015])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',
                  ticks=np.arange(-24, 25, 8))
cb.set_label('Overturning streamfunction (Sv)', fontsize=14)
cb.ax.tick_params(labelsize=12)

for c, (e, lab) in enumerate(zip(EXPS[1:], LABELS[1:])):
    m, lats, deps = data[e]
    ax = fig.add_subplot(gs[1, c + 1])
    anm = m - pi_m
    cf2 = ax.contourf(lats, -deps, anm, levels=lev_anm, cmap='PuOr_r',
                      norm=TwoSlopeNorm(vcenter=0, vmin=-12, vmax=12),
                      extend='both')
    ax.contour(lats, -deps, anm, levels=lev_anm, colors='k',
               linewidths=0.25, alpha=0.35)
    ax.set_ylim(6000, 0)
    ax.set_xlim(-78, 80)
    ax.set_title(f'{lab} $-$ PI', fontsize=18, fontweight='bold', pad=8)
    ax.text(0.025, 0.055, f'({letters[p]})', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    p += 1
    ax.set_xlabel('Latitude', fontsize=13)
    if c == 0:
        ax.set_ylabel('Depth (m)', fontsize=15)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=12)

cax2 = fig.add_axes([0.30, 0.062, 0.42, 0.015])
cb2 = fig.colorbar(cf2, cax=cax2, orientation='horizontal',
                   ticks=np.arange(-12, 13, 4))
cb2.set_label('Streamfunction anomaly (Sv)', fontsize=14)
cb2.ax.tick_params(labelsize=12)

fig.savefig('figures/figR2_moc_5exps.pdf', dpi=400)
fig.savefig('figures/figR2_moc_5exps.png', dpi=200)
print('saved figures/figR2_moc_5exps.{pdf,png}')

print()
print('%-6s %10s %10s' % ('EXP', 'AABW(Sv)', 'NADW(Sv)'))
for e, lab in zip(EXPS, LABELS):
    m, lats, deps = data[e]
    kz = deps <= -2000
    js = lats < -30
    aabw = np.nanmin(m[np.ix_(kz, js)])
    kn = (deps <= -500) & (deps >= -4000)
    jn = (lats > 0) & (lats < 70)
    nadw = np.nanmax(m[np.ix_(kn, jn)])
    print('%-6s %10.1f %10.1f' % (lab, aabw, nadw))
