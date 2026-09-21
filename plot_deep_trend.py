#!/usr/bin/env python
"""
Deep-ocean equilibration, for Reviewer 3's question about whether the
simulations are at equilibrium.

Surface temperature equilibrates in decades but the abyss takes millennia, so a
flat global mean surface temperature is not by itself evidence that the deep
ocean has stopped adjusting. This shows the volume-weighted mean temperature and
salinity below 2000 m over all available output years, with the analysed final
100 years shaded.

Data from calc_deep_ocean_trend.py -> deep_ocean_trend.npz.
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
LABELS = {'pi': 'PI', 'mh': 'MH', 'lig': 'LIG', 'lgm': 'LGM', 'mis': 'MIS3'}
COL = {'pi': '#000000', 'mh': '#E69F00', 'lig': '#009E73',
       'lgm': '#0072B2', 'mis': '#CC79A7'}
LS = {'pi': '-', 'mh': '-', 'lig': '--', 'lgm': '-', 'mis': '--'}

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})

d = np.load('deep_ocean_trend.npz', allow_pickle=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# ---------------------------------------------------------------- (a) T
ax = axes[0]
for e in EXPS:
    yrs = d[f'{e}_years']
    t = d[f'{e}_temp']
    x = yrs - yrs[-1]                       # years before the end of the run
    ax.plot(x, t, color=COL[e], ls=LS[e], lw=2.2, label=LABELS[e])
ax.axvspan(-100, 0, color='#cccccc', alpha=0.45, zorder=0)
ax.text(-50, ax.get_ylim()[1], 'analysed\n100 yr', ha='center', va='top',
        fontsize=11, color='#444444')
ax.set_xlabel('Model years before end of run', fontsize=13)
ax.set_ylabel('Mean potential temperature below 2000 m (°C)', fontsize=13)
ax.set_title('(a)  Deep ocean temperature', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=12, ncol=2)
ax.grid(alpha=0.25, lw=0.6)

# ---------------------------------------------------------------- (b) S
ax = axes[1]
for e in EXPS:
    yrs = d[f'{e}_years']
    sa = d[f'{e}_salt']
    x = yrs - yrs[-1]
    ax.plot(x, sa, color=COL[e], ls=LS[e], lw=2.2, label=LABELS[e])
ax.axvspan(-100, 0, color='#cccccc', alpha=0.45, zorder=0)
ax.set_xlabel('Model years before end of run', fontsize=13)
ax.set_ylabel('Mean salinity below 2000 m', fontsize=13)
ax.set_title('(b)  Deep ocean salinity', fontsize=15, fontweight='bold')
ax.grid(alpha=0.25, lw=0.6)

# ---------------------------------------------------------------- (c) trends
ax = axes[2]
x = np.arange(len(EXPS))
vals = [float(d[f'{e}_dT']) for e in EXPS]
ax.bar(x, vals, color=[COL[e] for e in EXPS], width=0.6)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[e] for e in EXPS], fontsize=13)
ax.set_ylabel('Trend over the analysed 100 yr (K century$^{-1}$)', fontsize=13)
ax.set_title('(c)  Residual deep drift', fontsize=15, fontweight='bold')
ax.axhline(0, color='#444444', lw=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.006, f'{v:.3f}', ha='center', fontsize=11.5)
ax.set_ylim(0, max(vals) * 1.25)

for a in axes:
    a.tick_params(labelsize=12)

fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.13, wspace=0.30)
fig.savefig('figures/figR10_deep_trend.pdf', dpi=400)
fig.savefig('figures/figR10_deep_trend.png', dpi=200)
print('saved figures/figR10_deep_trend.{pdf,png}')

print()
print('%-6s %8s %12s %12s %14s' % ('exp', 'nyears', 'T_final', 'dT/100yr', 'dT over 100yr'))
for e in EXPS:
    yrs = d[f'{e}_years']; t = d[f'{e}_temp']
    print('%-6s %8d %12.4f %12.5f %14.4f'
          % (LABELS[e], len(yrs), t[-100:].mean(), float(d[f'{e}_dT']),
             float(d[f'{e}_dT'])))
