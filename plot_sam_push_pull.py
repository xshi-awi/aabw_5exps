#!/usr/bin/env python
"""
Figure for Reviewer 3: the Southern Annular Mode acts on both the "push"
(Ekman divergence, upwelling) and the "pull" (salt pump via coastal brine input),
in every simulated climate state.

(a) SAM composite Ekman transport anomaly at 60 S
(b) SAM composite coastal brine-equivalent freshwater loss anomaly (<65 S)
(c) SAM composite peak WMT anomaly, for reference
2026-11-26 (user): the scatter of push against pull (old panel c, r = 0.77)
was dropped; the WMT panel is now (c). Old version figures/versions/*_v20261126pre.*
2026-11-27 (A4 readability): drawn at the printed width 5.81 in, so every size is the
printed size in pt. (a) and (b) side by side, (c) below at full width, because the
sigma_2 labels of (c) need about 34 pt per bar at 8 pt. Old: *_v20261127pre.*
2026-11-27 12:04 (user): WMT in 0.1 sigma_2 bins; (c) reads the new composites, the
sigma_2 of the maximum is a bin centre (x.x5) and is printed with two decimals.
Old 0.2-bin version: *_v20261127bin0p2.*
"""
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import xarray as xr

LABELS = ['PI', 'MH', 'LIG', 'LGM', 'MIS3']
EXPS = ['pi', 'mh', 'lig', 'lgm', 'mis']
# Okabe-Ito, CVD safe
COLS = {'PI': '#000000', 'MH': '#E69F00', 'LIG': '#009E73',
        'LGM': '#0072B2', 'MIS3': '#CC79A7'}
MARK = {'PI': 'o', 'MH': 's', 'LIG': '^', 'LGM': 'D', 'MIS3': 'v'}

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': 'k', 'axes.linewidth': 0.8, 'text.color': 'k', 'axes.labelcolor': 'k',
    'xtick.color': 'k', 'ytick.color': 'k', 'xtick.major.width': 0.8,
    'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
})
PRINT_W = 5.81          # in, width of the figure in the letter
PNG_DPI = 534           # 5.81 in x 534 = 3102 px, about the old 3100 px
FS_TICK, FS_LAB, FS_TITLE, FS_NUM = 8, 8.5, 9, 8     # printed pt

rows = json.loads(open('sam_push_pull.json').read())
R = {r['exp']: r for r in rows}


def peak_wmt_anom(exp):
    """Peak SAM composite WMT anomaly for the Southern Ocean, Sv."""
    d = xr.open_dataset(f'composite_sam/wmt_sam_composite_southern_ocean_{exp}.nc')
    sig = d['sigma2'].values if 'sigma2' in d else d['sigma2_l_target'].values

    def g(n):
        k = f'{n}_diff'
        return -d[k].values / 1e9 if k in d else np.zeros_like(sig)

    tot = (g('total_heat_surface_exchange_flux_nonadvective_heat')
           + g('surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt')
           + g('surface_ocean_flux_advective_negative_rhs_evaporation_salt')
           + g('surface_ocean_flux_advective_negative_rhs_snow_salt')
           + g('surface_ocean_flux_advective_negative_rhs_rain_and_ice_salt')
           + g('surface_ocean_flux_advective_negative_rhs_rivers_salt'))
    m = (sig >= 35.0) & (sig <= 39.0)
    idx = np.where(m)[0]
    i = idx[int(np.nanargmax(tot[idx]))]
    d.close()
    return tot[i], sig[i]


wmt = {lab: peak_wmt_anom(e) for e, lab in zip(EXPS, LABELS)}

# layout in inches: (a) and (b) on top, (c) at full width below
FW = PRINT_W
ax_h, xt, t2, t1, gap, bot = 1.42, 0.22, 0.40, 0.26, 0.10, 0.04
left, mid, right = 0.60, 0.62, 0.03
FH = t2 + ax_h + xt + gap + t1 + ax_h + xt + bot
wa = (FW - left - mid - right) / 2
y_c = bot + xt
y_ab = y_c + ax_h + t1 + gap + xt
fig = plt.figure(figsize=(FW, FH))
axes = [fig.add_axes([left / FW, y_ab / FH, wa / FW, ax_h / FH]),
        fig.add_axes([(left + wa + mid) / FW, y_ab / FH, wa / FW, ax_h / FH]),
        fig.add_axes([left / FW, y_c / FH, (FW - left - right) / FW, ax_h / FH])]
x = np.arange(len(LABELS))


def style(ax, ylabel, title, ystep):
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))      # same y ticks as before 11-27
    ax.set_ylabel(ylabel, fontsize=FS_LAB, fontweight='bold', labelpad=3, linespacing=1.1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=4, linespacing=1.1)
    ax.axhline(0, color='k', lw=0.8)
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.tick_params(labelsize=FS_TICK)


# ---------------------------------------------------------------- (a) push
ax = axes[0]
vals = [R[l]['d_ekman'] for l in LABELS]
ax.bar(x, vals, color=[COLS[l] for l in LABELS], edgecolor='none', width=0.62)
style(ax, '$\\Delta$ Ekman transport\nat 60°S  (Sv)', '(a)  The push:\nwind-driven upwelling', 2)
for i, v in enumerate(vals):
    ax.text(i, v + 0.35, f'{v:.1f}', ha='center', fontsize=FS_NUM)
ax.set_ylim(0, max(vals) * 1.24)

# ---------------------------------------------------------------- (b) pull
ax = axes[1]
vals2 = [R[l]['d_fwc'] * 1000 for l in LABELS]   # mSv
ax.bar(x, vals2, color=[COLS[l] for l in LABELS], edgecolor='none', width=0.62)
style(ax, '$\\Delta$ coastal brine\ninput  (mSv)', '(b)  The pull:\ncoastal salt pump', 10)
for i, v in enumerate(vals2):
    ax.text(i, v + 1.1, f'{v:.0f}', ha='center', fontsize=FS_NUM)
ax.set_ylim(0, max(vals2) * 1.24)

# ---------------------------------------------------------------- (c) WMT
ax = axes[2]
vals3 = [wmt[l][0] for l in LABELS]
ax.bar(x, vals3, color=[COLS[l] for l in LABELS], edgecolor='none', width=0.62)
style(ax, '$\\Delta$ peak\ntransformation  (Sv)', '(c)  The response: dense water formation', 2)
for i, l in enumerate(LABELS):
    ax.text(i, vals3[i] + 0.22, f'{vals3[i]:.1f}\n$\\sigma_2$={wmt[l][1]:.2f}',   # 0.1 bins: centres at x.x5
            ha='center', fontsize=FS_NUM, linespacing=1.1)
ax.set_ylim(0, max(vals3) * 1.34)

fig.savefig('figures/figR4_sam_push_pull.pdf', dpi=400)
fig.savefig('figures/figR4_sam_push_pull.png', dpi=PNG_DPI)
print('saved figures/figR4_sam_push_pull.{pdf,png}')

print()
print('%-6s %10s %12s %12s' % ('exp', 'dEkman_Sv', 'dbrine_mSv', 'dWMT_Sv'))
for l in LABELS:
    print('%-6s %10.2f %12.1f %12.2f' % (l, R[l]['d_ekman'], R[l]['d_fwc'] * 1000, wmt[l][0]))
r = np.corrcoef([R[l]['d_ekman'] for l in LABELS], [R[l]['d_fwc'] for l in LABELS])[0, 1]
print(f'\ncorrelation between the push and the pull across the five states: r = {r:.2f}')
