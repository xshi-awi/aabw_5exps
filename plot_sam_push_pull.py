#!/usr/bin/env python
"""
Figure for Reviewer 3: the Southern Annular Mode acts on both the "push"
(Ekman divergence, upwelling) and the "pull" (salt pump via coastal brine input),
in every simulated climate state.

(a) SAM composite Ekman transport anomaly at 60 S
(b) SAM composite coastal brine-equivalent freshwater loss anomaly (<65 S)
(c) the two against each other, showing they co-vary across states
(d) SAM composite peak WMT anomaly, for reference
"""
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
    'axes.edgecolor': '#333333',
})

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

fig, axes = plt.subplots(1, 4, figsize=(20, 5.0))
x = np.arange(len(LABELS))

# ---------------------------------------------------------------- (a) push
ax = axes[0]
vals = [R[l]['d_ekman'] for l in LABELS]
ax.bar(x, vals, color=[COLS[l] for l in LABELS], edgecolor='none', width=0.62)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylabel('$\\Delta$ Ekman transport at 60°S  (Sv)', fontsize=13)
ax.set_title('(a)  The push:\nwind-driven upwelling', fontsize=15, fontweight='bold')
ax.axhline(0, color='#444444', lw=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.35, f'{v:.1f}', ha='center', fontsize=11.5)
ax.set_ylim(0, max(vals) * 1.24)

# ---------------------------------------------------------------- (b) pull
ax = axes[1]
vals2 = [R[l]['d_fwc'] * 1000 for l in LABELS]   # mSv
ax.bar(x, vals2, color=[COLS[l] for l in LABELS], edgecolor='none', width=0.62)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylabel('$\\Delta$ coastal brine input  (mSv)', fontsize=13)
ax.set_title('(b)  The pull:\ncoastal salt pump', fontsize=15, fontweight='bold')
ax.axhline(0, color='#444444', lw=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.6)
for i, v in enumerate(vals2):
    ax.text(i, v + 1.1, f'{v:.0f}', ha='center', fontsize=11.5)
ax.set_ylim(0, max(vals2) * 1.24)

# ---------------------------------------------------------------- (c) both
ax = axes[2]
for l in LABELS:
    ax.scatter(R[l]['d_ekman'], R[l]['d_fwc'] * 1000, s=190, color=COLS[l],
               marker=MARK[l], edgecolor='white', linewidth=1.4, zorder=5, label=l)
ax.set_xlabel('$\\Delta$ Ekman transport  (Sv)', fontsize=13)
ax.set_ylabel('$\\Delta$ coastal brine input  (mSv)', fontsize=13)
ax.set_title('(c)  Push and pull\nrespond together', fontsize=15, fontweight='bold')
ax.grid(alpha=0.25, lw=0.6)
ax.legend(frameon=False, fontsize=11, loc='upper left')

# ---------------------------------------------------------------- (d) WMT
ax = axes[3]
vals3 = [wmt[l][0] for l in LABELS]
ax.bar(x, vals3, color=[COLS[l] for l in LABELS], edgecolor='none', width=0.62)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylabel('$\\Delta$ peak transformation  (Sv)', fontsize=13)
ax.set_title('(d)  The response:\ndense water formation', fontsize=15, fontweight='bold')
ax.axhline(0, color='#444444', lw=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.6)
for i, l in enumerate(LABELS):
    ax.text(i, vals3[i] + 0.22, f'{vals3[i]:.1f}\n$\\sigma_2$={wmt[l][1]:.1f}',
            ha='center', fontsize=10.5)
ax.set_ylim(0, max(vals3) * 1.34)

for a in axes:
    a.tick_params(labelsize=12)

fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.13, wspace=0.34)
fig.savefig('figures/figR4_sam_push_pull.pdf', dpi=400)
fig.savefig('figures/figR4_sam_push_pull.png', dpi=200)
print('saved figures/figR4_sam_push_pull.{pdf,png}')

print()
print('%-6s %10s %12s %12s' % ('exp', 'dEkman_Sv', 'dbrine_mSv', 'dWMT_Sv'))
for l in LABELS:
    print('%-6s %10.2f %12.1f %12.2f' % (l, R[l]['d_ekman'], R[l]['d_fwc'] * 1000, wmt[l][0]))
r = np.corrcoef([R[l]['d_ekman'] for l in LABELS], [R[l]['d_fwc'] for l in LABELS])[0, 1]
print(f'\ncorrelation between the push and the pull across the five states: r = {r:.2f}')
