#!/usr/bin/env python
"""
Transformation in the seasonal sea ice zone, five climate states, in fixed
0.5 kg/m3 sigma2 bins.

Top row: total in black, the thermal contribution in red, and the haline
contribution in blue. Haline is the sum of the sea ice and the other freshwater
terms, so total = thermal + haline exactly and the three lines can be read
against each other without further arithmetic.

Bottom row: the thermal share of each bin.

This version uses our own sigma2 with no conversion to neutral density and no
alignment between states. Absolute densities are therefore directly comparable
across panels, and the glacial shift toward denser classes is visible as a shift
along the axis rather than being removed. Glacial surface water is denser, so
the glacial states populate bins that the interglacials barely reach and leave
the lightest bins empty; that is a result, not an artefact of binning.

Domain is each state's own September 15% ice contour, the Pellichero-comparable
domain, and the average is annual.

Outputs figures/figR16_siz_bins_5exps.{pdf,png}.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

EXPS = [('pi', 'PI'), ('mh', 'MH'), ('lig', 'LIG'), ('lgm', 'LGM'), ('mis', 'MIS3')]

C_TOT = '#000000'
C_THERM = '#D62728'
C_HAL = '#1F5FD0'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})

EDGES = np.arange(35.5, 38.51, 0.5)
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])


def load(exp):
    d = xr.open_dataset(f'wmt_siz_{exp}.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    V = lambda n: -d[n].values.mean(axis=0) / 1e9
    h = V('siz_total_heat_heat')

    def g(n):
        k = f'siz_surface_ocean_flux_advective_negative_rhs_{n}_salt'
        return V(k) if k in d else np.zeros_like(h)

    hal = (g('sea_ice_melt') + g('evaporation') + g('snow')
           + g('rain_and_ice') + g('rivers'))
    return sig, h, hal, d.attrs['siz_area_m2']


def binned(exp):
    sig, h, hal, area = load(exp)
    T, H, A = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (sig >= lo) & (sig < hi)
        th, ha = np.nansum(h[m]), np.nansum(hal[m])
        H.append(th)
        A.append(ha)
        T.append(th + ha)
    T, H, A = np.array(T), np.array(H), np.array(A)
    den = np.abs(H) + np.abs(A)
    share = np.where(den > 1e-9, np.abs(H) / np.where(den > 0, den, 1) * 100, np.nan)
    return T, H, A, share, area


fig, axes = plt.subplots(2, 5, figsize=(22.5, 9.9), sharex=True)

for j, (exp, lab) in enumerate(EXPS):
    T, H, A, share, area = binned(exp)

    # ---------------------------------------------------------- top
    ax = axes[0, j]
    ax.axhline(0, color='#666666', lw=0.9, zorder=1)
    ax.plot(CENT, T, '-o', color=C_TOT, lw=2.8, ms=7, label='Total', zorder=5)
    ax.plot(CENT, H, '-s', color=C_THERM, lw=2.2, ms=6, label='Thermal', zorder=4)
    ax.plot(CENT, A, '-^', color=C_HAL, lw=2.2, ms=6,
            label='Haline (sea ice + other FW)', zorder=3)
    ax.set_title('%s\nice zone %.2f$\\times$10$^{13}$ m$^2$' % (lab, area / 1e13),
                 fontsize=13.5, fontweight='bold')
    if j == 0:
        ax.set_ylabel('Annual-mean transformation (Sv)', fontsize=12.5)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=11)

    # ---------------------------------------------------------- bottom
    ax = axes[1, j]
    ax.axhspan(0, 33, color='#009E73', alpha=0.11, zorder=0)
    ax.axhline(50, color='#999999', lw=1.0, ls=':', zorder=1)
    ok = np.isfinite(share)
    ax.plot(CENT[ok], share[ok], '-o', color=C_TOT, lw=2.6, ms=7)
    for c, v, t in zip(CENT, share, T):
        if np.isfinite(v) and abs(t) > 0.05:
            ax.annotate('%.0f' % v, xy=(c, v), xytext=(0, 13 if v < 45 else -19),
                        textcoords='offset points', ha='center',
                        fontsize=9.5, fontweight='bold',
                        bbox=dict(boxstyle='square,pad=0.08', fc='white',
                                  ec='none', alpha=0.75))
    ax.set_ylim(0, 100)
    ax.set_xlim(EDGES[0] - 0.1, EDGES[-1] + 0.1)
    ax.set_xticks(np.arange(35.5, 38.6, 0.5))
    ax.set_xlabel('$\\sigma_2$ (kg m$^{-3}$)', fontsize=12.5)
    if j == 0:
        ax.set_ylabel('Thermal share (%)', fontsize=12.5)
        ax.text(0.03, 26, 'haline dominated', fontsize=10.5, color='#00654a',
                transform=ax.get_yaxis_transform())
    ax.grid(axis='y', alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=11, labelrotation=45)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.935),
           ncol=3, frameon=False, fontsize=12)

fig.suptitle('Seasonal sea ice zone, annual mean, in 0.5 kg m$^{-3}$ '
             '$\\sigma_2$ bins', fontsize=16, fontweight='bold')
# one scale for the interglacials and one for the glacials: a single scale for
# all five would flatten PI/MH/LIG, and independent scales would hide that the
# glacial excursions are five times larger
for jj in range(3):
    axes[0, jj].set_ylim(-17, 24)
for jj in (3, 4):
    axes[0, jj].set_ylim(-72, 24)

fig.subplots_adjust(left=0.052, right=0.99, top=0.845, bottom=0.115,
                    wspace=0.24, hspace=0.20)
fig.savefig('figures/figR16_siz_bins_5exps.pdf', dpi=400)
fig.savefig('figures/figR16_siz_bins_5exps.png', dpi=200)
print('saved figures/figR16_siz_bins_5exps.{pdf,png}')

print()
hdr = '%-6s %-13s %9s %9s %9s %8s'
print(hdr % ('exp', 'sigma2 bin', 'total', 'thermal', 'haline', 'therm%'))
for exp, lab in EXPS:
    T, H, A, share, area = binned(exp)
    for c, lo, hi in zip(CENT, EDGES[:-1], EDGES[1:]):
        k = list(CENT).index(c)
        if abs(T[k]) > 0.02:
            print(hdr % (lab, '%.1f-%.1f' % (lo, hi), '%.2f' % T[k],
                         '%.2f' % H[k], '%.2f' % A[k],
                         '%.0f%%' % share[k] if np.isfinite(share[k]) else 'n/a'))
