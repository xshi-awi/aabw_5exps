#!/usr/bin/env python
"""
Formation rate for PI in the seasonal sea ice zone, drawn the way Pellichero
et al. (2018) draw their Fig. 3b.

This is the Pellichero-comparable domain, inside the September 15% ice contour,
and the annual mean, so the domain and the averaging both match theirs. Only the
density coordinate differs: ours is sigma2, theirs neutral density.

Formation is the convergence of the transformation, F = -dT/dsigma. Positive
means water accumulates in a class and must subduct; negative means it is
removed and must be supplied by upwelling. The sign convention is checked
against their own description, that classes lighter than 27.3 subduct while
27.3-27.9 upwells.

WHAT THIS FIGURE SHOWS, including the part that is unfavourable to us.

Our formation pattern is displaced toward lighter classes relative to theirs.
They find subduction in the light classes, upwelling through the middle, and
subduction again in the dense classes. We find subduction in the lightest
classes, upwelling over sigma2 36.1-36.7, and subduction from 36.9 upward. The
sense alternates the same way but the crossover sits at a lighter density, and
our magnitudes are several times smaller. Restricting the domain to the ice zone
does not remove this: the same pattern appears over the full domain south of
60 S. It follows from where the model convects, in the open Weddell gyre rather
than over the shelf, which displaces the whole transformation curve toward
lighter water.

Their published integrals are quoted as text rather than drawn as brackets over
our axis, because they are defined over neutral density ranges that our sigma2
axis cannot reproduce exactly.

Outputs figures/figR12b_formation_rate_siz.{pdf,png}.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

C_TOT = '#7F7F7F'
C_HEAT = '#D62728'
C_FW = '#1F4EF5'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})


def load():
    d = xr.open_dataset('wmt_siz_pi.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    V = lambda n: -d[n].values.mean(axis=0) / 1e9
    h = V('siz_total_heat_heat')

    def g(n):
        k = f'siz_surface_ocean_flux_advective_negative_rhs_{n}_salt'
        return V(k) if k in d else np.zeros_like(h)

    fw = (g('sea_ice_melt') + g('evaporation') + g('snow')
          + g('rain_and_ice') + g('rivers'))
    return sig, h, fw


SIG, HEAT, FW = load()
TOT = HEAT + FW

live = np.abs(TOT) + np.abs(HEAT) + np.abs(FW) > 1e-6
i0, i1 = np.where(live)[0][[0, -1]]
sl = slice(max(i0 - 1, 0), min(i1 + 2, len(SIG)))
sig, heat, fw, tot = SIG[sl], HEAT[sl], FW[sl], TOT[sl]
dsig = np.diff(sig).mean()


def formation(x):
    return -np.gradient(x, dsig) * dsig


f_tot, f_heat, f_fw = formation(tot), formation(heat), formation(fw)

fig, ax = plt.subplots(figsize=(13.5, 6.6))

w = dsig * 0.26
ax.bar(sig - w, f_tot, w * 0.95, color=C_TOT, label='Total', zorder=4)
ax.bar(sig, f_heat, w * 0.95, color=C_HEAT, label='Heat flux', zorder=4)
ax.bar(sig + w, f_fw, w * 0.95, color=C_FW, label='FW flux', zorder=4)
ax.axhline(0, color='#333333', lw=1.0, zorder=5)

# where our own curve changes sense, the analogue of their 27.6 divide
cross = None
for k in range(len(sig) - 1):
    if tot[k] < 0 <= tot[k + 1]:
        cross = sig[k] + dsig * (-tot[k]) / (tot[k + 1] - tot[k])
        break
if cross is not None:
    ax.axvline(cross, color=C_HEAT, ls='--', lw=1.6, zorder=3)
    ax.text(cross + 0.03, 0.97,
            'our lightening / densification\ndivide, $\\sigma_2$ = %.2f' % cross,
            transform=ax.get_xaxis_transform(), fontsize=10.5,
            color=C_HEAT, va='top')

lo, hi = sig.min() - 0.15, sig.max() + 0.15
ax.set_xlim(lo, hi)
ymax = max(abs(f_tot).max(), abs(f_heat).max(), abs(f_fw).max()) * 1.35
ax.set_ylim(-ymax, ymax)

ax.annotate('', xy=(lo + 0.18, ymax * 0.62), xytext=(lo + 0.18, ymax * 0.24),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#333333'))
ax.text(lo + 0.27, ymax * 0.43, 'Subduction', fontsize=12, va='center')
ax.annotate('', xy=(lo + 0.18, -ymax * 0.62), xytext=(lo + 0.18, -ymax * 0.24),
            arrowprops=dict(arrowstyle='->', lw=1.6, color='#333333'))
ax.text(lo + 0.27, -ymax * 0.43, 'Upwelling', fontsize=12, va='center')

ax.set_xlabel('$\\sigma_2$ (kg m$^{-3}$)', fontsize=13.5)
ax.set_ylabel('Water mass formation rate (Sv)', fontsize=13.5)
ax.set_title('PI formation rate, seasonal sea ice zone, annual mean',
             fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.22, lw=0.6)
ax.tick_params(labelsize=12)

ax.text(0.985, 0.035,
        'Observed (Pellichero et al. 2018, neutral density):\n'
        'subduction 22 $\\pm$ 4 Sv lighter than 27.3 $\\gamma$;  '
        'upwelling 27 $\\pm$ 7 Sv over 27.3$-$27.9 $\\gamma$;\n'
        'subduction 5 $\\pm$ 5 Sv denser than 27.9 $\\gamma$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        color='#444444', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.5', fc='#f4f4f4', ec='#cccccc'))

fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.125)
fig.savefig('figures/figR12b_formation_rate_siz.pdf', dpi=400)
fig.savefig('figures/figR12b_formation_rate_siz.png', dpi=200)
print('saved figures/figR12b_formation_rate_siz.{pdf,png}')

print()
print('%8s %9s %9s %9s %9s' % ('sigma2', 'T_total', 'F_total', 'F_heat', 'F_fw'))
for k in range(len(sig)):
    if abs(f_tot[k]) > 0.02 or abs(tot[k]) > 0.05:
        print('%8.1f %9.2f %9.2f %9.2f %9.2f'
              % (sig[k], tot[k], f_tot[k], f_heat[k], f_fw[k]))
if cross is not None:
    print('\nour lightening/densification divide: sigma2 = %.2f' % cross)
print('net subduction  (F>0): %+.2f Sv' % f_tot[f_tot > 0].sum())
print('net upwelling   (F<0): %+.2f Sv' % f_tot[f_tot < 0].sum())
