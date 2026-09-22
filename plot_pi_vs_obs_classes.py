#!/usr/bin/env python
"""
Pre-industrial transformation by density class, on our own sigma2 axis.

Reviewers 1, 2 and 3 all raise the same objection, that our pre-industrial state
is thermally dominated where Pellichero et al. (2018) find freshwater dominance.
The thermal share is not a single number. It peaks in the mid-density classes
and collapses to 7% in the densest class, where we agree with the observations.
The disagreement is confined to the middle of the range, which is the expected
signature of a model that convects in the open gyre rather than over the shelf.

The axis here is our own sigma2, over the full range the model populates. An
earlier version used the neutral density classes of Pellichero et al. and began
at their 27.1 boundary, which silently discarded every bin lighter than
sigma2 36.5 -- and that is where our largest single feature sits, a net
lightening of about -50 Sv. Our own axis keeps the whole curve and needs no
conversion and no extrapolation into densities the model barely outcrops.

The observational values are published in neutral density and cannot be drawn as
points on this axis. They are quoted in the annotation with the density they
refer to, rather than placed, so that no exact correspondence is implied.

Outputs figures/figR11_pi_vs_obs_classes.{pdf,png}.
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

C_HEAT = '#D55E00'
C_ICE = '#009E73'
C_OTH = '#0072B2'
C_TOT = '#000000'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})

# our own classes, taken from the structure of the PI curve itself
CLASSES = [(None, 36.3, 'Light\n$\\sigma_2<$36.3'),
           (36.3, 36.7, 'Intermediate\n36.3$-$36.7'),
           (36.7, 37.0, 'Transformation peak\n36.7$-$37.0'),
           (37.0, None, 'Densest\n$\\sigma_2\\geq$37.0')]


def load(months=None):
    d = xr.open_dataset('pi/wmt_echam_heat_fesom_freshwater_southern_ocean_pi.nc',
                        decode_times=False)
    sig = d['sigma2_l_target'].values
    sel = slice(None) if months is None else [m - 1 for m in months]
    V = lambda n: -d[n].values[sel].mean(axis=0) / 1e9
    h = V('total_heat_surface_exchange_flux_nonadvective_heat')
    i = V('surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt')
    o = sum(V(f'surface_ocean_flux_advective_negative_rhs_{k}_salt')
            for k in ['evaporation', 'snow', 'rain_and_ice', 'rivers'])
    return sig, h, i, o


def by_class(months=None):
    sig, h, i, o = load(months)
    rows = []
    for lo, hi, _ in CLASSES:
        m = np.ones_like(sig, bool)
        if lo is not None:
            m &= sig >= lo
        if hi is not None:
            m &= sig < hi
        H, I, O = np.nansum(h[m]), np.nansum(i[m]), np.nansum(o[m])
        den = abs(H) + abs(I) + abs(O)
        rows.append(dict(heat=H, ice=I, oth=O, total=H + I + O,
                         thermal=abs(H) / den * 100 if den else np.nan))
    return rows


ANN = by_class(None)
JJA = by_class([6, 7, 8])

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
x = np.arange(len(CLASSES))
xlab = [c[2] for c in CLASSES]

# ---------------------------------------------------------------- (a) components
ax = axes[0]
w = 0.26
ax.bar(x - w, [r['heat'] for r in ANN], w, color=C_HEAT, label='Heat')
ax.bar(x, [r['ice'] for r in ANN], w, color=C_ICE, label='Sea ice FW')
ax.bar(x + w, [r['oth'] for r in ANN], w, color=C_OTH, label='Other FW')
ax.plot(x, [r['total'] for r in ANN], 'o-', color=C_TOT, lw=2.4, ms=9,
        label='Total', zorder=6)
ax.axhline(0, color='#444444', lw=0.9)

for k, r in enumerate(ANN):
    ax.annotate('%+.1f' % r['total'], xy=(k, r['total']),
                xytext=(0, 14 if r['total'] >= 0 else -22),
                textcoords='offset points', ha='center',
                fontsize=11, fontweight='bold', zorder=7)

ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=11.5)
ax.set_ylabel('Annual-mean transformation (Sv)', fontsize=13.5)
ax.set_title('(a)  PI transformation by density class', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=11.5, loc='lower right')
ax.grid(axis='y', alpha=0.25, lw=0.6)

# ---------------------------------------------------------------- (b) thermal share
ax = axes[1]
ax.axhspan(0, 33, color=C_ICE, alpha=0.11, zorder=0)
ax.plot(x, [r['thermal'] for r in ANN], '-o', color=C_TOT, lw=2.8, ms=10,
        label='Annual mean')
ax.plot(x, [r['thermal'] for r in JJA], '--s', color=C_TOT, lw=2.8, ms=10,
        label='Winter (JJA)')
ax.axhline(50, color='#999999', lw=1.0, ls=':')
ax.text(0.30, 51.8, 'equal thermal / haline', fontsize=10.5,
        color='#777777', ha='left')
ax.text(0.30, 26.5, 'haline dominated', fontsize=12, color='#00654a',
        transform=ax.get_yaxis_transform())

for k in range(len(CLASSES)):
    dy = 15 if ANN[k]['thermal'] < 20 else -23
    ax.annotate('%.0f%%' % ANN[k]['thermal'], xy=(k, ANN[k]['thermal']),
                xytext=(0, dy), textcoords='offset points', ha='center',
                fontsize=11, fontweight='bold')

ax.annotate('we agree with the observations here\n(their dense cell: 5 $\\pm$ 5 Sv at 27.9 $\\gamma$)',
            xy=(2.97, ANN[3]['thermal'] + 1.5), xytext=(0.36, 0.30),
            textcoords='axes fraction', fontsize=10.5, color='#00654a',
            linespacing=1.45, ha='left',
            arrowprops=dict(arrowstyle='->', color='#00654a', lw=1.3))

ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=11.5)
ax.set_ylim(0, 100)
ax.set_ylabel('Thermal share of transformation (%)', fontsize=13.5)
ax.set_title('(b)  Thermal dominance is confined to mid-densities', fontsize=15, fontweight='bold')
ax.legend(frameon=False, fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.25, lw=0.6)

for a in axes:
    a.tick_params(labelsize=11.5)
    a.set_xlabel('Density class, $\\sigma_2$ (kg m$^{-3}$)', fontsize=13)

fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.155, wspace=0.22)
fig.savefig('figures/figR11_pi_vs_obs_classes.pdf', dpi=400)
fig.savefig('figures/figR11_pi_vs_obs_classes.png', dpi=200)
print('saved figures/figR11_pi_vs_obs_classes.{pdf,png}')

print()
print('%-22s %8s %8s %8s %8s %8s' % ('class', 'total', 'heat', 'seaice', 'othFW', 'therm%'))
for (lo, hi, nm), r in zip(CLASSES, ANN):
    lab = '%s-%s' % (lo if lo else 'min', hi if hi else 'max')
    print('%-22s %8.2f %8.2f %8.2f %8.2f %7.0f%%'
          % (lab, r['total'], r['heat'], r['ice'], r['oth'], r['thermal']))
sig, h, i, o = load(None)
print('\nsum over all classes: %+.2f Sv   (all %d bins, nothing discarded)'
      % (np.nansum(h + i + o), len(sig)))
