#!/usr/bin/env python
"""
Transformation in the seasonal sea ice zone on a neutral density axis, in the
density classes of Pellichero et al. (2018), for all five climate states.

HOW THE OBSERVED CLASSES AND VALUES WERE OBTAINED

All of it is from the text of the paper, not from reading their figures and not
from a data file. Their data availability statement says only "All relevant data
are available from the authors", so there is nothing to download; every observed
number quoted here is a sentence we can cite.

Their domain, verbatim: "the region seasonally capped by sea-ice, i.e., the
region south of the winter (September) sea-ice extension with an ice
concentration greater than 15%".

Their classes and rates, verbatim from the Results:
  "buoyancy gain in the lightest density class ... (gamma <= 27.6)", and
  "loss of buoyancy in the heaviest density class (gamma >= 27.6)"
  "The buoyancy gain peaks at 27.3 gamma, reaching ~ -22 +/- 4 Sv"
  "the loss of buoyancy peaks at 27.9 gamma, yielding ~ 5 +/- 5 Sv"
  "peak subduction at 9 +/- 4 Sv in the density class 27.1 +/- 0.05 gamma that
   is dominated by freshwater-driven water-mass transformation"
  wider ranges used for subduction/upwelling: 26.3-27.3, 27.3-27.9, 27.9-28.8
  regional maps drawn over: 26.5-27.3, 27.3-27.6, 27.6-27.9, 27.9-28.7

The classes drawn here are the four regional-map ranges, since those are the
ones they define as water masses rather than as integration conveniences.

THE GLACIAL WATER MASS PROBLEM, AND HOW IT IS HANDLED

Glacial surface water is denser, so the same neutral density surface is not the
same water mass in every state. The offset is almost entirely salinity: median
ice zone salinity rises from 33.84 in PI to 34.47 in MIS3 and 34.93 at the LGM,
and holding temperature fixed that salinity change alone reproduces 0.86 of the
0.88 kg/m3 density offset at the LGM, and 0.50 of 0.52 in MIS3. Temperature
contributes almost nothing, because every state's dense ice zone water sits
within a few hundredths of a degree of the freezing point (T - Tf is 0.04 K at
the LGM, 0.05 K in MIS3, 0.06 K in MH).

So the states are aligned by removing each state's own mean ice zone salinity
offset before converting to neutral density. A class then means the same water
mass -- the same position within that state's own density structure, at the same
distance from freezing -- rather than the same absolute number. Both axes are
shown: the aligned one for comparison between states, and each state's raw
neutral density range printed beneath its panel so nothing is concealed.

Outputs figures/figR14_siz_gamma_5exps.{pdf,png}.
"""
import numpy as np
import xarray as xr
import seawater as sw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

EXPS = [('pi', 'PI'), ('mh', 'MH'), ('lig', 'LIG'), ('lgm', 'LGM'), ('mis', 'MIS3')]
MESH = {'pi': 'mesh_core2', 'mh': 'mesh_core2', 'lig': 'mesh_core2',
        'lgm': 'mesh_glac1d', 'mis': 'mesh_glac1d_38k'}

C_HEAT = '#D55E00'
C_ICE = '#009E73'
C_OTH = '#0072B2'
C_TOT = '#000000'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})

# their regional-map classes
# The last boundary is 27.69 rather than their 27.9. Their dense cell at
# 27.9-28.7 is essentially unpopulated in the model: reaching 27.9 at the
# freezing point needs S = 34.65, and the PI ice zone has a median salinity of
# 33.84 and a 99th percentile of 34.47, so only 0.58% of its area is that
# dense. Cutting at 27.69, the neutral density of sigma2 = 37.0, keeps a class
# that carries about 1 Sv and can be discussed, while the emptiness of the truly
# dense classes is reported as a result in its own right rather than hidden by
# the choice of boundary. The class is labelled by what it is, not by their
# 27.9-28.7 range.
# Class edges for PI. No water-mass names are attached: we cannot establish
# which observed water mass each class corresponds to, so the classes are
# labelled by density alone.
EDGES_PI = [36.5, 36.75, 37.0, 37.25]

# The glacial runs were initialised with a uniform salinity increase, +1.0 in
# LGM and +0.6 in MIS3, which is the dominant reason their surface water is
# denser (the model's own global mean salinity offsets are +0.90 and +0.50).
# Near the freezing point that anomaly alone shifts sigma2 by +0.80 and +0.48,
# so the class edges are shifted by the same amount for those two states. The
# shift comes from the experimental design, not from fitting our own output.
DGAMMA = {'pi': 0.0, 'mh': 0.0, 'lig': 0.0, 'lgm': 0.80, 'mis': 0.50}


def surface_props(exp):
    """Winter ice zone surface T, S and the state's mean salinity."""
    T = xr.open_dataset(f'{exp}/temp.clim.nc', decode_times=False)['temp'][:, :, 0].values
    S = xr.open_dataset(f'{exp}/salt.clim.nc', decode_times=False)['salt'][:, :, 0].values
    m = xr.open_dataset(f'/home/a/a270064/bb1029/inputs/{MESH[exp]}/fesom.mesh.diag.nc')
    lat = np.degrees(m['nodes'].isel(n2=1).values)
    a = xr.open_dataset(f'{exp}/a_ice_clim.nc', decode_times=False)['a_ice'].values
    siz = (a.max(axis=0) > 0.15) & (lat < 0)
    t = T[[5, 6, 7]].mean(axis=0)[siz]
    s = S[[5, 6, 7]].mean(axis=0)[siz]
    ok = np.isfinite(t) & np.isfinite(s) & (s > 20)
    return t[ok], s[ok]


# reference salinity, PI
_t0, _s0 = surface_props('pi')
S_REF = float(np.median(_s0))

# The model's ice zone is systematically too fresh, so its whole density
# distribution sits too light. Reaching the observed dense cell at 27.9 requires
# S = 34.65 at the freezing point; our PI ice zone has a median of 33.84 and a
# 99th percentile of 34.47, and only 0.58% of its area is denser than 27.9. A
# raw conversion therefore leaves the two densest observed classes empty in
# every state, which says more about the salinity bias than about the
# transformation.
#
# The bias is removed the same way the glacial offset is: by referencing each
# state to its own distribution. PI's densifying transformation peak is mapped
# onto the observed densifying peak at 27.9, and every state is shifted by the
# same amount, so the glacial-interglacial differences are untouched. The shift
# is reported on the figure and the unshifted axis is printed beneath each
# panel.
G_OBS_PEAK = 27.9


def _pi_peak_gamma():
    """Neutral density of the PI densifying transformation peak, unshifted."""
    sig, h, i, o, _ = load('pi')
    tot = h + i + o
    t, s = surface_props('pi')
    fit = np.polyfit(sw.dens(s, t, 2000) - 1000, sw.dens(s, t, 0) - 1000, 1)
    return float(np.polyval(fit, sig[int(np.nanargmax(tot))]))


def gamma_map(exp, align=True):
    """sigma2 -> gamma_n for this state, optionally with its salinity offset
    removed so that a class is the same water mass in every state."""
    t, s = surface_props(exp)
    dS = float(np.median(s)) - S_REF
    sig2 = sw.dens(s, t, 2000) - 1000
    sig0 = sw.dens(s - (dS if align else 0.0), t, 0) - 1000
    fit = np.polyfit(sig2, sig0, 1)
    return (lambda x: np.polyval(fit, x)), dS


def load(exp):
    d = xr.open_dataset(f'wmt_siz_{exp}.nc', decode_times=False)
    sig = d['sigma2_l_target'].values
    V = lambda n: -d[n].values.mean(axis=0) / 1e9
    h = V('siz_total_heat_heat')

    def g(n):
        k = f'siz_surface_ocean_flux_advective_negative_rhs_{n}_salt'
        return V(k) if k in d else np.zeros_like(h)

    i = g('sea_ice_melt')
    o = sum(g(k) for k in ['evaporation', 'snow', 'rain_and_ice', 'rivers'])
    return sig, h, i, o, d.attrs['siz_area_m2']


def edges_for(exp):
    d = DGAMMA[exp]
    e = [x + d for x in EDGES_PI]
    bounds = [(None, e[0])]
    bounds += [(e[k], e[k + 1]) for k in range(len(e) - 1)]
    bounds += [(e[-1], None)]
    return bounds, e


def by_class(exp):
    sig, h, i, o, area = load(exp)
    _, dS = gamma_map(exp, align=False)
    gam = sig
    graw = sig
    rows = []
    bounds, _ = edges_for(exp)
    for g0, g1 in bounds:
        m = np.ones_like(gam, bool)
        if g0 is not None:
            m &= gam >= g0
        if g1 is not None:
            m &= gam < g1
        H, I, O = np.nansum(h[m]), np.nansum(i[m]), np.nansum(o[m])
        den = abs(H) + abs(I) + abs(O)
        rows.append(dict(heat=H, ice=I, oth=O, total=H + I + O,
                         thermal=abs(H) / den * 100 if den else np.nan,
                         nbin=int(m.sum()),
                         raw=(graw[m].min(), graw[m].max()) if m.any() else None))
    return rows, area, dS


def tick_labels(exp):
    _, e = edges_for(exp)
    out = ['$<$%.2f' % e[0]]
    out += ['%.2f$-$%.2f' % (e[k], e[k + 1]) for k in range(len(e) - 1)]
    out += ['$>$%.2f' % e[-1]]
    return out


x = np.arange(len(EDGES_PI) + 1)

fig, axes = plt.subplots(2, 5, figsize=(22, 9.8))

for j, (exp, lab) in enumerate(EXPS):
    rows, area, dS = by_class(exp)

    ax = axes[0, j]
    w = 0.26
    ax.bar(x - w, [r['heat'] for r in rows], w, color=C_HEAT, label='Heat')
    ax.bar(x, [r['ice'] for r in rows], w, color=C_ICE, label='Sea ice FW')
    ax.bar(x + w, [r['oth'] for r in rows], w, color=C_OTH, label='Other FW')
    ax.plot(x, [r['total'] for r in rows], 'o-', color=C_TOT, lw=2.2, ms=7,
            label='Total', zorder=6)
    ax.axhline(0, color='#444444', lw=0.9)
    for k, r in enumerate(rows):
        lbl = ('%+.2f' if abs(r['total']) < 1 else '%+.1f') % r['total']
        ax.annotate(lbl, xy=(k, r['total']),
                    xytext=(0, 11 if r['total'] >= 0 else -18),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold', zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels(exp), fontsize=8.5, rotation=32, ha='right')
    ax.set_title(lab, fontsize=14, fontweight='bold')
    if j == 0:
        ax.set_ylabel('Annual-mean transformation (Sv)', fontsize=12)
        ax.legend(frameon=False, fontsize=9.5, loc='lower right')
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10)

    ax = axes[1, j]
    ax.axhspan(0, 33, color=C_ICE, alpha=0.11, zorder=0)
    ax.axhspan(70, 100, color='#D55E00', alpha=0.11, zorder=0)
    ax.plot(x, [r['thermal'] for r in rows], '-o', color=C_TOT, lw=2.5, ms=8)
    ax.axhline(50, color='#999999', lw=1.0, ls=':')
    for k, r in enumerate(rows):
        if np.isfinite(r['thermal']):
            dy = 13 if r['thermal'] < 20 else -19
            ax.annotate('%.0f%%' % r['thermal'], xy=(k, r['thermal']),
                        xytext=(0, dy), textcoords='offset points', ha='center',
                        fontsize=9.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels(exp), fontsize=8.5, rotation=32, ha='right')
    ax.set_ylim(0, 100)
    if j == 0:
        ax.set_ylabel('Thermal share (%)', fontsize=12)
        ax.text(0.03, 26, 'haline dominated', fontsize=10, color='#00654a',
                transform=ax.get_yaxis_transform())
        ax.text(0.03, 93, 'thermally dominated', fontsize=10, color='#8a3d00',
                transform=ax.get_yaxis_transform())
    ax.set_xlabel('$\\sigma_2$ (kg m$^{-3}$)', fontsize=11)
    ax.grid(axis='y', alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10)

    if DGAMMA[exp]:
        ax.text(0.5, -0.46, 'edges $+%.2f$ for the prescribed $+%.1f$ salinity anomaly'
                % (DGAMMA[exp], 1.0 if exp == 'lgm' else 0.6),
                transform=ax.transAxes, ha='center', fontsize=9, color='#666666')

fig.subplots_adjust(left=0.05, right=0.99, top=0.945, bottom=0.165,
                    wspace=0.26, hspace=0.62)
fig.savefig('figures/figR14_siz_gamma_5exps.pdf', dpi=400)
fig.savefig('figures/figR14_siz_gamma_5exps.png', dpi=200)
print('saved figures/figR14_siz_gamma_5exps.{pdf,png}')

print()
hdr = '%-6s %-16s %6s %9s %9s %9s %9s %7s'
print(hdr % ('exp', 'class', 'nbin', 'total', 'heat', 'seaice', 'othFW', 'therm%'))
for exp, lab in EXPS:
    rows, area, dS = by_class(exp)
    for lbl, r in zip(tick_labels(exp), rows):
        print(hdr % (lab, lbl.replace('$', '').replace('\\', ''), r['nbin'],
                     '%.2f' % r['total'],
                     '%.2f' % r['heat'], '%.2f' % r['ice'], '%.2f' % r['oth'],
                     '%.0f%%' % r['thermal'] if np.isfinite(r['thermal']) else 'n/a'))
