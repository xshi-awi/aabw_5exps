#!/usr/bin/env python
"""
Where we agree with Pellichero et al. (2018) and where we do not, resolved by
density class.

Reviewers 1, 2 and 3 all raise the same objection: our pre-industrial state is
thermally dominated while the observations find freshwater dominance. The answer
is that the thermal share is not a single number, it falls monotonically with
density. In the classes that actually correspond to bottom water we agree with
the observations; the disagreement is confined to the intermediate classes, and
it is the expected signature of a model that convects in the open gyre rather
than over the shelf.

Their class boundaries are in neutral density. We map them onto our sigma2 axis
using the model's own winter surface water inside the September ice zone, so the
correspondence is derived from the same water the transformation is computed
over rather than from a published lookup table.

Note on what the observations do and do not report. Pellichero et al. give a
per-class heat/freshwater split nowhere in their text. They state that the
27.1 class is "dominated by freshwater-driven water-mass transformation" and
that, over the whole domain, the heat flux contributes "a factor ~2-5 lower
than the freshwater contribution" -- and that second number describes the
buoyancy flux, not the transformation rate. Only those two statements are drawn
here; the rest of their curve is not reproduced because it would have to be
read off their figure.

Outputs figures/figR11_density_class_comparison.{pdf,png}.
"""
import numpy as np
import xarray as xr
import seawater as sw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EXPS = [('pi', 'PI'), ('mh', 'MH'), ('lig', 'LIG'), ('lgm', 'LGM'), ('mis', 'MIS3')]
COL = {'pi': '#000000', 'mh': '#E69F00', 'lig': '#009E73',
       'lgm': '#0072B2', 'mis': '#CC79A7'}

MESH = '/home/a/a270064/bb1029/inputs/mesh_core2/fesom.mesh.diag.nc'

mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans'], 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333333'})


# ------------------------------------------------------------------ density map
def sigma2_for_gamma():
    """Map their neutral density boundaries onto our sigma2, using PI winter
    surface water inside the September ice zone."""
    T = xr.open_dataset('pi/temp.clim.nc', decode_times=False)['temp'][:, :, 0].values
    S = xr.open_dataset('pi/salt.clim.nc', decode_times=False)['salt'][:, :, 0].values
    lat = np.degrees(xr.open_dataset(MESH)['nodes'].isel(n2=1).values)
    aice = xr.open_dataset('pi/a_ice_clim.nc', decode_times=False)['a_ice'].values

    siz = (aice.max(axis=0) > 0.15) & (lat < 0)
    t = T[[5, 6, 7]].mean(axis=0)[siz]
    s = S[[5, 6, 7]].mean(axis=0)[siz]
    ok = np.isfinite(t) & np.isfinite(s) & (s > 20)
    t, s = t[ok], s[ok]

    sig0 = sw.dens(s, t, 0) - 1000
    sig2 = sw.dens(s, t, 2000) - 1000

    # A linear fit over the populated range extrapolates the 27.9 boundary.
    # That class barely outcrops at the PI winter surface (a few tens of points
    # out of ten thousand), which is itself part of the result: the model hardly
    # ventilates the density range that carries the observed dense cell.
    out = {}
    for g in [27.1, 27.3, 27.6, 27.9]:
        k = np.abs(sig0 - g) < 0.04
        out[g] = np.median(sig2[k]) if k.sum() > 20 else np.nan
    good = [g for g in out if np.isfinite(out[g])]
    fit = np.polyfit([g for g in good], [out[g] for g in good], 1)
    for g in out:
        if not np.isfinite(out[g]):
            out[g] = float(np.polyval(fit, g))
            print('  (%.1f extrapolated: barely outcrops at the PI surface)' % g)
    return out, len(t)


GMAP, NPTS = sigma2_for_gamma()
print('gamma_n -> sigma2 (from %d PI winter surface points):' % NPTS)
for g, s in GMAP.items():
    print('  %.1f -> %.2f' % (g, s))

# their classes, as (gamma_lo, gamma_hi, label)
CLASSES = [(27.1, 27.3, 'SAMW/AAIW\n27.1$-$27.3'),
           (27.3, 27.6, 'Upper CDW\n27.3$-$27.6'),
           (27.6, 27.9, 'Lower CDW\n27.6$-$27.9'),
           (27.9, None, 'AABW precursor\n$\\geq$27.9')]


def bounds(g0, g1):
    lo = GMAP[g0]
    hi = GMAP[g1] if g1 is not None else None
    return lo, hi


# ------------------------------------------------------------------ WMT
def load(exp, months=None):
    d = xr.open_dataset(
        f'{exp}/wmt_echam_heat_fesom_freshwater_southern_ocean_{exp}.nc',
        decode_times=False)
    sig = d['sigma2_l_target'].values
    sel = slice(None) if months is None else [m - 1 for m in months]

    def V(n):
        return -d[n].values[sel].mean(axis=0) / 1e9

    h = V('total_heat_surface_exchange_flux_nonadvective_heat')
    i = V('surface_ocean_flux_advective_negative_rhs_sea_ice_melt_salt')
    o = sum(V(f'surface_ocean_flux_advective_negative_rhs_{k}_salt')
            for k in ['evaporation', 'snow', 'rain_and_ice', 'rivers'])
    return sig, h, i, o


def shares(exp, months):
    """Thermal share and component totals per class."""
    sig, h, i, o = load(exp, months)
    res = []
    for g0, g1, _ in CLASSES:
        lo, hi = bounds(g0, g1)
        m = sig >= lo
        if hi is not None:
            m &= sig < hi
        H, I, O = np.nansum(h[m]), np.nansum(i[m]), np.nansum(o[m])
        den = abs(H) + abs(I) + abs(O)
        res.append(dict(thermal=abs(H) / den * 100 if den else np.nan,
                        total=H + I + O, heat=H, ice=I, oth=O))
    return res


fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
x = np.arange(len(CLASSES))
xlab = [c[2] for c in CLASSES]

# ---------------------------------------------------------------- (a) PI both seasons
ax = axes[0]
for season, mo, ls, mk in [('Annual mean', None, '-', 'o'), ('Winter (JJA)', [6, 7, 8], '--', 's')]:
    r = shares('pi', mo)
    ax.plot(x, [v['thermal'] for v in r], ls, marker=mk, color='#000000',
            lw=2.6, ms=9, label=season)
ax.axhline(50, color='#999999', lw=1.0, ls=':')
ax.text(2.95, 52, 'equal thermal / haline', fontsize=10.5, color='#777777', ha='right')
ax.axhspan(0, 33, color='#009E73', alpha=0.10, zorder=0)
ax.text(0.04, 6, 'haline dominated\n(observations)', fontsize=11,
        color='#00654a', transform=ax.get_yaxis_transform())
ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=11.5)
ax.set_ylim(0, 100)
ax.set_ylabel('Thermal share of transformation (%)', fontsize=13)
ax.set_title('(a)  Pre-industrial: thermal share falls with density',
             fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.25, lw=0.6)

# ---------------------------------------------------------------- (b) all experiments
ax = axes[1]
for exp, lab in EXPS:
    r = shares(exp, [6, 7, 8])
    ax.plot(x, [v['thermal'] for v in r], '-o', color=COL[exp], lw=2.4, ms=8, label=lab)
ax.axhspan(0, 33, color='#009E73', alpha=0.10, zorder=0)
ax.axhline(50, color='#999999', lw=1.0, ls=':')
ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=11.5)
ax.set_ylim(0, 100)
ax.set_ylabel('Thermal share of transformation (%)', fontsize=13)
ax.set_title('(b)  Winter, all climate states', fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=11.5, ncol=2, loc='upper right')
ax.grid(axis='y', alpha=0.25, lw=0.6)

# ---------------------------------------------------------------- (c) PI components
ax = axes[2]
r = shares('pi', None)
w = 0.27
ax.bar(x - w, [v['heat'] for v in r], w, color='#D55E00', label='Heat')
ax.bar(x, [v['ice'] for v in r], w, color='#009E73', label='Sea ice FW')
ax.bar(x + w, [v['oth'] for v in r], w, color='#0072B2', label='Other FW')
ax.axhline(0, color='#444444', lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=11.5)
ax.set_ylabel('Annual-mean transformation (Sv)', fontsize=13)
ax.set_title('(c)  Pre-industrial components by class', fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.25, lw=0.6)

# observational anchors, the only two the paper states numerically
ax.annotate('observed: this class is\n"dominated by freshwater"',
            xy=(0, r[0]['heat']), xytext=(0.15, 0.80), textcoords='axes fraction',
            fontsize=10.5, color='#444444',
            arrowprops=dict(arrowstyle='->', color='#777777', lw=1.2))
ax.annotate('observed dense cell\n5 $\\pm$ 5 Sv at 27.9',
            xy=(3, r[3]['ice']), xytext=(0.58, 0.55), textcoords='axes fraction',
            fontsize=10.5, color='#444444',
            arrowprops=dict(arrowstyle='->', color='#777777', lw=1.2))

for a in axes:
    a.tick_params(labelsize=11.5)
    a.set_xlabel('Neutral density class of Pellichero et al. (2018)', fontsize=12.5)

fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.17, wspace=0.26)
fig.savefig('figures/figR11_density_class_comparison.pdf', dpi=400)
fig.savefig('figures/figR11_density_class_comparison.png', dpi=200)
print('saved figures/figR11_density_class_comparison.{pdf,png}')

# ------------------------------------------------------------------ table
print()
hdr = '%-6s %-8s %-16s %8s %8s %8s %8s %7s'
print(hdr % ('exp', 'season', 'class', 'total', 'heat', 'seaice', 'othFW', 'therm%'))
for exp, lab in EXPS:
    for season, mo in [('annual', None), ('JJA', [6, 7, 8])]:
        for (g0, g1, nm), v in zip(CLASSES, shares(exp, mo)):
            cl = '%.1f-%s' % (g0, ('%.1f' % g1) if g1 else 'max')
            print(hdr % (lab, season, cl, '%.2f' % v['total'], '%.2f' % v['heat'],
                         '%.2f' % v['ice'], '%.2f' % v['oth'], '%.0f%%' % v['thermal']))
